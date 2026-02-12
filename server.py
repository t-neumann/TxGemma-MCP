#!/usr/bin/env python3
"""
TxGemma MCP Server

Entry point for the Model Context Protocol server exposing TxGemma tools.
Uses FastMCP for dual stdio/SSE support.
"""

import logging
from typing import Annotated

from fastmcp import FastMCP

from txgemma.chat_factory import register_chat_tool
from txgemma.config import get_config
from txgemma.executor import execute_tool
from txgemma.tool_factory import build_tools

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Load config
# -----------------------------------------------------------------------------

logger.info("Loading TxGemma configuration...")

config = get_config()

# -----------------------------------------------------------------------------
# Server initialization
# -----------------------------------------------------------------------------

mcp = FastMCP("TxGemma-MCP", version="0.1.0")

if config.tools.enable_chat:
    logger.info("Loading TxGemma chat tool...")
    register_chat_tool(mcp)
    logger.info("Registered TxGemma chat tool with FastMCP")
else:
    logger.info("Chat tool disabled in config")

# Load tools once at startup
logger.info("Loading TxGemma tools from TDC definitions...")

TOOLS = build_tools(
    filter_placeholder=config.tools.filter_placeholder,
    filter_placeholders=config.tools.filter_placeholders,
    match_all=config.tools.match_all,
    exact_match=config.tools.exact_match,
    exclude_complex=config.tools.exclude_complex,
    max_placeholders=config.tools.max_placeholders,
    exclude_name_pattern=config.tools.exclude_name_pattern,
)

logger.info(f"Loaded {len(TOOLS)} tools")

# -----------------------------------------------------------------------------
# Tool Registration Helper
# -----------------------------------------------------------------------------


def _get_python_type(json_type: str) -> type:
    """Map JSON Schema types to Python types."""
    type_map = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
    }
    return type_map.get(json_type, str)


def _make_tool_function_safe(tool_name: str, tool_schema: dict) -> callable:
    """
    Create a tool function with explicit parameters.

    Unfortunately, FastMCP requires explicit parameters (no **kwargs) and
    doesn't properly unwrap Pydantic models. The ONLY way to create functions
    with dynamic parameter names is exec(). However, we make this as safe as
    possible by:

    1. Validating all inputs before exec
    2. Using a minimal, auditable template
    3. Restricting the execution environment
    4. Thorough logging

    Args:
        tool_name: Name of the tool (validated to be safe)
        tool_schema: JSON Schema for tool parameters

    Returns:
        Callable function ready for FastMCP registration
    """

    # Validate tool name to prevent code injection
    if not tool_name.replace("_", "").replace("-", "").isalnum():
        raise ValueError(f"Invalid tool name: {tool_name}")

    properties = tool_schema.get("properties", {})
    required = set(tool_schema.get("required", []))

    # Build parameter list and annotations
    param_parts = []
    annotations = {}

    for param_name, param_info in properties.items():
        # Validate parameter name
        if not param_name.replace("_", "").isalnum():
            raise ValueError(f"Invalid parameter name: {param_name}")

        python_type = _get_python_type(param_info.get("type", "string"))
        description = param_info.get("description", "")

        # Add to annotations
        annotations[param_name] = Annotated[python_type, description]

        # Add to parameter list
        if param_name in required:
            param_parts.append(param_name)
        else:
            param_parts.append(f"{param_name}=None")

    params_str = ", ".join(param_parts)
    param_names = [p.split("=")[0] for p in param_parts]

    # Create minimal, safe function template
    # This is the ONLY place we use exec(), and it's carefully controlled
    func_template = f'''
def {tool_name}({params_str}) -> str:
    """Execute {tool_name} tool."""
    params_dict = {{{", ".join([f'"{p}": {p}' for p in param_names])}}}
    params_dict = {{k: v for k, v in params_dict.items() if v is not None}}
    return _execute_tool_safe("{tool_name}", params_dict)
'''

    # Execute in restricted environment with only necessary imports
    exec_globals = {
        "_execute_tool_safe": lambda name, args: execute_tool(name, args),
        "Annotated": Annotated,
    }
    exec_locals = {}

    # Execute the function definition
    try:
        exec(func_template, exec_globals, exec_locals)
    except Exception as e:
        logger.error(f"Failed to create function for {tool_name}: {e}")
        raise RuntimeError(f"Tool function creation failed: {e}") from e

    # Extract the generated function
    tool_func = exec_locals[tool_name]

    # Add annotations for FastMCP
    tool_func.__annotations__ = annotations
    tool_func.__annotations__["return"] = str

    logger.debug(f"Created function for tool: {tool_name} with params: {param_names}")

    return tool_func


# -----------------------------------------------------------------------------
# Register Tools
# -----------------------------------------------------------------------------

logger.info("Registering tools with FastMCP...")

for tool in TOOLS:
    try:
        tool_func = _make_tool_function_safe(tool.name, tool.inputSchema)
        mcp.tool(name=tool.name, description=tool.description)(tool_func)
    except Exception as e:
        logger.error(f"Failed to register tool {tool.name}: {e}")
        # Continue with other tools rather than failing completely
        continue

logger.info(f"Registered {len(TOOLS)} tools with FastMCP")

# -----------------------------------------------------------------------------
# Resources
# -----------------------------------------------------------------------------


@mcp.resource("txgemma://info")
def server_info() -> str:
    """Information about the TxGemma MCP server."""
    import json

    info = {
        "server": "TxGemma-MCP",
        "version": "0.1.0",
        "tools_loaded": len(TOOLS),
        "configuration": {
            "predict_model": config.predict.model,
            "chat_model": config.chat.model,
            "filter": config.tools.filter_placeholder or "None (all tools)",
            "exclude_pattern": config.tools.exclude_name_pattern or "None",
        },
        "documentation": "https://github.com/t-neumann/TxGemma-MCP",
    }

    return json.dumps(info, indent=2)


@mcp.resource("txgemma://tools")
def tools_list() -> str:
    """List all loaded tools."""
    import json

    tools_data = [
        {
            "name": tool.name,
            "description": tool.description,
            "parameters": list(tool.inputSchema.get("properties", {}).keys()),
            "required": tool.inputSchema.get("required", []),
        }
        for tool in TOOLS
    ]

    return json.dumps(tools_data, indent=2)


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------


def main():
    """Main entry point for MCP server."""
    import sys

    logger.info("Starting TxGemma MCP server...")
    mcp.run(sys.argv[1:] if len(sys.argv) > 1 else None)


if __name__ == "__main__":
    main()
