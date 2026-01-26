#!/usr/bin/env python3
"""
TxGemma MCP Server

Entry point for the Model Context Protocol server exposing TxGemma tools.
Uses FastMCP for dual stdio/SSE support.
"""

import logging
from typing import Any, Dict

from fastmcp import FastMCP

from txgemma.tool_factory import build_tools
from txgemma.chat_factory import register_chat_tool
from txgemma.executor import execute_tool

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Server initialization
# -----------------------------------------------------------------------------

mcp = FastMCP(
    "txgemma-mcp",
)

# Load chat tool once at startup (before model is loaded)
logger.info("Loading TxGemma chat tool...")

# Register chat tool
register_chat_tool(mcp)

logger.info(f"Registered TxGemma chat tool with FastMCP")

# Load tools once at startup (before model is loaded)
# Tools are auto-generated from TDC prompts downloaded from HuggingFace
logger.info("Loading TxGemma tools from TDC definitions...")

# Configuration: Choose your tool filtering strategy
# ===================================================

# Option 1: Load ALL tools (default - maximum flexibility)
TOOLS = build_tools()

# Option 2: Load only Drug SMILES tools (uncomment to use)
# TOOLS = build_tools(filter_placeholder="Drug SMILES")

# Option 3: Load only simple tools (≤2 parameters) (uncomment to use)
# TOOLS = build_tools(max_placeholders=2)

# Option 4: Load only drug-target interaction tools (uncomment to use)
# TOOLS = build_tools(
#     filter_placeholders=["Drug SMILES", "Target sequence"],
#     match_all=True
# )

# Option 5: Load sequence-related tools (fuzzy match) (uncomment to use)
# TOOLS = build_tools(
#     filter_placeholder="sequence",
#     exact_match=False
# )

logger.info(f"Loaded {len(TOOLS)} tools")

# Convert Tool objects to FastMCP format and register
for tool in TOOLS:
    tool_name = tool.name
    tool_description = tool.description
    tool_schema = tool.inputSchema
    
    # Enhance description with parameter information for agents
    # This helps agents understand what parameters to provide
    enhanced_description = tool_description
    if tool_schema.get("properties"):
        enhanced_description += "\n\nParameters:"
        for param_name, param_info in tool_schema["properties"].items():
            param_desc = param_info.get("description", "")
            param_type = param_info.get("type", "string")
            is_required = param_name in tool_schema.get("required", [])
            required_marker = " (required)" if is_required else " (optional)"
            enhanced_description += f"\n- {param_name}{required_marker}: {param_desc} (type: {param_type})"
    
    # Create a closure that captures the tool name
    def make_tool_func(name: str):
        def _tool_func(params: dict) -> str:
            """
            Execute a TxGemma tool with the provided parameters.
            
            Args:
                params: Dictionary of parameter name -> value mappings.
                        Parameter names may contain spaces (e.g., "Drug SMILES").
            
            Returns:
                Prediction result from the TxGemma model.
            """
            try:
                result = execute_tool(name, params)
                return result
            except Exception as e:
                logger.error(f"Tool execution failed for {name}: {e}")
                return f"ERROR: {str(e)}"
        
        _tool_func.__name__ = name
        
        return _tool_func
    
    # Register the tool
    tool_func = make_tool_func(tool_name)
    mcp.tool(name=tool_name, description=enhanced_description)(tool_func)

logger.info(f"Registered {len(TOOLS)} tools with FastMCP")


# -----------------------------------------------------------------------------
# Resources
# -----------------------------------------------------------------------------

@mcp.resource("txgemma://info")
def server_info() -> str:
    """Information about the TxGemma MCP server and available models."""
    from txgemma.tool_factory import analyze_tools
    
    stats = analyze_tools()
    
    info = f"""
TxGemma MCP Server
==================

This server provides access to Google DeepMind's TxGemma models for therapeutic
development and drug discovery tasks.

Server Configuration:
- Tools loaded: {len(TOOLS)}
- Total available tools: {stats['total_tools']}
- Unique placeholders: {stats['total_placeholders']}

Available Models:
- TxGemma 2B (default - fastest, good for basic predictions)
- TxGemma 9B (balanced speed and accuracy)
- TxGemma 27B (most accurate, slower)

Model Types:
- Predict: Optimized for property predictions (current)
- Chat: Conversational, can explain predictions

Current Tools: {', '.join([t.name for t in TOOLS[:5]])}{'...' if len(TOOLS) > 5 else ''}

Most Common Placeholders:
"""
    
    for placeholder, count in stats['most_common_placeholders'][:5]:
        info += f"- {placeholder}: {count} tools\n"
    
    info += """
For more information, visit:
https://developers.google.com/health-ai-developer-foundations/txgemma
"""
    return info


@mcp.resource("txgemma://stats")
def server_stats() -> str:
    """Detailed statistics about available tools."""
    from txgemma.tool_factory import analyze_tools
    import json
    
    stats = analyze_tools()
    return json.dumps(stats, indent=2)


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

def main():
    """Main entry point for MCP server."""
    # FastMCP will handle transport/host/port via CLI args
    # This is just for direct Python execution
    logger.info("Starting TxGemma MCP server...")
    mcp.run()


if __name__ == "__main__":
    main()