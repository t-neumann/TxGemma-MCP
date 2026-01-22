"""
Execute TxGemma tool calls.
"""

import logging
from typing import Any

from txgemma.model import get_model
from txgemma.prompts import get_loader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def execute_tool(tool_name: str, arguments: dict[str, Any]) -> str:
    """
    Execute a TxGemma tool call.

    Args:
        tool_name: Name of the tool (matches prompt template name)
        arguments: Dictionary of parameter values

    Returns:
        Generated prediction result as string

    Raises:
        KeyError: If tool_name doesn't exist
        ValueError: If arguments are invalid
        RuntimeError: If model fails
    """
    # Get the prompt template
    loader = get_loader()

    try:
        template = loader.get(tool_name)
    except KeyError:
        raise KeyError(f"Unknown tool: {tool_name}") from None

    # Format the prompt with arguments
    try:
        prompt = template.format(**arguments)
    except ValueError as e:
        raise ValueError(f"Invalid arguments for tool '{tool_name}': {e}") from e

    logger.info(f"Executing tool: {tool_name}")
    logger.debug(f"Prompt:\n{prompt}")

    # Get model and generate
    model = get_model()

    try:
        result = model.generate(prompt)
        logger.info(f"Generated result for {tool_name}")
        return result
    except Exception as e:
        logger.error(f"Model generation failed for {tool_name}: {e}")
        raise RuntimeError(f"Model generation failed: {e}") from e


async def execute_tool_async(tool_name: str, arguments: dict[str, Any]) -> str:
    """
    Async wrapper for execute_tool.

    Currently just calls the sync version, but could be extended to use
    async model inference in the future.

    Args:
        tool_name: Name of the tool
        arguments: Tool arguments

    Returns:
        Generated prediction result
    """
    # For now, just call sync version
    # In production, you might want to use asyncio.to_thread() or similar
    return execute_tool(tool_name, arguments)
