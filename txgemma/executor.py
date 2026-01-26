"""
Tool executor for TxGemma MCP server.

Executes tool calls by formatting prompts and running them through the model.
Supports both prediction tools (TDC) and chat queries.
"""

import logging
from typing import Any

from txgemma.model import get_chat_model, get_predict_model
from txgemma.prompts import get_loader

logger = logging.getLogger(__name__)


def execute_tool(tool_name: str, arguments: dict[str, Any]) -> str:
    """
    Execute a TxGemma tool with the given arguments.

    Args:
        tool_name: Name of the tool to execute
        arguments: Dictionary of parameter name -> value mappings

    Returns:
        Prediction result from the model (stripped of whitespace)

    Raises:
        KeyError: If tool_name is not found
        ValueError: If arguments are invalid for the tool
        RuntimeError: If model generation fails
    """
    logger.info(f"Executing tool: {tool_name}")

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
    logger.debug(f"Formatted prompt: {prompt[:100]}...")

    # Generate prediction using model
    model = get_predict_model()
    try:
        result = model.generate(prompt, max_new_tokens=64)
    except Exception as e:
        logger.error(f"Model generation failed for {tool_name}: {e}")
        raise RuntimeError(f"Model generation failed: {e}") from e

    logger.info(f"Tool {tool_name} completed successfully")

    # Strip whitespace from result
    return result.strip()


def execute_chat(question: str) -> str:
    """
    Execute a chat query with TxGemma chat model.

    Args:
        question: User's question about drug discovery, molecular properties, etc.

    Returns:
        Conversational response from TxGemma chat model

    Raises:
        RuntimeError: If chat model generation fails
    """
    logger.info(f"Executing chat query: {question[:100]}...")

    try:
        chat_model = get_chat_model()
        response = chat_model.generate(question)
        logger.info(f"Chat response generated (length: {len(response)})")
        return response
    except Exception as e:
        logger.error(f"Chat execution failed: {e}")
        raise RuntimeError(f"Chat model error: {e}") from e


async def execute_tool_async(tool_name: str, arguments: dict[str, Any]) -> str:
    """
    Async version of execute_tool.

    Currently just wraps the sync version.
    Could be enhanced with actual async model inference in the future.

    Args:
        tool_name: Name of the tool to execute
        arguments: Dictionary of parameter name -> value mappings

    Returns:
        Prediction result from the model
    """
    # For now, just call the sync version
    # In future, could use async model inference
    return execute_tool(tool_name, arguments)


async def execute_chat_async(question: str) -> str:
    """
    Async version of execute_chat.

    Currently just wraps the sync version.
    Could be enhanced with actual async model inference in the future.

    Args:
        question: User's question

    Returns:
        Conversational response from chat model
    """
    # For now, just call the sync version
    return execute_chat(question)
