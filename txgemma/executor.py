"""
Tool execution with model inference

Author: Tobias Neumann
License: MIT
Version: 0.1.1
"""

import logging
from typing import Any

from txgemma.cache_utils import get_cached_parameter_mapping
from txgemma.model import TxGemmaChatModel, TxGemmaPredictModel, get_chat_model, get_predict_model
from txgemma.prompts import PromptLoader, get_loader
from txgemma.validation import ValidationError, validate_tool_call

logger = logging.getLogger(__name__)


def execute_tool(
    tool_name: str,
    arguments: dict[str, Any],
    # Optional dependencies for testing (normally None)
    _loader: PromptLoader | None = None,
    _model: TxGemmaPredictModel | None = None,
    _param_mapping: dict[str, str] | None = None,
) -> str:
    """
    Execute a TxGemma tool with the given arguments.

    Args:
        tool_name: Name of the tool to execute
        arguments: Dictionary of parameter name -> value mappings.
                  Uses NORMALIZED parameter names (e.g., "drug_smiles")
        _loader: Optional PromptLoader for testing (defaults to get_loader())
        _model: Optional model for testing (defaults to get_predict_model())
        _param_mapping: Optional mapping for testing (defaults to cached mapping)

    Returns:
        Prediction result from the model (stripped of whitespace)

    Raises:
        ValidationError: If inputs are invalid
        KeyError: If tool_name is not found
        ValueError: If arguments are invalid for the tool
        RuntimeError: If model generation fails
    """
    logger.info(f"Executing tool: {tool_name}")

    # Validate inputs first (security)
    try:
        tool_name, arguments = validate_tool_call(tool_name, arguments)
    except ValidationError as e:
        logger.error(f"Validation failed: {e}")
        raise ValueError(f"Invalid input: {e}") from e

    # Get dependencies (with optional injection for testing)
    loader = _loader or get_loader()
    model = _model or get_predict_model()
    param_mapping = _param_mapping or get_cached_parameter_mapping()

    # Get the prompt template
    try:
        template = loader.get(tool_name)
    except KeyError:
        raise KeyError(f"Unknown tool: {tool_name}") from None

    # Map normalized parameter names back to original placeholder names
    # The MCP client sends us {"drug_smiles": "CCO"}
    # But the prompt template needs {"Drug SMILES": "CCO"}
    original_arguments = {}

    for norm_name, value in arguments.items():
        # Get the original placeholder name (e.g., "drug_smiles" -> "Drug SMILES")
        original_name = param_mapping.get(norm_name, norm_name)
        original_arguments[original_name] = value

        if norm_name != original_name:
            logger.debug(f"Mapped parameter: {norm_name} -> {original_name}")

    # Format the prompt with original placeholder names
    try:
        prompt = template.format(**original_arguments)
    except ValueError as e:
        raise ValueError(f"Invalid arguments for tool '{tool_name}': {e}") from e

    logger.debug(f"Formatted prompt: {prompt[:100]}...")

    # Generate prediction using model
    try:
        result = model.generate(prompt, max_new_tokens=64)
    except Exception as e:
        logger.error(f"Model generation failed for {tool_name}: {e}")
        raise RuntimeError(f"Model generation failed: {e}") from e

    logger.info(f"Tool {tool_name} completed successfully")

    # Strip whitespace from result
    return result.strip()


def execute_chat(
    question: str,
    # Optional dependency for testing
    _model: TxGemmaChatModel | None = None,
) -> str:
    """
    Execute a chat query with TxGemma chat model.

    Args:
        question: User's question about drug discovery, molecular properties, etc.
        _model: Optional model for testing (defaults to get_chat_model())

    Returns:
        Conversational response from TxGemma chat model

    Raises:
        ValidationError: If question is invalid
        RuntimeError: If chat model generation fails
    """
    logger.info(f"Executing chat query: {question[:100]}...")

    # Validate question
    from txgemma.validation import InputValidator

    try:
        question = InputValidator.validate_string_value(question, "question")
    except ValidationError as e:
        logger.error(f"Chat validation failed: {e}")
        raise ValueError(f"Invalid question: {e}") from e

    # Get model (with optional injection for testing)
    chat_model = _model or get_chat_model()

    try:
        logger.info(f"Chat model loaded, is_loaded: {chat_model.is_loaded}")

        response = chat_model.generate(question)

        logger.info(f"Chat response generated (length: {len(response)})")
        return response
    except Exception as e:
        logger.error(f"Chat execution failed: {e}", exc_info=True)
        raise RuntimeError(f"Chat model error: {e}") from e
