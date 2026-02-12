"""
TxGemma-MCP package initialization with lazy loading

Author: Tobias Neumann
License: MIT
Version: 0.1.1
"""

__version__ = "0.1.1"

# =============================================================================
# LAZY IMPORTS
# =============================================================================
# Import heavy modules (torch-dependent) only when accessed
# This prevents PyTorch loading during test collection with --cov


def __getattr__(name):
    """
    Lazy import handler for expensive imports.

    Delays importing torch-dependent modules until actually needed.
    This prevents PyTorch + coverage.py conflicts during test collection.
    """
    # Model classes and functions (imports torch)
    if name in ("TxGemmaPredictModel", "TxGemmaChatModel", "get_predict_model", "get_chat_model"):
        from txgemma.model import (  # noqa: F401
            TxGemmaChatModel,
            TxGemmaPredictModel,
            get_chat_model,
            get_predict_model,
        )

        return locals()[name]

    # Executor functions (imports model which imports torch)
    if name in ("execute_tool", "execute_tool_async", "execute_chat", "execute_chat_async"):
        from txgemma.executor import (
            execute_chat,
            execute_tool,
        )

        # Note: async versions removed in refactoring
        if name == "execute_chat":
            return execute_chat
        elif name == "execute_tool":
            return execute_tool
        elif name in ("execute_chat_async", "execute_tool_async"):
            raise AttributeError(f"'{name}' was removed in refactoring. Use '{name[:-6]}' instead.")

    # Chat factory (imports executor which imports model)
    if name == "register_chat_tool":
        from txgemma.chat_factory import register_chat_tool

        return register_chat_tool

    # Tool building (imports prompts and executor)
    if name == "build_tools":
        from txgemma.tool_factory import build_tools

        return build_tools

    # Prompt modules (lightweight, safe to import eagerly)
    if name in ("PromptTemplate", "PromptLoader", "get_loader"):
        from txgemma.prompts import PromptLoader, PromptTemplate, get_loader  # noqa: F401

        return locals()[name]

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


# =============================================================================
# EAGER IMPORTS (lightweight modules only)
# =============================================================================
# These don't import torch, safe for test collection

# Validation (no torch dependency)
# Cache utils (no torch dependency)
from txgemma.cache_utils import (  # noqa: E402
    get_cached_parameter_mapping,
    reset_all_caches,
    reset_parameter_mapping,
)

# Config (no torch dependency)
from txgemma.config import get_config, reset_config  # noqa: E402
from txgemma.validation import InputValidator, ValidationError, validate_tool_call  # noqa: E402

# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [
    "InputValidator",
    "PromptLoader",
    # Prompts (lazy)
    "PromptTemplate",
    "TxGemmaChatModel",
    # Models (lazy)
    "TxGemmaPredictModel",
    # Validation (eager - lightweight)
    "ValidationError",
    # Version
    "__version__",
    # Tool building (lazy)
    "build_tools",
    "execute_chat",
    # Execution (lazy)
    "execute_tool",
    "get_cached_parameter_mapping",
    "get_chat_model",
    # Config (eager - lightweight)
    "get_config",
    "get_loader",
    "get_predict_model",
    "register_chat_tool",
    # Cache (eager - lightweight)
    "reset_all_caches",
    "reset_config",
    "reset_parameter_mapping",
    "validate_tool_call",
]


# =============================================================================
# DIR SUPPORT
# =============================================================================


def __dir__():
    """Support for dir() to show all available attributes."""
    return __all__
