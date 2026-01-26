"""
TxGemma MCP package.

Provides Model Context Protocol tools for TxGemma therapeutic AI models.
"""

from txgemma.chat_factory import register_chat_tool
from txgemma.executor import execute_chat, execute_chat_async, execute_tool, execute_tool_async
from txgemma.model import (
    TxGemmaChatModel,
    TxGemmaPredictModel,
    get_chat_model,
    get_predict_model,
)
from txgemma.prompts import PromptLoader, PromptTemplate, get_loader
from txgemma.tool_factory import build_tools

__version__ = "0.1.0"

__all__ = [
    # Models
    "TxGemmaPredictModel",
    "TxGemmaChatModel",
    "get_predict_model",
    "get_chat_model",
    # Execution
    "execute_tool",
    "execute_tool_async",
    "execute_chat",
    "execute_chat_async",
    # Tool building
    "build_tools",
    "register_chat_tool",
    # Prompts
    "PromptTemplate",
    "PromptLoader",
    "get_loader",
]
