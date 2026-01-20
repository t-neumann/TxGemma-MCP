"""
TxGemma MCP package.

Provides Model Context Protocol tools for TxGemma therapeutic AI models.
"""

from txgemma.model import TxGemmaModel, get_model
from txgemma.executor import execute_tool, execute_tool_async
from txgemma.tool_factory import build_tools
from txgemma.prompts import PromptTemplate, PromptLoader, get_loader

__version__ = "0.1.0"

__all__ = [
    "TxGemmaModel",
    "get_model",
    "execute_tool",
    "execute_tool_async",
    "build_tools",
    "PromptTemplate",
    "PromptLoader",
    "get_loader",
]