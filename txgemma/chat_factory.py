"""
Chat tool for TxGemma conversational model.

Registers the chat tool with FastMCP server.
"""

import logging
from typing import Annotated

from txgemma.executor import execute_chat

logger = logging.getLogger(__name__)


def register_chat_tool(mcp):
    """
    Register the chat tool with FastMCP server.

    Args:
        mcp: FastMCP instance
    """
    
    description = """Ask TxGemma Chat model a question about drug discovery, molecular properties, 
or therapeutic development. The chat model provides detailed explanations and can discuss 
drug-target interactions, toxicity mechanisms, pharmacokinetics, and more.

Use this tool when you need:
- Explanations of molecular properties
- Discussion of drug mechanisms
- Advice on drug discovery strategies
- Interpretation of SMILES structures
- Understanding of biological targets

Examples:
- "Why might the drug CC(=O)OC1=CC=CC=C1C(=O)O cause liver toxicity?"
- "What makes a good blood-brain barrier penetrant drug?"
- "How does protein sequence affect drug binding?"
"""

    def txgemma_chat(
        question: Annotated[
            str,
            "Your question about drug discovery, molecular properties, or therapeutic development"
        ]
    ) -> str:
        """
        Execute TxGemma chat model with a question.

        Args:
            question: Your question about drugs, molecules, or therapeutic development

        Returns:
            Conversational response from chat model
        """
        try:
            if not question:
                return "ERROR: Missing required parameter 'question'"

            return execute_chat(question)
        except Exception as e:
            logger.error(f"Chat tool execution failed: {e}")
            return f"ERROR: {str(e)}"

    mcp.tool(name="txgemma_chat", description=description)(txgemma_chat)

    logger.info("Registered txgemma_chat tool")