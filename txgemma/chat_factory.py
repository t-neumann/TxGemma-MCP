"""
Chat tool for TxGemma conversational model.

Registers the chat tool with FastMCP server.
"""

import logging

from txgemma.executor import execute_chat

logger = logging.getLogger(__name__)


# Chat tool metadata for MCP
CHAT_TOOL = {
    "name": "txgemma_chat",
    "description": """Ask TxGemma Chat model a question about drug discovery, molecular properties, 
or therapeutic development. The chat model provides detailed explanations and can discuss 
drug-target interactions, toxicity mechanisms, pharmacokinetics, and more.

Use this tool when you need:
- Explanations of molecular properties
- Discussion of drug mechanisms
- Advice on drug discovery strategies
- Interpretation of SMILES structures
- Understanding of biological targets

Parameters:
- question (required): Your question about drugs, molecules, or therapeutic development (type: string)

Examples:
- "Why might the drug CC(=O)OC1=CC=CC=C1C(=O)O cause liver toxicity?"
- "What makes a good blood-brain barrier penetrant drug?"
- "How does protein sequence affect drug binding?"
""",
    "inputSchema": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "Your question about drug discovery, molecular properties, or therapeutic development"
            }
        },
        "required": ["question"]
    }
}


def register_chat_tool(mcp):
    """
    Register the chat tool with FastMCP server.
    
    Args:
        mcp: FastMCP instance
    """
    enhanced_description = CHAT_TOOL["description"]
    
    def _chat_tool_func(params: dict) -> str:
        """
        Execute TxGemma chat model.
        
        Args:
            params: Dictionary containing 'question' key
        
        Returns:
            Conversational response from chat model
        """
        try:
            question = params.get("question")
            if not question:
                return "ERROR: Missing required parameter 'question'"
            
            return execute_chat(question)
        except Exception as e:
            logger.error(f"Chat tool execution failed: {e}")
            return f"ERROR: {str(e)}"
    
    _chat_tool_func.__name__ = CHAT_TOOL["name"]
    
    # Register with FastMCP
    mcp.tool(
        name=CHAT_TOOL["name"],
        description=enhanced_description
    )(_chat_tool_func)
    
    logger.info("Registered txgemma_chat tool")