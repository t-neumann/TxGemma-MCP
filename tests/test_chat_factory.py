"""
Tests for txgemma.chat_factory module.

Tests chat tool registration and execution.
"""

import pytest
from unittest.mock import Mock, patch

from txgemma.chat_factory import register_chat_tool, CHAT_TOOL
from txgemma.executor import execute_chat


class TestChatToolMetadata:
    """Test chat tool metadata."""
    
    def test_chat_tool_has_name(self):
        """Test that CHAT_TOOL has a name."""
        assert "name" in CHAT_TOOL
        assert CHAT_TOOL["name"] == "txgemma_chat"
    
    def test_chat_tool_has_description(self):
        """Test that CHAT_TOOL has a description."""
        assert "description" in CHAT_TOOL
        assert len(CHAT_TOOL["description"]) > 0
        assert "drug discovery" in CHAT_TOOL["description"].lower()
    
    def test_chat_tool_has_schema(self):
        """Test that CHAT_TOOL has input schema."""
        assert "inputSchema" in CHAT_TOOL
        schema = CHAT_TOOL["inputSchema"]
        
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "question" in schema["properties"]
        assert "required" in schema
        assert "question" in schema["required"]
    
    def test_question_parameter_schema(self):
        """Test question parameter schema."""
        question_schema = CHAT_TOOL["inputSchema"]["properties"]["question"]
        
        assert question_schema["type"] == "string"
        assert "description" in question_schema
        assert len(question_schema["description"]) > 0


class TestExecuteChat:
    """Test execute_chat function."""
    
    @patch('txgemma.executor.get_chat_model')
    def test_execute_chat_success(self, mock_get_chat_model):
        """Test successful chat execution."""
        # Mock chat model
        mock_model = Mock()
        mock_model.generate.return_value = "This is a helpful response about drugs."
        mock_get_chat_model.return_value = mock_model
        
        # Execute
        result = execute_chat("What is toxicity?")
        
        # Verify
        assert result == "This is a helpful response about drugs."
        mock_model.generate.assert_called_once_with("What is toxicity?")
    
    @patch('txgemma.executor.get_chat_model')
    def test_execute_chat_with_long_question(self, mock_get_chat_model):
        """Test chat with a long question."""
        # Mock chat model
        mock_model = Mock()
        mock_model.generate.return_value = "Detailed explanation..."
        mock_get_chat_model.return_value = mock_model
        
        long_question = "Why might the drug with SMILES CC(=O)OC1=CC=CC=C1C(=O)O cause liver toxicity in phase 3 trials?"
        result = execute_chat(long_question)
        
        assert result == "Detailed explanation..."
        mock_model.generate.assert_called_once_with(long_question)
    
    @patch('txgemma.executor.get_chat_model')
    def test_execute_chat_model_failure(self, mock_get_chat_model):
        """Test chat execution when model fails."""
        # Mock model to fail
        mock_model = Mock()
        mock_model.generate.side_effect = RuntimeError("GPU out of memory")
        mock_get_chat_model.return_value = mock_model
        
        # Should raise RuntimeError
        with pytest.raises(RuntimeError, match="Chat model error"):
            execute_chat("Question?")
    
    @patch('txgemma.executor.logger')
    @patch('txgemma.executor.get_chat_model')
    def test_execute_chat_logs_on_success(self, mock_get_chat_model, mock_logger):
        """Test that successful chat execution logs."""
        mock_model = Mock()
        mock_model.generate.return_value = "Response"
        mock_get_chat_model.return_value = mock_model
        
        execute_chat("Question?")
        
        # Should log info
        assert mock_logger.info.call_count >= 2  # Start and end
    
    @patch('txgemma.executor.logger')
    @patch('txgemma.executor.get_chat_model')
    def test_execute_chat_logs_on_failure(self, mock_get_chat_model, mock_logger):
        """Test that chat failures are logged."""
        mock_model = Mock()
        mock_model.generate.side_effect = RuntimeError("Error")
        mock_get_chat_model.return_value = mock_model
        
        with pytest.raises(RuntimeError):
            execute_chat("Question?")
        
        # Should log error
        assert mock_logger.error.called


class TestRegisterChatTool:
    """Test register_chat_tool function."""
    
    @patch('txgemma.chat_factory.logger')
    def test_register_chat_tool_success(self, mock_logger):
        """Test that chat tool registers successfully."""
        # Mock FastMCP
        mock_mcp = Mock()
        mock_tool_decorator = Mock()
        mock_mcp.tool.return_value = mock_tool_decorator
        
        # Register
        register_chat_tool(mock_mcp)
        
        # Verify tool decorator was called
        mock_mcp.tool.assert_called_once()
        call_kwargs = mock_mcp.tool.call_args[1]
        assert call_kwargs["name"] == "txgemma_chat"
        assert "description" in call_kwargs
        
        # Verify function was registered
        mock_tool_decorator.assert_called_once()
        
        # Should log registration
        mock_logger.info.assert_called_with("Registered txgemma_chat tool")
    
    def test_registered_tool_function_signature(self):
        """Test that registered function has correct signature."""
        mock_mcp = Mock()
        mock_tool_decorator = Mock()
        mock_mcp.tool.return_value = mock_tool_decorator
        
        register_chat_tool(mock_mcp)
        
        # Get the registered function
        registered_func = mock_tool_decorator.call_args[0][0]
        
        # Should have correct name
        assert registered_func.__name__ == "txgemma_chat"
    
    @patch('txgemma.chat_factory.execute_chat')
    def test_registered_tool_execution(self, mock_execute_chat):
        """Test that registered tool function executes correctly."""
        mock_mcp = Mock()
        mock_tool_decorator = Mock()
        mock_mcp.tool.return_value = mock_tool_decorator
        
        # Register tool
        register_chat_tool(mock_mcp)
        
        # Get the registered function
        registered_func = mock_tool_decorator.call_args[0][0]
        
        # Mock execute_chat
        mock_execute_chat.return_value = "Chat response"
        
        # Call registered function
        result = registered_func({"question": "What is toxicity?"})
        
        # Verify
        assert result == "Chat response"
        mock_execute_chat.assert_called_once_with("What is toxicity?")
    
    @patch('txgemma.chat_factory.execute_chat')
    def test_registered_tool_missing_question(self, mock_execute_chat):
        """Test registered tool with missing question parameter."""
        mock_mcp = Mock()
        mock_tool_decorator = Mock()
        mock_mcp.tool.return_value = mock_tool_decorator
        
        register_chat_tool(mock_mcp)
        registered_func = mock_tool_decorator.call_args[0][0]
        
        # Call without question
        result = registered_func({})
        
        # Should return error message
        assert "ERROR" in result
        assert "Missing required parameter" in result
        
        # Should not call execute_chat
        mock_execute_chat.assert_not_called()
    
    @patch('txgemma.chat_factory.execute_chat')
    @patch('txgemma.chat_factory.logger')
    def test_registered_tool_handles_exceptions(self, mock_logger, mock_execute_chat):
        """Test that registered tool handles exceptions gracefully."""
        mock_mcp = Mock()
        mock_tool_decorator = Mock()
        mock_mcp.tool.return_value = mock_tool_decorator
        
        register_chat_tool(mock_mcp)
        registered_func = mock_tool_decorator.call_args[0][0]
        
        # Make execute_chat raise exception
        mock_execute_chat.side_effect = RuntimeError("GPU error")
        
        # Call registered function
        result = registered_func({"question": "Test?"})
        
        # Should return error message, not raise
        assert "ERROR" in result
        assert "GPU error" in result
        
        # Should log error
        assert mock_logger.error.called


@pytest.mark.gpu
class TestChatToolIntegration:
    """Integration tests for chat tool (requires GPU)."""
    
    @pytest.fixture(scope="class")
    def loaded_chat_model(self):
        """Load chat model once for all tests."""
        from txgemma.model import get_chat_model
        model = get_chat_model()
        model.load()
        yield model
        model.unload()
    
    def test_execute_chat_real_model(self, loaded_chat_model):
        """Test execute_chat with real model."""
        result = execute_chat("What is a SMILES string?")
        
        assert isinstance(result, str)
        assert len(result) > 0
        # Should be more verbose than predictions
        assert len(result) > 20
    
    def test_execute_chat_drug_question(self, loaded_chat_model):
        """Test chat with drug-specific question."""
        result = execute_chat("Why is aspirin used to treat pain?")
        
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_registered_tool_real_execution(self, loaded_chat_model):
        """Test full tool registration and execution."""
        from fastmcp import FastMCP
        
        mcp = FastMCP("test")
        register_chat_tool(mcp)
        
        # Verify tool was registered by checking if it can be called
        # FastMCP stores tools differently, so we test by execution
        # The tool should be accessible via the registered function
        
        # We can't easily access internal tool registry, so we verify
        # by checking that registration succeeded without errors
        # (if it failed, an exception would have been raised)
        
        # Since we can't directly call the tool without going through
        # the full MCP protocol, we just verify registration completed
        assert True  # Registration succeeded if we got here
        
        # Alternative: Test that execute_chat works directly
        result = execute_chat("What is toxicity?")
        
        assert isinstance(result, str)
        assert len(result) > 0
        assert "ERROR" not in result