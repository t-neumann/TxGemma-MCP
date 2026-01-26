"""
Tests for txgemma.executor module.

Tests the tool execution logic with mocked models.
"""

import pytest
from unittest.mock import Mock, patch

from txgemma.executor import execute_tool, execute_chat


class TestExecuteToolMocked:
    """Test execute_tool with mocked dependencies."""
    
    @patch('txgemma.executor.get_loader')
    @patch('txgemma.executor.get_predict_model')
    def test_execute_tool_success(self, mock_get_model, mock_get_loader):
        """Test successful tool execution."""
        # Mock loader
        mock_loader = Mock()
        mock_template = Mock()
        mock_template.format.return_value = "Formatted prompt"
        mock_loader.get.return_value = mock_template
        mock_get_loader.return_value = mock_loader
        
        # Mock model
        mock_model = Mock()
        mock_model.generate.return_value = "Model result"
        mock_get_model.return_value = mock_model
        
        # Execute
        result = execute_tool("test_tool", {"param": "value"})
        
        # Verify
        assert result == "Model result"
        mock_loader.get.assert_called_once_with("test_tool")
        mock_template.format.assert_called_once_with(param="value")
        mock_model.generate.assert_called_once_with("Formatted prompt", max_new_tokens=64)
    
    @patch('txgemma.executor.get_loader')
    def test_execute_tool_unknown_tool(self, mock_get_loader):
        """Test execution with unknown tool name."""
        mock_loader = Mock()
        mock_loader.get.side_effect = KeyError("not found")
        mock_get_loader.return_value = mock_loader
        
        with pytest.raises(KeyError, match="Unknown tool"):
            execute_tool("unknown_tool", {})
    
    @patch('txgemma.executor.get_loader')
    @patch('txgemma.executor.get_predict_model')
    def test_execute_tool_invalid_arguments(self, mock_get_model, mock_get_loader):
        """Test execution with invalid arguments."""
        # Mock loader
        mock_loader = Mock()
        mock_template = Mock()
        mock_template.format.side_effect = ValueError("Missing required")
        mock_loader.get.return_value = mock_template
        mock_get_loader.return_value = mock_loader
        
        with pytest.raises(ValueError, match="Invalid arguments"):
            execute_tool("test_tool", {"wrong": "param"})
    
    @patch('txgemma.executor.get_loader')
    @patch('txgemma.executor.get_predict_model')
    def test_execute_tool_model_failure(self, mock_get_model, mock_get_loader):
        """Test execution when model generation fails."""
        # Mock loader
        mock_loader = Mock()
        mock_template = Mock()
        mock_template.format.return_value = "Formatted prompt"
        mock_loader.get.return_value = mock_template
        mock_get_loader.return_value = mock_loader
        
        # Mock model to fail
        mock_model = Mock()
        mock_model.generate.side_effect = RuntimeError("GPU error")
        mock_get_model.return_value = mock_model
        
        with pytest.raises(RuntimeError, match="Model generation failed"):
            execute_tool("test_tool", {"param": "value"})
    
    @patch('txgemma.executor.get_loader')
    @patch('txgemma.executor.get_predict_model')
    def test_execute_tool_with_complex_params(self, mock_get_model, mock_get_loader):
        """Test execution with multiple parameters."""
        # Mock loader
        mock_loader = Mock()
        mock_template = Mock()
        mock_template.format.return_value = "Complex prompt"
        mock_loader.get.return_value = mock_template
        mock_get_loader.return_value = mock_loader
        
        # Mock model
        mock_model = Mock()
        mock_model.generate.return_value = "Complex result"
        mock_get_model.return_value = mock_model
        
        # Execute with multiple params
        result = execute_tool(
            "complex_tool",
            {
                "Drug SMILES": "CC(=O)O",
                "Target sequence": "MKTAYIAK",
                "Trial phase": "Phase 3"
            }
        )
        
        assert result == "Complex result"
        # Verify params were passed correctly (with spaces in names)
        mock_template.format.assert_called_once_with(
            **{
                "Drug SMILES": "CC(=O)O",
                "Target sequence": "MKTAYIAK",
                "Trial phase": "Phase 3"
            }
        )
    
    @patch('txgemma.executor.get_loader')
    @patch('txgemma.executor.get_predict_model')
    def test_execute_tool_strips_result(self, mock_get_model, mock_get_loader):
        """Test that result is stripped of whitespace."""
        # Mock loader
        mock_loader = Mock()
        mock_template = Mock()
        mock_template.format.return_value = "Prompt"
        mock_loader.get.return_value = mock_template
        mock_get_loader.return_value = mock_loader
        
        # Mock model returns result with whitespace
        mock_model = Mock()
        mock_model.generate.return_value = "  Result with spaces  \n"
        mock_get_model.return_value = mock_model
        
        result = execute_tool("test_tool", {"param": "value"})
        
        # Should be stripped
        assert result == "Result with spaces"


class TestExecuteChatMocked:
    """Test execute_chat with mocked dependencies."""
    
    @patch('txgemma.executor.get_chat_model')
    def test_execute_chat_success(self, mock_get_chat_model):
        """Test successful chat execution."""
        # Mock chat model
        mock_model = Mock()
        mock_model.generate.return_value = "This is a helpful response."
        mock_get_chat_model.return_value = mock_model
        
        # Execute
        result = execute_chat("What is toxicity?")
        
        # Verify
        assert result == "This is a helpful response."
        mock_model.generate.assert_called_once_with("What is toxicity?")
    
    @patch('txgemma.executor.get_chat_model')
    def test_execute_chat_with_long_question(self, mock_get_chat_model):
        """Test chat with a long question."""
        # Mock chat model
        mock_model = Mock()
        mock_model.generate.return_value = "Detailed explanation..."
        mock_get_chat_model.return_value = mock_model
        
        long_question = "Why might the drug CC(=O)OC1=CC=CC=C1C(=O)O cause liver toxicity?"
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
    
    @patch('txgemma.executor.get_chat_model')
    def test_execute_chat_empty_question(self, mock_get_chat_model):
        """Test chat with empty question."""
        # Mock chat model
        mock_model = Mock()
        mock_model.generate.return_value = "I need a question to answer."
        mock_get_chat_model.return_value = mock_model
        
        result = execute_chat("")
        
        assert isinstance(result, str)
        mock_model.generate.assert_called_once_with("")


class TestExecuteToolLogging:
    """Test logging behavior in execute_tool."""
    
    @patch('txgemma.executor.logger')
    @patch('txgemma.executor.get_loader')
    @patch('txgemma.executor.get_predict_model')
    def test_logging_on_success(self, mock_get_model, mock_get_loader, mock_logger):
        """Test that successful execution logs appropriately."""
        # Setup mocks
        mock_loader = Mock()
        mock_template = Mock()
        mock_template.format.return_value = "Prompt"
        mock_loader.get.return_value = mock_template
        mock_get_loader.return_value = mock_loader
        
        mock_model = Mock()
        mock_model.generate.return_value = "Result"
        mock_get_model.return_value = mock_model
        
        # Execute
        execute_tool("test_tool", {"param": "value"})
        
        # Should log info
        assert mock_logger.info.called
    
    @patch('txgemma.executor.logger')
    @patch('txgemma.executor.get_loader')
    @patch('txgemma.executor.get_predict_model')
    def test_logging_on_failure(self, mock_get_model, mock_get_loader, mock_logger):
        """Test that failures are logged as errors."""
        # Setup mocks
        mock_loader = Mock()
        mock_template = Mock()
        mock_template.format.return_value = "Prompt"
        mock_loader.get.return_value = mock_template
        mock_get_loader.return_value = mock_loader
        
        mock_model = Mock()
        mock_model.generate.side_effect = RuntimeError("Error")
        mock_get_model.return_value = mock_model
        
        # Execute (will raise)
        with pytest.raises(RuntimeError):
            execute_tool("test_tool", {"param": "value"})
        
        # Should log error
        assert mock_logger.error.called


class TestExecuteChatLogging:
    """Test logging behavior in execute_chat."""
    
    @patch('txgemma.executor.logger')
    @patch('txgemma.executor.get_chat_model')
    def test_logging_on_success(self, mock_get_chat_model, mock_logger):
        """Test that successful chat execution logs appropriately."""
        mock_model = Mock()
        mock_model.generate.return_value = "Response"
        mock_get_chat_model.return_value = mock_model
        
        execute_chat("Question?")
        
        # Should log info (at least twice: start and end)
        assert mock_logger.info.call_count >= 2
    
    @patch('txgemma.executor.logger')
    @patch('txgemma.executor.get_chat_model')
    def test_logging_on_failure(self, mock_get_chat_model, mock_logger):
        """Test that chat failures are logged as errors."""
        mock_model = Mock()
        mock_model.generate.side_effect = RuntimeError("Error")
        mock_get_chat_model.return_value = mock_model
        
        with pytest.raises(RuntimeError):
            execute_chat("Question?")
        
        # Should log error
        assert mock_logger.error.called