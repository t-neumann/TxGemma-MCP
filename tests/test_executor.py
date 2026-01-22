"""
Tests for txgemma.executor module.

These tests mock the model to avoid GPU requirements.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from txgemma.executor import execute_tool, execute_tool_async


class TestExecutorUnit:
    """Unit tests with mocked model (no GPU required)."""
    
    @patch('txgemma.executor.get_model')
    @patch('txgemma.executor.get_loader')
    def test_execute_tool_success(self, mock_loader, mock_get_model):
        """Test successful tool execution."""
        # Mock template
        mock_template = Mock()
        mock_template.format.return_value = "Formatted prompt"
        
        # Mock loader
        mock_loader_instance = Mock()
        mock_loader_instance.get.return_value = mock_template
        mock_loader.return_value = mock_loader_instance
        
        # Mock model
        mock_model = Mock()
        mock_model.generate.return_value = "Predicted result"
        mock_get_model.return_value = mock_model
        
        # Execute
        result = execute_tool("test_tool", {"Drug SMILES": "CC(=O)O"})
        
        # Assertions
        assert result == "Predicted result"
        mock_loader_instance.get.assert_called_once_with("test_tool")
        mock_template.format.assert_called_once_with(**{"Drug SMILES": "CC(=O)O"})
        mock_model.generate.assert_called_once_with("Formatted prompt")
    
    @patch('txgemma.executor.get_loader')
    def test_execute_tool_unknown_tool(self, mock_loader):
        """Test execution with unknown tool name."""
        mock_loader_instance = Mock()
        mock_loader_instance.get.side_effect = KeyError("Tool not found")
        mock_loader.return_value = mock_loader_instance
        
        with pytest.raises(KeyError, match="Unknown tool"):
            execute_tool("nonexistent_tool", {})
    
    @patch('txgemma.executor.get_model')
    @patch('txgemma.executor.get_loader')
    def test_execute_tool_invalid_arguments(self, mock_loader, mock_get_model):
        """Test execution with invalid arguments."""
        # Mock template that raises ValueError
        mock_template = Mock()
        mock_template.format.side_effect = ValueError("Missing placeholders")
        
        mock_loader_instance = Mock()
        mock_loader_instance.get.return_value = mock_template
        mock_loader.return_value = mock_loader_instance
        
        with pytest.raises(ValueError, match="Invalid arguments"):
            execute_tool("test_tool", {})
    
    @patch('txgemma.executor.get_model')
    @patch('txgemma.executor.get_loader')
    def test_execute_tool_model_failure(self, mock_loader, mock_get_model):
        """Test execution when model generation fails."""
        # Mock template
        mock_template = Mock()
        mock_template.format.return_value = "Prompt"
        
        mock_loader_instance = Mock()
        mock_loader_instance.get.return_value = mock_template
        mock_loader.return_value = mock_loader_instance
        
        # Mock model that fails
        mock_model = Mock()
        mock_model.generate.side_effect = RuntimeError("GPU out of memory")
        mock_get_model.return_value = mock_model
        
        with pytest.raises(RuntimeError, match="Model generation failed"):
            execute_tool("test_tool", {"Drug SMILES": "CC(=O)O"})
    
    @pytest.mark.asyncio
    @patch('txgemma.executor.execute_tool')
    async def test_execute_tool_async(self, mock_execute):
        """Test async wrapper."""
        mock_execute.return_value = "Result"
        
        result = await execute_tool_async("test_tool", {"arg": "value"})
        
        assert result == "Result"
        mock_execute.assert_called_once_with("test_tool", {"arg": "value"})


class TestExecutorWithRealLoader:
    """Tests using real loader but mocked model."""
    
    @pytest.fixture
    def mock_prompts_file(self, tmp_path):
        """Create a temporary prompts file."""
        import json
        prompts_file = tmp_path / "test_prompts.json"
        prompts_file.write_text(json.dumps({
            "test_tool": "Question: What is {Drug SMILES}?\nAnswer:"
        }))
        return prompts_file
    
    @patch('txgemma.executor.get_model')
    def test_execute_with_real_template(self, mock_get_model, mock_prompts_file):
        """Test execution with real template formatting."""
        from txgemma.prompts import PromptLoader
        
        # Use real loader with test file
        with patch('txgemma.executor.get_loader') as mock_get_loader:
            loader = PromptLoader(local_override=mock_prompts_file)
            mock_get_loader.return_value = loader
            
            # Mock model
            mock_model = Mock()
            mock_model.generate.return_value = "Low toxicity"
            mock_get_model.return_value = mock_model
            
            # Execute
            result = execute_tool("test_tool", {"Drug SMILES": "CC(=O)O"})
            
            # Check that prompt was formatted correctly
            call_args = mock_model.generate.call_args[0][0]
            assert "CC(=O)O" in call_args
            assert result == "Low toxicity"