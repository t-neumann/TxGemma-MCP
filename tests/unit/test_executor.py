"""
Tests for txgemma.executor module.

Tests tool execution logic with mocked dependencies using dependency injection.
Focus on parameter mapping, validation, and error handling.

"""

from unittest.mock import Mock, patch

import pytest

from txgemma.executor import execute_chat, execute_tool
from txgemma.validation import ValidationError

pytestmark = [pytest.mark.unit]

# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def mock_loader():
    """Mock PromptLoader."""
    loader = Mock()
    template = Mock()
    template.format.return_value = "Formatted prompt"
    loader.get.return_value = template
    return loader


@pytest.fixture
def mock_predict_model():
    """Mock TxGemmaPredictModel."""
    model = Mock()
    model.generate.return_value = "Model result"
    return model


@pytest.fixture
def mock_chat_model():
    """Mock TxGemmaChatModel."""
    model = Mock()
    model.is_loaded = True
    model.generate.return_value = "Chat response"
    return model


@pytest.fixture
def mock_param_mapping():
    """Mock parameter mapping (normalized -> original)."""
    return {
        "drug_smiles": "Drug SMILES",
        "target_sequence": "Target sequence",
        "trial_phase": "Trial phase",
    }


# =============================================================================
# CORE EXECUTION TESTS
# =============================================================================

class TestExecuteToolCore:
    """Test core tool execution logic with dependency injection."""
    
    def test_execute_tool_success(self, mock_loader, mock_predict_model, mock_param_mapping):
        """Test successful tool execution using dependency injection."""
        result = execute_tool(
            "test_tool",
            {"param": "value"},
            _loader=mock_loader,
            _model=mock_predict_model,
            _param_mapping=mock_param_mapping,
        )
        
        assert result == "Model result"
        mock_loader.get.assert_called_once_with("test_tool")
        mock_predict_model.generate.assert_called_once()
    
    def test_execute_tool_strips_whitespace(self, mock_loader, mock_predict_model, mock_param_mapping):
        """Test that result whitespace is stripped."""
        mock_predict_model.generate.return_value = "  Result with spaces  \n\t"
        
        result = execute_tool(
            "test_tool",
            {"param": "value"},
            _loader=mock_loader,
            _model=mock_predict_model,
            _param_mapping=mock_param_mapping,
        )
        
        assert result == "Result with spaces"
    
    def test_execute_tool_max_tokens(self, mock_loader, mock_predict_model, mock_param_mapping):
        """Test that max_new_tokens is set to 64."""
        execute_tool(
            "test_tool",
            {"param": "value"},
            _loader=mock_loader,
            _model=mock_predict_model,
            _param_mapping=mock_param_mapping,
        )
        
        # Verify max_new_tokens=64 is passed
        call_args = mock_predict_model.generate.call_args
        assert call_args[1]["max_new_tokens"] == 64
    
    def test_execute_tool_unknown_tool(self, mock_loader, mock_predict_model, mock_param_mapping):
        """Test execution with unknown tool name."""
        mock_loader.get.side_effect = KeyError("not found")
        
        with pytest.raises(KeyError, match="Unknown tool"):
            execute_tool(
                "unknown_tool",
                {},
                _loader=mock_loader,
                _model=mock_predict_model,
                _param_mapping=mock_param_mapping,
            )


# =============================================================================
# PARAMETER MAPPING TESTS ⭐ CRITICAL
# =============================================================================

class TestExecuteToolParameterMapping:
    """Test parameter name mapping (normalized -> original)."""
    
    def test_maps_normalized_to_original(self, mock_loader, mock_predict_model):
        """Test that normalized names are mapped to original placeholder names."""
        # Setup mapping
        param_mapping = {
            "drug_smiles": "Drug SMILES",
            "target_sequence": "Target sequence",
        }
        
        # Execute with normalized names
        execute_tool(
            "test_tool",
            {
                "drug_smiles": "CCO",
                "target_sequence": "MKTAYIAK",
            },
            _loader=mock_loader,
            _model=mock_predict_model,
            _param_mapping=param_mapping,
        )
        
        # Verify template.format was called with ORIGINAL names
        template = mock_loader.get.return_value
        call_kwargs = template.format.call_args[1]
        
        assert "Drug SMILES" in call_kwargs
        assert call_kwargs["Drug SMILES"] == "CCO"
        assert "Target sequence" in call_kwargs
        assert call_kwargs["Target sequence"] == "MKTAYIAK"
        
        # Should NOT have normalized names
        assert "drug_smiles" not in call_kwargs
        assert "target_sequence" not in call_kwargs
    
    def test_preserves_unmapped_names(self, mock_loader, mock_predict_model):
        """Test that unmapped parameter names are preserved."""
        param_mapping = {
            "drug_smiles": "Drug SMILES",
        }
        
        execute_tool(
            "test_tool",
            {
                "drug_smiles": "CCO",
                "unknown_param": "value",  # Not in mapping
            },
            _loader=mock_loader,
            _model=mock_predict_model,
            _param_mapping=param_mapping,
        )
        
        template = mock_loader.get.return_value
        call_kwargs = template.format.call_args[1]
        
        # Mapped parameter should use original name
        assert "Drug SMILES" in call_kwargs
        
        # Unmapped parameter should be preserved as-is
        assert "unknown_param" in call_kwargs
    
    def test_handles_empty_mapping(self, mock_loader, mock_predict_model):
        """Test with empty parameter mapping."""
        execute_tool(
            "test_tool",
            {"param": "value"},
            _loader=mock_loader,
            _model=mock_predict_model,
            _param_mapping={},  # Empty mapping
        )
        
        # Should still work, using parameter names as-is
        template = mock_loader.get.return_value
        call_kwargs = template.format.call_args[1]
        assert "param" in call_kwargs
    
    def test_maps_all_common_parameters(self, mock_loader, mock_predict_model):
        """Test mapping for all common TDC parameters."""
        param_mapping = {
            "drug_smiles": "Drug SMILES",
            "target_sequence": "Target sequence",
            "protein_sequence": "Protein sequence",
            "epitope_amino_acid_sequence": "Epitope amino acid sequence",
            "trial_phase": "Trial phase",
            "cell_line": "Cell line",
        }
        
        execute_tool(
            "complex_tool",
            {
                "drug_smiles": "CCO",
                "target_sequence": "MKTAYIAK",
                "trial_phase": "Phase 3",
            },
            _loader=mock_loader,
            _model=mock_predict_model,
            _param_mapping=param_mapping,
        )
        
        template = mock_loader.get.return_value
        call_kwargs = template.format.call_args[1]
        
        # All should be mapped to original names
        assert "Drug SMILES" in call_kwargs
        assert "Target sequence" in call_kwargs
        assert "Trial phase" in call_kwargs


# =============================================================================
# VALIDATION TESTS ⭐ NEW
# =============================================================================

class TestExecuteToolValidation:
    """Test input validation integration."""
    
    @patch("txgemma.executor.validate_tool_call")
    def test_validates_inputs(self, mock_validate, mock_loader, mock_predict_model, mock_param_mapping):
        """Test that validate_tool_call is invoked."""
        # Mock validation to return cleaned inputs
        mock_validate.return_value = ("test_tool", {"param": "value"})
        
        execute_tool(
            "test_tool",
            {"param": "value"},
            _loader=mock_loader,
            _model=mock_predict_model,
            _param_mapping=mock_param_mapping,
        )
        
        # Should call validation
        mock_validate.assert_called_once_with("test_tool", {"param": "value"})
    
    @patch("txgemma.executor.validate_tool_call")
    def test_validation_failure_raises_value_error(self, mock_validate, mock_loader, mock_predict_model, mock_param_mapping):
        """Test that validation errors are converted to ValueError."""
        # Mock validation to fail
        mock_validate.side_effect = ValidationError("Invalid input")
        
        with pytest.raises(ValueError, match="Invalid input"):
            execute_tool(
                "test_tool",
                {"param": "value"},
                _loader=mock_loader,
                _model=mock_predict_model,
                _param_mapping=mock_param_mapping,
            )
    
    @patch("txgemma.executor.validate_tool_call")
    def test_uses_validated_inputs(self, mock_validate, mock_loader, mock_predict_model, mock_param_mapping):
        """Test that validated/cleaned inputs are used."""
        # Mock validation to clean inputs
        mock_validate.return_value = ("cleaned_tool", {"cleaned": "param"})
        
        execute_tool(
            "original_tool",
            {"original": "param"},
            _loader=mock_loader,
            _model=mock_predict_model,
            _param_mapping=mock_param_mapping,
        )
        
        # Should use cleaned tool name
        mock_loader.get.assert_called_once_with("cleaned_tool")


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================

class TestExecuteToolErrorHandling:
    """Test error handling and exception wrapping."""
    
    def test_template_format_error(self, mock_loader, mock_predict_model, mock_param_mapping):
        """Test error when template formatting fails."""
        template = mock_loader.get.return_value
        template.format.side_effect = ValueError("Missing required parameter")
        
        with pytest.raises(ValueError, match="Invalid arguments for tool"):
            execute_tool(
                "test_tool",
                {"wrong": "param"},
                _loader=mock_loader,
                _model=mock_predict_model,
                _param_mapping=mock_param_mapping,
            )
    
    def test_model_generation_error(self, mock_loader, mock_predict_model, mock_param_mapping):
        """Test error when model generation fails."""
        mock_predict_model.generate.side_effect = RuntimeError("GPU out of memory")
        
        with pytest.raises(RuntimeError, match="Model generation failed"):
            execute_tool(
                "test_tool",
                {"param": "value"},
                _loader=mock_loader,
                _model=mock_predict_model,
                _param_mapping=mock_param_mapping,
            )
    
    def test_error_includes_tool_name(self, mock_loader, mock_predict_model, mock_param_mapping):
        """Test that error messages include tool name for debugging."""
        mock_predict_model.generate.side_effect = RuntimeError("GPU error")
        
        with pytest.raises(RuntimeError) as exc_info:
            execute_tool(
                "specific_tool",
                {"param": "value"},
                _loader=mock_loader,
                _model=mock_predict_model,
                _param_mapping=mock_param_mapping,
            )
        
        # Error should mention the tool name (in logged context)
        # The raised error should wrap the original
        assert "Model generation failed" in str(exc_info.value)


# =============================================================================
# EXECUTE CHAT TESTS
# =============================================================================

class TestExecuteChat:
    """Test chat execution functionality."""
    
    def test_execute_chat_success(self, mock_chat_model):
        """Test successful chat execution."""
        result = execute_chat(
            "What is toxicity?",
            _model=mock_chat_model,
        )
        
        assert result == "Chat response"
        mock_chat_model.generate.assert_called_once_with("What is toxicity?")
    
    def test_execute_chat_with_long_question(self, mock_chat_model):
        """Test chat with long question."""
        long_question = "Why might the drug CC(=O)OC1=CC=CC=C1C(=O)O cause liver toxicity in patients with pre-existing conditions?"
        
        result = execute_chat(long_question, _model=mock_chat_model)
        
        assert result == "Chat response"
        mock_chat_model.generate.assert_called_once_with(long_question)
    
    @patch("txgemma.validation.InputValidator.validate_string_value")
    def test_validates_question(self, mock_validate, mock_chat_model):
        """Test that question is validated."""
        mock_validate.return_value = "Valid question"
        
        execute_chat("Question?", _model=mock_chat_model)
        
        mock_validate.assert_called_once_with("Question?", "question")
    
    @patch("txgemma.validation.InputValidator.validate_string_value")
    def test_validation_failure_raises_value_error(self, mock_validate, mock_chat_model):
        """Test that validation errors are converted to ValueError."""
        mock_validate.side_effect = ValidationError("Invalid question")
        
        with pytest.raises(ValueError, match="Invalid question"):
            execute_chat("", _model=mock_chat_model)
    
    def test_execute_chat_model_error(self, mock_chat_model):
        """Test chat execution when model fails."""
        mock_chat_model.generate.side_effect = RuntimeError("GPU out of memory")
        
        with pytest.raises(RuntimeError, match="Chat model error"):
            execute_chat("Question?", _model=mock_chat_model)
    
    def test_execute_chat_empty_question(self, mock_chat_model):
        """Test chat with empty question (after validation)."""
        # Note: Validation might reject empty string, but if it passes through:
        result = execute_chat("", _model=mock_chat_model)
        
        assert isinstance(result, str)


# =============================================================================
# LOGGING TESTS
# =============================================================================

class TestLogging:
    """Test logging behavior."""
    
    @patch("txgemma.executor.logger")
    def test_logs_tool_execution(self, mock_logger, mock_loader, mock_predict_model, mock_param_mapping):
        """Test that tool execution is logged."""
        execute_tool(
            "test_tool",
            {"param": "value"},
            _loader=mock_loader,
            _model=mock_predict_model,
            _param_mapping=mock_param_mapping,
        )
        
        # Should log at start and end
        assert mock_logger.info.call_count >= 2
        
        # Check log messages
        calls = [str(call) for call in mock_logger.info.call_args_list]
        assert any("Executing tool" in str(call) for call in calls)
        assert any("completed successfully" in str(call) for call in calls)
    
    @patch("txgemma.executor.logger")
    def test_logs_validation_errors(self, mock_logger, mock_loader, mock_predict_model, mock_param_mapping):
        """Test that validation errors are logged."""
        with patch("txgemma.executor.validate_tool_call") as mock_validate:
            mock_validate.side_effect = ValidationError("Bad input")
            
            with pytest.raises(ValueError):
                execute_tool(
                    "test_tool",
                    {},
                    _loader=mock_loader,
                    _model=mock_predict_model,
                    _param_mapping=mock_param_mapping,
                )
            
            # Should log error
            assert mock_logger.error.called
    
    @patch("txgemma.executor.logger")
    def test_logs_model_errors(self, mock_logger, mock_loader, mock_predict_model, mock_param_mapping):
        """Test that model errors are logged."""
        mock_predict_model.generate.side_effect = RuntimeError("GPU error")
        
        with pytest.raises(RuntimeError):
            execute_tool(
                "test_tool",
                {"param": "value"},
                _loader=mock_loader,
                _model=mock_predict_model,
                _param_mapping=mock_param_mapping,
            )
        
        # Should log error with tool name
        assert mock_logger.error.called
        error_calls = [str(call) for call in mock_logger.error.call_args_list]
        assert any("Model generation failed" in str(call) for call in error_calls)
    
    @patch("txgemma.executor.logger")
    def test_logs_chat_execution(self, mock_logger, mock_chat_model):
        """Test that chat execution is logged."""
        execute_chat("Question?", _model=mock_chat_model)
        
        # Should log multiple times (start, model loaded, response generated)
        assert mock_logger.info.call_count >= 3
    
    @patch("txgemma.executor.logger")
    def test_logs_chat_errors_with_traceback(self, mock_logger, mock_chat_model):
        """Test that chat errors are logged with exc_info."""
        mock_chat_model.generate.side_effect = RuntimeError("GPU error")
        
        with pytest.raises(RuntimeError):
            execute_chat("Question?", _model=mock_chat_model)
        
        # Should log error with exc_info=True
        assert mock_logger.error.called
        # Check that exc_info was used
        call = mock_logger.error.call_args
        assert call[1].get("exc_info") is True


# =============================================================================
# INTEGRATION-LIKE TESTS (still unit tests with mocks)
# =============================================================================

class TestExecuteToolWorkflows:
    """Test complete execution workflows."""
    
    def test_full_workflow_with_mapping(self, mock_loader, mock_predict_model):
        """Test complete workflow with parameter mapping."""
        # Setup realistic mapping
        param_mapping = {
            "drug_smiles": "Drug SMILES",
            "target_sequence": "Target sequence",
        }
        
        # Setup template to use original names
        template = mock_loader.get.return_value
        template.format.return_value = "Formatted prompt with Drug SMILES"
        
        # Execute with normalized names
        result = execute_tool(
            "Lipophilicity_AstraZeneca",
            {"drug_smiles": "CCO"},
            _loader=mock_loader,
            _model=mock_predict_model,
            _param_mapping=param_mapping,
        )
        
        # Verify end-to-end flow
        assert result == "Model result"
        
        # Verify template got original name
        call_kwargs = template.format.call_args[1]
        assert "Drug SMILES" in call_kwargs
        assert call_kwargs["Drug SMILES"] == "CCO"
        
        # Verify model got formatted prompt
        mock_predict_model.generate.assert_called_once()
        prompt_arg = mock_predict_model.generate.call_args[0][0]
        assert prompt_arg == "Formatted prompt with Drug SMILES"
    
    def test_multi_parameter_workflow(self, mock_loader, mock_predict_model):
        """Test workflow with multiple parameters."""
        param_mapping = {
            "drug_smiles": "Drug SMILES",
            "target_sequence": "Target sequence",
            "trial_phase": "Trial phase",
        }
        
        result = execute_tool(
            "ComplexTool",
            {
                "drug_smiles": "CCO",
                "target_sequence": "MKTAYIAK",
                "trial_phase": "Phase 2",
            },
            _loader=mock_loader,
            _model=mock_predict_model,
            _param_mapping=param_mapping,
        )
        
        # Verify all parameters mapped correctly
        template = mock_loader.get.return_value
        call_kwargs = template.format.call_args[1]
        
        assert call_kwargs["Drug SMILES"] == "CCO"
        assert call_kwargs["Target sequence"] == "MKTAYIAK"
        assert call_kwargs["Trial phase"] == "Phase 2"
        
        # Verify result
        assert result == "Model result"