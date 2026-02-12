"""
Tests for txgemma.chat_factory module.

Tests chat tool registration with FastMCP server.
Uses mocked dependencies for unit tests, real models for GPU tests.

"""

from unittest.mock import Mock, patch

import pytest

from txgemma.chat_factory import register_chat_tool

# Mark all tests as unit tests (except GPU tests which override)
pytestmark = [pytest.mark.unit]

# =============================================================================
# TOOL REGISTRATION TESTS
# =============================================================================


class TestRegisterChatTool:
    """Test chat tool registration with FastMCP."""

    def test_register_chat_tool_calls_mcp_tool(self):
        """Test that register_chat_tool calls mcp.tool()."""
        mock_mcp = Mock()
        mock_tool_decorator = Mock()
        mock_mcp.tool.return_value = mock_tool_decorator

        register_chat_tool(mock_mcp)

        # Should call mcp.tool with name and description
        mock_mcp.tool.assert_called_once()
        call_kwargs = mock_mcp.tool.call_args[1]

        assert call_kwargs["name"] == "txgemma_chat"
        assert "description" in call_kwargs
        assert "drug discovery" in call_kwargs["description"].lower()

    def test_register_tool_description_content(self):
        """Test that tool description includes key information."""
        mock_mcp = Mock()
        mock_tool_decorator = Mock()
        mock_mcp.tool.return_value = mock_tool_decorator

        register_chat_tool(mock_mcp)

        description = mock_mcp.tool.call_args[1]["description"]

        # Should mention key use cases
        assert "molecular properties" in description.lower()
        assert "drug discovery" in description.lower()
        assert "examples" in description.lower()

    def test_register_tool_registers_function(self):
        """Test that the decorator is called with a function."""
        mock_mcp = Mock()
        mock_tool_decorator = Mock()
        mock_mcp.tool.return_value = mock_tool_decorator

        register_chat_tool(mock_mcp)

        # The decorator should be called with the function
        mock_tool_decorator.assert_called_once()

        # Get the registered function
        registered_func = mock_tool_decorator.call_args[0][0]
        assert callable(registered_func)
        assert registered_func.__name__ == "txgemma_chat"

    @patch("txgemma.chat_factory.logger")
    def test_register_logs_success(self, mock_logger):
        """Test that successful registration is logged."""
        mock_mcp = Mock()
        mock_tool_decorator = Mock()
        mock_mcp.tool.return_value = mock_tool_decorator

        register_chat_tool(mock_mcp)

        # Should log registration
        mock_logger.info.assert_called_with("Registered txgemma_chat tool")


# =============================================================================
# REGISTERED FUNCTION TESTS
# =============================================================================


class TestRegisteredFunction:
    """Test the txgemma_chat function that gets registered."""

    @patch("txgemma.chat_factory.execute_chat")
    def test_function_calls_execute_chat(self, mock_execute_chat):
        """Test that registered function calls execute_chat."""
        mock_mcp = Mock()
        mock_tool_decorator = Mock()
        mock_mcp.tool.return_value = mock_tool_decorator

        register_chat_tool(mock_mcp)

        # Get the registered function
        registered_func = mock_tool_decorator.call_args[0][0]

        # Mock execute_chat
        mock_execute_chat.return_value = "This is a helpful response."

        # Call function with question parameter
        result = registered_func(question="What is toxicity?")

        # Should call execute_chat with question
        mock_execute_chat.assert_called_once_with("What is toxicity?")
        assert result == "This is a helpful response."

    @patch("txgemma.chat_factory.execute_chat")
    def test_function_with_empty_question(self, mock_execute_chat):
        """Test function with empty question."""
        mock_mcp = Mock()
        mock_tool_decorator = Mock()
        mock_mcp.tool.return_value = mock_tool_decorator

        register_chat_tool(mock_mcp)
        registered_func = mock_tool_decorator.call_args[0][0]

        # Call with empty string
        result = registered_func(question="")

        # Should return error, not call execute_chat
        assert "ERROR" in result
        assert "Missing required parameter" in result
        mock_execute_chat.assert_not_called()

    @patch("txgemma.chat_factory.execute_chat")
    def test_function_handles_execute_chat_exception(self, mock_execute_chat):
        """Test that function handles execute_chat exceptions."""
        mock_mcp = Mock()
        mock_tool_decorator = Mock()
        mock_mcp.tool.return_value = mock_tool_decorator

        register_chat_tool(mock_mcp)
        registered_func = mock_tool_decorator.call_args[0][0]

        # Make execute_chat raise exception
        mock_execute_chat.side_effect = RuntimeError("GPU out of memory")

        # Call function - should NOT raise, should return error
        result = registered_func(question="Test question?")

        # Should return error message
        assert "ERROR" in result
        assert "GPU out of memory" in result

    @patch("txgemma.chat_factory.logger")
    @patch("txgemma.chat_factory.execute_chat")
    def test_function_logs_errors(self, mock_execute_chat, mock_logger):
        """Test that errors are logged."""
        mock_mcp = Mock()
        mock_tool_decorator = Mock()
        mock_mcp.tool.return_value = mock_tool_decorator

        register_chat_tool(mock_mcp)
        registered_func = mock_tool_decorator.call_args[0][0]

        # Make execute_chat raise exception
        mock_execute_chat.side_effect = RuntimeError("Model error")

        # Call function
        _ = registered_func(question="Test?")

        # Should log error
        assert mock_logger.error.called
        error_call = mock_logger.error.call_args[0][0]
        assert "Chat tool execution failed" in error_call

    @patch("txgemma.chat_factory.execute_chat")
    def test_function_with_long_question(self, mock_execute_chat):
        """Test function with long question."""
        mock_mcp = Mock()
        mock_tool_decorator = Mock()
        mock_mcp.tool.return_value = mock_tool_decorator

        register_chat_tool(mock_mcp)
        registered_func = mock_tool_decorator.call_args[0][0]

        mock_execute_chat.return_value = "Detailed response..."

        long_question = (
            "Why might the drug with SMILES CC(=O)OC1=CC=CC=C1C(=O)O "
            "cause liver toxicity in phase 3 clinical trials when "
            "administered at high doses to elderly patients?"
        )

        result = registered_func(question=long_question)

        # Should handle long questions
        assert result == "Detailed response..."
        mock_execute_chat.assert_called_once_with(long_question)

    @patch("txgemma.chat_factory.execute_chat")
    def test_function_with_smiles_in_question(self, mock_execute_chat):
        """Test function with SMILES string in question."""
        mock_mcp = Mock()
        mock_tool_decorator = Mock()
        mock_mcp.tool.return_value = mock_tool_decorator

        register_chat_tool(mock_mcp)
        registered_func = mock_tool_decorator.call_args[0][0]

        mock_execute_chat.return_value = "This molecule is aspirin..."

        question = "What is the molecule CC(=O)OC1=CC=CC=C1C(=O)O?"
        result = registered_func(question=question)

        assert "aspirin" in result.lower()
        mock_execute_chat.assert_called_once_with(question)


# =============================================================================
# FUNCTION SIGNATURE TESTS
# =============================================================================


class TestFunctionSignature:
    """Test the registered function signature and annotations."""

    def test_function_has_correct_name(self):
        """Test that function has correct name."""
        mock_mcp = Mock()
        mock_tool_decorator = Mock()
        mock_mcp.tool.return_value = mock_tool_decorator

        register_chat_tool(mock_mcp)
        registered_func = mock_tool_decorator.call_args[0][0]

        assert registered_func.__name__ == "txgemma_chat"

    def test_function_has_docstring(self):
        """Test that function has a docstring."""
        mock_mcp = Mock()
        mock_tool_decorator = Mock()
        mock_mcp.tool.return_value = mock_tool_decorator

        register_chat_tool(mock_mcp)
        registered_func = mock_tool_decorator.call_args[0][0]

        assert registered_func.__doc__ is not None
        assert len(registered_func.__doc__) > 0

    def test_function_has_type_annotations(self):
        """Test that function has proper type annotations."""
        mock_mcp = Mock()
        mock_tool_decorator = Mock()
        mock_mcp.tool.return_value = mock_tool_decorator

        register_chat_tool(mock_mcp)
        registered_func = mock_tool_decorator.call_args[0][0]

        # Check annotations
        annotations = registered_func.__annotations__
        assert "question" in annotations
        assert "return" in annotations
        assert annotations["return"] is str


# =============================================================================
# INTEGRATION TESTS (Unit-level with mocks)
# =============================================================================


class TestChatFactoryIntegration:
    """Integration-style tests with mocked dependencies."""

    @patch("txgemma.chat_factory.execute_chat")
    def test_full_registration_and_execution_flow(self, mock_execute_chat):
        """Test complete registration and execution flow."""
        # Setup
        mock_mcp = Mock()
        mock_tool_decorator = Mock()
        mock_mcp.tool.return_value = mock_tool_decorator
        mock_execute_chat.return_value = "Chat model response"

        # Register
        register_chat_tool(mock_mcp)

        # Verify registration
        assert mock_mcp.tool.called
        assert mock_tool_decorator.called

        # Get function and execute
        registered_func = mock_tool_decorator.call_args[0][0]
        result = registered_func(question="Test question")

        # Verify execution
        assert result == "Chat model response"
        mock_execute_chat.assert_called_once_with("Test question")

    @patch("txgemma.chat_factory.execute_chat")
    def test_multiple_executions(self, mock_execute_chat):
        """Test that function can be called multiple times."""
        mock_mcp = Mock()
        mock_tool_decorator = Mock()
        mock_mcp.tool.return_value = mock_tool_decorator

        register_chat_tool(mock_mcp)
        registered_func = mock_tool_decorator.call_args[0][0]

        # Multiple calls
        mock_execute_chat.return_value = "Response 1"
        result1 = registered_func(question="Question 1")

        mock_execute_chat.return_value = "Response 2"
        result2 = registered_func(question="Question 2")

        assert result1 == "Response 1"
        assert result2 == "Response 2"
        assert mock_execute_chat.call_count == 2


# =============================================================================
# GPU INTEGRATION TESTS (Requires actual model)
# =============================================================================


@pytest.mark.gpu
class TestChatToolGPU:
    """GPU integration tests with real chat model."""

    @pytest.fixture(scope="class")
    def loaded_chat_model(self):
        """Load chat model once for all GPU tests."""
        from txgemma.model import get_chat_model

        model = get_chat_model()
        model.load()
        yield model
        model.unload()

    def test_execute_chat_real_model(self, loaded_chat_model):
        """Test execute_chat with real model."""
        from txgemma.executor import execute_chat

        result = execute_chat("What is a SMILES string?")

        assert isinstance(result, str)
        assert len(result) > 0
        # Chat responses should be verbose
        assert len(result) > 20

    def test_registered_tool_with_real_model(self, loaded_chat_model):
        """Test registered tool with real model execution."""
        mock_mcp = Mock()
        mock_tool_decorator = Mock()
        mock_mcp.tool.return_value = mock_tool_decorator

        # Register tool (no mocking of execute_chat)
        register_chat_tool(mock_mcp)
        registered_func = mock_tool_decorator.call_args[0][0]

        # Execute with real model
        result = registered_func(question="Why is aspirin used to treat pain?")

        assert isinstance(result, str)
        assert len(result) > 0
        assert "ERROR" not in result

    def test_drug_discovery_question(self, loaded_chat_model):
        """Test with drug discovery specific question."""
        mock_mcp = Mock()
        mock_tool_decorator = Mock()
        mock_mcp.tool.return_value = mock_tool_decorator

        register_chat_tool(mock_mcp)
        registered_func = mock_tool_decorator.call_args[0][0]

        question = "What makes a good blood-brain barrier penetrant drug?"
        result = registered_func(question=question)

        assert isinstance(result, str)
        assert len(result) > 50  # Should be detailed

    def test_molecular_property_question(self, loaded_chat_model):
        """Test with molecular property question."""
        mock_mcp = Mock()
        mock_tool_decorator = Mock()
        mock_mcp.tool.return_value = mock_tool_decorator

        register_chat_tool(mock_mcp)
        registered_func = mock_tool_decorator.call_args[0][0]

        question = "How does lipophilicity affect drug absorption?"
        result = registered_func(question=question)

        assert isinstance(result, str)
        assert len(result) > 0
        assert "ERROR" not in result
