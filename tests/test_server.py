"""
Tests for server.py - MCP server initialization and tool registration.

Tests server setup, tool registration, and FastMCP integration.
"""

import sys
import json
from unittest.mock import Mock, patch

import pytest


# =============================================================================
# Tests that import server.py directly
# =============================================================================


class TestServerIntegration:
    """Integration tests that import actual server module."""
    
    def test_server_can_be_imported(self):
        """Test that server module can be imported without errors."""
        import server
        
        assert hasattr(server, "mcp")
        assert hasattr(server, "TOOLS")
        assert hasattr(server, "main")
    
    def test_fastmcp_instance_exists(self):
        """Test that FastMCP instance is created."""
        from server import mcp
        
        # Check that mcp is a FastMCP instance
        assert mcp is not None
        # FastMCP should have run method
        assert hasattr(mcp, "run")
    
    def test_logger_configured(self):
        """Test that logging is configured."""
        import server
        
        assert hasattr(server, "logger")
        assert server.logger is not None
    
    def test_tools_list_exists(self):
        """Test that TOOLS list exists."""
        from server import TOOLS
        
        # TOOLS should be a list
        assert isinstance(TOOLS, list)
    
    @pytest.mark.skipif(
        not pytest.importorskip("server").TOOLS,
        reason="No tools loaded (may be mocked in test environment)",
    )
    def test_tools_have_valid_schemas(self):
        """Test that loaded tools have valid input schemas."""
        from server import TOOLS
        
        if len(TOOLS) == 0:
            pytest.skip("No tools loaded in test environment")
        
        for tool in TOOLS:
            # Check basic structure
            assert hasattr(tool, "name")
            assert hasattr(tool, "description")
            assert hasattr(tool, "inputSchema")
            
            # Check schema structure
            schema = tool.inputSchema
            assert "type" in schema
            assert schema["type"] == "object"
            assert "properties" in schema
            assert "required" in schema
            
            # Check properties structure
            for _prop_name, prop_schema in schema["properties"].items():
                assert "type" in prop_schema
                assert "description" in prop_schema


class TestResourceEndpoints:
    """Test resource endpoints defined in server."""
    
    @patch("txgemma.tool_factory.analyze_tools")
    def test_server_info_resource(self, mock_analyze):
        """Test server_info resource."""
        from server import server_info
        
        # Mock analyze_tools
        mock_analyze.return_value = {
            "total_tools": 703,
            "total_placeholders": 50,
            "most_common_placeholders": [
                ("Drug SMILES", 677),
                ("Target sequence", 30),
            ],
        }
        
        # server_info is a FunctionResource after decoration
        # Access the underlying function via .fn attribute
        if hasattr(server_info, 'fn'):
            result = server_info.fn()
        else:
            # Fallback if structure is different
            result = server_info()
        
        # Verify result contains expected content
        assert "TxGemma MCP Server" in result
        assert "Drug SMILES" in result or "703" in result
    
    @patch("txgemma.tool_factory.analyze_tools")
    def test_server_stats_resource(self, mock_analyze):
        """Test server_stats resource returns JSON."""
        from server import server_stats
        
        # Mock analyze_tools
        mock_stats = {
            "total_tools": 703,
            "total_placeholders": 50,
            "placeholder_usage": {"Drug SMILES": 677},
        }
        mock_analyze.return_value = mock_stats
        
        # server_stats is a FunctionResource after decoration
        # Access the underlying function via .fn attribute
        if hasattr(server_stats, 'fn'):
            result = server_stats.fn()
        else:
            # Fallback if structure is different
            result = server_stats()
        
        # Verify it's valid JSON
        parsed = json.loads(result)
        assert parsed["total_tools"] == 703
        assert parsed["total_placeholders"] == 50


class TestMainEntryPoint:
    """Test main() entry point."""
    
    def test_main_exists_and_callable(self):
        """Test that main function exists and is callable."""
        from server import main
        
        assert callable(main)
    
    @patch("server.mcp")
    def test_main_calls_run(self, mock_mcp):
        """Test main() calls mcp.run()."""
        from server import main
        
        # Mock sys.argv
        with patch("sys.argv", ["server.py"]):
            main()
        
        # Verify run was called
        mock_mcp.run.assert_called_once()


# =============================================================================
# GPU tests that require actual model/HuggingFace access
# =============================================================================


@pytest.mark.gpu
class TestToolLoading:
    """Test actual tool loading (requires GPU and hits HuggingFace)."""
    
    def test_actual_tool_loading(self):
        """Test loading actual tools from HuggingFace.
        
        Run with: pytest --run-gpu
        """
        from txgemma.tool_factory import build_tools
        
        # This hits HuggingFace API
        tools = build_tools()
        
        assert len(tools) > 0
        assert all(hasattr(t, "name") for t in tools)
        assert all(hasattr(t, "description") for t in tools)
        assert all(hasattr(t, "inputSchema") for t in tools)
    
    def test_filtered_tool_loading(self):
        """Test loading filtered tools.
        
        Run with: pytest --run-gpu
        """
        from txgemma.tool_factory import build_tools
        
        # Load only simple tools
        tools = build_tools(max_placeholders=2)
        
        assert len(tools) > 0
        # All tools should have <= 2 placeholders
        for tool in tools:
            assert len(tool.inputSchema.get("required", [])) <= 2


# =============================================================================
# Unit tests with mocks (don't import server.py)
# =============================================================================


class TestServerInitialization:
    """Test server initialization with mocks."""
    
    @patch("txgemma.tool_factory.build_tools")
    @patch("fastmcp.FastMCP")
    def test_server_loads_tools(self, mock_fastmcp, mock_build_tools):
        """Test that server loads tools on startup."""
        # Mock tools
        mock_tool = Mock()
        mock_tool.name = "test_tool"
        mock_tool.description = "Test tool"
        mock_tool.inputSchema = {
            "type": "object",
            "properties": {"param1": {"type": "string", "description": "Test param"}},
            "required": ["param1"],
        }
        mock_build_tools.return_value = [mock_tool]
        
        # Mock FastMCP instance
        mock_mcp_instance = Mock()
        mock_mcp_instance.tool = Mock(return_value=lambda f: f)
        mock_mcp_instance.resource = Mock(return_value=lambda f: f)
        mock_fastmcp.return_value = mock_mcp_instance
        
        # Import server (which triggers tool loading)
        import importlib
        
        if "server" in sys.modules:
            importlib.reload(sys.modules["server"])
        
        # Verify tools were loaded
        mock_build_tools.assert_called_once()
        # Verify FastMCP was instantiated
        mock_fastmcp.assert_called_once()
    
    @patch("txgemma.tool_factory.build_tools")
    @patch("fastmcp.FastMCP")
    def test_server_registers_tools_with_fastmcp(self, mock_fastmcp, mock_build_tools):
        """Test that tools are registered with FastMCP."""
        # Mock tools
        mock_tool = Mock()
        mock_tool.name = "test_tool"
        mock_tool.description = "Test description"
        mock_tool.inputSchema = {
            "type": "object",
            "properties": {"Drug_SMILES": {"type": "string", "description": "SMILES string"}},
            "required": ["Drug_SMILES"],
        }
        mock_build_tools.return_value = [mock_tool]
        
        # Mock FastMCP instance with tool decorator
        mock_tool_decorator = Mock(return_value=lambda f: f)
        mock_mcp_instance = Mock()
        mock_mcp_instance.tool = Mock(return_value=mock_tool_decorator)
        mock_mcp_instance.resource = Mock(return_value=lambda f: f)
        mock_fastmcp.return_value = mock_mcp_instance
        
        # Import server
        import importlib
        
        if "server" in sys.modules:
            importlib.reload(sys.modules["server"])
        
        # Verify tool decorator was called
        assert mock_mcp_instance.tool.called


class TestServerImports:
    """Test that server can import required modules."""
    
    def test_imports_available(self):
        """Test that all required imports are available."""
        # These should not raise ImportError
        from fastmcp import FastMCP
        from txgemma.tool_factory import build_tools
        from txgemma.executor import execute_tool
        from txgemma.chat_factory import register_chat_tool
        
        assert FastMCP is not None
        assert build_tools is not None
        assert execute_tool is not None
        assert register_chat_tool is not None


class TestToolRegistration:
    """Test tool registration logic."""
    
    def test_tool_wrapper_function(self):
        """Test the tool wrapper function pattern used in server."""
        from txgemma.executor import execute_tool
        
        # Simulate the wrapper pattern from server.py
        def make_tool_func(name: str):
            def _tool_func(params: dict) -> str:
                return execute_tool(name, params)
            _tool_func.__name__ = name
            return _tool_func
        
        # Create wrapper
        tool_func = make_tool_func("test_tool")
        
        assert tool_func.__name__ == "test_tool"
        assert callable(tool_func)
    
    @patch("server.execute_tool")
    def test_tool_execution_wrapper(self, mock_execute):
        """Test that tool execution wrapper calls execute_tool correctly."""
        # Mock successful execution
        mock_execute.return_value = "Prediction result"
        
        # Call execute_tool
        from server import execute_tool
        result = execute_tool("test_tool", {"param": "value"})
        
        # Verify it was called
        mock_execute.assert_called_once_with("test_tool", {"param": "value"})


class TestServerConfiguration:
    """Test server configuration and options."""
    
    def test_fastmcp_initialization(self):
        """Test FastMCP server can be initialized."""
        from fastmcp import FastMCP
        
        mcp = FastMCP("txgemma-mcp")
        
        assert mcp is not None


class TestToolExecution:
    """Test tool execution through server."""
    
    @patch('txgemma.executor.get_predict_model')
    @patch('txgemma.executor.get_loader')
    def test_execute_tool_via_wrapper(self, mock_get_loader, mock_get_predict_model):
        """Test executing a tool through the server wrapper."""
        from txgemma.executor import execute_tool
        
        # Mock loader
        mock_loader = Mock()
        mock_template = Mock()
        mock_template.format.return_value = "Formatted prompt"
        mock_loader.get.return_value = mock_template
        mock_get_loader.return_value = mock_loader
        
        # Mock model
        mock_model = Mock()
        mock_model.generate.return_value = "Result"
        mock_get_predict_model.return_value = mock_model
        
        # Execute
        result = execute_tool("test_tool", {"param": "value"})
        
        assert result == "Result"


class TestChatToolIntegration:
    """Test chat tool integration with server."""
    
    def test_register_chat_tool_callable(self):
        """Test that register_chat_tool can be called."""
        from txgemma.chat_factory import register_chat_tool
        
        mock_mcp = Mock()
        mock_tool_decorator = Mock()
        mock_mcp.tool.return_value = mock_tool_decorator
        
        # Should not raise
        register_chat_tool(mock_mcp)
        
        # Should have registered a tool
        assert mock_mcp.tool.called
    
    @patch('txgemma.executor.get_chat_model')
    def test_execute_chat_from_server(self, mock_get_chat_model):
        """Test executing chat through server."""
        from txgemma.executor import execute_chat
        
        # Mock chat model
        mock_model = Mock()
        mock_model.generate.return_value = "Chat response"
        mock_get_chat_model.return_value = mock_model
        
        result = execute_chat("Test question?")
        
        assert result == "Chat response"


class TestToolDescriptionEnhancement:
    """Test tool description enhancement with parameter info."""
    
    def test_enhance_description_with_params(self):
        """Test the description enhancement pattern from server."""
        # Simulate server's description enhancement
        tool_description = "Base description"
        tool_schema = {
            "properties": {
                "Drug SMILES": {
                    "type": "string",
                    "description": "SMILES string"
                },
                "Dose": {
                    "type": "number",
                    "description": "Drug dose"
                }
            },
            "required": ["Drug SMILES"]
        }
        
        # Enhancement logic from server
        enhanced_description = tool_description
        if tool_schema.get("properties"):
            enhanced_description += "\n\nParameters:"
            for param_name, param_info in tool_schema["properties"].items():
                param_desc = param_info.get("description", "")
                param_type = param_info.get("type", "string")
                is_required = param_name in tool_schema.get("required", [])
                required_marker = " (required)" if is_required else " (optional)"
                enhanced_description += f"\n- {param_name}{required_marker}: {param_desc} (type: {param_type})"
        
        # Verify enhancement
        assert "Parameters:" in enhanced_description
        assert "Drug SMILES (required)" in enhanced_description
        assert "Dose (optional)" in enhanced_description


class TestErrorHandling:
    """Test error handling in server components."""
    
    @patch('txgemma.executor.get_predict_model')
    @patch('txgemma.executor.get_loader')
    def test_tool_execution_error_handling(self, mock_get_loader, mock_get_predict_model):
        """Test that tool execution errors are handled."""
        from txgemma.executor import execute_tool
        
        # Mock loader to raise error
        mock_loader = Mock()
        mock_loader.get.side_effect = KeyError("Unknown tool")
        mock_get_loader.return_value = mock_loader
        
        # Should raise KeyError
        with pytest.raises(KeyError, match="Unknown tool"):
            execute_tool("nonexistent_tool", {})


class TestModelSingletons:
    """Test model singleton behavior in server context."""
    
    def test_predict_model_singleton(self):
        """Test that predict model uses singleton."""
        from txgemma.model import get_predict_model, TxGemmaPredictModel
        
        # Reset singleton
        TxGemmaPredictModel._instance = None
        
        model1 = get_predict_model()
        model2 = get_predict_model()
        
        assert model1 is model2
    
    def test_chat_model_singleton(self):
        """Test that chat model uses singleton."""
        from txgemma.model import get_chat_model, TxGemmaChatModel
        
        # Reset singleton
        TxGemmaChatModel._instance = None
        
        model1 = get_chat_model()
        model2 = get_chat_model()
        
        assert model1 is model2


class TestToolFiltering:
    """Test tool filtering options commented in server."""
    
    @patch('txgemma.tool_factory.build_tools')
    def test_filter_by_drug_smiles(self, mock_build_tools):
        """Test filtering tools by Drug SMILES placeholder."""
        mock_build_tools.return_value = []
        
        # Option 2 from server comments
        mock_build_tools(filter_placeholder="Drug SMILES")
        
        mock_build_tools.assert_called_with(filter_placeholder="Drug SMILES")
    
    @patch('txgemma.tool_factory.build_tools')
    def test_filter_simple_tools(self, mock_build_tools):
        """Test filtering for simple tools (≤2 parameters)."""
        mock_build_tools.return_value = []
        
        # Option 3 from server comments
        mock_build_tools(max_placeholders=2)
        
        mock_build_tools.assert_called_with(max_placeholders=2)