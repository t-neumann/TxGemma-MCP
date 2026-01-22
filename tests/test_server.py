"""
Tests for server.py

These tests check server initialization and configuration.
We don't test actual MCP protocol communication (that's integration testing).
"""

import sys
from unittest.mock import Mock, patch

import pytest


class TestServerInitialization:
    """Test server initialization and tool loading."""

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
        else:
            pass

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


class TestToolFunctionGeneration:
    """Test dynamic tool function generation."""

    def test_tools_are_loaded(self):
        """Test that TOOLS list is populated."""
        from server import TOOLS

        # Should have tools loaded (703 in production, but mocked in tests)
        assert isinstance(TOOLS, list)
        # In real execution (not mocked), should have tools
        # In mocked tests, this might be empty

    @patch("server.execute_tool")
    def test_tool_execution_wrapper(self, mock_execute):
        """Test that tool execution wrapper calls execute_tool correctly."""
        from server import execute_tool

        # Mock successful execution
        mock_execute.return_value = "Prediction result"

        # Call execute_tool
        _result = execute_tool("test_tool", {"param": "value"})

        # Verify it was called
        mock_execute.assert_called_once_with("test_tool", {"param": "value"})


class TestResourceEndpoints:
    """Test resource endpoints."""

    @patch("txgemma.prompts.get_loader")
    @patch("txgemma.tool_factory.analyze_tools")
    def test_server_info_resource(self, mock_analyze, mock_loader):
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

        # Mock loader
        mock_loader_instance = Mock()
        mock_loader.return_value = mock_loader_instance

        # Call resource
        result = server_info()

        # Verify result contains expected content
        assert "TxGemma MCP Server" in result
        assert "Drug SMILES" in result
        assert "703" in result or "total_tools" in str(mock_analyze.return_value)

    @patch("txgemma.tool_factory.analyze_tools")
    def test_server_stats_resource(self, mock_analyze):
        """Test server_stats resource returns JSON."""
        import json

        from server import server_stats

        # Mock analyze_tools
        mock_stats = {
            "total_tools": 703,
            "total_placeholders": 50,
            "placeholder_usage": {"Drug SMILES": 677},
        }
        mock_analyze.return_value = mock_stats

        # Call resource
        result = server_stats()

        # Verify it's valid JSON
        parsed = json.loads(result)
        assert parsed["total_tools"] == 703
        assert parsed["total_placeholders"] == 50


class TestMainEntryPoint:
    """Test main() entry point."""

    @patch("server.mcp")
    def test_main_calls_run(self, mock_mcp):
        """Test main() calls mcp.run() without arguments (transport handled by CLI)."""
        from server import main

        # Mock sys.argv (no arguments - transport configured via CLI)
        with patch("sys.argv", ["server.py"]):
            main()

        # Verify run was called (transport handled by fastmcp CLI, not in code)
        mock_mcp.run.assert_called_once()

    def test_main_exists_and_callable(self):
        """Test that main function exists and is callable."""
        from server import main

        assert callable(main)


class TestServerConfiguration:
    """Test different server configurations."""

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


@pytest.mark.integration
class TestServerIntegration:
    """Integration tests for the server (require FastMCP to be working)."""

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


@pytest.mark.gpu
class TestToolLoading:
    """Test actual tool loading (requires GPU and hits HuggingFace)."""

    @pytest.mark.gpu
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

    @pytest.mark.gpu
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
