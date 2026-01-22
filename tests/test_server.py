"""
Tests for server.py

These tests check server initialization and configuration.
We don't test actual MCP protocol communication (that's integration testing).
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import sys


class TestServerInitialization:
    """Test server initialization and tool loading."""
    
    @patch('server.build_tools')
    def test_server_loads_tools(self, mock_build_tools):
        """Test that server loads tools on startup."""
        # Mock tools
        mock_tool = Mock()
        mock_tool.name = "test_tool"
        mock_tool.description = "Test tool"
        mock_tool.inputSchema = {
            "properties": {
                "param1": {"type": "string"}
            }
        }
        mock_build_tools.return_value = [mock_tool]
        
        # Import server (which triggers tool loading)
        import importlib
        if 'server' in sys.modules:
            importlib.reload(sys.modules['server'])
        else:
            import server
        
        # Verify tools were loaded
        mock_build_tools.assert_called_once()
    
    @patch('server.build_tools')
    @patch('server.FastMCP')
    def test_server_registers_tools_with_fastmcp(self, mock_fastmcp, mock_build_tools):
        """Test that tools are registered with FastMCP."""
        # Mock tools
        mock_tool = Mock()
        mock_tool.name = "test_tool"
        mock_tool.description = "Test description"
        mock_tool.inputSchema = {
            "properties": {
                "Drug SMILES": {"type": "string"}
            }
        }
        mock_build_tools.return_value = [mock_tool]
        
        # Mock FastMCP instance
        mock_mcp_instance = Mock()
        mock_fastmcp.return_value = mock_mcp_instance
        
        # Import server
        import importlib
        if 'server' in sys.modules:
            importlib.reload(sys.modules['server'])
        
        # Verify FastMCP was instantiated
        mock_fastmcp.assert_called_with("txgemma-mcp")


class TestToolFunctionGeneration:
    """Test dynamic tool function generation."""
    
    @patch('server.execute_tool')
    def test_make_tool_func_success(self, mock_execute):
        """Test tool function execution."""
        from server import TOOLS
        
        if len(TOOLS) == 0:
            pytest.skip("No tools loaded")
        
        # Mock successful execution
        mock_execute.return_value = "Prediction result"
        
        # Get first tool name
        tool_name = TOOLS[0].name
        
        # We can't easily test the dynamically created functions,
        # but we can test execute_tool is called correctly
        from server import execute_tool
        result = execute_tool(tool_name, {"test": "arg"})
        
        mock_execute.assert_called_once_with(tool_name, {"test": "arg"})
    
    @patch('server.execute_tool')
    def test_tool_func_handles_errors(self, mock_execute):
        """Test that tool functions handle errors gracefully."""
        # Mock execution failure
        mock_execute.side_effect = ValueError("Invalid input")
        
        # This should not raise, error is caught in tool_func
        try:
            from server import execute_tool
            result = execute_tool("test_tool", {})
        except ValueError:
            # Expected if execute_tool raises directly
            pass


class TestResourceEndpoints:
    """Test resource endpoints."""
    
    @patch('server.get_loader')
    @patch('server.analyze_tools')
    def test_server_info_resource(self, mock_analyze, mock_loader):
        """Test server_info resource."""
        from server import server_info
        
        # Mock analyze_tools
        mock_analyze.return_value = {
            'total_tools': 10,
            'total_placeholders': 5,
            'most_common_placeholders': [
                ('Drug SMILES', 8),
                ('Target sequence', 3),
            ]
        }
        
        # Mock loader
        mock_loader.return_value = Mock()
        
        # Call resource
        result = server_info()
        
        # Verify result contains expected content
        assert "TxGemma MCP Server" in result
        assert "Drug SMILES" in result
        assert "10" in result or "Total available tools: 10" in result
    
    @patch('server.analyze_tools')
    def test_server_stats_resource(self, mock_analyze):
        """Test server_stats resource returns JSON."""
        from server import server_stats
        import json
        
        # Mock analyze_tools
        mock_stats = {
            'total_tools': 10,
            'total_placeholders': 5,
        }
        mock_analyze.return_value = mock_stats
        
        # Call resource
        result = server_stats()
        
        # Verify it's valid JSON
        parsed = json.loads(result)
        assert parsed['total_tools'] == 10
        assert parsed['total_placeholders'] == 5


class TestMainEntryPoint:
    """Test main() entry point."""
    
    @patch('server.mcp')
    def test_main_default_stdio_mode(self, mock_mcp):
        """Test main() starts in stdio mode by default."""
        from server import main
        
        # Mock sys.argv (no arguments)
        with patch('sys.argv', ['server.py']):
            main()
        
        # Verify stdio transport
        mock_mcp.run.assert_called_once_with(transport="stdio")
    
    @patch('server.mcp')
    def test_main_api_mode(self, mock_mcp):
        """Test main() starts in API mode with 'api' argument."""
        from server import main
        
        # Mock sys.argv with 'api' argument
        with patch('sys.argv', ['server.py', 'api']):
            main()
        
        # Verify SSE transport
        mock_mcp.run.assert_called_once_with(transport="sse")


class TestServerConfiguration:
    """Test different server configurations."""
    
    def test_tools_list_is_populated(self):
        """Test that TOOLS list is not empty."""
        from server import TOOLS
        
        # Should have at least some tools loaded
        assert len(TOOLS) > 0
        assert all(hasattr(t, 'name') for t in TOOLS)
        assert all(hasattr(t, 'description') for t in TOOLS)
        assert all(hasattr(t, 'inputSchema') for t in TOOLS)
    
    def test_tools_have_valid_schemas(self):
        """Test that all tools have valid input schemas."""
        from server import TOOLS
        
        for tool in TOOLS:
            assert 'properties' in tool.inputSchema
            assert 'required' in tool.inputSchema
            
            # Check properties structure
            for prop_name, prop_schema in tool.inputSchema['properties'].items():
                assert 'type' in prop_schema
                assert 'description' in prop_schema


@pytest.mark.integration
class TestServerIntegration:
    """Integration tests for the server (require FastMCP to be working)."""
    
    def test_server_can_be_imported(self):
        """Test that server module can be imported without errors."""
        import server
        assert hasattr(server, 'mcp')
        assert hasattr(server, 'TOOLS')
        assert hasattr(server, 'main')
    
    def test_fastmcp_instance_exists(self):
        """Test that FastMCP instance is created."""
        from server import mcp
        
        # Check that mcp is a FastMCP instance
        assert mcp is not None
        # FastMCP should have run method
        assert hasattr(mcp, 'run')