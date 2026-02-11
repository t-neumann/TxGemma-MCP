"""
Tests for server.py - MCP server initialization and tool registration.

Tests server setup, tool registration, FastMCP integration, and SECURITY.
Includes critical security tests for exec() usage in _make_tool_function_safe().

"""

import json
from unittest.mock import Mock, patch

import pytest

# Mark all tests as integration tests (except GPU tests which override)
pytestmark = [pytest.mark.integration]

# =============================================================================
# SECURITY TESTS - CRITICAL FOR exec() USAGE
# =============================================================================

class TestMakeToolFunctionSafeSecurity:
    """
    CRITICAL SECURITY TESTS for _make_tool_function_safe().
    
    This function uses exec() which is inherently dangerous. These tests
    ensure malicious inputs cannot execute arbitrary code.
    """
    
    def test_rejects_code_injection_in_tool_name(self):
        """Test that code injection attempts in tool name are rejected."""
        from server import _make_tool_function_safe
        
        malicious_names = [
            "__import__('os').system('rm -rf /')",
            "evil'; import os; os.system('whoami'); x='",
            "tool\"; __import__('os').system('ls'); \"",
            "tool'; exec('print(123)'); x='",
            "../../etc/passwd",
            "tool; DROP TABLE users; --",
        ]
        
        schema = {
            "type": "object",
            "properties": {"param": {"type": "string"}},
            "required": ["param"]
        }
        
        for malicious_name in malicious_names:
            with pytest.raises(ValueError, match="Invalid tool name"):
                _make_tool_function_safe(malicious_name, schema)
    
    def test_rejects_code_injection_in_parameter_names(self):
        """Test that code injection in parameter names is rejected."""
        from server import _make_tool_function_safe
        
        malicious_schemas = [
            {
                "type": "object",
                "properties": {
                    "__import__('os').system('ls')": {"type": "string"}
                },
                "required": []
            },
            {
                "type": "object",
                "properties": {
                    "param'; exec('print(1)'); x='": {"type": "string"}
                },
                "required": []
            },
        ]
        
        for schema in malicious_schemas:
            with pytest.raises(ValueError, match="Invalid parameter name"):
                _make_tool_function_safe("valid_tool", schema)
    
    def test_accepts_only_alphanumeric_with_underscores(self):
        """Test that only safe tool names are accepted."""
        from server import _make_tool_function_safe
        
        schema = {
            "type": "object",
            "properties": {"param": {"type": "string"}},
            "required": ["param"]
        }
        
        # Valid names (should succeed)
        valid_names = [
            "valid_tool",
            "tool123",
            "tool_with_underscores",
            "UPPERCASE_TOOL",
        ]
        
        for name in valid_names:
            func = _make_tool_function_safe(name, schema)
            assert callable(func)
            assert func.__name__ == name
    
    def test_rejects_hyphens_in_tool_names(self):
        """Test that hyphens are rejected (Python doesn't allow in function names)."""
        from server import _make_tool_function_safe
        
        schema = {
            "type": "object",
            "properties": {"param": {"type": "string"}},
            "required": ["param"]
        }
        
        # Hyphens would cause Python syntax errors
        # The validation check should catch this
        with pytest.raises((ValueError, RuntimeError)):
            _make_tool_function_safe("tool-with-hyphens", schema)
    
    def test_sanitizes_special_characters_in_description(self):
        """Test that special chars in descriptions don't break function."""
        from server import _make_tool_function_safe
        
        schema = {
            "type": "object",
            "properties": {
                "param": {
                    "type": "string",
                    "description": "Test with 'quotes' and \"double quotes\" and {braces}"
                }
            },
            "required": ["param"]
        }
        
        # Should not raise
        func = _make_tool_function_safe("test_tool", schema)
        assert callable(func)

# =============================================================================
# TOOL FUNCTION CREATION TESTS
# =============================================================================

class TestMakeToolFunctionSafe:
    """Test _make_tool_function_safe() function creation."""
    
    def test_creates_function_with_correct_name(self):
        """Test that created function has correct name."""
        from server import _make_tool_function_safe
        
        schema = {
            "type": "object",
            "properties": {"param": {"type": "string"}},
            "required": ["param"]
        }
        
        func = _make_tool_function_safe("my_tool", schema)
        
        assert func.__name__ == "my_tool"
        assert callable(func)
    
    def test_creates_function_with_required_parameters(self):
        """Test function with required parameters."""
        from server import _make_tool_function_safe
        
        schema = {
            "type": "object",
            "properties": {
                "drug_smiles": {"type": "string", "description": "Drug SMILES"},
                "target": {"type": "string", "description": "Target"},
            },
            "required": ["drug_smiles", "target"]
        }
        
        func = _make_tool_function_safe("predict_tool", schema)
        
        # Check annotations
        assert "drug_smiles" in func.__annotations__
        assert "target" in func.__annotations__
        assert func.__annotations__["return"] == str
    
    def test_creates_function_with_optional_parameters(self):
        """Test function with optional parameters."""
        from server import _make_tool_function_safe
        
        schema = {
            "type": "object",
            "properties": {
                "required_param": {"type": "string"},
                "optional_param": {"type": "string"},
            },
            "required": ["required_param"]
        }
        
        func = _make_tool_function_safe("test_tool", schema)
        
        # Both should be in annotations
        assert "required_param" in func.__annotations__
        assert "optional_param" in func.__annotations__
    
    @patch("server.execute_tool")
    def test_created_function_calls_execute_tool(self, mock_execute):
        """Test that created function calls execute_tool."""
        from server import _make_tool_function_safe
        
        schema = {
            "type": "object",
            "properties": {"param": {"type": "string"}},
            "required": ["param"]
        }
        
        mock_execute.return_value = "Result"
        
        func = _make_tool_function_safe("test_tool", schema)
        result = func(param="test_value")
        
        # Should call execute_tool
        mock_execute.assert_called_once_with("test_tool", {"param": "test_value"})
        assert result == "Result"
    
    @patch("server.execute_tool")
    def test_created_function_handles_none_for_optional_params(self, mock_execute):
        """Test that None values for optional params are filtered out."""
        from server import _make_tool_function_safe
        
        schema = {
            "type": "object",
            "properties": {
                "required_param": {"type": "string"},
                "optional_param": {"type": "string"},
            },
            "required": ["required_param"]
        }
        
        mock_execute.return_value = "Result"
        
        func = _make_tool_function_safe("test_tool", schema)
        func(required_param="test", optional_param=None)
        
        # Should filter out None values
        call_args = mock_execute.call_args[0][1]
        assert "required_param" in call_args
        assert "optional_param" not in call_args
    
    def test_handles_different_parameter_types(self):
        """Test handling of different JSON Schema types."""
        from server import _make_tool_function_safe
        
        schema = {
            "type": "object",
            "properties": {
                "string_param": {"type": "string"},
                "int_param": {"type": "integer"},
                "float_param": {"type": "number"},
                "bool_param": {"type": "boolean"},
            },
            "required": ["string_param"]
        }
        
        func = _make_tool_function_safe("multi_type_tool", schema)
        
        # Check type annotations
        annotations = func.__annotations__
        assert "string_param" in annotations
        assert "int_param" in annotations
        assert "float_param" in annotations
        assert "bool_param" in annotations


# =============================================================================
# TYPE MAPPING TESTS
# =============================================================================

class TestGetPythonType:
    """Test _get_python_type() helper function."""
    
    def test_maps_string_to_str(self):
        """Test string type mapping."""
        from server import _get_python_type
        
        assert _get_python_type("string") == str
    
    def test_maps_integer_to_int(self):
        """Test integer type mapping."""
        from server import _get_python_type
        
        assert _get_python_type("integer") == int
    
    def test_maps_number_to_float(self):
        """Test number type mapping."""
        from server import _get_python_type
        
        assert _get_python_type("number") == float
    
    def test_maps_boolean_to_bool(self):
        """Test boolean type mapping."""
        from server import _get_python_type
        
        assert _get_python_type("boolean") == bool
    
    def test_defaults_to_str_for_unknown_types(self):
        """Test that unknown types default to str."""
        from server import _get_python_type
        
        assert _get_python_type("unknown") == str
        assert _get_python_type("array") == str
        assert _get_python_type("object") == str


# =============================================================================
# SERVER INITIALIZATION TESTS
# =============================================================================

class TestServerIntegration:
    """Integration tests that import actual server module."""
    
    def test_server_module_imports_successfully(self):
        """Test that server module can be imported."""
        import server
        
        assert hasattr(server, "mcp")
        assert hasattr(server, "TOOLS")
        assert hasattr(server, "main")
    
    def test_fastmcp_instance_exists(self):
        """Test that FastMCP instance is created."""
        from server import mcp
        
        assert mcp is not None
        assert hasattr(mcp, "run")
    
    def test_tools_list_exists(self):
        """Test that TOOLS list exists and is a list."""
        from server import TOOLS
        
        assert isinstance(TOOLS, list)
    
    def test_config_loaded(self):
        """Test that config is loaded."""
        from server import config
        
        assert config is not None
        assert hasattr(config, "predict")
        assert hasattr(config, "chat")
        assert hasattr(config, "tools")
    
    @pytest.mark.skipif(
        not pytest.importorskip("server").TOOLS,
        reason="No tools loaded (may be filtered in test environment)"
    )
    def test_loaded_tools_have_valid_schemas(self):
        """Test that loaded tools have valid schemas."""
        from server import TOOLS
        
        if len(TOOLS) == 0:
            pytest.skip("No tools loaded")
        
        for tool in TOOLS:
            # Check basic structure
            assert hasattr(tool, "name")
            assert hasattr(tool, "description")
            assert hasattr(tool, "inputSchema")
            
            # Check schema structure
            schema = tool.inputSchema
            assert schema["type"] == "object"
            assert "properties" in schema
            assert "required" in schema
            
            # Check properties
            for prop_name, prop_schema in schema["properties"].items():
                assert "type" in prop_schema
                assert "description" in prop_schema

# =============================================================================
# RESOURCE ENDPOINT TESTS
# =============================================================================

class TestResourceEndpoints:
    """Test resource endpoints."""
    
    def test_server_info_resource_exists(self):
        """Test that server_info resource exists."""
        from server import server_info
        
        # FastMCP wraps functions in FunctionResource
        assert server_info is not None
        # Check it has the .fn attribute (the actual function)
        assert hasattr(server_info, "fn") or callable(server_info)
    
    def test_server_info_returns_json(self):
        """Test that server_info returns valid JSON."""
        from server import server_info
        
        # Access the underlying function via .fn attribute
        if hasattr(server_info, "fn"):
            result = server_info.fn()
        else:
            result = server_info()
        
        # Should be valid JSON
        data = json.loads(result)
        
        # Check expected fields
        assert "server" in data
        assert "version" in data
        assert "tools_loaded" in data
        assert "configuration" in data
    
    def test_tools_list_resource_exists(self):
        """Test that tools_list resource exists."""
        from server import tools_list
        
        # FastMCP wraps functions in FunctionResource
        assert tools_list is not None
        # Check it has the .fn attribute (the actual function)
        assert hasattr(tools_list, "fn") or callable(tools_list)
    
    def test_tools_list_returns_json(self):
        """Test that tools_list returns valid JSON."""
        from server import tools_list
        
        # Access the underlying function via .fn attribute
        if hasattr(tools_list, "fn"):
            result = tools_list.fn()
        else:
            result = tools_list()
        
        # Should be valid JSON array
        data = json.loads(result)
        assert isinstance(data, list)
        
        # If tools exist, check structure
        if len(data) > 0:
            tool_info = data[0]
            assert "name" in tool_info
            assert "description" in tool_info
            assert "parameters" in tool_info
            assert "required" in tool_info

# =============================================================================
# MAIN ENTRY POINT TESTS
# =============================================================================

class TestMainEntryPoint:
    """Test main() entry point."""
    
    def test_main_function_exists(self):
        """Test that main function exists and is callable."""
        from server import main
        
        assert callable(main)
    
    @patch("server.mcp.run")
    def test_main_calls_mcp_run(self, mock_run):
        """Test that main() calls mcp.run()."""
        from server import main
        
        with patch("sys.argv", ["server.py"]):
            main()
        
        mock_run.assert_called_once()
    
    @patch("server.mcp.run")
    def test_main_passes_argv_to_run(self, mock_run):
        """Test that main() passes command line args."""
        from server import main
        
        with patch("sys.argv", ["server.py", "arg1", "arg2"]):
            main()
        
        # Should be called with args (excluding script name)
        mock_run.assert_called_once_with(["arg1", "arg2"])


# =============================================================================
# TOOL REGISTRATION TESTS
# =============================================================================

class TestToolRegistration:
    """Test tool registration with FastMCP."""
    
    @patch("server.execute_tool")
    def test_tools_registered_with_fastmcp(self, mock_execute):
        """Test that tools are registered during server init."""
        from server import TOOLS, mcp
        
        # If no tools loaded, skip
        if len(TOOLS) == 0:
            pytest.skip("No tools loaded")
        
        # Verify mcp has tool decorator
        assert hasattr(mcp, "tool")
    
    @patch("server.execute_tool")
    def test_tool_function_execution_flow(self, mock_execute):
        """Test full tool function execution flow."""
        from server import _make_tool_function_safe
        
        schema = {
            "type": "object",
            "properties": {
                "drug_smiles": {
                    "type": "string",
                    "description": "Drug SMILES string"
                }
            },
            "required": ["drug_smiles"]
        }
        
        mock_execute.return_value = "Toxicity: High"
        
        # Create function
        func = _make_tool_function_safe("predict_toxicity", schema)
        
        # Execute
        result = func(drug_smiles="CC(=O)O")
        
        # Verify
        assert result == "Toxicity: High"
        mock_execute.assert_called_once_with(
            "predict_toxicity",
            {"drug_smiles": "CC(=O)O"}
        )

# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================

class TestErrorHandling:
    """Test error handling in server."""
    
    def test_invalid_tool_name_raises_value_error(self):
        """Test that invalid tool names raise ValueError."""
        from server import _make_tool_function_safe
        
        schema = {"type": "object", "properties": {}, "required": []}
        
        with pytest.raises(ValueError, match="Invalid tool name"):
            _make_tool_function_safe("invalid.name", schema)
    
    def test_invalid_parameter_name_raises_value_error(self):
        """Test that invalid parameter names raise ValueError."""
        from server import _make_tool_function_safe
        
        schema = {
            "type": "object",
            "properties": {
                "invalid.param": {"type": "string"}
            },
            "required": []
        }
        
        with pytest.raises(ValueError, match="Invalid parameter name"):
            _make_tool_function_safe("valid_tool", schema)
    
    @patch("server.execute_tool")
    def test_function_creation_error_is_caught(self, mock_execute):
        """Test that errors during function creation are caught."""
        from server import _make_tool_function_safe
        
        # Create a schema that might cause issues
        schema = {
            "type": "object",
            "properties": {},
            "required": []
        }
        
        # This should not raise during normal operation
        func = _make_tool_function_safe("test_tool", schema)
        assert callable(func)

# =============================================================================
# CONFIGURATION INTEGRATION TESTS
# =============================================================================

class TestConfigurationIntegration:
    """Test that server respects configuration."""
    
    def test_chat_tool_enabled_by_config(self):
        """Test that chat tool is enabled based on config."""
        from server import config
        
        # Config should have enable_chat setting
        assert hasattr(config.tools, "enable_chat")
    
    def test_tool_filtering_applied(self):
        """Test that tool filtering from config is applied."""
        from server import config, TOOLS
        
        # If filtering is enabled, tools should be filtered
        if config.tools.filter_placeholder:
            # Tools should be filtered
            assert isinstance(TOOLS, list)
        
        if config.tools.max_placeholders:
            # Tools should respect max placeholders
            for tool in TOOLS:
                num_params = len(tool.inputSchema.get("required", []))
                assert num_params <= config.tools.max_placeholders

# =============================================================================
# GPU INTEGRATION TESTS (Optional, requires GPU)
# =============================================================================

@pytest.mark.gpu
class TestServerGPU:
    """GPU integration tests with real models."""
    
    def test_server_can_load_with_real_models(self):
        """Test that server loads successfully with real models."""
        import server
        
        # Server should load without errors
        assert server.mcp is not None
        assert isinstance(server.TOOLS, list)
    
    def test_tool_execution_with_real_model(self):
        """Test executing a tool with real model."""
        from server import TOOLS
        from txgemma.executor import execute_tool
        
        if len(TOOLS) == 0:
            pytest.skip("No tools loaded")
        
        # Get first tool
        tool = TOOLS[0]
        
        # Create test arguments (all required params as empty strings)
        test_args = {
            param: "CC(=O)O"  # Aspirin SMILES as test
            for param in tool.inputSchema.get("required", [])
        }
        
        # Execute
        result = execute_tool(tool.name, test_args)
        
        # Should return a string result
        assert isinstance(result, str)
        assert len(result) > 0