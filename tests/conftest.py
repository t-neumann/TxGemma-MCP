"""
Pytest configuration and shared fixtures.

Provides:
- Custom CLI options (--run-gpu)
- Marker-based test skipping
- Shared fixtures for common test needs
"""

import pytest
from unittest.mock import Mock

# =========================================================================
# CLI OPTIONS
# =========================================================================

def pytest_addoption(parser):
    """Add custom command-line options to pytest."""
    parser.addoption(
        "--run-gpu",
        action="store_true",
        default=False,
        help="Run tests that require GPU (marked with @pytest.mark.gpu)",
    )

# =========================================================================
# MARKER-BASED SKIPPING
# =========================================================================

def pytest_collection_modifyitems(config, items):
    """Skip GPU tests unless explicitly requested."""
    run_gpu = config.getoption("--run-gpu")
    skip_gpu = pytest.mark.skip(reason="need --run-gpu option to run")

    for item in items:
        if "gpu" in item.keywords and not run_gpu:
            item.add_marker(skip_gpu)

# =========================================================================
# SHARED FIXTURES
# =========================================================================

@pytest.fixture(autouse=True)
def reset_caches():
    """
    Reset all module-level caches before each test.
    
    This ensures test isolation by clearing:
    - Parameter mapping cache
    - Configuration singleton
    
    Runs automatically for all tests.
    """
    from txgemma.cache_utils import reset_all_caches
    reset_all_caches()
    yield
    # Cleanup after test if needed

@pytest.fixture
def mock_predict_model():
    """
    Mock TxGemmaPredictModel for testing.
    
    Returns a mock that simulates model behavior without
    actually loading the model.
    
    Example:
        def test_something(mock_predict_model):
            result = mock_predict_model.generate("test")
            assert result == "mocked result"
    """
    mock = Mock()
    mock.generate.return_value = "mocked result"
    mock.is_loaded = True
    mock.model_name = "test-model"
    mock.max_new_tokens = 64
    return mock

@pytest.fixture
def mock_chat_model():
    """
    Mock TxGemmaChatModel for testing.
    
    Returns a mock that simulates chat model behavior without
    actually loading the model.
    
    Example:
        def test_chat(mock_chat_model):
            result = mock_chat_model.generate("question")
            assert result == "mocked chat response"
    """
    mock = Mock()
    mock.generate.return_value = "mocked chat response"
    mock.is_loaded = True
    mock.model_name = "test-chat-model"
    mock.max_new_tokens = 100
    return mock

@pytest.fixture
def mock_prompt_loader():
    """
    Mock PromptLoader for testing.
    
    Returns a mock loader with a mock template that formats correctly.
    
    Example:
        def test_executor(mock_prompt_loader):
            template = mock_prompt_loader.get("tool_name")
            result = template.format(param="value")
            assert result == "formatted prompt"
    """
    mock_loader = Mock()
    mock_template = Mock()
    mock_template.format.return_value = "formatted prompt"
    mock_template.placeholder_count.return_value = 1
    mock_template.placeholders = ["Drug SMILES"]
    mock_loader.get.return_value = mock_template
    return mock_loader

@pytest.fixture
def sample_parameter_mapping():
    """
    Sample parameter mapping for testing.
    
    Maps normalized names (snake_case) to original placeholder names.
    
    Example:
        def test_mapping(sample_parameter_mapping):
            assert sample_parameter_mapping["drug_smiles"] == "Drug SMILES"
    """
    return {
        "drug_smiles": "Drug SMILES",
        "target_sequence": "Target sequence",
        "protein_sequence": "Protein sequence",
        "cell_line": "Cell line",
        "dose": "Dose",
    }

@pytest.fixture
def temp_config(tmp_path):
    """
    Create temporary config file for testing.
    
    Creates a valid config.yaml in a temp directory.
    
    Example:
        def test_config_loading(temp_config):
            from txgemma.config import load_config
            config = load_config(temp_config)
            assert config.predict.model == "test-model"
    """
    import yaml
    
    config_data = {
        "predict": {
            "model": "test-model",
            "max_new_tokens": 32,
        },
        "chat": {
            "model": "test-chat-model",
            "max_new_tokens": 64,
        },
        "tools": {
            "enable_chat": True,
            "filter_placeholder": None,
            "max_placeholders": None,
        }
    }
    
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)
    
    return config_path

@pytest.fixture
def sample_tool_schema():
    """
    Sample MCP tool schema for testing.
    
    Returns a valid tool schema with parameters.
    
    Example:
        def test_tool_creation(sample_tool_schema):
            assert "properties" in sample_tool_schema
            assert "drug_smiles" in sample_tool_schema["properties"]
    """
    return {
        "type": "object",
        "properties": {
            "drug_smiles": {
                "type": "string",
                "description": "SMILES string of drug molecule"
            },
            "target_sequence": {
                "type": "string",
                "description": "Amino acid sequence of target protein"
            }
        },
        "required": ["drug_smiles"]
    }

@pytest.fixture
def mock_mcp_server():
    """
    Mock FastMCP server for testing tool registration.
    
    Returns a mock MCP server with tool registration capability.
    
    Example:
        def test_registration(mock_mcp_server):
            register_chat_tool(mock_mcp_server)
            assert mock_mcp_server.tool.called
    """
    mock = Mock()
    mock_decorator = Mock()
    mock.tool.return_value = mock_decorator
    return mock

# =========================================================================
# PYTEST CONFIGURATION HOOKS
# =========================================================================

def pytest_configure(config):
    """
    Configure pytest with additional settings.
    
    This runs once before test collection.
    """
    # Could add dynamic configuration here if needed
    pass

def pytest_sessionstart(session):
    """
    Called before test session starts.
    
    Good place for one-time setup like downloading resources.
    """
    pass

def pytest_sessionfinish(session, exitstatus):
    """
    Called after entire test session.
    
    Good place for cleanup or reporting.
    """
    pass