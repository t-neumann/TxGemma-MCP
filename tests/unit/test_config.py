"""
Tests for txgemma.config module.

Tests configuration loading, validation, and environment variable overrides.
Uses isolated test fixtures - does NOT depend on project config.yaml.

"""

import os
from pathlib import Path
from unittest.mock import mock_open, patch

import pytest
import yaml
from pydantic import ValidationError

from txgemma.config import (
    ChatConfig,
    Config,
    PredictConfig,
    PromptsConfig,
    ToolsConfig,
    get_config,
    load_config,
    reset_config,
)

pytestmark = [pytest.mark.unit]

# =============================================================================
# TEST FIXTURES - Isolated Test Data
# =============================================================================


@pytest.fixture(autouse=True)
def reset_config_singleton():
    """
    Reset config singleton before and after each test.

    Ensures test isolation - each test starts with a clean slate.
    """
    reset_config()
    yield
    reset_config()


@pytest.fixture
def minimal_yaml():
    """Minimal valid YAML configuration."""
    return """
predict:
  model: "google/txgemma-2b-predict"
"""


@pytest.fixture
def full_yaml():
    """Complete YAML configuration with all options."""
    return """
predict:
  model: "google/txgemma-9b-predict"
  max_new_tokens: 64

chat:
  model: "google/txgemma-9b-chat"
  max_new_tokens: 200

tools:
  prompts:
    filename: "tdc_prompts.json"
    local_override: null
  filter_placeholder: "Drug SMILES"
  max_placeholders: null
  exclude_complex: false
  enable_chat: true
"""


@pytest.fixture
def production_yaml():
    """Production configuration preset."""
    return """
predict:
  model: "google/txgemma-9b-predict"

chat:
  model: "google/txgemma-9b-chat"
  max_new_tokens: 200

tools:
  filter_placeholder: "Drug SMILES"
  enable_chat: true
"""


@pytest.fixture
def test_yaml():
    """Test/development configuration preset."""
    return """
predict:
  model: "google/txgemma-2b-predict"

chat:
  model: "google/txgemma-9b-chat"

tools:
  max_placeholders: 1
  enable_chat: false
"""


# =============================================================================
# CONFIG MODEL TESTS
# =============================================================================


class TestConfigModels:
    """Test Pydantic config models."""

    def test_predict_config_defaults(self):
        """Test PredictConfig default values."""
        config = PredictConfig()

        assert config.model == "google/txgemma-2b-predict"
        assert config.max_new_tokens == 64

    def test_chat_config_defaults(self):
        """Test ChatConfig default values."""
        config = ChatConfig()

        assert config.model == "google/txgemma-9b-chat"
        assert config.max_new_tokens == 100

    def test_prompts_config_defaults(self):
        """Test PromptsConfig default values."""
        config = PromptsConfig()

        assert config.filename == "tdc_prompts.json"
        assert config.local_override is None

    def test_tools_config_defaults(self):
        """Test ToolsConfig default values."""
        config = ToolsConfig()

        assert config.filter_placeholder is None  # Default is None
        assert config.max_placeholders is None
        assert config.enable_chat is True
        assert isinstance(config.prompts, PromptsConfig)

    def test_main_config_defaults(self):
        """Test Config default values."""
        config = Config()

        assert isinstance(config.predict, PredictConfig)
        assert isinstance(config.chat, ChatConfig)
        assert isinstance(config.tools, ToolsConfig)

    def test_predict_config_custom_values(self):
        """Test PredictConfig with custom values."""
        config = PredictConfig(model="google/txgemma-9b-predict", max_new_tokens=128)

        assert config.model == "google/txgemma-9b-predict"
        assert config.max_new_tokens == 128

    def test_chat_config_custom_values(self):
        """Test ChatConfig with custom values."""
        config = ChatConfig(model="google/txgemma-27b-chat", max_new_tokens=500)

        assert config.model == "google/txgemma-27b-chat"
        assert config.max_new_tokens == 500


# =============================================================================
# FILE LOADING TESTS
# =============================================================================


class TestLoadConfigFromFile:
    """Test loading configuration from YAML file."""

    def test_load_config_file_not_exists(self):
        """Test loading config when file doesn't exist uses defaults."""
        config = load_config(Path("nonexistent.yaml"))

        # Should use defaults
        assert config.predict.model == "google/txgemma-2b-predict"
        assert config.chat.model == "google/txgemma-9b-chat"
        assert config.tools.filter_placeholder is None  # Default

    def test_load_minimal_yaml(self, minimal_yaml):
        """Test loading minimal valid YAML."""
        with (
            patch("builtins.open", mock_open(read_data=minimal_yaml)),
            patch("pathlib.Path.exists", return_value=True),
        ):
            config = load_config(Path("test_config.yaml"))

        assert config.predict.model == "google/txgemma-2b-predict"
        # Other fields should use defaults
        assert config.chat.model == "google/txgemma-9b-chat"

    def test_load_full_yaml(self, full_yaml):
        """Test loading complete YAML configuration."""
        with (
            patch("builtins.open", mock_open(read_data=full_yaml)),
            patch("pathlib.Path.exists", return_value=True),
        ):
            config = load_config(Path("test_config.yaml"))

        assert config.predict.model == "google/txgemma-9b-predict"
        assert config.predict.max_new_tokens == 64
        assert config.chat.model == "google/txgemma-9b-chat"
        assert config.chat.max_new_tokens == 200
        assert config.tools.filter_placeholder == "Drug SMILES"

    def test_load_yaml_with_tmp_path(self, tmp_path):
        """Test loading from real file using tmp_path."""
        # Create actual test file
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("""
predict:
  model: "google/txgemma-2b-predict"
  max_new_tokens: 128
""")

        config = load_config(config_file)

        assert config.predict.model == "google/txgemma-2b-predict"
        assert config.predict.max_new_tokens == 128

    def test_invalid_yaml(self):
        """Test handling of invalid YAML."""
        invalid_yaml = "predict:\n  model: [invalid: yaml"

        with (
            patch("builtins.open", mock_open(read_data=invalid_yaml)),
            patch("pathlib.Path.exists", return_value=True),
            pytest.raises((ValueError, yaml.YAMLError)),
        ):
            load_config(Path("test_config.yaml"))

    def test_empty_yaml(self):
        """Test handling of empty YAML file."""
        with (
            patch("builtins.open", mock_open(read_data="")),
            patch("pathlib.Path.exists", return_value=True),
        ):
            config = load_config(Path("test_config.yaml"))

        # Should use all defaults
        assert config.predict.model == "google/txgemma-2b-predict"
        assert config.chat.model == "google/txgemma-9b-chat"


# =============================================================================
# ENVIRONMENT VARIABLE TESTS
# =============================================================================


class TestEnvironmentVariableOverrides:
    """Test environment variable overrides."""

    def test_env_predict_model_override(self):
        """Test TXGEMMA_PREDICT_MODEL override."""
        with patch.dict(os.environ, {"TXGEMMA_PREDICT_MODEL": "google/custom-model"}):
            config = load_config(Path("nonexistent.yaml"))

        assert config.predict.model == "google/custom-model"

    def test_env_chat_model_override(self):
        """Test TXGEMMA_CHAT_MODEL override."""
        with patch.dict(os.environ, {"TXGEMMA_CHAT_MODEL": "google/custom-chat"}):
            config = load_config(Path("nonexistent.yaml"))

        assert config.chat.model == "google/custom-chat"

    def test_env_max_tokens_override(self):
        """Test TXGEMMA_CHAT_MAX_TOKENS override."""
        with patch.dict(os.environ, {"TXGEMMA_CHAT_MAX_TOKENS": "500"}):
            config = load_config(Path("nonexistent.yaml"))

        assert config.chat.max_new_tokens == 500

    def test_env_filter_placeholder_override(self):
        """Test TXGEMMA_FILTER_PLACEHOLDER override."""
        with patch.dict(os.environ, {"TXGEMMA_FILTER_PLACEHOLDER": "Target sequence"}):
            config = load_config(Path("nonexistent.yaml"))

        assert config.tools.filter_placeholder == "Target sequence"

    def test_env_boolean_overrides(self):
        """Test boolean environment variable overrides."""
        env_vars = {
            "TXGEMMA_EXACT_MATCH": "0",
            "TXGEMMA_MATCH_ALL": "0",
            "TXGEMMA_EXCLUDE_COMPLEX": "1",
            "TXGEMMA_ENABLE_CHAT": "0",
        }

        with patch.dict(os.environ, env_vars):
            config = load_config(Path("nonexistent.yaml"))

        assert config.tools.exact_match is False
        assert config.tools.match_all is False
        assert config.tools.exclude_complex is True
        assert config.tools.enable_chat is False

    def test_env_overrides_yaml(self, production_yaml):
        """Test that environment variables override YAML values."""
        env_vars = {
            "TXGEMMA_PREDICT_MODEL": "google/env-model",
            "TXGEMMA_FILTER_PLACEHOLDER": "Protein sequence",
        }

        with (
            patch.dict(os.environ, env_vars),
            patch("builtins.open", mock_open(read_data=production_yaml)),
            patch("pathlib.Path.exists", return_value=True),
        ):
            config = load_config(Path("test_config.yaml"))

        # Environment should win over YAML
        assert config.predict.model == "google/env-model"
        assert config.tools.filter_placeholder == "Protein sequence"
        # YAML values without env override should be preserved
        assert config.chat.max_new_tokens == 200

    def test_env_priority_order(self, production_yaml):
        """Test priority: env > yaml > defaults."""
        env_vars = {"TXGEMMA_PREDICT_MODEL": "google/env-model"}

        with (
            patch.dict(os.environ, env_vars),
            patch("builtins.open", mock_open(read_data=production_yaml)),
            patch("pathlib.Path.exists", return_value=True),
        ):
            config = load_config(Path("test_config.yaml"))

        # Env (highest priority)
        assert config.predict.model == "google/env-model"

        # YAML (middle priority)
        assert config.chat.max_new_tokens == 200

        # Default (lowest priority)
        assert config.tools.max_placeholders is None


# =============================================================================
# SINGLETON TESTS
# =============================================================================


class TestGetConfigSingleton:
    """Test get_config singleton behavior."""

    def test_get_config_returns_instance(self):
        """Test that get_config returns a Config instance."""
        config = get_config()

        assert isinstance(config, Config)

    def test_get_config_singleton(self):
        """Test that get_config returns the same instance."""
        config1 = get_config()
        config2 = get_config()

        assert config1 is config2

    def test_reset_config_clears_singleton(self):
        """Test that reset_config clears the singleton."""
        config1 = get_config()

        reset_config()

        config2 = get_config()

        # Should be different instances
        assert config1 is not config2

    def test_get_config_reloads_after_reset(self):
        """Test that config reloads after reset."""
        config1 = get_config()
        _ = config1.predict.model

        reset_config()

        # Should create new instance
        config2 = get_config()
        assert isinstance(config2, Config)


# =============================================================================
# USE CASE TESTS
# =============================================================================


class TestConfigUseCases:
    """Test realistic configuration use cases."""

    def test_development_config(self):
        """Test typical development configuration."""
        dev_yaml = """
predict:
  model: "google/txgemma-2b-predict"

chat:
  model: "google/txgemma-9b-chat"

tools:
  max_placeholders: 1
  enable_chat: true
"""

        with (
            patch("builtins.open", mock_open(read_data=dev_yaml)),
            patch("pathlib.Path.exists", return_value=True),
        ):
            config = load_config(Path("test_config.yaml"))

        assert config.predict.model == "google/txgemma-2b-predict"
        assert config.tools.max_placeholders == 1
        assert config.tools.enable_chat is True

    def test_production_config(self, production_yaml):
        """Test production configuration preset."""
        with (
            patch("builtins.open", mock_open(read_data=production_yaml)),
            patch("pathlib.Path.exists", return_value=True),
        ):
            config = load_config(Path("test_config.yaml"))

        assert config.predict.model == "google/txgemma-9b-predict"
        assert config.chat.model == "google/txgemma-9b-chat"
        assert config.chat.max_new_tokens == 200
        assert config.tools.filter_placeholder == "Drug SMILES"

    def test_testing_config(self, test_yaml):
        """Test configuration for testing/CI."""
        with (
            patch("builtins.open", mock_open(read_data=test_yaml)),
            patch("pathlib.Path.exists", return_value=True),
        ):
            config = load_config(Path("test_config.yaml"))

        assert config.predict.model == "google/txgemma-2b-predict"
        assert config.tools.max_placeholders == 1
        assert config.tools.enable_chat is False

    def test_custom_prompts_config(self):
        """Test configuration with custom local prompts."""
        custom_yaml = """
tools:
  prompts:
    filename: "tdc_prompts.json"
    local_override: "/app/custom_prompts.json"
  filter_placeholder: "Drug SMILES"
  max_placeholders: 2
  enable_chat: false
"""

        with (
            patch("builtins.open", mock_open(read_data=custom_yaml)),
            patch("pathlib.Path.exists", return_value=True),
        ):
            config = load_config(Path("test_config.yaml"))

        assert config.tools.prompts.filename == "tdc_prompts.json"
        assert config.tools.prompts.local_override == "/app/custom_prompts.json"
        assert config.tools.filter_placeholder == "Drug SMILES"
        assert config.tools.max_placeholders == 2
        assert config.tools.enable_chat is False


# =============================================================================
# VALIDATION TESTS
# =============================================================================


class TestConfigValidation:
    """Test Pydantic validation."""

    def test_invalid_max_tokens_type(self):
        """Test that invalid max_tokens type raises error."""
        with pytest.raises(ValidationError):
            PredictConfig(max_new_tokens="invalid")

    def test_negative_max_tokens(self):
        """Test that negative max_tokens is accepted (Pydantic doesn't enforce)."""
        # Note: Pydantic allows negative values unless we add validators
        config = PredictConfig(max_new_tokens=-1)
        assert config.max_new_tokens == -1
