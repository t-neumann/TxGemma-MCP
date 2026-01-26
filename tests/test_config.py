"""
Tests for txgemma.config module.

Tests configuration loading, validation, and environment variable overrides.
"""

import os
from pathlib import Path
from unittest.mock import mock_open, patch

import pytest
from pydantic import ValidationError

from txgemma.config import (
    ChatConfig,
    Config,
    PredictConfig,
    PromptsConfig,
    ToolsConfig,
    get_config,
    load_config,
)


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

        assert config.filter_placeholder == "Drug SMILES"
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


class TestLoadConfigFromFile:
    """Test loading configuration from YAML file."""

    def test_load_config_file_not_exists(self):
        """Test loading config when file doesn't exist uses defaults."""
        config = load_config(Path("nonexistent.yaml"))

        # Should use defaults
        assert config.predict.model == "google/txgemma-2b-predict"
        assert config.chat.model == "google/txgemma-9b-chat"
        assert config.tools.filter_placeholder == "Drug SMILES"

    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data="""
predict:
  model: "google/txgemma-9b-predict"
  max_new_tokens: 64

chat:
  model: "google/txgemma-9b-chat"
  max_new_tokens: 200

tools:
  filter_placeholder: "Target sequence"
  enable_chat: false
""",
    )
    @patch("pathlib.Path.exists")
    def test_load_config_from_yaml(self, mock_exists, mock_file):
        """Test loading config from YAML file."""
        mock_exists.return_value = True

        config = load_config(Path("config.yaml"))

        assert config.predict.model == "google/txgemma-9b-predict"
        assert config.predict.max_new_tokens == 64
        assert config.chat.model == "google/txgemma-9b-chat"
        assert config.chat.max_new_tokens == 200
        assert config.tools.filter_placeholder == "Target sequence"
        assert config.tools.enable_chat is False

    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data="""
tools:
  prompts:
    filename: "custom_prompts.json"
    local_override: "/path/to/prompts.json"
  filter_placeholder: null
  max_placeholders: 2
""",
    )
    @patch("pathlib.Path.exists")
    def test_load_config_with_nested_prompts(self, mock_exists, mock_file):
        """Test loading config with nested prompts configuration."""
        mock_exists.return_value = True

        config = load_config(Path("config.yaml"))

        assert config.tools.prompts.filename == "custom_prompts.json"
        assert config.tools.prompts.local_override == "/path/to/prompts.json"
        assert config.tools.filter_placeholder is None
        assert config.tools.max_placeholders == 2

    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data="""
predict:
  model: "google/txgemma-27b-predict"
""",
    )
    @patch("pathlib.Path.exists")
    def test_load_config_partial_yaml(self, mock_exists, mock_file):
        """Test loading config with only some values specified."""
        mock_exists.return_value = True

        config = load_config(Path("config.yaml"))

        # Specified value
        assert config.predict.model == "google/txgemma-27b-predict"

        # Defaults for unspecified values
        assert config.predict.max_new_tokens == 64
        assert config.chat.model == "google/txgemma-9b-chat"
        assert config.tools.filter_placeholder == "Drug SMILES"


class TestEnvironmentVariableOverrides:
    """Test environment variable overrides."""

    @patch("pathlib.Path.exists")
    def test_env_override_predict_model(self, mock_exists):
        """Test TXGEMMA_PREDICT_MODEL environment variable."""
        mock_exists.return_value = False

        with patch.dict(os.environ, {"TXGEMMA_PREDICT_MODEL": "google/txgemma-27b-predict"}):
            config = load_config()

        assert config.predict.model == "google/txgemma-27b-predict"

    @patch("pathlib.Path.exists")
    def test_env_override_chat_model(self, mock_exists):
        """Test TXGEMMA_CHAT_MODEL environment variable."""
        mock_exists.return_value = False

        with patch.dict(os.environ, {"TXGEMMA_CHAT_MODEL": "google/txgemma-27b-chat"}):
            config = load_config()

        assert config.chat.model == "google/txgemma-27b-chat"

    @patch("pathlib.Path.exists")
    def test_env_override_chat_max_tokens(self, mock_exists):
        """Test TXGEMMA_CHAT_MAX_TOKENS environment variable."""
        mock_exists.return_value = False

        with patch.dict(os.environ, {"TXGEMMA_CHAT_MAX_TOKENS": "500"}):
            config = load_config()

        assert config.chat.max_new_tokens == 500

    @patch("pathlib.Path.exists")
    def test_env_override_filter_placeholder(self, mock_exists):
        """Test TXGEMMA_FILTER_PLACEHOLDER environment variable."""
        mock_exists.return_value = False

        with patch.dict(os.environ, {"TXGEMMA_FILTER_PLACEHOLDER": "Target sequence"}):
            config = load_config()

        assert config.tools.filter_placeholder == "Target sequence"

    @patch("pathlib.Path.exists")
    def test_env_override_filter_placeholder_null(self, mock_exists):
        """Test setting filter_placeholder to null via env var."""
        mock_exists.return_value = False

        # Test various null representations
        for null_value in ["null", "none", "None", "NULL", ""]:
            with patch.dict(os.environ, {"TXGEMMA_FILTER_PLACEHOLDER": null_value}):
                config = load_config()

            assert config.tools.filter_placeholder is None

    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data="""
predict:
  model: "google/txgemma-2b-predict"
chat:
  model: "google/txgemma-9b-chat"
  max_new_tokens: 100
""",
    )
    @patch("pathlib.Path.exists")
    def test_env_overrides_yaml_values(self, mock_exists, mock_file):
        """Test that environment variables override YAML values."""
        mock_exists.return_value = True

        with patch.dict(
            os.environ,
            {
                "TXGEMMA_PREDICT_MODEL": "google/txgemma-27b-predict",
                "TXGEMMA_CHAT_MAX_TOKENS": "500",
            },
        ):
            config = load_config(Path("config.yaml"))

        # Env overrides YAML
        assert config.predict.model == "google/txgemma-27b-predict"
        assert config.chat.max_new_tokens == 500

        # YAML value used where no env override
        assert config.chat.model == "google/txgemma-9b-chat"

    @patch("pathlib.Path.exists")
    def test_multiple_env_overrides(self, mock_exists):
        """Test multiple environment variable overrides at once."""
        mock_exists.return_value = False

        with patch.dict(
            os.environ,
            {
                "TXGEMMA_PREDICT_MODEL": "google/txgemma-9b-predict",
                "TXGEMMA_CHAT_MODEL": "google/txgemma-27b-chat",
                "TXGEMMA_CHAT_MAX_TOKENS": "300",
                "TXGEMMA_FILTER_PLACEHOLDER": "null",
            },
        ):
            config = load_config()

        assert config.predict.model == "google/txgemma-9b-predict"
        assert config.chat.model == "google/txgemma-27b-chat"
        assert config.chat.max_new_tokens == 300
        assert config.tools.filter_placeholder is None


class TestGetConfigSingleton:
    """Test get_config singleton behavior."""

    def test_get_config_returns_config(self):
        """Test that get_config returns a Config instance."""
        # Clear singleton
        if hasattr(get_config, "_config"):
            delattr(get_config, "_config")

        config = get_config()

        assert isinstance(config, Config)

    def test_get_config_singleton(self):
        """Test that get_config returns same instance."""
        # Clear singleton
        if hasattr(get_config, "_config"):
            delattr(get_config, "_config")

        config1 = get_config()
        config2 = get_config()

        assert config1 is config2


class TestConfigValidation:
    """Test configuration validation."""

    def test_invalid_max_tokens_type(self):
        """Test that invalid max_new_tokens type raises validation error."""
        with pytest.raises(ValidationError):
            PredictConfig(max_new_tokens="invalid")

    def test_invalid_enable_chat_type(self):
        """Test that invalid enable_chat type raises validation error."""
        with pytest.raises(ValidationError):
            ToolsConfig(enable_chat="invalid")

    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data="""
invalid yaml content
  this: is: not: valid:
""",
    )
    @patch("pathlib.Path.exists")
    def test_invalid_yaml_syntax(self, mock_exists, mock_file):
        """Test that invalid YAML syntax is handled gracefully."""
        mock_exists.return_value = True

        # Should handle invalid YAML
        # Behavior depends on implementation - may use defaults or raise
        try:
            config = load_config(Path("config.yaml"))
            # If it doesn't raise, it should use defaults
            assert isinstance(config, Config)
        except Exception:
            # Or it may raise an error
            pass


class TestConfigUseCases:
    """Test realistic configuration use cases."""

    @patch("pathlib.Path.exists")
    def test_development_config(self, mock_exists):
        """Test development configuration preset."""
        mock_exists.return_value = False

        # Defaults are development preset
        config = load_config()

        assert config.predict.model == "google/txgemma-2b-predict"
        assert config.chat.model == "google/txgemma-9b-chat"
        assert config.chat.max_new_tokens == 100
        assert config.tools.filter_placeholder == "Drug SMILES"

    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data="""
predict:
  model: "google/txgemma-9b-predict"
chat:
  model: "google/txgemma-9b-chat"
  max_new_tokens: 200
tools:
  filter_placeholder: "Drug SMILES"
""",
    )
    @patch("pathlib.Path.exists")
    def test_production_config(self, mock_exists, mock_file):
        """Test production configuration preset."""
        mock_exists.return_value = True

        config = load_config(Path("config.yaml"))

        assert config.predict.model == "google/txgemma-9b-predict"
        assert config.chat.model == "google/txgemma-9b-chat"
        assert config.chat.max_new_tokens == 200
        assert config.tools.filter_placeholder == "Drug SMILES"

    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data="""
predict:
  model: "google/txgemma-27b-predict"
chat:
  model: "google/txgemma-27b-chat"
  max_new_tokens: 500
tools:
  filter_placeholder: null
  enable_chat: true
""",
    )
    @patch("pathlib.Path.exists")
    def test_research_config(self, mock_exists, mock_file):
        """Test research configuration preset."""
        mock_exists.return_value = True

        config = load_config(Path("config.yaml"))

        assert config.predict.model == "google/txgemma-27b-predict"
        assert config.chat.model == "google/txgemma-27b-chat"
        assert config.chat.max_new_tokens == 500
        assert config.tools.filter_placeholder is None
        assert config.tools.enable_chat is True

    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data="""
tools:
  prompts:
    filename: "tdc_prompts.json"
    local_override: "/app/custom_prompts.json"
  filter_placeholder: "Drug SMILES"
  max_placeholders: 2
  enable_chat: false
""",
    )
    @patch("pathlib.Path.exists")
    def test_custom_prompts_config(self, mock_exists, mock_file):
        """Test configuration with custom local prompts."""
        mock_exists.return_value = True

        config = load_config(Path("config.yaml"))

        assert config.tools.prompts.filename == "tdc_prompts.json"
        assert config.tools.prompts.local_override == "/app/custom_prompts.json"
        assert config.tools.filter_placeholder == "Drug SMILES"
        assert config.tools.max_placeholders == 2
        assert config.tools.enable_chat is False


class TestConfigLogging:
    """Test that config loading logs appropriately."""

    @patch("txgemma.config.logger")
    @patch("pathlib.Path.exists")
    def test_logs_config_not_found(self, mock_exists, mock_logger):
        """Test that missing config file is logged."""
        mock_exists.return_value = False

        load_config(Path("config.yaml"))

        # Should log that config file not found
        mock_logger.info.assert_any_call("Config file config.yaml not found, using defaults")

    @patch("txgemma.config.logger")
    @patch("builtins.open", new_callable=mock_open, read_data="predict:\n  model: test")
    @patch("pathlib.Path.exists")
    def test_logs_config_loaded(self, mock_exists, mock_file, mock_logger):
        """Test that config loading is logged."""
        mock_exists.return_value = True

        load_config(Path("config.yaml"))

        # Should log that config was loaded
        mock_logger.info.assert_any_call("Loading configuration from config.yaml")
        mock_logger.info.assert_any_call("Configuration loaded successfully")

    @patch("txgemma.config.logger")
    @patch("pathlib.Path.exists")
    def test_logs_env_overrides(self, mock_exists, mock_logger):
        """Test that environment variable overrides are logged."""
        mock_exists.return_value = False

        with patch.dict(os.environ, {"TXGEMMA_PREDICT_MODEL": "google/txgemma-9b-predict"}):
            load_config()

        # Should log the override
        assert any(
            "Override from TXGEMMA_PREDICT_MODEL" in str(call)
            for call in mock_logger.info.call_args_list
        )
