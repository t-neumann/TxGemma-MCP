"""
Configuration management for TxGemma MCP server.

Loads settings from config.yaml with environment variable overrides.
"""

import logging
import os
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class PredictConfig(BaseModel):
    """Prediction model configuration."""

    model: str = Field(default="google/txgemma-2b-predict")
    max_new_tokens: int = Field(default=64)


class ChatConfig(BaseModel):
    """Chat model configuration."""

    model: str = Field(default="google/txgemma-9b-chat")
    max_new_tokens: int = Field(default=100)


class PromptsConfig(BaseModel):
    """Prompts source configuration."""

    filename: str = Field(default="tdc_prompts.json")
    local_override: str | None = Field(default=None)


class ToolsConfig(BaseModel):
    """Tool loading configuration."""

    prompts: PromptsConfig = Field(default_factory=PromptsConfig)
    filter_placeholder: str | None = Field(default="Drug SMILES")
    max_placeholders: int | None = Field(default=None)
    enable_chat: bool = Field(default=True)


class Config(BaseModel):
    """Main configuration."""

    predict: PredictConfig = Field(default_factory=PredictConfig)
    chat: ChatConfig = Field(default_factory=ChatConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)


def load_config(config_path: Path | None = None) -> Config:
    """
    Load configuration from YAML file with environment variable overrides.

    Priority (highest to lowest):
    1. Environment variables (TXGEMMA_*)
    2. Config file
    3. Defaults

    Args:
        config_path: Path to config.yaml (default: ./config.yaml)

    Returns:
        Loaded and validated configuration

    Environment variable overrides:
        TXGEMMA_PREDICT_MODEL: Override predict.model
        TXGEMMA_CHAT_MODEL: Override chat.model
        TXGEMMA_CHAT_MAX_TOKENS: Override chat.max_new_tokens
        TXGEMMA_FILTER_PLACEHOLDER: Override tools.filter_placeholder
    """
    # Default config path
    if config_path is None:
        config_path = Path("config.yaml")

    # Load from file if exists
    config_dict = {}
    if config_path.exists():
        logger.info(f"Loading configuration from {config_path}")
        with open(config_path) as f:
            config_dict = yaml.safe_load(f) or {}
    else:
        logger.info(f"Config file {config_path} not found, using defaults")

    # Apply environment variable overrides
    env_overrides = {
        "TXGEMMA_PREDICT_MODEL": ("predict", "model"),
        "TXGEMMA_CHAT_MODEL": ("chat", "model"),
        "TXGEMMA_CHAT_MAX_TOKENS": ("chat", "max_new_tokens"),
        "TXGEMMA_FILTER_PLACEHOLDER": ("tools", "filter_placeholder"),
    }

    for env_var, (section, key) in env_overrides.items():
        if env_var in os.environ:
            value = os.environ[env_var]

            # Convert to int if needed
            if key in ["max_new_tokens", "max_placeholders"]:
                value = int(value)

            # Handle null/none for filter_placeholder
            if key == "filter_placeholder" and value.lower() in ["null", "none", ""]:
                value = None

            # Ensure section exists
            if section not in config_dict:
                config_dict[section] = {}

            config_dict[section][key] = value
            logger.info(f"Override from {env_var}: {section}.{key} = {value}")

    # Create and validate config
    try:
        config = Config(**config_dict)
        logger.info("Configuration loaded successfully")
        logger.info(f"  Predict model: {config.predict.model}")
        logger.info(f"  Chat model: {config.chat.model}")
        logger.info(f"  Chat max tokens: {config.chat.max_new_tokens}")
        logger.info(f"  Tool filter: {config.tools.filter_placeholder or 'None (all tools)'}")
        logger.info(f"  Chat enabled: {config.tools.enable_chat}")
        return config
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        raise


def get_config() -> Config:
    """
    Get configuration singleton.

    Loads on first call, returns cached instance thereafter.
    """
    if not hasattr(get_config, "_config"):
        get_config._config = load_config()
    return get_config._config
