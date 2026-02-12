"""
Configuration management for TxGemma-MCP

Author: Tobias Neumann
License: MIT
Version: 0.1.1
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

    filter_placeholder: str | None = Field(
        default=None, description="Filter tools by a single placeholder (e.g., 'Drug SMILES')"
    )

    filter_placeholders: list[str] | None = Field(
        default=None, description="Filter tools by multiple placeholders"
    )

    match_all: bool = Field(
        default=True, description="If True, match ALL placeholders; if False, match ANY"
    )
    exact_match: bool = Field(
        default=True,
        description="If True, exact placeholder match; if False, fuzzy substring match",
    )

    exclude_complex: bool = Field(default=False, description="Exclude tools with >2 placeholders")
    max_placeholders: int | None = Field(
        default=None, description="Maximum number of placeholders per tool"
    )

    exclude_name_pattern: str | None = Field(
        default=None, description="Regex pattern to exclude tools by name (e.g., '^ToxCast')"
    )

    enable_chat: bool = Field(default=True, description="Enable the TxGemma chat tool")


class Config(BaseModel):
    """Main configuration."""

    predict: PredictConfig = Field(default_factory=PredictConfig)
    chat: ChatConfig = Field(default_factory=ChatConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)


def _parse_env_value(value: str, key: str) -> any:
    """
    Parse environment variable value based on expected type.

    Args:
        value: Raw string value from environment
        key: Configuration key to determine type

    Returns:
        Parsed value (int, bool, str, or None)
    """
    if value.lower() in ["null", "none", ""]:
        return None

    if key in ["max_new_tokens", "max_placeholders"]:
        try:
            return int(value)
        except ValueError:
            logger.warning(f"Invalid integer for {key}: {value}")
            return None

    if key in ["exact_match", "match_all", "exclude_complex", "enable_chat"]:
        return value.lower() in ["1", "true", "yes", "on"]

    return value


def _apply_env_overrides(config_dict: dict) -> None:
    """
    Apply environment variable overrides to config dictionary.

    Modifies config_dict in place with values from environment variables.

    Environment variables:
        TXGEMMA_PREDICT_MODEL: Override predict.model
        TXGEMMA_CHAT_MODEL: Override chat.model
        TXGEMMA_CHAT_MAX_TOKENS: Override chat.max_new_tokens
        TXGEMMA_FILTER_PLACEHOLDER: Override tools.filter_placeholder
        TXGEMMA_MAX_PLACEHOLDERS: Override tools.max_placeholders
        TXGEMMA_EXCLUDE_PATTERN: Override tools.exclude_name_pattern
        TXGEMMA_EXACT_MATCH: Override tools.exact_match (0 or 1)
        TXGEMMA_MATCH_ALL: Override tools.match_all (0 or 1)
        TXGEMMA_EXCLUDE_COMPLEX: Override tools.exclude_complex (0 or 1)
        TXGEMMA_ENABLE_CHAT: Override tools.enable_chat (0 or 1)
    """
    env_mappings = {
        "TXGEMMA_PREDICT_MODEL": ("predict", "model"),
        "TXGEMMA_CHAT_MODEL": ("chat", "model"),
        "TXGEMMA_CHAT_MAX_TOKENS": ("chat", "max_new_tokens"),
        "TXGEMMA_FILTER_PLACEHOLDER": ("tools", "filter_placeholder"),
        "TXGEMMA_MAX_PLACEHOLDERS": ("tools", "max_placeholders"),
        "TXGEMMA_EXCLUDE_PATTERN": ("tools", "exclude_name_pattern"),
        "TXGEMMA_EXACT_MATCH": ("tools", "exact_match"),
        "TXGEMMA_MATCH_ALL": ("tools", "match_all"),
        "TXGEMMA_EXCLUDE_COMPLEX": ("tools", "exclude_complex"),
        "TXGEMMA_ENABLE_CHAT": ("tools", "enable_chat"),
    }

    for env_var, (section, key) in env_mappings.items():
        if env_var in os.environ:
            raw_value = os.environ[env_var]
            parsed_value = _parse_env_value(raw_value, key)

            if section not in config_dict:
                config_dict[section] = {}

            config_dict[section][key] = parsed_value
            logger.info(f"Override from {env_var}: {section}.{key} = {parsed_value}")


def load_config(config_path: Path | None = None) -> Config:
    """
    Load configuration from YAML file with environment variable overrides.

    Priority (highest to lowest):
    1. Environment variables (TXGEMMA_*)
    2. Config file (config.yaml)
    3. Defaults

    Args:
        config_path: Path to config.yaml (default: ./config.yaml)

    Returns:
        Loaded and validated configuration

    Raises:
        Exception: If config file is invalid
    """
    if config_path is None:
        config_path = Path("config.yaml")

    config_dict = {}
    if config_path.exists():
        logger.info(f"Loading configuration from {config_path}")
        with open(config_path) as f:
            config_dict = yaml.safe_load(f) or {}
    else:
        logger.info(f"Config file {config_path} not found, using defaults")

    _apply_env_overrides(config_dict)

    try:
        config = Config(**config_dict)
        logger.info("Configuration loaded successfully")
        logger.info(f"  Predict model: {config.predict.model}")
        logger.info(f"  Chat model: {config.chat.model}")
        logger.info(f"  Chat max tokens: {config.chat.max_new_tokens}")
        logger.info(f"  Tool filter: {config.tools.filter_placeholder or 'None (all tools)'}")
        logger.info(f"  Max placeholders: {config.tools.max_placeholders or 'None'}")
        logger.info(f"  Exclude pattern: {config.tools.exclude_name_pattern or 'None'}")
        logger.info(f"  Chat enabled: {config.tools.enable_chat}")
        return config
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        raise


_config: Config | None = None


def get_config() -> Config:
    """
    Get configuration singleton.

    Loads on first call, returns cached instance thereafter.

    Returns:
        Loaded configuration instance
    """
    global _config
    if _config is None:
        _config = load_config()
    return _config


def reset_config() -> None:
    """
    Reset configuration singleton.

    Useful for testing where you want to reload configuration
    with different values or environment variables.
    """
    global _config
    _config = None
