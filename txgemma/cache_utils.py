"""
Global state management improvements for tool_factory.py

This module adds helper functions to manage global caches in a more testable way.
"""

import logging

logger = logging.getLogger(__name__)

# Global parameter mapping cache
_PARAMETER_MAPPING: dict[str, str] | None = None


def get_cached_parameter_mapping() -> dict[str, str]:
    """
    Get cached parameter mapping (normalized → original names).

    This is cached globally for performance, but can be reset for testing.

    Returns:
        Dictionary mapping normalized names to original placeholder names

    Example:
        {
            "drug_smiles": "Drug SMILES",
            "target_sequence": "Target sequence"
        }
    """
    global _PARAMETER_MAPPING

    if _PARAMETER_MAPPING is None:
        from txgemma.tool_factory import get_parameter_mapping

        _PARAMETER_MAPPING = get_parameter_mapping()
        logger.debug(f"Cached parameter mapping with {len(_PARAMETER_MAPPING)} entries")

    return _PARAMETER_MAPPING


def reset_parameter_mapping() -> None:
    """
    Reset the cached parameter mapping.

    This is primarily useful for testing where you want to reload
    the mapping with different templates or configurations.

    Example:
        >>> reset_parameter_mapping()
        >>> # Mapping will be reloaded on next call to get_cached_parameter_mapping()
    """
    global _PARAMETER_MAPPING
    _PARAMETER_MAPPING = None
    logger.debug("Reset parameter mapping cache")


def reset_all_caches() -> None:
    """
    Reset all module-level caches.

    This is a convenience function for testing that resets:
    - Parameter mapping cache (tool_factory)
    - Configuration singleton (config)

    Example:
        >>> from txgemma.cache_utils import reset_all_caches
        >>> reset_all_caches()
        >>> # All caches cleared, will reload from source on next access
    """
    reset_parameter_mapping()

    from txgemma.config import reset_config

    reset_config()

    logger.info("Reset all module-level caches")


class ParameterMappingOverride:
    """
    Context manager to temporarily override parameter mapping.

    This is useful for testing where you want to inject a specific
    mapping without affecting global state.

    Example:
        >>> test_mapping = {"test_param": "Test Param"}
        >>> with ParameterMappingOverride(test_mapping):
        ...     # Code here sees the test mapping
        ...     mapping = get_cached_parameter_mapping()
        ...     assert mapping == test_mapping
        >>> # Original mapping restored
    """

    def __init__(self, override_mapping: dict[str, str]):
        """
        Initialize with override mapping.

        Args:
            override_mapping: Dictionary to use instead of cached mapping
        """
        self.override_mapping = override_mapping
        self.original_mapping = None

    def __enter__(self):
        """Save current mapping and install override."""
        global _PARAMETER_MAPPING
        self.original_mapping = _PARAMETER_MAPPING
        _PARAMETER_MAPPING = self.override_mapping
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore original mapping."""
        global _PARAMETER_MAPPING
        _PARAMETER_MAPPING = self.original_mapping
        return False


__all__ = [
    "ParameterMappingOverride",
    "get_cached_parameter_mapping",
    "reset_all_caches",
    "reset_parameter_mapping",
]
