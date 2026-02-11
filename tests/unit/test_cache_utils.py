"""
Tests for txgemma.cache_utils module.

Tests global state management, caching behavior, and context managers.
Focus on ensuring caches work correctly and can be reset for testing.

"""

from unittest.mock import Mock, patch

import pytest

from txgemma.cache_utils import (
    get_cached_parameter_mapping,
    reset_parameter_mapping,
    reset_all_caches,
    ParameterMappingOverride,
)

pytestmark = [pytest.mark.unit]

# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture(autouse=True)
def reset_cache_before_each_test():
    """
    Reset cache before each test to ensure clean state.
    
    This is autouse=True so every test starts with a clean cache.
    """
    reset_parameter_mapping()
    yield
    # Clean up after test
    reset_parameter_mapping()


@pytest.fixture
def mock_mapping():
    """Sample parameter mapping for testing."""
    return {
        "drug_smiles": "Drug SMILES",
        "target_sequence": "Target sequence",
        "trial_phase": "Trial phase",
    }


# =============================================================================
# CACHE LOADING TESTS
# =============================================================================

class TestGetCachedParameterMapping:
    """Test cached parameter mapping retrieval."""
    
    @patch("txgemma.tool_factory.get_parameter_mapping")
    def test_loads_mapping_on_first_access(self, mock_get_mapping, mock_mapping):
        """Test that mapping is loaded on first access."""
        mock_get_mapping.return_value = mock_mapping
        
        result = get_cached_parameter_mapping()
        
        # Should call get_parameter_mapping
        mock_get_mapping.assert_called_once()
        
        # Should return the mapping
        assert result == mock_mapping
    
    @patch("txgemma.tool_factory.get_parameter_mapping")
    def test_caches_mapping_for_subsequent_access(self, mock_get_mapping, mock_mapping):
        """Test that mapping is cached and not reloaded."""
        mock_get_mapping.return_value = mock_mapping
        
        # First access - should load
        result1 = get_cached_parameter_mapping()
        assert mock_get_mapping.call_count == 1
        
        # Second access - should use cache
        result2 = get_cached_parameter_mapping()
        assert mock_get_mapping.call_count == 1  # Still 1, not called again
        
        # Should return same mapping
        assert result1 == result2
        assert result2 == mock_mapping
    
    @patch("txgemma.tool_factory.get_parameter_mapping")
    def test_returns_same_dict_instance(self, mock_get_mapping, mock_mapping):
        """Test that cached mapping returns the same dict instance."""
        mock_get_mapping.return_value = mock_mapping
        
        result1 = get_cached_parameter_mapping()
        result2 = get_cached_parameter_mapping()
        
        # Should be the exact same object (not a copy)
        assert result1 is result2
    
    @patch("txgemma.tool_factory.get_parameter_mapping")
    def test_handles_empty_mapping(self, mock_get_mapping):
        """Test with empty parameter mapping."""
        mock_get_mapping.return_value = {}
        
        result = get_cached_parameter_mapping()
        
        assert result == {}
        assert isinstance(result, dict)
    
    @patch("txgemma.tool_factory.get_parameter_mapping")
    def test_loads_large_mapping(self, mock_get_mapping):
        """Test with large parameter mapping."""
        # Create a large mapping
        large_mapping = {f"param_{i}": f"Param {i}" for i in range(100)}
        mock_get_mapping.return_value = large_mapping
        
        result = get_cached_parameter_mapping()
        
        assert len(result) == 100
        assert result["param_0"] == "Param 0"
        assert result["param_99"] == "Param 99"


# =============================================================================
# RESET TESTS
# =============================================================================

class TestResetParameterMapping:
    """Test cache reset functionality."""
    
    @patch("txgemma.tool_factory.get_parameter_mapping")
    def test_clears_cache(self, mock_get_mapping, mock_mapping):
        """Test that reset clears the cache."""
        mock_get_mapping.return_value = mock_mapping
        
        # Load cache
        get_cached_parameter_mapping()
        assert mock_get_mapping.call_count == 1
        
        # Reset cache
        reset_parameter_mapping()
        
        # Next access should reload
        get_cached_parameter_mapping()
        assert mock_get_mapping.call_count == 2  # Called again
    
    @patch("txgemma.tool_factory.get_parameter_mapping")
    def test_allows_new_mapping_after_reset(self, mock_get_mapping):
        """Test that reset allows loading a different mapping."""
        # First mapping
        mock_get_mapping.return_value = {"param1": "Param 1"}
        result1 = get_cached_parameter_mapping()
        assert result1 == {"param1": "Param 1"}
        
        # Reset and change mapping
        reset_parameter_mapping()
        mock_get_mapping.return_value = {"param2": "Param 2"}
        result2 = get_cached_parameter_mapping()
        
        # Should have new mapping
        assert result2 == {"param2": "Param 2"}
        assert result2 != result1
    
    def test_reset_is_idempotent(self):
        """Test that multiple resets are safe."""
        # Multiple resets should not cause errors
        reset_parameter_mapping()
        reset_parameter_mapping()
        reset_parameter_mapping()
        
        # Should still work after multiple resets
        with patch("txgemma.tool_factory.get_parameter_mapping") as mock:
            mock.return_value = {}
            result = get_cached_parameter_mapping()
            assert result == {}
    
    def test_reset_before_first_access(self):
        """Test that reset works even if cache was never accessed."""
        # Reset before any access
        reset_parameter_mapping()
        
        # Should still work normally
        with patch("txgemma.tool_factory.get_parameter_mapping") as mock:
            mock.return_value = {"test": "Test"}
            result = get_cached_parameter_mapping()
            assert result == {"test": "Test"}


# =============================================================================
# RESET ALL CACHES TESTS
# =============================================================================

class TestResetAllCaches:
    """Test resetting all module caches."""
    
    @patch("txgemma.config.reset_config")
    @patch("txgemma.tool_factory.get_parameter_mapping")
    def test_resets_parameter_mapping(self, mock_get_mapping, mock_reset_config, mock_mapping):
        """Test that reset_all_caches resets parameter mapping."""
        mock_get_mapping.return_value = mock_mapping
        
        # Load cache
        get_cached_parameter_mapping()
        assert mock_get_mapping.call_count == 1
        
        # Reset all
        reset_all_caches()
        
        # Should reload parameter mapping
        get_cached_parameter_mapping()
        assert mock_get_mapping.call_count == 2
    
    @patch("txgemma.config.reset_config")
    @patch("txgemma.tool_factory.get_parameter_mapping")
    def test_resets_config(self, mock_get_mapping, mock_reset_config):
        """Test that reset_all_caches resets config."""
        mock_get_mapping.return_value = {}
        
        reset_all_caches()
        
        # Should call reset_config
        mock_reset_config.assert_called_once()
    
    @patch("txgemma.config.reset_config")
    @patch("txgemma.tool_factory.get_parameter_mapping")
    def test_is_idempotent(self, mock_get_mapping, mock_reset_config):
        """Test that multiple resets are safe."""
        mock_get_mapping.return_value = {}
        
        # Multiple resets should not cause errors
        reset_all_caches()
        reset_all_caches()
        reset_all_caches()
        
        # Should work normally after
        result = get_cached_parameter_mapping()
        assert isinstance(result, dict)


# =============================================================================
# CONTEXT MANAGER TESTS
# =============================================================================

class TestParameterMappingOverride:
    """Test ParameterMappingOverride context manager."""
    
    @patch("txgemma.tool_factory.get_parameter_mapping")
    def test_overrides_mapping_in_context(self, mock_get_mapping):
        """Test that mapping is overridden within context."""
        mock_get_mapping.return_value = {"original": "Original"}
        override_mapping = {"override": "Override"}
        
        # Load original
        original = get_cached_parameter_mapping()
        assert original == {"original": "Original"}
        
        # Override in context
        with ParameterMappingOverride(override_mapping):
            result = get_cached_parameter_mapping()
            assert result == {"override": "Override"}
        
        # Should be restored after context
        after = get_cached_parameter_mapping()
        assert after == {"original": "Original"}
    
    @patch("txgemma.tool_factory.get_parameter_mapping")
    def test_restores_original_after_context(self, mock_get_mapping):
        """Test that original mapping is restored."""
        mock_get_mapping.return_value = {"original": "Original"}
        
        # Load original
        get_cached_parameter_mapping()
        
        # Override and exit
        with ParameterMappingOverride({"temp": "Temp"}):
            pass
        
        # Original should be restored
        result = get_cached_parameter_mapping()
        assert result == {"original": "Original"}
        # Should not reload (still cached)
        assert mock_get_mapping.call_count == 1
    
    def test_restores_on_exception(self):
        """Test that original is restored even on exception."""
        with patch("txgemma.tool_factory.get_parameter_mapping") as mock:
            mock.return_value = {"original": "Original"}
            get_cached_parameter_mapping()
            
            # Override with exception
            try:
                with ParameterMappingOverride({"temp": "Temp"}):
                    raise ValueError("Test error")
            except ValueError:
                pass
            
            # Original should be restored
            result = get_cached_parameter_mapping()
            assert result == {"original": "Original"}
    
    def test_nested_overrides(self):
        """Test nested context managers."""
        with patch("txgemma.tool_factory.get_parameter_mapping") as mock:
            mock.return_value = {"original": "Original"}
            get_cached_parameter_mapping()
            
            with ParameterMappingOverride({"level1": "Level 1"}):
                assert get_cached_parameter_mapping() == {"level1": "Level 1"}
                
                with ParameterMappingOverride({"level2": "Level 2"}):
                    assert get_cached_parameter_mapping() == {"level2": "Level 2"}
                
                # Back to level 1
                assert get_cached_parameter_mapping() == {"level1": "Level 1"}
            
            # Back to original
            assert get_cached_parameter_mapping() == {"original": "Original"}
    
    def test_override_with_empty_mapping(self):
        """Test override with empty dictionary."""
        with patch("txgemma.tool_factory.get_parameter_mapping") as mock:
            mock.return_value = {"original": "Original"}
            get_cached_parameter_mapping()
            
            with ParameterMappingOverride({}):
                result = get_cached_parameter_mapping()
                assert result == {}
            
            # Original restored
            result = get_cached_parameter_mapping()
            assert result == {"original": "Original"}
    
    def test_override_before_first_access(self):
        """Test override before cache is initialized."""
        override_mapping = {"override": "Override"}
        
        # Use override before any cache access
        with ParameterMappingOverride(override_mapping):
            result = get_cached_parameter_mapping()
            assert result == {"override": "Override"}
        
        # After context, cache should be None (not loaded)
        # Next access will load normally
        with patch("txgemma.tool_factory.get_parameter_mapping") as mock:
            mock.return_value = {"normal": "Normal"}
            result = get_cached_parameter_mapping()
            assert result == {"normal": "Normal"}


# =============================================================================
# CACHE PERFORMANCE TESTS
# =============================================================================

class TestCachePerformance:
    """Test that caching actually improves performance."""
    
    @patch("txgemma.tool_factory.get_parameter_mapping")
    def test_only_calls_get_parameter_mapping_once(self, mock_get_mapping, mock_mapping):
        """Test that get_parameter_mapping is only called once."""
        mock_get_mapping.return_value = mock_mapping
        
        # Access cache multiple times
        for _ in range(10):
            get_cached_parameter_mapping()
        
        # Should only call once
        assert mock_get_mapping.call_count == 1
    
    @patch("txgemma.tool_factory.get_parameter_mapping")
    def test_cache_survives_multiple_accesses(self, mock_get_mapping, mock_mapping):
        """Test that cache persists across many accesses."""
        mock_get_mapping.return_value = mock_mapping
        
        # Access 100 times
        results = [get_cached_parameter_mapping() for _ in range(100)]
        
        # All should be the same instance
        assert all(r is results[0] for r in results)
        
        # Only loaded once
        assert mock_get_mapping.call_count == 1


# =============================================================================
# INTEGRATION-LIKE TESTS
# =============================================================================

class TestCacheUtilsIntegration:
    """Test realistic usage patterns."""
    
    @patch("txgemma.tool_factory.get_parameter_mapping")
    def test_typical_usage_pattern(self, mock_get_mapping):
        """Test typical usage: load, use, reset, reload."""
        # Initial load
        mock_get_mapping.return_value = {"v1": "Version 1"}
        result1 = get_cached_parameter_mapping()
        assert result1 == {"v1": "Version 1"}
        
        # Use cache multiple times
        for _ in range(5):
            result = get_cached_parameter_mapping()
            assert result == {"v1": "Version 1"}
        
        # Reset for new configuration
        reset_parameter_mapping()
        mock_get_mapping.return_value = {"v2": "Version 2"}
        
        # Reload with new version
        result2 = get_cached_parameter_mapping()
        assert result2 == {"v2": "Version 2"}
    
    @patch("txgemma.config.reset_config")
    @patch("txgemma.tool_factory.get_parameter_mapping")
    def test_test_isolation_pattern(self, mock_get_mapping, mock_reset_config):
        """Test pattern for test isolation."""
        mock_get_mapping.return_value = {"test1": "Test 1"}
        
        # Test 1
        result1 = get_cached_parameter_mapping()
        assert result1 == {"test1": "Test 1"}
        
        # Reset between tests
        reset_all_caches()
        
        # Test 2 with different data
        mock_get_mapping.return_value = {"test2": "Test 2"}
        result2 = get_cached_parameter_mapping()
        assert result2 == {"test2": "Test 2"}
        assert result2 != result1
    
    @patch("txgemma.tool_factory.get_parameter_mapping")
    def test_temporary_override_pattern(self, mock_get_mapping):
        """Test pattern for temporary testing overrides."""
        mock_get_mapping.return_value = {"prod": "Production"}
        
        # Normal usage
        prod_mapping = get_cached_parameter_mapping()
        assert prod_mapping == {"prod": "Production"}
        
        # Temporary test override
        test_mapping = {"test": "Test"}
        with ParameterMappingOverride(test_mapping):
            # Test code here sees test mapping
            result = get_cached_parameter_mapping()
            assert result == {"test": "Test"}
        
        # Back to production mapping
        result = get_cached_parameter_mapping()
        assert result == {"prod": "Production"}


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    @patch("txgemma.tool_factory.get_parameter_mapping")
    def test_get_parameter_mapping_raises_exception(self, mock_get_mapping):
        """Test behavior when get_parameter_mapping fails."""
        mock_get_mapping.side_effect = RuntimeError("Failed to load")
        
        # Should propagate the exception
        with pytest.raises(RuntimeError, match="Failed to load"):
            get_cached_parameter_mapping()
    
    @patch("txgemma.tool_factory.get_parameter_mapping")
    def test_mapping_with_special_characters(self, mock_get_mapping):
        """Test mapping with special characters in keys/values."""
        special_mapping = {
            "param_with_ümlauts": "Param with Ümlauts",
            "param_with_spaces": "Param With Spaces",
            "param_with_emoji_🎉": "Param With Emoji 🎉",
        }
        mock_get_mapping.return_value = special_mapping
        
        result = get_cached_parameter_mapping()
        assert result == special_mapping
    
    def test_multiple_resets_and_accesses(self):
        """Test alternating resets and accesses."""
        with patch("txgemma.tool_factory.get_parameter_mapping") as mock:
            for i in range(5):
                # Set return value for this iteration
                expected = {f"test{i}": f"Test{i}"}
                mock.return_value = expected
                
                # Access should get current value
                result = get_cached_parameter_mapping()
                assert result == expected
                
                # Reset for next iteration
                reset_parameter_mapping()