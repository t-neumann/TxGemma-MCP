"""
Tests for txgemma.model module.

These tests require GPU and model download (~5GB).
Run with: pytest tests/test_model.py --run-gpu
"""

import pytest
import torch

from txgemma.model import TxGemmaModel, get_model


# Mark all tests in this file as requiring GPU
pytestmark = pytest.mark.gpu


class TestTxGemmaModelUnit:
    """Unit tests that don't require model loading."""
    
    def setup_method(self):
        """Reset singleton before each test."""
        # Reset the singleton instance for testing
        TxGemmaModel._instance = None
    
    def test_init_default(self):
        """Test model initialization with defaults."""
        model = TxGemmaModel()
        
        assert model.model_name == "google/txgemma-2b-predict"
        assert model.max_new_tokens == 64
        assert not model.is_loaded
        assert model.tokenizer is None
        assert model.model is None
    
    def test_init_custom(self):
        """Test model initialization with custom parameters."""
        model = TxGemmaModel(
            model_name="google/txgemma-9b-predict",
            max_new_tokens=128
        )
        
        assert model.model_name == "google/txgemma-9b-predict"
        assert model.max_new_tokens == 128
        assert not model.is_loaded
    
    def test_singleton_pattern(self):
        """Test that TxGemmaModel uses singleton pattern."""
        model1 = TxGemmaModel()
        model2 = TxGemmaModel()
        
        assert model1 is model2
    
    def test_get_model_singleton(self):
        """Test that get_model returns singleton."""
        model1 = get_model()
        model2 = get_model()
        
        assert model1 is model2
    
    def test_is_loaded_before_load(self):
        """Test is_loaded property before loading."""
        model = TxGemmaModel()
        assert not model.is_loaded


class TestTxGemmaModelIntegration:
    """Integration tests that require model download and GPU."""
    
    @pytest.fixture(scope="class")
    def loaded_model(self):
        """Fixture that loads model once for all tests in class."""
        model = TxGemmaModel()
        model.load()
        yield model
        # Cleanup after all tests
        model.unload()
    
    def test_load_model(self, loaded_model):
        """Test model loading."""
        assert loaded_model.is_loaded
        assert loaded_model.tokenizer is not None
        assert loaded_model.model is not None
    
    def test_model_device(self, loaded_model):
        """Test that model is on correct device."""
        # Should be on GPU if available
        if torch.cuda.is_available():
            assert "cuda" in str(loaded_model.model.device)
        elif torch.backends.mps.is_available():
            assert "mps" in str(loaded_model.model.device)
    
    def test_generate_simple(self, loaded_model):
        """Test simple text generation."""
        prompt = "Question: What is 2+2?\nAnswer:"
        result = loaded_model.generate(prompt, max_new_tokens=10)
        
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_generate_with_custom_tokens(self, loaded_model):
        """Test generation with custom max_new_tokens."""
        prompt = "Question: Explain photosynthesis.\nAnswer:"
        result = loaded_model.generate(prompt, max_new_tokens=50)
        
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_generate_deterministic(self, loaded_model):
        """Test that generation is deterministic (do_sample=False)."""
        prompt = "Question: What is the capital of France?\nAnswer:"
        
        result1 = loaded_model.generate(prompt, max_new_tokens=20)
        result2 = loaded_model.generate(prompt, max_new_tokens=20)
        
        # Should be identical with do_sample=False
        assert result1 == result2
    
    def test_generate_with_smiles(self, loaded_model):
        """Test generation with SMILES input (typical TDC use case)."""
        prompt = """Instruction: Predict the toxicity of the given drug molecule.
Context: Drug toxicity prediction is critical for early-stage drug discovery.
Question: Given the drug SMILES 'CC(=O)OC1=CC=CC=C1C(=O)O', predict its toxicity level.
Answer:"""
        
        result = loaded_model.generate(prompt, max_new_tokens=64)
        
        assert isinstance(result, str)
        assert len(result) > 0
        # Model may return short responses like "298" or "Low toxicity"
        # Just verify we got something back
        assert len(result.strip()) > 0
    
    def test_unload_and_reload(self):
        """Test unloading and reloading model."""
        model = TxGemmaModel()
        
        # Load
        model.load()
        assert model.is_loaded
        
        # Unload
        model.unload()
        assert not model.is_loaded
        
        # Reload
        model.load()
        assert model.is_loaded
        
        # Test it still works
        result = model.generate("Test prompt", max_new_tokens=10)
        assert isinstance(result, str)
        
        # Cleanup
        model.unload()


class TestTxGemmaModelEdgeCases:
    """Test edge cases and error handling."""
    
    def setup_method(self):
        """Reset singleton before each test."""
        TxGemmaModel._instance = None
    
    def teardown_method(self):
        """Clean up after each test."""
        if TxGemmaModel._instance and TxGemmaModel._instance.is_loaded:
            TxGemmaModel._instance.unload()
    
    def test_load_invalid_model(self):
        """Test loading with invalid model name."""
        model = TxGemmaModel(model_name="invalid/model-name")
        
        with pytest.raises(RuntimeError, match="Could not load TxGemma model"):
            model.load()
    
    def test_generate_empty_prompt(self):
        """Test generation with empty prompt."""
        model = TxGemmaModel()
        model.load()
        
        try:
            result = model.generate("")
            # May return empty or error, either is acceptable
            assert isinstance(result, str)
        finally:
            model.unload()
    
    def test_generate_very_long_prompt(self):
        """Test generation with very long prompt."""
        model = TxGemmaModel()
        model.load()
        
        try:
            # Create a very long prompt
            long_prompt = "Question: " + " ".join(["word"] * 1000) + "\nAnswer:"
            result = model.generate(long_prompt, max_new_tokens=10)
            
            assert isinstance(result, str)
        finally:
            model.unload()