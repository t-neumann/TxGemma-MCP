"""
Tests for txgemma.model module.

These tests require GPU and model download (~5GB for predict, ~18GB for chat).
Run with: pytest tests/test_model.py --run-gpu
"""

import pytest
import torch

from txgemma.model import TxGemmaChatModel, TxGemmaPredictModel, get_chat_model, get_predict_model

# Mark all tests in this file as requiring GPU
pytestmark = pytest.mark.gpu


class TestTxGemmaPredictModelUnit:
    """Unit tests for predict model that don't require model loading."""

    def setup_method(self):
        """Reset singleton before each test."""
        TxGemmaPredictModel._instance = None

    def test_init_default(self):
        """Test model initialization with defaults."""
        model = TxGemmaPredictModel()

        assert model.model_name == "google/txgemma-2b-predict"
        assert model.max_new_tokens == 64
        assert not model.is_loaded
        assert model.tokenizer is None
        assert model.model is None

    def test_init_custom(self):
        """Test model initialization with custom parameters."""
        model = TxGemmaPredictModel(model_name="google/txgemma-9b-predict", max_new_tokens=128)

        assert model.model_name == "google/txgemma-9b-predict"
        assert model.max_new_tokens == 128
        assert not model.is_loaded

    def test_singleton_pattern(self):
        """Test that TxGemmaPredictModel uses singleton pattern."""
        model1 = TxGemmaPredictModel()
        model2 = TxGemmaPredictModel()

        assert model1 is model2

    def test_get_predict_model_singleton(self):
        """Test that get_predict_model returns singleton."""
        model1 = get_predict_model()
        model2 = get_predict_model()

        assert model1 is model2

    def test_is_loaded_before_load(self):
        """Test is_loaded property before loading."""
        model = TxGemmaPredictModel()
        assert not model.is_loaded


class TestTxGemmaChatModelUnit:
    """Unit tests for chat model that don't require model loading."""

    def setup_method(self):
        """Reset singleton before each test."""
        TxGemmaChatModel._instance = None

    def test_init_default(self):
        """Test model initialization with defaults."""
        model = TxGemmaChatModel()

        assert model.model_name == "google/txgemma-9b-chat"
        assert model.max_new_tokens == 100
        assert not model.is_loaded
        assert model.tokenizer is None
        assert model.model is None

    def test_init_custom(self):
        """Test model initialization with custom parameters."""
        model = TxGemmaChatModel(model_name="google/txgemma-27b-chat", max_new_tokens=300)

        assert model.model_name == "google/txgemma-27b-chat"
        assert model.max_new_tokens == 300
        assert not model.is_loaded

    def test_singleton_pattern(self):
        """Test that TxGemmaChatModel uses singleton pattern."""
        model1 = TxGemmaChatModel()
        model2 = TxGemmaChatModel()

        assert model1 is model2

    def test_get_chat_model_singleton(self):
        """Test that get_chat_model returns singleton."""
        model1 = get_chat_model()
        model2 = get_chat_model()

        assert model1 is model2

    def test_is_loaded_before_load(self):
        """Test is_loaded property before loading."""
        model = TxGemmaChatModel()
        assert not model.is_loaded


class TestTxGemmaPredictModelIntegration:
    """Integration tests for predict model that require GPU."""

    @pytest.fixture(scope="class")
    def loaded_model(self):
        """Fixture that loads predict model once for all tests in class."""
        model = TxGemmaPredictModel()
        model.load()
        yield model
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
        assert len(result.strip()) > 0

    def test_unload_and_reload(self):
        """Test unloading and reloading predict model."""
        model = TxGemmaPredictModel()

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


class TestTxGemmaChatModelIntegration:
    """Integration tests for chat model that require GPU."""

    @pytest.fixture(scope="class")
    def loaded_chat_model(self):
        """Fixture that loads chat model once for all tests in class."""
        model = TxGemmaChatModel()
        model.load()
        yield model
        model.unload()

    def test_load_chat_model(self, loaded_chat_model):
        """Test chat model loading."""
        assert loaded_chat_model.is_loaded
        assert loaded_chat_model.tokenizer is not None
        assert loaded_chat_model.model is not None

    def test_chat_model_device(self, loaded_chat_model):
        """Test that chat model is on correct device."""
        if torch.cuda.is_available():
            assert "cuda" in str(loaded_chat_model.model.device)
        elif torch.backends.mps.is_available():
            assert "mps" in str(loaded_chat_model.model.device)

    def test_generate_chat_simple(self, loaded_chat_model):
        """Test simple chat generation."""
        prompt = "What is a SMILES string?"
        result = loaded_chat_model.generate(prompt)

        assert isinstance(result, str)
        assert len(result) > 0
        # Chat responses should be more verbose than predictions
        assert len(result) > 10

    def test_generate_chat_with_drug_question(self, loaded_chat_model):
        """Test chat with drug-specific question."""
        prompt = "Why might aspirin cause stomach issues?"
        result = loaded_chat_model.generate(prompt)

        assert isinstance(result, str)
        assert len(result) > 0

    def test_chat_deterministic(self, loaded_chat_model):
        """Test that chat generation is deterministic."""
        prompt = "What is toxicity?"

        result1 = loaded_chat_model.generate(prompt, max_new_tokens=50)
        result2 = loaded_chat_model.generate(prompt, max_new_tokens=50)

        # Should be identical with do_sample=False
        assert result1 == result2

    def test_unload_and_reload_chat(self):
        """Test unloading and reloading chat model."""
        model = TxGemmaChatModel()

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
        result = model.generate("What is a drug?", max_new_tokens=20)
        assert isinstance(result, str)

        # Cleanup
        model.unload()


class TestPredictModelEdgeCases:
    """Test edge cases and error handling for predict model."""

    def setup_method(self):
        """Reset singleton before each test."""
        TxGemmaPredictModel._instance = None

    def teardown_method(self):
        """Clean up after each test."""
        if TxGemmaPredictModel._instance and TxGemmaPredictModel._instance.is_loaded:
            TxGemmaPredictModel._instance.unload()

    def test_load_invalid_model(self):
        """Test loading with invalid model name."""
        model = TxGemmaPredictModel(model_name="invalid/model-name")

        with pytest.raises(RuntimeError, match="Could not load TxGemma predict model"):
            model.load()

    def test_generate_empty_prompt(self):
        """Test generation with empty prompt."""
        model = TxGemmaPredictModel()
        model.load()

        try:
            result = model.generate("")
            assert isinstance(result, str)
        finally:
            model.unload()

    def test_generate_very_long_prompt(self):
        """Test generation with very long prompt."""
        model = TxGemmaPredictModel()
        model.load()

        try:
            long_prompt = "Question: " + " ".join(["word"] * 1000) + "\nAnswer:"
            result = model.generate(long_prompt, max_new_tokens=10)

            assert isinstance(result, str)
        finally:
            model.unload()


class TestChatModelEdgeCases:
    """Test edge cases and error handling for chat model."""

    def setup_method(self):
        """Reset singleton before each test."""
        TxGemmaChatModel._instance = None

    def teardown_method(self):
        """Clean up after each test."""
        if TxGemmaChatModel._instance and TxGemmaChatModel._instance.is_loaded:
            TxGemmaChatModel._instance.unload()

    def test_load_invalid_chat_model(self):
        """Test loading with invalid chat model name."""
        model = TxGemmaChatModel(model_name="invalid/chat-model")

        with pytest.raises(RuntimeError, match="Could not load TxGemma chat model"):
            model.load()

    def test_generate_chat_empty_prompt(self):
        """Test chat generation with empty prompt."""
        model = TxGemmaChatModel()
        model.load()

        try:
            result = model.generate("")
            assert isinstance(result, str)
        finally:
            model.unload()
