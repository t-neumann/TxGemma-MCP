"""
TxGemma model wrapper with lazy loading and singleton pattern.
"""

import logging
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TxGemmaModel:
    """
    Lazy-loaded TxGemma model with singleton pattern.

    The model is not loaded until the first generate() call, which is important
    for server startup time and memory management.
    """

    _instance: Optional["TxGemmaModel"] = None

    def __new__(cls, model_name: str = "google/txgemma-2b-predict", max_new_tokens: int = 64):
        """Singleton: only one model instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        model_name: str = "google/txgemma-2b-predict",
        max_new_tokens: int = 64,
    ):
        """
        Initialize model configuration (but don't load yet).

        Args:
            model_name: HuggingFace model identifier
            max_new_tokens: Maximum tokens to generate per request
        """
        # Skip if already initialized
        if self._initialized:
            return

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens

        self.tokenizer = None
        self.model = None
        self._loaded = False

        logger.info(f"TxGemmaModel configured: {model_name}")
        self._initialized = True

    def load(self):
        """
        Load model and tokenizer into GPU memory.
        This is called automatically on first generate().
        """
        if self._loaded:
            return

        logger.info(f"Loading model: {self.model_name}")
        logger.info("This may take a few minutes on first load...")

        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            # Load model with auto device mapping
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
                dtype=torch.float16,
            )

            self._loaded = True
            logger.info(f"Model loaded successfully on device: {self.model.device}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Could not load TxGemma model: {e}") from e

    def generate(self, prompt: str, max_new_tokens: int | None = None) -> str:
        """
        Generate text from prompt.

        Args:
            prompt: Input prompt (typically from TDC template)
            max_new_tokens: Override default max tokens

        Returns:
            Generated text (excluding prompt)
        """
        # Lazy load on first call
        self.load()

        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens

        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Deterministic for predictions
            )

        # Decode only the generated part (skip input prompt)
        generated_ids = outputs[0][inputs.input_ids.shape[1] :]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return generated_text.strip()

    def unload(self):
        """
        Unload model from memory (useful for testing or cleanup).
        """
        if self._loaded:
            del self.model
            del self.tokenizer
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            self._loaded = False
            logger.info("Model unloaded from memory")

    @property
    def is_loaded(self) -> bool:
        """Check if model is currently loaded."""
        return self._loaded


# Global getter function
_model_instance: TxGemmaModel | None = None


def get_model(
    model_name: str = "google/txgemma-2b-predict", max_new_tokens: int = 64
) -> TxGemmaModel:
    """
    Get the global TxGemma model instance.
    Creates it if it doesn't exist.

    Args:
        model_name: HuggingFace model identifier
        max_new_tokens: Maximum tokens to generate per request

    Returns:
        TxGemmaModel singleton instance

    Note:
        Arguments are only used on first call. Subsequent calls return
        the existing singleton regardless of arguments passed.
    """
    global _model_instance
    if _model_instance is None:
        _model_instance = TxGemmaModel(model_name=model_name, max_new_tokens=max_new_tokens)
    return _model_instance
