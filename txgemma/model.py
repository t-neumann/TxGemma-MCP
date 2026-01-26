"""
TxGemma model wrappers - separate classes for predict and chat models.

Each model type has its own singleton class since they serve different purposes:
- TxGemmaPredictModel: Fast, deterministic predictions for TDC tasks
- TxGemmaChatModel: Conversational explanations and Q&A
"""

import logging
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from txgemma.config import get_config

logger = logging.getLogger(__name__)


class TxGemmaPredictModel:
    """
    Singleton wrapper for TxGemma prediction models.

    Used for property predictions from TDC prompts.
    Optimized for fast, deterministic, short-form outputs.

    Configuration loaded from config.yaml by default.
    """

    _instance: Optional["TxGemmaPredictModel"] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        model_name: str | None = None,
        max_new_tokens: int | None = None,
    ):
        """
        Initialize prediction model configuration.

        Priority (highest to lowest):
        1. Explicitly passed arguments
        2. Config file values
        3. Hardcoded defaults

        Args:
            model_name: HuggingFace model ID (overrides config if provided)
            max_new_tokens: Max tokens for predictions (overrides config if provided)
        """
        if self._initialized:
            return

        # Load config (may fail if config.yaml doesn't exist)
        try:
            config = get_config()
            config_model = config.predict.model
            config_max_tokens = config.predict.max_new_tokens
        except Exception as e:
            logger.warning(f"Could not load config, using defaults: {e}")
            config_model = None
            config_max_tokens = None

        # Priority: argument → config → hardcoded default
        self.model_name = (
            model_name
            if model_name is not None
            else (config_model if config_model is not None else "google/txgemma-2b-predict")
        )
        self.max_new_tokens = (
            max_new_tokens
            if max_new_tokens is not None
            else (config_max_tokens if config_max_tokens is not None else 64)
        )

        self.tokenizer: AutoTokenizer | None = None
        self.model: AutoModelForCausalLM | None = None
        self._initialized = True

        logger.info(
            f"TxGemmaPredictModel configured: {self.model_name}, max_tokens: {self.max_new_tokens}"
        )

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None

    def load(self) -> None:
        """Load the prediction model."""
        if self.is_loaded:
            logger.info("Predict model already loaded")
            return

        logger.info(f"Loading predict model: {self.model_name}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
                dtype=torch.float16,
            )
            logger.info("Predict model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load predict model: {e}")
            raise RuntimeError(f"Could not load TxGemma predict model: {e}") from e

    def generate(self, prompt: str, max_new_tokens: int | None = None) -> str:
        """
        Generate a prediction.

        Args:
            prompt: TDC-formatted prompt
            max_new_tokens: Override default max tokens

        Returns:
            Model prediction (short, deterministic)
        """
        if not self.is_loaded:
            self.load()

        max_tokens = max_new_tokens or self.max_new_tokens

        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
        )

        generated_ids = outputs[0][len(inputs["input_ids"][0]) :]
        result = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return result.strip()

    def unload(self) -> None:
        """Unload model to free memory."""
        if self.model is not None:
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            torch.cuda.empty_cache()
            logger.info("Predict model unloaded")


class TxGemmaChatModel:
    """
    Singleton wrapper for TxGemma chat models.

    Used for conversational Q&A and explanations.
    Optimized for detailed, explanatory responses.

    Configuration loaded from config.yaml by default.
    """

    _instance: Optional["TxGemmaChatModel"] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        model_name: str | None = None,
        max_new_tokens: int | None = None,
    ):
        """
        Initialize chat model configuration.

        Priority (highest to lowest):
        1. Explicitly passed arguments
        2. Config file values
        3. Hardcoded defaults

        Args:
            model_name: HuggingFace model ID (overrides config if provided)
            max_new_tokens: Max tokens for chat responses (overrides config if provided)
        """
        if self._initialized:
            return

        # Load config (may fail if config.yaml doesn't exist)
        try:
            config = get_config()
            config_model = config.chat.model
            config_max_tokens = config.chat.max_new_tokens
        except Exception as e:
            logger.warning(f"Could not load config, using defaults: {e}")
            config_model = None
            config_max_tokens = None

        # Priority: argument → config → hardcoded default
        self.model_name = (
            model_name
            if model_name is not None
            else (config_model if config_model is not None else "google/txgemma-9b-chat")
        )
        self.max_new_tokens = (
            max_new_tokens
            if max_new_tokens is not None
            else (config_max_tokens if config_max_tokens is not None else 200)
        )

        self.tokenizer: AutoTokenizer | None = None
        self.model: AutoModelForCausalLM | None = None
        self._initialized = True

        logger.info(
            f"TxGemmaChatModel configured: {self.model_name}, max_tokens: {self.max_new_tokens}"
        )

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None

    def load(self) -> None:
        """Load the chat model."""
        if self.is_loaded:
            logger.info("Chat model already loaded")
            return

        logger.info(f"Loading chat model: {self.model_name}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
                dtype=torch.float16,
            )
            logger.info("Chat model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load chat model: {e}")
            raise RuntimeError(f"Could not load TxGemma chat model: {e}") from e

    def generate(self, prompt: str, max_new_tokens: int | None = None) -> str:
        """
        Generate a conversational response.

        Args:
            prompt: User question or prompt
            max_new_tokens: Override default max tokens

        Returns:
            Conversational response with explanation
        """
        if not self.is_loaded:
            self.load()

        max_tokens = max_new_tokens or self.max_new_tokens

        # Format as chat message
        messages = [{"role": "user", "content": prompt}]

        # Apply chat template
        result = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )

        # CRITICAL: Extract tensor if result is BatchEncoding or dict-like
        if hasattr(result, "input_ids"):
            # It's a BatchEncoding object
            inputs = result.input_ids.to(self.model.device)
        elif isinstance(result, dict) and "input_ids" in result:
            # It's a dict
            inputs = result["input_ids"].to(self.model.device)
        else:
            # It's already a tensor
            inputs = result.to(self.model.device)

        # Generate response
        outputs = self.model.generate(input_ids=inputs, max_new_tokens=max_tokens)

        # Decode response only
        response = self.tokenizer.decode(outputs[0, len(inputs[0]) :], skip_special_tokens=True)

        return response.strip()

    def unload(self) -> None:
        """Unload model to free memory."""
        if self.model is not None:
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            torch.cuda.empty_cache()
            logger.info("Chat model unloaded")


# Singleton accessors
def get_predict_model() -> TxGemmaPredictModel:
    """Get the singleton TxGemmaPredictModel instance."""
    return TxGemmaPredictModel()


def get_chat_model() -> TxGemmaChatModel:
    """Get the singleton TxGemmaChatModel instance."""
    return TxGemmaChatModel()
