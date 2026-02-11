"""
TxGemma model wrappers - separate classes for predict and chat models.

Each model type has its own singleton class since they serve different purposes:
- TxGemmaPredictModel: Fast, deterministic predictions for TDC tasks
- TxGemmaChatModel: Conversational explanations and Q&A
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from txgemma.config import get_config

logger = logging.getLogger(__name__)


class TxGemmaModelBase(ABC):
    """
    Base class for TxGemma models with singleton pattern.
    
    Provides common functionality:
    - Singleton pattern implementation
    - Configuration loading with priority (args → config → defaults)
    - Model loading/unloading
    - Memory management
    
    Subclasses must implement:
    - _get_default_model_name()
    - _get_default_max_tokens()
    - _load_config_values()
    - generate()
    """
    
    _instance: Optional["TxGemmaModelBase"] = None
    
    def __new__(cls, *args, **kwargs):
        """Singleton pattern: only one instance per class."""
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
        Initialize model configuration.

        Priority (highest to lowest):
        1. Explicitly passed arguments
        2. Config file values
        3. Hardcoded defaults

        Args:
            model_name: HuggingFace model ID (overrides config if provided)
            max_new_tokens: Max tokens for generation (overrides config if provided)
        """
        if self._initialized:
            return

        config_model, config_max_tokens = self._load_config_values()

        # Priority: argument -> config -> default
        self.model_name = (
            model_name
            if model_name is not None
            else (config_model if config_model is not None else self._get_default_model_name())
        )
        self.max_new_tokens = (
            max_new_tokens
            if max_new_tokens is not None
            else (config_max_tokens if config_max_tokens is not None else self._get_default_max_tokens())
        )

        self.tokenizer: AutoTokenizer | None = None
        self.model: AutoModelForCausalLM | None = None
        self._initialized = True

        logger.info(
            f"{self.__class__.__name__} configured: {self.model_name}, max_tokens: {self.max_new_tokens}"
        )
    
    @abstractmethod
    def _get_default_model_name(self) -> str:
        """Get default model name for this model type."""
        pass
    
    @abstractmethod
    def _get_default_max_tokens(self) -> int:
        """Get default max tokens for this model type."""
        pass
    
    @abstractmethod
    def _load_config_values(self) -> tuple[str | None, int | None]:
        """
        Load configuration values for this model type.
        
        Returns:
            Tuple of (model_name, max_tokens) from config, or (None, None) if config unavailable
        """
        pass
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded in memory."""
        return self.model is not None

    def load(self) -> None:
        """Load the model into memory."""
        if self.is_loaded:
            logger.info(f"{self.__class__.__name__} already loaded")
            return

        logger.info(f"Loading {self.__class__.__name__}: {self.model_name}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
                dtype=torch.float16,
            )
            logger.info(f"{self.__class__.__name__} loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load {self.__class__.__name__}: {e}")
            raise RuntimeError(f"Could not load {self.__class__.__name__}: {e}") from e

    def unload(self) -> None:
        """Unload model to free memory."""
        if self.model is not None:
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            torch.cuda.empty_cache()
            logger.info(f"{self.__class__.__name__} unloaded")

    @abstractmethod
    def generate(self, prompt: str, max_new_tokens: int | None = None) -> str:
        """
        Generate output from the model.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Override default max tokens
            
        Returns:
            Generated text
        """
        pass


class TxGemmaPredictModel(TxGemmaModelBase):
    """
    Singleton wrapper for TxGemma prediction models.

    Used for property predictions from TDC prompts.
    Optimized for fast, deterministic, short-form outputs.

    Configuration loaded from config.yaml by default.
    """
    
    _instance: Optional["TxGemmaPredictModel"] = None
    
    def _get_default_model_name(self) -> str:
        """Default prediction model."""
        return "google/txgemma-2b-predict"
    
    def _get_default_max_tokens(self) -> int:
        """Default max tokens for predictions."""
        return 64
    
    def _load_config_values(self) -> tuple[str | None, int | None]:
        """Load prediction model config."""
        try:
            config = get_config()
            return config.predict.model, config.predict.max_new_tokens
        except Exception as e:
            logger.warning(f"Could not load config, using defaults: {e}")
            return None, None

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


class TxGemmaChatModel(TxGemmaModelBase):
    """
    Singleton wrapper for TxGemma chat models.

    Used for conversational Q&A and explanations.
    Optimized for detailed, explanatory responses.

    Configuration loaded from config.yaml by default.
    """
    
    _instance: Optional["TxGemmaChatModel"] = None
    
    def _get_default_model_name(self) -> str:
        """Default chat model."""
        return "google/txgemma-9b-chat"
    
    def _get_default_max_tokens(self) -> int:
        """Default max tokens for chat."""
        return 200
    
    def _load_config_values(self) -> tuple[str | None, int | None]:
        """Load chat model config."""
        try:
            config = get_config()
            return config.chat.model, config.chat.max_new_tokens
        except Exception as e:
            logger.warning(f"Could not load config, using defaults: {e}")
            return None, None

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

        outputs = self.model.generate(input_ids=inputs, max_new_tokens=max_tokens)

        response = self.tokenizer.decode(outputs[0, len(inputs[0]) :], skip_special_tokens=True)

        return response.strip()

def get_predict_model() -> TxGemmaPredictModel:
    """Get the singleton TxGemmaPredictModel instance."""
    return TxGemmaPredictModel()


def get_chat_model() -> TxGemmaChatModel:
    """Get the singleton TxGemmaChatModel instance."""
    return TxGemmaChatModel()