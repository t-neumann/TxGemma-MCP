"""
TxGemma model wrappers - separate classes for predict and chat models.

Each model type has its own singleton class since they serve different purposes:
- TxGemmaPredictModel: Fast, deterministic predictions for TDC tasks
- TxGemmaChatModel: Conversational explanations and Q&A
"""

import logging
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

logger = logging.getLogger(__name__)


class TxGemmaPredictModel:
    """
    Singleton wrapper for TxGemma prediction models.
    
    Used for property predictions from TDC prompts.
    Optimized for fast, deterministic, short-form outputs.
    """
    
    _instance: Optional["TxGemmaPredictModel"] = None
    
    def __new__(cls, *args, **kwargs):
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
        Initialize prediction model configuration.
        
        Args:
            model_name: HuggingFace model ID
            max_new_tokens: Max tokens for predictions
        """
        if self._initialized:
            return
        
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModelForCausalLM] = None
        self._initialized = True
    
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
                torch_dtype=torch.float16,
            )
            logger.info("Predict model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load predict model: {e}")
            raise RuntimeError(f"Could not load TxGemma predict model: {e}") from e
    
    def generate(self, prompt: str, max_new_tokens: Optional[int] = None) -> str:
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
            do_sample=False,  # Deterministic
        )
        
        generated_ids = outputs[0][len(inputs["input_ids"][0]):]
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
    """
    
    _instance: Optional["TxGemmaChatModel"] = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(
        self,
        model_name: str = "google/txgemma-9b-chat",
        max_new_tokens: int = 200,
    ):
        """
        Initialize chat model configuration.
        
        Args:
            model_name: HuggingFace model ID
            max_new_tokens: Max tokens for chat responses
        """
        if self._initialized:
            return
        
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModelForCausalLM] = None
        self._initialized = True
    
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
                torch_dtype=torch.float16,
            )
            logger.info("Chat model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load chat model: {e}")
            raise RuntimeError(f"Could not load TxGemma chat model: {e}") from e
    
    def generate(self, prompt: str, max_new_tokens: Optional[int] = None) -> str:
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
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)
        
        # Generate response
        outputs = self.model.generate(
            input_ids=inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
        )
        
        # Decode response only
        response = self.tokenizer.decode(
            outputs[0, len(inputs[0]):],
            skip_special_tokens=True
        )
        
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


# Legacy compatibility - can be removed later
def get_model() -> TxGemmaPredictModel:
    """Get the prediction model (for backward compatibility)."""
    return get_predict_model()