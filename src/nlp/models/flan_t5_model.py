"""
Flan-T5 model loader and utilities.

Provides shared model loading and generation utilities for
both intent classification and entity extraction.
"""

import os
from typing import Optional, Tuple

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Default to local fine-tuned model if available
DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
    "models",
    "flan-t5-travel",
)


class FlanT5ModelLoader:
    """
    Shared Flan-T5 model loader with caching.

    Ensures the model is loaded only once and shared between
    intent classifier and entity extractor (Singleton pattern).

    By default, uses the locally fine-tuned model at models/flan-t5-travel/.
    Falls back to HuggingFace model if local model not found.
    """

    _instance: Optional["FlanT5ModelLoader"] = None
    _model: Optional[T5ForConditionalGeneration] = None
    _tokenizer: Optional[T5Tokenizer] = None
    _model_name: Optional[str] = None
    _device: str = "cpu"

    def __new__(cls, model_name: Optional[str] = None) -> "FlanT5ModelLoader":
        # Use local model by default if available
        if model_name is None:
            if os.path.exists(DEFAULT_MODEL_PATH):
                model_name = DEFAULT_MODEL_PATH
            else:
                model_name = "google/flan-t5-base"
        """Singleton pattern to avoid loading model multiple times."""
        if cls._instance is None or cls._model_name != model_name:
            cls._instance = super().__new__(cls)
            cls._model_name = model_name
            cls._model = None
            cls._tokenizer = None
        return cls._instance

    def load(self) -> Tuple[T5ForConditionalGeneration, T5Tokenizer]:
        """
        Load and return the model and tokenizer.

        Returns:
            Tuple of (model, tokenizer)
        """
        if self._model is None or self._tokenizer is None:
            print(f"Loading Flan-T5 model: {self._model_name}...")

            self.__class__._tokenizer = T5Tokenizer.from_pretrained(self._model_name)
            self.__class__._model = T5ForConditionalGeneration.from_pretrained(
                self._model_name,
                torch_dtype=torch.float32,
            )
            self.__class__._model.eval()

            # Determine device (CUDA > MPS > CPU)
            if torch.cuda.is_available():
                self.__class__._device = "cuda"
            elif torch.backends.mps.is_available():
                self.__class__._device = "mps"
            else:
                self.__class__._device = "cpu"
            self.__class__._model.to(self._device)

            print(f"Flan-T5 loaded on {self._device}")

        return self._model, self._tokenizer  # type: ignore[return-value]

    @property
    def device(self) -> str:
        """Return the device the model is on."""
        return self._device

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        temperature: float = 0.1,
    ) -> str:
        """
        Generate text from a prompt.

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (lower = more deterministic)

        Returns:
            Generated text
        """
        model, tokenizer = self.load()

        inputs = tokenizer(  # pylint: disable=not-callable
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self._device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                num_beams=1,
            )

        return tokenizer.decode(outputs[0], skip_special_tokens=True)
