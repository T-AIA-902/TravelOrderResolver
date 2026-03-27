"""
Flan-T5 model loader and utilities.

Provides model loading and generation utilities for
both intent classification and entity extraction.

Two separate instances are expected:
- Entity extractor: uses the local fine-tuned model (models/flan-t5-travel/)
- Intent classifier: uses the base model (google/flan-t5-base)

A per-path cache avoids reloading the same model twice.
"""

import os
from typing import Dict, Optional, Tuple

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Default to local fine-tuned model if available
DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
    "models",
    "flan-t5-travel",
)

# Per-path cache: allows one loader per model path without reloading
_loader_cache: Dict[str, "FlanT5ModelLoader"] = {}


def get_flan_t5_loader(model_name: Optional[str] = None) -> "FlanT5ModelLoader":
    """Return a cached FlanT5ModelLoader for the given model path.

    Different model paths get independent loaders so that the fine-tuned
    entity model and the base intent model can coexist.
    """
    if model_name is None:
        if os.path.exists(DEFAULT_MODEL_PATH):
            model_name = DEFAULT_MODEL_PATH
        else:
            model_name = "google/flan-t5-base"

    if model_name not in _loader_cache:
        _loader_cache[model_name] = FlanT5ModelLoader(model_name)
    return _loader_cache[model_name]


def _select_device() -> str:
    """Select the best available device, avoiding MPS for T5 compatibility."""
    if torch.cuda.is_available():
        return "cuda"
    # MPS causes RuntimeError with T5 safetensors ("Placeholder storage has
    # not been allocated on MPS device"), so we fall back to CPU on Apple Silicon.
    return "cpu"


class FlanT5ModelLoader:
    """
    Flan-T5 model loader with per-path caching.

    Each distinct model path gets its own loader instance so that the
    fine-tuned entity model and the base intent model can coexist.
    """

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model: Optional[T5ForConditionalGeneration] = None
        self._tokenizer: Optional[T5Tokenizer] = None
        self._device: str = "cpu"

    def load(self) -> Tuple[T5ForConditionalGeneration, T5Tokenizer]:
        """
        Load and return the model and tokenizer.

        Returns:
            Tuple of (model, tokenizer)
        """
        if self._model is None or self._tokenizer is None:
            print(f"Loading Flan-T5 model: {self._model_name}...")

            self._tokenizer = T5Tokenizer.from_pretrained(self._model_name)
            self._model = T5ForConditionalGeneration.from_pretrained(
                self._model_name,
                dtype=torch.float32,
            )
            self._model.eval()

            self._device = _select_device()
            self._model.to(self._device)

            print(f"Flan-T5 loaded on {self._device}")

        return self._model, self._tokenizer

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

        return str(tokenizer.decode(outputs[0], skip_special_tokens=True))
