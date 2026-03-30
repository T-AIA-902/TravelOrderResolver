"""
Mistral 7B intent classifier with QLoRA fine-tuning support.

Derives intent from the unified JSON response.
Supports CUDA, MPS (Apple Silicon), and CPU.
"""

import json
import re
from typing import Any, Callable, List, Optional, Tuple

from ..interfaces import IntentClassifier

INTENT_PROMPT = """[INST] Analyse cette phrase de voyage.
Reponds en JSON avec: langue (fr/en/other), intention (TRIP/NOT_TRIP/UNKNOWN),
et si TRIP: depart, destination, intermediaires.

Phrase: {text} [/INST]"""


class MistralIntentClassifier(IntentClassifier):
    """
    Intent classifier using Mistral 7B with optional QLoRA adapter.
    Shares model with MistralEntityExtractor when possible.
    """

    def __init__(
        self,
        adapter_path: Optional[str] = "models/mistral-unified-lora",
        device: str = "auto",
        ner_model: Any = None,
    ) -> None:
        self._adapter_path = adapter_path
        self._model = None
        self._tokenizer = None
        self._shared_model = ner_model
        self._device = self._select_device(device)

    @staticmethod
    def _select_device(device: str) -> str:
        import torch
        if device != "auto":
            return device
        if torch.cuda.is_available():
            return "cuda"
        # MPS crashes with Mistral GQA, so skip it
        return "cpu"

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        if self._shared_model is not None:
            self._shared_model._ensure_loaded()
            self._model = self._shared_model._model
            self._tokenizer = self._shared_model._tokenizer
            self._device = self._shared_model._device
            return

        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        base_model_name = "mistralai/Mistral-7B-Instruct-v0.3"
        print(f"Loading Mistral 7B for intent: {base_model_name} ({self._device})...")

        self._tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            dtype=torch.float16,
            trust_remote_code=True,
        )

        if self._adapter_path:
            self._model = PeftModel.from_pretrained(self._model, self._adapter_path)

        self._model.to(self._device)
        self._model.eval()
        print(f"Mistral intent classifier ready (device: {self._device})")

    @property
    def name(self) -> str:
        return "Mistral-LoRA" if self._adapter_path else "Mistral"

    def classify(self, text: str) -> Tuple[str, float]:
        import torch

        if len(text.strip()) < 3:
            return ("UNKNOWN", 0.5)

        self._ensure_loaded()
        prompt = INTENT_PROMPT.format(text=text)
        inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self._tokenizer.pad_token_id,
            )

        generated = outputs[0][inputs["input_ids"].shape[1]:]
        response = self._tokenizer.decode(generated, skip_special_tokens=True)

        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                intent = data.get("intention", "UNKNOWN")
                if intent in ("TRIP", "NOT_TRIP"):
                    return (intent, 0.9)
                return ("UNKNOWN", 0.5)
            except json.JSONDecodeError:
                pass

        upper = response.upper()
        if "NOT_TRIP" in upper or "NOT TRIP" in upper:
            return ("NOT_TRIP", 0.7)
        if "TRIP" in upper:
            return ("TRIP", 0.7)
        return ("UNKNOWN", 0.5)

    def classify_batch(
        self,
        texts: List[str],
        batch_size: int = 4,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[Tuple[str, float]]:
        results: List[Tuple[str, float]] = []
        total = len(texts)
        for i, text in enumerate(texts):
            results.append(self.classify(text))
            if progress_callback and (i + 1) % batch_size == 0:
                progress_callback(i + 1, total)
        if progress_callback:
            progress_callback(total, total)
        return results
