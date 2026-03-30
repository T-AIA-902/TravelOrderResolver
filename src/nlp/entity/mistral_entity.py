"""
Mistral 7B entity extractor with QLoRA fine-tuning support.

Uses Mistral 7B Instruct for generative entity extraction, outputting structured JSON
with departure, destination, and intermediate stations.
"""

import json
import re
from typing import Any, Callable, Dict, List, Optional

from ..interfaces import EntityExtractor

ENTITY_PROMPT = """[INST] Analyse cette phrase de voyage.
Reponds en JSON avec: langue (fr/en/other), intention (TRIP/NOT_TRIP/UNKNOWN),
et si TRIP: depart, destination, intermediaires.

Phrase: {text} [/INST]"""


class MistralEntityExtractor(EntityExtractor):
    """
    Entity extractor using Mistral 7B with optional QLoRA adapter.
    Supports CUDA, MPS (Apple Silicon), and CPU.
    """

    def __init__(
        self,
        adapter_path: Optional[str] = "models/mistral-unified-lora",
        device: str = "auto",
        max_new_tokens: int = 256,
    ) -> None:
        self._adapter_path = adapter_path
        self._max_new_tokens = max_new_tokens
        self._model = None
        self._tokenizer = None
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

        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        base_model_name = "mistralai/Mistral-7B-Instruct-v0.3"
        print(f"Loading Mistral 7B: {base_model_name} ({self._device})...")

        self._tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            dtype=torch.float16,
            trust_remote_code=True,
        )

        if self._adapter_path:
            print(f"Loading LoRA adapter from: {self._adapter_path}")
            self._model = PeftModel.from_pretrained(self._model, self._adapter_path)

        self._model.to(self._device)
        self._model.eval()
        print(f"Mistral entity extractor ready (device: {self._device})")

    @property
    def name(self) -> str:
        return "Mistral-LoRA" if self._adapter_path else "Mistral"

    def _generate(self, text: str) -> str:
        import torch
        self._ensure_loaded()
        prompt = ENTITY_PROMPT.format(text=text)
        inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self._tokenizer.pad_token_id,
            )

        generated = outputs[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(generated, skip_special_tokens=True)

    def _parse_response(self, response: str) -> Dict[str, Any]:
        default = {"departure": None, "destination": None, "intermediate": []}

        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                dep = data.get("depart") or data.get("departure")
                dest = data.get("destination") or data.get("dest")
                inter = data.get("intermediaires") or data.get("intermediate") or []
                if isinstance(inter, str):
                    inter = [s.strip() for s in inter.split(",") if s.strip()]
                return {
                    "departure": dep if dep and str(dep).lower() not in ("null", "none", "") else None,
                    "destination": dest if dest and str(dest).lower() not in ("null", "none", "") else None,
                    "intermediate": [s for s in inter if s and str(s).lower() not in ("null", "none", "")],
                }
            except json.JSONDecodeError:
                pass
        return default

    def extract(self, text: str) -> Dict[str, Any]:
        if len(text.strip()) < 3:
            return {"departure": None, "destination": None, "intermediate": []}
        return self._parse_response(self._generate(text))

    def extract_batch(
        self,
        texts: List[str],
        batch_size: int = 4,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        total = len(texts)
        for i, text in enumerate(texts):
            results.append(self.extract(text))
            if progress_callback and (i + 1) % batch_size == 0:
                progress_callback(i + 1, total)
        if progress_callback:
            progress_callback(total, total)
        return results
