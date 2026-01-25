"""
Ministral-based entity extractor with QLoRA fine-tuning support.

Uses Ministral 3B for generative entity extraction, outputting structured JSON
with departure, destination, and intermediate stations.
"""

import json
import re
from typing import Any, Callable, Dict, List, Literal, Optional

from ..interfaces import EntityExtractor

DeviceType = Literal["auto", "cuda", "cpu"]


# Prompt template for entity extraction
ENTITY_PROMPT_TEMPLATE = (
    "[INST] Extrait les entités de voyage du texte.\n"
    "Réponds UNIQUEMENT avec un JSON: "
    '{"departure": "...", "destination": "...", "intermediate": [...]}\n\n'
    "Texte: {text}\n\n"
    "JSON: [/INST]"
)


class MinistralEntityExtractor(EntityExtractor):
    """
    Entity extractor using Ministral 3B with optional QLoRA fine-tuning.

    Uses a generative approach: the model outputs structured JSON
    containing departure, destination, and intermediate stations.
    """

    def __init__(
        self,
        model_name: str = "unsloth/Ministral-3-3B-Base-2512-bnb-4bit",
        adapter_path: Optional[str] = None,
        device: DeviceType = "auto",
        max_new_tokens: int = 100,
        temperature: float = 0.1,
    ) -> None:
        """
        Initialize the Ministral entity extractor.

        Args:
            model_name: HuggingFace model name or path
            tokenizer_name: Tokenizer name (defaults to official Mistral tokenizer)
            adapter_path: Path to LoRA adapter (None for base model)
            device: Device preference - "auto", "cuda", or "cpu"
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (lower = more deterministic)
        """
        from src.utils.device import get_torch_device

        self.device = get_torch_device(device)
        self.model_name = model_name
        self.adapter_path = adapter_path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        self._load_model()

    def _load_model(self) -> None:
        """Load the model and tokenizer."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading Ministral model for entity extraction: {self.model_name}...")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto" if self.device == "cuda" else None,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )

        if self.device == "cpu":
            self.model = self.model.to("cpu")

        # Load LoRA adapter if provided
        if self.adapter_path:
            self._load_adapter()

        self.model.eval()
        print(f"Ministral entity extractor ready (device: {self.device})")

    def _load_adapter(self) -> None:
        """Load LoRA adapter weights."""
        from peft import PeftModel

        print(f"Loading LoRA adapter from: {self.adapter_path}")
        self.model = PeftModel.from_pretrained(
            self.model,
            self.adapter_path,
        )
        print("LoRA adapter loaded successfully")

    @property
    def name(self) -> str:
        """Return display name for this extractor."""
        if self.adapter_path:
            return "Ministral-LoRA"
        return "Ministral"

    def _parse_json_output(self, generated_text: str) -> Dict[str, Any]:
        """
        Parse JSON from generated text.

        Args:
            generated_text: Raw generated text from model

        Returns:
            Dictionary with departure, destination, intermediate
        """
        default_result = {
            "departure": None,
            "destination": None,
            "intermediate": [],
        }

        # Try to find JSON in the output
        text = generated_text.strip()

        # Try direct JSON parsing
        try:
            result = json.loads(text)
            return self._normalize_result(result)
        except json.JSONDecodeError:
            pass

        # Try to extract JSON from text
        json_patterns = [
            r"\{[^{}]*\}",  # Simple JSON object
            r"\{.*?\}",  # Greedy match
        ]

        for pattern in json_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    result = json.loads(match)
                    return self._normalize_result(result)
                except json.JSONDecodeError:
                    continue

        # Fallback: try to extract entities from text
        return self._extract_from_text(text, default_result)

    def _normalize_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize parsed result to expected format."""
        normalized = {
            "departure": None,
            "destination": None,
            "intermediate": [],
        }

        # Handle departure
        dep = result.get("departure") or result.get("depart") or result.get("from")
        if dep and isinstance(dep, str) and dep.lower() not in ("null", "none", ""):
            normalized["departure"] = dep.strip()

        # Handle destination
        dest = result.get("destination") or result.get("arrivee") or result.get("to")
        if dest and isinstance(dest, str) and dest.lower() not in ("null", "none", ""):
            normalized["destination"] = dest.strip()

        # Handle intermediate
        inter = result.get("intermediate") or result.get("intermediaire") or result.get("via")
        if inter:
            if isinstance(inter, list):
                normalized["intermediate"] = [s.strip() for s in inter if s and isinstance(s, str)]
            elif isinstance(inter, str) and inter.lower() not in ("null", "none", ""):
                normalized["intermediate"] = [inter.strip()]

        return normalized

    def _extract_from_text(self, text: str, default: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fallback extraction from unstructured text.

        Args:
            text: Generated text
            default: Default result dictionary

        Returns:
            Extracted entities
        """
        result = default.copy()

        # Look for patterns like "departure: X" or "depart: X"
        dep_patterns = [
            r"(?:departure|depart|from)\s*[:\-]\s*[\"']?([A-Za-zÀ-ÿ\-\s]+)[\"']?",
            r"de\s+([A-Z][a-zÀ-ÿ]+(?:[\-\s][A-Z][a-zÀ-ÿ]+)*)",
        ]
        for pattern in dep_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                result["departure"] = match.group(1).strip()
                break

        # Look for destination patterns
        dest_patterns = [
            r"(?:destination|arrivee|to)\s*[:\-]\s*[\"']?([A-Za-zÀ-ÿ\-\s]+)[\"']?",
            r"(?:à|vers|pour)\s+([A-Z][a-zÀ-ÿ]+(?:[\-\s][A-Z][a-zÀ-ÿ]+)*)",
        ]
        for pattern in dest_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                result["destination"] = match.group(1).strip()
                break

        return result

    def extract(self, text: str) -> Dict[str, Any]:
        """
        Extract travel entities from the input text.

        Args:
            text: Input text to process

        Returns:
            Dictionary with keys:
            - departure: Optional[str] - Starting location
            - destination: Optional[str] - Ending location
            - intermediate: List[str] - Intermediate stops
        """
        import torch

        # Handle empty text
        if len(text.strip()) < 3:
            return {"departure": None, "destination": None, "intermediate": []}

        # Format prompt
        prompt = ENTITY_PROMPT_TEMPLATE.format(text=text)

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Decode only the new tokens
        generated_ids = outputs[0][inputs["input_ids"].shape[1] :]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return self._parse_json_output(generated_text)

    def extract_batch(
        self,
        texts: List[str],
        batch_size: int = 4,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> List[Dict[str, Any]]:
        """
        Extract entities from multiple texts.

        Args:
            texts: List of input texts
            batch_size: Batch size for processing
            progress_callback: Optional callback(current, total) for progress

        Returns:
            List of entity dictionaries
        """
        results: List[Dict[str, Any]] = []
        total = len(texts)

        for i in range(0, total, batch_size):
            batch = texts[i : i + batch_size]
            batch_results = [self.extract(text) for text in batch]
            results.extend(batch_results)

            if progress_callback:
                progress_callback(min(i + batch_size, total), total)

        return results
