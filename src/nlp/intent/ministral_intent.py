"""
Ministral-based intent classifier with QLoRA fine-tuning support.

Uses Ministral 3B for intent classification, supporting both:
- Prompt-based inference (zero-shot or few-shot)
- Fine-tuned inference with LoRA adapters
"""

import re
from typing import Callable, List, Literal, Optional, Tuple

from ..interfaces import IntentClassifier

DeviceType = Literal["auto", "cuda", "cpu"]


# Prompt template for intent classification
INTENT_PROMPT_TEMPLATE = (
    "[INST] Tu es un assistant spécialisé dans la détection d'intentions de voyage.\n"
    "Analyse le texte et détermine s'il s'agit d'une demande de voyage en train.\n"
    "Réponds uniquement par 'TRIP' (voyage) ou 'NOT_TRIP' (autre).\n\n"
    "Texte: {text}\n\n"
    "Intention: [/INST]"
)


class MinistralIntentClassifier(IntentClassifier):
    """
    Intent classifier using Ministral 3B with optional QLoRA fine-tuning.

    Can operate in two modes:
    - Prompt-based: Uses instruction prompting for classification
    - Fine-tuned: Loads LoRA adapters for improved accuracy
    """

    INTENT_LABELS = ["TRIP", "NOT_TRIP", "UNKNOWN"]

    def __init__(
        self,
        model_name: str = "unsloth/Ministral-3-3B-Base-2512-bnb-4bit",
        adapter_path: Optional[str] = None,
        adapter_path: Optional[str] = None,
        device: DeviceType = "auto",
        max_new_tokens: int = 10,
        temperature: float = 0.1,
    ) -> None:
        """
        Initialize the Ministral intent classifier.

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

        print(f"Loading Ministral model: {self.model_name}...")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model - pre-quantized models from Unsloth don't need BitsAndBytesConfig
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
        print(f"Ministral intent classifier ready (device: {self.device})")

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
        """Return display name for this classifier."""
        if self.adapter_path:
            return "Ministral-LoRA"
        return "Ministral"

    def _extract_intent(self, generated_text: str) -> Tuple[str, float]:
        """
        Extract intent from generated text.

        Args:
            generated_text: Raw generated text from model

        Returns:
            Tuple of (intent_label, confidence)
        """
        text_upper = generated_text.upper().strip()

        # Check for explicit labels
        if "TRIP" in text_upper and "NOT_TRIP" not in text_upper:
            return ("TRIP", 0.9)
        elif "NOT_TRIP" in text_upper or "NOT TRIP" in text_upper:
            return ("NOT_TRIP", 0.9)

        # Check for French equivalents
        trip_patterns = [
            r"\b(voyage|trajet|aller|partir|billet|train)\b",
            r"\b(oui|demande.*(voyage|trajet))\b",
        ]
        not_trip_patterns = [
            r"\b(non|autre|pas.*(voyage|trajet))\b",
        ]

        for pattern in not_trip_patterns:
            if re.search(pattern, text_upper, re.IGNORECASE):
                return ("NOT_TRIP", 0.7)

        for pattern in trip_patterns:
            if re.search(pattern, generated_text, re.IGNORECASE):
                return ("TRIP", 0.7)

        return ("UNKNOWN", 0.5)

    def classify(self, text: str) -> Tuple[str, float]:
        """
        Classify the intent of the input text.

        Args:
            text: Input text to classify

        Returns:
            Tuple of (intent_label, confidence) where:
            - intent_label: "TRIP", "NOT_TRIP", or "UNKNOWN"
            - confidence: Float between 0.0 and 1.0
        """
        import torch

        # Handle empty or very short text
        if len(text.strip()) < 3:
            return ("UNKNOWN", 0.5)

        # Format prompt
        prompt = INTENT_PROMPT_TEMPLATE.format(text=text)

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

        return self._extract_intent(generated_text)

    def classify_batch(
        self,
        texts: List[str],
        batch_size: int = 8,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> List[Tuple[str, float]]:
        """
        Classify multiple texts.

        Args:
            texts: List of input texts to classify
            batch_size: Batch size for processing
            progress_callback: Optional callback(current, total) for progress

        Returns:
            List of (intent_label, confidence) tuples
        """
        results: List[Tuple[str, float]] = []
        total = len(texts)

        for i in range(0, total, batch_size):
            batch = texts[i : i + batch_size]
            batch_results = [self.classify(text) for text in batch]
            results.extend(batch_results)

            if progress_callback:
                progress_callback(min(i + batch_size, total), total)

        return results
