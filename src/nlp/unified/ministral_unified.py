"""
Unified NLP pipeline using fine-tuned Ministral 3B.

Handles language detection, intent classification, and entity extraction
in a single inference pass using a generative approach.

The model is trained to output structured JSON responses containing all
three components, enabling efficient single-pass inference.

Usage:
    nlp = MinistralUnifiedNLP("models/ministral-unified-lora")
    result = nlp.process("Je voudrais aller de Paris a Lyon")
    # Returns: {"langue": "fr", "intention": "TRIP", "depart": "Paris", ...}
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from src.nlp.interfaces import EntityExtractor, IntentClassifier, LanguageDetector


class MinistralUnifiedNLP(LanguageDetector, IntentClassifier, EntityExtractor):
    """
    Single model that performs language detection, intent classification,
    and entity extraction in one inference pass.

    This class implements all three NLP interfaces (LanguageDetector,
    IntentClassifier, EntityExtractor) using a fine-tuned Ministral 3B model.

    Attributes:
        model: The loaded Ministral model with LoRA adapter
        tokenizer: The tokenizer for the model
        _name: Display name for this model
    """

    PROMPT_TEMPLATE = """[INST] Analyse cette phrase de voyage.
Reponds en JSON avec: langue (fr/en/other), intention (TRIP/NOT_TRIP/UNKNOWN),
et si TRIP: depart, destination, intermediaires.

Phrase: {text} [/INST]"""

    def __init__(
        self,
        adapter_path: str = "models/ministral-unified-lora",
        base_model: str = "unsloth/Ministral-3-3B-Instruct-2512",
        max_seq_length: int = 512,
        load_in_4bit: bool = True,
        device: Optional[str] = None,
    ):
        """
        Initialize the unified NLP model.

        Args:
            adapter_path: Path to the LoRA adapter directory
            base_model: HuggingFace model name for the base model
            max_seq_length: Maximum sequence length for generation
            load_in_4bit: Whether to load model in 4-bit quantization
            device: Device to load model on (auto-detected if None)
        """
        self._name = "MinistralUnified"
        self.adapter_path = adapter_path
        self.base_model = base_model
        self.max_seq_length = max_seq_length
        self.load_in_4bit = load_in_4bit
        self._model = None
        self._tokenizer = None
        self._device = device

    def _ensure_loaded(self) -> None:
        """Lazy load the model and tokenizer."""
        if self._model is not None:
            return

        try:
            from unsloth import FastVisionModel
        except ImportError as e:
            raise ImportError(
                "unsloth is required for MinistralUnifiedNLP. " "Install with: pip install unsloth"
            ) from e

        # Load base model
        self._model, self._tokenizer = FastVisionModel.from_pretrained(
            self.base_model,
            max_seq_length=self.max_seq_length,
            load_in_4bit=self.load_in_4bit,
            dtype=None,
        )

        # Load LoRA adapter
        self._model.load_adapter(self.adapter_path)

        # Set to inference mode
        FastVisionModel.for_inference(self._model)

    @property
    def name(self) -> str:
        """Return the name of this model."""
        return self._name

    def _generate(self, text: str) -> Dict[str, Any]:
        """
        Generate unified response from the model.

        Args:
            text: Input text to process

        Returns:
            Dictionary with language, intent, and entity information
        """
        self._ensure_loaded()

        prompt = self.PROMPT_TEMPLATE.format(text=text)
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)

        outputs = self._model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.1,
            do_sample=False,
            pad_token_id=self._tokenizer.pad_token_id,
        )

        response = self._tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract JSON from response (after [/INST])
        if "[/INST]" in response:
            response = response.split("[/INST]")[-1].strip()

        # Parse JSON
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        # Fallback if parsing fails
        return {
            "langue": "unknown",
            "intention": "UNKNOWN",
            "depart": None,
            "destination": None,
            "intermediaires": [],
        }

    def process(self, text: str) -> Dict[str, Any]:
        """
        Full unified pipeline in one call.

        Args:
            text: Input text to process

        Returns:
            Dictionary with all extracted information:
            - langue: Detected language (fr/en/other)
            - intention: Classified intent (TRIP/NOT_TRIP/UNKNOWN)
            - depart: Departure station (or None)
            - destination: Destination station (or None)
            - intermediaires: List of intermediate stops
        """
        return self._generate(text)

    def process_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Process multiple texts.

        Args:
            texts: List of input texts

        Returns:
            List of result dictionaries
        """
        return [self.process(text) for text in texts]

    # LanguageDetector interface
    def detect(self, text: str) -> Tuple[str, float]:
        """
        Detect the language of input text.

        Args:
            text: Input text to analyze

        Returns:
            Tuple of (language, confidence) where language is
            "FRENCH", "ENGLISH", or "UNKNOWN"
        """
        result = self._generate(text)
        lang = result.get("langue", "unknown")

        # Map to expected format
        if lang == "fr":
            return ("FRENCH", 0.9)
        elif lang == "en":
            return ("ENGLISH", 0.9)
        else:
            return ("UNKNOWN", 0.5)

    def detect_batch(self, texts: List[str]) -> List[Tuple[str, float]]:
        """Detect language for multiple texts."""
        return [self.detect(text) for text in texts]

    # IntentClassifier interface
    def classify(self, text: str) -> Tuple[str, float]:
        """
        Classify the intent of input text.

        Args:
            text: Input text to classify

        Returns:
            Tuple of (intent_label, confidence) where intent_label
            is "TRIP", "NOT_TRIP", or "UNKNOWN"
        """
        result = self._generate(text)
        intent = result.get("intention", "UNKNOWN")

        if intent in ("TRIP", "NOT_TRIP", "UNKNOWN"):
            return (intent, 0.9)
        return ("UNKNOWN", 0.5)

    # EntityExtractor interface
    def extract(self, text: str) -> Dict[str, Any]:
        """
        Extract travel entities from input text.

        Args:
            text: Input text to process

        Returns:
            Dictionary with:
            - departure: Optional departure station
            - destination: Optional destination station
            - intermediate: List of intermediate stops
        """
        result = self._generate(text)

        return {
            "departure": result.get("depart") or "",
            "destination": result.get("destination") or "",
            "intermediate": result.get("intermediaires", []),
        }
