"""
Shared CamemBERT NER model wrapper.

Loads a fine-tuned CamembertForTokenClassification model once
and provides NER inference methods used by both
CamembertEntityExtractor and CamembertIntentClassifier.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple

DeviceType = Literal["auto", "cuda", "cpu"]

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_MODEL_DIR = _PROJECT_ROOT / "models" / "camembert-ner-retrain"


class CamembertNERModel:
    """Fine-tuned CamemBERT token-classification model.

    Wraps CamembertForTokenClassification and provides:
    - predict_ner: token-level NER with subword alignment
    - extract_entities: BIO tag grouping into DEP/DEST/STEP
    - predict_and_extract: convenience combining both
    - predict_batch: batched inference for evaluation
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        device: DeviceType = "auto",
    ) -> None:
        """Initialize CamemBERT NER model with tokenizer and label mapping."""
        try:
            import torch
            from transformers import CamembertForTokenClassification, CamembertTokenizerFast
        except ImportError:
            raise ImportError(
                "transformers/torch not available. " "Install with: pip install transformers torch"
            )

        from src.utils.device import get_torch_device

        self._torch = torch

        if model_path is None:
            model_path = _DEFAULT_MODEL_DIR
        self.model_path = Path(model_path)

        # Load label mapping from label_config.json
        label_config_path = self.model_path / "label_config.json"
        if label_config_path.exists():
            with open(label_config_path, "r", encoding="utf-8") as f:
                label_config = json.load(f)
            self.id2label: Dict[int, str] = {int(k): v for k, v in label_config["id2label"].items()}
        else:
            # Fallback: read from model config.json
            config_path = self.model_path / "config.json"
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            self.id2label = {int(k): v for k, v in config["id2label"].items()}

        self.device = get_torch_device(device)

        print(f"Loading fine-tuned CamemBERT NER from {self.model_path}...")
        self.tokenizer = CamembertTokenizerFast.from_pretrained(str(self.model_path))
        self.model = CamembertForTokenClassification.from_pretrained(str(self.model_path))
        self.model.to(self.device)
        self.model.eval()
        print(
            f"CamemBERT NER ready (device: {self.device}, "
            f"labels: {list(self.id2label.values())})"
        )

    def predict_ner(self, sentence: str) -> Tuple[List[str], List[str]]:
        """Run NER prediction on a sentence.

        Args:
            sentence: Input text.

        Returns:
            Tuple of (words, predicted_labels) at word level.
        """
        words = sentence.split()
        if not words:
            return [], []

        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            max_length=128,
            truncation=True,
            return_tensors="pt",
        )
        encoding = {k: v.to(self.device) for k, v in encoding.items()}

        with self._torch.no_grad():
            outputs = self.model(**encoding)

        predictions = outputs.logits.argmax(dim=-1).squeeze().cpu().tolist()
        if isinstance(predictions, int):
            predictions = [predictions]

        # Subword -> word-level alignment
        word_ids_list = self.tokenizer(
            words,
            is_split_into_words=True,
            max_length=128,
            truncation=True,
        ).word_ids()

        word_preds: List[str] = []
        prev_word_id = None
        for idx, word_id in enumerate(word_ids_list):
            if word_id is not None and word_id != prev_word_id:
                if idx < len(predictions):
                    word_preds.append(self.id2label[predictions[idx]])
                else:
                    word_preds.append("O")
            prev_word_id = word_id

        # Pad or truncate to match word count
        while len(word_preds) < len(words):
            word_preds.append("O")
        word_preds = word_preds[: len(words)]

        return words, word_preds

    @staticmethod
    def extract_entities(words: List[str], labels: List[str]) -> Dict[str, Any]:
        """Group BIO-tagged words into entities.

        Handles B-DEP/I-DEP, B-DEST/I-DEST, and B-STEP/I-STEP tags.

        Args:
            words: List of words.
            labels: Corresponding BIO labels.

        Returns:
            Dict with keys: departure, destination, intermediate.
        """
        entities: List[Tuple[str, str]] = []  # (type, text)
        current_tokens: List[str] = []
        current_type: str | None = None

        for word, label in zip(words, labels):
            if label.startswith("B-"):
                # Save previous entity
                if current_tokens and current_type:
                    entities.append((current_type, " ".join(current_tokens)))
                current_type = label[2:]  # DEP, DEST, or STEP
                current_tokens = [word]
            elif label.startswith("I-") and current_type:
                current_tokens.append(word)
            else:
                if current_tokens and current_type:
                    entities.append((current_type, " ".join(current_tokens)))
                current_tokens = []
                current_type = None

        # Final entity
        if current_tokens and current_type:
            entities.append((current_type, " ".join(current_tokens)))

        # Map to output format
        departure: str | None = None
        destination: str | None = None
        intermediate: List[str] = []

        for entity_type, entity_text in entities:
            if entity_type == "DEP" and departure is None:
                departure = entity_text
            elif entity_type == "DEST" and destination is None:
                destination = entity_text
            elif entity_type == "STEP":
                intermediate.append(entity_text)

        return {
            "departure": departure,
            "destination": destination,
            "intermediate": intermediate,
        }

    def predict_and_extract(self, sentence: str) -> Dict[str, Any]:
        """Run NER prediction and extract entities in one step.

        Args:
            sentence: Input text.

        Returns:
            Dict with keys: departure, destination, intermediate.
        """
        words, labels = self.predict_ner(sentence)
        return self.extract_entities(words, labels)

    def predict_batch(
        self,
        sentences: List[str],
        batch_size: int = 32,
    ) -> List[Dict[str, Any]]:
        """Run batched NER prediction and entity extraction.

        Args:
            sentences: List of input texts.
            batch_size: Number of sentences per batch.

        Returns:
            List of entity dicts.
        """
        results: List[Dict[str, Any]] = []

        for i in range(0, len(sentences), batch_size):
            batch = sentences[i : i + batch_size]
            for sentence in batch:
                results.append(self.predict_and_extract(sentence))

        return results
