"""
CamemBERT NER retrain entity extractor.

Uses the fine-tuned CamemBERT model (models/camembert-ner-retrain/) for
token classification NER with labels: O, B-DEP, I-DEP, B-DEST, I-DEST.

Integrates fuzzy matching against the SNCF station database to normalize
extracted entity names to real station names.
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..interfaces import EntityExtractor

PROJECT_ROOT = Path(__file__).resolve().parents[3]
# Use camembert-ner-finetuned if retrain weights are not available locally
_RETRAIN_DIR = PROJECT_ROOT / "models" / "camembert-ner-retrain"
_FINETUNED_DIR = PROJECT_ROOT / "models" / "camembert-ner-finetuned"
DEFAULT_MODEL_DIR = (
    _RETRAIN_DIR if (_RETRAIN_DIR / "model.safetensors").exists()
    else _FINETUNED_DIR
)

LABEL_LIST = ["O", "B-DEP", "I-DEP", "B-DEST", "I-DEST"]
ID2LABEL = {idx: label for idx, label in enumerate(LABEL_LIST)}


class CamembertNerRetrainExtractor(EntityExtractor):
    """
    Entity extractor using the fine-tuned CamemBERT NER model.

    Performs token-level classification to identify departure (DEP)
    and destination (DEST) entities, then fuzzy-matches them against
    the SNCF station database.
    """

    def __init__(
        self,
        model_dir: Optional[str] = None,
        device: str = "auto",
        fuzzy_threshold: int = 80,
    ) -> None:
        model_dir = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR

        import torch
        from transformers import CamembertTokenizerFast, CamembertForTokenClassification

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"Loading CamemBERT NER retrain model from {model_dir}...")
        self.tokenizer = CamembertTokenizerFast.from_pretrained(str(model_dir))
        self.model = CamembertForTokenClassification.from_pretrained(str(model_dir))
        self.model.to(self.device)
        self.model.eval()
        self.torch = torch

        self.fuzzy_threshold = fuzzy_threshold
        self.stations = self._load_station_names()
        print(
            f"CamemBERT NER retrain ready (device: {self.device}, "
            f"{len(self.stations)} stations loaded)"
        )

    @property
    def name(self) -> str:
        return "CamemBERT-NER-Retrain"

    def _load_station_names(self) -> List[str]:
        """Load SNCF station names for fuzzy matching."""
        import json
        stations = set()

        gares_csv = PROJECT_ROOT / "datasets" / "raw" / "sncf" / "gares.csv"
        gares_json = PROJECT_ROOT / "datasets" / "raw" / "sncf" / "gares-de-voyageurs.json"

        if gares_csv.exists():
            import pandas as pd
            df = pd.read_csv(gares_csv)
            for _, row in df.iterrows():
                for col in ("city_name", "stations_name"):
                    if col in df.columns:
                        val = str(row[col]).strip()
                        if val and val.lower() != "nan":
                            stations.add(val)

        if gares_json.exists():
            with open(gares_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            for s in data:
                nom = s.get("nom", "").strip()
                if nom:
                    stations.add(nom)

        return sorted(stations)

    def _predict_ner(self, sentence: str) -> tuple[list[str], list[str]]:
        """Run NER prediction on a sentence. Returns (words, labels)."""
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

        with self.torch.no_grad():
            outputs = self.model(**encoding)

        predictions = outputs.logits.argmax(dim=-1).squeeze().cpu().tolist()
        if isinstance(predictions, int):
            predictions = [predictions]

        word_ids_list = self.tokenizer(
            words,
            is_split_into_words=True,
            max_length=128,
            truncation=True,
        ).word_ids()

        word_preds = []
        prev_word_id = None
        for idx, word_id in enumerate(word_ids_list):
            if word_id is not None and word_id != prev_word_id:
                if idx < len(predictions):
                    word_preds.append(ID2LABEL[predictions[idx]])
                else:
                    word_preds.append("O")
            prev_word_id = word_id

        while len(word_preds) < len(words):
            word_preds.append("O")
        word_preds = word_preds[:len(words)]

        return words, word_preds

    def _extract_entities(
        self, words: List[str], labels: List[str]
    ) -> Dict[str, Optional[str]]:
        """Group B-/I- tokens into departure and destination strings."""
        departure_tokens: List[str] = []
        destination_tokens: List[str] = []
        current_tokens: List[str] = []
        current_type: Optional[str] = None

        for word, label in zip(words, labels):
            if label.startswith("B-"):
                if current_tokens and current_type:
                    if "DEP" in current_type:
                        departure_tokens = current_tokens
                    elif "DEST" in current_type:
                        destination_tokens = current_tokens
                current_type = label[2:]
                current_tokens = [word]
            elif label.startswith("I-") and current_type:
                current_tokens.append(word)
            else:
                if current_tokens and current_type:
                    if "DEP" in current_type:
                        departure_tokens = current_tokens
                    elif "DEST" in current_type:
                        destination_tokens = current_tokens
                current_tokens = []
                current_type = None

        if current_tokens and current_type:
            if "DEP" in current_type:
                departure_tokens = current_tokens
            elif "DEST" in current_type:
                destination_tokens = current_tokens

        departure = " ".join(departure_tokens) if departure_tokens else None
        destination = " ".join(destination_tokens) if destination_tokens else None
        return {"departure": departure, "destination": destination}

    def _fuzzy_match(self, entity: str) -> str:
        """Fuzzy match an entity against the station database."""
        try:
            from rapidfuzz import fuzz, process
            result = process.extractOne(
                entity, self.stations, scorer=fuzz.ratio,
                score_cutoff=self.fuzzy_threshold,
            )
            if result:
                return result[0]
        except ImportError:
            entity_lower = entity.lower()
            for station in self.stations:
                if station.lower() == entity_lower:
                    return station
        return entity

    def extract(self, text: str) -> Dict[str, Any]:
        """
        Extract travel entities from input text.

        Returns:
            Dictionary with keys: departure, destination, intermediate
        """
        words, labels = self._predict_ner(text)
        entities = self._extract_entities(words, labels)

        departure = entities["departure"]
        destination = entities["destination"]

        if departure:
            departure = self._fuzzy_match(departure)
        if destination:
            destination = self._fuzzy_match(destination)

        return {
            "departure": departure,
            "destination": destination,
            "intermediate": [],
        }

    def extract_batch(
        self,
        texts: List[str],
        batch_size: int = 128,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> List[Dict[str, Any]]:
        """Extract entities from multiple texts."""
        results: List[Dict[str, Any]] = []
        total = len(texts)

        for i, text in enumerate(texts):
            results.append(self.extract(text))
            if progress_callback and (i + 1) % batch_size == 0:
                progress_callback(i + 1, total)

        if progress_callback:
            progress_callback(total, total)

        return results
