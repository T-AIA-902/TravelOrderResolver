"""
CamemBERT-based entity extractor.

Uses exact string matching against station database.
This is the zero-shot baseline (no fine-tuning on SNCF data).
"""

from typing import Any, Callable, Dict, List, Literal, Set

from ..interfaces import EntityExtractor

DeviceType = Literal["auto", "cuda", "cpu"]


class CamembertEntityExtractor(EntityExtractor):
    """
    Zero-shot entity extraction using camembert-base.

    This extractor loads the CamemBERT model and precomputes embeddings
    for all station names, then uses exact string matching to find
    stations in input text.

    Note: Embedding similarity is disabled due to false positives.
    This provides a pure exact-match baseline for comparison.
    """

    def __init__(
        self,
        model_name: str = "almanach/camembert-base",
        threshold: float = 0.85,
        device: DeviceType = "auto",
    ) -> None:
        """
        Initialize the zero-shot CamemBERT extractor.

        Args:
            model_name: HuggingFace model name (default: almanach/camembert-base)
            threshold: Minimum cosine similarity for station matching (default: 0.85)
            device: Device to use - "auto", "cuda", or "cpu" (default: auto)
        """
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError:
            raise ImportError(
                "transformers/torch not available. " "Install with: pip install transformers torch"
            )

        from src.utils.device import get_torch_device

        self.device = get_torch_device(device)

        print(f"Loading CamemBERT base model: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model = self.model.to(self.device)  # Move model to device
        self.model.eval()  # Set to evaluation mode
        self.threshold = threshold
        self.torch = torch

        # Load station database and precompute embeddings
        from src.data import StationDatabase

        self.station_db = StationDatabase()
        self.station_db.load()
        self._precompute_station_embeddings()
        print(
            f"CamemBERT zero-shot extractor ready (device: {self.device}, threshold: {threshold})"
        )

    @property
    def name(self) -> str:
        return "CamemBERT"

    def _get_embedding(self, text: str) -> Any:
        """Get the mean pooled embedding for a text."""
        with self.torch.no_grad():
            inputs = self.tokenizer(
                text, return_tensors="pt", padding=True, truncation=True, max_length=32
            )
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            # Mean pooling over token embeddings (excluding special tokens)
            attention_mask = inputs["attention_mask"]
            embeddings = outputs.last_hidden_state
            mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size())
            sum_embeddings = (embeddings * mask_expanded).sum(1)
            sum_mask = mask_expanded.sum(1).clamp(min=1e-9)
            return sum_embeddings / sum_mask

    def _precompute_station_embeddings(self) -> None:
        """Precompute embeddings for all station names."""
        print("Precomputing station embeddings...")
        self.station_embeddings: Dict[str, Any] = {}
        self.city_names: List[str] = []

        # Get unique city names from station database
        cities: Set[str] = set()
        for station in self.station_db.get_all_stations():
            # Station is a dataclass with name and commune attributes
            city = station.commune if station.commune else station.name
            if city and len(city) >= 2:
                # Capitalize properly (commune is often uppercase)
                cities.add(city.title())

        self.city_names = sorted(cities)

        # Batch compute embeddings for efficiency
        batch_size = 64
        for i in range(0, len(self.city_names), batch_size):
            batch = self.city_names[i : i + batch_size]
            for city in batch:
                self.station_embeddings[city.lower()] = self._get_embedding(city)

        print(f"Computed embeddings for {len(self.station_embeddings)} cities")

    def _find_station_matches(self, text: str) -> List[Dict[str, Any]]:
        """
        Find potential station matches in text using exact string matching.

        Returns list of matches with position, text, and similarity score.
        """
        matches: List[Dict[str, Any]] = []
        text_lower = text.lower()

        # Stage 1: Check for exact city name matches in text
        # Sort cities by length (longest first) to match multi-word names first
        matched_positions: Set[int] = set()
        cities_by_length = sorted(self.station_embeddings.keys(), key=len, reverse=True)

        # Word boundary characters
        boundary_chars = set(" .,;:!?()[]{}\"'\t\n-")

        for city in cities_by_length:
            city_lower = city.lower()

            # Skip very short city names (< 3 chars) to avoid false positives
            if len(city_lower) < 3:
                continue

            # Search for the city name in the text
            pos = text_lower.find(city_lower)
            if pos != -1:
                end_pos = pos + len(city_lower)

                # Check word boundaries to avoid matching substrings
                is_word_start = pos == 0 or text_lower[pos - 1] in boundary_chars
                is_word_end = end_pos >= len(text_lower) or text_lower[end_pos] in boundary_chars

                if not is_word_start or not is_word_end:
                    continue

                # Check if this position overlaps with already matched positions
                overlap = any(p >= pos and p < end_pos for p in matched_positions)
                if not overlap:
                    # Mark these positions as matched
                    for p in range(pos, end_pos):
                        matched_positions.add(p)

                    matches.append(
                        {
                            "text": city_lower,
                            "matched_city": city.title(),
                            "score": 1.0,
                            "position": pos,
                            "start": pos,
                        }
                    )

        # Stage 2: For words not yet matched, skip embedding similarity
        # (disabled due to false positives - gives pure exact-match baseline)

        return matches

    def extract(self, text: str) -> Dict[str, Any]:
        """
        Extract travel entities from the input text.

        Uses exact string matching against the station database.

        Args:
            text: Input text to process

        Returns:
            Dictionary with keys:
            - departure: Optional[str] - Starting location
            - destination: Optional[str] - Ending location
            - intermediate: List[str] - Intermediate stops
        """
        result: Dict[str, Any] = {
            "departure": None,
            "destination": None,
            "intermediate": [],
        }

        # Find all station matches
        matches = self._find_station_matches(text)

        if not matches:
            return result

        # Sort by position in text
        matches.sort(key=lambda x: x["position"])

        # Apply heuristics: first = departure, last = destination
        if len(matches) == 1:
            result["destination"] = matches[0]["matched_city"]
        elif len(matches) >= 2:
            result["departure"] = matches[0]["matched_city"]
            result["destination"] = matches[-1]["matched_city"]

            # Middle matches are intermediates
            if len(matches) > 2:
                result["intermediate"] = [m["matched_city"] for m in matches[1:-1]]

        return result

    def extract_batch(
        self,
        texts: List[str],
        batch_size: int = 128,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> List[Dict[str, Any]]:
        """
        Extract entities from multiple texts.

        Args:
            texts: List of input texts to process
            batch_size: Batch size for progress reporting
            progress_callback: Optional callback(current, total) for progress updates

        Returns:
            List of entity dictionaries
        """
        results: List[Dict[str, Any]] = []
        total = len(texts)

        for i, text in enumerate(texts):
            results.append(self.extract(text))

            # Report progress at batch boundaries
            if progress_callback and (i + 1) % batch_size == 0:
                progress_callback(i + 1, total)

        # Final progress update
        if progress_callback:
            progress_callback(total, total)

        return results
