"""
Fuzzy matching post-processor.

Normalizes extracted station names to official SNCF station names
using fuzzy string matching.

Performance optimizations:
- LRU cache in underlying StationMatcher (repeated city names are instant)
- Pre-built search indices (computed once at init)
"""

from typing import Any, Dict, List, Optional

from ..interfaces import PostProcessor


class FuzzyPostProcessor(PostProcessor):
    """
    Post-processor that applies fuzzy matching to entity extraction results.

    Uses RapidFuzz to match extracted location names (which may have typos,
    case variations, or be incomplete) to official SNCF station names.

    This can be applied to ANY entity extractor's output.
    """

    def __init__(self, threshold: int = 80) -> None:
        """
        Initialize the fuzzy post-processor.

        Args:
            threshold: Minimum similarity score (0-100) to accept a match
        """
        # Import here to avoid circular imports and allow lazy loading
        from src.nlp.fuzzy_matcher import StationMatcher

        self.threshold = threshold
        self.matcher = StationMatcher(threshold=threshold)

    @property
    def name(self) -> str:
        return "Fuzzy"

    def _match_entity(self, entity: Optional[str]) -> Optional[str]:
        """
        Match a single entity to a station name.

        Args:
            entity: Extracted entity text (may be None)

        Returns:
            Matched station name or original entity if no match
        """
        if not entity:
            return None

        result = self.matcher.match_station(entity)
        if result:
            station_name, score = result
            return station_name

        # Return original if no match found
        return entity

    def process(self, entities: Dict[str, Any], text: str) -> Dict[str, Any]:
        """
        Post-process extracted entities using fuzzy matching.

        Applies fuzzy station matching to departure, destination,
        and all intermediate stops.

        Args:
            entities: Dictionary with departure, destination, intermediate
            text: Original input text (not used, but required by interface)

        Returns:
            Dictionary with same structure, but station names normalized
        """
        result = entities.copy()

        # Match departure
        if entities.get("departure"):
            result["departure"] = self._match_entity(entities["departure"])

        # Match destination
        if entities.get("destination"):
            result["destination"] = self._match_entity(entities["destination"])

        # Match intermediates
        if entities.get("intermediate"):
            intermediates: List[str] = entities["intermediate"]
            result["intermediate"] = [self._match_entity(inter) or inter for inter in intermediates]

        return result

    def process_batch(
        self, entities_list: List[Dict[str, Any]], texts: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Post-process multiple entity extraction results in batch.

        Uses caching for efficiency - repeated city names across the batch
        are matched instantly from cache.

        Args:
            entities_list: List of entity dictionaries
            texts: List of original texts (not used, but required for interface)

        Returns:
            List of processed entity dictionaries
        """
        return [self.process(entities, text) for entities, text in zip(entities_list, texts)]

    def get_cache_stats(self) -> Dict[str, int]:
        """Return cache statistics from underlying matcher."""
        return self.matcher.get_cache_stats()
