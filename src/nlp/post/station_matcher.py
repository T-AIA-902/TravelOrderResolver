"""
Fuzzy matching module for normalizing location names to SNCF stations.

This module uses RapidFuzz to match extracted location entities (potentially
with typos, case variations) to the official SNCF stations database.

Uses StationDatabase as the single source of truth for station data.

Performance optimizations:
- In-memory cache for repeated queries (same city name = same result)
- Pre-built first_words index (computed once at init, not per query)
"""

from typing import Dict, List, Optional, Tuple

from rapidfuzz import fuzz, process

from src.data.station_database import StationDatabase, normalize_name


class StationMatcher:
    """
    Fuzzy matcher for SNCF station names.

    Uses RapidFuzz to find the best matching station name from the database,
    handling typos, case variations, and accent differences.

    Uses StationDatabase as data source to avoid duplication.
    """

    def __init__(self, station_database: Optional[StationDatabase] = None, threshold: int = 80):
        """
        Initialize the station matcher with SNCF stations database.

        Args:
            station_database: StationDatabase instance. Creates a new one if None.
            threshold: Minimum similarity score (0-100) to accept a match (default: 80)
        """
        self.threshold = threshold

        # Use provided database or create a new one
        self.db = station_database or StationDatabase()
        self.db.load()

        # Build search index from StationDatabase
        # Map: normalized name -> original name
        self.search_index: Dict[str, str] = {}
        for station in self.db.get_all_stations(passenger_only=False):
            self.search_index[station.name_normalized] = station.name

        # List of normalized names for fuzzy search
        self.normalized_names = list(self.search_index.keys())

        # Pre-build first_words index (computed once, reused for all queries)
        # Maps first word -> list of full normalized names
        self.first_words: Dict[str, List[str]] = {}
        for normalized_name in self.normalized_names:
            first_word = normalized_name.split(" ")[0]
            if first_word not in self.first_words:
                self.first_words[first_word] = []
            self.first_words[first_word].append(normalized_name)
        self.first_word_keys = list(self.first_words.keys())

        # Initialize LRU cache for match_station
        self._match_cache: Dict[str, Optional[Tuple[str, float]]] = {}

        print(f"Loaded {len(self.search_index)} SNCF stations from StationDatabase")
        print(f"Fuzzy matching threshold: {self.threshold}%")

    def normalize_query(self, text: str) -> str:
        """
        Normalize a query string for matching.

        Uses the same normalize_name function as StationDatabase for consistency.

        Args:
            text: Input text (e.g., "Pari", "LYON", "Straßbourg")

        Returns:
            Normalized text (e.g., "pari", "lyon", "strassbourg")
        """
        return normalize_name(text)

    def match_station(self, query: str, top_n: int = 1) -> Optional[Tuple[str, float]]:
        """
        Find the best matching station for a query.

        Uses fuzzy string matching to handle typos and variations.
        Special handling: if query is a simple city name (e.g., "Paris"),
        it will match any station starting with that city (e.g., "Paris-Gare-de-Lyon").

        Results are cached for performance (same query = same result).

        Args:
            query: Location name to match (e.g., "pari", "Lion", "Strasbourg")
            top_n: Number of top matches to return (default: 1)

        Returns:
            Tuple of (matched_station_name, confidence_score) or None if no match
            above threshold.

        Example:
            >>> matcher = StationMatcher(threshold=75)
            >>> result = matcher.match_station("pari")
            >>> print(result)
            ('Paris-Gare-de-Lyon', 90.0)
        """
        if not query:
            return None

        # Normalize the query
        normalized_query = self.normalize_query(query)

        # Check cache first (huge speedup for repeated queries)
        if normalized_query in self._match_cache:
            return self._match_cache[normalized_query]

        # Step 1: Check for exact prefix match (e.g., "paris" -> "paris gare de lyon")
        # This handles cases where user says "Paris" but stations are "Paris-Est", etc.
        # Note: normalize_name() converts hyphens to spaces
        for normalized_name in self.normalized_names:
            if normalized_name.startswith(normalized_query + " "):
                # Found a station that starts with query
                # Return with high confidence (95%) since it's a prefix match
                original_name = self.search_index[normalized_name]
                result = (original_name, 95.0)
                self._match_cache[normalized_query] = result
                return result

        # Step 2: Use pre-built first_words index (no longer rebuilt every call)
        # Step 3: Try fuzzy matching on first words only
        first_word_matches = process.extract(
            normalized_query,
            self.first_word_keys,
            scorer=fuzz.ratio,
            limit=10,
        )

        # Step 4: Also do full fuzzy matching on complete station names
        full_matches = process.extract(
            normalized_query,
            self.normalized_names,
            scorer=fuzz.WRatio,
            limit=10,
        )

        # Step 5: Combine results with boosted scores for first-word matches
        all_matches: Dict[str, float] = {}

        # Add first-word matches with boost
        for first_word, score, _ in first_word_matches:
            # Pick the first station with this first word
            station_name = self.first_words[first_word][0]
            # Boost score significantly since we matched the primary city name
            # Higher boost ensures major cities are preferred over obscure stations
            boosted_score = min(100, score + 20)
            all_matches[station_name] = boosted_score

        # Add full matches
        for station_name, score, _ in full_matches:
            if station_name in all_matches:
                # Keep the higher score
                all_matches[station_name] = max(all_matches[station_name], score)
            else:
                all_matches[station_name] = score

        if not all_matches:
            self._match_cache[normalized_query] = None
            return None

        # Step 6: Find best match
        best_match = max(all_matches.items(), key=lambda x: x[1])
        best_station, best_score = best_match

        if best_score < self.threshold:
            self._match_cache[normalized_query] = None
            return None

        # Return the original station name (not normalized)
        original_name = self.search_index[best_station]
        result = (original_name, best_score)
        self._match_cache[normalized_query] = result
        return result

    def match_stations_batch(self, queries: List[str]) -> List[Optional[Tuple[str, float]]]:
        """
        Match multiple queries in batch.

        Uses caching for efficiency - repeated queries are instant.

        Args:
            queries: List of location names to match

        Returns:
            List of match results (same order as input)

        Example:
            >>> matcher = StationMatcher()
            >>> queries = ["pari", "lion", "Marseille"]
            >>> results = matcher.match_stations_batch(queries)
            >>> print(results)
            [('Paris', 90.0), ('Lyon', 90.0), ('Marseille', 100.0)]
        """
        return [self.match_station(query) for query in queries]

    def clear_cache(self) -> None:
        """Clear the match cache. Useful for testing or memory management."""
        self._match_cache.clear()

    def get_cache_stats(self) -> Dict[str, int]:
        """Return cache statistics for monitoring."""
        return {
            "cache_size": len(self._match_cache),
            "stations_count": len(self.search_index),
        }

    def match_with_details(self, query: str, top_n: int = 5) -> List[Tuple[str, float]]:
        """
        Get multiple match candidates with scores for debugging.

        Useful for understanding why a particular match was selected.

        Args:
            query: Location name to match
            top_n: Number of top matches to return (default: 5)

        Returns:
            List of (station_name, score) tuples, sorted by score descending

        Example:
            >>> matcher = StationMatcher()
            >>> results = matcher.match_with_details("pari", top_n=3)
            >>> for name, score in results:
            ...     print(f"{name}: {score:.1f}%")
            Paris: 90.0%
            Paray-le-Monial: 65.0%
            Parigny: 60.0%
        """
        if not query:
            return []

        normalized_query = self.normalize_query(query)

        matches = process.extract(
            normalized_query,
            self.normalized_names,
            scorer=fuzz.WRatio,
            limit=top_n,
        )

        # Convert to original names
        results = [(self.search_index[match[0]], match[1]) for match in matches]

        return results


def main():
    """Demo function to test the fuzzy matcher."""
    print("=" * 60)
    print("Fuzzy Station Matcher - Demo")
    print("=" * 60)

    # Create matcher
    matcher = StationMatcher(threshold=75)

    # Test queries with various issues
    test_queries = [
        "pari",  # Misspelling
        "Lion",  # Misspelling
        "MARSEILLE",  # All caps
        "paris",  # Lowercase
        "Strasbourg",  # Correct
        "Nante",  # Missing 's'
        "bordeaux saint-jean",  # Full station name
        "invalid_city_xyz",  # Should not match
    ]

    print("\nTesting fuzzy matching:\n")

    for query in test_queries:
        result = matcher.match_station(query)
        if result:
            station, score = result
            print(f"Query: {query:25} -> Match: {station:25} (Score: {score:.1f}%)")
        else:
            print(f"Query: {query:25} -> No match found")

    # Show detailed matches for "pari"
    print("\n" + "=" * 60)
    print("Detailed matches for 'pari' (top 5):")
    print("=" * 60)
    details = matcher.match_with_details("pari", top_n=5)
    for station, score in details:
        print(f"  {station:30} {score:.1f}%")


if __name__ == "__main__":
    main()
