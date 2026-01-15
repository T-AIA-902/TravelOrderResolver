"""
Travel Order Resolver - Main orchestrator and CLI.

This module provides the TravelOrderResolver class that orchestrates
entity extraction, pathfinding, and visualization for travel requests.

Supports multiple extraction backends:
- camembert: Fine-tuned CamemBERT NER (best accuracy)
- spacy: spaCy fr_core_news_lg NER
- fuzzy: SpaCy + RapidFuzz station matching
- regex: Rule-based baseline

Usage:
    python -m src.main --extractor camembert
    python -m src.main --extractor spacy
"""

import argparse
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type

from src.pathfinding.graph import TrainGraph


@dataclass
class TravelResult:
    """Result of travel order resolution."""

    departure: Optional[str]
    destination: Optional[str]
    intermediate: List[str]
    path: Optional[List[str]]
    error: Optional[str]


class TravelOrderResolver:
    """
    Main orchestrator for travel order resolution.

    Combines entity extraction with pathfinding to resolve
    natural language travel requests into concrete routes.
    """

    # Available extraction backends
    EXTRACTORS: Dict[str, str] = {
        "camembert": "src.nlp.entity_extractor.CamembertEntityExtractor",
        "spacy": "src.nlp.entity_extractor.SpacyEntityExtractor",
        "fuzzy": "src.nlp.entity_extractor.FuzzyEntityExtractor",
        "regex": "src.nlp.models.baseline_regex.BaselineRegexModel",
    }

    # Instance attributes
    extractor_name: str
    extractor: Any
    graph: Optional[TrainGraph]
    visualizer: Optional[Any]

    def __init__(
        self,
        extractor: str = "camembert",
        load_graph: bool = True,
    ) -> None:
        """
        Initialize the travel order resolver.

        Args:
            extractor: Name of extraction backend to use
            load_graph: Whether to load the railway graph (set False for testing)
        """
        if extractor not in self.EXTRACTORS:
            raise ValueError(
                f"Unknown extractor: {extractor}. " f"Available: {list(self.EXTRACTORS.keys())}"
            )

        self.extractor_name = extractor
        self.extractor = self._load_extractor(extractor)

        if load_graph:
            self.graph = TrainGraph()
            self._load_visualizer()
        else:
            self.graph = None
            self.visualizer = None

    def _load_extractor(self, name: str) -> Any:
        """Dynamically load an extractor by name."""
        module_path = self.EXTRACTORS[name]
        module_name, class_name = module_path.rsplit(".", 1)

        try:
            module = __import__(module_name, fromlist=[class_name])
            extractor_class: Type[Any] = getattr(module, class_name)
            return extractor_class()
        except ImportError as e:
            raise ImportError(
                f"Failed to load extractor '{name}': {e}. " f"Make sure dependencies are installed."
            )

    def _load_visualizer(self) -> None:
        """Load the map visualizer if available."""
        if self.graph is None:
            self.visualizer = None
            return

        try:
            from src.visualization.map_visualizer import MapVisualizer

            self.visualizer = MapVisualizer(self.graph)
        except ImportError:
            self.visualizer = None
            print(
                "Warning: Visualization module not available.",
                file=sys.stderr,
            )

    def resolve(self, text: str) -> TravelResult:
        """
        Resolve a natural language travel request.

        Args:
            text: Travel request (e.g., "Je veux aller de Paris à Lyon")

        Returns:
            TravelResult with extracted entities and computed path
        """
        # Extract entities
        entities = self.extractor.extract_entities(text)

        departure = entities.get("departure")
        destination = entities.get("destination")
        intermediate = entities.get("intermediate", [])

        # Ensure intermediate is a list
        if not isinstance(intermediate, list):
            intermediate = []

        # If no graph loaded, return just the entities
        if self.graph is None:
            return TravelResult(
                departure=departure,
                destination=destination,
                intermediate=intermediate,
                path=None,
                error="Graph not loaded",
            )

        # Check if we have enough information
        if not departure or not destination:
            return TravelResult(
                departure=departure,
                destination=destination,
                intermediate=intermediate,
                path=None,
                error="Missing departure or destination",
            )

        # Find path
        path, error, uics = self.graph.get_path(departure, destination)

        return TravelResult(
            departure=departure,
            destination=destination,
            intermediate=intermediate,
            path=path,
            error=error,
        )

    def resolve_and_format(self, sentence_id: str, text: str) -> str:
        """
        Resolve a request and format as CSV output.

        Args:
            sentence_id: Identifier for the request
            text: Travel request text

        Returns:
            Formatted string: "id, station1, station2, ..." or "id, INVALID"
        """
        result = self.resolve(text)

        if result.error or not result.path:
            return f"{sentence_id}, INVALID"

        route_str = ", ".join(result.path)
        return f"{sentence_id}, {route_str}"

    def visualize(
        self,
        text: str,
        filename: str = "trajet.html",
    ) -> Optional[str]:
        """
        Resolve a request and generate a map visualization.

        Args:
            text: Travel request text
            filename: Output HTML file name

        Returns:
            Path to generated HTML file, or None if visualization failed
        """
        if self.visualizer is None:
            print("Visualization not available.", file=sys.stderr)
            return None

        result = self.resolve(text)

        if result.error or not result.path:
            print(f"Cannot visualize: {result.error}", file=sys.stderr)
            return None

        if self.graph is None or not result.departure or not result.destination:
            return None

        # Get full UIC path for visualization
        _, _, uics = self.graph.get_path(result.departure, result.destination)

        if uics:
            try:
                self.visualizer.draw_path(uics, filename=filename)
                return filename
            except Exception as e:
                print(f"Visualization error: {e}", file=sys.stderr)
                return None

        return None


def interactive_cli(resolver: TravelOrderResolver) -> None:
    """Run interactive command-line interface."""
    print("\n" + "=" * 60)
    print(f"Travel Order Resolver (extractor: {resolver.extractor_name})")
    print("=" * 60)
    print("\nCommands:")
    print("  Enter a travel request (e.g., 'De Paris à Lyon')")
    print("  Format: 'ID, text' or just 'text' (auto-assigns ID)")
    print("  Ctrl+C to quit\n")

    # Handle stdin encoding for French characters
    import io

    stdin = io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8", errors="replace")

    for line in stdin:
        line = line.strip()
        if not line:
            continue

        try:
            # Parse input
            if "," in line:
                parts = line.split(",", 1)
                sentence_id = parts[0].strip()
                text = parts[1].strip()
            else:
                sentence_id = "AUTO"
                text = line

            # Resolve and print
            output = resolver.resolve_and_format(sentence_id, text)
            print(output)
            sys.stdout.flush()

            # Generate HTML map if path found
            if "INVALID" not in output:
                html_file = resolver.visualize(text, filename=f"trajet_{sentence_id}.html")
                if html_file:
                    print(f"   Map saved: {html_file}", file=sys.stderr)

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)


def main() -> None:
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Travel Order Resolver - Extract travel entities and find routes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.main --extractor camembert
  python -m src.main --extractor spacy
  python -m src.main --extractor fuzzy
  python -m src.main --extractor regex

Available extractors:
  camembert  Fine-tuned CamemBERT NER (best accuracy)
  spacy      spaCy fr_core_news_lg NER
  fuzzy      SpaCy + RapidFuzz station matching
  regex      Rule-based baseline
        """,
    )
    parser.add_argument(
        "--extractor",
        "-e",
        choices=list(TravelOrderResolver.EXTRACTORS.keys()),
        default="camembert",
        help="Entity extraction backend (default: camembert)",
    )
    parser.add_argument(
        "--no-graph",
        action="store_true",
        help="Don't load the railway graph (extraction only)",
    )

    args = parser.parse_args()

    try:
        resolver = TravelOrderResolver(
            extractor=args.extractor,
            load_graph=not args.no_graph,
        )
        interactive_cli(resolver)
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
