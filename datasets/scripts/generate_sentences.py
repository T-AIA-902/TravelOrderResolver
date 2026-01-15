#!/usr/bin/env python3
"""
NLP Dataset Generator for Travel Order Resolver.

This script generates a dataset of French sentences for training
NLP models to extract travel intentions and locations.

Dataset schema:
- sentence_id: Unique identifier
- sentence: Input text
- intent: TRIP | NOT_TRIP | NOT_FRENCH | UNKNOWN
- departure: Departure station (or empty)
- destination: Destination station (or empty)
- intermediate: Comma-separated intermediate stations (or empty)

Usage:
    python generate_sentences.py [--output DIR] [--count N] [--seed S]
"""

import argparse
import csv
import json
import random
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from data import StationDatabase  # noqa: E402

# =============================================================================
# DATASET SCHEMA
# =============================================================================


@dataclass
class DatasetEntry:
    """Single entry in the NLP dataset."""

    sentence_id: str
    sentence: str
    intent: str  # TRIP, NOT_TRIP, NOT_FRENCH, UNKNOWN
    departure: str = ""
    destination: str = ""
    intermediate: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary for CSV/JSON export."""
        return {
            "sentence_id": self.sentence_id,
            "sentence": self.sentence,
            "intent": self.intent,
            "departure": self.departure,
            "destination": self.destination,
            "intermediate": self.intermediate,
        }


# =============================================================================
# SENTENCE TEMPLATES
# =============================================================================

# Templates for TRIP intent - {dep} = departure, {dest} = destination
TRIP_TEMPLATES = [
    # Basic patterns
    "Je voudrais aller de {dep} a {dest}",
    "Je veux aller de {dep} a {dest}",
    "Je souhaite aller de {dep} a {dest}",
    "Je dois aller de {dep} a {dest}",
    "Je vais de {dep} a {dest}",
    "Aller de {dep} a {dest}",
    "De {dep} a {dest}",
    "{dep} {dest}",
    "{dep} - {dest}",
    "{dep} vers {dest}",
    "{dep} direction {dest}",
    # Prendre le train
    "Je voudrais prendre un train de {dep} a {dest}",
    "Je veux prendre le train de {dep} a {dest}",
    "Prendre le train de {dep} a {dest}",
    "Un train de {dep} a {dest}",
    "Train de {dep} a {dest}",
    "Un billet de {dep} a {dest}",
    "Billet {dep} {dest}",
    # Partir / Arriver
    "Je pars de {dep} pour aller a {dest}",
    "Je pars de {dep} et je vais a {dest}",
    "Depart de {dep} arrivee a {dest}",
    "Depart {dep} arrivee {dest}",
    "Partir de {dep} pour {dest}",
    "Partir de {dep} vers {dest}",
    # Me rendre
    "Je voudrais me rendre a {dest} depuis {dep}",
    "Je veux me rendre a {dest} en partant de {dep}",
    "Me rendre de {dep} a {dest}",
    "Je me rends a {dest} depuis {dep}",
    # Voyager
    "Je voyage de {dep} a {dest}",
    "Voyage de {dep} a {dest}",
    "Voyager de {dep} vers {dest}",
    # Questions
    "Comment aller de {dep} a {dest}",
    "Comment aller de {dep} a {dest} ?",
    "Quel train pour aller de {dep} a {dest}",
    "Quel train de {dep} a {dest} ?",
    "Y a-t-il un train de {dep} a {dest}",
    "Y a-t-il un train de {dep} a {dest} ?",
    "Est-ce qu'il y a un train de {dep} a {dest}",
    # Informal
    "Je vais a {dest} depuis {dep}",
    "{dep} puis {dest}",
    "De {dep} jusqu'a {dest}",
    "Depuis {dep} jusqu'a {dest}",
    "Direction {dest} au depart de {dep}",
    # With time context (time will be ignored but sentences should still work)
    "Je dois aller de {dep} a {dest} demain",
    "Un train de {dep} a {dest} pour demain",
    "Demain je pars de {dep} pour {dest}",
    "Ce soir je vais de {dep} a {dest}",
    # Polite forms
    "Bonjour, je voudrais aller de {dep} a {dest}",
    "S'il vous plait, un trajet de {dep} a {dest}",
    "Pourriez-vous me trouver un train de {dep} a {dest}",
    "J'aimerais aller de {dep} a {dest}",
    "J'aimerais un billet de {dep} a {dest}",
]

# Templates with intermediate stops - {via} = intermediate station
TRIP_INTERMEDIATE_TEMPLATES = [
    "Je voudrais aller de {dep} a {dest} en passant par {via}",
    "De {dep} a {dest} via {via}",
    "De {dep} a {dest} en passant par {via}",
    "{dep} {dest} via {via}",
    "Aller de {dep} a {dest} avec un arret a {via}",
    "Je veux passer par {via} pour aller de {dep} a {dest}",
    "Train de {dep} a {dest} passant par {via}",
    "{dep} puis {via} puis {dest}",
    "De {dep} vers {dest} avec correspondance a {via}",
    "Je pars de {dep}, je passe par {via} et j'arrive a {dest}",
]

# Templates for NOT_TRIP intent (not a travel request)
NOT_TRIP_TEMPLATES = [
    # Greetings
    "Bonjour",
    "Salut",
    "Bonsoir",
    "Coucou",
    "Hello",
    "Hey",
    # Questions about the service
    "Comment ca marche",
    "Qu'est-ce que tu peux faire",
    "Aide moi",
    "Help",
    "C'est quoi ce service",
    "Tu fais quoi",
    # General statements
    "Il fait beau aujourd'hui",
    "J'aime les trains",
    "Les trains sont souvent en retard",
    "La SNCF c'est pas mal",
    "Je n'aime pas voyager",
    "C'est trop cher",
    "Merci",
    "Au revoir",
    "A bientot",
    "Ok",
    "D'accord",
    "Super",
    "Genial",
    # Incomplete or unclear
    "Je voudrais",
    "Je veux aller",
    "Un train",
    "Voyage",
    "Partir",
    "Arriver",
    "Demain",
    "Aujourd'hui",
    # Names that could be confused with cities
    "Je m'appelle Paris",
    "Mon ami Lyon va bien",
    "J'ai rencontre Marseille hier",
    "Albert est mon cousin",
    "Nice de te rencontrer",
    "Nancy est ma soeur",
    "Troyes est un prenom original",
]

# Templates for NOT_FRENCH intent (other languages)
NOT_FRENCH_TEMPLATES = [
    # English
    "I want to go from Paris to Lyon",
    "I would like a train ticket",
    "How do I get to Marseille",
    "Train from Paris to Bordeaux please",
    "What is the best route",
    "Can you help me",
    "I need to travel tomorrow",
    "Book a ticket for me",
    # German
    "Ich mochte von Paris nach Lyon fahren",
    "Ein Zug nach Marseille bitte",
    "Wie komme ich nach Bordeaux",
    # Spanish
    "Quiero ir de Paris a Lyon",
    "Un billete de tren por favor",
    "Como llego a Marsella",
    # Italian
    "Vorrei andare da Parigi a Lione",
    "Un biglietto per Marsiglia",
    # Mixed
    "I want aller a Paris",
    "Je want to go Lyon",
]

# Templates for UNKNOWN intent (gibberish, errors)
UNKNOWN_TEMPLATES = [
    "asdfghjkl",
    "qwerty",
    "123456",
    "???",
    "...",
    "!!!",
    "@#$%",
    "aaaaaaaa",
    "test test test",
    "blablabla",
    "lorem ipsum",
    "",
    "   ",
    "a",
    "ab",
]


# =============================================================================
# TEXT AUGMENTATION FUNCTIONS
# =============================================================================


def remove_accents(text: str) -> str:
    """Remove accents from text."""
    nfkd = unicodedata.normalize("NFD", text)
    return "".join(c for c in nfkd if unicodedata.category(c) != "Mn")


def to_lowercase(text: str) -> str:
    """Convert to lowercase."""
    return text.lower()


def to_uppercase(text: str) -> str:
    """Convert to uppercase."""
    return text.upper()


def random_case(text: str) -> str:
    """Randomly change case of some characters."""
    return "".join(c.upper() if random.random() > 0.7 else c.lower() for c in text)


def add_typos(text: str, prob: float = 0.1) -> str:
    """Add random typos to text."""
    if not text:
        return text

    typo_funcs = [
        lambda c: "",  # Delete character
        lambda c: c + c,  # Double character
        lambda c: c + random.choice("aeiou"),  # Add vowel
    ]

    # Common keyboard neighbors for typos
    neighbors = {
        "a": "zqs",
        "z": "aes",
        "e": "zrd",
        "r": "etf",
        "t": "ryg",
        "y": "tuh",
        "u": "yij",
        "i": "uok",
        "o": "ipl",
        "p": "om",
        "q": "aws",
        "s": "qdzx",
        "d": "sfec",
        "f": "dgvr",
        "g": "fhbt",
        "h": "gjny",
        "j": "hknu",
        "k": "jlmi",
        "l": "kmo",
        "m": "lkn",
    }

    result = []
    for c in text:
        if random.random() < prob and c.isalpha():
            lower_c = c.lower()
            if lower_c in neighbors and random.random() > 0.5:
                # Swap with neighbor key
                new_c = random.choice(neighbors[lower_c])
                result.append(new_c if c.islower() else new_c.upper())
            else:
                # Apply random typo
                result.append(random.choice(typo_funcs)(c))
        else:
            result.append(c)

    return "".join(result)


def add_common_misspellings(text: str) -> str:
    """Replace words with common misspellings."""
    misspellings = {
        "voudrais": ["voudrai", "voudrais", "voudrait"],
        "aller": ["allez", "alle", "aler"],
        "prendre": ["prandre", "prendr"],
        "train": ["trin", "tran"],
        "billet": ["bilet", "billlet"],
        "depuis": ["depui", "depuit"],
        "arrivee": ["arivee", "arrivé", "arrive"],
        "depart": ["depar", "départ"],
        "voyage": ["voyaje", "voiage"],
        "gare": ["gar", "guare"],
    }

    words = text.split()
    result = []
    for word in words:
        lower_word = word.lower()
        if lower_word in misspellings and random.random() > 0.5:
            result.append(random.choice(misspellings[lower_word]))
        else:
            result.append(word)

    return " ".join(result)


# =============================================================================
# DATASET GENERATOR
# =============================================================================


class DatasetGenerator:
    """Generator for NLP training datasets."""

    def __init__(self, station_db: StationDatabase, seed: int | None = None):
        """
        Initialize the generator.

        Args:
            station_db: StationDatabase instance with loaded stations.
            seed: Random seed for reproducibility.
        """
        self.station_db = station_db
        self.stations = station_db.get_all_stations(passenger_only=True)
        self.station_names = [s.name for s in self.stations]

        # Filter to major stations for more realistic dataset
        self.major_stations = [
            s.name
            for s in self.stations
            if s.short_code  # Has a short code = more important station
        ]

        if seed is not None:
            random.seed(seed)

        self.entry_count = 0

    def _get_id(self) -> str:
        """Generate unique sentence ID."""
        self.entry_count += 1
        return f"S{self.entry_count:06d}"

    def _random_station(self, exclude: list[str] | None = None) -> str:
        """Get a random station name."""
        pool = self.major_stations if random.random() > 0.3 else self.station_names
        if exclude:
            pool = [s for s in pool if s not in exclude]
        return random.choice(pool) if pool else random.choice(self.station_names)

    def generate_trip_sentence(
        self,
        with_intermediate: bool = False,
        augmentations: list[Callable[[str], str]] | None = None,
    ) -> DatasetEntry:
        """Generate a TRIP intent sentence."""
        dep = self._random_station()
        dest = self._random_station(exclude=[dep])

        if with_intermediate:
            via = self._random_station(exclude=[dep, dest])
            template = random.choice(TRIP_INTERMEDIATE_TEMPLATES)
            sentence = template.format(dep=dep, dest=dest, via=via)
            intermediate = via
        else:
            template = random.choice(TRIP_TEMPLATES)
            sentence = template.format(dep=dep, dest=dest)
            intermediate = ""

        # Apply augmentations
        if augmentations:
            for aug in augmentations:
                sentence = aug(sentence)

        return DatasetEntry(
            sentence_id=self._get_id(),
            sentence=sentence,
            intent="TRIP",
            departure=dep,
            destination=dest,
            intermediate=intermediate,
        )

    def generate_not_trip_sentence(
        self,
        augmentations: list[Callable[[str], str]] | None = None,
    ) -> DatasetEntry:
        """Generate a NOT_TRIP intent sentence."""
        sentence = random.choice(NOT_TRIP_TEMPLATES)

        if augmentations:
            for aug in augmentations:
                sentence = aug(sentence)

        return DatasetEntry(
            sentence_id=self._get_id(),
            sentence=sentence,
            intent="NOT_TRIP",
        )

    def generate_not_french_sentence(self) -> DatasetEntry:
        """Generate a NOT_FRENCH intent sentence."""
        sentence = random.choice(NOT_FRENCH_TEMPLATES)

        return DatasetEntry(
            sentence_id=self._get_id(),
            sentence=sentence,
            intent="NOT_FRENCH",
        )

    def generate_unknown_sentence(self) -> DatasetEntry:
        """Generate an UNKNOWN intent sentence."""
        sentence = random.choice(UNKNOWN_TEMPLATES)

        return DatasetEntry(
            sentence_id=self._get_id(),
            sentence=sentence,
            intent="UNKNOWN",
        )

    def generate_dataset(
        self,
        trip_count: int = 7000,
        not_trip_count: int = 1500,
        not_french_count: int = 500,
        unknown_count: int = 500,
        intermediate_ratio: float = 0.15,
    ) -> list[DatasetEntry]:
        """
        Generate a complete dataset.

        Args:
            trip_count: Number of TRIP sentences.
            not_trip_count: Number of NOT_TRIP sentences.
            not_french_count: Number of NOT_FRENCH sentences.
            unknown_count: Number of UNKNOWN sentences.
            intermediate_ratio: Ratio of TRIP sentences with intermediate stops.

        Returns:
            List of DatasetEntry objects.
        """
        entries: list[DatasetEntry] = []

        # Define augmentation combinations
        aug_none: list[Callable[[str], str]] = []
        aug_lower: list[Callable[[str], str]] = [to_lowercase]
        aug_no_accent: list[Callable[[str], str]] = [remove_accents]
        aug_lower_no_accent: list[Callable[[str], str]] = [to_lowercase, remove_accents]
        aug_typos: list[Callable[[str], str]] = [lambda t: add_typos(t, 0.05)]
        aug_misspell: list[Callable[[str], str]] = [add_common_misspellings]

        augmentation_options = [
            (0.40, aug_none),  # 40% normal
            (0.20, aug_lower),  # 20% lowercase
            (0.15, aug_no_accent),  # 15% no accents
            (0.10, aug_lower_no_accent),  # 10% lowercase + no accents
            (0.10, aug_typos),  # 10% with typos
            (0.05, aug_misspell),  # 5% with misspellings
        ]

        def pick_augmentation() -> list[Callable[[str], str]]:
            r = random.random()
            cumulative = 0.0
            for prob, aug in augmentation_options:
                cumulative += prob
                if r < cumulative:
                    return aug
            return aug_none

        # Generate TRIP sentences
        intermediate_count = int(trip_count * intermediate_ratio)
        normal_trip_count = trip_count - intermediate_count

        print(f"Generating {normal_trip_count} TRIP sentences...")
        for _ in range(normal_trip_count):
            entries.append(
                self.generate_trip_sentence(
                    with_intermediate=False,
                    augmentations=pick_augmentation(),
                )
            )

        print(f"Generating {intermediate_count} TRIP sentences with intermediate...")
        for _ in range(intermediate_count):
            entries.append(
                self.generate_trip_sentence(
                    with_intermediate=True,
                    augmentations=pick_augmentation(),
                )
            )

        # Generate NOT_TRIP sentences
        print(f"Generating {not_trip_count} NOT_TRIP sentences...")
        for _ in range(not_trip_count):
            entries.append(
                self.generate_not_trip_sentence(
                    augmentations=pick_augmentation() if random.random() > 0.3 else None,
                )
            )

        # Generate NOT_FRENCH sentences
        print(f"Generating {not_french_count} NOT_FRENCH sentences...")
        for _ in range(not_french_count):
            entries.append(self.generate_not_french_sentence())

        # Generate UNKNOWN sentences
        print(f"Generating {unknown_count} UNKNOWN sentences...")
        for _ in range(unknown_count):
            entries.append(self.generate_unknown_sentence())

        # Shuffle the dataset
        random.shuffle(entries)

        return entries


# =============================================================================
# DATASET EXPORT
# =============================================================================


def split_dataset(
    entries: list[DatasetEntry],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> tuple[list[DatasetEntry], list[DatasetEntry], list[DatasetEntry]]:
    """
    Split dataset into train/val/test sets.

    Args:
        entries: List of dataset entries.
        train_ratio: Ratio for training set.
        val_ratio: Ratio for validation set.

    Returns:
        Tuple of (train, val, test) lists.
    """
    n = len(entries)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    return (
        entries[:train_end],
        entries[train_end:val_end],
        entries[val_end:],
    )


def export_to_csv(entries: list[DatasetEntry], filepath: Path) -> None:
    """Export dataset to CSV file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sentence_id",
                "sentence",
                "intent",
                "departure",
                "destination",
                "intermediate",
            ],
        )
        writer.writeheader()
        for entry in entries:
            writer.writerow(entry.to_dict())


def export_to_json(entries: list[DatasetEntry], filepath: Path) -> None:
    """Export dataset to JSON file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump([e.to_dict() for e in entries], f, ensure_ascii=False, indent=2)


def print_stats(entries: list[DatasetEntry], name: str = "Dataset") -> None:
    """Print dataset statistics."""
    intent_counts = {}
    for entry in entries:
        intent_counts[entry.intent] = intent_counts.get(entry.intent, 0) + 1

    intermediate_count = sum(1 for e in entries if e.intermediate)

    print(f"\n{name} Statistics:")
    print(f"  Total entries: {len(entries)}")
    for intent, count in sorted(intent_counts.items()):
        pct = 100 * count / len(entries)
        print(f"  {intent}: {count} ({pct:.1f}%)")
    print(f"  With intermediate: {intermediate_count}")


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate NLP training dataset for travel order resolution."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: datasets/generated/)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=10000,
        help="Approximate total number of sentences (default: 10000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--format",
        choices=["csv", "json", "both"],
        default="both",
        help="Output format (default: both)",
    )

    args = parser.parse_args()

    # Setup output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path(__file__).parent.parent / "generated"

    # Load station database
    print("Loading station database...")
    station_db = StationDatabase()
    station_db.load()
    stats = station_db.stats()
    print(f"Loaded {stats['passenger_stations']} passenger stations.")

    # Calculate distribution
    total = args.count
    trip_count = int(total * 0.70)
    not_trip_count = int(total * 0.15)
    not_french_count = int(total * 0.10)
    unknown_count = total - trip_count - not_trip_count - not_french_count

    # Generate dataset
    print(f"\nGenerating dataset with {total} entries...")
    generator = DatasetGenerator(station_db, seed=args.seed)
    entries = generator.generate_dataset(
        trip_count=trip_count,
        not_trip_count=not_trip_count,
        not_french_count=not_french_count,
        unknown_count=unknown_count,
    )

    # Split into train/val/test
    train, val, test = split_dataset(entries)

    print_stats(entries, "Full Dataset")
    print_stats(train, "Train Set")
    print_stats(val, "Validation Set")
    print_stats(test, "Test Set")

    # Export
    print(f"\nExporting to {output_dir}...")

    if args.format in ("csv", "both"):
        export_to_csv(train, output_dir / "train.csv")
        export_to_csv(val, output_dir / "val.csv")
        export_to_csv(test, output_dir / "test.csv")
        export_to_csv(entries, output_dir / "full_dataset.csv")
        print("  CSV files exported.")

    if args.format in ("json", "both"):
        export_to_json(train, output_dir / "train.json")
        export_to_json(val, output_dir / "val.json")
        export_to_json(test, output_dir / "test.json")
        export_to_json(entries, output_dir / "full_dataset.json")
        print("  JSON files exported.")

    print("\nDone!")


if __name__ == "__main__":
    main()
