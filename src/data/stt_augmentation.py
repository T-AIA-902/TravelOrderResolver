"""
STT (Speech-to-Text) Augmentation Module.

Simulates realistic Whisper speech-to-text transcription errors
for generating synthetic training data.

Error types based on Whisper research:
- Filler words and hesitations
- False starts and restarts
- Word repetitions
- Incomplete sentences
- Phonetic confusions
- Proper noun (station) misspellings
- Punctuation errors
- Capitalization inconsistencies
- Number format variations
- Code-switching (language mixing)
- Hallucinations
- Noise artifacts
- Regional accent transcriptions
- Homophone confusions
"""

import random
import re
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class STTErrorConfig:
    """Configuration for STT error simulation probabilities."""

    filler_words: float = 0.15
    false_starts: float = 0.08
    repetitions: float = 0.05
    incomplete: float = 0.03
    phonetic_confusion: float = 0.10
    station_misspelling: float = 0.12
    punctuation_errors: float = 0.20
    capitalization_errors: float = 0.25
    number_format: float = 0.08
    code_switching: float = 0.02
    hallucinations: float = 0.015
    noise_artifacts: float = 0.05
    accent_variations: float = 0.06
    homophones: float = 0.08


# Predefined error intensity profiles
ERROR_PROFILES = {
    "clean": STTErrorConfig(
        filler_words=0,
        false_starts=0,
        repetitions=0,
        incomplete=0,
        phonetic_confusion=0,
        station_misspelling=0,
        punctuation_errors=0,
        capitalization_errors=0,
        number_format=0,
        code_switching=0,
        hallucinations=0,
        noise_artifacts=0,
        accent_variations=0,
        homophones=0,
    ),
    "light": STTErrorConfig(
        filler_words=0.10,
        false_starts=0.05,
        repetitions=0.03,
        incomplete=0.01,
        phonetic_confusion=0.05,
        station_misspelling=0.05,
        punctuation_errors=0.15,
        capitalization_errors=0.20,
        number_format=0.05,
        code_switching=0.01,
        hallucinations=0.005,
        noise_artifacts=0.02,
        accent_variations=0.03,
        homophones=0.05,
    ),
    "moderate": STTErrorConfig(
        filler_words=0.15,
        false_starts=0.08,
        repetitions=0.05,
        incomplete=0.03,
        phonetic_confusion=0.10,
        station_misspelling=0.12,
        punctuation_errors=0.20,
        capitalization_errors=0.25,
        number_format=0.08,
        code_switching=0.02,
        hallucinations=0.015,
        noise_artifacts=0.05,
        accent_variations=0.06,
        homophones=0.08,
    ),
    "heavy": STTErrorConfig(
        filler_words=0.25,
        false_starts=0.15,
        repetitions=0.10,
        incomplete=0.08,
        phonetic_confusion=0.20,
        station_misspelling=0.20,
        punctuation_errors=0.35,
        capitalization_errors=0.40,
        number_format=0.15,
        code_switching=0.05,
        hallucinations=0.03,
        noise_artifacts=0.10,
        accent_variations=0.12,
        homophones=0.15,
    ),
}

# =============================================================================
# FRENCH FILLER WORDS AND HESITATIONS
# =============================================================================

FRENCH_FILLERS = [
    "euh",
    "hum",
    "hmm",
    "ben",
    "bah",
    "alors",
    "enfin",
    "donc",
    "voila",
    "tu vois",
    "genre",
    "quoi",
    "en fait",
    "bon",
    "hein",
    "ouais",
]

# =============================================================================
# FALSE START PATTERNS
# =============================================================================

FALSE_START_PATTERNS = [
    ("Je veux", "Je veux... enfin je voudrais"),
    ("Je voudrais", "Je... je voudrais"),
    ("Je souhaite", "Je sou... je souhaite"),
    ("un train", "un... un train"),
    ("aller de", "aller... de"),
    ("aller a", "aller... a"),
    ("prendre", "pren... prendre"),
    ("partir de", "partir... de"),
    ("Un billet", "Un... un billet"),
    ("Bonjour", "Bon... bonjour"),
]

# =============================================================================
# PHONETIC CONFUSIONS (French)
# =============================================================================

FRENCH_PHONETIC_CONFUSIONS = {
    # Nasal sounds
    "an": ["en", "ant", "and"],
    "en": ["an", "ent", "ant"],
    "in": ["ain", "ein", "un", "aint"],
    "ain": ["in", "ein", "aint"],
    "on": ["ont", "ond"],
    # E sounds
    "e": ["ai", "ais", "et"],
    "ai": ["e", "ais", "et", "ait"],
    "ais": ["ai", "ait", "e"],
    "er": ["e", "ez", "ais"],
    "ez": ["e", "er", "ais"],
    # Common confusions
    "s": ["ss", "c"],
    "ss": ["s", "c"],
    "c": ["s", "ss", "k"],
    "qu": ["c", "k", "q"],
    "k": ["c", "qu"],
    "ph": ["f"],
    "f": ["ph"],
    # Silent letters issues
    "ent": ["e", "an"],
    "ait": ["ai", "ais", "e"],
}

# Common word phonetic errors
WORD_PHONETIC_ERRORS = {
    "train": ["trin", "tran", "trein", "traing"],
    "billet": ["bilet", "billait", "biyé"],
    "aller": ["allez", "aler", "haller", "ale"],
    "gare": ["gar", "guare", "garre"],
    "voyage": ["voyaje", "voiage", "voyag"],
    "voudrais": ["voudrai", "voudrait", "voudrè"],
    "partir": ["partire", "parti", "partiir"],
    "arriver": ["ariver", "arrivé", "arivé"],
    "prendre": ["prandre", "prendr", "prende"],
    "depuis": ["depui", "depuit", "depu"],
    "destination": ["destinacion", "destinassion"],
    "depart": ["depar", "dépar", "depard"],
}

# =============================================================================
# STATION NAME PHONETIC RULES
# =============================================================================

STATION_PHONETIC_RULES = {
    # Common prefixes
    "Saint": ["Sain", "Sin", "Cent", "St"],
    "Sainte": ["Sainte", "Sinte", "Ste"],
    # Major stations
    "Montparnasse": ["Monparnasse", "Montparnace", "Mon Parnasse", "Montparnass"],
    "Lyon": ["Lion", "Léon", "Lillon", "Lyons"],
    "Lille": ["Lil", "Lile", "Lilles", "Lill"],
    "Marseille": ["Marseye", "Marceille", "Marsei", "Marceil"],
    "Bordeaux": ["Bordo", "Bordau", "Bordeau", "Bordo"],
    "Toulouse": ["Toulouze", "Tolouse", "Toulous"],
    "Strasbourg": ["Strasbour", "Strassbourg", "Strasburg"],
    "Nantes": ["Nante", "Nant", "Nants"],
    "Nice": ["Nisse", "Nis", "Nices"],
    "Montpellier": ["Monpellier", "Montpelier", "Montpelié"],
    "Rennes": ["Renne", "Ren", "Renn"],
    "Grenoble": ["Grenobl", "Grennoble", "Grenobe"],
    "Dijon": ["Dijonn", "Dijons", "Digeon"],
    "Avignon": ["Avignion", "Avignonn", "Avignon TGV"],
    # Common suffixes
    "-sur-": [" sur ", "sur", "-sûr-", " sûr "],
    "-les-": [" les ", "les", "-lès-", " lès "],
    "-la-": [" la ", "la", " La "],
    "-le-": [" le ", "le", " Le "],
    # Airport
    "Aeroport": ["Aéroport", "Aeroport", "Airoport"],
    "Charles de Gaulle": ["Charles De Gaulle", "CDG", "Charle de Gaulle"],
}

# =============================================================================
# HOMOPHONES (French)
# =============================================================================

FRENCH_HOMOPHONES = {
    "a": "à",
    "à": "a",
    "ou": "où",
    "où": "ou",
    "vers": "vert",
    "vert": "vers",
    "ce": "se",
    "se": "ce",
    "ces": "ses",
    "ses": "ces",
    "c'est": "sait",
    "sait": "c'est",
    "et": "est",
    "est": "et",
    "son": "sont",
    "sont": "son",
    "ont": "on",
    "on": "ont",
    "mes": "mais",
    "mais": "mes",
    "la": "là",
    "là": "la",
    "du": "dû",
    "sur": "sûr",
    "sûr": "sur",
    "notre": "nôtre",
    "votre": "vôtre",
}

# =============================================================================
# CODE-SWITCHING REPLACEMENTS (French/English)
# =============================================================================

CODE_SWITCH_REPLACEMENTS = {
    "Je veux": ["I want", "Je want"],
    "Je voudrais": ["I would like", "Je would like"],
    "aller": ["to go", "go", "aller to"],
    "un train": ["a train", "un train"],
    "s'il vous plait": ["please", "s'il vous please", "please s'il vous plait"],
    "billet": ["ticket", "billette"],
    "de": ["from"],
    "pour": ["for", "to"],
    "vers": ["to", "towards"],
    "gare": ["station", "gare station"],
    "merci": ["thank you", "merci beaucoup thank you"],
}

# =============================================================================
# WHISPER HALLUCINATIONS
# =============================================================================

WHISPER_HALLUCINATIONS = [
    "Merci d'avoir regarde cette video.",
    "N'oubliez pas de vous abonner.",
    "Like and subscribe.",
    "Sous-titres par la communaute.",
    "Copyright 2024.",
    "Thanks for watching.",
    "[Musique]",
    "[Applaudissements]",
    "...",
    "Abonnez-vous a notre chaine.",
    "Rendez-vous sur notre site.",
    "Pour plus d'informations.",
]

# =============================================================================
# NOISE MARKERS
# =============================================================================

NOISE_MARKERS = [
    "[inaudible]",
    "[bruit]",
    "[musique]",
    "[annonce]",
    "...",
    "---",
    "[coupure]",
    "[static]",
    "*bruit*",
    "[incomprehensible]",
    "[???]",
    "[silence]",
]

# =============================================================================
# NUMBER MAPPINGS (French)
# =============================================================================

NUMBER_WORD_MAPPINGS = {
    "1": "un",
    "2": "deux",
    "3": "trois",
    "4": "quatre",
    "5": "cinq",
    "6": "six",
    "7": "sept",
    "8": "huit",
    "9": "neuf",
    "10": "dix",
    "11": "onze",
    "12": "douze",
    "13": "treize",
    "14": "quatorze",
    "15": "quinze",
    "16": "seize",
    "17": "dix-sept",
    "18": "dix-huit",
    "19": "dix-neuf",
    "20": "vingt",
}

# =============================================================================
# AZERTY KEYBOARD NEIGHBORS (for typos)
# =============================================================================

AZERTY_NEIGHBORS = {
    "a": "zqs",
    "z": "aesq",
    "e": "zrds",
    "r": "etfd",
    "t": "rygf",
    "y": "tuhg",
    "u": "yijh",
    "i": "uokj",
    "o": "iplk",
    "p": "olm",
    "q": "azws",
    "s": "qdzxwe",
    "d": "sfxcer",
    "f": "dgcvrt",
    "g": "fhvbty",
    "h": "gjbnuy",
    "j": "hknui",
    "k": "jlmio",
    "l": "kmop",
    "m": "lkp",
    "w": "qsx",
    "x": "wsdc",
    "c": "xdfv",
    "v": "cfgb",
    "b": "vghn",
    "n": "bhj",
}


@dataclass
class STTAugmenter:
    """
    Applies realistic STT errors to text.

    Simulates various types of speech-to-text transcription errors
    based on Whisper research and real-world STT behavior.
    """

    config: STTErrorConfig = field(default_factory=STTErrorConfig)
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.seed is not None:
            random.seed(self.seed)

    def augment(
        self,
        text: str,
        intensity: str | None = None,
        apply_to_station: str | None = None,
    ) -> str:
        """
        Apply STT errors to text.

        Args:
            text: Input text to augment.
            intensity: Error intensity profile ("clean", "light", "moderate", "heavy").
                      If None, uses instance config.
            apply_to_station: If provided, also corrupt this station name in the text.

        Returns:
            Augmented text with STT errors.
        """
        if not text or not text.strip():
            return text

        # Use intensity profile if specified
        config = ERROR_PROFILES.get(intensity, self.config) if intensity else self.config

        result = text

        # Apply station misspelling first if station name provided
        if apply_to_station and random.random() < config.station_misspelling:
            corrupted = self._corrupt_station_name(apply_to_station)
            if corrupted != apply_to_station:
                result = result.replace(apply_to_station, corrupted)

        # Apply various error types based on probabilities
        if random.random() < config.filler_words:
            result = self._add_filler_words(result)

        if random.random() < config.false_starts:
            result = self._add_false_starts(result)

        if random.random() < config.repetitions:
            result = self._add_repetitions(result)

        if random.random() < config.incomplete:
            result = self._make_incomplete(result)

        if random.random() < config.phonetic_confusion:
            result = self._apply_phonetic_confusion(result)

        if random.random() < config.homophones:
            result = self._apply_homophones(result)

        if random.random() < config.punctuation_errors:
            result = self._corrupt_punctuation(result)

        if random.random() < config.capitalization_errors:
            result = self._corrupt_capitalization(result)

        if random.random() < config.number_format:
            result = self._vary_number_format(result)

        if random.random() < config.code_switching:
            result = self._apply_code_switching(result)

        if random.random() < config.hallucinations:
            result = self._add_hallucination(result)

        if random.random() < config.noise_artifacts:
            result = self._add_noise_artifacts(result)

        if random.random() < config.accent_variations:
            result = self._apply_accent_variation(result)

        return result

    def augment_with_intensity(self, text: str, intensity: str) -> str:
        """Convenience method to augment with a specific intensity."""
        return self.augment(text, intensity=intensity)

    # =========================================================================
    # INDIVIDUAL ERROR FUNCTIONS
    # =========================================================================

    def _add_filler_words(self, text: str) -> str:
        """Insert oral French filler words randomly."""
        words = text.split()
        if len(words) < 2:
            return text

        num_fillers = random.randint(1, min(3, len(words) - 1))
        positions = random.sample(range(1, len(words)), k=num_fillers)

        for pos in sorted(positions, reverse=True):
            filler = random.choice(FRENCH_FILLERS)
            words.insert(pos, filler)

        return " ".join(words)

    def _add_false_starts(self, text: str) -> str:
        """Simulate speaker corrections and restarts."""
        for original, replacement in FALSE_START_PATTERNS:
            if original.lower() in text.lower():
                # Case-insensitive replacement
                pattern = re.compile(re.escape(original), re.IGNORECASE)
                if random.random() < 0.5:
                    return pattern.sub(replacement, text, count=1)
        return text

    def _add_repetitions(self, text: str) -> str:
        """Duplicate words to simulate stuttering/repetition."""
        words = text.split()
        if len(words) < 2:
            return text

        idx = random.randint(0, len(words) - 1)
        word = words[idx]
        # Don't repeat very short words or punctuation
        if len(word) > 1 and word.isalpha():
            words.insert(idx, word.lower())

        return " ".join(words)

    def _make_incomplete(self, text: str) -> str:
        """Truncate sentence to simulate cut-off audio."""
        words = text.split()
        if len(words) < 3:
            return text

        cut_point = random.randint(max(2, len(words) // 2), len(words) - 1)
        truncated = " ".join(words[:cut_point])

        # Optionally truncate the last word
        if random.random() < 0.3 and len(truncated) > 3:
            truncated = truncated[:-random.randint(1, 3)]

        suffix = random.choice(["...", "---", " [coupure]", ""])
        return truncated + suffix

    def _apply_phonetic_confusion(self, text: str) -> str:
        """Replace sounds with phonetically similar ones."""
        result = text

        # Apply word-level phonetic errors
        for word, replacements in WORD_PHONETIC_ERRORS.items():
            if word in result.lower():
                pattern = re.compile(re.escape(word), re.IGNORECASE)
                if random.random() < 0.4:
                    replacement = random.choice(replacements)
                    result = pattern.sub(replacement, result, count=1)

        # Apply character-level phonetic confusions
        for original, confusions in FRENCH_PHONETIC_CONFUSIONS.items():
            if original in result.lower() and random.random() < 0.2:
                pattern = re.compile(re.escape(original), re.IGNORECASE)
                result = pattern.sub(random.choice(confusions), result, count=1)

        return result

    def _corrupt_station_name(self, station_name: str) -> str:
        """Apply realistic transcription errors to station names."""
        result = station_name

        # Apply station-specific phonetic rules
        for pattern, replacements in STATION_PHONETIC_RULES.items():
            if pattern in result:
                if random.random() < 0.5:
                    result = result.replace(pattern, random.choice(replacements), 1)

        # Apply AZERTY keyboard typos
        if random.random() < 0.3:
            result = self._add_keyboard_typo(result)

        return result

    def _add_keyboard_typo(self, text: str) -> str:
        """Add typos based on AZERTY keyboard layout."""
        if not text:
            return text

        chars = list(text)
        # Choose a random position
        valid_positions = [
            i for i, c in enumerate(chars) if c.lower() in AZERTY_NEIGHBORS
        ]

        if not valid_positions:
            return text

        pos = random.choice(valid_positions)
        char = chars[pos]
        lower_char = char.lower()

        if lower_char in AZERTY_NEIGHBORS:
            neighbors = AZERTY_NEIGHBORS[lower_char]
            new_char = random.choice(neighbors)
            chars[pos] = new_char.upper() if char.isupper() else new_char

        return "".join(chars)

    def _corrupt_punctuation(self, text: str) -> str:
        """Add/remove/change punctuation."""
        option = random.choice(["remove", "add_comma", "wrong_end", "none"])

        if option == "remove":
            # Remove all punctuation
            return re.sub(r"[.,!?;:]", "", text)

        elif option == "add_comma":
            # Add random comma
            words = text.split()
            if len(words) > 2:
                pos = random.randint(1, len(words) - 1)
                words[pos - 1] += ","
                return " ".join(words)

        elif option == "wrong_end":
            # Wrong ending punctuation
            text = text.rstrip(".,!?;:")
            return text + random.choice(["", ".", ",", "..."])

        return text

    def _corrupt_capitalization(self, text: str) -> str:
        """Simulate Whisper capitalization inconsistencies."""
        option = random.choice(["lower", "random", "first_only", "none"])

        if option == "lower":
            return text.lower()

        elif option == "random":
            return "".join(
                c.upper() if random.random() < 0.15 else c.lower() for c in text
            )

        elif option == "first_only":
            return text[0].upper() + text[1:].lower() if text else text

        return text

    def _vary_number_format(self, text: str) -> str:
        """Convert between digit and word representations."""
        result = text

        # Convert digits to words or vice versa
        for digit, word in NUMBER_WORD_MAPPINGS.items():
            if digit in result:
                if random.random() < 0.5:
                    result = result.replace(digit, word, 1)
            elif word in result.lower():
                if random.random() < 0.5:
                    pattern = re.compile(re.escape(word), re.IGNORECASE)
                    result = pattern.sub(digit, result, count=1)

        return result

    def _apply_code_switching(self, text: str) -> str:
        """Mix French and English (bilingual speaker artifact)."""
        for french, english_options in CODE_SWITCH_REPLACEMENTS.items():
            if french.lower() in text.lower():
                if random.random() < 0.4:
                    pattern = re.compile(re.escape(french), re.IGNORECASE)
                    return pattern.sub(random.choice(english_options), text, count=1)
        return text

    def _add_hallucination(self, text: str) -> str:
        """Add Whisper hallucination artifacts."""
        hallucination = random.choice(WHISPER_HALLUCINATIONS)
        position = random.choice(["start", "end"])

        if position == "start":
            return hallucination + " " + text
        return text + " " + hallucination

    def _add_noise_artifacts(self, text: str) -> str:
        """Insert noise/inaudible markers."""
        words = text.split()
        if len(words) < 2:
            return text

        marker = random.choice(NOISE_MARKERS)
        pos = random.randint(0, len(words))
        words.insert(pos, marker)

        return " ".join(words)

    def _apply_accent_variation(self, text: str) -> str:
        """Transcribe regional accents phonetically."""
        # Southern French variations
        southern = {
            "pain": ["pang", "paing"],
            "vin": ["vang", "ving"],
            "demain": ["demaing"],
            "bien": ["bieng"],
        }

        # Northern/Ch'ti variations
        northern = {"je": ["ch'"], "c'est": ["ch'est"], "ce": ["ch'"]}

        # Choose one regional pattern
        patterns = random.choice([southern, northern])

        result = text
        for standard, variants in patterns.items():
            if standard in result.lower():
                if random.random() < 0.3:
                    pattern = re.compile(re.escape(standard), re.IGNORECASE)
                    result = pattern.sub(random.choice(variants), result, count=1)

        return result

    def _apply_homophones(self, text: str) -> str:
        """Replace words with their homophones."""
        words = text.split()
        result = []

        for word in words:
            # Check without punctuation
            clean_word = word.strip(".,!?;:'\"")
            punctuation = word[len(clean_word) :] if len(word) > len(clean_word) else ""

            word_lower = clean_word.lower()
            if word_lower in FRENCH_HOMOPHONES and random.random() < 0.3:
                replacement = FRENCH_HOMOPHONES[word_lower]
                # Preserve original case
                if clean_word[0].isupper():
                    replacement = replacement.capitalize()
                result.append(replacement + punctuation)
            else:
                result.append(word)

        return " ".join(result)


def create_augmenter(
    intensity: str = "moderate", seed: int | None = None
) -> STTAugmenter:
    """
    Factory function to create an STTAugmenter with preset intensity.

    Args:
        intensity: One of "clean", "light", "moderate", "heavy".
        seed: Random seed for reproducibility.

    Returns:
        Configured STTAugmenter instance.
    """
    config = ERROR_PROFILES.get(intensity, ERROR_PROFILES["moderate"])
    return STTAugmenter(config=config, seed=seed)
