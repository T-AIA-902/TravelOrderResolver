import random
import sys
import unicodedata
from pathlib import Path

import pandas as pd

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    from faker import Faker
except ImportError:
    Faker = None

OUTPUT_CSV = "dataset_train_sncf.csv"
NUM_SAMPLES = 100000
RATIO_INVALID = 0.20

fake = Faker("fr_FR") if Faker else None


def load_stations():
    """Load stations from StationDatabase (JSON source)."""
    from data import StationDatabase

    print("📂 Chargement des gares depuis StationDatabase...")
    db = StationDatabase()
    db.load()

    # Get all passenger station names
    stations = [s.name for s in db.get_all_stations(passenger_only=True)]
    print(f"✅ {len(stations)} gares chargées avec succès.")
    return stations


synonyms = {
    "intent_polite": [
        "Je voudrais",
        "J'aimerais",
        "Je souhaite",
        "Pourriez-vous me trouver",
        "Je veux",
        "Il me faut",
        "J'ai besoin de",
        "Recherche",
        "Je cherche",
    ],
    "intent_imperative": ["Trouve-moi", "Cherche", "Donne-moi", "Affiche", "Sors-moi", "Go pour"],
    "intent_question": [
        "Y a-t-il",
        "Est-ce qu'il y a",
        "Quels sont les",
        "Existe-t-il",
        "A quelle heure part",
    ],
    "noun_trip": [
        "un train",
        "un billet",
        "un trajet",
        "un voyage",
        "une place",
        "un aller",
        "un aller-simple",
        "une liaison",
        "un TGV",
        "un TER",
    ],
    "prep_from": [
        "de",
        "depuis",
        "au départ de",
        "en partant de",
        "partant de",
        "venant de",
        "du côté de",
    ],
    "prep_to": ["à", "pour", "vers", "à destination de", "jusqu'à", "en direction de", "au"],
    "time_adverb": [
        "maintenant",
        "demain",
        "tout de suite",
        "plus tard",
        "ce soir",
        "ce matin",
        "lundi",
    ],
}


def generate_valid_sentence(dep, dest):
    """
    Generate a valid travel request sentence.

    Returns:
        Tuple of (sentence, actual_dep, actual_dest) where actual_dep may be None
        for destination-only templates.
    """
    polite = random.choice(synonyms["intent_polite"])
    noun = random.choice(synonyms["noun_trip"])
    prep_from = random.choice(synonyms["prep_from"])
    prep_to = random.choice(synonyms["prep_to"])
    question = random.choice(synonyms["intent_question"])
    imperative = random.choice(synonyms["intent_imperative"])

    # Templates with both departure and destination
    structures_both = [
        f"{polite} {noun} {prep_from} {dep} {prep_to} {dest}",
        f"{noun} {prep_to} {dest} {prep_from} {dep}",
        f"{imperative} {noun} entre {dep} et {dest}",
        f"train {dep} {dest}",
        f"billet {dest} depuis {dep}",
        f"{dep} - {dest}",
        f"Je dois aller à {dest} pour une réunion, je pars de {dep}",
        f"Réserve moi une place pour {dest} en provenance de {dep}",
    ]

    # Templates with only destination (no departure in text)
    structures_dest_only = [
        f"{question} {noun} {prep_to} {dest} ?",
    ]

    # 85% both, 15% dest-only (realistic ratio)
    if random.random() < 0.85:
        sentence = random.choice(structures_both)
        return sentence, dep, dest
    else:
        sentence = random.choice(structures_dest_only)
        return sentence, None, dest  # No departure for dest-only templates


def generate_invalid_sentence(stations):
    city_trap = random.choice(stations)
    first_name = fake.first_name() if fake else "Albert"

    traps = [
        f"Je m'appelle {city_trap}",
        f"Il habite à {city_trap}",
        f"La ville de {city_trap} est belle",
        f"Quel temps fait-il à {city_trap} ?",
        "Je voudrais une pizza",
        "Raconte moi une blague",
        f"Bonjour {first_name}, comment ça va ?",
        "C'est quoi le sens de la vie ?",
        f"Je cherche mon ami {first_name}",
    ]
    return random.choice(traps)


def apply_noise(text):
    text_list = list(text)

    if random.random() < 0.5:
        text = text.lower()
        text_list = list(text)

    if random.random() < 0.3:
        text = "".join(
            c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
        )
        text_list = list(text)

    if random.random() < 0.1 and len(text) > 5:
        idx = random.randint(0, len(text) - 2)
        text_list[idx], text_list[idx + 1] = text_list[idx + 1], text_list[idx]

    return "".join(text_list)


def main():
    cities = load_stations()
    data = []

    print(f"🚀 Génération de {NUM_SAMPLES} phrases...")

    for i in range(NUM_SAMPLES):
        is_valid = random.random() > RATIO_INVALID

        if is_valid:
            dep = random.choice(cities)
            dest = random.choice(cities)
            while dep == dest:
                dest = random.choice(cities)

            raw_text, actual_dep, actual_dest = generate_valid_sentence(dep, dest)
            final_text = apply_noise(raw_text)

            data.append(
                {
                    "id": i,
                    "text": final_text,
                    "label": "VALID",
                    "departure": actual_dep,  # May be None for dest-only templates
                    "destination": actual_dest,
                }
            )

        else:
            raw_text = generate_invalid_sentence(cities)
            final_text = apply_noise(raw_text)

            data.append(
                {
                    "id": i,
                    "text": final_text,
                    "label": "INVALID",
                    "departure": None,
                    "destination": None,
                }
            )

    df_output = pd.DataFrame(data)
    df_output.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    print(f"✨ Terminé ! Fichier généré : {OUTPUT_CSV}")
    valid_count = len(df_output[df_output["label"] == "VALID"])
    invalid_count = len(df_output[df_output["label"] == "INVALID"])
    print(f"📊 Statistiques : {valid_count} valides / {invalid_count} invalides.")
    print(df_output.head(10))


if __name__ == "__main__":
    main()
