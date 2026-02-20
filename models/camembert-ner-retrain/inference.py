"""
Inference CamemBERT NER + post-processing.

Pipeline complet :
  1. Tokenisation + prediction NER (B-DEP, I-DEP, B-DEST, I-DEST, O)
  2. Extraction des entites (depart, destination)
  3. Post-processing (fuzzy matching gares SNCF, detection langue, validation)
  4. Sortie structuree :
       - Phrase valide  -> sentenceID, Departure, Destination
       - Phrase invalide -> sentenceID, Code
         Codes : NOT_FRENCH | UNKNOWN | NOT_TRIP

Usage:
    python models/camembert-ner-retrain/inference.py --sentence "Je veux aller de Paris a Lyon"
    python models/camembert-ner-retrain/inference.py --input_csv data.csv --output_csv results.csv
"""

import json
import csv
import argparse
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import torch
from transformers import CamembertTokenizerFast, CamembertForTokenClassification

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "camembert-ner-retrain"
GARES_CSV = PROJECT_ROOT / "datasets" / "raw" / "sncf" / "gares.csv"
GARES_JSON = PROJECT_ROOT / "datasets" / "raw" / "sncf" / "gares-de-voyageurs.json"

LABEL_LIST = ["O", "B-DEP", "I-DEP", "B-DEST", "I-DEST"]
ID2LABEL = {idx: label for idx, label in enumerate(LABEL_LIST)}

# Indicateurs linguistiques pour la detection de langue
FRENCH_INDICATORS = {
    "je", "de", "du", "des", "le", "la", "les", "un", "une",
    "pour", "dans", "en", "au", "aux", "et", "ou", "mais",
    "que", "qui", "ce", "cette", "mon", "ma", "mes", "ton",
    "sa", "ses", "nous", "vous", "ils", "elles", "est", "sont",
    "suis", "veux", "voudrais", "souhaite", "peux", "faut",
    "aller", "partir", "prendre", "avoir", "faire",
}

# Indicateurs de voyage
TRIP_INDICATORS = {
    "aller", "billet", "train", "trajet", "voyage", "voyager",
    "partir", "gare", "destination", "depart", "reserve",
    "horaire", "direct", "correspondance", "tgv", "ter",
    "aller-retour", "simple", "rendre", "emmene", "amene",
    "direction", "vers", "depuis",
}


def load_station_names() -> List[str]:
    """Charge la liste des noms de gares SNCF pour le fuzzy matching."""
    stations = set()

    if GARES_CSV.exists():
        import pandas as pd
        df = pd.read_csv(GARES_CSV)
        for _, row in df.iterrows():
            stations.add(str(row["city_name"]).strip())
            stations.add(str(row["stations_name"]).strip())

    if GARES_JSON.exists():
        with open(GARES_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        for s in data:
            name = s.get("nom", "").strip()
            if name:
                stations.add(name)

    return sorted(stations)


def fuzzy_match_station(entity: str, stations: List[str], threshold: int = 80) -> Optional[str]:
    """Fuzzy matching d'une entite extraite contre la base de gares."""
    try:
        from rapidfuzz import fuzz, process
        result = process.extractOne(entity, stations, scorer=fuzz.ratio, score_cutoff=threshold)
        if result:
            return result[0]
    except ImportError:
        # Fallback : matching exact insensible a la casse
        entity_lower = entity.lower()
        for station in stations:
            if station.lower() == entity_lower:
                return station
    return entity


# ---------------------------------------------------------------------------
# NER Prediction
# ---------------------------------------------------------------------------
def predict_ner(
    model: CamembertForTokenClassification,
    tokenizer: CamembertTokenizerFast,
    sentence: str,
    device: torch.device,
) -> Tuple[List[str], List[str]]:
    """
    Prediction NER sur une phrase.

    Retourne (words, predicted_labels) au niveau mot.
    """
    words = sentence.split()
    if not words:
        return [], []

    encoding = tokenizer(
        words,
        is_split_into_words=True,
        max_length=128,
        truncation=True,
        return_tensors="pt",
    )
    encoding = {k: v.to(device) for k, v in encoding.items()}

    with torch.no_grad():
        outputs = model(**encoding)

    predictions = outputs.logits.argmax(dim=-1).squeeze().cpu().tolist()
    if isinstance(predictions, int):
        predictions = [predictions]

    # Subword -> word-level
    word_ids_list = tokenizer(
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
    word_preds = word_preds[: len(words)]

    return words, word_preds


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------
def extract_entities(words: List[str], labels: List[str]) -> Dict[str, Optional[str]]:
    """
    Extraction des entites DEP et DEST a partir des labels NER.

    Regroupe les tokens B-/I- consecutifs en entites completes.
    """
    departure_tokens = []
    destination_tokens = []
    current_tokens = []
    current_type = None

    for word, label in zip(words, labels):
        if label.startswith("B-"):
            # Sauvegarder l'entite precedente
            if current_tokens and current_type:
                if "DEP" in current_type:
                    departure_tokens = current_tokens
                elif "DEST" in current_type:
                    destination_tokens = current_tokens
            current_type = label[2:]  # DEP ou DEST
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

    # Derniere entite
    if current_tokens and current_type:
        if "DEP" in current_type:
            departure_tokens = current_tokens
        elif "DEST" in current_type:
            destination_tokens = current_tokens

    departure = " ".join(departure_tokens) if departure_tokens else None
    destination = " ".join(destination_tokens) if destination_tokens else None

    return {"departure": departure, "destination": destination}


def detect_language(sentence: str) -> str:
    """Detection simplifiee : francais ou non."""
    words_lower = set(sentence.lower().split())
    french_count = len(words_lower & FRENCH_INDICATORS)
    ratio = french_count / max(len(words_lower), 1)
    if ratio >= 0.15 or len(words_lower) <= 2:
        return "FRENCH"
    return "OTHER"


def classify_sentence(
    sentence: str,
    departure: Optional[str],
    destination: Optional[str],
) -> Dict:
    """
    Classification finale de la phrase.

    Sortie :
      - Phrase valide  : {"departure": str, "destination": str, "code": None}
      - NOT_FRENCH     : phrase non francaise
      - NOT_TRIP       : phrase francaise mais pas une commande de voyage
      - UNKNOWN        : cas ambigu
    """
    # Cas 1 : depart et destination trouves
    if departure and destination:
        return {"departure": departure, "destination": destination, "code": None}

    # Cas 2 : detection de langue
    lang = detect_language(sentence)
    if lang != "FRENCH":
        return {"departure": None, "destination": None, "code": "NOT_FRENCH"}

    # Cas 3 : phrase francaise sans entites de voyage
    words_lower = set(sentence.lower().split())
    has_trip_indicator = bool(words_lower & TRIP_INDICATORS)

    if not has_trip_indicator and not departure and not destination:
        return {"departure": None, "destination": None, "code": "NOT_TRIP"}

    # Cas 4 : partiellement detecte (1 seule entite) ou ambigu
    return {
        "departure": departure,
        "destination": destination,
        "code": "UNKNOWN",
    }


def process_sentence(
    model: CamembertForTokenClassification,
    tokenizer: CamembertTokenizerFast,
    sentence: str,
    device: torch.device,
    stations: Optional[List[str]] = None,
) -> Dict:
    """Pipeline complet : NER + extraction + post-processing."""
    # 1. NER
    words, labels = predict_ner(model, tokenizer, sentence, device)

    # 2. Extraction des entites
    entities = extract_entities(words, labels)

    # 3. Fuzzy matching (optionnel)
    if stations:
        if entities["departure"]:
            entities["departure"] = fuzzy_match_station(entities["departure"], stations)
        if entities["destination"]:
            entities["destination"] = fuzzy_match_station(entities["destination"], stations)

    # 4. Classification
    result = classify_sentence(sentence, entities["departure"], entities["destination"])
    result["ner_labels"] = list(zip(words, labels))

    return result


def format_output(sentence_id: str, result: Dict) -> str:
    """Formatte la sortie selon le format attendu."""
    if result["code"] is None:
        return f"{sentence_id},{result['departure']},{result['destination']}"
    elif result["departure"] or result["destination"]:
        dep = result["departure"] or "?"
        dest = result["destination"] or "?"
        return f"{sentence_id},{dep},{dest},{result['code']}"
    else:
        return f"{sentence_id},{result['code']}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="CamemBERT NER inference + post-processing")
    parser.add_argument("--model_dir", type=str, default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--sentence", type=str, help="Single sentence to process")
    parser.add_argument("--input_csv", type=str, help="Input CSV (sentenceID, sentence)")
    parser.add_argument("--output_csv", type=str, help="Output CSV path")
    parser.add_argument("--no_fuzzy", action="store_true", help="Disable fuzzy matching")
    parser.add_argument("--verbose", action="store_true", help="Show NER details")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)

    # Charger le modele
    print(f"Chargement du modele depuis {model_dir}...")
    tokenizer = CamembertTokenizerFast.from_pretrained(str(model_dir))
    model = CamembertForTokenClassification.from_pretrained(str(model_dir))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Charger les gares (pour fuzzy matching)
    stations = None
    if not args.no_fuzzy:
        stations = load_station_names()
        print(f"  {len(stations)} gares chargees pour le matching")

    # --- Mode phrase unique ---
    if args.sentence:
        result = process_sentence(model, tokenizer, args.sentence, device, stations)
        output = format_output("INPUT", result)
        print(f"\nPhrase : {args.sentence}")
        print(f"Sortie : {output}")

        if args.verbose and result.get("ner_labels"):
            print("\nDetail NER :")
            for word, label in result["ner_labels"]:
                tag = f"  [{label}]" if label != "O" else ""
                print(f"  {word:<25} {label}{tag}")
        return

    # --- Mode CSV batch ---
    if args.input_csv:
        input_path = Path(args.input_csv)
        output_path = Path(args.output_csv) if args.output_csv else input_path.with_name("results.csv")

        with open(input_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        results = []
        for row in rows:
            sid = row.get("sentenceID", row.get("sentence_id", "?"))
            sentence = row.get("sentence", "")
            result = process_sentence(model, tokenizer, sentence, device, stations)
            output_line = format_output(sid, result)
            results.append(output_line)
            print(output_line)

        with open(output_path, "w", encoding="utf-8", newline="") as f:
            f.write("sentenceID,result\n")
            for line in results:
                f.write(line + "\n")

        print(f"\nResultats sauvegardes dans {output_path}")
        return

    # --- Mode interactif ---
    print("\nMode interactif (tapez 'quit' pour quitter)")
    print("-" * 40)
    idx = 0
    while True:
        try:
            sentence = input("\nPhrase > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not sentence or sentence.lower() in ("quit", "exit", "q"):
            break

        idx += 1
        result = process_sentence(model, tokenizer, sentence, device, stations)
        output = format_output(f"S{idx:04d}", result)
        print(f"  -> {output}")

        if args.verbose and result.get("ner_labels"):
            for word, label in result["ner_labels"]:
                if label != "O":
                    print(f"     {word} [{label}]")


if __name__ == "__main__":
    main()
