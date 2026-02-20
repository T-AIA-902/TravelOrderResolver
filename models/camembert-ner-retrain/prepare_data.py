"""
Preparation des donnees NER pour l'entrainement CamemBERT natif.

Utilise les datasets raw :
  - datasets/raw/sentences/travel_sentences.csv  (phrases annotees)
  - datasets/raw/sncf/gares.csv                  (noms de gares)
  - datasets/raw/sncf/gares-de-voyageurs.json    (liste complete des gares)

Genere un dataset NER au format JSON avec les labels :
  O, B-DEP, I-DEP, B-DEST, I-DEST

Garanties de non-fuite entre splits :
  - Templates TRIP separes : train vs eval (val/test)
  - Gares separees : 80% train / 10% val / 10% test
  - Phrases NOT_TRIP et AMBIGUES separees par pool (zero overlap)
  - Augmentations de texte (casse, mots parasites) sur ~20% du train

Usage:
    python models/camembert-ner-retrain/prepare_data.py
    python models/camembert-ner-retrain/prepare_data.py --n_trip 10000 --n_not_trip 3000
"""

import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "datasets" / "raw"
SENTENCES_CSV = RAW_DIR / "sentences" / "travel_sentences.csv"
GARES_CSV = RAW_DIR / "sncf" / "gares.csv"
GARES_JSON = RAW_DIR / "sncf" / "gares-de-voyageurs.json"
OUTPUT_DIR = PROJECT_ROOT / "datasets" / "processed" / "ner_retrain"

LABEL_LIST = ["O", "B-DEP", "I-DEP", "B-DEST", "I-DEST"]
LABEL2ID = {label: idx for idx, label in enumerate(LABEL_LIST)}
ID2LABEL = {idx: label for idx, label in enumerate(LABEL_LIST)}

# ---------------------------------------------------------------------------
# Templates de phrases de voyage - TRAIN uniquement
# ---------------------------------------------------------------------------
TRAIN_TEMPLATES = [
    # --- Sans accent ---
    "Je veux aller de {DEP} a {DEST}",
    "Je souhaite aller de {DEP} a {DEST}",
    "Je voudrais aller de {DEP} a {DEST}",
    "Emmene-moi de {DEP} a {DEST}",
    "Je vais de {DEP} a {DEST}",
    "Aller de {DEP} a {DEST}",
    "De {DEP} a {DEST}",
    "De {DEP} a {DEST} s'il vous plait",
    "Comment aller de {DEP} a {DEST}",
    "Comment me rendre de {DEP} a {DEST}",
    "Comment me rendre a {DEST} depuis {DEP}",
    "Je veux me rendre a {DEST} depuis {DEP}",
    "Je veux un billet de {DEP} a {DEST}",
    "Reserve-moi un billet de {DEP} a {DEST}",
    "Un billet de {DEP} a {DEST}",
    "Un aller simple de {DEP} a {DEST}",
    "Un aller-retour de {DEP} a {DEST}",
    "Je cherche un train de {DEP} a {DEST}",
    "Y a-t-il un train de {DEP} a {DEST}",
    "Quel train pour aller de {DEP} a {DEST}",
    "Je pars de {DEP} pour aller a {DEST}",
    "Depart de {DEP} arrivee a {DEST}",
    "Depart {DEP} destination {DEST}",
    "Je voudrais partir de {DEP} vers {DEST}",
    "Faut que j'aille de {DEP} a {DEST}",
    "Je dois me rendre de {DEP} a {DEST}",
    "Amene-moi de {DEP} a {DEST}",
    "Trajet de {DEP} a {DEST}",
    "Voyage de {DEP} a {DEST}",
    "Je veux voyager de {DEP} a {DEST}",
    "Quand part le prochain train de {DEP} a {DEST}",
    "A quelle heure y a-t-il des trains de {DEP} a {DEST}",
    "Un billet pour {DEST} au depart de {DEP}",
    "Pour aller a {DEST} depuis {DEP}",
    "Direction {DEST} depuis {DEP}",
    # --- Avec accent à ---
    "Je veux aller de {DEP} à {DEST}",
    "Je voudrais aller de {DEP} à {DEST}",
    "De {DEP} à {DEST}",
    "De {DEP} à {DEST} s'il vous plaît",
    "Comment aller de {DEP} à {DEST}",
    "Comment me rendre à {DEST} depuis {DEP}",
    "Je veux un billet de {DEP} à {DEST}",
    "Un billet de {DEP} à {DEST}",
    "Un aller simple de {DEP} à {DEST}",
    "Je cherche un train de {DEP} à {DEST}",
    "Je pars de {DEP} pour aller à {DEST}",
    "Faut que j'aille de {DEP} à {DEST}",
    "Trajet de {DEP} à {DEST}",
    "Je veux voyager de {DEP} à {DEST}",
    "Pour aller à {DEST} depuis {DEP}",
    "Emmène-moi de {DEP} à {DEST}",
    "Amène-moi de {DEP} à {DEST}",
    # --- Avec étape intermédiaire (étape = O, pas d'entité) ---
    "Je veux aller de {DEP} à {DEST} en passant par Valence",
    "De {DEP} à {DEST} via Mâcon",
    "Un billet de {DEP} à {DEST} avec correspondance à Lyon",
    "Je pars de {DEP} pour {DEST} en passant par Tours",
    "De {DEP} à {DEST} avec un arrêt à Dijon",
    "Trajet de {DEP} à {DEST} via Bordeaux",
    "Je veux aller de {DEP} à {DEST} en passant par Lille",
    "Un train de {DEP} à {DEST} avec changement à Nantes",
]

# ---------------------------------------------------------------------------
# Templates de phrases de voyage - EVAL uniquement (val + test)
# Structures differentes, jamais vues a l'entrainement
# ---------------------------------------------------------------------------
EVAL_TEMPLATES = [
    "Je souhaite me deplacer de {DEP} jusqu'a {DEST}",
    "Peux-tu me trouver un trajet de {DEP} a {DEST}",
    "Il me faut un transport de {DEP} a {DEST}",
    "J'ai besoin d'aller a {DEST} en partant de {DEP}",
    "Conduisez-moi de {DEP} a {DEST}",
    "Quel est le meilleur itineraire de {DEP} a {DEST}",
    "Je desire voyager depuis {DEP} jusqu'a {DEST}",
    "Existe-t-il une liaison entre {DEP} et {DEST}",
    "Programme-moi un deplacement de {DEP} vers {DEST}",
    "Le trajet {DEP} {DEST} est-il possible",
    "Je compte partir de {DEP} et arriver a {DEST}",
    "Trouve-moi un moyen d'aller de {DEP} a {DEST}",
    "J'aimerais rejoindre {DEST} au depart de {DEP}",
    "Reservation de {DEP} en direction de {DEST}",
    "Planifie mon voyage de {DEP} a {DEST}",
    # Avec accent
    "Je souhaite me déplacer de {DEP} jusqu'à {DEST}",
    "J'ai besoin d'aller à {DEST} en partant de {DEP}",
    "Trouve-moi un moyen d'aller de {DEP} à {DEST}",
    "Je compte partir de {DEP} et arriver à {DEST}",
    "Planifie mon voyage de {DEP} à {DEST}",
    # Avec étape intermédiaire
    "Je souhaite me déplacer de {DEP} à {DEST} via Strasbourg",
    "Trouve-moi un trajet de {DEP} à {DEST} en passant par Marseille",
]

# ---------------------------------------------------------------------------
# Phrases NOT_TRIP - pools separes par split (zero overlap)
# ---------------------------------------------------------------------------
NOT_TRIP_TRAIN = [
    "Quel temps fait-il demain",
    "Bonjour comment ca va",
    "Quelle heure est-il",
    "Raconte-moi une blague",
    "Manger une pomme",
    "J'aime le chocolat",
    "Le chat dort sur le canape",
    "Il fait beau aujourd'hui",
    "Je cherche un restaurant",
    "Quel est le prix du cafe",
    "Merci beaucoup",
    "Au revoir et bonne journee",
    "Est-ce que tu parles anglais",
    "Je ne comprends pas",
    "C'est une bonne idee",
    "Le film etait super",
    "J'ai faim",
    "Il pleut dehors",
    "Mon telephone ne marche plus",
    "Ou est la sortie",
    "Je suis en retard",
    "Peux-tu m'aider",
    "La reunion est a quelle heure",
    "Je voudrais un cafe",
    "Le match commence bientot",
    "Il y a du monde ici",
    "C'est trop cher",
    "Je reviens tout de suite",
    "Bonne nuit",
    "Joyeux anniversaire",
    "Je suis fatigue",
    "Il faut acheter du pain",
    "Le cours commence dans cinq minutes",
    "Tu as vu les nouvelles",
    "Je n'ai plus de batterie",
    "On mange ou ce soir",
    "Appelle-moi demain matin",
    "C'est ferme le dimanche",
    "J'ai perdu mes cles",
    "La connexion internet ne marche pas",
    "Mon chien s'appelle Rex",
    "Le soleil se couche tard en ete",
    "Tu viens a la fete ce soir",
    "J'ai oublie mon mot de passe",
    "La pizza est delicieuse ici",
    "Il neige depuis ce matin",
    "Mon frere habite a l'etranger",
    "Je dois finir ce rapport avant vendredi",
    "L'examen est dans deux semaines",
    "Tu connais un bon dentiste",
    "Le magasin ouvre a neuf heures",
    "Les enfants jouent dans le jardin",
    "J'ai mal a la tete",
    "La voiture est au garage",
    "Le livre etait passionnant",
    "On regarde un film ce soir",
    "Je dois appeler le plombier",
    "La machine a laver est en panne",
    "Mon voisin fait trop de bruit",
    "Le prix de l'essence a augmente",
]

NOT_TRIP_VAL = [
    "Il faut arroser les plantes",
    "La pharmacie est ouverte le dimanche",
    "Je cherche un cadeau pour ma mere",
    "Le Wi-Fi ne fonctionne pas",
    "On joue au foot demain apres-midi",
    "La cantine est fermee aujourd'hui",
    "Mon ordinateur rame beaucoup",
    "J'ai un rendez-vous chez le medecin",
    "Le concert est complet",
    "Elle a reussi son permis de conduire",
    "Les soldes commencent la semaine prochaine",
    "Le chat a renverse le vase",
    "Mon abonnement expire bientot",
    "La boulangerie du coin fait de bons croissants",
    "Le prof est absent aujourd'hui",
    "La climatisation ne marche plus",
    "Tu as fini tes devoirs",
    "Le parking est plein",
    "J'ai besoin de nouveaux ecouteurs",
    "La seance de cinema est a vingt heures",
]

NOT_TRIP_TEST = [
    "Le four est casse depuis hier",
    "Mon fils a eu une bonne note",
    "La piscine est ouverte jusqu'a dix-neuf heures",
    "J'ai un entretien d'embauche lundi",
    "Le supermarche est a cote de la banque",
    "Tu peux me preter ton chargeur",
    "La serie est vraiment bien",
    "Le loyer a augmente cette annee",
    "Les vacances scolaires commencent vendredi",
    "Mon imprimante n'a plus d'encre",
    "Le restaurant italien est excellent",
    "La reunion a ete annulee",
    "J'ai commande un colis sur internet",
    "Le voisin a adopte un chien",
    "La tempete arrive ce week-end",
    "Mon cousin se marie en juin",
    "Le musee est gratuit le premier dimanche",
    "L'ascenseur est en panne",
    "Elle a change de numero de telephone",
    "Le match de rugby est ce samedi",
]

# ---------------------------------------------------------------------------
# Phrases AMBIGUES - mentionnent deplacement/lieux mais PAS commande voyage
# ---------------------------------------------------------------------------
AMBIGUOUS_TRAIN = [
    "Je veux aller au restaurant ce soir",
    "On se retrouve a la gare pour prendre un cafe",
    "Je reviens de vacances la semaine prochaine",
    "Il faut aller chercher les enfants a l'ecole",
    "J'ai fait un beau voyage cet ete",
    "La gare est en travaux en ce moment",
    "Le train avait du retard hier",
    "J'adore prendre le train pour lire",
    "Mon pere travaille a la SNCF",
    "La gare de Lyon est tres grande",
    "Je dois aller chez le coiffeur",
    "On va au cinema ce soir",
    "Il faut que j'aille a la banque",
    "Je passe par la gare en rentrant",
    "Le voyage en avion etait fatiguant",
    "On se voit a Paris la semaine prochaine",
    "J'ai visite Marseille le mois dernier",
]

AMBIGUOUS_VAL = [
    "La gare routiere est mal desservie",
    "Je vais au supermarche acheter du lait",
    "Le voyage scolaire est annule",
    "On prend un taxi pour aller au restaurant",
    "L'arret de bus est juste en face",
    "J'ai perdu mon billet de spectacle",
]

AMBIGUOUS_TEST = [
    "Le TGV est plus rapide que la voiture",
    "J'ai rate mon bus ce matin",
    "La station de metro est en travaux",
    "On devrait prendre le velo pour y aller",
    "Mon abonnement SNCF expire demain",
    "Le parking de la gare est toujours plein",
    "Je prends le bus tous les matins pour le travail",
]

# ---------------------------------------------------------------------------
# Mots parasites pour augmentation
# ---------------------------------------------------------------------------
FILLER_WORDS = ["euh", "donc", "bon", "alors", "genre", "enfin", "bref", "hein"]


def load_stations() -> List[str]:
    """Charge les noms de gares/villes depuis les fichiers raw SNCF."""
    cities = set()

    # gares.csv : colonne city_name
    if GARES_CSV.exists():
        df = pd.read_csv(GARES_CSV)
        for _, row in df.iterrows():
            name = str(row["city_name"]).strip()
            if name and name.lower() != "nan":
                cities.add(name)

    # gares-de-voyageurs.json : champ "nom"
    if GARES_JSON.exists():
        with open(GARES_JSON, "r", encoding="utf-8") as f:
            stations = json.load(f)
        for s in stations:
            name = s.get("nom", "").strip()
            if name:
                cities.add(name)

    return sorted(cities)


def split_stations(
    stations: List[str], seed: int = 42
) -> Tuple[List[str], List[str], List[str]]:
    """Separe les gares en 3 pools disjoints (80/10/10)."""
    rng = random.Random(seed)
    shuffled = list(stations)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)

    return shuffled[:n_train], shuffled[n_train:n_train + n_val], shuffled[n_train + n_val:]


def tokenize_and_label(
    sentence: str, dep: str, dest: str
) -> Tuple[List[str], List[str]]:
    """
    Tokenisation par espaces et etiquetage BIO.

    Retourne (tokens, labels) avec labels parmi
    O / B-DEP / I-DEP / B-DEST / I-DEST.
    """
    tokens = sentence.split()
    labels = ["O"] * len(tokens)

    dep_tokens = dep.split()
    dest_tokens = dest.split()

    # Recherche du depart
    for i in range(len(tokens) - len(dep_tokens) + 1):
        window = [t.lower().rstrip(".,!?;:'\"") for t in tokens[i : i + len(dep_tokens)]]
        target = [t.lower() for t in dep_tokens]
        if window == target:
            labels[i] = "B-DEP"
            for j in range(1, len(dep_tokens)):
                labels[i + j] = "I-DEP"
            break

    # Recherche de la destination (sans ecraser le depart)
    for i in range(len(tokens) - len(dest_tokens) + 1):
        if any(labels[i + j] != "O" for j in range(len(dest_tokens))):
            continue
        window = [t.lower().rstrip(".,!?;:'\"") for t in tokens[i : i + len(dest_tokens)]]
        target = [t.lower() for t in dest_tokens]
        if window == target:
            labels[i] = "B-DEST"
            for j in range(1, len(dest_tokens)):
                labels[i + j] = "I-DEST"
            break

    return tokens, labels


def augment_sentence(sentence: str, dep: str, dest: str, rng: random.Random) -> str:
    """Applique des augmentations aleatoires a une phrase.

    - Variation de casse sur tous les mots (y compris noms de gares)
    - Insertion de mots parasites
    """
    words = sentence.split()

    # Insertion d'un mot parasite (30% de chance)
    if rng.random() < 0.3 and len(words) > 2:
        filler = rng.choice(FILLER_WORDS)
        # Inserer a une position qui n'est pas dans une entite
        dep_tokens = set(dep.lower().split())
        dest_tokens = set(dest.lower().split())
        safe_positions = [
            i for i, w in enumerate(words)
            if w.lower().rstrip(".,!?;:'\"") not in dep_tokens
            and w.lower().rstrip(".,!?;:'\"") not in dest_tokens
            and i > 0
        ]
        if safe_positions:
            pos = rng.choice(safe_positions)
            words.insert(pos, filler)

    # Variation de casse (50% de chance) - inclut les noms de gares
    if rng.random() < 0.5:
        for i, w in enumerate(words):
            r = rng.random()
            if r < 0.3:
                words[i] = w.lower()
            elif r < 0.35:
                words[i] = w.upper()

    return " ".join(words)


def convert_raw_sentences() -> List[Dict]:
    """
    Convertit les phrases brutes de travel_sentences.csv.

    Les labels B-LOC / I-LOC sont convertis en B-DEP/I-DEP et B-DEST/I-DEST
    selon l'ordre d'apparition (1ere localite = depart, 2e = destination).
    """
    samples = []
    if not SENTENCES_CSV.exists():
        return samples

    df = pd.read_csv(SENTENCES_CSV)
    for _, row in df.iterrows():
        tokens = str(row["sentence"]).split()
        raw_labels = str(row["labels"]).split()

        labels = []
        loc_count = 0
        for lab in raw_labels:
            lab_upper = lab.upper()
            if lab_upper == "B-LOC":
                loc_count += 1
                labels.append("B-DEP" if loc_count == 1 else "B-DEST")
            elif lab_upper == "I-LOC":
                labels.append("I-DEP" if loc_count <= 1 else "I-DEST")
            else:
                labels.append("O")

        # Ajuster la longueur si necessaire
        while len(labels) < len(tokens):
            labels.append("O")
        labels = labels[: len(tokens)]

        sid = row["sentenceID"]
        samples.append(
            {
                "id": f"RAW_{sid}",
                "tokens": tokens,
                "labels": labels,
                "sentence": str(row["sentence"]),
                "intent": "TRIP" if any(l != "O" for l in labels) else "NOT_TRIP",
            }
        )
    return samples


def generate_split(
    split_name: str,
    templates: List[str],
    stations: List[str],
    not_trip_pool: List[str],
    ambiguous_pool: List[str],
    n_trip: int,
    n_not_trip: int,
    rng: random.Random,
    augment: bool = False,
) -> List[Dict]:
    """Genere les exemples pour un split donne avec ses propres pools."""
    samples = []

    # --- Phrases de voyage generees ---
    for i in range(n_trip):
        template = rng.choice(templates)
        dep = rng.choice(stations)
        dest = rng.choice(stations)
        while dest == dep:
            dest = rng.choice(stations)

        sentence = template.format(DEP=dep, DEST=dest)

        # Augmentation (~30% du train)
        if augment and rng.random() < 0.3:
            sentence = augment_sentence(sentence, dep, dest, rng)

        tokens, labels = tokenize_and_label(sentence, dep, dest)

        samples.append(
            {
                "id": f"GEN_TRIP_{split_name}_{i:06d}",
                "tokens": tokens,
                "labels": labels,
                "sentence": sentence,
                "intent": "TRIP",
            }
        )

    # --- Phrases hors-voyage ---
    for i in range(n_not_trip):
        sentence = rng.choice(not_trip_pool)
        tokens = sentence.split()
        labels = ["O"] * len(tokens)

        samples.append(
            {
                "id": f"GEN_NOT_TRIP_{split_name}_{i:06d}",
                "tokens": tokens,
                "labels": labels,
                "sentence": sentence,
                "intent": "NOT_TRIP",
            }
        )

    # --- Phrases ambigues (toutes, pas d'echantillonnage) ---
    for i, sentence in enumerate(ambiguous_pool):
        tokens = sentence.split()
        labels = ["O"] * len(tokens)

        samples.append(
            {
                "id": f"GEN_AMBIGUOUS_{split_name}_{i:06d}",
                "tokens": tokens,
                "labels": labels,
                "sentence": sentence,
                "intent": "NOT_TRIP",
            }
        )

    rng.shuffle(samples)
    return samples


def main():
    parser = argparse.ArgumentParser(description="Prepare NER dataset for CamemBERT training")
    parser.add_argument("--n_trip", type=int, default=5000, help="Number of TRIP sentences for train")
    parser.add_argument("--n_not_trip", type=int, default=1500, help="Number of NOT_TRIP sentences for train")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    print("=" * 60)
    print("  Preparation des donnees NER - CamemBERT natif")
    print("=" * 60)

    # 1. Charger et separer les gares
    print("\n[1/5] Chargement des gares SNCF...")
    all_stations = load_stations()
    train_stations, val_stations, test_stations = split_stations(all_stations, seed=args.seed)
    print(f"      {len(all_stations)} gares total")
    print(f"      Train: {len(train_stations)} | Val: {len(val_stations)} | Test: {len(test_stations)}")

    # 2. Generer le split train
    print(f"\n[2/5] Generation du train ({args.n_trip} TRIP + {args.n_not_trip} NOT_TRIP)...")
    train_samples = generate_split(
        split_name="train",
        templates=TRAIN_TEMPLATES,
        stations=train_stations,
        not_trip_pool=NOT_TRIP_TRAIN,
        ambiguous_pool=AMBIGUOUS_TRAIN,
        n_trip=args.n_trip,
        n_not_trip=args.n_not_trip,
        rng=rng,
        augment=True,
    )

    # Ajouter les phrases brutes annotees (travel_sentences.csv) au train
    raw = convert_raw_sentences()
    train_samples.extend(raw)
    rng.shuffle(train_samples)

    trip_count = sum(1 for s in train_samples if s["intent"] == "TRIP")
    raw_count = sum(1 for s in train_samples if s["id"].startswith("RAW_"))
    print(f"      {len(train_samples)} exemples (TRIP: {trip_count} | NOT_TRIP: {len(train_samples) - trip_count} | Raw: {raw_count})")

    # 3. Generer le split val (templates eval, gares val)
    n_val_trip = args.n_trip // 8
    n_val_not_trip = args.n_not_trip // 8
    print(f"\n[3/5] Generation du val ({n_val_trip} TRIP + {n_val_not_trip} NOT_TRIP)...")
    val_samples = generate_split(
        split_name="val",
        templates=EVAL_TEMPLATES,
        stations=val_stations,
        not_trip_pool=NOT_TRIP_VAL,
        ambiguous_pool=AMBIGUOUS_VAL,
        n_trip=n_val_trip,
        n_not_trip=n_val_not_trip,
        rng=rng,
        augment=False,
    )
    print(f"      {len(val_samples)} exemples")

    # 4. Generer le split test (templates eval, gares test)
    n_test_trip = args.n_trip // 8
    n_test_not_trip = args.n_not_trip // 8
    print(f"\n[4/5] Generation du test ({n_test_trip} TRIP + {n_test_not_trip} NOT_TRIP)...")
    test_samples = generate_split(
        split_name="test",
        templates=EVAL_TEMPLATES,
        stations=test_stations,
        not_trip_pool=NOT_TRIP_TEST,
        ambiguous_pool=AMBIGUOUS_TEST,
        n_trip=n_test_trip,
        n_not_trip=n_test_not_trip,
        rng=rng,
        augment=False,
    )
    print(f"      {len(test_samples)} exemples")

    # 5. Sauvegarder
    print(f"\n[5/5] Sauvegarde dans {OUTPUT_DIR}...")
    save_json(train_samples, OUTPUT_DIR / "train.json")
    save_json(val_samples, OUTPUT_DIR / "val.json")
    save_json(test_samples, OUTPUT_DIR / "test.json")

    label_config = {
        "label_list": LABEL_LIST,
        "label2id": LABEL2ID,
        "id2label": ID2LABEL,
    }
    save_json(label_config, OUTPUT_DIR / "label_config.json")

    print("\n  Fichiers generes :")
    print(f"    - train.json   ({len(train_samples)} exemples)")
    print(f"    - val.json     ({len(val_samples)} exemples)")
    print(f"    - test.json    ({len(test_samples)} exemples)")
    print(f"    - label_config.json")

    # Stats de non-fuite
    print("\n  Garanties de non-fuite :")
    print(f"    - Templates train : {len(TRAIN_TEMPLATES)} (reserves au train)")
    print(f"    - Templates eval  : {len(EVAL_TEMPLATES)} (reserves au val/test)")
    print(f"    - NOT_TRIP train  : {len(NOT_TRIP_TRAIN)} uniques")
    print(f"    - NOT_TRIP val    : {len(NOT_TRIP_VAL)} uniques (zero overlap)")
    print(f"    - NOT_TRIP test   : {len(NOT_TRIP_TEST)} uniques (zero overlap)")
    print(f"    - Ambigues train  : {len(AMBIGUOUS_TRAIN)} | val: {len(AMBIGUOUS_VAL)} | test: {len(AMBIGUOUS_TEST)}")
    print(f"    - Gares train     : {len(train_stations)} | val: {len(val_stations)} | test: {len(test_stations)}")
    print("\n  Termine.")


def save_json(data, filepath: Path) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
