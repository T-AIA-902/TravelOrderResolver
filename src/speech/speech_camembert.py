# src/speech/speech_camembert.py
"""
Script amélioré pour extraire origine et destination depuis les phrases de voyage.
- Normalise les textes (minuscules, suppression des accents, ponctuation éliminée)
- Recherche par n-grams pour matcher des gares multi-mots
- Utilise rapidfuzz pour fuzzy matching
- Détecte origine/destination via motifs (de/depuis -> origine, à/vers/pour -> destination)
- Garantit toujours 3 colonnes dans la sortie CSV (sentenceID, Departure, Destination)
"""

import re
import unicodedata
import pandas as pd
from rapidfuzz import process

# Chemins des fichiers
SENTENCES_CSV = "datasets/raw/sentences/travel_sentences.csv"
STATIONS_CSV = "datasets/raw/sncf/gares.csv"
OUTPUT_CSV = "travel_orders_output.csv"

# Débogage
DEBUG = True

# Chargement sécurisé des CSVs (essayer utf-8 puis latin-1)
def read_csv_fallback(path):
    try:
        return pd.read_csv(path, encoding="utf-8")
    except Exception:
        return pd.read_csv(path, encoding="latin-1")

sentences_df = read_csv_fallback(SENTENCES_CSV)
stations_df = read_csv_fallback(STATIONS_CSV)

stations_list = stations_df['city_name'].astype(str).tolist()

# Normalisation des chaînes
def normalize_text(s: str) -> str:
    if not isinstance(s, str):
        s = str(s or "")
    s = s.lower()
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^\w\s]", " ", s)  # retirer ponctuation
    s = re.sub(r"\s+", " ", s).strip()
    return s

# Stopwords de base pour éviter de matcher des prépositions / petits mots
STOPWORDS = {
    'de', 'depuis', 'a', 'à', 'y', 'il', 'des', 'le', 'la', 'les', 'en', 'du', 'pour',
    'me', 'je', 'un', 'une', 'se', 'te', 'nous', 'vous', 'qui', 'que', 'quoi', 'ou', 'où',
    'quand', 'comment', 'a-t-il', 'a', 'est', 'sont', 'y-a-t-il'
}

# Construire dictionnaires normalisés -> original
stations_norm = [normalize_text(s) for s in stations_list]
norm_to_original = {n: o for n, o in zip(stations_norm, stations_list)}
max_station_words = max((len(n.split()) for n in stations_norm), default=1)

# Extraire villes via n-grams + fuzzy matching
def extract_cities(sentence: str, threshold: int = 80):
    words = normalize_text(sentence).split()
    found = []
    # parcourir n-grams du plus long au plus court pour privilégier les correspondances multi-mots
    for n in range(max_station_words, 0, -1):
        for i in range(len(words) - n + 1):
            ngram = " ".join(words[i : i + n])
            if not ngram:
                continue
            # ignorer n-grams trop courts ou mots vides
            ngram_compact = ngram.replace(' ', '')
            if len(ngram_compact) < 3:
                continue
            if ngram in STOPWORDS:
                continue
            match = process.extractOne(ngram, stations_norm)
            if match:
                match_norm, score, _ = match
                if score >= threshold:
                    found.append((i, n, ngram, match_norm, score))
    # Trier par position d'apparition et garder uniques tout en conservant le premier match
    found.sort(key=lambda x: x[0])
    seen = set()
    ordered = []
    for _, _, ngram, match_norm, score in found:
        orig = norm_to_original.get(match_norm)
        if orig and orig not in seen:
            seen.add(orig)
            ordered.append((orig, ngram, score))
    return ordered

# Détecter origine et destination via motifs simples
def get_origin_destination(sentence: str, cities: list):
    s_norm = normalize_text(sentence)
    origin = None
    destination = None
    for city in cities:
        city_name = city[0] if isinstance(city, tuple) else city
        city_norm = normalize_text(city_name)
        # motifs pour origine
        if re.search(rf"\b(de|depuis)\s+{re.escape(city_norm)}\b", s_norm):
            if origin is None:
                origin = city_name
        # motifs pour destination
        if re.search(rf"\b(a|à|vers|pour|destination)\s+{re.escape(city_norm)}\b", s_norm):
            if destination is None:
                destination = city_name
    # si au moins deux villes détectées, remplir les manquantes
    city_names = [c[0] if isinstance(c, tuple) else c for c in cities]
    if (origin is None or destination is None) and len(city_names) >= 2:
        if origin is None:
            origin = city_names[0]
        if destination is None and len(city_names) >= 2:
            destination = city_names[1]
    return origin, destination

# Traitement
output_rows = []
for _, row in sentences_df.iterrows():
    sentence_id = row.get('sentenceID')
    sentence = str(row.get('sentence', ''))

    cities = extract_cities(sentence, threshold=80)

    if DEBUG:
        print("---")
        print(f"ID={sentence_id} | sentence= {sentence}")
        print("extracted cities (orig,matched_ngram,score):", cities)

    if not cities:
        output_rows.append([sentence_id, 'INVALID', 'INVALID'])
        continue

    origin, destination = get_origin_destination(sentence, cities)

    if DEBUG:
        print(f"origin={origin} destination={destination}")

    if origin is None or destination is None:
        output_rows.append([sentence_id, 'INVALID', 'INVALID'])
    else:
        output_rows.append([sentence_id, origin, destination])

# Sauvegarde en garantissant 3 colonnes
output_df = pd.DataFrame(output_rows, columns=['sentenceID', 'Departure', 'Destination'])
output_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
print(f"Traitement terminé ! Résultat sauvegardé dans {OUTPUT_CSV}")
