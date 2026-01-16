"""
Script amélioré pour extraire origine et destination depuis les phrases de voyage.
Ajouts récents:
- classification simple travel / not-travel (is_travel_sentence)
- statut (Travel) ajouté au CSV de sortie
- affichage des résultats sous forme de tableau lisible dans le terminal
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

# Termes liés aux voyages (pour la classification)
TRAVEL_KEYWORDS = {
    'aller', 'partir', 'prendre', 'arriver', 'départ', 'destination', 'billet',
    'train', 'tgv', 'avion', 'vol', 'bus', 'horaire', 'réserver', 'réservation',
    'gare', 'terminus', 'correspondance', 'aller', 'venir', 'rendre', 'trajet'
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

# Classifier si la phrase correspond à un voyage
def is_travel_sentence(sentence: str, detected_cities: list) -> bool:
    s_norm = normalize_text(sentence)
    # si au moins deux villes détectées -> vraisemblablement une requête de voyage
    if len(detected_cities) >= 2:
        return True
    # rechercher mots-clés liés aux voyages
    for kw in TRAVEL_KEYWORDS:
        if re.search(rf"\b{re.escape(kw)}\b", s_norm):
            return True
    # si le texte contient 'à' ou 'de' suivi d'un mot ressemblant à une ville (heuristique)
    if re.search(r"\b(de|depuis|à|a|vers|pour)\b", s_norm) and len(detected_cities) >= 1:
        return True
    return False

# Calculer un score de confiance (0-100)
def compute_confidence(sentence: str, cities: list, origin: str, destination: str) -> float:
    """
    cities: list of tuples (orig, matched_ngram, score)
    origin/destination: chosen city names or None
    Logique simple:
    - Si deux villes extraites et origin/destination trouvés: moyenne des scores fuzzy pour origin & destination
    - Sinon si au moins une ville: moyenne des scores fuzzy pour les villes détectées
    - Additionner un petit bonus si des mots-clés de voyage sont présents
    - Bonus de complétude si extraction complète
    - Capiage 0-100
    """
    s_norm = normalize_text(sentence)
    # construire mapping city -> score
    scores = {}
    for item in cities:
        if isinstance(item, tuple) and len(item) >= 3:
            city_name = item[0]
            score = float(item[2])
            scores[city_name] = max(scores.get(city_name, 0.0), score)
    # cas: aucune ville détectée
    if not scores:
        base = 10.0 if any(re.search(rf"\b{re.escape(kw)}\b", s_norm) for kw in TRAVEL_KEYWORDS) else 0.0
        return round(min(100.0, base), 1)
    # si origin & destination présents
    if origin and destination and origin in scores and destination in scores:
        base = (scores.get(origin, 0.0) + scores.get(destination, 0.0)) / 2.0
    else:
        # moyenne des scores détectés
        base = sum(scores.values()) / len(scores.values())
    # bonus mots-clés
    keyword_bonus = 5.0 if any(re.search(rf"\b{re.escape(kw)}\b", s_norm) for kw in TRAVEL_KEYWORDS) else 0.0
    # bonus complétude
    completeness_bonus = 10.0 if (origin and destination) else 0.0
    confidence = base + keyword_bonus + completeness_bonus
    confidence = max(0.0, min(100.0, confidence))
    return round(confidence, 1)

# Traitement
output_rows = []
for _, row in sentences_df.iterrows():
    sentence_id = row.get('sentenceID')
    sentence = str(row.get('sentence', ''))

    cities = extract_cities(sentence, threshold=80)
    # cities is list of tuples (orig, ngram, score)
    city_names = [c[0] if isinstance(c, tuple) else c for c in cities]

    travel_flag = is_travel_sentence(sentence, city_names)

    origin, destination = get_origin_destination(sentence, cities)

    extraction_ok = origin is not None and destination is not None

    # Déterminer statut final
    if not travel_flag:
        status = 'NOT_TRAVEL'
        dep = origin if origin is not None else 'INVALID'
        dest = destination if destination is not None else 'INVALID'
    else:
        if not extraction_ok:
            status = 'INVALID_EXTRACTION'
            dep = origin if origin is not None else 'INVALID'
            dest = destination if destination is not None else 'INVALID'
        else:
            status = 'VALID'
            dep = origin
            dest = destination

    # calculer confiance
    confidence = compute_confidence(sentence, cities, origin, destination)

    output_rows.append([sentence_id, sentence, dep, dest, status, confidence])

# Sauvegarde en garantissant les colonnes (sentenceID, sentence, Departure, Destination, Travel, Confidence)
output_df = pd.DataFrame(output_rows, columns=['sentenceID', 'sentence', 'Departure', 'Destination', 'Travel', 'Confidence'])
# Écrire une version simplifiée pour compatibilité (sans le texte de la phrase) et garder le statut + confidence
output_df[['sentenceID', 'Departure', 'Destination', 'Travel', 'Confidence']].to_csv(OUTPUT_CSV, index=False, encoding='utf-8')

# Afficher les résultats dans le terminal sous forme de tableau lisible
# On affiche un aperçu en tronquant éventuellement la phrase si trop longue
DISPLAY_MAX_SENTENCE = 80
display_df = output_df.copy()
if display_df['sentence'].apply(len).max() > DISPLAY_MAX_SENTENCE:
    display_df['sentence'] = display_df['sentence'].apply(lambda s: (s[:DISPLAY_MAX_SENTENCE-3] + '...') if len(s) > DISPLAY_MAX_SENTENCE else s)

print('\nRésultats du traitement :')
print(display_df.to_string(index=False))
print(f"\nTraitement terminé ! Résultat sauvegardé dans {OUTPUT_CSV}")
