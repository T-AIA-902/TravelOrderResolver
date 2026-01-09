# travel_order_resolver_native.py

import pandas as pd
from transformers import CamembertTokenizer, CamembertModel
from rapidfuzz import process

# Charger les fichiers à traiter

sentences_df = pd.read_csv("datasets/raw/sentences/travel_sentences.csv",encoding="latin-1")  # phrases de voyage
stations_df = pd.read_csv("datasets/raw/sncf/gares.csv",encoding="latin-1")  # liste des gares / villes

stations_list = stations_df['city_name'].tolist()

# Charger CamemBERT natif

tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
model = CamembertModel.from_pretrained("camembert-base")


# Fonctions utilitaires pour extraire les villes et identifier origine/destination


def extract_cities(sentence, stations_list, threshold=80):
    """
    Trouver les villes/gares présentes dans une phrase via fuzzy matching.
    """
    # Nettoyer la phrase
    words = sentence.replace(',', '').replace('?', '').replace('.', '').split()
    found_cities = []

    for word in words:
        match, score, _ = process.extractOne(word, stations_list)
        if score >= threshold:
            found_cities.append(match)

    return list(set(found_cities))  # retirer doublons


def get_origin_destination(sentence, cities):
    """
    Identifier l'origine et la destination en utilisant les mots-clés
    'de', 'depuis', 'à', 'vers'. Si non trouvé, assigner simplement
    la première et la deuxième ville détectées.
    """
    sentence_lower = sentence.lower()
    origin, destination = None, None

    for city in cities:
        city_lower = city.lower()
        if f"de {city_lower}" in sentence_lower or f"depuis {city_lower}" in sentence_lower:
            origin = city
        elif f"à {city_lower}" in sentence_lower or f"vers {city_lower}" in sentence_lower:
            destination = city

    # Si toujours non défini mais deux villes détectées
    if len(cities) == 2 and (origin is None or destination is None):
        origin, destination = cities[0], cities[1]

    return origin, destination


# Traitement des phrases

output_rows = []

for idx, row in sentences_df.iterrows():
    sentence_id = row['sentenceID']
    sentence = row['sentence']

    # Extraire les villes
    cities = extract_cities(sentence, stations_list)

    if not cities:
        # Phrase invalide
        output_rows.append([sentence_id, "INVALID"])
        continue

    # Identifier origine et destination
    origin, destination = get_origin_destination(sentence, cities)

    if origin is None or destination is None:
        output_rows.append([sentence_id, "INVALID"])
    else:
        output_rows.append([sentence_id, origin, destination])

# -----------------------------
# 5. Sauvegarde CSV final
# -----------------------------
output_df = pd.DataFrame(output_rows, columns=['sentenceID', 'Departure', 'Destination'])
output_df.to_csv("travel_orders_output.csv", index=False, encoding='utf-8')

print("Traitement terminé ! Résultat sauvegardé dans travel_orders_output.csv")
