import pandas as pd
import random
import unicodedata
try:
    from faker import Faker
except ImportError:
    Faker = None

INPUT_CSV = "gares-de-voyageurs.csv"
OUTPUT_CSV = "dataset_train_sncf.csv"
NUM_SAMPLES = 100000
RATIO_INVALID = 0.20

fake = Faker('fr_FR') if Faker else None

def load_stations(filename):
    print(f"📂 Chargement des gares depuis {filename}...")
    try:
        df = pd.read_csv(filename, sep=";", on_bad_lines='skip', encoding='utf-8')

        if 'Nom' not in df.columns:
            df = pd.read_csv(filename, sep=";", encoding='latin-1')

        if 'Nom' not in df.columns:
            raise ValueError("Colonne 'Nom' introuvable.")

        stations = df['Nom'].dropna().unique().tolist()
        stations = [s for s in stations if len(s) > 2]
        
        print(f"✅ {len(stations)} gares chargées avec succès.")
        return stations

    except Exception as e:
        print(f"⚠️ Erreur lecture CSV ({e}). Utilisation mode secours.")
        return ["Paris", "Lyon", "Marseille", "Bordeaux", "Lille", "Toulouse", "Albert"]

synonyms = {
    "intent_polite": [
        "Je voudrais", "J'aimerais", "Je souhaite", "Pourriez-vous me trouver", 
        "Je veux", "Il me faut", "J'ai besoin de", "Recherche", "Je cherche"
    ],
    "intent_imperative": [
        "Trouve-moi", "Cherche", "Donne-moi", "Affiche", "Sors-moi", "Go pour"
    ],
    "intent_question": [
        "Y a-t-il", "Est-ce qu'il y a", "Quels sont les", "Existe-t-il", "A quelle heure part"
    ],
    "noun_trip": [
        "un train", "un billet", "un trajet", "un voyage", "une place", 
        "un aller", "un aller-simple", "une liaison", "un TGV", "un TER"
    ],
    "prep_from": [
        "de", "depuis", "au départ de", "en partant de", "partant de", "venant de", "du côté de"
    ],
    "prep_to": [
        "à", "pour", "vers", "à destination de", "jusqu'à", "en direction de", "au"
    ],
    "time_adverb": [
        "maintenant", "demain", "tout de suite", "plus tard", "ce soir", "ce matin", "lundi"
    ]
}

def generate_valid_sentence(dep, dest):
    structures = [
        f"{random.choice(synonyms['intent_polite'])} {random.choice(synonyms['noun_trip'])} {random.choice(synonyms['prep_from'])} {dep} {random.choice(synonyms['prep_to'])} {dest}",
        f"{random.choice(synonyms['noun_trip'])} {random.choice(synonyms['prep_to'])} {dest} {random.choice(synonyms['prep_from'])} {dep}",
        f"{random.choice(synonyms['intent_question'])} {random.choice(synonyms['noun_trip'])} {random.choice(synonyms['prep_to'])} {dest} ?",
        f"{random.choice(synonyms['intent_imperative'])} {random.choice(synonyms['noun_trip'])} entre {dep} et {dest}",
        f"train {dep} {dest}",
        f"billet {dest} depuis {dep}",
        f"{dep} - {dest}",
        f"Je dois aller à {dest} pour une réunion, je pars de {dep}",
        f"Réserve moi une place pour {dest} en provenance de {dep}"
    ]
    return random.choice(structures)

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
        f"Je cherche mon ami {first_name}"
    ]
    return random.choice(traps)

def apply_noise(text):
    text_list = list(text)

    if random.random() < 0.5:
        text = text.lower()
        text_list = list(text)

    if random.random() < 0.3:
        text = ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
        text_list = list(text)

    if random.random() < 0.1 and len(text) > 5:
        idx = random.randint(0, len(text) - 2)
        text_list[idx], text_list[idx+1] = text_list[idx+1], text_list[idx]

    return "".join(text_list)

def main():
    cities = load_stations(INPUT_CSV)
    data = []

    print(f"🚀 Génération de {NUM_SAMPLES} phrases...")

    for i in range(NUM_SAMPLES):
        is_valid = random.random() > RATIO_INVALID

        if is_valid:
            dep = random.choice(cities)
            dest = random.choice(cities)
            while dep == dest:
                dest = random.choice(cities)

            raw_text = generate_valid_sentence(dep, dest)
            final_text = apply_noise(raw_text)

            data.append({
                "id": i,
                "text": final_text,
                "label": "VALID",
                "departure": dep,
                "destination": dest
            })

        else:
            raw_text = generate_invalid_sentence(cities)
            final_text = apply_noise(raw_text)

            data.append({
                "id": i,
                "text": final_text,
                "label": "INVALID",
                "departure": None,
                "destination": None
            })

    df_output = pd.DataFrame(data)
    df_output.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
    
    print(f"✨ Terminé ! Fichier généré : {OUTPUT_CSV}")
    print(f"📊 Statistiques : {len(df_output[df_output['label']=='VALID'])} valides / {len(df_output[df_output['label']=='INVALID'])} invalides.")
    print(df_output.head(10))

if __name__ == "__main__":
    main()