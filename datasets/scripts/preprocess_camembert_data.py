import json

import pandas as pd
from sklearn.model_selection import train_test_split

INPUT_CSV = "dataset_train_sncf.csv"
OUTPUT_TRAIN = "train.jsonl"
OUTPUT_VAL = "validation.jsonl"

LABEL_MAP = {"O": 0, "B-DEP": 1, "I-DEP": 2, "B-DEST": 3, "I-DEST": 4}


def find_sublist(sublist, main_list):
    if not sublist:
        return -1
    len_sub = len(sublist)
    for i in range(len(main_list) - len_sub + 1):
        if main_list[i : i + len_sub] == sublist:  # noqa: E203
            return i
    return -1


def process_data(input_file):
    print("⏳ Chargement du CSV et tokenization...")
    df = pd.read_csv(input_file)

    formatted_data = []

    for _, row in df.iterrows():
        text = str(row["text"])
        tokens = text.split()

        ner_tags = [0] * len(tokens)

        if row["label"] == "INVALID":
            formatted_data.append({"tokens": tokens, "ner_tags": ner_tags})
            continue

        dep = str(row["departure"]) if pd.notna(row["departure"]) else ""
        dest = str(row["destination"]) if pd.notna(row["destination"]) else ""

        dep_tokens = dep.split()
        dest_tokens = dest.split()

        tokens_lower = [t.lower() for t in tokens]
        dep_lower = [t.lower() for t in dep_tokens]
        dest_lower = [t.lower() for t in dest_tokens]

        start_dep = find_sublist(dep_lower, tokens_lower)
        if start_dep != -1:
            ner_tags[start_dep] = LABEL_MAP["B-DEP"]
            for i in range(1, len(dep_tokens)):
                if start_dep + i < len(ner_tags):
                    ner_tags[start_dep + i] = LABEL_MAP["I-DEP"]

        start_dest = find_sublist(dest_lower, tokens_lower)
        if start_dest != -1:
            ner_tags[start_dest] = LABEL_MAP["B-DEST"]
            for i in range(1, len(dest_tokens)):
                if start_dest + i < len(ner_tags):
                    ner_tags[start_dest + i] = LABEL_MAP["I-DEST"]

        formatted_data.append({"id": row["id"], "tokens": tokens, "ner_tags": ner_tags})

    return formatted_data


data = process_data(INPUT_CSV)

train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
print(f"Sauvegarde de {len(train_data)} exemples d'entraînement...")
with open(OUTPUT_TRAIN, "w", encoding="utf-8") as f:
    for entry in train_data:
        json.dump(entry, f)
        f.write("\n")

print(f"Sauvegarde de {len(val_data)} exemples de validation...")
with open(OUTPUT_VAL, "w", encoding="utf-8") as f:
    for entry in val_data:
        json.dump(entry, f)
        f.write("\n")

print("✅ Terminé ! Prêt pour le fine-tuning.")
