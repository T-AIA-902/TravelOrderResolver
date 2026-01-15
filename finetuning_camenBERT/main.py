import sys
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Répertoire de base du script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from transformers import pipeline
from pathfinder import TrainGraph

try:
    from visualizer import MapVisualizer
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("⚠️ 'visualizer.py' manquant. Cartes désactivées.", file=sys.stderr)


MODEL_PATH = os.path.join(BASE_DIR, "saved_model")
CSV_GARES = os.path.join(BASE_DIR, "gares-de-voyageurs.csv")
CSV_LIGNES = os.path.join(BASE_DIR, "lignes-par-type.csv")


class TravelOrderResolver:
    def __init__(self):
        print("🧠 Chargement du modèle NLP...", file=sys.stderr)
        try:
            self.nlp = pipeline("ner", model=MODEL_PATH, aggregation_strategy="simple")
        except Exception as e:
            print(f"ERREUR FATALE: Modèle introuvable dans {MODEL_PATH}.", file=sys.stderr)
            print(f"Détail: {e}", file=sys.stderr)
            sys.exit(1)

        self.graph_engine = TrainGraph(CSV_GARES, CSV_LIGNES)

        print("🚀 Système prêt !", file=sys.stderr)

    def extract_entities(self, text):
        results = self.nlp(text)
        dep, dest = None, None
        for entity in results:
            label = entity["entity_group"]
            word = entity["word"]
            if "DEP" in label:
                dep = word
            elif "DEST" in label:
                dest = word
        return dep, dest

    def process_request(self, sentence_id, text):
        dep, dest = self.extract_entities(text)

        if not dep or not dest:
            print(f"{sentence_id}, INVALID")
            return

        dep = dep.strip()
        dest = dest.strip()

        names_list, error, full_uics = self.graph_engine.get_path(dep, dest)

        if error:
            print(f"{sentence_id}, INVALID")
        else:
            route_str = ", ".join(names_list)
            print(f"{sentence_id}, {route_str}")

            if VISUALIZATION_AVAILABLE and full_uics:
                try:
                    viz = MapVisualizer(self.graph_engine)
                    viz.draw_path(full_uics, filename=f"trajet_{sentence_id}.html")
                except Exception as e:
                    print(f"Erreur Visualisation: {e}", file=sys.stderr)


if __name__ == "__main__":
    resolver = TravelOrderResolver()
    
    print("\n👉 MENU COMMANDES :")
    print("   Écrire : '1, Je vais à Paris' (ou juste 'Je vais à Paris')")
    print("   (Ctrl+C pour quitter)\n")

    for line in sys.stdin:
        line = line.strip()
        if not line: continue

        try:
            if "," in line:
                parts = line.split(",", 1)
                s_id = parts[0].strip()
                text = parts[1].strip()
            else:
                s_id, text = "AUTO", line

            resolver.process_request(s_id, text)
            sys.stdout.flush()

        except Exception as e:
            print(f"Erreur traitement: {e}", file=sys.stderr)