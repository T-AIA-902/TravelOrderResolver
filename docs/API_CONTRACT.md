# Contrat API -- Travel Order Resolver

> **Version** : 1.0.0
> **Date** : 2026-02-20
> **Base URL** : `http://localhost:8000`
> **Framework** : FastAPI (Python 3.11+)

Ce document decrit l'ensemble des endpoints REST que le frontend Vue 3 appelle.
Le developpeur backend doit implementer chaque route en s'appuyant sur le code Python existant.

---

## Table des matieres

1. [Architecture generale](#1-architecture-generale)
2. [Conventions](#2-conventions)
3. [Endpoints](#3-endpoints)
   - 3.1 [Health Check](#31-get-apihealth)
   - 3.2 [Resolution complete](#32-post-apiresolve)
   - 3.3 [NLP -- Detection de langue](#33-post-apinlplanguage)
   - 3.4 [NLP -- Classification d'intent](#34-post-apinlpintent)
   - 3.5 [NLP -- Extraction d'entites](#35-post-apinlpentities)
   - 3.6 [Pathfinding -- Recherche de gares](#36-get-apipathfindingstations)
   - 3.7 [Pathfinding -- Calcul d'itineraire](#37-post-apipathfindingroute)
   - 3.8 [Speech -- Transcription audio](#38-post-apispeechtranscribe)
   - 3.9 [Evaluation -- Liste des rapports](#39-get-apievaluationreports)
   - 3.10 [Evaluation -- Detail d'un rapport](#310-get-apievaluationreportsid)
   - 3.11 [Evaluation -- Lancer une evaluation](#311-post-apievaluationrun)
   - 3.12 [Evaluation -- Statut d'une tache](#312-get-apievaluationstatustaskid)
   - 3.13 [Monitoring -- Metriques de requetes](#313-get-apimonitoringmetrics)
   - 3.14 [Monitoring -- Ressources systeme](#314-get-apimonitoringresources)
   - 3.15 [Monitoring -- Empreinte carbone](#315-get-apimonitoringcarbon)
4. [Mapping code Python existant](#4-mapping-code-python-existant)
5. [Types de reference](#5-types-de-reference)
6. [Notes d'implementation](#6-notes-dimplementation)

---

## 1. Architecture generale

```
Frontend (Vue 3)  --HTTP JSON-->  FastAPI  --appels Python-->  Modules existants
                                    |
                                    +-- src/nlp/pipeline.py        (NLPPipeline)
                                    +-- src/pathfinding/graph.py   (TrainGraph)
                                    +-- src/speech/whisper_model.py (WhisperModel)
                                    +-- src/evaluation/cli.py      (Evaluateurs)
                                    +-- src/monitoring/             (MetricsLogger, ResourceTracker, CarbonCalculator)
```

Le serveur FastAPI doit etre cree dans `src/api/` (nouveau dossier). Il agit comme une **couche mince** : il recoit les requetes HTTP, appelle le code Python existant, et formate les reponses au format attendu par le frontend.

---

## 2. Conventions

| Element             | Convention                                              |
| ------------------- | ------------------------------------------------------- |
| Format des corps    | `application/json` (sauf `/api/speech/transcribe`)      |
| Encodage            | UTF-8                                                   |
| Codes HTTP succes   | `200 OK`                                                |
| Codes HTTP erreur   | `400` (requete invalide), `404` (non trouve), `500` (erreur interne) |
| Erreurs             | `{ "detail": "message d'erreur" }`                      |
| CORS                | Autoriser `http://localhost:5173` (Vite dev server)     |
| Prefixe             | Toutes les routes commencent par `/api/`                |
| Champs optionnels   | Suffixe `?` dans la doc = champ non obligatoire         |

---

## 3. Endpoints

---

### 3.1 GET /api/health

**Description** : Verification de l'etat du serveur et des modeles charges. Appelee par le frontend au demarrage pour afficher le statut dans le dashboard.

**Parametres** : Aucun

**Reponse** (`200 OK`) :

```json
{
  "status": "ok",
  "version": "1.0.0",
  "models_loaded": ["regex", "spacy", "camembert", "flant5"],
  "stations_count": 2300,
  "graph_nodes": 2300,
  "graph_edges": 5000,
  "device": "cpu",
  "gpu_name": null
}
```

| Champ            | Type             | Description                                              |
| ---------------- | ---------------- | -------------------------------------------------------- |
| `status`         | `string`         | Toujours `"ok"` si le serveur repond                     |
| `version`        | `string`         | Version de l'API                                         |
| `models_loaded`  | `string[]`       | Liste des modeles NLP disponibles                        |
| `stations_count` | `number`         | Nombre de gares dans `TrainGraph.stations`               |
| `graph_nodes`    | `number`         | `TrainGraph.graph.number_of_nodes()`                     |
| `graph_edges`    | `number`         | `TrainGraph.graph.number_of_edges()`                     |
| `device`         | `string`         | `"cpu"` ou `"cuda"` -- resultat de `get_torch_device()`  |
| `gpu_name`       | `string \| null` | Nom du GPU si disponible, sinon `null`                   |

**Code Python a utiliser** :
```python
from src.utils.device import get_torch_device, get_device_info
# Agreger les infos de NLPPipeline (modeles charges) + TrainGraph (graph stats)
# + get_device_info() pour device/gpu_name
```

---

### 3.2 POST /api/resolve

**Description** : Endpoint principal. Prend une phrase en langage naturel, effectue l'analyse NLP complete (langue, intent, entites), puis calcule l'itineraire ferroviaire si un trajet est detecte.

**Corps de la requete** :

```json
{
  "text": "Je veux aller de Paris a Lyon en passant par Marseille",
  "intent_model": "camembert",
  "entity_model": "spacy",
  "use_fuzzy": true
}
```

| Champ           | Type      | Requis | Description                                                    |
| --------------- | --------- | ------ | -------------------------------------------------------------- |
| `text`          | `string`  | Oui    | Texte en langage naturel a analyser                            |
| `intent_model`  | `string`  | Non    | Modele d'intent : `"regex"`, `"camembert"`, `"flant5"`         |
| `entity_model`  | `string`  | Non    | Modele d'entites : `"regex"`, `"spacy"`, `"camembert"`, `"flant5"` |
| `use_fuzzy`     | `boolean` | Non    | Activer le fuzzy matching sur les noms de gares (defaut: `true`) |

**Reponse** (`200 OK`) :

```json
{
  "nlp": {
    "language": {
      "detected": "FRENCH",
      "confidence": 0.98,
      "model": "regex"
    },
    "intent": {
      "value": "TRIP",
      "confidence": 0.95,
      "model": "camembert"
    },
    "entities": {
      "departure": {
        "raw": "Paris",
        "matched": "Paris Gare de Lyon",
        "confidence": 0.92
      },
      "destination": {
        "raw": "Lyon",
        "matched": "Lyon Part-Dieu",
        "confidence": 0.88
      },
      "intermediates": [],
      "model": "spacy",
      "fuzzy_enabled": true
    },
    "processed_text": "je veux aller de paris a lyon",
    "latency_ms": 150
  },
  "pathfinding": {
    "found": true,
    "route": ["Paris Gare de Lyon", "Lyon Part-Dieu"],
    "route_details": [
      {
        "from_station": "Paris Gare de Lyon",
        "from_uic": "87686006",
        "to_station": "Lyon Part-Dieu",
        "to_uic": "87723197",
        "line": "830000",
        "type": "TRAIN",
        "geometry": [[48.8449, 2.3735], [45.7606, 4.8599]]
      }
    ],
    "total_stops": 2,
    "transfers": 0,
    "error": null
  }
}
```

**Sous-objet `nlp`** :

| Champ                          | Type             | Description                                           |
| ------------------------------ | ---------------- | ----------------------------------------------------- |
| `language.detected`            | `string`         | `"FRENCH"`, `"ENGLISH"`, `"UNKNOWN"`                  |
| `language.confidence`          | `number`         | Score de confiance (0.0 a 1.0)                        |
| `language.model`               | `string`         | Nom du modele utilise                                 |
| `intent.value`                 | `string`         | `"TRIP"`, `"NOT_TRIP"`, `"UNKNOWN"`                   |
| `intent.confidence`            | `number`         | Score de confiance (0.0 a 1.0)                        |
| `intent.model`                 | `string`         | Nom du modele utilise                                 |
| `entities.departure.raw`       | `string`         | Texte brut extrait pour le depart                     |
| `entities.departure.matched`   | `string`         | Nom de gare apres matching                            |
| `entities.departure.confidence`| `number`         | Score de confiance                                    |
| `entities.destination.raw`     | `string`         | Texte brut extrait pour la destination                |
| `entities.destination.matched` | `string`         | Nom de gare apres matching                            |
| `entities.destination.confidence`| `number`       | Score de confiance                                    |
| `entities.intermediates`       | `array`          | Liste des arrets intermediaires (meme format)         |
| `entities.model`               | `string`         | Nom du modele d'extraction utilise                    |
| `entities.fuzzy_enabled`       | `boolean`        | Si le fuzzy matching etait actif                      |
| `processed_text`               | `string`         | Texte apres preprocessing                             |
| `latency_ms`                   | `number`         | Temps d'execution NLP en millisecondes                |

**Sous-objet `pathfinding`** :

| Champ                             | Type             | Description                                        |
| --------------------------------- | ---------------- | -------------------------------------------------- |
| `found`                           | `boolean`        | `true` si un itineraire a ete trouve               |
| `route`                           | `string[]`       | Liste simplifiee des noms de gares                 |
| `route_details`                   | `RouteSegment[]` | Details par segment (voir ci-dessous)              |
| `total_stops`                     | `number`         | Nombre total d'arrets dans `route`                 |
| `transfers`                       | `number`         | Nombre de correspondances (changements de ligne)   |
| `error`                           | `string \| null` | Message d'erreur si pas de trajet trouve           |

**Type `RouteSegment`** :

| Champ          | Type           | Description                                           |
| -------------- | -------------- | ----------------------------------------------------- |
| `from_station` | `string`       | Nom de la gare de depart du segment                   |
| `from_uic`     | `string`       | Code UIC de la gare de depart                         |
| `to_station`   | `string`       | Nom de la gare d'arrivee du segment                   |
| `to_uic`       | `string`       | Code UIC de la gare d'arrivee                         |
| `line`         | `string`       | Code de la ligne (`"830000"`, `"TRANSFERT"`, etc.)    |
| `type`         | `string`       | `"TRAIN"` ou `"WALK"` (correspondance a pied)        |
| `geometry`     | `[number, number][]` | Coordonnees `[lat, lon]` du trace de la ligne   |

**Code Python a utiliser** :

```python
# 1. Creer le pipeline NLP avec les modeles demandes
pipeline = NLPPipeline(
    language_detector=...,    # selon intent_model
    intent_classifier=...,    # selon intent_model (regex/camembert/flant5)
    entity_extractor=...,     # selon entity_model (regex/spacy/camembert/flant5)
    station_db=station_db,    # si use_fuzzy=True
)

# 2. Traiter le texte
result: PredictionResult = pipeline.process(text)

# 3. Si intent == TRIP et departure + destination trouves :
simplified, error, full_uic_path = graph.get_path(
    result.departure, result.destination, result.intermediates
)

# 4. Construire route_details a partir de full_uic_path :
route_details = []
for i in range(len(full_uic_path) - 1):
    u, v = full_uic_path[i], full_uic_path[i + 1]
    edge_data = graph.graph.get_edge_data(u, v)[0]  # IMPORTANT: [0] car MultiGraph
    route_details.append({
        "from_station": graph.graph.nodes[u]["name"],
        "from_uic": u,
        "to_station": graph.graph.nodes[v]["name"],
        "to_uic": v,
        "line": edge_data.get("line", "UNKNOWN"),
        "type": edge_data.get("type", "TRAIN"),
        "geometry": edge_data.get("geometry", [
            list(graph.graph.nodes[u]["pos"]),
            list(graph.graph.nodes[v]["pos"]),
        ]),
    })
```

> **ATTENTION** : `graph.get_edge_data(u, v)` retourne un dictionnaire indexe par cle d'arete (c'est un `MultiGraph`). Il faut utiliser `[0]` pour obtenir la premiere arete. Les donnees contiennent les champs `geometry` (liste de `[lat, lon]`), `line` (code ligne), et `type` (`"TRAIN"` ou `"WALK"`).

---

### 3.3 POST /api/nlp/language

**Description** : Execute tous les detecteurs de langue disponibles sur le texte donne. Utilise pour la page de debug/comparaison des modeles.

**Corps de la requete** :

```json
{
  "text": "Je veux aller de Paris a Lyon"
}
```

| Champ  | Type     | Requis | Description        |
| ------ | -------- | ------ | ------------------ |
| `text` | `string` | Oui    | Texte a analyser   |

**Reponse** (`200 OK`) :

```json
{
  "results": [
    {
      "model": "regex",
      "detected": "FRENCH",
      "confidence": 0.98,
      "latency_ms": 1
    },
    {
      "model": "langdetect",
      "detected": "FRENCH",
      "confidence": 0.95,
      "latency_ms": 3
    }
  ]
}
```

| Champ              | Type     | Description                                 |
| ------------------ | -------- | ------------------------------------------- |
| `results[].model`  | `string` | Nom du detecteur (`"regex"`, `"langdetect"`) |
| `results[].detected` | `string` | Langue detectee (`"FRENCH"`, `"ENGLISH"`, `"UNKNOWN"`) |
| `results[].confidence` | `number` | Score de confiance (0.0 a 1.0)          |
| `results[].latency_ms` | `number` | Temps d'execution en ms                 |

**Code Python a utiliser** :

```python
from src.nlp.language import RegexLanguageDetector
from src.nlp.language.langdetect_language import LangdetectLanguageDetector

detectors = [RegexLanguageDetector(), LangdetectLanguageDetector()]
results = []
for det in detectors:
    start = time.time()
    lang, conf = det.detect(text)
    latency = (time.time() - start) * 1000
    results.append({
        "model": det.name,
        "detected": lang,
        "confidence": conf,
        "latency_ms": round(latency, 1),
    })
```

---

### 3.4 POST /api/nlp/intent

**Description** : Execute tous les classifieurs d'intent disponibles sur le texte donne.

**Corps de la requete** :

```json
{
  "text": "Je veux aller de Paris a Lyon"
}
```

| Champ  | Type     | Requis | Description        |
| ------ | -------- | ------ | ------------------ |
| `text` | `string` | Oui    | Texte a analyser   |

**Reponse** (`200 OK`) :

```json
{
  "results": [
    {
      "model": "regex",
      "intent": "TRIP",
      "confidence": 0.9,
      "latency_ms": 5
    },
    {
      "model": "camembert",
      "intent": "TRIP",
      "confidence": 0.97,
      "latency_ms": 120
    },
    {
      "model": "flant5",
      "intent": "TRIP",
      "confidence": 0.93,
      "latency_ms": 200
    }
  ]
}
```

| Champ               | Type     | Description                                           |
| -------------------- | -------- | ----------------------------------------------------- |
| `results[].model`    | `string` | Nom du classifieur (`"regex"`, `"camembert"`, `"flant5"`) |
| `results[].intent`   | `string` | Intent detecte (`"TRIP"`, `"NOT_TRIP"`, `"UNKNOWN"`)  |
| `results[].confidence` | `number` | Score de confiance (0.0 a 1.0)                     |
| `results[].latency_ms` | `number` | Temps d'execution en ms                            |

**Code Python a utiliser** :

```python
from src.nlp.intent import RegexIntentClassifier
from src.nlp.intent.camembert_intent import CamembertIntentClassifier
from src.nlp.intent.flant5_intent import FlanT5IntentClassifier
# Utiliser src/evaluation/model_factory.py :: create_intent_classifiers(["all"])
# pour instancier tous les modeles disponibles.

classifiers = [RegexIntentClassifier(), CamembertIntentClassifier(), ...]
for clf in classifiers:
    start = time.time()
    intent, conf = clf.classify(text)
    latency = (time.time() - start) * 1000
    results.append({
        "model": clf.name,
        "intent": intent,
        "confidence": conf,
        "latency_ms": round(latency, 1),
    })
```

---

### 3.5 POST /api/nlp/entities

**Description** : Execute tous les extracteurs d'entites disponibles sur le texte donne.

**Corps de la requete** :

```json
{
  "text": "Je veux aller de Paris a Lyon"
}
```

| Champ  | Type     | Requis | Description        |
| ------ | -------- | ------ | ------------------ |
| `text` | `string` | Oui    | Texte a analyser   |

**Reponse** (`200 OK`) :

```json
{
  "results": [
    {
      "model": "spacy",
      "fuzzy": true,
      "departure": {
        "raw": "Paris",
        "matched": "Paris Gare de Lyon",
        "confidence": 0.9
      },
      "destination": {
        "raw": "Lyon",
        "matched": "Lyon Part-Dieu",
        "confidence": 0.88
      },
      "intermediates": [],
      "latency_ms": 50
    },
    {
      "model": "regex",
      "fuzzy": true,
      "departure": {
        "raw": "Paris",
        "matched": "Paris Gare de Lyon",
        "confidence": 0.85
      },
      "destination": {
        "raw": "Lyon",
        "matched": "Lyon Part-Dieu",
        "confidence": 0.85
      },
      "intermediates": [],
      "latency_ms": 5
    }
  ]
}
```

| Champ                        | Type      | Description                                         |
| ---------------------------- | --------- | --------------------------------------------------- |
| `results[].model`            | `string`  | Nom de l'extracteur (`"regex"`, `"spacy"`, `"camembert"`, `"flant5"`) |
| `results[].fuzzy`            | `boolean` | Si le fuzzy matching a ete applique                 |
| `results[].departure.raw`    | `string`  | Texte brut extrait                                  |
| `results[].departure.matched`| `string`  | Nom de gare apres matching                          |
| `results[].departure.confidence` | `number` | Score de confiance                               |
| `results[].destination.raw`  | `string`  | Texte brut extrait                                  |
| `results[].destination.matched` | `string` | Nom de gare apres matching                        |
| `results[].destination.confidence` | `number` | Score de confiance                             |
| `results[].intermediates`    | `array`   | Liste d'arrets intermediaires (meme format)         |
| `results[].latency_ms`       | `number`  | Temps d'execution en ms                             |

**Code Python a utiliser** :

```python
from src.nlp.entity import RegexEntityExtractor
from src.nlp.entity.spacy_entity import SpacyEntityExtractor
from src.nlp.entity.camembert_entity import CamembertEntityExtractor
from src.nlp.entity.flant5_entity import FlanT5EntityExtractor
# Utiliser src/evaluation/model_factory.py :: create_entity_extractors(["all"])

extractors = [RegexEntityExtractor(), SpacyEntityExtractor(), ...]
for ext in extractors:
    start = time.time()
    entities = ext.extract(text)
    latency = (time.time() - start) * 1000
    # entities retourne: {"departure": str|None, "destination": str|None, "intermediate": list[str]}
    # Appliquer fuzzy matching via StationDatabase.search_by_name() si necessaire
    results.append({
        "model": ext.name,
        "fuzzy": True,  # ou False selon config
        "departure": {"raw": entities.get("departure", ""), "matched": ..., "confidence": ...},
        "destination": {"raw": entities.get("destination", ""), "matched": ..., "confidence": ...},
        "intermediates": [...],
        "latency_ms": round(latency, 1),
    })
```

> **Note** : Le champ `confidence` par entite n'est pas retourne nativement par tous les extracteurs. Pour les extracteurs qui ne fournissent pas de confidence par entite (regex, spacy), utiliser `1.0` si l'entite est trouvee, `0.0` sinon. Les modeles CamemBERT et FlanT5 peuvent fournir un score via `TravelEntity.confidence`.

---

### 3.6 GET /api/pathfinding/stations

**Description** : Recherche de gares par nom pour l'autocompletion dans le frontend.

**Parametres de requete (query string)** :

| Parametre | Type     | Requis | Description                        |
| --------- | -------- | ------ | ---------------------------------- |
| `q`       | `string` | Oui    | Terme de recherche                 |
| `limit`   | `number` | Non    | Nombre max de resultats (defaut: 10) |

**Exemple** : `GET /api/pathfinding/stations?q=paris&limit=10`

**Reponse** (`200 OK`) :

```json
{
  "stations": [
    {
      "name": "Paris Gare de Lyon",
      "uic": "87686006",
      "lat": 48.8449,
      "lon": 2.3735
    },
    {
      "name": "Paris Montparnasse",
      "uic": "87391003",
      "lat": 48.8414,
      "lon": 2.3187
    }
  ]
}
```

| Champ              | Type     | Description                      |
| ------------------ | -------- | -------------------------------- |
| `stations[].name`  | `string` | Nom officiel de la gare          |
| `stations[].uic`   | `string` | Code UIC de la gare              |
| `stations[].lat`   | `number` | Latitude (WGS84)                 |
| `stations[].lon`   | `number` | Longitude (WGS84)                |

**Code Python a utiliser** :

```python
# Rechercher dans TrainGraph.graph.nodes
results = []
q_lower = q.lower()
for uic, data in graph.graph.nodes(data=True):
    if q_lower in data["name"].lower():
        lat, lon = data["pos"]  # pos est un tuple (lat, lon)
        results.append({
            "name": data["name"],
            "uic": uic,
            "lat": lat,
            "lon": lon,
        })
    if len(results) >= limit:
        break
```

> **Conseil** : Trier les resultats par pertinence (match exact d'abord, puis prefixe, puis contient) pour une meilleure experience utilisateur. S'inspirer de `TrainGraph._find_uic_by_name()` pour la logique de tri.

---

### 3.7 POST /api/pathfinding/route

**Description** : Calcul d'itineraire ferroviaire entre deux gares, avec arrets intermediaires optionnels. Le format de reponse est identique au sous-objet `pathfinding` de `/api/resolve`.

**Corps de la requete** :

```json
{
  "departure": "Paris Gare de Lyon",
  "destination": "Lyon Part-Dieu",
  "intermediates": ["Marseille Saint-Charles"]
}
```

| Champ           | Type       | Requis | Description                                 |
| --------------- | ---------- | ------ | ------------------------------------------- |
| `departure`     | `string`   | Oui    | Nom de la gare de depart                    |
| `destination`   | `string`   | Oui    | Nom de la gare de destination               |
| `intermediates` | `string[]` | Non    | Liste des arrets intermediaires (dans l'ordre) |

**Reponse** (`200 OK`) :

```json
{
  "found": true,
  "route": ["Paris Gare de Lyon", "Marseille Saint-Charles", "Lyon Part-Dieu"],
  "route_details": [
    {
      "from_station": "Paris Gare de Lyon",
      "from_uic": "87686006",
      "to_station": "...",
      "to_uic": "...",
      "line": "830000",
      "type": "TRAIN",
      "geometry": [[48.8449, 2.3735], ...]
    }
  ],
  "total_stops": 3,
  "transfers": 1,
  "error": null
}
```

Le format de reponse est **identique** a `pathfinding` dans `/api/resolve` (voir section 3.2).

**Code Python a utiliser** :

```python
# 1. Appeler get_path
simplified, error, full_uic_path = graph.get_path(
    departure, destination, intermediates, algorithm="astar"
)

# 2. Si error:
if error:
    return {"found": False, "route": [], "route_details": [], "total_stops": 0, "transfers": 0, "error": error}

# 3. Construire route_details (CRITIQUE : voir le detail dans la section 3.2)
route_details = []
transfers = 0
prev_line = None
for i in range(len(full_uic_path) - 1):
    u, v = full_uic_path[i], full_uic_path[i + 1]
    edge_data = graph.graph.get_edge_data(u, v)[0]  # [0] car MultiGraph !
    line = edge_data.get("line", "UNKNOWN")
    seg_type = edge_data.get("type", "TRAIN")

    # Compter les correspondances (changement de ligne)
    if prev_line is not None and line != prev_line:
        transfers += 1
    prev_line = line

    route_details.append({
        "from_station": graph.graph.nodes[u]["name"],
        "from_uic": u,
        "to_station": graph.graph.nodes[v]["name"],
        "to_uic": v,
        "line": line,
        "type": seg_type,
        "geometry": edge_data.get("geometry", [
            list(graph.graph.nodes[u]["pos"]),
            list(graph.graph.nodes[v]["pos"]),
        ]),
    })

return {
    "found": True,
    "route": simplified,
    "route_details": route_details,
    "total_stops": len(simplified),
    "transfers": transfers,
    "error": None,
}
```

> **CRITIQUE** : Pour chaque arete `(u, v)` dans `full_uic_path`, recuperer les donnees avec `graph.graph.get_edge_data(u, v)[0]`. Le `[0]` est indispensable car le graphe est un `MultiGraph` (plusieurs aretes possibles entre deux noeuds). Les donnees contiennent :
> - `geometry` : liste de coordonnees `[lat, lon]` du trace de la ligne
> - `line` : code de la ligne ferroviaire
> - `type` : `"TRAIN"` ou `"WALK"` (transfert a pied entre gares d'une meme ville)
>
> Si `geometry` est `None` (certaines aretes n'ont pas de geometrie chargee), utiliser les positions des noeuds comme fallback : `[list(graph.graph.nodes[u]["pos"]), list(graph.graph.nodes[v]["pos"])]`.

---

### 3.8 POST /api/speech/transcribe

**Description** : Transcription d'un fichier audio en texte via le modele Whisper.

**Corps de la requete** : `multipart/form-data`

| Champ      | Type   | Requis | Description                                 |
| ---------- | ------ | ------ | ------------------------------------------- |
| `audio`    | `File` | Oui    | Fichier audio (blob) -- formats : wav, mp3, webm, ogg |
| `language` | `string` | Non  | Code langue ISO (defaut: `"fr"`)            |

**Reponse** (`200 OK`) :

```json
{
  "text": "je veux aller de paris a lyon",
  "language": "fr",
  "confidence": 0.95,
  "duration_seconds": 5.0,
  "latency_ms": 1200
}
```

| Champ              | Type     | Description                                    |
| ------------------ | -------- | ---------------------------------------------- |
| `text`             | `string` | Texte transcrit                                |
| `language`         | `string` | Langue detectee par Whisper                    |
| `confidence`       | `number` | Score de confiance moyen sur les segments      |
| `duration_seconds` | `number` | Duree de l'audio en secondes                   |
| `latency_ms`       | `number` | Temps de traitement en ms                      |

**Code Python a utiliser** :

```python
from src.speech.whisper_model import WhisperModel
import tempfile, time

whisper = WhisperModel(model_name="medium", language=language or "fr")

# 1. Sauvegarder le fichier audio temporairement
with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
    tmp.write(await audio.read())
    tmp_path = tmp.name

# 2. Transcrire
start = time.time()
result: TranscriptionResult = whisper.transcribe(tmp_path)
latency = (time.time() - start) * 1000

# 3. Calculer la confidence moyenne des segments
confidence = 0.95  # Whisper ne fournit pas directement un score global
# Optionnel : calculer a partir de result.segments si disponible

# 4. Calculer la duree audio
# Utiliser result.segments[-1]["end"] si des segments sont disponibles
duration = result.segments[-1]["end"] if result.segments else 0.0

# 5. Nettoyer le fichier temporaire
os.unlink(tmp_path)

return {
    "text": result.text,
    "language": result.language,
    "confidence": confidence,
    "duration_seconds": duration,
    "latency_ms": round(latency, 1),
}
```

> **Note** : Le modele Whisper est **lourd en memoire** (~5 Go pour `medium`). Le charger une seule fois au demarrage du serveur et le reutiliser. Si la memoire est limitee, utiliser `"base"` ou `"small"`.

---

### 3.9 GET /api/evaluation/reports

**Description** : Liste tous les rapports d'evaluation disponibles.

**Parametres** : Aucun

**Reponse** (`200 OK`) :

```json
{
  "reports": [
    {
      "id": "evaluation_2024-01-15_14-30-00",
      "date": "2024-01-15",
      "eval_type": "entity",
      "dataset": "test_set",
      "samples": 500
    },
    {
      "id": "evaluation_2024-01-14_10-00-00",
      "date": "2024-01-14",
      "eval_type": "all",
      "dataset": "test_set",
      "samples": 500
    }
  ]
}
```

| Champ                | Type     | Description                                          |
| -------------------- | -------- | ---------------------------------------------------- |
| `reports[].id`       | `string` | Identifiant unique (nom du fichier sans extension)   |
| `reports[].date`     | `string` | Date de creation (ISO 8601, date seule)              |
| `reports[].eval_type`| `string` | Type d'evaluation (`"intent"`, `"entity"`, `"all"`, etc.) |
| `reports[].dataset`  | `string` | Nom du dataset utilise                               |
| `reports[].samples`  | `number` | Nombre d'echantillons evalues                        |

**Code Python a utiliser** :

```python
from pathlib import Path
import json

reports_dir = Path("src/evaluation/reports")  # ou "reports/" selon config
reports = []
for f in sorted(reports_dir.glob("*.json"), reverse=True):
    try:
        data = json.loads(f.read_text())
        # Extraire la date du nom de fichier : evaluation_2024-01-15_14-30-00.json
        date_str = f.stem.split("_")[1] if "_" in f.stem else ""
        reports.append({
            "id": f.stem,
            "date": date_str,
            "eval_type": data.get("eval_type", "all"),
            "dataset": data.get("dataset", "test_set"),
            "samples": data.get("samples", 0),
        })
    except Exception:
        continue
```

---

### 3.10 GET /api/evaluation/reports/:id

**Description** : Retourne le contenu complet d'un rapport d'evaluation specifique, incluant les resultats par modele et les matrices de confusion.

**Parametres de chemin** :

| Parametre | Type     | Description                          |
| --------- | -------- | ------------------------------------ |
| `id`      | `string` | Identifiant du rapport (ex: `evaluation_2024-01-15_14-30-00`) |

**Reponse** (`200 OK`) :

Le contenu JSON complet du fichier de rapport. La structure correspond a la sortie de `export_results_json()` dans `src/evaluation/reporting.py` :

```json
{
  "intent": {
    "regex": {
      "accuracy": 0.85,
      "macro_f1": 0.83,
      "trip_precision": 0.88,
      "trip_recall": 0.90,
      "trip_f1": 0.89,
      "trip_support": 250,
      "not_trip_precision": 0.80,
      "not_trip_recall": 0.75,
      "not_trip_f1": 0.77,
      "not_trip_support": 200,
      "unknown_precision": 0.0,
      "unknown_recall": 0.0,
      "unknown_f1": 0.0,
      "unknown_support": 50,
      "latency_ms": 2.5
    },
    "camembert": { "..." : "..." }
  },
  "entity": { "..." : "..." },
  "entity_fuzzy": { "..." : "..." },
  "combined": [ "..." ],
  "combined_fuzzy": [ "..." ],
  "language": { "..." : "..." }
}
```

**Reponse erreur** (`404 Not Found`) :

```json
{
  "detail": "Rapport non trouve : evaluation_xxx"
}
```

**Code Python a utiliser** :

```python
report_path = reports_dir / f"{report_id}.json"
if not report_path.exists():
    raise HTTPException(status_code=404, detail=f"Rapport non trouve : {report_id}")
return json.loads(report_path.read_text())
```

---

### 3.11 POST /api/evaluation/run

**Description** : Lance une evaluation en tache de fond. Retourne immediatement un identifiant de tache pour suivre la progression.

**Corps de la requete** :

```json
{
  "eval_type": "entity",
  "intent_models": ["regex", "camembert"],
  "entity_models": ["spacy", "camembert"],
  "use_fuzzy": true,
  "device": "cpu"
}
```

| Champ           | Type       | Requis | Description                                                      |
| --------------- | ---------- | ------ | ---------------------------------------------------------------- |
| `eval_type`     | `string`   | Oui    | Type : `"intent"`, `"entity"`, `"entity_fuzzy"`, `"combined"`, `"language"`, `"all"` |
| `intent_models` | `string[]` | Non    | Modeles d'intent a evaluer                                       |
| `entity_models` | `string[]` | Non    | Modeles d'entites a evaluer                                      |
| `use_fuzzy`     | `boolean`  | Non    | Activer le fuzzy matching (defaut: `true`)                       |
| `device`        | `string`   | Non    | Device : `"auto"`, `"cuda"`, `"cpu"` (defaut: `"cpu"`)          |

**Reponse** (`200 OK`) :

```json
{
  "task_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "status": "started",
  "message": "Evaluation 'entity' lancee avec 2 modeles"
}
```

| Champ     | Type     | Description                          |
| --------- | -------- | ------------------------------------ |
| `task_id` | `string` | UUID unique de la tache              |
| `status`  | `string` | Toujours `"started"`                 |
| `message` | `string` | Description de la tache lancee       |

**Code Python a utiliser** :

```python
import uuid
from fastapi import BackgroundTasks

# Store en memoire pour le suivi des taches
# Dict[str, {"status": str, "progress": dict, "report_id": str | None}]
tasks_store: dict = {}

task_id = str(uuid.uuid4())
tasks_store[task_id] = {"status": "running", "progress": {"percent": 0, "elapsed_seconds": 0, "current_step": "Initialisation..."}, "report_id": None}

# Lancer en tache de fond avec FastAPI BackgroundTasks
background_tasks.add_task(run_evaluation_task, task_id, eval_type, intent_models, entity_models, use_fuzzy, device)

# La fonction run_evaluation_task() doit :
# 1. Charger le dataset (src/evaluation/data_loader.py :: load_dataset)
# 2. Creer les modeles (src/evaluation/model_factory.py)
# 3. Lancer les evaluateurs (src/evaluation/evaluators/)
# 4. Mettre a jour tasks_store[task_id]["progress"] regulierement
# 5. Exporter le rapport JSON (src/evaluation/reporting.py :: export_results_json)
# 6. Mettre a jour tasks_store[task_id] = {"status": "completed", "report_id": "..."}
```

---

### 3.12 GET /api/evaluation/status/:taskId

**Description** : Verifie l'avancement d'une tache d'evaluation en cours.

**Parametres de chemin** :

| Parametre | Type     | Description       |
| --------- | -------- | ----------------- |
| `taskId`  | `string` | UUID de la tache  |

**Reponse** (`200 OK`) -- Tache en cours :

```json
{
  "task_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "status": "running",
  "progress": {
    "percent": 45,
    "elapsed_seconds": 12,
    "current_step": "Evaluating spacy..."
  }
}
```

**Reponse** (`200 OK`) -- Tache terminee :

```json
{
  "task_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "status": "completed",
  "progress": {
    "percent": 100,
    "elapsed_seconds": 30,
    "current_step": "Termine"
  },
  "report_id": "evaluation_2024-01-15_14-30-00"
}
```

**Reponse** (`200 OK`) -- Tache echouee :

```json
{
  "task_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "status": "failed",
  "progress": {
    "percent": 45,
    "elapsed_seconds": 12,
    "current_step": "Erreur : ..."
  }
}
```

| Champ                       | Type             | Description                                   |
| --------------------------- | ---------------- | --------------------------------------------- |
| `task_id`                   | `string`         | UUID de la tache                               |
| `status`                    | `string`         | `"running"`, `"completed"`, `"failed"`         |
| `progress.percent`          | `number`         | Pourcentage d'avancement (0-100)               |
| `progress.elapsed_seconds`  | `number`         | Temps ecoule depuis le debut                   |
| `progress.current_step`     | `string`         | Description de l'etape en cours                |
| `report_id`                 | `string \| null` | ID du rapport genere (uniquement si `completed`) |

**Reponse erreur** (`404 Not Found`) :

```json
{
  "detail": "Tache non trouvee : a1b2c3d4..."
}
```

---

### 3.13 GET /api/monitoring/metrics

**Description** : Retourne l'historique des requetes traitees avec leurs metriques de performance.

**Parametres de requete (query string)** :

| Parametre | Type     | Requis | Description                                |
| --------- | -------- | ------ | ------------------------------------------ |
| `limit`   | `number` | Non    | Nombre max de requetes (defaut: 100)       |

**Exemple** : `GET /api/monitoring/metrics?limit=50`

**Reponse** (`200 OK`) :

```json
{
  "requests": [
    {
      "timestamp": "2024-01-15T14:30:00Z",
      "input_text": "Je veux aller de Paris a Lyon",
      "model_name": "camembert",
      "duration_s": 0.15,
      "cpu_percent_avg": 25.0,
      "ram_peak_mb": 512,
      "gpu_peak_mb": null,
      "carbon_kg": 0.000001
    }
  ],
  "total_requests": 150,
  "avg_latency_ms": 120
}
```

| Champ                         | Type             | Description                               |
| ----------------------------- | ---------------- | ----------------------------------------- |
| `requests[]`                  | `array`          | Liste des requetes recentes               |
| `requests[].timestamp`        | `string`         | Horodatage ISO 8601                       |
| `requests[].input_text`       | `string`         | Texte d'entree                            |
| `requests[].model_name`       | `string`         | Modele utilise                             |
| `requests[].duration_s`       | `number`         | Duree d'execution en secondes             |
| `requests[].cpu_percent_avg`  | `number \| null` | Utilisation CPU moyenne                   |
| `requests[].ram_peak_mb`      | `number \| null` | Pic de RAM en Mo                          |
| `requests[].gpu_peak_mb`      | `number \| null` | Pic de VRAM en Mo (null si pas de GPU)    |
| `requests[].carbon_kg`        | `number \| null` | Emissions CO2 en kg                       |
| `total_requests`              | `number`         | Nombre total de requetes dans l'historique |
| `avg_latency_ms`              | `number`         | Latence moyenne en ms                     |

**Code Python a utiliser** :

```python
from src.monitoring.metrics_logger import MetricsLogger

metrics_logger = MetricsLogger(output_dir="reports/metrics")

# get_request_history retourne les N dernieres requetes (plus recentes en premier)
requests = metrics_logger.get_request_history(limit=limit)

# Calculer les stats agregees
total_requests = len(requests)
avg_latency_ms = (
    sum(r["duration_s"] for r in requests) / total_requests * 1000
    if total_requests > 0 else 0
)

return {
    "requests": requests,
    "total_requests": total_requests,
    "avg_latency_ms": round(avg_latency_ms, 1),
}
```

> **Note** : Les donnees proviennent du fichier `reports/metrics/requests.jsonl`. Chaque ligne est un objet JSON correspondant a un `RequestMetrics`. Le champ `output` et `extra` du `RequestMetrics` ne doivent **pas** etre inclus dans la reponse API (les filtrer cote serveur).

---

### 3.14 GET /api/monitoring/resources

**Description** : Retourne un instantane des ressources systeme (CPU, RAM, GPU).

**Parametres** : Aucun

**Reponse** (`200 OK`) :

```json
{
  "cpu_percent": 25.0,
  "ram_used_mb": 4096,
  "ram_total_mb": 16384,
  "ram_percent": 25.0,
  "gpu_used_mb": null,
  "gpu_total_mb": null,
  "gpu_percent": null
}
```

| Champ          | Type             | Description                              |
| -------------- | ---------------- | ---------------------------------------- |
| `cpu_percent`  | `number`         | Utilisation CPU en pourcentage           |
| `ram_used_mb`  | `number`         | RAM utilisee en Mo                       |
| `ram_total_mb` | `number`         | RAM totale en Mo                         |
| `ram_percent`  | `number`         | RAM utilisee en pourcentage              |
| `gpu_used_mb`  | `number \| null` | VRAM utilisee en Mo (null si pas de GPU) |
| `gpu_total_mb` | `number \| null` | VRAM totale en Mo (null si pas de GPU)   |
| `gpu_percent`  | `number \| null` | VRAM utilisee en % (null si pas de GPU)  |

**Code Python a utiliser** :

```python
from src.monitoring.resource_tracker import ResourceTracker

# La methode _get_snapshot() est statique et retourne un ResourceSnapshot
snapshot = ResourceTracker._get_snapshot()

return {
    "cpu_percent": snapshot.cpu_percent,
    "ram_used_mb": round(snapshot.ram_used_mb, 1),
    "ram_total_mb": round(snapshot.ram_total_mb, 1),
    "ram_percent": snapshot.ram_percent,
    "gpu_used_mb": round(snapshot.gpu_used_mb, 1) if snapshot.gpu_used_mb is not None else None,
    "gpu_total_mb": round(snapshot.gpu_total_mb, 1) if snapshot.gpu_total_mb is not None else None,
    "gpu_percent": round(snapshot.gpu_percent, 1) if snapshot.gpu_percent is not None else None,
}
```

> **Note** : `ResourceTracker._get_snapshot()` est une methode `@staticmethod`. Elle necessite `psutil` et optionnellement `torch` pour les metriques GPU.

---

### 3.15 GET /api/monitoring/carbon

**Description** : Retourne les emissions carbone cumulees de toutes les requetes traitees.

**Parametres** : Aucun

**Reponse** (`200 OK`) :

```json
{
  "total_emissions_kg": 0.0001,
  "total_energy_kwh": 0.005,
  "total_requests": 150,
  "country": "FRA"
}
```

| Champ                | Type     | Description                            |
| -------------------- | -------- | -------------------------------------- |
| `total_emissions_kg` | `number` | Emissions CO2 cumulees en kg           |
| `total_energy_kwh`   | `number` | Energie consommee cumulee en kWh       |
| `total_requests`     | `number` | Nombre total de requetes               |
| `country`            | `string` | Code pays ISO 3166-1 alpha-3           |

**Code Python a utiliser** :

```python
from src.monitoring.metrics_logger import MetricsLogger

metrics_logger = MetricsLogger(output_dir="reports/metrics")

# Lire tout l'historique pour agreger
all_requests = metrics_logger.get_request_history(limit=999999)

total_emissions = sum(r.get("carbon_kg", 0) or 0 for r in all_requests)
total_energy = total_emissions * 50  # Approximation : 1 kg CO2 ~ 50 kWh en France
# Ou mieux : stocker energy_kwh dans RequestMetrics si CodeCarbon est utilise

return {
    "total_emissions_kg": round(total_emissions, 8),
    "total_energy_kwh": round(total_energy, 8),
    "total_requests": len(all_requests),
    "country": "FRA",
}
```

> **Note** : Le champ `carbon_kg` dans `RequestMetrics` peut etre `null` si CodeCarbon n'etait pas actif. Traiter `None` comme `0`. Pour `energy_kwh`, idealement utiliser les donnees CodeCarbon via `CarbonCalculator`. A defaut, utiliser un facteur de conversion approximatif pour la France (mix electrique bas carbone).

---

## 4. Mapping code Python existant

Tableau recapitulatif du code Python a appeler pour chaque endpoint :

| Endpoint                        | Module(s) Python                                         | Methode(s) principale(s)                                |
| ------------------------------- | -------------------------------------------------------- | ------------------------------------------------------- |
| `GET /api/health`               | `src.utils.device`, `src.nlp.pipeline`, `src.pathfinding.graph` | `get_device_info()`, `TrainGraph.graph.number_of_nodes()` |
| `POST /api/resolve`             | `src.nlp.pipeline`, `src.pathfinding.graph`              | `NLPPipeline.process()`, `TrainGraph.get_path()`        |
| `POST /api/nlp/language`        | `src.nlp.language`                                       | `LanguageDetector.detect()`                              |
| `POST /api/nlp/intent`          | `src.nlp.intent`                                         | `IntentClassifier.classify()`                            |
| `POST /api/nlp/entities`        | `src.nlp.entity`                                         | `EntityExtractor.extract()`                              |
| `GET /api/pathfinding/stations` | `src.pathfinding.graph`                                  | `TrainGraph.graph.nodes` (iteration)                     |
| `POST /api/pathfinding/route`   | `src.pathfinding.graph`                                  | `TrainGraph.get_path()`                                  |
| `POST /api/speech/transcribe`   | `src.speech.whisper_model`                               | `WhisperModel.transcribe()`                              |
| `GET /api/evaluation/reports`   | `src.evaluation.reporting`                               | Scan du dossier `reports/`                               |
| `GET /api/evaluation/reports/:id` | `src.evaluation.reporting`                             | Lecture du fichier JSON                                  |
| `POST /api/evaluation/run`      | `src.evaluation.cli`, `src.evaluation.model_factory`     | `BackgroundTasks` + evaluateurs                          |
| `GET /api/evaluation/status/:id`| (store en memoire)                                       | Lecture du dictionnaire `tasks_store`                    |
| `GET /api/monitoring/metrics`   | `src.monitoring.metrics_logger`                          | `MetricsLogger.get_request_history()`                    |
| `GET /api/monitoring/resources` | `src.monitoring.resource_tracker`                        | `ResourceTracker._get_snapshot()`                        |
| `GET /api/monitoring/carbon`    | `src.monitoring.metrics_logger`                          | Aggregation de `carbon_kg` depuis `requests.jsonl`       |

---

## 5. Types de reference

### PredictionResult (src/nlp/types.py)

```python
@dataclass
class PredictionResult:
    intent: Intent           # TRIP, NOT_TRIP, UNKNOWN
    intent_confidence: float
    language: Language        # FRENCH, ENGLISH, UNKNOWN
    language_confidence: float
    departure: str
    destination: str
    intermediates: list[str]
    entities: list[TravelEntity]
    raw_text: str
    processed_text: str
    model_name: str
    metadata: dict[str, Any]
```

### ResourceSnapshot (src/monitoring/resource_tracker.py)

```python
@dataclass
class ResourceSnapshot:
    cpu_percent: float
    ram_used_mb: float
    ram_total_mb: float
    ram_percent: float
    gpu_used_mb: Optional[float]
    gpu_total_mb: Optional[float]
    gpu_percent: Optional[float]
    gpu_name: Optional[str]
```

### RequestMetrics (src/monitoring/metrics_logger.py)

```python
@dataclass
class RequestMetrics:
    timestamp: str
    input_text: str
    model_name: str
    duration_s: float
    cpu_percent_avg: Optional[float]
    ram_peak_mb: Optional[float]
    gpu_peak_mb: Optional[float]
    carbon_kg: Optional[float]
    output: Optional[dict]
    extra: dict
```

### TranscriptionResult (src/speech/whisper_model.py)

```python
@dataclass
class TranscriptionResult:
    text: str
    language: str
    segments: list[dict]  # [{"start": float, "end": float, "text": str}]
```

### TrainGraph.get_path() (src/pathfinding/graph.py)

```python
def get_path(
    dep_name: str,
    dest_name: str,
    intermediates: Optional[List[str]] = None,
    algorithm: str = "astar",
) -> Tuple[Optional[List[str]], Optional[str], Optional[List[str]]]:
    """
    Returns: (simplified_names, error_message, full_uic_path)
    - simplified_names : liste de noms de gares simplifiee (arrets cles)
    - error_message : None si succes, sinon message d'erreur
    - full_uic_path : liste complete des UICs traverses
    """
```

---

## 6. Notes d'implementation

### 6.1 Structure du serveur FastAPI

```
src/api/
  __init__.py
  main.py              # app FastAPI, startup events, CORS
  dependencies.py      # Singletons : TrainGraph, NLPPipeline, WhisperModel, MetricsLogger
  routers/
    __init__.py
    health.py          # GET /api/health
    resolve.py         # POST /api/resolve
    nlp.py             # POST /api/nlp/{language,intent,entities}
    pathfinding.py     # GET /api/pathfinding/stations, POST /api/pathfinding/route
    speech.py          # POST /api/speech/transcribe
    evaluation.py      # GET/POST /api/evaluation/*
    monitoring.py      # GET /api/monitoring/*
```

### 6.2 Initialisation au demarrage

Les objets lourds doivent etre initialises **une seule fois** au demarrage du serveur (event `startup` de FastAPI ou `lifespan`) :

```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Charger les ressources au demarrage
    app.state.graph = TrainGraph()          # Charge le graphe ferroviaire (~5s)
    app.state.station_db = StationDatabase()
    app.state.whisper = WhisperModel()      # Charge Whisper lazily
    app.state.metrics_logger = MetricsLogger()
    yield
    # Cleanup si necessaire

app = FastAPI(lifespan=lifespan)
```

### 6.3 CORS

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### 6.4 Gestion des erreurs

Toutes les erreurs doivent retourner un JSON avec le champ `detail` :

```python
from fastapi import HTTPException

# Exemple
raise HTTPException(status_code=404, detail="Station non trouvee : XYZ")
raise HTTPException(status_code=400, detail="Le champ 'text' est requis")
```

### 6.5 Logging des requetes

Chaque appel a `/api/resolve` doit etre enregistre via `MetricsLogger` pour alimenter les endpoints de monitoring :

```python
with metrics_logger.start_request(text, model_name) as timer:
    result = pipeline.process(text)
    timer.set_output(result.to_dict())
```

### 6.6 MultiGraph -- Attention

Le graphe ferroviaire est un `nx.MultiGraph`. Pour recuperer les donnees d'une arete :

```python
# CORRECT :
edge_data = graph.graph.get_edge_data(u, v)[0]

# INCORRECT (ne fonctionne pas avec MultiGraph) :
edge_data = graph.graph[u][v]  # Retourne un dict de dicts, pas les donnees directement
```

### 6.7 Geometry fallback

Certaines aretes n'ont pas de champ `geometry` (transferts a pied, lignes sans geometrie chargee). Toujours prevoir un fallback :

```python
geometry = edge_data.get("geometry") or [
    list(graph.graph.nodes[u]["pos"]),
    list(graph.graph.nodes[v]["pos"]),
]
```

### 6.8 Modeles disponibles

| Categorie         | Modeles disponibles                              | Fichier source                        |
| ----------------- | ------------------------------------------------ | ------------------------------------- |
| Language Detector  | `regex`, `langdetect`                           | `src/nlp/language/`                   |
| Intent Classifier  | `regex`, `camembert`, `spacy`, `flant5`         | `src/nlp/intent/`                     |
| Entity Extractor   | `regex`, `spacy`, `camembert`, `flant5`         | `src/nlp/entity/`                     |
| Post-Processor     | `fuzzy` (FuzzyPostProcessor)                    | `src/nlp/post/fuzzy_post_processor.py`|
| Speech-to-Text     | `whisper` (tiny/base/small/medium/large)        | `src/speech/whisper_model.py`         |

Pour instancier les modeles, utiliser les fonctions de `src/evaluation/model_factory.py` :
- `create_intent_classifiers(models, device)`
- `create_entity_extractors(models, device)`
- `create_language_detectors(models)`
- `create_fuzzy_post_processor()`
