# Travel Order Resolver - Questions / Réponses pour la Présentation

---

## 1. QUESTIONS GÉNÉRALES SUR LE PROJET

### Q: C'est quoi le projet en une phrase ?

Un système NLP qui transforme des demandes de voyage en langage naturel français (texte ou voix) en itinéraires ferroviaires optimaux sur le réseau SNCF.

### Q: Quel problème vous résolvez ?

L'utilisateur dit par exemple "Je veux aller de Paris à Lyon en passant par Dijon" — le système comprend automatiquement l'intention de voyage, extrait les villes (départ, arrivée, étapes), puis calcule le meilleur itinéraire sur le réseau ferré français réel (3000+ gares).

### Q: Quelle est la stack technique ?

- **Backend** : Python 3.10, FastAPI, PyTorch 2.1, Transformers (HuggingFace), spaCy, NetworkX
- **Frontend** : Vue 3, TypeScript, Tailwind CSS, Leaflet (cartes), ECharts (graphiques)
- **ML/NLP** : CamemBERT (BERT français), Flan-T5, spaCy fr_core_news_lg, OpenAI Whisper
- **Infra** : Docker, CUDA 12.1 (GPU), Prometheus + Grafana (monitoring), CodeCarbon (empreinte carbone)
- **Outils** : Poetry (deps Python), Vite (build frontend), Git LFS (modèles lourds)

### Q: Combien de temps a duré le projet ? Combien de personnes ?

Projet Epitech T9-AIA, 4 personnes : Romain Bernier (architecte), Victor Vattier (ML lead), Marine Gayet (backend & ML), Camille Kerserho (backend & ML).

---

## 2. QUESTIONS SUR LE PIPELINE NLP

### Q: Décrivez le pipeline NLP de bout en bout.

Le pipeline est composé de 5 étapes séquentielles :

1. **Preprocessing** (~<1ms) : nettoyage des artéfacts STT (ex: `[musique]`, `[bruit]`), normalisation unicode, normalisation des apostrophes et tirets
2. **Détection de langue** (~6ms) : identifie si le texte est en français, anglais ou autre. On utilise la lib `langdetect` (77.9% accuracy)
3. **Classification d'intent** (~0.8ms) : détermine si c'est une demande de voyage (TRIP), pas un voyage (NOT_TRIP), ou indéterminé (UNKNOWN). Meilleur modèle : spaCy (79.1%)
4. **Extraction d'entités** (~10ms) : extrait départ, destination, et étapes intermédiaires. Meilleur modèle : Regex + Fuzzy (57.1%)
5. **Post-processing** : fuzzy matching avec rapidfuzz pour corriger les noms de gares vers les vrais noms SNCF. Apporte **+15% d'accuracy**

**Latence totale : ~17ms par requête.**

### Q: Pourquoi une architecture modulaire ?

Toutes les composantes implémentent des interfaces abstraites (`LanguageDetector`, `IntentClassifier`, `EntityExtractor`, `PostProcessor`). Ça permet de :
- Swapper facilement un modèle par un autre (ex: remplacer Regex par CamemBERT)
- Évaluer chaque modèle indépendamment avec les mêmes métriques
- Tester différentes combinaisons de modèles
- Ajouter un nouveau modèle sans toucher au reste du pipeline

### Q: Quels modèles avez-vous comparés et lequel est le meilleur ?

On a implémenté et benchmarké **4 approches** pour chaque tâche :

**Intent Classification :**

| Modèle | Accuracy | Latence | Principe |
|--------|----------|---------|----------|
| Regex | 65.8% | <0.1ms | Patterns de mots-clés ("aller", "train", "billet"...) |
| **SpaCy** | **79.1%** | **0.8ms** | Compte les entités LOC/GPE détectées |
| CamemBERT | 32.4% | ~1ms | Classification fine-tunée |
| Flan-T5 | - | plus lent | Seq2seq |

**Entity Extraction :**

| Modèle | Accuracy | Accuracy + Fuzzy | Latence | Principe |
|--------|----------|-------------------|---------|----------|
| **Regex** | 33.3% | **57.1%** | **9.8ms** | Patterns "de X à Y" + fuzzy matching |
| SpaCy | 27.8% | 34.2% | 2.5ms | NER pré-entraîné (LOC, GPE) |
| CamemBERT | 16.3% | 31.2% | 0.8ms | Token classification BIO fine-tuné |
| Flan-T5 | **87% F1** | - | plus lent | Génération séq2séq promptée |

**Conclusion** : pour la production, on recommande SpaCy (intent) + Regex+Fuzzy (entities) pour le meilleur compromis accuracy/latence. Flan-T5 a le meilleur F1 brut (87%) mais est plus lent.

### Q: Comment marche l'extracteur Regex ?

Il utilise des patterns regex pour capturer les structures de phrases françaises et anglaises :

**Patterns français (par priorité) :**
1. Pattern 3 gares : `X puis Y puis Z` → départ, via, destination
2. `de X à/vers Y` : `(?:de|depuis|au depart de)\s+(.+?)\s+(?:à|vers|direction|pour)\s+(.+?)`
3. `à Y depuis X` : pattern inversé
4. `X vers Y` : pattern simple direction

**Patterns anglais :** `from X to Y`, `X to Y`

**Intermédiaires :** `via`, `en passant par`, `avec arrêt à`, `puis`

Ensuite le fuzzy post-processing corrige "Liyon" → "Lyon", "Pari" → "Paris" etc.

### Q: Comment marche CamemBERT pour le NER ?

CamemBERT est un modèle BERT pré-entraîné sur du français (par l'équipe ALMAnaCH de l'INRIA). On l'a **fine-tuné** pour la classification de tokens avec des tags BIO :

- **B-DEPARTURE** : début d'un nom de gare de départ
- **I-DEPARTURE** : suite du nom de gare de départ
- **B-DESTINATION** : début d'un nom de gare d'arrivée
- **I-DESTINATION** : suite du nom de gare d'arrivée
- **O** : token non pertinent

Entraîné sur ~50k samples du dataset augmenté via le HuggingFace Trainer.

**Résultat** : 31.2% accuracy avec fuzzy. C'est décevant car le modèle a du mal avec les noms de gares rares. Les patterns regex + fuzzy matching sont plus efficaces pour ce cas d'usage spécifique.

### Q: Comment marche Flan-T5 ?

Flan-T5 est un modèle seq2seq (encoder-decoder) de Google, fine-tuné pour suivre des instructions. On l'a re-fine-tuné sur nos données de voyage avec PEFT (Parameter-Efficient Fine-Tuning).

**Principe :** on lui donne un prompt et il génère une réponse structurée.

```
Input:  "Extrais les villes: Je veux aller de Paris à Lyon via Dijon"
Output: "DEPART: Paris | ARRIVEE: Lyon | VIA: Dijon"
```

On parse ensuite la sortie avec des regex pour extraire les champs.

**Configuration :** température 0.1 (peu de créativité), max 128 tokens.

**Résultat** : 87% F1, le meilleur score brut, mais plus lent que le regex à l'inférence.

### Q: Qu'est-ce que le fuzzy matching et pourquoi c'est aussi impactant (+15%) ?

Le fuzzy matching utilise la bibliothèque **rapidfuzz** pour trouver la gare SNCF la plus proche d'un texte extrait, même avec des fautes.

**Algorithme en 5 étapes :**
1. **Normalisation** : minuscules, suppression accents, remplacement tirets/apostrophes
2. **Cache** : si déjà vu, retour immédiat (LRU cache)
3. **Match exact par préfixe** : "paris" matche "paris gare de lyon" → confiance 95%
4. **Match fuzzy premier mot** : index pré-calculé par premier mot, bonus de +20 points
5. **Match fuzzy complet** : score WRatio de rapidfuzz sur tous les noms normalisés

**Seuil** : score minimum de 80/100 pour accepter un match.

**Pourquoi +15%** : beaucoup d'erreurs viennent de petites différences (accents, casse, abréviations). Le fuzzy matching les corrige automatiquement. C'est le **plus gros gain de performance** du pipeline.

### Q: Qu'est-ce que le preprocessing STT et pourquoi c'est nécessaire ?

Quand on utilise Whisper (speech-to-text), la transcription contient des artéfacts :
- Marqueurs de bruit : `[musique]`, `[bruit]`, `[silence]`, `[inaudible]`
- Patterns de gibberish : `...`, `---`, `___`
- Marqueurs entre crochets : `[quelque chose]`

Le filtre STT retire ces 33+ marqueurs connus et 6 patterns regex avant de passer le texte au NLP. Impact mesuré : **+0.2% accuracy** (marginal mais nécessaire pour éviter des faux positifs).

---

## 3. QUESTIONS SUR LE PATHFINDING

### Q: Comment est construit le graphe ferroviaire ?

Le graphe est un **MultiGraph NetworkX** (plusieurs arêtes possibles entre 2 nœuds) construit à partir de 3 fichiers de données SNCF open data :

1. **gares-de-voyageurs.json** : ~3000+ gares avec nom, code UIC, coordonnées GPS
2. **liste-des-gares.json** : association gare ↔ ligne avec Position Kilométrique (PK)
3. **lignes-par-type.json** : types de lignes (TGV, TER, Intercités) + géométrie GeoJSON

**Construction en 5 étapes :**
1. Charger les gares → créer les nœuds (UIC = identifiant, attributs = nom + position GPS)
2. Détecter les lignes LGV (nom contient "LGV" ou "VITESSE")
3. Charger les géométries GeoJSON pour la visualisation
4. Construire la topologie : pour chaque ligne, connecter les gares consécutives triées par PK
5. Ajouter les correspondances urbaines : arêtes piétonnes entre gares < 8km dans les hubs (Paris, Lyon, Lille, Marseille, Bordeaux, Nantes)

### Q: Comment sont calculés les poids des arêtes ?

Le poids représente un coût de trajet (plus c'est bas, mieux c'est) :

| Type | Formule | Vitesse simulée |
|------|---------|-----------------|
| **Ligne LGV** | `distance_km / 3.0` | ~280 km/h |
| **Ligne classique** | `distance_km` | ~80 km/h |
| **Correspondance piétonne** | `15` (fixe) | ~5 km/h |

La division par 3 pour les LGV simule le fait qu'un TGV est ~3x plus rapide qu'un TER. L'algorithme va donc naturellement favoriser les lignes à grande vitesse.

**Distance** : calculée via la différence de PK (point kilométrique) entre 2 gares sur la même ligne. Si la valeur est aberrante (<0.1km ou >500km), on utilise la distance géodésique (à vol d'oiseau) comme fallback.

### Q: Expliquez l'algorithme A* et pourquoi vous l'avez choisi.

**A*** est une extension de Dijkstra qui utilise une heuristique pour guider la recherche vers la destination.

**Formule** : `f(n) = g(n) + h(n)`
- `g(n)` = coût réel depuis le départ
- `h(n)` = estimation du coût restant (heuristique)
- `f(n)` = coût total estimé

**Notre heuristique** : distance géodésique (à vol d'oiseau en km) divisée par 3.0
```python
h(n) = geodesic(position_n, position_destination).km / 3.0
```

**Pourquoi diviser par 3 ?** Pour rester **admissible** (ne jamais surestimer). Comme les lignes LGV ont un poids de `dist_km / 3`, l'heuristique doit diviser aussi par 3 pour ne pas surestimer le coût réel. Distance à vol d'oiseau ≤ distance réelle, donc `geodesic/3 ≤ dist_réelle/3`.

**Pourquoi A* plutôt que Dijkstra ?**
- Même complexité théorique : O((V+E) log V)
- Mais en pratique, A* explore **beaucoup moins de nœuds** car il "sait" dans quelle direction chercher
- Garantit le même chemin optimal (grâce à l'admissibilité)
- Dijkstra explore dans toutes les directions ; A* va droit au but

### Q: Expliquez Dijkstra.

**Dijkstra** est l'algorithme classique du plus court chemin dans un graphe à poids positifs.

**Principe :**
1. File de priorité (min-heap) ordonnée par coût
2. On démarre du nœud source avec coût 0
3. À chaque itération, on pop le nœud avec le plus petit coût
4. On relaxe ses voisins : si `coût_actuel + poids_arête < coût_voisin`, on met à jour
5. On s'arrête quand on atteint la destination

**Structures de données :**
- `dist[node]` : meilleure distance trouvée
- `came_from[node]` : parent pour reconstruire le chemin
- `visited` : ensemble des nœuds déjà explorés
- `open_set` : min-heap (distance, compteur, nœud)

**Complexité** : O((V + E) log V) avec un heap binaire.

**Limite** : explore uniformément dans toutes les directions, ce qui est sous-optimal quand on connaît la position géographique de la destination.

### Q: C'est quoi MOA* et pourquoi l'avoir implémenté ?

**MOA*** (Multi-Objective A*) est une extension de A* qui optimise **3 objectifs simultanément** :

1. **Temps** : `distance / vitesse` (en heures)
2. **Distance** : km réels
3. **Correspondances** : nombre de changements de ligne

**Concept clé : la dominance de Pareto.**
- Un chemin A **domine** B si A est meilleur ou égal sur TOUS les objectifs, et strictement meilleur sur au moins un.
- La **frontière de Pareto** = ensemble des solutions non-dominées (aucune n'est meilleure que les autres sur tous les critères).

**Exemple concret :**
- Chemin 1 : 2h, 450km, 0 correspondance (TGV direct)
- Chemin 2 : 3h30, 380km, 2 correspondances (TER, moins cher)
- Chemin 3 : 2h30, 420km, 1 correspondance (compromis)

Aucun ne domine les autres → tous sont sur la frontière de Pareto → on les propose tous les 3 à l'utilisateur.

**Implémentation :**
- Vecteur de coût `CostVector(time, distance, transfers)` au lieu d'un scalaire
- Frontière de Pareto par nœud (au lieu d'un simple `dist[n]`)
- Pruning : on abandonne un chemin s'il est dominé par une solution existante
- Heuristique multi-objectif : `h = (geodesic/280, geodesic, 0)` (optimiste sur chaque dimension)
- Retourne jusqu'à 3 solutions Pareto-optimales

**Complexité** : O((V·P + E) log V) où P = taille de la frontière de Pareto par nœud.

### Q: Comment gérez-vous les gares intermédiaires ?

Si l'utilisateur dit "Paris à Marseille via Lyon" :

1. On découpe en legs : Paris→Lyon, puis Lyon→Marseille
2. On calcule le plus court chemin pour chaque leg
3. On concatène : `chemin1 + chemin2[1:]` (on évite de dupliquer Lyon)

**Pour MOA*** : on fait le produit cartésien des frontières de Pareto de chaque leg, on additionne les coûts, et on re-filtre pour garder la frontière de Pareto combinée (top 3).

### Q: Comment simplifiez-vous le chemin pour l'affichage ?

Le chemin brut peut contenir des dizaines de gares. On ne garde que les **arrêts clés** :
- Première et dernière gare (toujours)
- Gares de **changement de ligne** (quand le code ligne change)
- **Hubs majeurs** : gares dont le nom contient "Paris", "Lyon", etc. ET "TGV"
- **Waypoints explicites** : les étapes demandées par l'utilisateur

---

## 4. QUESTIONS SUR LE SPEECH-TO-TEXT

### Q: Comment marche la partie voix ?

On utilise **OpenAI Whisper**, un modèle de transcription audio pré-entraîné sur 680 000 heures d'audio multilingue.

**Configuration :**
- Modèle : `medium` (bon compromis précision/vitesse, ~1.5GB en mémoire)
- Hint de langue : `fr` (français)
- GPU : utilisé si disponible (fp16 pour accélérer)

**Pipeline audio complet :**
1. L'utilisateur enregistre via le micro (frontend Vue 3)
2. L'audio est envoyé au backend via `POST /api/speech/transcribe`
3. Whisper transcrit en texte
4. Le texte passe dans le pipeline NLP classique
5. L'itinéraire est calculé et renvoyé

**Gestion du silence :** `AudioRecorder` peut enregistrer une durée fixe ou s'arrêter automatiquement après 2s de silence (seuil RMS < 0.01).

### Q: Pourquoi avoir simulé les erreurs STT dans le dataset ?

Whisper fait des erreurs typiques : confusions phonétiques ("Lyon"→"Lion"), oublis d'accents ("Montpellier"→"montpelier"), artéfacts ([pause], [musique]). Pour que notre pipeline soit robuste en conditions réelles, on a **augmenté** notre dataset de base avec un simulateur d'erreurs STT :

- Erreurs au niveau caractère : s→z, é→e
- Confusions phonétiques
- Artéfacts Whisper
- Résultat : 15 005 samples de test + 100 000 samples augmentés

Ça nous a permis de mesurer la robustesse de chaque modèle face aux erreurs de transcription.

---

## 5. QUESTIONS SUR L'API ET LE FRONTEND

### Q: Comment est structurée l'API ?

FastAPI avec 7 routers, chacun préfixé par `/api/` :

| Router | Endpoint principal | Rôle |
|--------|-------------------|------|
| **health** | `GET /api/health` | Vérification que le serveur tourne |
| **resolve** | `POST /api/resolve` | Pipeline complet : texte → NLP → pathfinding |
| **nlp** | `POST /api/nlp/{language,intent,entities}` | Debug NLP : tester chaque modèle individuellement |
| **pathfinding** | `GET /api/pathfinding/stations`, `POST /api/pathfinding/route` | Recherche de gares et calcul de route |
| **speech** | `POST /api/speech/transcribe` | Transcription audio |
| **evaluation** | `POST /api/evaluation/run`, `GET /api/evaluation/reports` | Lancer et consulter des évaluations |
| **monitoring** | `GET /api/monitoring/{metrics,resources,carbon}` | Métriques de performance et empreinte carbone |

### Q: Comment marche l'endpoint /api/resolve ?

C'est le point d'entrée principal. Requête :
```json
{
  "text": "Je veux aller de Paris à Lyon",
  "intent_model": "spacy",       // optionnel
  "entity_model": "regex",       // optionnel
  "use_fuzzy": true               // optionnel, défaut true
}
```

**Traitement :**
1. Sélection des modèles (par défaut : premier disponible)
2. Pipeline NLP complet (language → intent → entities)
3. Fuzzy post-processing si activé
4. Si intent = TRIP et départ + destination trouvés → pathfinding A*
5. Retour JSON avec NLP results + route

**Réponse :**
```json
{
  "nlp": {
    "language": { "detected": "FRENCH", "confidence": 0.95 },
    "intent": { "value": "TRIP", "confidence": 0.8 },
    "entities": {
      "departure": { "raw": "Paris", "matched": "Paris Gare de Lyon", "confidence": 0.95 },
      "destination": { "raw": "Lyon", "matched": "Lyon Part Dieu", "confidence": 0.92 }
    },
    "latency_ms": 17.3
  },
  "pathfinding": {
    "simplified_path": ["Paris Gare de Lyon", "Dijon Ville", "Lyon Part Dieu"],
    "full_uic_path": ["87686006", "87713040", "87723197"]
  }
}
```

### Q: Comment est fait le frontend ?

**Vue 3 + TypeScript + Tailwind CSS**, avec 3 pages principales :

1. **Dashboard** : interface chat pour les requêtes de voyage, bouton micro pour la voix, carte interactive Leaflet qui trace les itinéraires
2. **Monitoring** : graphiques temps réel (CPU, RAM, latence, empreinte carbone) avec ECharts
3. **Évaluation** : lancement de benchmarks, affichage des résultats et matrices de confusion

**Architecture frontend :**
- **Composables** (hooks Vue 3) : `useResolve()`, `useMapRoute()`, `useMonitoring()`, `useAudioRecorder()`
- **Services API** : un fichier par domaine (`resolve.ts`, `nlp.ts`, `pathfinding.ts`, `speech.ts`)
- **Composants** : `ChatWindow`, `RecordButton`, `RailwayMap`, `IntentBadge`, `EntityHighlight`, `ResourceGauge`, `CarbonCard`

### Q: Comment est gérée l'initialisation des modèles ?

Via le **lifespan** de FastAPI (context manager async). Au démarrage :
1. Détection du device (CUDA ou CPU)
2. Chargement du graphe ferroviaire (~3000 gares)
3. Chargement de la base de gares
4. Instanciation de tous les modèles NLP (language, intent, entity)
5. Chargement du fuzzy post-processor
6. Chargement de Whisper
7. Initialisation du metrics logger

Tout est stocké dans `app.state` et injecté dans les endpoints via des fonctions de dépendance FastAPI (`Depends(get_graph)`, etc.).

---

## 6. QUESTIONS SUR LES DONNÉES

### Q: D'où viennent vos données ?

**Données SNCF (open data) :**
- `gares-de-voyageurs.json` : toutes les gares voyageurs avec coordonnées GPS
- `liste-des-gares.json` : gares + lignes desservies + PK
- `lignes-par-type.json` : types de lignes (TGV, TER, Intercités)

**Dataset NLP (généré par nous) :**
- Scripts de génération automatique dans `datasets/scripts/`
- Samples la liste réelle des gares SNCF
- Génère des phrases variées : "de X à Y", "je veux aller depuis X vers Y", etc.
- Distribution : 89% français, 6% anglais, 5% inconnu | 74% TRIP, 25% NOT_TRIP

### Q: Comment avez-vous augmenté les données ?

On a un script `augment_stt.py` qui simule les erreurs typiques de Whisper :

- **Erreurs caractère** : s→z, é→e, è→e
- **Confusions phonétiques** : "Lyon"→"Lion", "Marseille"→"Marseil"
- **Artéfacts Whisper** : ajout de `[pause]`, `[musique]`, `[bruit de fond]`
- **Casse** : majuscules/minuscules aléatoires

Résultat : un dataset augmenté de 15 005 samples de test et 100 000 samples d'entraînement pour des benchmarks robustes.

### Q: Quel est le format d'un sample ?

```json
{
  "sentence_id": "STT072273",
  "sentence": "Je veux aller de Paris à Lyon",
  "intent": "TRIP",
  "language": "FRENCH",
  "departure": "Paris",
  "destination": "Lyon",
  "intermediate": "Dijon"
}
```

---

## 7. QUESTIONS SUR L'ÉVALUATION ET LES MÉTRIQUES

### Q: Quelles métriques utilisez-vous ?

**Par classe (TRIP, NOT_TRIP, UNKNOWN) :**
- **Precision** = TP / (TP + FP) — "quand je dis TRIP, ai-je raison ?"
- **Recall** = TP / (TP + FN) — "est-ce que je détecte bien tous les TRIP ?"
- **F1** = 2 × (Precision × Recall) / (Precision + Recall) — moyenne harmonique

**Globales :**
- **Accuracy** = correct / total
- **Macro F1** = moyenne des F1 par classe (traite chaque classe également)

**Pour les entités :**
- Precision, Recall, F1 séparés pour départ et destination
- Accuracy = les deux (départ ET destination) sont corrects

**Opérationnelles :**
- Latence moyenne (ms)
- CPU % / RAM peak (MB) / GPU peak (MB)
- Émissions carbone (kg CO2)

### Q: Comment lancez-vous une évaluation ?

```bash
# Toutes les évaluations
python -m src.evaluation --eval-type all --models all

# Juste l'entity extraction avec fuzzy
python -m src.evaluation --eval-type entity_fuzzy --models regex spacy

# Via l'API
POST /api/evaluation/run
```

Les résultats sont exportés en JSON avec timestamp dans `src/evaluation/reports/` et affichés en tables ASCII dans le terminal (style sklearn classification_report).

### Q: Pourquoi CamemBERT a un score aussi bas (16-31%) ?

Plusieurs raisons :
1. Le fine-tuning NER sur des noms de gares est difficile car le vocabulaire est très spécifique (noms propres rares)
2. Le dataset d'entraînement contient des erreurs STT qui perturbent le token classification
3. Le modèle de base (camembert-base) n'a pas été entraîné spécifiquement sur des noms de lieux français
4. L'approche regex + fuzzy est plus adaptée car elle exploite directement la structure syntaxique ("de X à Y") plutôt que de deviner les entités

C'est un **résultat intéressant** : dans ce cas d'usage, le rule-based bat le deep learning grâce à la structure très prévisible des phrases de voyage.

### Q: Quel est l'impact de chaque étape du pipeline ?

| Étape | Impact sur l'accuracy |
|-------|----------------------|
| Preprocessing STT | +0.2% |
| Filtrage par langue | +3.7% |
| **Fuzzy post-processing** | **+15.0%** (le plus impactant !) |

Le fuzzy matching est de loin l'amélioration la plus significative. C'est logique : corriger "Liyon" en "Lyon" résout directement l'erreur sans avoir besoin d'un modèle plus complexe.

---

## 8. QUESTIONS SUR LE MONITORING ET L'EMPREINTE CARBONE

### Q: Comment mesurez-vous l'empreinte carbone ?

On utilise **CodeCarbon**, une lib Python qui mesure les émissions CO2 par requête :

- Mesure la consommation électrique (CPU + GPU)
- Multiplie par le facteur d'émission du pays (France : ~60g CO2/kWh, un des plus bas grâce au nucléaire)
- Chaque requête est loggée avec : durée, énergie (kWh), émissions (kg CO2)

Ça permet de quantifier l'impact environnemental de l'inférence et de comparer l'efficacité des modèles.

### Q: Quelle stack de monitoring ?

**Docker Compose** avec le profil `monitoring` :
- **Prometheus** (port 9090) : collecte de métriques toutes les 15s
- **Grafana** (port 3000) : dashboards visuels
- **Node Exporter** (port 9100) : métriques système (CPU, RAM, disque)
- **cAdvisor** (port 8080) : métriques au niveau container Docker
- **Gatus** (port 8080) : health checks et page de statut

On a aussi un **MetricsLogger** interne qui loggue chaque requête en JSONL (format streaming) avec timestamp, input, output, durée, CPU, RAM, GPU, carbone.

---

## 9. QUESTIONS SUR LE DÉPLOIEMENT

### Q: Comment est containerisé le projet ?

**2 Dockerfiles :**
1. **Backend** : basé sur `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime`, installe Poetry + dépendances + modèles spaCy. Support GPU natif via NVIDIA Container Toolkit.
2. **Frontend** : build multi-stage Node.js → Nginx. Vite compile les assets, Nginx les sert en production.

**docker-compose.yml** avec des profils :
- Profil par défaut : `app` (backend) + `frontend`
- Profil `monitoring` : Prometheus + Grafana + Node Exporter + cAdvisor + Gatus
- Profil `mlops` : MLflow pour le tracking d'expériences
- Profil `training` : container GPU pour l'entraînement

### Q: Pourquoi Docker avec GPU ?

Les modèles de deep learning (CamemBERT, Flan-T5, Whisper) sont **2 à 5x plus rapides** sur GPU. L'image Docker est basée sur CUDA 12.1 et utilise `--gpus all` pour accéder au GPU de la machine hôte. Sur CPU ça fonctionne aussi, mais c'est plus lent pour l'inférence des gros modèles.

---

## 10. QUESTIONS PIÈGES / TECHNIQUES

### Q: Pourquoi ne pas utiliser un LLM (GPT, Claude) directement ?

Plusieurs raisons :
1. **Latence** : notre pipeline fait ~17ms, un appel API à GPT fait 500ms-2s
2. **Coût** : pas d'API payante, tout tourne en local
3. **Offline** : fonctionne sans internet
4. **Contrôle** : on maîtrise exactement ce que fait chaque composant
5. **Données sensibles** : pas d'envoi de données à un tiers
6. **Reproductibilité** : résultats déterministes (surtout regex)

### Q: Votre heuristique A* est-elle admissible ? Prouvez-le.

Oui. L'heuristique est `h(n) = geodesic(n, goal) / 3.0`.

**Preuve d'admissibilité** (h ne surestime jamais le coût réel) :
- Le coût réel minimum est sur une LGV directe : `distance_réelle / 3.0`
- La distance géodésique (vol d'oiseau) ≤ distance réelle (par inégalité triangulaire)
- Donc `geodesic / 3.0 ≤ distance_réelle / 3.0 ≤ coût_réel`
- Donc h(n) ≤ coût_réel pour tout n. CQFD.

### Q: Pourquoi un MultiGraph et pas un simple Graph ?

Parce que deux gares peuvent être reliées par **plusieurs lignes différentes** (ex: Paris-Lyon peut avoir une ligne TGV et une ligne TER). Le MultiGraph de NetworkX permet d'avoir plusieurs arêtes entre les mêmes nœuds, chacune avec ses propres attributs (poids, vitesse, type de ligne). C'est essentiel pour MOA* qui doit considérer toutes les options.

### Q: Comment gérez-vous les noms ambigus de gares (ex: "Paris") ?

"Paris" matche plusieurs gares (Gare de Lyon, Gare du Nord, Montparnasse...). Le matching se fait par priorité :

1. **Match exact** : si le nom correspond exactement → retour immédiat
2. **Match par préfixe** : "Paris" matche "Paris Gare de Lyon" (première trouvée)
3. **Match par contenu** : le terme est contenu dans le nom de la gare

De plus, la `StationDatabase` a des **alias codés en dur** pour les cas courants :
- "paris gare de lyon" → alias : "paris lyon", "gare de lyon"
- "cdg" → "aéroport charles de gaulle 2 tgv"
- "lyon" → "lyon part dieu"

### Q: Que se passe-t-il si l'intent est TRIP mais qu'aucune gare n'est trouvée ?

Le pipeline **downgrade** l'intent de TRIP à NOT_TRIP si aucune gare de départ ET d'arrivée n'est trouvée après le matching. C'est une sécurité pour éviter de lancer le pathfinding sur des données vides.

### Q: Comment gérez-vous les langues autres que le français ?

- Le détecteur de langue identifie FR, EN, et "UNKNOWN" (pour espagnol, allemand, italien, etc.)
- Les patterns regex existent en français ET en anglais
- SpaCy utilise le modèle français `fr_core_news_lg`, donc il est moins performant en anglais
- **Résultat par langue** : SpaCy fait 82.2% accuracy sur le français mais seulement 57.1% sur l'anglais

Le système est clairement **optimisé pour le français** (réseau SNCF oblige), mais fonctionne en mode dégradé sur l'anglais.

### Q: Quelle est la complexité mémoire du système ?

| Composant | Mémoire |
|-----------|---------|
| Python de base | ~200 MB |
| spaCy fr_core_news_lg | ~60 MB |
| CamemBERT | ~400 MB |
| Flan-T5 | ~900 MB |
| Whisper (medium) | ~1.5 GB |
| Graphe SNCF | ~300 MB |
| **Total** | **~3.4 GB** |

Avec GPU : la VRAM est utilisée pour les modèles Transformers (CamemBERT, Flan-T5, Whisper).

### Q: Si vous deviez améliorer le projet, que feriez-vous ?

1. **Meilleur entity extraction** : fine-tuner CamemBERT avec plus de données propres (sans erreurs STT), ou utiliser un modèle plus gros
2. **Données horaires** : intégrer les horaires SNCF réels pour calculer des itinéraires avec heures de départ/arrivée
3. **Multi-modal** : combiner les résultats de plusieurs extracteurs (ensemble/vote)
4. **Cache** : mettre en cache les routes fréquentes (Paris→Lyon, etc.)
5. **Tests de bout en bout** : plus de tests d'intégration frontend ↔ backend
6. **Modèle de langue** : fine-tuner un détecteur de langue plus performant sur les textes courts

---

## 11. GLOSSAIRE RAPIDE

| Terme | Définition |
|-------|-----------|
| **NER** | Named Entity Recognition - reconnaissance d'entités nommées dans du texte |
| **BIO tagging** | Begin-Inside-Outside - format d'annotation pour le NER token par token |
| **Seq2Seq** | Sequence-to-Sequence - modèle qui transforme une séquence d'entrée en séquence de sortie |
| **PEFT** | Parameter-Efficient Fine-Tuning - fine-tuning qui ne modifie qu'une petite partie des paramètres |
| **UIC** | Union Internationale des Chemins de fer - code unique identifiant chaque gare |
| **PK** | Point Kilométrique - position d'une gare sur une ligne en km depuis l'origine |
| **LGV** | Ligne à Grande Vitesse - voie ferrée pour TGV (280+ km/h) |
| **Admissibilité** | Propriété d'une heuristique qui ne surestime jamais le coût réel |
| **Pareto** | Ensemble de solutions où aucune n'est meilleure que les autres sur tous les critères |
| **Fuzzy matching** | Correspondance approximative de chaînes (tolère les fautes) |
| **F1 Score** | Moyenne harmonique de la precision et du recall |
| **Macro F1** | Moyenne des F1 par classe (chaque classe a le même poids) |
| **CORS** | Cross-Origin Resource Sharing - autorise le frontend à appeler le backend |
| **Lifespan** | Pattern FastAPI pour initialiser/libérer les ressources au démarrage/arrêt |
| **STT** | Speech-To-Text - transcription audio vers texte |
| **CamemBERT** | Modèle BERT pré-entraîné sur du français (ALMAnaCH/INRIA) |
| **Flan-T5** | Modèle T5 de Google fine-tuné pour le suivi d'instructions |
| **rapidfuzz** | Bibliothèque Python de fuzzy matching rapide (alternative à fuzzywuzzy) |
| **CodeCarbon** | Lib Python pour mesurer les émissions CO2 du code |
