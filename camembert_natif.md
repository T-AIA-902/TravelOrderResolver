# 🧠 Résumé du modèle _Travel Order Resolver_ — Version CamemBERT natif

## 1\. Objectif du modèle

Ce modèle a pour objectif de **traiter des phrases en langage naturel (français)** exprimant une demande de voyage, afin d’en extraire automatiquement :

-   le **lieu de départ**
-   le **lieu de destination**

Il s’inscrit dans un pipeline plus large de _Travel Order Resolver_, dont la finalité est de produire un **itinéraire ferroviaire** à partir d’une commande utilisateur (chatbot, CLI, fichier texte, etc.).

Dans cette version, le système utilise **CamemBERT natif**, **sans fine-tuning**, afin d’établir une baseline simple, interprétable et robuste.

## 2\. Architecture générale du modèle

Le modèle repose sur **trois couches principales**, volontairement séparées :

Entrée texte
   ↓
Prétraitement linguistique
   ↓
Extraction des villes (fuzzy matching)
   ↓
Inférence origine / destination (règles linguistiques)
   ↓
Sortie structurée (CSV)

### 2.1 Entrées

-   Un fichier CSV `travel_sentences.csv`
-   Format attendu :

sentenceID,sentence,labels

> ⚠️ Les labels sont présents uniquement pour évaluation future (NER), ils **ne sont pas utilisés** par le modèle natif.

## 3\. Composants techniques

### 3.1 CamemBERT natif

-   Modèle : `camembert-base`
-   Usage :
-   Tokenisation française robuste
-   Encodage linguistique implicite (pas d’apprentissage)

📌 **Important**

Dans cette version :

-   CamemBERT **n’est pas utilisé pour prédire**
-   Il sert de **socle NLP prêt à évoluer** (fine-tuning futur)

### 3.2 Base de connaissances SNCF

Le modèle s’appuie sur un fichier `gares.csv` contenant :

station\_name;city\_name

Utilisation :

-   Référentiel des villes reconnues
-   Indépendant des phrases utilisateur
-   Peut être enrichi sans modifier le modèle

### 3.3 Extraction des villes (fuzzy matching)

Le système détecte les villes présentes dans la phrase à l’aide de :

-   **RapidFuzz**
-   Comparaison mot à mot avec la liste des villes SNCF
-   Seuil de similarité (ex. 80%)

📌 Avantages :

-   Tolérance aux fautes (`toulous` → `Toulouse`)
-   Indépendant de la casse et des accents
-   Fonctionne sans apprentissage

📌 Limites :

-   Sensible aux phrases longues
-   Faux positifs possibles si homonymes

### 3.4 Inférence origine / destination

Une fois les villes détectées, le modèle applique des **règles linguistiques déterministes** basées sur des mots-clés :

Mot-cléRôlede / depuisorigineà / vers / pourdestinationen partant deorigine

Si aucune règle explicite n’est trouvée :

-   Le modèle assigne **la première ville comme origine**
-   Et **la seconde comme destination**

📌 Cette étape est **interprétable**, mais **fragile linguistiquement**.

## 4\. Format de sortie

Le modèle produit un fichier :

sentenceID,Departure,Destination

Cas possibles :

SituationSortiePhrase comprisesentenceID,CityA,CityBPhrase non liée au voyagesentenceID,INVALIDAmbiguïté non résoluesentenceID,INVALID

## 5\. Interprétation des résultats

### 5.1 Ce que signifie `INVALID`

Un résultat `INVALID` signifie que **le modèle ne peut pas garantir** une extraction fiable, par exemple :

-   aucune ville reconnue
-   une seule ville détectée
-   ambiguïté syntaxique
-   phrase hors domaine

➡️ **Choix volontairement conservateur**, adapté à un système de production.

### 5.2 Qualité actuelle du modèle

✔️ Points forts :

-   fonctionne sans données d’apprentissage
-   tolère fautes et variations linguistiques
-   explicable et traçable
-   robuste pour une baseline

❌ Limites :

-   dépend fortement de règles
-   incapable de comprendre la sémantique profonde
-   difficulté avec :
-   phrases complexes
-   ordre implicite
-   structures longues
-   ne reconnait pas la provenance / destination en ne nommant que le nom de la gare (ex. "Part-Dieu")

## 6\. Évaluation qualitative (baseline)

Sur le dataset de test (~10 phrases) :

-   Précision élevée sur les structures simples
-   Faux négatifs fréquents sur :
-   phrases implicites
-   formulations non standard

📌 Cette performance est **attendue** pour un modèle non entraîné.

## 7\. Améliorations possibles (Makefile roadmap)

Voici une **roadmap claire et réaliste**, que tu peux présenter comme plan d’évolution.

### 🛠️ Phase 1 — Amélioration règle-based (facile)

improve\_rules:
	# Normalisation accents
	# Ajout nouveaux patterns linguistiques
	# Ordre d’apparition conservé

✔️ Gain rapide

✔️ Aucun coût de calcul

### 🧠 Phase 2 — NER CamemBERT fine-tuné

train\_ner:
	# Fine-tuning CamemBERT pour B-LOC / I-LOC

✔️ Suppression des règles

✔️ Meilleure généralisation

❌ Besoin dataset annoté

### 🔗 Phase 3 — Classification de relation

train\_relation:
	# Déterminer origine / destination

✔️ Gestion phrases complexes

✔️ Moins d’INVALID

❌ Complexité ML plus élevée

### 🚆 Phase 4 — Intégration graphe SNCF

pathfinding:
	# Construction graphe
	# Dijkstra / A\*

✔️ Itinéraires réalistes

✔️ Optimisation temps / distance

### 🎙️ Phase 5 — Speech-to-text (bonus)

speech:
	# Reconnaissance vocale

## 8\. Conclusion

Ce modèle constitue une **baseline solide**, justifiée scientifiquement, et parfaitement alignée avec les attentes du projet :

-   séparation claire NLP / logique métier
-   résultats interprétables
-   architecture évolutive
-   métriques mesurables

👉 Il est **idéal comme point de départ** avant fine-tuning CamemBERT.