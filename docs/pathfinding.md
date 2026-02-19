# Pathfinding - Documentation technique

## 1. Construction du graphe ferroviaire

### Sources de données

Le graphe est construit à partir de 3 fichiers SNCF open data :

| Fichier | Contenu | Usage |
|---------|---------|-------|
| `gares-de-voyageurs.json` | Gares voyageurs avec coordonnées GPS | Noeuds du graphe |
| `liste-des-gares.json` | Mapping gare ↔ ligne avec point kilométrique (pk) | Arêtes du graphe |
| `lignes-par-type.json` | Lignes ferroviaires avec type et géométrie | Détection LGV + visualisation |

### Approche hybride : topologique + géométrique

La construction du graphe combine deux approches complémentaires :

**Topologique (connexions)** : Les connexions entre gares sont déterminées par leur appartenance commune à une ligne ferroviaire. Chaque gare dans `liste-des-gares.json` est associée à un `code_ligne` et à un `pk` (point kilométrique). Les gares partageant la même ligne sont triées par pk croissant, puis les gares consécutives sont connectées par une arête. La distance entre deux gares est calculée par la différence de pk (ex: pk 602.834 - pk 590.200 = 12.634 km).

**Géométrique (coordonnées)** : Les coordonnées GPS des gares (issues de `gares-de-voyageurs.json`) servent à deux choses :
- L'heuristique des algorithmes A* et MOA* (distance à vol d'oiseau vers la destination)
- Le fallback de distance quand la différence de pk est aberrante (< 0.1 km ou > 500 km)

### Gestion des LGV

Les lignes à grande vitesse sont détectées par leur nom dans `lignes-par-type.json` (contient "LGV" ou "VITESSE"). Sur ces lignes, le poids des arêtes est divisé par 3 (`weight = dist_km / 3.0`) pour refléter le fait qu'un TGV parcourt la même distance ~3x plus vite qu'un train classique (280 km/h vs 80 km/h). Cela permet aux algorithmes de pathfinding de favoriser naturellement les itinéraires LGV.

### Correspondances urbaines

Dans les grandes villes (Paris, Lyon, Lille, Marseille, Bordeaux, Nantes), plusieurs gares coexistent sans être reliées par des lignes ferroviaires directes (ex: Paris Gare de Lyon et Paris Montparnasse). Des arêtes de transfert piéton sont ajoutées entre toutes les gares d'un même hub distantes de moins de 8 km, avec un poids fixe de 15 (équivalent à ~15 km de trajet, ce qui pénalise légèrement les correspondances).

### Structure du graphe

Le graphe est un **MultiGraph** NetworkX (plusieurs arêtes possibles entre deux mêmes noeuds, car deux gares peuvent être reliées par plusieurs lignes différentes).

Chaque **noeud** porte :
- `name` : nom de la gare
- `pos` : coordonnées (lat, lon)

Chaque **arête** porte :
- `weight` : coût pour le pathfinding (dist_km, divisé par 3 pour les LGV)
- `dist_km` : distance réelle en km
- `speed` : vitesse estimée (280 km/h LGV, 80 km/h classique, 5 km/h transfert)
- `line` : code de la ligne ferroviaire
- `type` : "TRAIN" ou "WALK"

---

## 2. Algorithmes de pathfinding

### A* (single-objective)

A* est une extension de Dijkstra qui utilise une **heuristique** pour guider la recherche vers la destination, réduisant le nombre de noeuds explorés.

**Principe** : À chaque étape, A* choisit le noeud qui minimise `f(n) = g(n) + h(n)` où :
- `g(n)` = coût réel du chemin depuis le départ jusqu'à `n`
- `h(n)` = estimation du coût restant de `n` jusqu'à la destination

**Heuristique utilisée** : `h(n) = geodesic(n, destination) / 3.0`

La distance géodésique (à vol d'oiseau) est divisée par 3 car le meilleur cas possible est une LGV en ligne droite (weight = dist/3). Cette heuristique est **admissible** : elle ne surestime jamais le coût réel, ce qui garantit que A* trouve le chemin optimal.

**Résultat** : Un unique chemin optimal (minimise le poids total).

### MOA* (Multi-Objective A*)

MOA* étend A* pour optimiser **simultanément plusieurs objectifs** sans les combiner en un score unique. Au lieu de chercher LE meilleur chemin, il cherche l'ensemble des chemins **Pareto-optimaux**.

#### Les 3 objectifs

| Objectif | Calcul | Unité |
|----------|--------|-------|
| Temps | `dist_km / speed` | heures |
| Distance | `dist_km` | km |
| Correspondances | +1 à chaque changement de ligne | count |

#### Dominance de Pareto

Un chemin A **domine** un chemin B si A est au moins aussi bon que B sur **tous** les objectifs, et strictement meilleur sur **au moins un**.

Exemple :
- `(3h, 500km, 1 corresp.)` domine `(4h, 600km, 2 corresp.)` → B est éliminé
- `(3h, 600km, 0 corresp.)` vs `(5h, 400km, 1 corresp.)` → aucun ne domine l'autre, les deux sont Pareto-optimaux

Un chemin est **Pareto-optimal** si aucun autre chemin ne le domine. L'ensemble de tous les chemins Pareto-optimaux forme la **frontière de Pareto**.

#### Fonctionnement de MOA*

1. **File de priorité** : triée par `f_time = g_time + h_time` (temps estimé total)

2. **Frontière de Pareto par noeud** : pour chaque noeud du graphe, on maintient l'ensemble des coûts non-dominés atteints. Un nouveau chemin arrivant à un noeud est élagué si son coût est dominé par un coût déjà enregistré pour ce noeud.

3. **Frontière de Pareto des solutions** : quand un chemin atteint la destination, il est ajouté aux solutions s'il n'est dominé par aucune solution existante. Les solutions existantes qu'il domine sont supprimées.

4. **Élagage** : un chemin en cours d'exploration est abandonné si :
   - Son coût est dominé par la frontière du noeud courant
   - Son coût est dominé par une solution déjà trouvée

5. **Heuristique multi-objectif admissible** :
   - Temps : `geodesic / 280` (meilleur cas : LGV en ligne droite)
   - Distance : `geodesic` (la distance ferroviaire est toujours >= la distance à vol d'oiseau)
   - Correspondances : `0` (optimiste, jamais surestimé)

**Résultat** : Jusqu'à 3 chemins Pareto-optimaux, chacun représentant un compromis différent entre temps, distance et correspondances.

#### Exemple concret

Pour Paris → Marseille, MOA* pourrait trouver :
- Chemin 1 : 3.2h, 900km, 0 corresp. (TGV direct, rapide mais plus long en km)
- Chemin 2 : 4.5h, 750km, 2 corresp. (route plus courte mais avec changements)
- Chemin 3 : 5.0h, 720km, 1 corresp. (compromis intermédiaire)

Aucun de ces chemins ne domine les autres : chacun est meilleur sur au moins un critère.

---

## 3. Gestion des villes intermédiaires

Quand l'utilisateur spécifie une ville de passage (ex: Paris → Marseille via Bordeaux), le système :

1. Calcule un chemin départ → intermédiaire
2. Calcule un chemin intermédiaire → destination
3. Concatène les deux chemins (en évitant de dupliquer le noeud intermédiaire)

Pour MOA*, les chemins Pareto de chaque leg sont combinés par **produit cartésien** : chaque chemin du leg 1 est apparié avec chaque chemin du leg 2, les coûts sont additionnés, puis les chemins dominés sont éliminés pour ne garder que la frontière de Pareto finale.

---

## 4. Visualisation

Les résultats sont affichés sur une carte interactive Folium :

- **A*/Dijkstra** : un seul chemin tracé en rouge, avec marqueurs départ (vert) et arrivée (rouge)
- **MOA*** : jusqu'à 3 chemins Pareto affichés en couleurs distinctes, avec :
  - Un contrôle de couches (layer toggle) pour activer/désactiver chaque chemin
  - Une légende indiquant temps, distance et correspondances pour chaque chemin
- **Correspondances** : segments piétons affichés en bleu pointillé
- Chaque segment est cliquable et affiche le numéro de ligne et la distance
