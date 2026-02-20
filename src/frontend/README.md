# Frontend — Travel Order Resolver

Interface Vue 3 pour le projet Travel Order Resolver : page d'accueil, chat NLP avec carte, évaluation des modèles, rapports benchmark et dataset.

## Stack

- **Vue 3** (Composition API, `<script setup>`)
- **TypeScript** (strict, `noUncheckedIndexedAccess`)
- **Vite 7** — dev server / build
- **Tailwind CSS 4** — styling
- **Vue Router 4** — navigation
- **Leaflet** + vue-leaflet — carte ferroviaire
- **Lucide** — icônes

## Lancer le projet

### Docker (recommandé)

Le plus simple pour lancer le frontend sans installer Node :

```bash
# Depuis la racine du projet
make front-run       # Build + lance sur http://localhost:3000
make front-stop      # Stopper
make front-rebuild   # Rebuild + relance après modification du code
```

Ouvrir http://localhost:3000

Le proxy `/api/` redirige vers le backend (`app:8000` en docker-compose). Si le backend n'est pas lancé, le frontend démarre quand même — seuls les appels API échoueront.

### Dev local (avec Node)

```bash
cd src/frontend
npm install
npm run dev
```

Le serveur de dev démarre sur `http://localhost:5173` avec hot-reload.

L'URL de l'API backend se configure via `VITE_API_URL` dans `.env` (par défaut : même origine).

## Scripts

| Commande            | Description                              |
| ------------------- | ---------------------------------------- |
| `npm run dev`       | Serveur de dev Vite                      |
| `npm run build`     | Type-check + build production            |
| `npm run check`     | Type-check + ESLint + Prettier (CI gate) |
| `npm run lint`      | ESLint seul                              |
| `npm run lint:fix`  | ESLint avec auto-fix                     |
| `npm run format`    | Prettier — formater les fichiers         |
| `npm run typecheck` | vue-tsc seul                             |

## Pages

| Route         | Page             | Description                                          |
| ------------- | ---------------- | ---------------------------------------------------- |
| `/`           | DashboardPage    | Page d'accueil avec liens rapides vers chaque section |
| `/chat`       | ChatPage         | Chat NLP (50/50 : conversation + carte ferroviaire)   |
| `/evaluation` | EvaluationPage   | Lancer une évaluation, voir les métriques par onglet  |
| `/dataset`    | DatasetPage      | **À implémenter** — répartition et métriques du dataset |
| `/rapports`   | RapportsPage     | Benchmark par étape du pipeline                       |

## Structure

```
src/
├── api/
│   ├── client.ts          # apiGet / apiPost / apiPostForm (fetch wrapper)
│   ├── types.ts           # Tous les types TS partagés (NLP, Eval, Pathfinding…)
│   ├── resolve.ts         # POST /resolve
│   ├── evaluation.ts      # GET/POST /eval/*
│   ├── speech.ts          # POST /speech/transcribe
│   └── pathfinding.ts     # POST /pathfinding/route, GET /pathfinding/stations
├── components/
│   ├── common/            # LoadingSpinner, ErrorAlert, StatusDot
│   ├── chat/              # ChatWindow, ChatMessage, ChatInput, RecordButton (STT)
│   ├── evaluation/        # MetricsTable, ConfusionMatrixView, EvalProgress
│   ├── map/               # RailwayMap, RouteLayer, StationMarker (Leaflet)
│   ├── nlp/               # PipelineVisualizer, IntentBadge, EntityHighlight
│   └── layout/            # AppLayout, AppSidebar, AppHeader
├── composables/
│   ├── useResolve.ts      # Appel pipeline complet (resolve)
│   ├── useEvaluation.ts   # Gestion éval (rapports, lancement, polling statut)
│   ├── useMapRoute.ts     # Extraction route depuis la réponse NLP → segments Leaflet
│   └── useAudioRecorder.ts # Enregistrement micro → envoi STT
├── pages/                 # Pages routées (voir tableau ci-dessus)
└── router/                # Configuration Vue Router
```

## Layout

- **Home (`/`)** : pas de sidebar ni header — page d'accueil centrée avec liens rapides + Grafana
- **Autres pages** : sidebar fixe à gauche (`w-56`) + header fixe en haut (`h-16`) + zone de contenu scrollable

## Données mockées (TODO backend)

### Page Évaluation

`EvaluationPage.vue` contient un `MOCK_REPORT` en dur qui simule un rapport d'évaluation complet (langue, intent, entités, entités+fuzzy).

**Quand le backend éval sera prêt :**

1. Supprimer `MOCK_REPORT` dans `EvaluationPage.vue`
2. Supprimer le fallback `?? MOCK_REPORT` dans le computed `currentReport`
3. Les données réelles viendront de `useEvaluation()` → `reportDetail` (API `/eval/reports/:id`)

### Page Rapports

`RapportsPage.vue` contient un `benchmarkRows` en dur avec les meilleurs résultats par étape du pipeline. À remplacer par un appel API quand disponible.

### Page Dataset

`DatasetPage.vue` est un placeholder. Contenu prévu :

- Courbes de répartition du dataset (langues, intents, entités)
- Métriques sur le dataset (nombre d'échantillons, distribution des classes, etc.)
- À ajouter quand les métriques seront définies

## API attendue

Le frontend consomme les endpoints suivants (voir `src/api/` pour les types) :

| Endpoint                     | Méthode | Description                        |
| ---------------------------- | ------- | ---------------------------------- |
| `/health`                    | GET     | Health check (version, GPU, etc.)  |
| `/resolve`                   | POST    | Pipeline NLP complet + pathfinding |
| `/resolve/language`          | POST    | Détection de langue seule          |
| `/resolve/intent`            | POST    | Classification d'intent seule      |
| `/resolve/entities`          | POST    | Extraction d'entités seule         |
| `/speech/transcribe`         | POST    | Transcription audio (STT)          |
| `/pathfinding/route`         | POST    | Calcul d'itinéraire                |
| `/pathfinding/stations`      | GET     | Liste des gares                    |
| `/eval/reports`              | GET     | Liste des rapports d'évaluation    |
| `/eval/reports/:id`          | GET     | Détail d'un rapport                |
| `/eval/run`                  | POST    | Lancer une évaluation              |
| `/eval/status/:task_id`      | GET     | Statut d'une évaluation en cours   |

## Conventions

- Composants en PascalCase, fichiers `.vue`
- Composables préfixés `use` dans `composables/`
- ESLint 9 flat config + Prettier (`semi: false`, `singleQuote: true`, `printWidth: 100`)
- Quality gate avant commit : `npm run check` (vue-tsc + eslint + prettier)
