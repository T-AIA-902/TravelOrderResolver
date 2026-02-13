# Frontend — TravelOrderResolver

Interface Vue 3 pour le projet TravelOrderResolver (dashboard, chat, évaluation).

## Stack

- **Vue 3** (Composition API, `<script setup>`)
- **TypeScript** (strict, `noUncheckedIndexedAccess`)
- **Vite 7** — dev server / build
- **Tailwind CSS 4** — styling
- **Vue Router 4** — navigation
- **ECharts** + vue-echarts — graphiques dashboard
- **Lucide** — icônes

## Lancer le projet

```bash
cd src/frontend
npm install
npm run dev
```

Le serveur de dev démarre sur `http://localhost:5173`.

L'URL de l'API backend se configure via `VITE_API_URL` dans `.env` (par défaut : même origine).

## Scripts

| Commande           | Description                              |
| ------------------- | ---------------------------------------- |
| `npm run dev`       | Serveur de dev Vite                      |
| `npm run build`     | Type-check + build production            |
| `npm run check`     | Type-check + ESLint + Prettier (CI gate) |
| `npm run lint`      | ESLint seul                              |
| `npm run lint:fix`  | ESLint avec auto-fix                     |
| `npm run format`    | Prettier — formater les fichiers         |
| `npm run typecheck` | vue-tsc seul                             |

## Structure

```
src/
├── api/            # Client HTTP et types partagés (fetch vers le backend)
├── components/
│   ├── common/     # LoadingSpinner, ErrorAlert
│   ├── chat/       # ChatWindow, ChatMessage, RecordButton (STT)
│   ├── evaluation/ # MetricsTable, ConfusionMatrixView, EvalProgress
│   └── layout/     # AppLayout, AppSidebar
├── composables/    # useChat, useEvaluation, useSpeech, etc.
├── pages/          # Pages routées (Dashboard, Chat, Evaluation)
└── router/         # Configuration Vue Router
```

## Pages

| Route         | Page                | Description                              |
| ------------- | ------------------- | ---------------------------------------- |
| `/`           | DashboardPage       | Vue d'ensemble (graphiques ECharts)      |
| `/chat`       | ChatPage            | Chat conversationnel + STT              |
| `/evaluation` | EvaluationPage      | Lancer une évaluation, voir les métriques |

## Données mockées (TODO)

La page **Évaluation** (`src/pages/EvaluationPage.vue`) contient un `MOCK_REPORT` en dur qui simule un rapport d'évaluation. Ces données servent à prévisualiser l'affichage des métriques tant que le backend n'est pas connecté.

**À faire quand le backend éval sera prêt :**
1. Supprimer la constante `MOCK_REPORT` dans `EvaluationPage.vue`
2. Supprimer le fallback `?? MOCK_REPORT` dans le computed `currentReport`
3. Les données réelles viendront de `useEvaluation()` → `reportDetail` (API `/eval/reports/:id`)

## API attendue

Le frontend consomme les endpoints suivants (voir `src/api/` pour le détail) :

- `POST /resolve/language` — détection de langue
- `POST /resolve/intent` — classification d'intent
- `POST /resolve/entities` — extraction d'entités
- `POST /speech/transcribe` — transcription audio (STT)
- `GET /eval/reports` — liste des rapports d'évaluation
- `GET /eval/reports/:id` — détail d'un rapport
- `POST /eval/run` — lancer une évaluation
- `GET /eval/status/:task_id` — statut d'une évaluation en cours
