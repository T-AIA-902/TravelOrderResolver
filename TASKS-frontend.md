# Tâches Frontend — Travel Order Resolver

## Légende
- [ ] À faire
- [x] Fait
- [~] En cours

---

## Sprint actuel : Préparation backend + monitoring

### Documentation
- [x] Contrat API (`docs/API_CONTRACT.md`)
- [x] Fichier de suivi (`TASKS-frontend.md`)

### Monitoring — Frontend
- [x] Types TypeScript monitoring (`api/types.ts`)
- [x] Service API monitoring (`api/monitoring.ts`)
- [x] Composable `useMonitoring.ts`
- [x] Composant `ResourceGauge.vue`
- [x] Composant `CarbonCard.vue`
- [x] Composant `LatencyChart.vue`
- [x] Page `MonitoringPage.vue`
- [x] Route `/monitoring` dans le router
- [x] Lien dans la sidebar
- [x] Quick action dans le dashboard

### Intégration Pathfinding ↔ Carte
- [x] Vérification format `RouteSegment` (déjà compatible)
- [x] Documenté dans API_CONTRACT.md

### En attente du backend
- [ ] Implémenter le serveur FastAPI (`src/api/`)
- [ ] Endpoint `/api/health`
- [ ] Endpoint `/api/resolve`
- [ ] Endpoint `/api/nlp/*`
- [ ] Endpoint `/api/pathfinding/*`
- [ ] Endpoint `/api/speech/transcribe`
- [ ] Endpoint `/api/evaluation/*`
- [ ] Endpoint `/api/monitoring/*`
- [ ] Tests d'intégration frontend ↔ backend
