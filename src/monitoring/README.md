# Monitoring

Module de monitoring pour le suivi des ressources, de l'empreinte carbone et de l'infrastructure.

## Installation

```bash
# Tracking Python (CPU/RAM/Carbone)
pip install codecarbon psutil

# Stack Docker (Prometheus + Grafana + Gatus)
# Necessite Docker + Docker Compose
```

## Tracking Python

### Suivi CPU/RAM

```python
from src.monitoring import ResourceTracker

with ResourceTracker() as tracker:
    result = model.predict(data)

print(f"CPU: {tracker.usage.cpu_percent_avg:.1f}%")
print(f"RAM peak: {tracker.usage.ram_peak_mb:.0f} MB")
```

### Estimation empreinte carbone

```python
from src.monitoring import CarbonCalculator

with CarbonCalculator() as calc:
    result = model.predict(data)

print(f"Emissions: {calc.metrics.emissions_kg:.6f} kg CO2")
print(f"Energie: {calc.metrics.energy_kwh:.6f} kWh")
print(f"Duree: {calc.metrics.duration_s:.1f}s")
```

### Logging des metriques de requetes

```python
from src.monitoring import MetricsLogger

logger = MetricsLogger()
with logger.start_request() as timer:
    result = model.predict(data)
# Metriques sauvegardees dans reports/metrics/ (format JSONL)
```

## Stack Docker (Prometheus + Grafana + Gatus)

### Lancement

```bash
cd docker
docker compose --profile monitoring up -d
```

### Arret

```bash
cd docker
docker compose --profile monitoring down
```

### Services

| Service | URL | Description |
|---|---|---|
| Prometheus | http://localhost:49090 | Collecte de metriques (CPU/RAM/containers) |
| Grafana | http://localhost:43000 | Dashboards visuels (`admin`/`admin`) |
| Node Exporter | http://localhost:49100 | Metriques systeme |
| cAdvisor | http://localhost:48080 | Metriques containers Docker |
| Gatus | http://localhost:48081 | Health checks & uptime |

### Troubleshooting

L'erreur NVIDIA (`nvidia-container-cli: initialization error`) concerne uniquement le container `app` (GPU requis) et n'affecte pas le monitoring.

Si Gatus ou Grafana restent en etat "Created" sans demarrer :

```bash
# Demarrer manuellement un container bloque
docker start docker-gatus-1
docker start docker-grafana-1

# En cas de probleme persistant, recreer la stack
docker compose --profile monitoring down -v
docker compose --profile monitoring up -d

# Verifier l'etat des containers
docker compose --profile monitoring ps
```

## Architecture

```
src/monitoring/
  __init__.py            # Exports publics
  resource_tracker.py    # ResourceTracker (CPU/RAM via psutil)
  carbon_calculator.py   # CarbonCalculator (CO2 via CodeCarbon)
  metrics_logger.py      # MetricsLogger (JSONL logging)

docker/
  prometheus/prometheus.yml              # Config scrape Prometheus
  grafana/provisioning/datasources/      # Auto-provision datasource
  grafana/provisioning/dashboards/       # Auto-provision dashboards
  grafana/dashboards/system-monitoring.json  # Dashboard principal
  gatus/config.yaml                      # Health checks
```
