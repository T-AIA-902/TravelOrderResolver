"""Monitoring endpoints: metrics, resources, and carbon footprint."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from src.api.dependencies import get_metrics_logger
from src.monitoring.resource_tracker import ResourceTracker

router = APIRouter(prefix="/monitoring", tags=["monitoring"])


@router.get("/metrics")
def get_metrics(limit: int = 100, metrics_logger=Depends(get_metrics_logger)):
    """Return recent request metrics with aggregate statistics."""
    requests = metrics_logger.get_request_history(limit=limit)

    filtered_requests = [
        {k: v for k, v in entry.items() if k not in ("output", "extra")} for entry in requests
    ]

    total_requests = len(requests)
    avg_latency_ms = 0.0
    if total_requests > 0:
        avg_latency_ms = (
            sum(entry.get("duration_s", 0) for entry in requests) / total_requests * 1000
        )

    return {
        "requests": filtered_requests,
        "total_requests": total_requests,
        "avg_latency_ms": round(avg_latency_ms, 2),
    }


@router.get("/resources")
def get_resources():
    """Return a live snapshot of CPU, RAM, and GPU usage."""
    snapshot = ResourceTracker._get_snapshot()
    return {
        "cpu_percent": round(snapshot.cpu_percent, 1),
        "ram_used_mb": round(snapshot.ram_used_mb, 1),
        "ram_total_mb": round(snapshot.ram_total_mb, 1),
        "ram_percent": round(snapshot.ram_percent, 1),
        "gpu_used_mb": round(snapshot.gpu_used_mb, 1) if snapshot.gpu_used_mb is not None else None,
        "gpu_total_mb": (
            round(snapshot.gpu_total_mb, 1) if snapshot.gpu_total_mb is not None else None
        ),
        "gpu_percent": (
            round(snapshot.gpu_percent, 1) if snapshot.gpu_percent is not None else None
        ),
    }


@router.get("/carbon")
def get_carbon(metrics_logger=Depends(get_metrics_logger)):
    """Return cumulative carbon emissions and energy estimates."""
    requests = metrics_logger.get_request_history(limit=999999)
    total_requests = len(requests)

    total_emissions = sum(entry.get("carbon_kg") or 0 for entry in requests)
    total_energy = total_emissions * 50  # 1 kg CO2 ≈ 50 kWh in France

    return {
        "total_emissions_kg": round(total_emissions, 6),
        "total_energy_kwh": round(total_energy, 4),
        "total_requests": total_requests,
        "country": "FRA",
    }
