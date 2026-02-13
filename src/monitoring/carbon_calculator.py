"""
Carbon footprint estimation using CodeCarbon.

Wraps the CodeCarbon EmissionsTracker to measure CO2 emissions
during inference and training operations.
"""

from dataclasses import dataclass
from typing import Optional

from loguru import logger


@dataclass
class CarbonMetrics:
    """Carbon emissions for a tracked operation."""

    emissions_kg: float
    energy_kwh: float
    duration_s: float
    country_iso_code: str


class CarbonCalculator:
    """
    Estimates carbon emissions using CodeCarbon.

    Can be used as a context manager around any code block.

    Args:
        country_iso_code: ISO 3166-1 alpha-3 country code. Default "FRA" (France).
        project_name: Project name for CodeCarbon tracking.

    Example:
        calculator = CarbonCalculator()
        with calculator:
            model.train(data)
        print(f"Emissions: {calculator.metrics.emissions_kg:.6f} kg CO2")
    """

    def __init__(
        self,
        country_iso_code: str = "FRA",
        project_name: str = "travel-order-resolver",
    ):
        self._country_iso_code = country_iso_code
        self._project_name = project_name
        self._tracker = None
        self.metrics: Optional[CarbonMetrics] = None

    def _ensure_codecarbon(self) -> None:
        """Check that codecarbon is available."""
        try:
            import codecarbon  # noqa: F401
        except ImportError:
            raise ImportError(
                "codecarbon is required for carbon tracking. "
                "Install with: poetry install --with ml"
            )

    def start(self) -> None:
        """Start tracking carbon emissions."""
        self._ensure_codecarbon()
        from codecarbon import EmissionsTracker

        self._tracker = EmissionsTracker(
            project_name=self._project_name,
            country_iso_code=self._country_iso_code,
            log_level="warning",
            save_to_file=False,
        )
        self._tracker.start()
        logger.debug("Carbon tracking started")

    def stop(self) -> CarbonMetrics:
        """
        Stop tracking and return carbon metrics.

        Returns:
            CarbonMetrics with emissions, energy, and duration.
        """
        if self._tracker is None:
            raise RuntimeError("Carbon tracking not started. Call start() first.")

        emissions_kg = self._tracker.stop()

        self.metrics = CarbonMetrics(
            emissions_kg=emissions_kg if emissions_kg is not None else 0.0,
            energy_kwh=self._tracker._total_energy.kWh if self._tracker._total_energy else 0.0,
            duration_s=self._tracker._last_measured_time or 0.0,
            country_iso_code=self._country_iso_code,
        )

        logger.debug(
            f"Carbon tracking stopped: {self.metrics.emissions_kg:.6f} kg CO2, "
            f"{self.metrics.energy_kwh:.6f} kWh"
        )
        return self.metrics

    def __enter__(self) -> "CarbonCalculator":
        self.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.stop()
