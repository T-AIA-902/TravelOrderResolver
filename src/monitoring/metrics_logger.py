"""
Metrics logging for request-level and training-level performance tracking.

Records inference latency, model info, and resource usage to JSON files
for later analysis or dashboard consumption.
"""

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from loguru import logger


@dataclass
class RequestMetrics:
    """Metrics for a single inference request."""

    timestamp: str
    input_text: str
    model_name: str
    duration_s: float
    cpu_percent_avg: Optional[float] = None
    ram_peak_mb: Optional[float] = None
    gpu_peak_mb: Optional[float] = None
    carbon_kg: Optional[float] = None
    output: Optional[dict] = None
    extra: dict = field(default_factory=dict)


class MetricsLogger:
    """
    Logs per-request and per-training metrics to JSON files.

    Each log entry is appended as a JSON line (JSONL format)
    for easy streaming and analysis.

    Args:
        output_dir: Directory to store metrics files.
    """

    def __init__(self, output_dir: str = "reports/metrics"):
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._request_file = self._output_dir / "requests.jsonl"
        self._training_file = self._output_dir / "training.jsonl"

    def log_request(self, metrics: RequestMetrics) -> None:
        """
        Log metrics for a single inference request.

        Args:
            metrics: The request metrics to log.
        """
        entry = asdict(metrics)
        self._append_jsonl(self._request_file, entry)
        logger.debug(
            f"Request logged: model={metrics.model_name}, " f"duration={metrics.duration_s:.3f}s"
        )

    def log_training(self, metrics: dict[str, Any]) -> None:
        """
        Log metrics for a training run.

        Args:
            metrics: Dictionary of training metrics (epochs, loss, accuracy, etc.)
        """
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **metrics,
        }
        self._append_jsonl(self._training_file, entry)
        logger.debug("Training metrics logged")

    def start_request(self, input_text: str, model_name: str) -> "_RequestTimer":
        """
        Start timing a request. Use as a context manager.

        Args:
            input_text: The input text being processed.
            model_name: Name of the model being used.

        Returns:
            A timer context manager that logs metrics on exit.

        Example:
            with metrics_logger.start_request("De Paris à Lyon", "camembert") as timer:
                result = model.predict(text)
                timer.set_output(result)
        """
        return _RequestTimer(self, input_text, model_name)

    @staticmethod
    def _append_jsonl(path: Path, entry: dict) -> None:
        """Append a JSON entry to a JSONL file."""
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")

    def get_request_history(self, limit: int = 100) -> list[dict]:
        """
        Read the most recent request metrics.

        Args:
            limit: Maximum number of entries to return.

        Returns:
            List of metric dictionaries, most recent first.
        """
        if not self._request_file.exists():
            return []

        entries = []
        with open(self._request_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))

        return entries[-limit:][::-1]


# Energy estimation constants
# Average power draw during inference (Watts) — conservative estimates
_POWER_ESTIMATES_W: dict[str, float] = {
    "cpu_idle": 5.0,
    "cpu_light": 15.0,   # Regex, SpaCy
    "cpu_medium": 25.0,  # CamemBERT, Flan-T5
    "cpu_heavy": 45.0,   # Mistral 7B on CPU
    "gpu": 80.0,         # GPU inference
}
# France electricity carbon intensity (ADEME 2024): 52 g CO2/kWh
_CARBON_INTENSITY_KG_PER_KWH = 0.052


def estimate_carbon_kg(duration_s: float, model_name: str = "") -> float:
    """Estimate CO2 emissions from inference duration and model type.

    Uses average power draw estimates and French grid emission factor.
    """
    name = model_name.lower()
    if "mistral" in name:
        power_w = _POWER_ESTIMATES_W["cpu_heavy"]
    elif any(k in name for k in ("camembert", "flan", "t5", "bert")):
        power_w = _POWER_ESTIMATES_W["cpu_medium"]
    elif any(k in name for k in ("spacy", "regex")):
        power_w = _POWER_ESTIMATES_W["cpu_light"]
    else:
        power_w = _POWER_ESTIMATES_W["cpu_medium"]

    energy_kwh = power_w * duration_s / 3_600_000
    return energy_kwh * _CARBON_INTENSITY_KG_PER_KWH


class _RequestTimer:
    """Context manager that times a request and logs metrics."""

    def __init__(self, logger_instance: MetricsLogger, input_text: str, model_name: str):
        self._logger = logger_instance
        self._input_text = input_text
        self._model_name = model_name
        self._start: float = 0.0
        self._output: Optional[dict] = None
        self._extra: dict = {}
        self.carbon_kg: float = 0.0

    def set_output(self, output: dict) -> None:
        """Set the output result to include in metrics."""
        self._output = output

    def set_extra(self, **kwargs: Any) -> None:
        """Set extra metadata fields."""
        self._extra.update(kwargs)

    def __enter__(self) -> "_RequestTimer":
        self._start = time.time()
        return self

    def __exit__(self, *args: object) -> None:
        duration = time.time() - self._start
        self.carbon_kg = estimate_carbon_kg(duration, self._model_name)
        metrics = RequestMetrics(
            timestamp=datetime.now(timezone.utc).isoformat(),
            input_text=self._input_text,
            model_name=self._model_name,
            duration_s=round(duration, 4),
            carbon_kg=round(self.carbon_kg, 10),
            output=self._output,
            extra=self._extra,
        )
        self._logger.log_request(metrics)
