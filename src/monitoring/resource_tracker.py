"""
System resource tracking for CPU, RAM, and GPU usage.

Provides a context manager to measure resource consumption
during any operation (inference, training, etc.).
"""

import threading
import time
from dataclasses import dataclass, field
from typing import Optional

from loguru import logger


@dataclass
class ResourceSnapshot:
    """A snapshot of system resource usage."""

    cpu_percent: float
    ram_used_mb: float
    ram_total_mb: float
    ram_percent: float
    gpu_used_mb: Optional[float] = None
    gpu_total_mb: Optional[float] = None
    gpu_percent: Optional[float] = None
    gpu_name: Optional[str] = None


@dataclass
class ResourceUsage:
    """Resource usage measured over a time period."""

    duration_s: float
    cpu_percent_avg: float
    ram_peak_mb: float
    ram_avg_mb: float
    gpu_peak_mb: Optional[float] = None
    gpu_avg_mb: Optional[float] = None
    snapshots: list[ResourceSnapshot] = field(default_factory=list)


class ResourceTracker:
    """
    Tracks CPU, RAM, and GPU usage over a period of time.

    Can be used as a context manager to measure resource consumption
    of any code block.

    Example:
        tracker = ResourceTracker()
        with tracker:
            # run inference or training
            result = model.predict(data)
        print(tracker.usage.cpu_percent_avg)
        print(tracker.usage.ram_peak_mb)
    """

    def __init__(self, interval: float = 0.5):
        """
        Args:
            interval: Sampling interval in seconds.
        """
        self._interval = interval
        self._snapshots: list[ResourceSnapshot] = []
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None
        self._thread: Optional["threading.Thread"] = None
        self._running = False
        self.usage: Optional[ResourceUsage] = None

    @staticmethod
    def _get_snapshot() -> ResourceSnapshot:
        """Take a single resource snapshot."""
        try:
            import psutil
        except ImportError:
            raise ImportError(
                "psutil is required for resource tracking. " "Install with: pip install psutil"
            )

        cpu_percent = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()

        snapshot = ResourceSnapshot(
            cpu_percent=cpu_percent,
            ram_used_mb=mem.used / (1024 * 1024),
            ram_total_mb=mem.total / (1024 * 1024),
            ram_percent=mem.percent,
        )

        # Try to get GPU info
        try:
            import torch

            if torch.cuda.is_available():
                snapshot.gpu_name = torch.cuda.get_device_name(0)
                allocated = torch.cuda.memory_allocated(0)
                total = torch.cuda.get_device_properties(0).total_memory
                snapshot.gpu_used_mb = allocated / (1024 * 1024)
                snapshot.gpu_total_mb = total / (1024 * 1024)
                snapshot.gpu_percent = (allocated / total) * 100 if total > 0 else 0.0
        except ImportError:
            pass

        return snapshot

    def _sample_loop(self) -> None:
        """Background sampling loop."""
        import psutil

        # Initialize CPU percent measurement
        psutil.cpu_percent(interval=None)

        while self._running:
            try:
                self._snapshots.append(self._get_snapshot())
            except Exception as e:
                logger.warning(f"Resource sampling error: {e}")
            time.sleep(self._interval)

    def start(self) -> None:
        """Start tracking resources in a background thread."""
        import threading

        import psutil

        # Initialize CPU measurement baseline
        psutil.cpu_percent(interval=None)

        self._snapshots = []
        self._running = True
        self._start_time = time.time()

        # Take an initial snapshot synchronously
        self._snapshots.append(self._get_snapshot())

        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        logger.debug("Resource tracking started")

    def stop(self) -> ResourceUsage:
        """Stop tracking and compute usage summary."""
        self._running = False
        self._end_time = time.time()

        if self._thread is not None:
            self._thread.join(timeout=2.0)

        # Take a final snapshot to ensure we have end-of-operation data
        try:
            self._snapshots.append(self._get_snapshot())
        except Exception:
            pass

        duration = (self._end_time - self._start_time) if self._start_time else 0.0

        cpu_values = [s.cpu_percent for s in self._snapshots]
        ram_values = [s.ram_used_mb for s in self._snapshots]

        gpu_peak = None
        gpu_avg = None
        gpu_values = [s.gpu_used_mb for s in self._snapshots if s.gpu_used_mb is not None]
        if gpu_values:
            gpu_peak = max(gpu_values)
            gpu_avg = sum(gpu_values) / len(gpu_values)

        self.usage = ResourceUsage(
            duration_s=duration,
            cpu_percent_avg=sum(cpu_values) / len(cpu_values),
            ram_peak_mb=max(ram_values),
            ram_avg_mb=sum(ram_values) / len(ram_values),
            gpu_peak_mb=gpu_peak,
            gpu_avg_mb=gpu_avg,
            snapshots=self._snapshots,
        )

        logger.debug(
            f"Resource tracking stopped: {duration:.1f}s, "
            f"CPU avg {self.usage.cpu_percent_avg:.1f}%, "
            f"RAM peak {self.usage.ram_peak_mb:.0f}MB"
        )
        return self.usage

    def __enter__(self) -> "ResourceTracker":
        self.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.stop()
