"""
Performance Metrics Tracking Module

Provides lightweight instrumentation for tracking execution time
and throughput of key operations across the pipeline.
"""

import time
import logging
import functools
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class OperationMetric:
    """Metrics for a single tracked operation."""

    name: str
    total_calls: int = 0
    total_time: float = 0.0
    min_time: float = float("inf")
    max_time: float = 0.0

    @property
    def avg_time(self) -> float:
        if self.total_calls == 0:
            return 0.0
        return self.total_time / self.total_calls

    def record(self, elapsed: float) -> None:
        self.total_calls += 1
        self.total_time += elapsed
        if elapsed < self.min_time:
            self.min_time = elapsed
        if elapsed > self.max_time:
            self.max_time = elapsed


class PerformanceTracker:
    """Tracks execution time and call counts for named operations.

    Usage as a context manager::

        tracker = PerformanceTracker()
        with tracker.track("predict"):
            result = model.predict(X)

    Usage as a decorator::

        tracker = PerformanceTracker()

        @tracker.timed("train_models")
        def train():
            ...

    Retrieve a summary dict via ``tracker.summary()``.
    """

    def __init__(self) -> None:
        self._metrics: Dict[str, OperationMetric] = {}

    def _get_metric(self, name: str) -> OperationMetric:
        if name not in self._metrics:
            self._metrics[name] = OperationMetric(name=name)
        return self._metrics[name]

    @contextmanager
    def track(self, operation: str):
        """Context manager that records the elapsed time of *operation*."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            metric = self._get_metric(operation)
            metric.record(elapsed)
            logger.info(
                "[perf] %s completed in %.4fs (calls=%d, avg=%.4fs)",
                operation,
                elapsed,
                metric.total_calls,
                metric.avg_time,
            )

    def timed(self, operation: str):
        """Decorator that records the elapsed time of a function call."""

        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                with self.track(operation):
                    return func(*args, **kwargs)

            return wrapper

        return decorator

    def get(self, operation: str) -> Optional[OperationMetric]:
        """Return the metric for *operation*, or ``None``."""
        return self._metrics.get(operation)

    def summary(self) -> Dict[str, Dict]:
        """Return a summary dict of all tracked operations."""
        result = {}
        for name, m in self._metrics.items():
            result[name] = {
                "total_calls": m.total_calls,
                "total_time": round(m.total_time, 4),
                "avg_time": round(m.avg_time, 4),
                "min_time": round(m.min_time, 4) if m.total_calls > 0 else None,
                "max_time": round(m.max_time, 4) if m.total_calls > 0 else None,
            }
        return result

    def log_summary(self) -> None:
        """Log a summary of all tracked operations."""
        if not self._metrics:
            logger.info("[perf] No operations tracked yet")
            return
        logger.info("[perf] === Performance Summary ===")
        for name, m in self._metrics.items():
            logger.info(
                "[perf]   %-30s  calls=%-5d  total=%.4fs  avg=%.4fs  "
                "min=%.4fs  max=%.4fs",
                name,
                m.total_calls,
                m.total_time,
                m.avg_time,
                m.min_time if m.total_calls > 0 else 0.0,
                m.max_time,
            )

    def reset(self) -> None:
        """Clear all tracked metrics."""
        self._metrics.clear()
