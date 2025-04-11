"""
Metrics Framework

This module provides a lightweight metrics collection system that can track
application metrics and send them to various backends. It supports counters,
gauges, histograms, and timers with labels.
"""

import time
import logging
import threading
import contextlib
import functools
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Callable, TypeVar, cast
from abc import ABC, abstractmethod
from collections import defaultdict

from backend.common.logger import app_logger

# Module logger
logger = app_logger.getChild("metrics")

# Type variables
F = TypeVar('F', bound=Callable[..., Any])
T = TypeVar('T')

class MetricType(Enum):
    """Types of metrics supported by the system."""
    COUNTER = "counter"  # Counts things, only increases
    GAUGE = "gauge"      # Current value, can go up or down
    HISTOGRAM = "histogram"  # Distribution of values
    TIMER = "timer"      # Time measurement


class MetricsBackend(ABC):
    """Abstract base class for metrics backends."""
    
    @abstractmethod
    def increment_counter(self, name: str, value: float = 1.0, labels: Dict[str, str] = None) -> None:
        """Increment a counter."""
        pass
    
    @abstractmethod
    def set_gauge(self, name: str, value: float, labels: Dict[str, str] = None) -> None:
        """Set a gauge value."""
        pass
    
    @abstractmethod
    def observe_histogram(self, name: str, value: float, labels: Dict[str, str] = None) -> None:
        """Observe a value for a histogram."""
        pass
    
    @abstractmethod
    def observe_timer(self, name: str, value_ms: float, labels: Dict[str, str] = None) -> None:
        """Record a timing observation."""
        pass
    
    @abstractmethod
    def flush(self) -> None:
        """Flush metrics to the backend."""
        pass


class LoggingMetricsBackend(MetricsBackend):
    """Metrics backend that logs metrics to a logger."""
    
    def __init__(self, logger: logging.Logger = None):
        """
        Initialize the logging metrics backend.
        
        Args:
            logger: Optional logger to use (defaults to metrics logger)
        """
        self.logger = logger or app_logger.getChild("metrics")
    
    def increment_counter(self, name: str, value: float = 1.0, labels: Dict[str, str] = None) -> None:
        """Increment a counter and log it."""
        labels_str = self._format_labels(labels)
        self.logger.info(f"METRIC_COUNTER {name}{labels_str} {value}")
    
    def set_gauge(self, name: str, value: float, labels: Dict[str, str] = None) -> None:
        """Set a gauge value and log it."""
        labels_str = self._format_labels(labels)
        self.logger.info(f"METRIC_GAUGE {name}{labels_str} {value}")
    
    def observe_histogram(self, name: str, value: float, labels: Dict[str, str] = None) -> None:
        """Observe a value for a histogram and log it."""
        labels_str = self._format_labels(labels)
        self.logger.info(f"METRIC_HISTOGRAM {name}{labels_str} {value}")
    
    def observe_timer(self, name: str, value_ms: float, labels: Dict[str, str] = None) -> None:
        """Record a timing observation and log it."""
        labels_str = self._format_labels(labels)
        self.logger.info(f"METRIC_TIMER {name}{labels_str} {value_ms:.2f}ms")
    
    def flush(self) -> None:
        """Flush metrics to the backend (no-op for logging)."""
        pass
    
    def _format_labels(self, labels: Dict[str, str] = None) -> str:
        """Format labels for logging."""
        if not labels:
            return ""
        return "{" + ", ".join(f"{k}={v}" for k, v in labels.items()) + "}"


class InMemoryMetricsBackend(MetricsBackend):
    """
    Metrics backend that stores metrics in memory.
    
    This is useful for testing or for aggregating metrics before flushing them
    to another backend.
    """
    
    def __init__(self):
        """Initialize the in-memory metrics backend."""
        self.counters = defaultdict(float)
        self.gauges = {}
        self.histograms = defaultdict(list)
        self.timers = defaultdict(list)
        self.lock = threading.RLock()
    
    def increment_counter(self, name: str, value: float = 1.0, labels: Dict[str, str] = None) -> None:
        """Increment a counter in memory."""
        key = self._get_key(name, labels)
        with self.lock:
            self.counters[key] += value
    
    def set_gauge(self, name: str, value: float, labels: Dict[str, str] = None) -> None:
        """Set a gauge value in memory."""
        key = self._get_key(name, labels)
        with self.lock:
            self.gauges[key] = value
    
    def observe_histogram(self, name: str, value: float, labels: Dict[str, str] = None) -> None:
        """Observe a value for a histogram in memory."""
        key = self._get_key(name, labels)
        with self.lock:
            self.histograms[key].append(value)
    
    def observe_timer(self, name: str, value_ms: float, labels: Dict[str, str] = None) -> None:
        """Record a timing observation in memory."""
        key = self._get_key(name, labels)
        with self.lock:
            self.timers[key].append(value_ms)
    
    def flush(self) -> None:
        """Clear all stored metrics."""
        with self.lock:
            self.counters.clear()
            self.gauges.clear()
            self.histograms.clear()
            self.timers.clear()
    
    def get_counter(self, name: str, labels: Dict[str, str] = None) -> float:
        """Get the current value of a counter."""
        key = self._get_key(name, labels)
        with self.lock:
            return self.counters.get(key, 0.0)
    
    def get_gauge(self, name: str, labels: Dict[str, str] = None) -> Optional[float]:
        """Get the current value of a gauge."""
        key = self._get_key(name, labels)
        with self.lock:
            return self.gauges.get(key)
    
    def get_histogram_values(self, name: str, labels: Dict[str, str] = None) -> List[float]:
        """Get all values observed for a histogram."""
        key = self._get_key(name, labels)
        with self.lock:
            return self.histograms.get(key, [])[:]
    
    def get_timer_values(self, name: str, labels: Dict[str, str] = None) -> List[float]:
        """Get all timing observations."""
        key = self._get_key(name, labels)
        with self.lock:
            return self.timers.get(key, [])[:]
    
    def _get_key(self, name: str, labels: Dict[str, str] = None) -> str:
        """Generate a unique key for a metric with labels."""
        if not labels:
            return name
        
        # Sort labels by key to ensure consistent key generation
        sorted_labels = [f"{k}:{v}" for k, v in sorted(labels.items())]
        return f"{name}:{','.join(sorted_labels)}"


class MultiMetricsBackend(MetricsBackend):
    """Metrics backend that forwards metrics to multiple other backends."""
    
    def __init__(self, backends: List[MetricsBackend]):
        """
        Initialize the multi-backend.
        
        Args:
            backends: List of metrics backends to forward to
        """
        self.backends = backends
    
    def increment_counter(self, name: str, value: float = 1.0, labels: Dict[str, str] = None) -> None:
        """Increment a counter in all backends."""
        for backend in self.backends:
            backend.increment_counter(name, value, labels)
    
    def set_gauge(self, name: str, value: float, labels: Dict[str, str] = None) -> None:
        """Set a gauge value in all backends."""
        for backend in self.backends:
            backend.set_gauge(name, value, labels)
    
    def observe_histogram(self, name: str, value: float, labels: Dict[str, str] = None) -> None:
        """Observe a value for a histogram in all backends."""
        for backend in self.backends:
            backend.observe_histogram(name, value, labels)
    
    def observe_timer(self, name: str, value_ms: float, labels: Dict[str, str] = None) -> None:
        """Record a timing observation in all backends."""
        for backend in self.backends:
            backend.observe_timer(name, value_ms, labels)
    
    def flush(self) -> None:
        """Flush metrics in all backends."""
        for backend in self.backends:
            backend.flush()


class MetricsService:
    """
    Service for collecting and reporting metrics.
    
    This class provides a facade for the metrics system, making it easy to
    record metrics without directly interacting with backends.
    """
    
    _instance = None  # Singleton instance
    
    @classmethod
    def get_instance(cls) -> 'MetricsService':
        """Get the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self, backend: MetricsBackend = None):
        """
        Initialize the metrics service.
        
        Args:
            backend: Optional metrics backend to use
        """
        self.backend = backend or LoggingMetricsBackend()
    
    def set_backend(self, backend: MetricsBackend) -> None:
        """Set the metrics backend."""
        self.backend = backend
    
    def counter(self, name: str, value: float = 1.0, labels: Dict[str, str] = None) -> None:
        """Increment a counter."""
        self.backend.increment_counter(name, value, labels)
    
    def gauge(self, name: str, value: float, labels: Dict[str, str] = None) -> None:
        """Set a gauge value."""
        self.backend.set_gauge(name, value, labels)
    
    def histogram(self, name: str, value: float, labels: Dict[str, str] = None) -> None:
        """Record a value in a histogram."""
        self.backend.observe_histogram(name, value, labels)
    
    def timer(self, name: str, value_ms: float, labels: Dict[str, str] = None) -> None:
        """Record a timing in milliseconds."""
        self.backend.observe_timer(name, value_ms, labels)
    
    @contextlib.contextmanager
    def timer_context(self, name: str, labels: Dict[str, str] = None):
        """
        Context manager for timing a block of code.
        
        Args:
            name: Metric name
            labels: Optional labels
            
        Yields:
            None
        """
        start_time = time.time()
        try:
            yield
        finally:
            elapsed_ms = (time.time() - start_time) * 1000.0
            self.timer(name, elapsed_ms, labels)
    
    def flush(self) -> None:
        """Flush metrics to the backend."""
        self.backend.flush()


def timed(name: str, labels: Optional[Dict[str, str]] = None):
    """
    Decorator for timing a function.
    
    Args:
        name: Metric name
        labels: Optional static labels
        
    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            metrics = get_metrics_service()
            with metrics.timer_context(name, labels):
                return func(*args, **kwargs)
        return cast(F, wrapper)
    return decorator


def counted(name: str, labels: Optional[Dict[str, str]] = None):
    """
    Decorator for counting function calls.
    
    Args:
        name: Metric name
        labels: Optional static labels
        
    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            metrics = get_metrics_service()
            metrics.counter(name, 1.0, labels)
            return func(*args, **kwargs)
        return cast(F, wrapper)
    return decorator


# Default metrics service
_default_service = MetricsService()

def get_metrics_service() -> MetricsService:
    """Get the default metrics service."""
    return MetricsService.get_instance()


def set_metrics_backend(backend: MetricsBackend) -> None:
    """Set the backend for the default metrics service."""
    get_metrics_service().set_backend(backend)
