"""
Metrics Module

Provides services and repositories for managing user aggregate performance metrics.
"""

# Expose key components for easier import
from .models import UserMetrics
from .repository import UserMetricsRepository
from .service import UserMetricsService

__all__ = [
    "UserMetrics",
    "UserMetricsRepository",
    "UserMetricsService",
] 