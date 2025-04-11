"""
Middleware Package

This package contains middleware components for the TradeIQ assessment platform.
"""

from backend.middleware.legacy_tracking import LegacyAPITrackingMiddleware

__all__ = ['LegacyAPITrackingMiddleware'] 