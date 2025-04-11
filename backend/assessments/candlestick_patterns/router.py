"""
Candlestick Pattern Assessment Router

This module exports the router from the candlestick_controller module.
"""

import logging
from backend.assessments.candlestick_patterns.candlestick_controller import router

logger = logging.getLogger(__name__)
logger.info(f"Candlestick router loaded with {len(router.routes)} routes")

__all__ = ['router']
