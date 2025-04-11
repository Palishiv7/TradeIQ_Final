"""
Database Module

This module provides database configuration and models for the TradeIQ backend.
"""

from backend.database.base import Base, ModelBase, metadata

__all__ = ['Base', 'ModelBase', 'metadata'] 