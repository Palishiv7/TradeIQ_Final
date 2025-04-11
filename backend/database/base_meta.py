"""
SQLAlchemy Base Metadata Configuration

This module provides the SQLAlchemy declarative base class that all models will inherit from.
It ensures consistent metadata configuration across all models.
"""

from sqlalchemy.orm import declarative_base
from sqlalchemy.ext.declarative import DeclarativeMeta

# Create the declarative base class that all models will inherit from
Base: DeclarativeMeta = declarative_base()

# Export the Base class for use in models
__all__ = ['Base'] 