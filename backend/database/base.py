"""
SQLAlchemy Base Configuration

This module provides the SQLAlchemy declarative base and session configuration
for the TradeIQ backend database models.
"""

from typing import Any, Dict
import logging
from sqlalchemy import MetaData
from sqlalchemy.ext.declarative import declarative_base

# Configure module logger
logger = logging.getLogger(__name__)

# Configure naming convention for constraints
convention = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s"
}

# Create metadata with naming convention
metadata = MetaData(naming_convention=convention)

# Create the declarative base class with configured metadata
Base = declarative_base(metadata=metadata)

# Export all symbols
class ModelBase(Base):
    """Base class for all SQLAlchemy models."""
    
    __abstract__ = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model instance to dictionary."""
        return {
            column.name: getattr(self, column.name)
            for column in self.__table__.columns
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Any:
        """Create model instance from dictionary."""
        return cls(**{
            k: v for k, v in data.items()
            if k in cls.__table__.columns
        })
    
    def update(self, data: Dict[str, Any]) -> None:
        """Update model instance from dictionary."""
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value) 