"""
Repository Base Module

This module provides base repository classes and exceptions for database operations.
It implements common functionality and patterns for data access.
"""

import logging
from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Optional, List, Dict, Any, Type, cast

# Set up logging
logger = logging.getLogger(__name__)

# Type variable for entity objects
T = TypeVar('T')

class NotFoundError(Exception):
    """Exception raised when an entity is not found."""
    
    def __init__(self, entity_type: str, entity_id: Any):
        """
        Initialize the exception.
        
        Args:
            entity_type: The type of entity that wasn't found
            entity_id: The ID of the entity that wasn't found
        """
        self.entity_type = entity_type
        self.entity_id = entity_id
        message = f"{entity_type} with ID {entity_id} not found"
        super().__init__(message)


class ConflictError(Exception):
    """Exception raised when an entity already exists."""
    
    def __init__(self, entity_type: str, entity_id: Any):
        """
        Initialize the exception.
        
        Args:
            entity_type: The type of entity that has a conflict
            entity_id: The ID of the entity that has a conflict
        """
        self.entity_type = entity_type
        self.entity_id = entity_id
        message = f"{entity_type} with ID {entity_id} already exists"
        super().__init__(message)


class ValidationError(Exception):
    """Exception raised when entity validation fails."""
    
    def __init__(self, entity_type: str, errors: Dict[str, str]):
        """
        Initialize the exception.
        
        Args:
            entity_type: The type of entity that failed validation
            errors: Dictionary of field errors
        """
        self.entity_type = entity_type
        self.errors = errors
        message = f"Validation failed for {entity_type}: {errors}"
        super().__init__(message)


class BaseRepository(Generic[T], ABC):
    """
    Abstract base repository for all entity types.
    
    This class provides a common interface and implementation for
    basic CRUD operations on entities.
    """
    
    def __init__(self, entity_type: str):
        """
        Initialize the base repository.
        
        Args:
            entity_type: The type of entity managed by this repository
        """
        self.entity_type = entity_type
    
    @abstractmethod
    async def get(self, entity_id: Any) -> T:
        """
        Get an entity by ID.
        
        Args:
            entity_id: The entity ID
            
        Returns:
            The entity
            
        Raises:
            NotFoundError: If the entity doesn't exist
        """
        pass
    
    @abstractmethod
    async def create(self, entity: T) -> T:
        """
        Create a new entity.
        
        Args:
            entity: The entity to create
            
        Returns:
            The created entity with any generated fields
            
        Raises:
            ConflictError: If an entity with the same ID already exists
            ValidationError: If the entity fails validation
        """
        pass
    
    @abstractmethod
    async def update(self, entity_id: Any, entity: T) -> T:
        """
        Update an existing entity.
        
        Args:
            entity_id: The entity ID
            entity: The entity with updated fields
            
        Returns:
            The updated entity
            
        Raises:
            NotFoundError: If the entity doesn't exist
            ValidationError: If the entity fails validation
        """
        pass
    
    @abstractmethod
    async def delete(self, entity_id: Any) -> bool:
        """
        Delete an entity by ID.
        
        Args:
            entity_id: The entity ID
            
        Returns:
            True if deleted, False if not found
        """
        pass
    
    @abstractmethod
    async def list(self, filters: Optional[Dict[str, Any]] = None, limit: int = 100, offset: int = 0) -> List[T]:
        """
        List entities matching the given filters.
        
        Args:
            filters: Optional filters to apply
            limit: Maximum number of entities to return
            offset: Offset for pagination
            
        Returns:
            List of entities
        """
        pass
    
    @abstractmethod
    async def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """
        Count entities matching the given filters.
        
        Args:
            filters: Optional filters to apply
            
        Returns:
            Number of matching entities
        """
        pass 