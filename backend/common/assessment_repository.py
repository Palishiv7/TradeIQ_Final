"""
Common Assessment Repository Components [DEPRECATED]

IMPORTANT: This module is being deprecated in favor of the repositories in
backend.assessments.base.repositories. New code should import directly from
those modules instead.

This module now serves as a compatibility layer to ensure existing
imports continue to work during the transition period.
"""

from abc import ABC, abstractmethod
import logging
from typing import Dict, List, Any, Optional, TypeVar, Generic, Type

from backend.common.logger import app_logger
from backend.common.base_assessment import Question, AssessmentSession
from backend.common.cache import CacheBackend, get_cache, CacheEntry, CacheResult

# Import from the canonical location
from backend.assessments.base.repositories import (
    QuestionRepository,
    SessionRepository,
    AssessmentRepository
)

# Module logger
logger = app_logger.getChild("common.assessment_repository")

# Type variables for generic implementations
T = TypeVar('T')  # Generic type for entities
T_Question = TypeVar('T_Question', bound=Question)
T_Session = TypeVar('T_Session', bound=AssessmentSession)

# Re-export types from assessments/base for backward compatibility
__all__ = [
    'Repository',
    'CachingRepository',
    'QuestionRepository',
    'SessionRepository',
    'AssessmentRepository'
]

# Create base repository classes for backward compatibility
class Repository(Generic[T], ABC):
    """Generic repository interface for domain objects."""
    
    @abstractmethod
    async def get(self, object_id: str) -> Optional[T]:
        """Get an object by ID."""
        pass
    
    @abstractmethod
    async def save(self, obj: T) -> bool:
        """Save an object."""
        pass
    
    @abstractmethod
    async def delete(self, object_id: str) -> bool:
        """Delete an object by ID."""
        pass
    
    @abstractmethod
    async def find(self, criteria: Dict[str, Any], limit: int = 100) -> List[T]:
        """Find objects matching criteria."""
        pass


class CachingRepository(Generic[T], Repository[T]):
    """
    Repository implementation with caching.
    
    This repository uses a cache-first strategy, checking the cache before
    hitting the backing store. It automatically updates the cache on writes.
    """
    
    def __init__(self, cache_backend: Optional[CacheBackend] = None, ttl_seconds: int = 3600):
        """
        Initialize the caching repository.
        
        Args:
            cache_backend: Cache backend implementation
            ttl_seconds: TTL for cached objects in seconds
        """
        self.cache = cache_backend or get_cache()
        self.ttl_seconds = ttl_seconds
        self.prefix = self.__class__.__name__
    
    def _build_key(self, object_id: str) -> str:
        """Build cache key for an object ID."""
        return f"{self.prefix}:{object_id}"
    
    async def get(self, object_id: str) -> Optional[T]:
        """
        Get an object by ID, using cache first.
        
        Args:
            object_id: Object identifier
            
        Returns:
            Object if found, None otherwise
        """
        # Try cache first
        cache_key = self._build_key(object_id)
        cached_obj = await self.cache.get(cache_key)
        
        if cached_obj is not None:
            logger.debug(f"Cache hit for {object_id}")
            return cached_obj
        
        # Cache miss, get from backing store
        logger.debug(f"Cache miss for {object_id}")
        obj = await self._get_from_store(object_id)
        
        # Update cache if object found
        if obj is not None:
            await self.cache.set(cache_key, obj, ttl=self.ttl_seconds)
        
        return obj
    
    async def save(self, obj: T) -> bool:
        """
        Save an object, updating the cache.
        
        Args:
            obj: Object to save
            
        Returns:
            True if successful, False otherwise
        """
        # Save to backing store
        result = await self._save_to_store(obj)
        
        if result:
            # Update cache
            object_id = self._get_object_id(obj)
            cache_key = self._build_key(object_id)
            await self.cache.set(cache_key, obj, ttl=self.ttl_seconds)
        
        return result
    
    async def delete(self, object_id: str) -> bool:
        """
        Delete an object, removing from cache.
        
        Args:
            object_id: Object identifier
            
        Returns:
            True if successful, False otherwise
        """
        # Delete from backing store
        result = await self._delete_from_store(object_id)
        
        if result:
            # Remove from cache
            cache_key = self._build_key(object_id)
            await self.cache.delete(cache_key)
        
        return result
    
    @abstractmethod
    async def _get_from_store(self, object_id: str) -> Optional[T]:
        """Get an object from the backing store."""
        pass
    
    @abstractmethod
    async def _save_to_store(self, obj: T) -> bool:
        """Save an object to the backing store."""
        pass
    
    @abstractmethod
    async def _delete_from_store(self, object_id: str) -> bool:
        """Delete an object from the backing store."""
        pass
    
    @abstractmethod
    def _get_object_id(self, obj: T) -> str:
        """Get the ID of an object."""
        pass 