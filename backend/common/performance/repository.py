"""
Performance Repository

This module provides repository classes for persisting performance tracking data,
including user performance metrics, difficulty settings, and spaced repetition data.
"""

import json
import uuid
import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Set, TypeVar, Generic, Type, Callable

from backend.common.logger import app_logger
from backend.common.cache import get_cache, configure_cache_sync
from backend.common.performance.tracker import PerformanceTracker
from backend.common.performance.difficulty import DifficultyManager, AdaptiveDifficultyEngine
from backend.common.performance.forgetting import SpacedRepetitionScheduler

# Type variables for generics
T = TypeVar('T')

# Module logger
logger = app_logger.getChild("performance.repository")

# Initialize the default cache - ensures we have a memory cache available
try:
    # Configure with synchronous cache implementation that doesn't need an event loop
    cache_manager = configure_cache_sync(
        memory_cache_size=10000,
        write_through=True
    )
    logger.info(f"Performance repository module initialized default cache with {len(cache_manager.backends)} backends")
except Exception as e:
    logger.warning(f"Failed to initialize default cache: {e}")


class PerformanceRepository(ABC, Generic[T]):
    """
    Abstract base class for performance data repositories.
    
    Provides a common interface for storing and retrieving performance data,
    regardless of the specific storage mechanism used.
    """
    
    @abstractmethod
    def get(self, id: str) -> Optional[T]:
        """
        Retrieve an entity by ID.
        
        Args:
            id: Entity identifier
            
        Returns:
            Entity or None if not found
        """
        pass
    
    @abstractmethod
    def save(self, entity: T) -> str:
        """
        Save an entity to the repository.
        
        Args:
            entity: Entity to save
            
        Returns:
            Entity identifier
        """
        pass
    
    @abstractmethod
    def delete(self, id: str) -> bool:
        """
        Delete an entity from the repository.
        
        Args:
            id: Entity identifier
            
        Returns:
            Whether the entity was deleted
        """
        pass
    
    @abstractmethod
    def list(self, filter_args: Optional[Dict[str, Any]] = None) -> List[T]:
        """
        List entities matching the filter.
        
        Args:
            filter_args: Optional filter arguments
            
        Returns:
            List of matching entities
        """
        pass


class CachePerformanceRepository(PerformanceRepository[T]):
    """
    Cache-based implementation of the performance repository.
    
    Uses a cache backend for storing and retrieving performance data.
    This is suitable for development and testing, but not for production use
    unless the cache is persistent.
    """
    
    def __init__(
        self,
        entity_type: Type[T],
        cache_prefix: str,
        ttl_seconds: int = 86400 * 30,  # 30 days
        serializer: Optional[callable] = None,
        deserializer: Optional[callable] = None
    ):
        """
        Initialize the cache repository.
        
        Args:
            entity_type: Type of entity stored in this repository
            cache_prefix: Prefix for cache keys
            ttl_seconds: Time-to-live in seconds for cached entities
            serializer: Optional function to serialize entities
            deserializer: Optional function to deserialize entities
        """
        self.entity_type = entity_type
        self.cache_prefix = cache_prefix
        self.ttl_seconds = ttl_seconds
        self.cache = get_cache()
        
        # Use default serialization/deserialization if not provided
        self.serializer = serializer or self._default_serializer
        self.deserializer = deserializer or self._default_deserializer
        
        # Index of all entity IDs
        self._id_index_key = f"{cache_prefix}:index"
    
    def _key_for_id(self, id: str) -> str:
        """Generate a cache key for an entity ID."""
        return f"{self.cache_prefix}:{id}"
    
    def _default_serializer(self, entity: T) -> Dict[str, Any]:
        """
        Default serializer for entities.
        
        Args:
            entity: Entity to serialize
            
        Returns:
            Serialized entity as a dictionary
        """
        if hasattr(entity, 'to_dict'):
            return entity.to_dict()
        elif hasattr(entity, '__dict__'):
            return entity.__dict__
        else:
            return entity  # Hope it's JSON serializable
    
    def _default_deserializer(self, data: Dict[str, Any]) -> T:
        """
        Default deserializer for entities.
        
        Args:
            data: Serialized entity data
            
        Returns:
            Deserialized entity
        """
        if hasattr(self.entity_type, 'from_dict'):
            return self.entity_type.from_dict(data)
        else:
            return self.entity_type(**data)
    
    def get(self, id: str) -> Optional[T]:
        """
        Retrieve an entity by ID.
        
        Args:
            id: Entity identifier
            
        Returns:
            Entity or None if not found
        """
        key = self._key_for_id(id)
        data = self.cache.get(key)
        
        if data is None:
            return None
        
        try:
            entity = self.deserializer(data)
            return entity
        except Exception as e:
            logger.error(f"Error deserializing entity {id}: {e}")
            return None
    
    def save(self, entity: T) -> str:
        """
        Save an entity to the repository.
        
        Args:
            entity: Entity to save
            
        Returns:
            Entity identifier
        """
        # Get the entity ID
        if hasattr(entity, 'id'):
            id = entity.id
        elif hasattr(entity, 'user_id'):
            id = entity.user_id
        elif hasattr(entity, 'get_id'):
            id = entity.get_id()
        else:
            # Generate a new ID if none exists
            id = str(uuid.uuid4())
            if hasattr(entity, 'id'):
                entity.id = id
        
        # Serialize the entity
        try:
            data = self.serializer(entity)
        except Exception as e:
            logger.error(f"Error serializing entity: {e}")
            raise
        
        # Save to cache
        key = self._key_for_id(id)
        self.cache.set(key, data, self.ttl_seconds)
        
        # Update index
        self._update_index(id)
        
        return id
    
    def _update_index(self, id: str) -> None:
        """
        Update the index of entity IDs.
        
        Args:
            id: Entity ID to add to the index
        """
        # Get current index
        index = self.cache.get(self._id_index_key) or set()
        
        # Add the ID to the index
        index.add(id)
        
        # Save the updated index
        self.cache.set(self._id_index_key, index, self.ttl_seconds)
    
    def delete(self, id: str) -> bool:
        """
        Delete an entity from the repository.
        
        Args:
            id: Entity identifier
            
        Returns:
            Whether the entity was deleted
        """
        key = self._key_for_id(id)
        
        # Check if entity exists
        if self.cache.get(key) is None:
            return False
        
        # Delete from cache
        self.cache.delete(key)
        
        # Update index
        index = self.cache.get(self._id_index_key) or set()
        if id in index:
            index.remove(id)
            self.cache.set(self._id_index_key, index, self.ttl_seconds)
        
        return True
    
    def list(self, filter_args: Optional[Dict[str, Any]] = None) -> List[T]:
        """
        List entities matching the filter.
        
        Args:
            filter_args: Optional filter arguments
            
        Returns:
            List of matching entities
        """
        # Get index of all IDs
        index = self.cache.get(self._id_index_key) or set()
        
        # Load all entities
        entities = []
        for id in index:
            entity = self.get(id)
            if entity is not None:
                entities.append(entity)
        
        # Apply filters if provided
        if filter_args:
            return self._filter_entities(entities, filter_args)
        
        return entities
    
    def _filter_entities(
        self, 
        entities: List[T], 
        filter_args: Dict[str, Any]
    ) -> List[T]:
        """
        Filter entities based on filter arguments.
        
        Args:
            entities: List of entities to filter
            filter_args: Filter arguments
            
        Returns:
            Filtered list of entities
        """
        result = []
        
        for entity in entities:
            matches = True
            
            for key, value in filter_args.items():
                if hasattr(entity, key):
                    entity_value = getattr(entity, key)
                    if entity_value != value:
                        matches = False
                        break
                else:
                    matches = False
                    break
            
            if matches:
                result.append(entity)
        
        return result


class TrackerRepository(CachePerformanceRepository[PerformanceTracker]):
    """Repository for performance tracker entities."""
    
    def __init__(self):
        """Initialize the tracker repository."""
        super().__init__(
            entity_type=PerformanceTracker,
            cache_prefix="performance:tracker",
            ttl_seconds=86400 * 90  # 90 days
        )
    
    def get_for_user_and_type(
        self, 
        user_id: str, 
        assessment_type: str
    ) -> Optional[PerformanceTracker]:
        """
        Get a performance tracker for a specific user and assessment type.
        
        Args:
            user_id: User identifier
            assessment_type: Assessment type
            
        Returns:
            Performance tracker or None if not found
        """
        # Generate a compound ID
        id = f"{user_id}:{assessment_type}"
        
        return self.get(id)
    
    def save(self, tracker: PerformanceTracker) -> str:
        """
        Save a performance tracker.
        
        Args:
            tracker: Performance tracker to save
            
        Returns:
            Performance tracker identifier
        """
        # Generate a compound ID
        id = f"{tracker.user_id}:{tracker.assessment_type}"
        
        # Save with the compound ID
        key = self._key_for_id(id)
        data = self.serializer(tracker)
        self.cache.set(key, data, self.ttl_seconds)
        
        # Update index
        self._update_index(id)
        
        return id
    
    def list_for_user(self, user_id: str) -> List[PerformanceTracker]:
        """
        List all performance trackers for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            List of performance trackers
        """
        return self.list({"user_id": user_id})
    
    def list_by_assessment_type(self, assessment_type: str) -> List[PerformanceTracker]:
        """
        List all performance trackers for an assessment type.
        
        Args:
            assessment_type: Assessment type
            
        Returns:
            List of performance trackers
        """
        return self.list({"assessment_type": assessment_type})


class DifficultyRepository(CachePerformanceRepository[AdaptiveDifficultyEngine]):
    """Repository for difficulty engine entities."""
    
    def __init__(self):
        """Initialize the difficulty repository."""
        super().__init__(
            entity_type=AdaptiveDifficultyEngine,
            cache_prefix="performance:difficulty",
            ttl_seconds=86400 * 90  # 90 days
        )
    
    def get_for_user_and_type(
        self, 
        user_id: str, 
        assessment_type: str
    ) -> Optional[AdaptiveDifficultyEngine]:
        """
        Get a difficulty engine for a specific user and assessment type.
        
        Args:
            user_id: User identifier
            assessment_type: Assessment type
            
        Returns:
            Difficulty engine or None if not found
        """
        # Generate a compound ID
        id = f"{user_id}:{assessment_type}"
        
        return self.get(id)
    
    def save(self, engine: AdaptiveDifficultyEngine, user_id: str, assessment_type: str) -> str:
        """
        Save a difficulty engine.
        
        Args:
            engine: Difficulty engine to save
            user_id: User identifier
            assessment_type: Assessment type
            
        Returns:
            Difficulty engine identifier
        """
        # Generate a compound ID
        id = f"{user_id}:{assessment_type}"
        
        # Save with the compound ID
        key = self._key_for_id(id)
        data = self.serializer(engine)
        self.cache.set(key, data, self.ttl_seconds)
        
        # Update index
        self._update_index(id)
        
        return id


class SpacedRepetitionRepository(CachePerformanceRepository[SpacedRepetitionScheduler]):
    """Repository for spaced repetition scheduler entities."""
    
    def __init__(self):
        """Initialize the spaced repetition repository."""
        super().__init__(
            entity_type=SpacedRepetitionScheduler,
            cache_prefix="performance:spaced_repetition",
            ttl_seconds=86400 * 180  # 180 days
        )
    
    def get_for_user_and_type(
        self, 
        user_id: str, 
        assessment_type: str
    ) -> Optional[SpacedRepetitionScheduler]:
        """
        Get a spaced repetition scheduler for a specific user and assessment type.
        
        Args:
            user_id: User identifier
            assessment_type: Assessment type
            
        Returns:
            Spaced repetition scheduler or None if not found
        """
        # Generate a compound ID
        id = f"{user_id}:{assessment_type}"
        
        return self.get(id)
    
    def save(
        self, 
        scheduler: SpacedRepetitionScheduler, 
        user_id: str, 
        assessment_type: str
    ) -> str:
        """
        Save a spaced repetition scheduler.
        
        Args:
            scheduler: Spaced repetition scheduler to save
            user_id: User identifier
            assessment_type: Assessment type
            
        Returns:
            Spaced repetition scheduler identifier
        """
        # Generate a compound ID
        id = f"{user_id}:{assessment_type}"
        
        # Save with the compound ID
        key = self._key_for_id(id)
        data = self.serializer(scheduler)
        self.cache.set(key, data, self.ttl_seconds)
        
        # Update index
        self._update_index(id)
        
        return id


# Singleton instances for common use
tracker_repository = TrackerRepository()
difficulty_repository = DifficultyRepository()
spaced_repetition_repository = SpacedRepetitionRepository()


def create_performance_repository(
    entity_type: Type[T],
    cache_prefix: str,
    ttl_seconds: int = 86400 * 30  # 30 days
) -> PerformanceRepository[T]:
    """
    Create a new performance repository for a custom entity type.
    
    Args:
        entity_type: Type of entity stored in this repository
        cache_prefix: Prefix for cache keys
        ttl_seconds: Time-to-live in seconds for cached entities
        
    Returns:
        New repository instance
    """
    return CachePerformanceRepository(
        entity_type=entity_type,
        cache_prefix=cache_prefix,
        ttl_seconds=ttl_seconds
    ) 