"""
Base Cache Module

This module defines the core interfaces and types for the caching system,
including the base cache backend interface and cache policies.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Generic, Optional, TypeVar, Union, List, Tuple

# Type variables for generic cache key and value types
K = TypeVar('K')
V = TypeVar('V')

# Cache key type
CacheKey = Union[str, Tuple[str, ...]]


class CachePolicy(Enum):
    """Cache eviction and management policies."""
    
    # No automatic eviction
    NONE = "none"
    
    # Time-to-live based eviction
    TTL = "ttl"
    
    # Least Recently Used eviction
    LRU = "lru"
    
    # Least Frequently Used eviction
    LFU = "lfu"
    
    # First In, First Out eviction
    FIFO = "fifo"


class CacheError(Exception):
    """Base exception for cache-related errors."""
    pass


@dataclass
class CacheEntry(Generic[V]):
    """
    Represents a cached value with metadata.
    
    Attributes:
        key: The cache key
        value: The cached value
        created_at: Unix timestamp when the entry was created
        expires_at: Optional Unix timestamp when the entry expires
        ttl: Time-to-live in seconds (0 means no expiration)
        hits: Number of times this entry has been accessed
        size: Approximate size of the entry in bytes
        metadata: Optional additional metadata
    """
    key: CacheKey
    value: V
    created_at: float
    expires_at: Optional[float] = None
    ttl: int = 0
    hits: int = 0
    size: int = 0
    metadata: Dict[str, Any] = None
    
    def is_expired(self, current_time: float) -> bool:
        """Check if the entry is expired."""
        if self.expires_at is None or self.ttl == 0:
            return False
        return current_time >= self.expires_at


@dataclass
class CacheResult(Generic[V]):
    """
    Result of a cache operation.
    
    Attributes:
        success: Whether the operation was successful
        value: The value retrieved or stored
        hit: Whether the value was found in cache (for get operations)
        ttl: Remaining time-to-live in seconds
        source: Source of the cached value (e.g., 'memory', 'redis')
        error: Optional error message if the operation failed
    """
    success: bool
    value: Optional[V] = None
    hit: bool = False
    ttl: Optional[int] = None
    source: Optional[str] = None
    error: Optional[str] = None


class CacheBackend(Generic[K, V], ABC):
    """
    Abstract interface for cache backends.
    
    This interface defines the operations that all cache backends must support.
    Concrete implementations should handle the specifics of different cache stores
    like memory, Redis, etc.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Get the name of this cache backend."""
        pass
    
    @abstractmethod
    async def get(self, key: K) -> CacheResult[V]:
        """
        Retrieve a value from the cache.
        
        Args:
            key: The cache key
            
        Returns:
            CacheResult with the value and metadata
        """
        pass
    
    @abstractmethod
    async def set(
        self, 
        key: K, 
        value: V, 
        ttl: int = 0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> CacheResult[V]:
        """
        Store a value in the cache.
        
        Args:
            key: The cache key
            value: The value to cache
            ttl: Time-to-live in seconds (0 means no expiration)
            metadata: Optional additional metadata
            
        Returns:
            CacheResult indicating success/failure
        """
        pass
    
    @abstractmethod
    async def delete(self, key: K) -> bool:
        """
        Delete a value from the cache.
        
        Args:
            key: The cache key
            
        Returns:
            True if the value was deleted, False otherwise
        """
        pass
    
    @abstractmethod
    async def has(self, key: K) -> bool:
        """
        Check if a key exists in the cache.
        
        Args:
            key: The cache key
            
        Returns:
            True if the key exists, False otherwise
        """
        pass
    
    @abstractmethod
    async def clear(self) -> bool:
        """
        Clear all entries from the cache.
        
        Returns:
            True if the cache was cleared, False otherwise
        """
        pass
    
    @abstractmethod
    async def get_many(self, keys: List[K]) -> Dict[K, CacheResult[V]]:
        """
        Retrieve multiple values from the cache.
        
        Args:
            keys: List of cache keys
            
        Returns:
            Dictionary mapping keys to CacheResults
        """
        pass
    
    @abstractmethod
    async def set_many(
        self, 
        entries: Dict[K, V], 
        ttl: int = 0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[K, CacheResult[V]]:
        """
        Store multiple values in the cache.
        
        Args:
            entries: Dictionary mapping keys to values
            ttl: Time-to-live in seconds (0 means no expiration)
            metadata: Optional additional metadata
            
        Returns:
            Dictionary mapping keys to CacheResults
        """
        pass
    
    @abstractmethod
    async def delete_many(self, keys: List[K]) -> Dict[K, bool]:
        """
        Delete multiple values from the cache.
        
        Args:
            keys: List of cache keys
            
        Returns:
            Dictionary mapping keys to deletion success status
        """
        pass
    
    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the cache.
        
        Returns:
            Dictionary containing cache statistics
        """
        pass 