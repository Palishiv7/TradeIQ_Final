"""
Memory Cache Backend Module

This module implements an in-memory cache backend using a dictionary-based storage
with thread safety and LRU eviction.
"""

import asyncio
import logging
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Set, TypeVar

from .base import CacheBackend, CacheResult
from .entry import CacheEntry

# Setup logging
logger = logging.getLogger(__name__)

# Type variables
K = TypeVar('K')
V = TypeVar('V')

class MemoryCacheBackend(CacheBackend[K, V]):
    """
    In-memory cache backend implementation.
    
    This class implements the CacheBackend interface using a dictionary-based
    storage with thread safety and LRU eviction. It is designed for speed and
    simplicity, making it ideal for development and small-scale deployments.
    
    Features:
    - Thread-safe operations
    - LRU eviction when reaching maximum size
    - Automatic expired entry cleanup
    - Comprehensive statistics tracking
    """
    
    def __init__(self, max_size: int = 10000, cleanup_interval: int = 60, name: str = "memory"):
        """
        Initialize the memory cache backend.
        
        Args:
            max_size: Maximum number of entries to store (default: 10000)
            cleanup_interval: Interval in seconds for cleanup task (default: 60)
            name: Name for this cache backend (default: "memory")
        """
        self._cache: Dict[K, CacheEntry[V]] = OrderedDict()
        self._lock = threading.RLock()
        self._max_size = max_size
        self._cleanup_interval = cleanup_interval
        self._name = name
        
        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._expirations = 0
        
        # Start cleanup task
        self._start_cleanup_task()
    
    @property
    def name(self) -> str:
        """Get the name of this cache backend."""
        return self._name
    
    async def get(self, key: K) -> CacheResult[V]:
        """
        Get a value from the cache.
        
        Args:
            key: The cache key
            
        Returns:
            A CacheResult containing the value and metadata
        """
        with self._lock:
            entry = self._cache.get(key)
            
            if entry is None:
                self._misses += 1
                return CacheResult(
                    success=False,
                    value=None,
                    hit=False,
                    source=self.name,
                    error="Key not found"
                )
            
            if entry.is_expired():
                del self._cache[key]
                self._expirations += 1
                self._misses += 1
                return CacheResult(
                    success=False,
                    value=None,
                    hit=False,
                    source=self.name,
                    error="Entry expired"
                )
            
            # Update access stats and move to end of OrderedDict (LRU)
            entry.access()
            self._cache.move_to_end(key)
            self._hits += 1
            
            return CacheResult(
                success=True,
                value=entry.value,
                hit=True,
                ttl=entry.ttl,
                source=self.name
            )
    
    async def set(
        self, 
        key: K, 
        value: V, 
        ttl: int = 0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> CacheResult[V]:
        """
        Set a value in the cache.
        
        Args:
            key: The cache key
            value: The value to cache
            ttl: Time-to-live in seconds, or None for no expiration
            metadata: Optional additional metadata
            
        Returns:
            CacheResult indicating success/failure
        """
        with self._lock:
            try:
                # Create new entry
                entry = CacheEntry(key=key, value=value, ttl=ttl, metadata=metadata)
                
                # Check if key exists
                if key in self._cache:
                    # Update existing entry
                    self._cache[key] = entry
                    self._cache.move_to_end(key)
                else:
                    # Ensure capacity before adding new entry
                    if len(self._cache) >= self._max_size:
                        self._evict_entries()
                    
                    # Add new entry
                    self._cache[key] = entry
                    self._cache.move_to_end(key)
                
                return CacheResult(
                    success=True,
                    value=value,
                    hit=False,
                    ttl=ttl,
                    source=self.name
                )
            except Exception as e:
                return CacheResult(
                    success=False,
                    value=None,
                    hit=False,
                    source=self.name,
                    error=str(e)
                )
    
    async def delete(self, key: K) -> bool:
        """
        Delete a value from the cache.
        
        Args:
            key: The cache key
            
        Returns:
            True if the key was found and deleted, False otherwise
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    async def has(self, key: K) -> bool:
        """
        Check if a key exists in the cache.
        
        Args:
            key: The cache key
            
        Returns:
            True if the key exists and has not expired, False otherwise
        """
        with self._lock:
            entry = self._cache.get(key)
            
            if entry is None:
                return False
            
            if entry.is_expired():
                del self._cache[key]
                self._expirations += 1
                return False
                
            return True
    
    async def clear(self) -> bool:
        """
        Clear all values from the cache.
        
        Returns:
            True if the operation was successful, False otherwise
        """
        with self._lock:
            self._cache.clear()
            return True
    
    async def get_many(self, keys: List[K]) -> Dict[K, CacheResult[V]]:
        """
        Get multiple values from the cache.
        
        Args:
            keys: List of cache keys
            
        Returns:
            Dictionary mapping keys to CacheResults
        """
        results = {}
        for key in keys:
            results[key] = await self.get(key)
        return results
    
    async def set_many(
        self, 
        entries: Dict[K, V], 
        ttl: int = 0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[K, CacheResult[V]]:
        """
        Set multiple values in the cache.
        
        Args:
            entries: Dictionary mapping keys to values
            ttl: Time-to-live in seconds, or None for no expiration
            metadata: Optional additional metadata
            
        Returns:
            Dictionary mapping keys to CacheResults
        """
        results = {}
        for key, value in entries.items():
            results[key] = await self.set(key, value, ttl, metadata)
        return results
    
    async def delete_many(self, keys: List[K]) -> Dict[K, bool]:
        """
        Delete multiple values from the cache.
        
        Args:
            keys: List of cache keys
            
        Returns:
            Dictionary mapping keys to deletion success status
        """
        results = {}
        for key in keys:
            results[key] = await self.delete(key)
        return results
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the cache.
        
        Returns:
            Dictionary containing cache statistics
        """
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0
            
            return {
                'backend': 'memory',
                'size': len(self._cache),
                'max_size': self._max_size,
                'hits': self._hits,
                'misses': self._misses,
                'hit_rate': hit_rate,
                'evictions': self._evictions,
                'expirations': self._expirations
            }
    
    def _evict_entries(self) -> None:
        """
        Evict entries using LRU policy until cache is under max size.
        """
        while len(self._cache) >= self._max_size:
            self._cache.popitem(last=False)  # Remove from start (least recently used)
            self._evictions += 1
    
    def _start_cleanup_task(self) -> None:
        """
        Start the background cleanup task.
        """
        async def cleanup():
            while True:
                await asyncio.sleep(self._cleanup_interval)
                self._cleanup_expired()
        
        asyncio.create_task(cleanup())
    
    def _cleanup_expired(self) -> None:
        """
        Remove expired entries from the cache.
        """
        with self._lock:
            now = time.time()
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired(now)
            ]
            for key in expired_keys:
                del self._cache[key]
                self._expirations += 1
    
    def __len__(self) -> int:
        """Return the number of entries in the cache."""
        with self._lock:
            return len(self._cache) 