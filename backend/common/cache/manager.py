"""
Cache Manager Module

This module provides a hierarchical cache manager that can coordinate between
multiple cache backends (e.g., memory and Redis) to provide a multi-level
caching system with fallbacks and write-through capabilities.
"""

import logging
import threading
from typing import Any, Dict, List, Optional, Tuple, Union, TypeVar

from backend.common.cache.base import (
    CacheBackend,
    CacheKey,
    CachePolicy,
    CacheResult
)

# Type variable for generic cache value types
T = TypeVar('T')

# Setup logging
logger = logging.getLogger(__name__)


class CacheManager(CacheBackend):
    """
    Hierarchical cache manager that coordinates between multiple cache backends.
    
    This class implements a multi-level caching strategy that can use different
    backends (e.g., memory, Redis) in a hierarchical manner, with configurable
    write policies and fallbacks.
    
    Features:
    - Support for multiple cache levels (e.g., L1: memory, L2: Redis)
    - Write-through or write-back policies
    - Automatic fallback to lower cache levels when data is not found
    - Cache warming and population from slower/deeper levels
    - Coordinated cache invalidation across all levels
    """
    
    def __init__(self, name: str = "cache_manager"):
        """
        Initialize a new cache manager.
        
        Args:
            name: Name of this cache manager
        """
        self._name = name
        self._backends: List[CacheBackend] = []
        self._write_through = True
        self._lock = threading.RLock()
        self._stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0
        }
    
    @property
    def name(self) -> str:
        """Get the name of this cache manager."""
        return self._name
    
    @property
    def backends(self) -> List[CacheBackend]:
        """Get the list of cache backends."""
        return self._backends.copy()
    
    def add_backend(self, backend: CacheBackend, index: Optional[int] = None) -> None:
        """
        Add a cache backend to the manager.
        
        Args:
            backend: The cache backend to add
            index: Optional index at which to insert the backend
                   (None means append to the end)
        """
        with self._lock:
            if index is None:
                self._backends.append(backend)
            else:
                self._backends.insert(index, backend)
    
    def remove_backend(self, backend_name: str) -> bool:
        """
        Remove a cache backend from the manager.
        
        Args:
            backend_name: Name of the backend to remove
            
        Returns:
            True if the backend was found and removed, False otherwise
        """
        with self._lock:
            for i, backend in enumerate(self._backends):
                if backend.name == backend_name:
                    self._backends.pop(i)
                    return True
            return False
    
    def set_write_through(self, write_through: bool) -> None:
        """
        Set the write-through policy.
        
        Args:
            write_through: If True, writes will propagate to all cache levels immediately.
                          If False, writes will only go to the highest level.
        """
        self._write_through = write_through
    
    def get(self, key: CacheKey) -> CacheResult:
        """
        Retrieve a value from the cache hierarchy.
        
        This method will check each cache level in order until it finds
        the value or exhausts all levels. If the value is found in a lower
        level, it will be populated into higher levels.
        
        Args:
            key: The cache key
            
        Returns:
            CacheResult with the value and metadata
        """
        if not self._backends:
            return CacheResult(
                success=False,
                hit=False,
                source=self.name,
                error="No cache backends configured"
            )
        
        # Try to get from each cache level
        value_found = False
        result = None
        source_level = -1
        
        for i, backend in enumerate(self._backends):
            try:
                level_result = backend.get(key)
                
                if level_result.success and level_result.hit:
                    result = level_result
                    value_found = True
                    source_level = i
                    break
                
            except Exception as e:
                logger.error(f"Error getting from {backend.name}: {str(e)}")
        
        if not value_found:
            self._stats["misses"] += 1
            return CacheResult(
                success=False,
                hit=False,
                source=self.name,
                error="Key not found in any cache level"
            )
        
        # Populate higher cache levels if the value was found in a lower level
        if source_level > 0 and result and result.value is not None:
            self._populate_higher_levels(key, result.value, result.ttl, source_level)
        
        self._stats["hits"] += 1
        return result
    
    def set(
        self, 
        key: CacheKey, 
        value: Any, 
        ttl: int = 0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> CacheResult:
        """
        Store a value in the cache hierarchy.
        
        Args:
            key: The cache key
            value: The value to cache
            ttl: Time-to-live in seconds (0 means no expiration)
            metadata: Optional additional metadata
            
        Returns:
            CacheResult indicating success/failure
        """
        if not self._backends:
            return CacheResult(
                success=False,
                value=value,
                source=self.name,
                error="No cache backends configured"
            )
        
        result = None
        if self._write_through:
            # Write to all levels
            for backend in self._backends:
                try:
                    level_result = backend.set(key, value, ttl, metadata)
                    # Keep the result from the highest level (first one)
                    if result is None:
                        result = level_result
                except Exception as e:
                    logger.error(f"Error setting in {backend.name}: {str(e)}")
        else:
            # Write only to the highest level
            try:
                result = self._backends[0].set(key, value, ttl, metadata)
            except Exception as e:
                logger.error(f"Error setting in {self._backends[0].name}: {str(e)}")
        
        self._stats["sets"] += 1
        
        # If result is still None, there was a problem
        if result is None:
            return CacheResult(
                success=False,
                value=value,
                source=self.name,
                error="Failed to set value in any cache level"
            )
            
        return result
    
    def delete(self, key: CacheKey) -> bool:
        """
        Delete a value from all cache levels.
        
        Args:
            key: The cache key
            
        Returns:
            True if the key was found and deleted from at least one level,
            False otherwise
        """
        if not self._backends:
            return False
        
        any_deleted = False
        
        # Delete from all levels
        for backend in self._backends:
            try:
                if backend.delete(key):
                    any_deleted = True
            except Exception as e:
                logger.error(f"Error deleting from {backend.name}: {str(e)}")
        
        if any_deleted:
            self._stats["deletes"] += 1
            
        return any_deleted
    
    def has(self, key: CacheKey) -> bool:
        """
        Check if a key exists in any cache level.
        
        Args:
            key: The cache key
            
        Returns:
            True if the key exists in at least one level, False otherwise
        """
        if not self._backends:
            return False
        
        # Check each level
        for backend in self._backends:
            try:
                if backend.has(key):
                    return True
            except Exception as e:
                logger.error(f"Error checking existence in {backend.name}: {str(e)}")
        
        return False
    
    def clear(self) -> bool:
        """
        Clear all cache levels.
        
        Returns:
            True if all levels were cleared, False otherwise
        """
        if not self._backends:
            return False
        
        success = True
        
        # Clear all levels
        for backend in self._backends:
            try:
                if not backend.clear():
                    success = False
            except Exception as e:
                logger.error(f"Error clearing {backend.name}: {str(e)}")
                success = False
        
        return success
    
    def get_many(self, keys: List[CacheKey]) -> Dict[CacheKey, CacheResult]:
        """
        Retrieve multiple values from the cache hierarchy.
        
        Args:
            keys: List of cache keys
            
        Returns:
            Dictionary mapping keys to CacheResults
        """
        if not self._backends:
            return {key: CacheResult(
                success=False,
                hit=False,
                source=self.name,
                error="No cache backends configured"
            ) for key in keys}
        
        results = {}
        missing_keys = set(keys)
        
        # Try each cache level
        for level, backend in enumerate(self._backends):
            if not missing_keys:
                break
                
            try:
                # Only query for keys that haven't been found yet
                level_results = backend.get_many(list(missing_keys))
                
                for key, result in level_results.items():
                    if result.success and result.hit:
                        results[key] = result
                        missing_keys.remove(key)
                        
                        # Populate higher levels
                        if level > 0 and result.value is not None:
                            self._populate_higher_levels(key, result.value, result.ttl, level)
            except Exception as e:
                logger.error(f"Error getting many from {backend.name}: {str(e)}")
        
        # Add results for missing keys
        for key in missing_keys:
            results[key] = CacheResult(
                success=False,
                hit=False,
                source=self.name,
                error="Key not found in any cache level"
            )
            self._stats["misses"] += 1
        
        self._stats["hits"] += len(keys) - len(missing_keys)
        return results
    
    def set_many(
        self, 
        entries: Dict[CacheKey, Any], 
        ttl: int = 0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[CacheKey, CacheResult]:
        """
        Store multiple values in the cache hierarchy.
        
        Args:
            entries: Dictionary mapping keys to values
            ttl: Time-to-live in seconds (0 means no expiration)
            metadata: Optional additional metadata
            
        Returns:
            Dictionary mapping keys to CacheResults
        """
        if not self._backends or not entries:
            return {key: CacheResult(
                success=False,
                value=value,
                source=self.name,
                error="No cache backends configured or empty entries"
            ) for key, value in entries.items()}
        
        results = {}
        
        if self._write_through:
            # Write to all levels
            for backend in self._backends:
                try:
                    level_results = backend.set_many(entries, ttl, metadata)
                    
                    # Keep results from the highest level (first one)
                    if not results:
                        results = level_results
                except Exception as e:
                    logger.error(f"Error setting many in {backend.name}: {str(e)}")
        else:
            # Write only to highest level
            try:
                results = self._backends[0].set_many(entries, ttl, metadata)
            except Exception as e:
                logger.error(f"Error setting many in {self._backends[0].name}: {str(e)}")
        
        self._stats["sets"] += len(entries)
        
        # Fill in results for any keys that failed
        for key, value in entries.items():
            if key not in results:
                results[key] = CacheResult(
                    success=False,
                    value=value,
                    source=self.name,
                    error="Failed to set value in any cache level"
                )
                
        return results
    
    def delete_many(self, keys: List[CacheKey]) -> Dict[CacheKey, bool]:
        """
        Delete multiple values from all cache levels.
        
        Args:
            keys: List of cache keys
            
        Returns:
            Dictionary mapping keys to deletion results
        """
        if not self._backends:
            return {key: False for key in keys}
        
        results = {key: False for key in keys}
        
        # Delete from all levels
        for backend in self._backends:
            try:
                level_results = backend.delete_many(keys)
                
                # Merge results - a key is considered deleted if it was
                # deleted from at least one level
                for key, deleted in level_results.items():
                    if deleted:
                        results[key] = True
            except Exception as e:
                logger.error(f"Error deleting many from {backend.name}: {str(e)}")
        
        self._stats["deletes"] += sum(1 for deleted in results.values() if deleted)
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get stats for all cache levels.
        
        Returns:
            Dict with cache stats from all levels
        """
        stats = {
            "manager": self._stats.copy(),
            "backends": {}
        }
        
        for backend in self._backends:
            try:
                backend_stats = backend.get_stats() if hasattr(backend, "get_stats") else {}
                stats["backends"][backend.name] = backend_stats
            except Exception as e:
                logger.error(f"Error getting stats from {backend.name}: {str(e)}")
                stats["backends"][backend.name] = {"error": str(e)}
        
        return stats
    
    def get_cache(self, name: str = 'default') -> CacheBackend:
        """
        Get a specific cache backend by name.
        
        Args:
            name: Name of the cache backend to retrieve
            
        Returns:
            The requested cache backend
            
        Raises:
            ValueError: If no backend with the given name is found
        """
        for backend in self._backends:
            if backend.name == name:
                return backend
                
        # If not found, return the first backend as default
        if self._backends and name == 'default':
            return self._backends[0]
            
        raise ValueError(f"No cache backend found with name '{name}'")
    
    def _populate_higher_levels(
        self, 
        key: CacheKey, 
        value: Any, 
        ttl: Optional[int],
        source_level: int
    ) -> None:
        """
        Populate the value into higher cache levels.
        
        Args:
            key: The cache key
            value: The value to cache
            ttl: Time-to-live in seconds
            source_level: The level index where the value was found
        """
        for i in range(source_level):
            try:
                self._backends[i].set(key, value, ttl)
            except Exception as e:
                logger.error(f"Error populating {self._backends[i].name}: {str(e)}")


# Global cache manager instance
_cache_manager = None
_manager_lock = threading.RLock()


def get_cache_manager() -> CacheManager:
    """
    Get the global cache manager instance.
    
    Returns:
        The global cache manager
    """
    global _cache_manager
    
    with _manager_lock:
        if _cache_manager is None:
            _cache_manager = CacheManager()
            
    return _cache_manager


def configure_cache(
    memory_cache_size: int = 10000,
    redis_url: Optional[str] = None,
    write_through: bool = True
) -> CacheManager:
    """
    Configure the global cache manager with specified settings.
    
    Args:
        memory_cache_size: Size of the memory cache
        redis_url: Optional Redis connection URL
        write_through: Whether to use write-through caching
        
    Returns:
        The configured cache manager
    """
    from backend.common.cache.memory import MemoryCacheBackend
    
    manager = get_cache_manager()
    
    # Reset backends
    with _manager_lock:
        for backend in manager.backends:
            manager.remove_backend(backend.name)
        
        # Add memory cache as first level
        memory_cache = MemoryCacheBackend(max_size=memory_cache_size, name="default")
        manager.add_backend(memory_cache)
        
        # Add Redis cache as second level if configured
        if redis_url:
            try:
                from backend.common.cache.redis import RedisCacheBackend
                redis_cache = RedisCacheBackend(
                    redis_url=redis_url,
                    name="redis"
                )
                manager.add_backend(redis_cache)
            except ImportError:
                logger.warning("Redis dependencies not installed, skipping Redis cache")
            except Exception as e:
                logger.error(f"Error configuring Redis cache: {str(e)}")
        
        # Set write-through policy
        manager.set_write_through(write_through)
    
    return manager


def configure_cache_sync(
    memory_cache_size: int = 10000,
    redis_url: Optional[str] = None,
    write_through: bool = True
) -> CacheManager:
    """
    Configure the global cache manager with specified settings, using a synchronous
    memory cache implementation that doesn't require an event loop.
    
    Args:
        memory_cache_size: Size of the memory cache
        redis_url: Optional Redis connection URL
        write_through: Whether to use write-through caching
        
    Returns:
        The configured cache manager
    """
    # Create a basic synchronous cache implementation for bootstrapping
    class SimpleCacheBackend(CacheBackend):
        def __init__(self, name="default"):
            self._name = name
            self._cache = {}
            
        @property
        def name(self):
            return self._name
            
        def get(self, key):
            value = self._cache.get(key)
            if value:
                return CacheResult(success=True, hit=True, source=self._name, value=value)
            return CacheResult(success=True, hit=False, source=self._name)
            
        def set(self, key, value, ttl=0, metadata=None):
            self._cache[key] = value
            return CacheResult(success=True, source=self._name, value=value)
            
        def delete(self, key):
            if key in self._cache:
                del self._cache[key]
                return True
            return False
            
        def has(self, key):
            return key in self._cache
            
        def clear(self):
            self._cache.clear()
            return True
            
        def get_many(self, keys):
            result = {}
            for key in keys:
                if key in self._cache:
                    result[key] = CacheResult(success=True, hit=True, source=self._name, value=self._cache[key])
                else:
                    result[key] = CacheResult(success=True, hit=False, source=self._name)
            return result
            
        def set_many(self, entries, ttl=0, metadata=None):
            result = {}
            for key, value in entries.items():
                self._cache[key] = value
                result[key] = CacheResult(success=True, source=self._name, value=value)
            return result
            
        def delete_many(self, keys):
            result = {}
            for key in keys:
                if key in self._cache:
                    del self._cache[key]
                    result[key] = True
                else:
                    result[key] = False
            return result
            
        def get_stats(self):
            return {
                "backend": "simple",
                "size": len(self._cache),
                "keys": list(self._cache.keys())
            }
    
    manager = get_cache_manager()
    
    # Reset backends
    with _manager_lock:
        for backend in manager.backends:
            manager.remove_backend(backend.name)
        
        # Add simple memory cache
        simple_cache = SimpleCacheBackend(name="default")
        manager.add_backend(simple_cache)
        
        # Set write-through policy
        manager.set_write_through(write_through)
    
    return manager


def clear_all_caches() -> bool:
    """
    Clear all caches in the global cache manager.
    
    Returns:
        True if all caches were cleared, False otherwise
    """
    manager = get_cache_manager()
    return manager.clear() 