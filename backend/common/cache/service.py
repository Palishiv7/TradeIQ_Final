"""
Cache Service Module

This module provides the high-level cache service API, abstracting away the details
of the underlying cache backends and providing additional functionality like
decorators for function result caching.
"""

import asyncio
import functools
import inspect
import logging
import time
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union, cast

from .. import cache
from .backend import CacheBackend
from .entry import CacheEntry
from .key_builder import KeyBuilder
from .memory import MemoryCacheBackend

# Setup logging
logger = logging.getLogger(__name__)

# Type variables
T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])

# Global default cache service
_default_cache_service = None

def get_cache_service():
    """
    Get the default cache service instance.
    
    Returns:
        The default cache service, or raises an exception if not set
    """
    if _default_cache_service is None:
        raise RuntimeError("Default cache service has not been set. Call set_default_cache_service first.")
    return _default_cache_service

def set_default_cache_service(service):
    """
    Set the default cache service instance.
    
    Args:
        service: The cache service to set as default
    """
    global _default_cache_service
    _default_cache_service = service

class CacheService:
    """
    High-level caching service for the TradeIQ platform.
    
    Provides a unified interface to different cache backends with advanced features
    like function result caching, hierarchical invalidation, and monitoring.
    """
    
    def __init__(
        self,
        backend_type: 'cache.CacheBackendType' = cache.CacheBackendType.MEMORY,
        default_ttl: Optional[float] = 300,  # 5 minutes
        backend_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the cache service with specified backend.
        
        Args:
            backend_type: Type of cache backend to use
            default_ttl: Default TTL for cache entries in seconds
            backend_config: Configuration options for the backend
        """
        self._default_ttl = default_ttl
        self._backend = self._create_backend(backend_type, backend_config or {})
        
    def _create_backend(
        self,
        backend_type: 'cache.CacheBackendType',
        config: Dict[str, Any]
    ) -> CacheBackend:
        """
        Create a cache backend based on type and configuration.
        
        Args:
            backend_type: Type of cache backend to create
            config: Configuration options for the backend
            
        Returns:
            An initialized cache backend
        """
        if backend_type == cache.CacheBackendType.MEMORY:
            return MemoryCacheBackend(
                max_size=config.get('max_size', 10000),
                cleanup_interval=config.get('cleanup_interval', 60)
            )
        elif backend_type == cache.CacheBackendType.REDIS:
            # Lazy import to avoid circular dependencies
            from .redis import RedisCacheBackend
            
            return RedisCacheBackend(
                redis_client=config.get('redis_client'),
                host=config.get('host', 'localhost'),
                port=config.get('port', 6379),
                db=config.get('db', 0),
                password=config.get('password'),
                key_prefix=config.get('key_prefix', 'tradeiq:'),
                serialization=config.get('serialization', cache.SerializationFormat.JSON)
            )
        else:
            raise ValueError(f"Unsupported backend type: {backend_type}")
            
    async def get(self, key: str, default: Any = None) -> Any:
        """
        Get a value from the cache.
        
        Args:
            key: The cache key
            default: Value to return if key is not found
            
        Returns:
            The cached value, or default if not found
        """
        entry = await self._backend.get(key)
        
        if entry is None:
            return default
            
        return entry.value
    
    async def set(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """
        Set a value in the cache.
        
        Args:
            key: The cache key
            value: The value to cache
            ttl: Time-to-live in seconds, or None to use default TTL
            
        Returns:
            True if the value was successfully stored, False otherwise
        """
        # Use default TTL if not specified
        if ttl is None:
            ttl = self._default_ttl
            
        return await self._backend.set(key, value, ttl)
    
    async def delete(self, key: str) -> bool:
        """
        Delete a value from the cache.
        
        Args:
            key: The cache key
            
        Returns:
            True if the key was found and deleted, False otherwise
        """
        return await self._backend.delete(key)
    
    async def exists(self, key: str) -> bool:
        """
        Check if a key exists in the cache.
        
        Args:
            key: The cache key
            
        Returns:
            True if the key exists and has not expired, False otherwise
        """
        return await self._backend.exists(key)
    
    async def clear(self) -> bool:
        """
        Clear all values from the cache.
        
        Returns:
            True if the operation was successful, False otherwise
        """
        return await self._backend.clear()
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the cache.
        
        Returns:
            A dictionary containing statistics about the cache
        """
        return await self._backend.get_stats()
    
    async def get_many(self, keys: List[str], default: Any = None) -> Dict[str, Any]:
        """
        Get multiple values from the cache in a single operation.
        
        Args:
            keys: A list of cache keys
            default: Value to use for keys that are not found
            
        Returns:
            A dictionary mapping keys to values for found keys
        """
        entries = await self._backend.get_many(keys)
        
        # Extract values from entries
        result = {}
        for key in keys:
            if key in entries:
                result[key] = entries[key].value
            else:
                result[key] = default
                
        return result
    
    async def set_many(self, items: Dict[str, Any], ttl: Optional[float] = None) -> bool:
        """
        Set multiple values in the cache in a single operation.
        
        Args:
            items: A dictionary mapping keys to values
            ttl: Time-to-live in seconds, or None to use default TTL
            
        Returns:
            True if all values were successfully stored, False otherwise
        """
        # Use default TTL if not specified
        if ttl is None:
            ttl = self._default_ttl
            
        return await self._backend.set_many(items, ttl)
    
    async def delete_many(self, keys: List[str]) -> int:
        """
        Delete multiple keys from the cache in a single operation.
        
        Args:
            keys: A list of cache keys to delete
            
        Returns:
            The number of keys that were found and deleted
        """
        return await self._backend.delete_many(keys)
    
    async def touch(self, key: str, ttl: Optional[float] = None) -> bool:
        """
        Update the TTL of a cache entry.
        
        Args:
            key: The cache key
            ttl: New TTL in seconds, or None to use default TTL
            
        Returns:
            True if the key was found and TTL updated, False otherwise
        """
        # Use default TTL if not specified
        if ttl is None:
            ttl = self._default_ttl
            
        # Get the existing entry
        entry = await self._backend.get(key)
        
        if entry is None:
            return False
            
        # Set the entry back with new TTL
        return await self._backend.set(key, entry.value, ttl)
    
    def cached(self, key: Optional[str] = None, ttl: Optional[float] = None,
               namespace: Optional[str] = None, version: Optional[str] = None):
        """
        Decorator for caching function results.
        
        Args:
            key: Explicit cache key to use, or None to generate from function and args
            ttl: Time-to-live in seconds, or None to use default TTL
            namespace: Namespace for generated keys
            version: Version string for generated keys
            
        Returns:
            A decorator function
        """
        def decorator(func: F) -> F:
            """The actual decorator."""
            
            # Flag the function as having a cache decorator
            setattr(func, "_has_cache", True)
            
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                """Async wrapper for the cached function."""
                
                # Generate or use the provided cache key
                cache_key = key
                if cache_key is None:
                    cache_key = KeyBuilder.function_key(
                        func, *args, namespace=namespace, version=version, **kwargs
                    )
                
                # Try to get from cache
                result = await self.get(cache_key)
                if result is not None:
                    logger.debug(f"Cache hit for key: {cache_key}")
                    return result
                
                # Cache miss, call the function
                logger.debug(f"Cache miss for key: {cache_key}")
                result = await func(*args, **kwargs)
                
                # Store result in cache
                if result is not None:  # Don't cache None results
                    await self.set(cache_key, result, ttl)
                    
                return result
                
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                """Sync wrapper for the cached function."""
                
                # Generate or use the provided cache key
                cache_key = key
                if cache_key is None:
                    cache_key = KeyBuilder.function_key(
                        func, *args, namespace=namespace, version=version, **kwargs
                    )
                
                # For sync functions, we need to run the async operations in an event loop
                # This is a simplistic approach - in a real implementation, we'd need
                # to consider the current event loop and execution context
                loop = asyncio.get_event_loop()
                
                # Try to get from cache
                try:
                    result = loop.run_until_complete(self.get(cache_key))
                    if result is not None:
                        logger.debug(f"Cache hit for key: {cache_key}")
                        return result
                except Exception as e:
                    logger.error(f"Error getting from cache: {e}")
                    # Proceed with function call on cache error
                
                # Cache miss, call the function
                logger.debug(f"Cache miss for key: {cache_key}")
                result = func(*args, **kwargs)
                
                # Store result in cache
                if result is not None:  # Don't cache None results
                    try:
                        loop.run_until_complete(self.set(cache_key, result, ttl))
                    except Exception as e:
                        logger.error(f"Error setting cache: {e}")
                        
                return result
            
            # Choose the appropriate wrapper based on whether the function is async
            if inspect.iscoroutinefunction(func):
                return cast(F, async_wrapper)
            else:
                return cast(F, sync_wrapper)
                
        return decorator
    
    def memoize(self, ttl: Optional[float] = None, namespace: Optional[str] = None,
                version: Optional[str] = None):
        """
        Decorator for memoizing function results.
        
        This is an alias for cached with no explicit key.
        
        Args:
            ttl: Time-to-live in seconds, or None to use default TTL
            namespace: Namespace for generated keys
            version: Version string for generated keys
            
        Returns:
            A decorator function
        """
        return self.cached(key=None, ttl=ttl, namespace=namespace, version=version)
    
    async def invalidate_by_prefix(self, prefix: str) -> int:
        """
        Invalidate all cache entries with keys starting with the given prefix.
        
        Args:
            prefix: Prefix to match
            
        Returns:
            Number of invalidated entries
        """
        # This method is highly backend-specific and may not be efficient
        # for all backends. For Redis, we could use SCAN and pipeline commands.
        # For memory backend, we can scan all keys.
        
        # This is a simplified implementation that will only work with the
        # memory backend. For Redis, we'd need to implement scanning.
        if not hasattr(self._backend, '_cache'):
            logger.warning("Invalidate by prefix not supported for this backend")
            return 0
            
        keys_to_delete = []
        
        # Access the underlying cache dictionary directly (for MemoryCacheBackend)
        # In a real implementation, we'd need to make this more backend-agnostic
        for key in getattr(self._backend, '_cache').keys():
            if isinstance(key, str) and key.startswith(prefix):
                keys_to_delete.append(key)
                
        # Delete the keys
        count = 0
        for key in keys_to_delete:
            if await self.delete(key):
                count += 1
                
        return count
    
    async def invalidate_by_pattern(self, pattern: str) -> int:
        """
        Invalidate all cache entries matching the given pattern.
        
        Args:
            pattern: Pattern to match (glob-style)
            
        Returns:
            Number of invalidated entries
        """
        # This is also highly backend-specific. For Redis, we could use SCAN with
        # pattern matching. For memory backend, we need to implement glob matching.
        
        # This is not implemented for the simple example
        logger.warning("Invalidate by pattern not fully implemented")
        return 0 