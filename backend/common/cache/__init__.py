"""
Hierarchical Caching System

This package provides a flexible caching system with support for multiple backends
including in-memory caching and Redis, with features such as TTL (Time-To-Live)
and LRU (Least Recently Used) eviction policies.
"""

import functools
import inspect
import logging
import hashlib
import json
from enum import Enum
from typing import Any, Callable, Dict, Optional, TypeVar, cast, Union, Tuple

# Set up logging
logger = logging.getLogger(__name__)

class SerializationFormat(str, Enum):
    """Enum for cache serialization formats."""
    JSON = "json"
    PICKLE = "pickle"

from backend.common.cache.base import (
    CacheBackend,
    CachePolicy,
    CacheKey,
    CacheEntry,
    CacheResult,
    CacheError
)

from backend.common.cache.memory import (
    MemoryCacheBackend as MemoryCache,
    MemoryCacheBackend as LRUMemoryCache  # Alias for backward compatibility
)

from backend.common.cache.redis import (
    RedisCacheBackend as RedisCache
)

from backend.common.cache.manager import (
    CacheManager,
    get_cache_manager,
    configure_cache,
    configure_cache_sync,
    clear_all_caches
)

# Configure default cache with memory backend
configure_cache_sync(memory_cache_size=10000)

# Convenience functions
def get_cache(name='default'):
    """Get a cache instance by name"""
    return get_cache_manager().get_cache(name)

def get_cache_client(name='default') -> CacheBackend:
    """
    Get a cache client instance by name.
    This is a convenience function that returns a cache instance with async support.
    
    Args:
        name: Name of the cache to use (default: 'default')
        
    Returns:
        A cache backend instance with async support
    """
    return get_cache(name)

# Create default cache instance
cache = get_cache('default')

# Public API
__all__ = [
    # Cache types and policies
    'CacheBackend',
    'CachePolicy',
    'CacheKey',
    'CacheEntry',
    'CacheResult',
    'CacheError',
    'SerializationFormat',
    
    # Cache implementations
    'MemoryCache',
    'LRUMemoryCache',
    'RedisCache',
    
    # Cache management
    'CacheManager',
    'get_cache_manager',
    'configure_cache',
    'configure_cache_sync',
    'clear_all_caches',
    'get_cache',
    'get_cache_client',
    'cache',  # Add default cache instance to exports
    
    # Cache decorators
    'async_cached',
]

# Type variables for the decorator
T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])

def async_cached(
    ttl: int = 3600,
    key_prefix: str = "",
    prefix: str = "",  # For backward compatibility
    cache_name: str = "default",
    key_builder: Optional[Callable[..., str]] = None
) -> Callable[[F], F]:
    """
    Decorator for caching async function results.
    
    Args:
        ttl: Time to live for the cached result in seconds
        key_prefix: Prefix for the cache key
        prefix: Alias for key_prefix (backward compatibility)
        cache_name: Name of the cache to use
        key_builder: Optional function to build custom cache keys
        
    Returns:
        Decorated function that uses the cache
    """
    # Use prefix if key_prefix is not set
    if not key_prefix and prefix:
        key_prefix = prefix
        
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Get the cache instance
            cache = get_cache(cache_name)
            
            # Generate cache key
            if key_builder:
                cache_key = key_builder(*args, **kwargs)
            else:
                # Default key generation using function name, args, and kwargs
                key_parts = [func.__module__, func.__name__]
                
                # Add args to key
                for arg in args:
                    try:
                        # Convert arg to a string representation
                        key_parts.append(str(arg))
                    except Exception:
                        # If we can't convert to string, use its id
                        key_parts.append(f"id:{id(arg)}")
                
                # Add kwargs to key (sorted to ensure consistent order)
                for k in sorted(kwargs.keys()):
                    v = kwargs[k]
                    try:
                        # Convert value to a string representation
                        key_parts.append(f"{k}:{v}")
                    except Exception:
                        # If we can't convert to string, use its id
                        key_parts.append(f"{k}:id:{id(v)}")
                
                # Create a single string, then hash it
                key_str = ":".join(key_parts)
                hashed = hashlib.md5(key_str.encode('utf-8')).hexdigest()
                cache_key = f"{key_prefix}:{hashed}" if key_prefix else hashed
            
            # Try to get from cache
            try:
                result = await cache.get(cache_key)
                if result.success and result.value is not None:
                    logger.debug(f"Cache hit for {func.__name__} with key {cache_key}")
                    return result.value
            except Exception as e:
                logger.warning(f"Error retrieving from cache: {e}")
            
            # Cache miss, call the function
            logger.debug(f"Cache miss for {func.__name__} with key {cache_key}")
            result = await func(*args, **kwargs)
            
            # Store in cache
            try:
                await cache.set(cache_key, result, ttl=ttl)
            except Exception as e:
                logger.warning(f"Error storing in cache: {e}")
            
            return result
        
        return cast(F, wrapper)
    
    return decorator 