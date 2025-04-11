"""
Redis Cache Backend Module

This module implements a Redis cache backend for the TradeIQ caching system.
It provides distributed caching capabilities with configurable serialization.
"""

import asyncio
import json
import logging
import pickle
import time
from typing import Any, Dict, List, Optional, Union, TypeVar, cast

import redis
from redis.exceptions import RedisError

from . import SerializationFormat
from .base import CacheBackend, CacheResult
from .entry import CacheEntry

# Setup logging
logger = logging.getLogger(__name__)

# Type variables
K = TypeVar('K')
V = TypeVar('V')

class RedisCacheBackend(CacheBackend[str, Any]):
    """
    Redis cache backend implementation.
    
    This class implements the CacheBackend interface using Redis as the storage
    backend. It supports configurable serialization formats, key prefixing, and
    connection pooling.
    
    Features:
    - Distributed cache across multiple application instances
    - Configurable serialization (JSON or Pickle)
    - Automatic handling of complex Python types
    - Connection pooling for efficient Redis connections
    - Separate metadata storage for cache entries
    """
    
    def __init__(
        self,
        redis_client: Optional[redis.Redis] = None,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        key_prefix: str = "tradeiq:",
        serialization: SerializationFormat = SerializationFormat.JSON,
        read_client: Optional[redis.Redis] = None,
        name: str = "redis"
    ):
        """
        Initialize the Redis cache backend.
        
        Args:
            redis_client: Optional existing Redis client to use
            host: Redis server hostname (default: "localhost")
            port: Redis server port (default: 6379)
            db: Redis database number (default: 0)
            password: Redis password (optional)
            key_prefix: Prefix for all Redis keys (default: "tradeiq:")
            serialization: Serialization format (JSON or PICKLE)
            read_client: Optional separate Redis client for reads (read/write splitting)
            name: Name for this cache backend (default: "redis")
        """
        # Store configuration
        self._key_prefix = key_prefix
        self._serialization = serialization
        self._name = name
        
        # Create or use Redis client
        if redis_client:
            self._redis = redis_client
        else:
            self._redis = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                decode_responses=False  # We handle decoding ourselves
            )
        
        # Setup read/write splitting if configured
        self._read_client = read_client or self._redis
        
        # Statistics counters
        self._hits = 0
        self._misses = 0
        
        # Test connection
        try:
            self._redis.ping()
        except RedisError as e:
            logger.warning(f"Redis connection test failed: {e}")
    
    @property
    def name(self) -> str:
        """Get the name of this cache backend."""
        return self._name
    
    def _build_key(self, key: str) -> str:
        """
        Build a Redis key with the configured prefix.
        
        Args:
            key: The original cache key
            
        Returns:
            The prefixed Redis key
        """
        return f"{self._key_prefix}{key}"
    
    def _build_metadata_key(self, key: str) -> str:
        """
        Build a Redis key for metadata with the configured prefix.
        
        Args:
            key: The original cache key
            
        Returns:
            The prefixed Redis metadata key
        """
        return f"{self._key_prefix}{key}:meta"
    
    def _serialize(self, value: Any) -> bytes:
        """
        Serialize a value for storage in Redis.
        
        Args:
            value: The value to serialize
            
        Returns:
            Serialized bytes
        """
        if self._serialization == SerializationFormat.JSON:
            try:
                return json.dumps(value).encode('utf-8')
            except (TypeError, ValueError) as e:
                logger.warning(f"JSON serialization failed, falling back to pickle: {e}")
                return pickle.dumps(value)
        else:  # PICKLE
            return pickle.dumps(value)
    
    def _deserialize(self, data: bytes) -> Any:
        """
        Deserialize data from Redis.
        
        Args:
            data: The serialized data
            
        Returns:
            The deserialized value
        """
        if not data:
            return None
            
        if self._serialization == SerializationFormat.JSON:
            try:
                return json.loads(data.decode('utf-8'))
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                logger.warning(f"JSON deserialization failed, trying pickle: {e}")
                try:
                    return pickle.loads(data)
                except Exception as e2:
                    logger.error(f"Both JSON and pickle deserialization failed: {e2}")
                    return None
        else:  # PICKLE
            try:
                return pickle.loads(data)
            except Exception as e:
                logger.error(f"Pickle deserialization failed: {e}")
                return None
    
    async def get(self, key: str) -> CacheResult[Any]:
        """
        Get a value from Redis.
        
        Args:
            key: The cache key
            
        Returns:
            CacheResult containing the value and metadata
        """
        redis_key = self._build_key(key)
        try:
            # Get value and metadata in a pipeline
            pipe = self._read_client.pipeline()
            pipe.get(redis_key)
            pipe.ttl(redis_key)
            value_data, ttl = await pipe.execute()
            
            if value_data is None:
                self._misses += 1
                return CacheResult(
                    success=False,
                    value=None,
                    hit=False,
                    source=self.name,
                    error="Key not found"
                )
            
            # Deserialize value
            value = self._deserialize(value_data)
            if value is None:
                self._misses += 1
                return CacheResult(
                    success=False,
                    value=None,
                    hit=False,
                    source=self.name,
                    error="Deserialization failed"
                )
            
            self._hits += 1
            return CacheResult(
                success=True,
                value=value,
                hit=True,
                ttl=ttl if ttl > 0 else None,
                source=self.name
            )
            
        except RedisError as e:
            logger.error(f"Redis error in get: {e}")
            return CacheResult(
                success=False,
                value=None,
                hit=False,
                source=self.name,
                error=str(e)
            )
    
    async def set(
        self, 
        key: str, 
        value: Any, 
        ttl: int = 0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> CacheResult[Any]:
        """
        Set a value in Redis.
        
        Args:
            key: The cache key
            value: The value to cache
            ttl: Time-to-live in seconds (0 means no expiration)
            metadata: Optional additional metadata
            
        Returns:
            CacheResult indicating success/failure
        """
        redis_key = self._build_key(key)
        try:
            # Serialize value
            value_data = self._serialize(value)
            if value_data is None:
                return CacheResult(
                    success=False,
                    value=None,
                    hit=False,
                    source=self.name,
                    error="Serialization failed"
                )
            
            # Set value and TTL in a pipeline
            pipe = self._redis.pipeline()
            pipe.set(redis_key, value_data)
            if ttl > 0:
                pipe.expire(redis_key, ttl)
            await pipe.execute()
            
            return CacheResult(
                success=True,
                value=value,
                hit=False,
                ttl=ttl if ttl > 0 else None,
                source=self.name
            )
            
        except RedisError as e:
            logger.error(f"Redis error in set: {e}")
            return CacheResult(
                success=False,
                value=None,
                hit=False,
                source=self.name,
                error=str(e)
            )
    
    async def delete(self, key: str) -> bool:
        """
        Delete a value from Redis.
        
        Args:
            key: The cache key
            
        Returns:
            True if the key was found and deleted, False otherwise
        """
        redis_key = self._build_key(key)
        try:
            return bool(await self._redis.delete(redis_key))
        except RedisError as e:
            logger.error(f"Redis error in delete: {e}")
            return False
    
    async def has(self, key: str) -> bool:
        """
        Check if a key exists in Redis.
        
        Args:
            key: The cache key
            
        Returns:
            True if the key exists, False otherwise
        """
        redis_key = self._build_key(key)
        try:
            return bool(await self._read_client.exists(redis_key))
        except RedisError as e:
            logger.error(f"Redis error in has: {e}")
            return False
    
    async def clear(self) -> bool:
        """
        Clear all values with our prefix from Redis.
        
        Returns:
            True if the operation was successful, False otherwise
        """
        try:
            # Get all keys with our prefix
            pattern = f"{self._key_prefix}*"
            keys = await self._redis.keys(pattern)
            
            if keys:
                # Delete all matching keys
                await self._redis.delete(*keys)
            
            return True
            
        except RedisError as e:
            logger.error(f"Redis error in clear: {e}")
            return False
    
    async def get_many(self, keys: List[str]) -> Dict[str, CacheResult[Any]]:
        """
        Get multiple values from Redis.
        
        Args:
            keys: List of cache keys
            
        Returns:
            Dictionary mapping keys to CacheResults
        """
        results = {}
        redis_keys = [self._build_key(key) for key in keys]
        
        try:
            # Get values and TTLs in a pipeline
            pipe = self._read_client.pipeline()
            for redis_key in redis_keys:
                pipe.get(redis_key)
                pipe.ttl(redis_key)
            responses = await pipe.execute()
            
            # Process responses in pairs (value, ttl)
            for i, key in enumerate(keys):
                value_data = responses[i * 2]
                ttl = responses[i * 2 + 1]
                
                if value_data is None:
                    self._misses += 1
                    results[key] = CacheResult(
                        success=False,
                        value=None,
                        hit=False,
                        source=self.name,
                        error="Key not found"
                    )
                    continue
                
                # Deserialize value
                value = self._deserialize(value_data)
                if value is None:
                    self._misses += 1
                    results[key] = CacheResult(
                        success=False,
                        value=None,
                        hit=False,
                        source=self.name,
                        error="Deserialization failed"
                    )
                    continue
                
                self._hits += 1
                results[key] = CacheResult(
                    success=True,
                    value=value,
                    hit=True,
                    ttl=ttl if ttl > 0 else None,
                    source=self.name
                )
            
            return results
            
        except RedisError as e:
            logger.error(f"Redis error in get_many: {e}")
            # Return error results for all keys
            return {
                key: CacheResult(
                    success=False,
                    value=None,
                    hit=False,
                    source=self.name,
                    error=str(e)
                )
                for key in keys
            }
    
    async def set_many(
        self, 
        entries: Dict[str, Any], 
        ttl: int = 0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, CacheResult[Any]]:
        """
        Set multiple values in Redis.
        
        Args:
            entries: Dictionary mapping keys to values
            ttl: Time-to-live in seconds (0 means no expiration)
            metadata: Optional additional metadata
            
        Returns:
            Dictionary mapping keys to CacheResults
        """
        results = {}
        
        try:
            # Set values and TTLs in a pipeline
            pipe = self._redis.pipeline()
            for key, value in entries.items():
                redis_key = self._build_key(key)
                
                # Serialize value
                value_data = self._serialize(value)
                if value_data is None:
                    results[key] = CacheResult(
                        success=False,
                        value=None,
                        hit=False,
                        source=self.name,
                        error="Serialization failed"
                    )
                    continue
                
                pipe.set(redis_key, value_data)
                if ttl > 0:
                    pipe.expire(redis_key, ttl)
                
                results[key] = CacheResult(
                    success=True,
                    value=value,
                    hit=False,
                    ttl=ttl if ttl > 0 else None,
                    source=self.name
                )
            
            await pipe.execute()
            return results
            
        except RedisError as e:
            logger.error(f"Redis error in set_many: {e}")
            # Return error results for all keys
            return {
                key: CacheResult(
                    success=False,
                    value=None,
                    hit=False,
                    source=self.name,
                    error=str(e)
                )
                for key in entries.keys()
            }
    
    async def delete_many(self, keys: List[str]) -> Dict[str, bool]:
        """
        Delete multiple values from Redis.
        
        Args:
            keys: List of cache keys
            
        Returns:
            Dictionary mapping keys to deletion success status
        """
        results = {}
        redis_keys = [self._build_key(key) for key in keys]
        
        try:
            # Delete keys in a pipeline
            pipe = self._redis.pipeline()
            for redis_key in redis_keys:
                pipe.delete(redis_key)
            responses = await pipe.execute()
            
            # Process responses
            for key, deleted in zip(keys, responses):
                results[key] = bool(deleted)
            
            return results
            
        except RedisError as e:
            logger.error(f"Redis error in delete_many: {e}")
            # Return False for all keys
            return {key: False for key in keys}
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the Redis cache.
        
        Returns:
            Dictionary containing cache statistics
        """
        try:
            info = await self._redis.info()
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0
            
            return {
                'backend': 'redis',
                'hits': self._hits,
                'misses': self._misses,
                'hit_rate': hit_rate,
                'memory_used': info.get('used_memory', 0),
                'total_connections': info.get('total_connections_received', 0),
                'evicted_keys': info.get('evicted_keys', 0),
                'expired_keys': info.get('expired_keys', 0)
            }
            
        except RedisError as e:
            logger.error(f"Redis error in get_stats: {e}")
            return {
                'backend': 'redis',
                'error': str(e)
            } 