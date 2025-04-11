"""
Redis Client Utility Module

This module provides a singleton Redis client for direct use
by components that need to interact with Redis directly,
rather than through the caching system.
"""

import os
import logging
import redis
from redis.exceptions import RedisError
from typing import Optional, Dict, Any

# Setup logging
logger = logging.getLogger(__name__)

# Singleton Redis client instance
_redis_client: Optional[redis.Redis] = None

def get_redis_settings() -> Dict[str, Any]:
    """
    Get Redis connection settings from environment variables.
    
    Returns:
        Dictionary with Redis connection settings
    """
    return {
        "host": os.environ.get("REDIS_HOST", "localhost"),
        "port": int(os.environ.get("REDIS_PORT", "6379")),
        "db": int(os.environ.get("REDIS_DB", "0")),
        "password": os.environ.get("REDIS_PASSWORD", None),
        "decode_responses": False  # Let client code handle decoding as needed
    }

def get_redis_client() -> redis.Redis:
    """
    Get a Redis client instance.
    
    Returns the singleton Redis client instance, creating it if it doesn't exist.
    
    Returns:
        Redis client instance
    """
    global _redis_client
    
    if _redis_client is None:
        settings = get_redis_settings()
        
        try:
            _redis_client = redis.Redis(**settings)
            # Test connection
            _redis_client.ping()
            logger.info(f"Connected to Redis at {settings['host']}:{settings['port']}")
        except RedisError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            # Return a client anyway, operations will fail with proper errors
            _redis_client = redis.Redis(**settings)
    
    return _redis_client

def reset_redis_client() -> None:
    """
    Reset the Redis client.
    
    This forces a new connection on the next call to get_redis_client().
    """
    global _redis_client
    
    if _redis_client is not None:
        try:
            _redis_client.close()
        except Exception as e:
            logger.error(f"Error closing Redis connection: {e}")
        
        _redis_client = None
        logger.info("Redis client reset") 