"""
Cache Entry Module

This module provides the CacheEntry class, which encapsulates cached values with metadata
for tracking expiration, access patterns, and other cache management information.
"""

import time
from typing import Any, Generic, Optional, TypeVar

# Type variable for cached value
V = TypeVar('V')

class CacheEntry(Generic[V]):
    """
    Represents a cached value with metadata.
    
    This class stores a value along with metadata such as creation time,
    expiration time, access count, and last access time. This metadata
    is used for cache management decisions like expiration and promotion/demotion.
    
    Attributes:
        value: The cached value
        created_at: When the entry was created (epoch time)
        expires_at: When the entry expires (epoch time), or None for no expiration
        access_count: Number of times the entry has been accessed
        last_accessed: When the entry was last accessed (epoch time)
        size_estimate: Estimated size of the value in bytes
    """
    
    def __init__(self, value: V, ttl: Optional[float] = None):
        """
        Initialize a cache entry with a value and optional TTL.
        
        Args:
            value: The value to cache
            ttl: Time-to-live in seconds, or None for no expiration
        """
        self.value = value
        self.created_at = time.time()
        self.expires_at = None if ttl is None else self.created_at + ttl
        self.access_count = 0
        self.last_accessed = self.created_at
        # Rough size estimation - adjust for specific use cases
        self.size_estimate = len(str(value)) if value is not None else 0
    
    def is_expired(self) -> bool:
        """
        Check if the entry has expired.
        
        Returns:
            True if the entry has expired, False otherwise
        """
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at
    
    def access(self) -> None:
        """
        Record an access to this entry.
        
        Updates the access count and last accessed time.
        """
        self.access_count += 1
        self.last_accessed = time.time()
    
    def should_promote(self) -> bool:
        """
        Check if this entry should be promoted to a faster cache level.
        
        Determines promotion based on access frequency, recency, and size.
        
        Returns:
            True if the entry should be promoted, False otherwise
        """
        # Promote if accessed frequently and recently, and not too large
        return (self.access_count > 5 and  # Frequently accessed
                time.time() - self.last_accessed < 60 and  # Recently accessed
                self.size_estimate < 10000)  # Not too large (< 10KB)
    
    def should_demote(self) -> bool:
        """
        Check if this entry should be demoted to a slower cache level.
        
        Determines demotion based on access frequency, recency, and size.
        
        Returns:
            True if the entry should be demoted, False otherwise
        """
        # Demote if rarely accessed or large
        return (self.access_count < 2 or  # Rarely accessed
                time.time() - self.last_accessed > 300 or  # Not recently accessed
                self.size_estimate > 100000)  # Large (> 100KB)
    
    def get_age(self) -> float:
        """
        Get the age of this cache entry in seconds.
        
        Returns:
            Number of seconds since the entry was created
        """
        return time.time() - self.created_at
    
    def get_ttl(self) -> Optional[float]:
        """
        Get the remaining TTL in seconds.
        
        Returns:
            Remaining TTL in seconds, or None if no expiration
        """
        if self.expires_at is None:
            return None
        remaining = self.expires_at - time.time()
        return max(0.0, remaining)
    
    def extend_ttl(self, additional_seconds: float) -> None:
        """
        Extend the TTL by the specified number of seconds.
        
        Args:
            additional_seconds: Number of seconds to add to TTL
        """
        if self.expires_at is not None:
            self.expires_at += additional_seconds 