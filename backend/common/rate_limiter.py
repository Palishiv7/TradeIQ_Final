"""
Rate Limiter Module

This module provides rate limiting functionality to protect the system
from excessive requests and maintain performance under load.
It utilizes Redis for distributed rate limiting across multiple instances.
"""

import time
import asyncio
import logging
from typing import Dict, Optional, Tuple, Callable, Any
from datetime import datetime, timedelta
import hashlib

from redis.asyncio import Redis
from fastapi import Request, Response, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware

from backend.common.logger import get_logger

logger = get_logger(__name__)

class RateLimiter:
    """
    Rate limiting utility to control request frequency.
    
    Uses Redis for distributed rate limiting to ensure consistent
    behavior across multiple application instances.
    
    Examples:
        # Initialize with Redis connection
        limiter = RateLimiter(redis_client)
        
        # Check if action is allowed
        allowed, reset_time = await limiter.check("user:123", max_requests=10, period=60)
        
        # As a FastAPI dependency
        @app.get("/api/resource")
        async def get_resource(is_allowed: bool = Depends(limiter.rate_limit_dependency(10, 60))):
            return {"data": "resource data"}
    """
    
    def __init__(self, redis: Optional[Redis] = None, prefix: str = "rate_limit:"):
        """
        Initialize the rate limiter.
        
        Args:
            redis: Redis client instance (optional, uses in-memory if None)
            prefix: Key prefix for Redis storage
        """
        self.redis = redis
        self.prefix = prefix
        self.local_storage: Dict[str, Tuple[int, float]] = {}  # Fallback for no Redis
    
    async def check(
        self, 
        key: str, 
        max_requests: int, 
        period: int,
        increment: bool = True
    ) -> Tuple[bool, Optional[int]]:
        """
        Check if the rate limit allows another request.
        
        Args:
            key: Unique identifier for the client (e.g., user ID, IP address)
            max_requests: Maximum number of requests allowed in the period
            period: Time period in seconds
            increment: Whether to increment the counter if allowed
            
        Returns:
            Tuple of (is_allowed, reset_time)
            - is_allowed: Whether the request is allowed
            - reset_time: Seconds until the rate limit resets (None if allowed)
        """
        now = time.time()
        redis_key = f"{self.prefix}{key}:{period}"
        
        # Use Redis if available
        if self.redis is not None:
            try:
                current_count = await self.check_redis(redis_key, max_requests, period, increment)
                
                # If count exceeds limit, calculate reset time
                if current_count > max_requests:
                    ttl = await self.redis.ttl(redis_key)
                    return False, max(1, ttl)
                
                return True, None
                
            except Exception as e:
                logger.error(f"Redis rate limit error: {str(e)}")
                # Fall back to local storage if Redis fails
        
        # Use local memory storage as fallback
        return await self.check_local(redis_key, max_requests, period, increment, now)
    
    async def check_redis(
        self, 
        key: str, 
        max_requests: int, 
        period: int,
        increment: bool
    ) -> int:
        """Check rate limit using Redis."""
        # Create key if it doesn't exist
        if increment:
            current = await self.redis.incr(key)
            if current == 1:
                await self.redis.expire(key, period)
        else:
            current = int(await self.redis.get(key) or 0)
            
        return current
    
    async def check_local(
        self, 
        key: str, 
        max_requests: int, 
        period: int,
        increment: bool,
        now: float
    ) -> Tuple[bool, Optional[int]]:
        """Check rate limit using local storage (fallback)."""
        # Get or create entry
        count, expire_time = self.local_storage.get(key, (0, now + period))
        
        # Clean expired entries
        self._clean_expired_local()
        
        # Check if expired
        if now > expire_time:
            if increment:
                self.local_storage[key] = (1, now + period)
            return True, None
            
        # Increment if needed and check limit
        if increment:
            count += 1
            self.local_storage[key] = (count, expire_time)
            
        if count > max_requests:
            return False, int(expire_time - now)
            
        return True, None
    
    def _clean_expired_local(self):
        """Remove expired entries from local storage."""
        now = time.time()
        keys_to_remove = [
            key for key, (_, expire_time) in self.local_storage.items()
            if now > expire_time
        ]
        for key in keys_to_remove:
            del self.local_storage[key]
    
    def rate_limit_dependency(
        self, 
        max_requests: int, 
        period: int, 
        key_func: Optional[Callable[[Request], str]] = None
    ) -> Callable[[Request], Any]:
        """
        Create a FastAPI dependency for rate limiting.
        
        Args:
            max_requests: Maximum number of requests allowed
            period: Time period in seconds
            key_func: Function to extract the key from request (default: uses IP)
            
        Returns:
            FastAPI dependency function
        """
        async def dependency(request: Request) -> bool:
            # Get key from request
            if key_func is None:
                key = self._get_client_ip(request)
            else:
                key = key_func(request)
                
            # Check rate limit
            allowed, reset_time = await self.check(key, max_requests, period)
            
            if not allowed:
                # Add rate limit headers
                headers = {
                    "X-RateLimit-Limit": str(max_requests),
                    "X-RateLimit-Reset": str(reset_time),
                    "Retry-After": str(reset_time)
                }
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"Rate limit exceeded. Try again in {reset_time} seconds.",
                    headers=headers
                )
                
            return True
            
        return dependency
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request."""
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host
    
    def create_middleware(
        self, 
        max_requests: int, 
        period: int,
        key_func: Optional[Callable[[Request], str]] = None,
        error_message: str = "Rate limit exceeded. Please try again later."
    ) -> BaseHTTPMiddleware:
        """
        Create middleware for rate limiting.
        
        Args:
            max_requests: Maximum requests per period
            period: Time period in seconds
            key_func: Function to extract key from request
            error_message: Custom error message
            
        Returns:
            Starlette middleware
        """
        class RateLimitMiddleware(BaseHTTPMiddleware):
            async def dispatch(
                self, request: Request, call_next
            ) -> Response:
                # Get client identifier
                if key_func is None:
                    key = self._rate_limiter._get_client_ip(request)
                else:
                    key = key_func(request)
                    
                # Check rate limit
                allowed, reset_time = await self._rate_limiter.check(
                    key, max_requests, period
                )
                
                if not allowed:
                    # Create response with rate limit headers
                    headers = {
                        "X-RateLimit-Limit": str(max_requests),
                        "X-RateLimit-Reset": str(reset_time),
                        "Retry-After": str(reset_time),
                        "Content-Type": "application/json"
                    }
                    
                    return Response(
                        content=f'{{"error": "{error_message}", "retry_after": {reset_time}}}',
                        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                        headers=headers
                    )
                
                # Process the request normally if allowed
                return await call_next(request)
                
        # Set reference to this rate limiter instance
        middleware = RateLimitMiddleware()
        middleware._rate_limiter = self
        
        return middleware
