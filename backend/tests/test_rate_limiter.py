"""
Tests for the rate limiter component.

This module contains tests for the RateLimiter class, focusing on:
1. The check method 
2. Rate limit enforcement
3. Handling of various inputs
"""

import pytest
import time
from unittest.mock import AsyncMock, patch, MagicMock

@pytest.mark.asyncio
async def test_rate_limiter_check():
    """Test that RateLimiter.check works correctly."""
    # Import here to avoid module-level import issues
    from backend.common.rate_limiter import RateLimiter
    
    # Create mocks
    redis_mock = AsyncMock()
    
    # Setup the mock redis calls
    redis_mock.incr.return_value = 1  # First request
    redis_mock.expire.return_value = True
    redis_mock.get.return_value = None
    
    # Create rate limiter with mock redis
    rate_limiter = RateLimiter(redis_mock)
    
    # Test with a new key (should be allowed)
    result, reset_time = await rate_limiter.check(
        "test:key:1", 
        max_requests=5,
        period=60
    )
    
    # Should be allowed since it's a new key
    assert result is True
    assert reset_time is None
    
    # Verify Redis interactions
    redis_mock.incr.assert_called_once()
    redis_mock.expire.assert_called_once()

@pytest.mark.asyncio
async def test_rate_limiter_enforces_limits():
    """Test that rate limits are properly enforced."""
    # Import here to avoid module-level import issues
    from backend.common.rate_limiter import RateLimiter
    
    # Create mocks
    redis_mock = AsyncMock()
    
    # Mock Redis calls for a key that exceeds the limit
    redis_mock.incr.return_value = 6  # Exceeds limit of 5
    redis_mock.ttl.return_value = 30  # 30 seconds remaining
    
    # Create rate limiter with mock redis
    rate_limiter = RateLimiter(redis_mock)
    
    # Test with key that exceeds the limit
    result, reset_time = await rate_limiter.check(
        "test:key:2", 
        max_requests=5,
        period=60
    )
    
    # Should be rate limited
    assert result is False
    assert reset_time == 30  # Should match the ttl 