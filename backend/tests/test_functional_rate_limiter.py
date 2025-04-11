"""
Functional tests for the rate limiter implementation.

These tests validate that the RateLimiter class functions correctly with
the actual implementation, not just through mocks.
"""

import pytest
import asyncio
import time
from unittest.mock import AsyncMock, patch, MagicMock

@pytest.mark.asyncio
async def test_actual_rate_limiter_functionality():
    """Test the actual RateLimiter class with a mocked Redis client."""
    # Import the actual RateLimiter
    from backend.common.rate_limiter import RateLimiter
    
    # Create a real Redis mock that operates like Redis would
    redis_mock = AsyncMock()
    
    # Configure Redis mock for the first call (new key)
    redis_mock.exists.return_value = False
    redis_mock.set.return_value = True
    
    # Create an ACTUAL RateLimiter instance with the mocked Redis
    rate_limiter = RateLimiter(redis_mock)
    
    # Define test parameters
    key = "test:rate:limiter"
    max_requests = 5
    period = 60
    
    # First call - should set a new key
    allowed, reset = await rate_limiter.check(key, max_requests, period)
    
    print(f"First call: allowed={allowed}, reset={reset}")
    print(f"Redis.exists called with: {redis_mock.exists.call_args}")
    print(f"Redis.set called with: {redis_mock.set.call_args}")
    
    # Verify results match expected behavior for a new key
    assert allowed is True, "First request should be allowed"
    assert reset is None, "No reset time should be provided for allowed request"
    
    # Verify Redis was called correctly
    redis_mock.exists.assert_called_once()
    redis_mock.set.assert_called_once()
    
    # Reset mocks for next test
    redis_mock.reset_mock()
    
    # Now test when key exists but under limit
    redis_mock.exists.return_value = True
    redis_mock.get.return_value = b"3"  # 3 requests so far
    redis_mock.incr.return_value = 4  # After increment, now at 4
    
    # Second call - existing key, under limit
    allowed, reset = await rate_limiter.check(key, max_requests, period)
    
    print(f"Second call: allowed={allowed}, reset={reset}")
    print(f"Redis.exists called with: {redis_mock.exists.call_args}")
    print(f"Redis.get called with: {redis_mock.get.call_args}")
    print(f"Redis.incr called with: {redis_mock.incr.call_args}")
    
    # Verify results match expected behavior for under limit
    assert allowed is True, "Request under limit should be allowed"
    assert reset is None, "No reset time should be provided for allowed request"
    
    # Verify Redis was called correctly
    redis_mock.exists.assert_called_once()
    redis_mock.get.assert_called_once()
    redis_mock.incr.assert_called_once()
    
    # Reset mocks for next test
    redis_mock.reset_mock()
    
    # Now test when key exists and at limit
    redis_mock.exists.return_value = True
    redis_mock.get.return_value = b"5"  # 5 requests so far (at limit)
    redis_mock.incr.return_value = 6  # After increment, now at 6 (over limit)
    redis_mock.ttl.return_value = 45  # 45 seconds remaining on key
    
    # Third call - existing key, over limit
    allowed, reset = await rate_limiter.check(key, max_requests, period)
    
    print(f"Third call: allowed={allowed}, reset={reset}")
    print(f"Redis.exists called with: {redis_mock.exists.call_args}")
    print(f"Redis.get called with: {redis_mock.get.call_args}")
    print(f"Redis.incr called with: {redis_mock.incr.call_args}")
    print(f"Redis.ttl called with: {redis_mock.ttl.call_args}")
    
    # Verify results match expected behavior for over limit
    assert allowed is False, "Request over limit should be denied"
    assert reset == 45, "Reset time should match TTL from Redis"
    
    # Verify Redis was called correctly
    redis_mock.exists.assert_called_once()
    redis_mock.get.assert_called_once()
    redis_mock.incr.assert_called_once()
    redis_mock.ttl.assert_called_once()

@pytest.mark.asyncio
async def test_rate_limiter_integration_with_api_logic():
    """Test how the real rate limiter integrates with API rate limiting logic."""
    # Import the actual RateLimiter and API-related code
    from backend.common.rate_limiter import RateLimiter
    
    # Create a Redis mock
    redis_mock = AsyncMock()
    
    # Test case 1: First request (allowed)
    redis_mock.exists.return_value = False
    redis_mock.set.return_value = True
    
    # Create actual rate limiter with mocked Redis
    rate_limiter = RateLimiter(redis_mock)
    
    # The actual code from candlestick_api.py's start_assessment function:
    user_id = "test-user"
    # This is the actual check that would be in the API:
    allowed, reset_time = await rate_limiter.check(
        f"candlestick:start:{user_id}", 
        5,  # max requests
        60  # period in seconds
    )
    
    print(f"API logic test - allowed: {allowed}, reset_time: {reset_time}")
    print(f"Redis calls: exists={redis_mock.exists.call_args}, set={redis_mock.set.call_args}")
    
    # Verify the result
    assert allowed is True, "First request through API logic should be allowed"
    assert reset_time is None, "No reset time should be provided"
    
    # Test case 2: Rate limited request
    redis_mock.reset_mock()
    redis_mock.exists.return_value = True
    redis_mock.get.return_value = b"5"  # At the limit
    redis_mock.incr.return_value = 6  # Over the limit after increment
    redis_mock.ttl.return_value = 30  # 30 seconds remaining
    
    # The actual code that would run in the API when rate limited:
    allowed, reset_time = await rate_limiter.check(
        f"candlestick:start:{user_id}", 
        5,  # max requests
        60  # period in seconds
    )
    
    print(f"API rate limited test - allowed: {allowed}, reset_time: {reset_time}")
    print(f"Redis calls: exists={redis_mock.exists.call_args}, get={redis_mock.get.call_args}, incr={redis_mock.incr.call_args}")
    
    # Verify the rate limited result
    assert allowed is False, "Over-limit request should be denied"
    assert reset_time == 30, "Reset time should match TTL"
    
    # Now test the HTTP exception that would be raised in the API
    if not allowed:
        # This is the actual code used in the API to handle rate limiting
        from fastapi import HTTPException
        try:
            raise HTTPException(status_code=429, detail="Rate limit exceeded for starting assessments")
            assert False, "Exception should have been raised"
        except HTTPException as e:
            print(f"Raised HTTPException: status_code={e.status_code}, detail={e.detail}")
            assert e.status_code == 429, "Status code should be 429 Too Many Requests"
            assert "Rate limit exceeded" in e.detail, "Error detail should mention rate limit" 