import asyncio
import json
import pickle
import time
import unittest
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

import pytest
import redis

from backend.common.cache import SerializationFormat
from backend.common.cache.service import CacheService
from backend.common.cache.entry import CacheEntry
from backend.common.cache.memory import MemoryCacheBackend
from backend.common.cache.redis import RedisCacheBackend
from backend.common.cache.key_builder import KeyBuilder


class TestCacheEntry(unittest.TestCase):
    """Test the CacheEntry class."""

    def test_init(self):
        """Test initializing a CacheEntry."""
        value = {"test": "value"}
        entry = CacheEntry(value)
        self.assertEqual(entry.value, value)
        self.assertIsNone(entry.expires_at)
        self.assertEqual(entry.access_count, 0)
        self.assertTrue(entry.created_at > 0)  # Should be set to current time
        
        # Test with TTL
        entry = CacheEntry(value, ttl=10)
        self.assertEqual(entry.value, value)
        self.assertTrue(entry.expires_at > entry.created_at)
        self.assertEqual(entry.expires_at, entry.created_at + 10)
        
    def test_is_expired(self):
        """Test checking if a CacheEntry is expired."""
        entry = CacheEntry("test")
        self.assertFalse(entry.is_expired())  # No expiration
        
        # Set expiration in the past
        entry.expires_at = time.time() - 10
        self.assertTrue(entry.is_expired())
        
        # Set expiration in the future
        entry.expires_at = time.time() + 10
        self.assertFalse(entry.is_expired())
        
    def test_access(self):
        """Test accessing a CacheEntry updates stats."""
        entry = CacheEntry("test")
        self.assertEqual(entry.access_count, 0)
        self.assertEqual(entry.last_accessed, entry.created_at)
        
        # Wait a moment to ensure time changes
        time.sleep(0.01)
        
        # Access the entry
        entry.access()
        self.assertEqual(entry.access_count, 1)
        self.assertTrue(entry.last_accessed > entry.created_at)
        
        # Access again
        last_accessed = entry.last_accessed
        time.sleep(0.01)
        entry.access()
        self.assertEqual(entry.access_count, 2)
        self.assertTrue(entry.last_accessed > last_accessed)


class TestMemoryCacheBackend(unittest.TestCase):
    """Test the MemoryCacheBackend class."""
    
    def setUp(self):
        """Set up a MemoryCacheBackend instance for testing."""
        self.cache = MemoryCacheBackend()
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
    def tearDown(self):
        """Clean up the event loop."""
        self.loop.close()
        
    def test_get_set(self):
        """Test setting and getting a value."""
        key = "test_key"
        value = {"test": "value"}
        
        # Set a value
        success = self.loop.run_until_complete(self.cache.set(key, value))
        self.assertTrue(success)
        
        # Get the value
        entry = self.loop.run_until_complete(self.cache.get(key))
        self.assertIsNotNone(entry)
        self.assertEqual(entry.value, value)
        self.assertEqual(entry.access_count, 1)  # Access count should be incremented
        
        # Get a non-existent key
        entry = self.loop.run_until_complete(self.cache.get("nonexistent"))
        self.assertIsNone(entry)
        
    def test_ttl(self):
        """Test time-to-live functionality."""
        key = "test_ttl"
        value = "expires_quickly"
        
        # Set with a short TTL
        self.loop.run_until_complete(self.cache.set(key, value, ttl=0.1))
        
        # Get immediately should succeed
        entry = self.loop.run_until_complete(self.cache.get(key))
        self.assertIsNotNone(entry)
        self.assertEqual(entry.value, value)
        
        # Wait for expiration
        time.sleep(0.2)
        
        # Get after expiration should return None
        entry = self.loop.run_until_complete(self.cache.get(key))
        self.assertIsNone(entry)
        
    def test_delete(self):
        """Test deleting a value."""
        key = "test_delete"
        
        # Set a value
        self.loop.run_until_complete(self.cache.set(key, "to_be_deleted"))
        
        # Verify it exists
        exists = self.loop.run_until_complete(self.cache.exists(key))
        self.assertTrue(exists)
        
        # Delete it
        success = self.loop.run_until_complete(self.cache.delete(key))
        self.assertTrue(success)
        
        # Verify it's gone
        exists = self.loop.run_until_complete(self.cache.exists(key))
        self.assertFalse(exists)
        
        # Delete non-existent key should return False
        success = self.loop.run_until_complete(self.cache.delete("nonexistent"))
        self.assertFalse(success)
        
    def test_clear(self):
        """Test clearing the cache."""
        # Set multiple values
        self.loop.run_until_complete(self.cache.set("key1", "value1"))
        self.loop.run_until_complete(self.cache.set("key2", "value2"))
        
        # Clear the cache
        success = self.loop.run_until_complete(self.cache.clear())
        self.assertTrue(success)
        
        # Verify everything is gone
        exists1 = self.loop.run_until_complete(self.cache.exists("key1"))
        exists2 = self.loop.run_until_complete(self.cache.exists("key2"))
        self.assertFalse(exists1)
        self.assertFalse(exists2)
        
    def test_get_stats(self):
        """Test getting cache statistics."""
        # Set and get some values to generate stats
        self.loop.run_until_complete(self.cache.set("stat_key1", "value1"))
        self.loop.run_until_complete(self.cache.get("stat_key1"))
        self.loop.run_until_complete(self.cache.get("nonexistent"))
        
        # Get stats
        stats = self.loop.run_until_complete(self.cache.get_stats())
        
        # Check the stats
        self.assertEqual(stats["backend"], "memory")
        self.assertEqual(stats["size"], 1)
        self.assertEqual(stats["hits"], 1)
        self.assertEqual(stats["misses"], 1)
        self.assertEqual(stats["hit_rate"], 0.5)
        
    def test_bulk_operations(self):
        """Test bulk get/set/delete operations."""
        # Set multiple values
        items = {
            "bulk1": "value1",
            "bulk2": "value2",
            "bulk3": "value3"
        }
        success = self.loop.run_until_complete(self.cache.set_many(items))
        self.assertTrue(success)
        
        # Get multiple values
        results = self.loop.run_until_complete(self.cache.get_many(["bulk1", "bulk2", "nonexistent"]))
        self.assertEqual(len(results), 2)
        self.assertIn("bulk1", results)
        self.assertIn("bulk2", results)
        self.assertNotIn("nonexistent", results)
        self.assertEqual(results["bulk1"].value, "value1")
        self.assertEqual(results["bulk2"].value, "value2")
        
        # Delete multiple values
        count = self.loop.run_until_complete(self.cache.delete_many(["bulk1", "bulk3", "nonexistent"]))
        self.assertEqual(count, 2)  # Only 2 existed
        
        # Verify deletions
        exists1 = self.loop.run_until_complete(self.cache.exists("bulk1"))
        exists2 = self.loop.run_until_complete(self.cache.exists("bulk2"))
        exists3 = self.loop.run_until_complete(self.cache.exists("bulk3"))
        self.assertFalse(exists1)
        self.assertTrue(exists2)
        self.assertFalse(exists3)


@pytest.mark.asyncio
class TestRedisCacheBackend:
    """Test the RedisCacheBackend class."""
    
    @pytest.fixture
    def mock_redis(self):
        """Create a mock Redis client."""
        mock = MagicMock(spec=redis.Redis)
        # Set up methods to return appropriate values
        mock.ping.return_value = True
        mock.pipeline.return_value = mock  # Return self for pipeline
        mock.execute.return_value = [True, True]  # Default pipeline execution result
        
        return mock
    
    @pytest.fixture
    def cache(self, mock_redis):
        """Create a RedisCacheBackend instance with a mock Redis client."""
        return RedisCacheBackend(redis_client=mock_redis)
    
    async def test_get_set(self, cache, mock_redis):
        """Test setting and getting a value."""
        key = "test_key"
        value = {"test": "value"}
        
        # Set up mock for get
        mock_redis.get.return_value = pickle.dumps(value)
        mock_redis.hgetall.return_value = {
            b"created_at": b"1600000000.0",
            b"expires_at": b"",
            b"access_count": b"0",
            b"last_accessed": b"1600000000.0",
            b"size_estimate": b"0"
        }
        
        # Set a value
        success = await cache.set(key, value)
        self.assertTrue(success)
        
        # Verify Redis calls
        mock_redis.set.assert_called_with("tradeiq:test_key", pickle.dumps(value))
        mock_redis.hset.assert_called()  # Just verify it was called
        
        # Get the value
        entry = await cache.get(key)
        self.assertIsNotNone(entry)
        self.assertEqual(entry.value, value)
        
        # Verify Redis calls for get
        mock_redis.get.assert_called_with("tradeiq:test_key")
        mock_redis.hgetall.assert_called_with("tradeiq:test_key:meta")
        
    async def test_get_nonexistent(self, cache, mock_redis):
        """Test getting a non-existent key."""
        # Set up mock for get to return None
        mock_redis.get.return_value = None
        
        # Get a non-existent key
        entry = await cache.get("nonexistent")
        self.assertIsNone(entry)
        
        # Verify Redis call
        mock_redis.get.assert_called_with("tradeiq:nonexistent")
        
    async def test_ttl(self, cache, mock_redis):
        """Test setting a value with a TTL."""
        key = "test_ttl"
        value = "expires"
        ttl = 60
        
        # Set with TTL
        await cache.set(key, value, ttl)
        
        # Verify Redis calls
        mock_redis.set.assert_called_with("tradeiq:test_ttl", pickle.dumps(value))
        mock_redis.expire.assert_called_with("tradeiq:test_ttl", 60)
        
    async def test_delete(self, cache, mock_redis):
        """Test deleting a value."""
        key = "test_delete"
        
        # Set up mock for delete
        mock_redis.delete.return_value = 2  # Both value and metadata
        
        # Delete the key
        success = await cache.delete(key)
        self.assertTrue(success)
        
        # Verify Redis call
        mock_redis.delete.assert_called_with("tradeiq:test_delete", "tradeiq:test_delete:meta")
        
    async def test_exists(self, cache, mock_redis):
        """Test checking if a key exists."""
        key = "test_exists"
        
        # Set up mocks
        mock_redis.get.return_value = pickle.dumps("test_value")
        mock_redis.hgetall.return_value = {
            b"created_at": b"1600000000.0",
            b"expires_at": b"",
            b"access_count": b"0",
            b"last_accessed": b"1600000000.0",
            b"size_estimate": b"0"
        }
        
        # Check if exists
        exists = await cache.exists(key)
        self.assertTrue(exists)
        
        # Verify Redis calls
        mock_redis.get.assert_called_with("tradeiq:test_exists")
        
    async def test_clear(self, cache, mock_redis):
        """Test clearing the cache."""
        # Set up mock for scan and delete
        mock_redis.scan.side_effect = [(0, [b"tradeiq:key1", b"tradeiq:key2"])]
        mock_redis.delete.return_value = 2
        
        # Clear the cache
        success = await cache.clear()
        self.assertTrue(success)
        
        # Verify Redis calls
        mock_redis.scan.assert_called_with(0, "tradeiq:*", 100)
        mock_redis.delete.assert_called_with(b"tradeiq:key1", b"tradeiq:key2")
        
    async def test_bulk_operations(self, cache, mock_redis):
        """Test bulk operations."""
        # Set up mocks for get_many
        mock_redis.mget.return_value = [pickle.dumps("value1"), pickle.dumps("value2"), None]
        mock_redis.hgetall.return_value = {
            b"created_at": b"1600000000.0",
            b"expires_at": b"",
            b"access_count": b"0",
            b"last_accessed": b"1600000000.0",
            b"size_estimate": b"0"
        }
        
        # Test get_many
        results = await cache.get_many(["bulk1", "bulk2", "nonexistent"])
        self.assertEqual(len(results), 2)  # nonexistent not included
        
        # Verify Redis calls
        mock_redis.mget.assert_called()
        
        # Test set_many
        items = {
            "bulk1": "value1",
            "bulk2": "value2"
        }
        success = await cache.set_many(items)
        self.assertTrue(success)
        
        # Verify Redis pipeline was used
        mock_redis.pipeline.assert_called()
        
        # Test delete_many
        mock_redis.delete.return_value = 4  # 2 keys × 2 (value + metadata)
        count = await cache.delete_many(["bulk1", "bulk2"])
        self.assertEqual(count, 2)


class TestKeyBuilder(unittest.TestCase):
    """Test the KeyBuilder class."""
    
    def test_object_key(self):
        """Test building an object key."""
        # Simple key
        key = KeyBuilder.object_key("user", "123")
        self.assertEqual(key, "user:123")
        
        # With prefix
        key = KeyBuilder.object_key("user", "123", prefix="app")
        self.assertEqual(key, "app:user:123")
        
        # With postfix
        key = KeyBuilder.object_key("user", "123", postfix="details")
        self.assertEqual(key, "user:123:details")
        
        # With both
        key = KeyBuilder.object_key("user", "123", prefix="app", postfix="details")
        self.assertEqual(key, "app:user:123:details")
        
    def test_collection_key(self):
        """Test building a collection key."""
        # Simple collection
        key = KeyBuilder.collection_key("users")
        self.assertEqual(key, "users")
        
        # With prefix
        key = KeyBuilder.collection_key("users", prefix="app")
        self.assertEqual(key, "app:users")
        
        # With filter
        key = KeyBuilder.collection_key("users", filter_dict={"status": "active"})
        self.assertEqual(key, "users:status=active")
        
        # With multiple filters
        key = KeyBuilder.collection_key(
            "users", 
            filter_dict={"status": "active", "role": "admin"}
        )
        # Order is not guaranteed, so check parts
        self.assertTrue(key.startswith("users:"))
        self.assertIn("status=active", key)
        self.assertIn("role=admin", key)
        
        # With prefix and filter
        key = KeyBuilder.collection_key(
            "users", 
            prefix="app",
            filter_dict={"status": "active"}
        )
        self.assertEqual(key, "app:users:status=active")
        
        # With list value in filter
        key = KeyBuilder.collection_key(
            "products", 
            filter_dict={"categories": ["electronics", "phones"]}
        )
        self.assertEqual(key, "products:categories=[electronics,phones]")
        
    def test_function_key(self):
        """Test building a function result key."""
        # Simple function
        key = KeyBuilder.function_key("get_user", {"id": 123})
        self.assertEqual(key, "fn:get_user:id=123")
        
        # With prefix
        key = KeyBuilder.function_key("get_user", {"id": 123}, prefix="app")
        self.assertEqual(key, "app:fn:get_user:id=123")
        
        # With complex args
        key = KeyBuilder.function_key(
            "search_users", 
            {
                "query": "john",
                "filters": {"status": "active", "age": [20, 30]}
            }
        )
        # Order is not guaranteed, so check parts
        self.assertTrue(key.startswith("fn:search_users:"))
        self.assertIn("query=john", key)
        self.assertIn("filters=", key)


class TestCacheService(unittest.TestCase):
    """Test the CacheService class."""
    
    def setUp(self):
        """Set up a CacheService instance for testing."""
        self.memory_backend = MemoryCacheBackend()
        self.service = CacheService(self.memory_backend)
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
    def tearDown(self):
        """Clean up the event loop."""
        self.loop.close()
        
    def test_get_set(self):
        """Test basic get/set operations."""
        key = "test_service_key"
        value = {"name": "test"}
        
        # Set a value
        self.loop.run_until_complete(self.service.set(key, value))
        
        # Get the value
        result = self.loop.run_until_complete(self.service.get(key))
        self.assertEqual(result, value)
        
        # Get with default
        nonexistent = self.loop.run_until_complete(
            self.service.get("nonexistent", default={"default": True})
        )
        self.assertEqual(nonexistent, {"default": True})
        
    def test_object_cache(self):
        """Test caching objects with object_key."""
        # Cache a user object
        user = {"id": 123, "name": "John"}
        key = KeyBuilder.object_key("user", 123)
        
        self.loop.run_until_complete(self.service.set(key, user))
        
        # Get the user
        result = self.loop.run_until_complete(self.service.get(key))
        self.assertEqual(result, user)
        
    def test_collection_cache(self):
        """Test caching collections with collection_key."""
        # Cache a collection of users
        users = [
            {"id": 1, "name": "John"},
            {"id": 2, "name": "Jane"}
        ]
        key = KeyBuilder.collection_key("users", filter_dict={"status": "active"})
        
        self.loop.run_until_complete(self.service.set(key, users))
        
        # Get the collection
        result = self.loop.run_until_complete(self.service.get(key))
        self.assertEqual(result, users)
        
    def test_function_result_cache(self):
        """Test caching function results."""
        # Set up a test function
        call_count = 0
        
        async def get_user(user_id):
            nonlocal call_count
            call_count += 1
            return {"id": user_id, "name": f"User {user_id}"}
        
        # Create a cached version
        cached_get_user = self.service.cached_function(get_user)
        
        # Call the function for the first time
        result1 = self.loop.run_until_complete(cached_get_user(123))
        self.assertEqual(result1, {"id": 123, "name": "User 123"})
        self.assertEqual(call_count, 1)
        
        # Call again with the same arg (should use cache)
        result2 = self.loop.run_until_complete(cached_get_user(123))
        self.assertEqual(result2, {"id": 123, "name": "User 123"})
        self.assertEqual(call_count, 1)  # No additional call
        
        # Call with a different arg
        result3 = self.loop.run_until_complete(cached_get_user(456))
        self.assertEqual(result3, {"id": 456, "name": "User 456"})
        self.assertEqual(call_count, 2)  # One additional call
        
    def test_invalidation(self):
        """Test cache invalidation."""
        # Set multiple values
        self.loop.run_until_complete(self.service.set("key1", "value1"))
        self.loop.run_until_complete(self.service.set("key2", "value2"))
        
        # Invalidate one key
        self.loop.run_until_complete(self.service.invalidate("key1"))
        
        # Check results
        val1 = self.loop.run_until_complete(self.service.get("key1"))
        val2 = self.loop.run_until_complete(self.service.get("key2"))
        self.assertIsNone(val1)
        self.assertEqual(val2, "value2")
        
        # Invalidate all
        self.loop.run_until_complete(self.service.invalidate_all())
        
        # Check all gone
        val2 = self.loop.run_until_complete(self.service.get("key2"))
        self.assertIsNone(val2)


if __name__ == "__main__":
    unittest.main() 