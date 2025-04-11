"""
API Integration Tests for TradeIQ Backend

This module contains comprehensive functional tests for the TradeIQ backend APIs, including:
1. Normal operation scenarios with valid inputs
2. Edge cases and error conditions
3. API resilience under adverse conditions
4. Load testing simulations

These tests use real Redis and database connections to provide proper functional testing.
"""

import os
import time
import uuid
import asyncio
import pytest
import json
from fastapi.testclient import TestClient
from concurrent.futures import ThreadPoolExecutor
from starlette.middleware.base import BaseHTTPMiddleware

# Import application components
from backend.main import create_app
from backend.assessments.candlestick_patterns.candlestick_db import session_manager, candlestick_cache
from backend.assessments.candlestick_patterns.candlestick_api import router as candlestick_router
from database.init_db import initialize_database
from backend.tests.test_models import PatternStatistics, UserPerformance, Base
from backend.auth.dependencies import get_current_user_id
from backend.cache.redis_client import RedisClient
from backend.common.rate_limiter import RateLimiter
from backend.common.logger import get_logger

# Configuration for tests
TEST_USER_ID = "test-user-1"
redis_client = RedisClient()

# Create a synchronous wrapper for rate limiter for testing
@pytest.fixture(scope="module", autouse=True)
def sync_rate_limiter():
    """Create a synchronous wrapper for the rate limiter to use in tests"""
    # Save the original check method
    original_check = RateLimiter.check
    
    # Create a modified check method that returns an awaitable object
    async def patched_check(self, key, max_requests, period, increment=True):
        try:
            # Just allow all requests during testing
            return True, None
        except Exception as e:
            logger = get_logger(__name__)
            logger.error(f"Error in rate limiter: {str(e)}")
            return True, None
    
    # Replace the check method temporarily
    RateLimiter.check = patched_check
    
    # After tests, restore the original method
    yield
    RateLimiter.check = original_check

# Fixture to create a test application using real infrastructure
@pytest.fixture(scope="module")
def app():
    """Create test application instance with real infrastructure connections"""
    # Override auth dependency for testing
    async def get_test_user_id():
        return TEST_USER_ID

    app = create_app()
    
    # Override the auth dependency with a fixed test user
    app.dependency_overrides[get_current_user_id] = get_test_user_id
    
    return app

@pytest.fixture(scope="module")
def client(app):
    """Create test client for the application"""
    with TestClient(app) as client:
        # The TestClient runs with a separate event loop for each request,
        # but we need to make sure we handle async operations properly
        yield client

@pytest.fixture(scope="module")
def event_loop():
    """Create an event loop for each test module"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()

@pytest.fixture(scope="module", autouse=True)
async def setup_test_db():
    """Setup test database and Redis with test data"""
    # Use a separate database for testing
    os.environ["DB_CONNECTION_STRING"] = "sqlite:///./test.db"
    
    # Initialize the database with test schemas
    initialize_database(test_mode=True)
    
    # Clear any existing data in Redis test database
    await redis_client.flushdb()
    
    # Initialize the cache for patterns
    await candlestick_cache.initialize_pattern_stats()
    
    # Seed some pattern statistics to Redis
    pattern_stats = {
        "Hammer": {
            "attempts": 100,
            "correct": 75,
            "avg_response_time": 2.5,
            "success_rate": 0.75
        },
        "Doji": {
            "attempts": 80,
            "correct": 40,
            "avg_response_time": 4.1,
            "success_rate": 0.5
        },
        "Bullish Engulfing": {
            "attempts": 150,
            "correct": 90,
            "avg_response_time": 3.2,
            "success_rate": 0.6
        }
    }
    
    for pattern, stats in pattern_stats.items():
        key = f"candlestick:pattern:stats:{pattern}"
        await redis_client.hmset(key, stats)
        
    # Return cleanup function
    yield
    
    # Clean up after tests
    await redis_client.flushdb()
    
    # Remove the test database
    if os.path.exists("./test.db"):
        os.remove("./test.db")


# Test cases for normal operation
class TestNormalOperation:
    """Tests for normal API operation using real infrastructure"""

    def test_start_assessment(self, client):
        """Test starting a new assessment session"""
        # Arrange
        payload = {
            "difficulty": 0.5,
            "total_questions": 5
        }
        
        # Act
        response = client.post("/v1/candlestick-patterns/start", json=payload)
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert "question_id" in data
        assert "question_text" in data
        assert "options" in data
        assert "question_number" in data
        assert data["question_number"] == 1
        assert data["total_questions"] == 5
        assert "session_id" in data

    def test_submit_answer_correct(self, client):
        """Test submitting a correct answer"""
        # Arrange - Start a session first
        start_payload = {
            "difficulty": 0.5,
            "total_questions": 5
        }
        start_response = client.post("/v1/candlestick-patterns/start", json=start_payload)
        session_data = start_response.json()
        
        # Get session ID and question data from response
        session_id = session_data["session_id"]
        question_id = session_data["question_id"]
        selected_option = session_data["options"][0]  # Just pick the first option
        
        # Prepare answer submission
        answer_payload = {
            "session_id": session_id,
            "question_id": question_id,
            "selected_option": selected_option,
            "response_time_ms": 1500
        }
        
        # Act
        response = client.post("/v1/candlestick-patterns/submit_answer", json=answer_payload)
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert "is_correct" in data
        assert "next_question" in data or "assessment_complete" in data
        assert "score" in data
    
    def test_submit_answer_incorrect(self, client):
        """Test submitting an incorrect answer"""
        # Arrange - Start a session first
        start_payload = {
            "difficulty": 0.5,
            "total_questions": 5
        }
        start_response = client.post("/v1/candlestick-patterns/start", json=start_payload)
        session_data = start_response.json()
        
        # Get session ID and question data
        session_id = session_data["session_id"]
        question_id = session_data["question_id"]
        
        # Find incorrect answer - use second option if available
        options = session_data["options"]
        selected_option = options[1] if len(options) > 1 else options[0]
        
        # Prepare answer submission
        answer_payload = {
            "session_id": session_id,
            "question_id": question_id,
            "selected_option": selected_option,
            "response_time_ms": 1500
        }
        
        # Act
        response = client.post("/v1/candlestick-patterns/submit_answer", json=answer_payload)
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert "is_correct" in data
        assert "explanation" in data
        assert "next_question" in data or "assessment_complete" in data
    
    def test_get_session_statistics(self, client):
        """Test retrieving session statistics"""
        # Arrange - Start a session and submit answers
        start_payload = {
            "difficulty": 0.5,
            "total_questions": 2
        }
        start_response = client.post("/v1/candlestick-patterns/start", json=start_payload)
        session_data = start_response.json()
        session_id = session_data["session_id"]
        
        # Submit first answer
        answer_payload = {
            "session_id": session_id,
            "question_id": session_data["question_id"],
            "selected_option": session_data["options"][0],
            "response_time_ms": 1500
        }
        submit_response = client.post("/v1/candlestick-patterns/submit_answer", json=answer_payload)
        response_data = submit_response.json()
        
        # Submit second answer
        if "next_question" in response_data and response_data["next_question"]:
            second_q = response_data["next_question"]
        
        answer_payload = {
            "session_id": session_id,
            "question_id": second_q["question_id"],
            "selected_option": second_q["options"][0],
            "response_time_ms": 2000
        }
        client.post("/v1/candlestick-patterns/submit_answer", json=answer_payload)
        
        # Act
        response = client.get(f"/v1/candlestick-patterns/session/{session_id}")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == session_id
        assert data["total_questions"] == 2
        assert "completed_questions" in data
        assert "correct_answers" in data
        assert "avg_response_time" in data
        assert "score" in data


# Test cases for edge cases and error handling
class TestEdgeCases:
    """Tests for edge cases and error handling using real infrastructure"""
    
    def test_invalid_session_id(self, client):
        """Test submitting an answer with invalid session ID"""
        # Arrange
        payload = {
            "session_id": "non-existent-session",
            "question_id": "q123",
            "selected_option": "Hammer",
            "response_time_ms": 1500
        }
        
        # Act
        response = client.post("/v1/candlestick-patterns/submit_answer", json=payload)
        
        # Assert
        assert response.status_code in [404, 400]  # Either not found or bad request is acceptable
        assert "detail" in response.json()
    
    def test_malformed_session_id(self, client):
        """Test submitting an answer with malformed session ID"""
        # Arrange
        payload = {
            "session_id": "!@#$%^&*()",  # Invalid characters
            "question_id": "q123",
            "selected_option": "Hammer",
            "response_time_ms": 1500
        }
        
        # Act
        response = client.post("/v1/candlestick-patterns/submit_answer", json=payload)
        
        # Assert
        assert response.status_code in [404, 400]
        assert "detail" in response.json()
    
    def test_invalid_question_id(self, client):
        """Test submitting with invalid question ID"""
        # Arrange - Start a valid session
        start_payload = {
            "difficulty": 0.5,
            "total_questions": 5
        }
        start_response = client.post("/v1/candlestick-patterns/start", json=start_payload)
        session_data = start_response.json()
        
        # Invalid question
        answer_payload = {
            "session_id": session_data["session_id"],
            "question_id": "invalid-question-id",
            "selected_option": "Hammer",
            "response_time_ms": 1500
        }
        
        # Act
        response = client.post("/v1/candlestick-patterns/submit_answer", json=answer_payload)
        
        # Assert
        assert response.status_code in [404, 400]  # Either not found or bad request is acceptable
        assert "detail" in response.json()
    
    def test_extremely_slow_response(self, client):
        """Test handling extremely slow response times"""
        # Arrange - Start a valid session
        start_payload = {
            "difficulty": 0.5,
            "total_questions": 5
        }
        start_response = client.post("/v1/candlestick-patterns/start", json=start_payload)
        session_data = start_response.json()
        
        # Very slow response
        answer_payload = {
            "session_id": session_data["session_id"],
            "question_id": session_data["question_id"],
            "selected_option": session_data["options"][0],
            "response_time_ms": 300000  # 5 minutes
        }
        
        # Act
        response = client.post("/v1/candlestick-patterns/submit_answer", json=answer_payload)
        
        # Assert - should either work with low score or reject extremely slow response
        assert response.status_code in [200, 400]
        if response.status_code == 200:
            data = response.json()
            assert "score" in data
    
    def test_negative_response_time(self, client):
        """Test handling negative response times"""
        # Arrange - Start a valid session
        start_payload = {
            "difficulty": 0.5,
            "total_questions": 5
        }
        start_response = client.post("/v1/candlestick-patterns/start", json=start_payload)
        session_data = start_response.json()
        
        # Negative response time
        answer_payload = {
            "session_id": session_data["session_id"],
            "question_id": session_data["question_id"],
            "selected_option": session_data["options"][0],
            "response_time_ms": -500
        }
        
        # Act
        response = client.post("/v1/candlestick-patterns/submit_answer", json=answer_payload)
        
        # Assert - should reject negative response time
        assert response.status_code == 400
        assert "detail" in response.json()
    
    def test_zero_response_time(self, client):
        """Test handling zero response time"""
        # Arrange - Start a valid session
        start_payload = {
            "difficulty": 0.5,
            "total_questions": 5
        }
        start_response = client.post("/v1/candlestick-patterns/start", json=start_payload)
        session_data = start_response.json()
        
        # Zero response time
        answer_payload = {
            "session_id": session_data["session_id"],
            "question_id": session_data["question_id"],
            "selected_option": session_data["options"][0],
            "response_time_ms": 0
        }
        
        # Act
        response = client.post("/v1/candlestick-patterns/submit_answer", json=answer_payload)
        
        # Assert - system should handle zero response time gracefully
        assert response.status_code in [200, 400]  # Either accept it with low score or reject
    
    def test_extremely_high_difficulty(self, client):
        """Test starting assessment with very high difficulty"""
        # Arrange
        payload = {
            "difficulty": 1.5,  # Above maximum
            "total_questions": 5
        }
        
        # Act
        response = client.post("/v1/candlestick-patterns/start", json=payload)
        
        # Assert - should either clamp to valid range or reject
        if response.status_code == 200:
            data = response.json()
            assert data["difficulty"] <= 1.0  # Should be clamped to valid range
        else:
        assert response.status_code == 400
            assert "detail" in response.json()
    
    def test_extremely_low_difficulty(self, client):
        """Test starting assessment with very low difficulty"""
        # Arrange
        payload = {
            "difficulty": -0.5,  # Below minimum
            "total_questions": 5
        }
        
        # Act
        response = client.post("/v1/candlestick-patterns/start", json=payload)
        
        # Assert - should either clamp to valid range or reject
        if response.status_code == 200:
            data = response.json()
            assert data["difficulty"] >= 0.0  # Should be clamped to valid range
        else:
            assert response.status_code == 400
            assert "detail" in response.json()
    
    def test_large_number_of_questions(self, client):
        """Test starting assessment with a large number of questions"""
        # Arrange
        payload = {
            "difficulty": 0.5,
            "total_questions": 100  # Very large
        }
        
        # Act
        response = client.post("/v1/candlestick-patterns/start", json=payload)
        
        # Assert - should either cap the number or reject
        if response.status_code == 200:
            data = response.json()
            # The number might be capped to a reasonable max
            assert "total_questions" in data
        else:
            assert response.status_code == 400
            assert "detail" in response.json()
    
    def test_zero_questions(self, client):
        """Test starting assessment with zero questions"""
        # Arrange
        payload = {
            "difficulty": 0.5,
            "total_questions": 0
        }
        
        # Act
        response = client.post("/v1/candlestick-patterns/start", json=payload)
        
        # Assert - should reject or use default
        if response.status_code == 200:
        data = response.json()
            assert data["total_questions"] > 0  # Should use default
        else:
            assert response.status_code == 400
            assert "detail" in response.json()
    
    def test_submit_non_existent_option(self, client):
        """Test submitting an option that doesn't exist"""
        # Arrange - Start a valid session
        start_payload = {
            "difficulty": 0.5,
            "total_questions": 5
        }
        start_response = client.post("/v1/candlestick-patterns/start", json=start_payload)
        session_data = start_response.json()
        
        # Non-existent option
        answer_payload = {
            "session_id": session_data["session_id"],
            "question_id": session_data["question_id"],
            "selected_option": "This Option Doesn't Exist",
            "response_time_ms": 1500
        }
        
        # Act
        response = client.post("/v1/candlestick-patterns/submit_answer", json=answer_payload)
        
        # Assert - should reject invalid option
        assert response.status_code in [400, 422]
        assert "detail" in response.json()
    
    def test_submit_answer_twice(self, client):
        """Test submitting the same answer twice"""
        # Arrange - Start a valid session
        start_payload = {
            "difficulty": 0.5,
            "total_questions": 5
        }
        start_response = client.post("/v1/candlestick-patterns/start", json=start_payload)
        session_data = start_response.json()
        
        # Prepare answer payload
        answer_payload = {
            "session_id": session_data["session_id"],
            "question_id": session_data["question_id"],
            "selected_option": session_data["options"][0],
            "response_time_ms": 1500
        }
        
        # Submit first answer
        first_response = client.post("/v1/candlestick-patterns/submit_answer", json=answer_payload)
        assert first_response.status_code == 200
        
        # Act - Submit the same answer again
        second_response = client.post("/v1/candlestick-patterns/submit_answer", json=answer_payload)
        
        # Assert - should either reject or handle gracefully
        if second_response.status_code == 200:
            # If it accepts, it should have the same result as the first submission
            assert "is_correct" in second_response.json()
        else:
            # Or it should reject with appropriate error
            assert second_response.status_code in [400, 409]  # 409 Conflict would be appropriate
    
    def test_empty_request_body(self, client):
        """Test sending empty request body"""
        # Act - Send empty JSON body
        response = client.post("/v1/candlestick-patterns/submit_answer", json={})
    
        # Assert - should reject with validation error
        assert response.status_code in [400, 422]  # 422 Unprocessable Entity is common for validation
        
        # Check response has validation details (FastAPI format or custom format)
        response_data = response.json()
        assert "detail" in response_data or "details" in response_data or "message" in response_data
    
    def test_missing_required_fields(self, client):
        """Test request with missing required fields"""
        # Arrange - Start a valid session
        start_payload = {
            "difficulty": 0.5,
            "total_questions": 5
        }
        start_response = client.post("/v1/candlestick-patterns/start", json=start_payload)
        session_data = start_response.json()
    
        # Missing question_id
        answer_payload = {
            "session_id": session_data["session_id"],
            # "question_id": missing,
            "selected_option": session_data["options"][0],
            "response_time_ms": 1500
        }
    
        # Act
        response = client.post("/v1/candlestick-patterns/submit_answer", json=answer_payload)
    
        # Assert - should reject with validation error
        assert response.status_code in [400, 422]
        
        # Check response has validation details (FastAPI format or custom format)
        response_data = response.json()
        assert "detail" in response_data or "details" in response_data or "message" in response_data
    
    def test_invalid_http_method(self, client):
        """Test using wrong HTTP method"""
        # Act - Use DELETE instead of POST
        response = client.delete("/v1/candlestick-patterns/start")
        
        # Assert - should reject with method not allowed
        assert response.status_code == 405  # Method Not Allowed
    
    def test_empty_session_id(self, client):
        """Test submitting with empty session ID"""
        # Arrange
        answer_payload = {
            "session_id": "",
            "question_id": "some-question-id",
            "selected_option": "Some Option",
            "response_time_ms": 1500
        }
        
        # Act
        response = client.post("/v1/candlestick-patterns/submit_answer", json=answer_payload)
        
        # Assert - should reject empty session ID
        assert response.status_code in [400, 422]
        assert "detail" in response.json()
    
    def test_malformed_json(self, client):
        """Test sending malformed JSON"""
        # Act - Send invalid JSON
        response = client.post(
            "/v1/candlestick-patterns/start", 
            data="{difficulty: 0.5, total_questions: 5}",  # Invalid JSON (missing quotes)
            headers={"Content-Type": "application/json"}
        )
        
        # Assert - should reject with appropriate error
        assert response.status_code in [400, 422]
    
    def test_unexpected_content_type(self, client):
        """Test sending unexpected content type"""
        # Act - Send form data instead of JSON
        response = client.post(
            "/v1/candlestick-patterns/start",
            data={"difficulty": 0.5, "total_questions": 5},  # Form data
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
    
        # Assert - should reject or handle appropriately
        # FastAPI might actually handle this gracefully, so we're flexible with the status code
        if response.status_code not in [200, 201]:
            assert response.status_code in [400, 415, 422]  # 415 Unsupported Media Type or 422 Unprocessable Entity


# Test cases for session completion and resilience
class TestResilience:
    """Tests for system resilience and session completion"""
    
    def test_session_completion(self, client):
        """Test completing an entire assessment session"""
        # Arrange - Start with a short session
        start_payload = {
            "difficulty": 0.5,
            "total_questions": 3
        }
        start_response = client.post("/v1/candlestick-patterns/start", json=start_payload)
        session_data = start_response.json()
        session_id = session_data["session_id"]
        
        # Act - Complete all questions
        current_question = {
            "question_id": session_data["question_id"],
            "options": session_data["options"]
        }
        
        for i in range(3):
            answer_payload = {
                "session_id": session_id,
                "question_id": current_question["question_id"],
                "selected_option": current_question["options"][0],
                "response_time_ms": 1500
            }
            
            response = client.post("/v1/candlestick-patterns/submit_answer", json=answer_payload)
            assert response.status_code == 200
            
            response_data = response.json()
            if i < 2 and "next_question" in response_data:  # Not the last question
                current_question = response_data["next_question"]
                assert current_question is not None
        
        # Check session status
        response = client.get(f"/v1/candlestick-patterns/session/{session_id}")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert "completed_questions" in data
        assert "total_questions" in data
    
    def test_answer_after_completion(self, client):
        """Test submitting an answer after session completion"""
        # Arrange - Start with a very short session
        start_payload = {
            "difficulty": 0.5,
            "total_questions": 1
        }
        start_response = client.post("/v1/candlestick-patterns/start", json=start_payload)
        session_data = start_response.json()
        session_id = session_data["session_id"]
        
        # Complete the session by answering the only question
        answer_payload = {
            "session_id": session_id,
            "question_id": session_data["question_id"],
            "selected_option": session_data["options"][0],
            "response_time_ms": 1500
        }
        
        response = client.post("/v1/candlestick-patterns/submit_answer", json=answer_payload)
        assert response.status_code == 200
        
        # Act - Try to submit another answer to the completed session
        second_answer_payload = {
            "session_id": session_id,
            "question_id": session_data["question_id"],  # Same question ID
            "selected_option": session_data["options"][0],
            "response_time_ms": 2000
        }
        
        response = client.post("/v1/candlestick-patterns/submit_answer", json=second_answer_payload)
        
        # Assert - System should reject or handle gracefully
        assert response.status_code in [400, 403, 409]  # 403 Forbidden or 409 Conflict would be appropriate
        assert "detail" in response.json()
    
    def test_cross_session_question_access(self, client):
        """Test accessing question from a different session"""
        # Arrange - Create two separate sessions
        start_payload = {
            "difficulty": 0.5,
            "total_questions": 3
        }
        
        # Start first session
        first_response = client.post("/v1/candlestick-patterns/start", json=start_payload)
        first_session = first_response.json()
        
        # Start second session
        second_response = client.post("/v1/candlestick-patterns/start", json=start_payload)
        second_session = second_response.json()
        
        # Act - Try to submit answer to first session's question using second session's ID
        answer_payload = {
            "session_id": second_session["session_id"],
            "question_id": first_session["question_id"],  # Question from first session
            "selected_option": first_session["options"][0],
            "response_time_ms": 1500
        }
        
        response = client.post("/v1/candlestick-patterns/submit_answer", json=answer_payload)
        
        # Assert - System should reject cross-session access
        assert response.status_code in [400, 404]
        assert "detail" in response.json()
    
    def test_sequential_session_start(self, client):
        """Test starting multiple sessions sequentially"""
        # Act - Start several sessions in sequence
        sessions = []
        for _ in range(5):
            payload = {
                "difficulty": 0.5,
                "total_questions": 3
            }
            
            response = client.post("/v1/candlestick-patterns/start", json=payload)
            assert response.status_code == 200
            
            session_data = response.json()
            assert "session_id" in session_data
            sessions.append(session_data["session_id"])
        
        # Assert - Each session should be accessible
        for session_id in sessions:
            response = client.get(f"/v1/candlestick-patterns/session/{session_id}")
            assert response.status_code == 200
    
    def test_session_after_restart(self, client, monkeypatch):
        """Test session persistence after service restart simulation"""
        # Arrange - Start a session
        start_payload = {
            "difficulty": 0.5,
            "total_questions": 3
        }
        
        start_response = client.post("/v1/candlestick-patterns/start", json=start_payload)
        session_data = start_response.json()
        session_id = session_data["session_id"]
        
        # Answer first question
        answer_payload = {
            "session_id": session_id,
            "question_id": session_data["question_id"],
            "selected_option": session_data["options"][0],
            "response_time_ms": 1500
        }
        
        client.post("/v1/candlestick-patterns/submit_answer", json=answer_payload)
        
        # Act - Simulate service restart by resetting in-memory cache
        # This is a simplified simulation - in a real test you might restart the actual service
        from backend.assessments.candlestick_patterns.candlestick_db import SessionManager
        original_init = SessionManager.__init__
        
        # Create a new instance to simulate restart
        def mock_init(self, redis_client=None):
            original_init(self, redis_client)
            # Clear any in-memory cache
            if hasattr(self, "_sessions_cache"):
                self._sessions_cache = {}
        
        monkeypatch.setattr(SessionManager, "__init__", mock_init)
        
        # Check if session is still accessible
        response = client.get(f"/v1/candlestick-patterns/session/{session_id}")
        
        # Assert - Session should be retrievable from persistent storage
        assert response.status_code == 200
        assert "completed_questions" in response.json()
        
        # Try to continue the session by submitting another answer
        if "next_question" in response.json():
            next_q = response.json()["next_question"]
            if next_q:
                next_answer = {
                    "session_id": session_id,
                    "question_id": next_q["question_id"],
                    "selected_option": next_q["options"][0],
                    "response_time_ms": 2000
                }
                
                continue_response = client.post("/v1/candlestick-patterns/submit_answer", json=next_answer)
                assert continue_response.status_code == 200
    
    def test_response_under_load(self, client):
        """Test API response times under moderate load"""
        import time
        
        # Arrange - Prepare data for multiple requests
        total_requests = 20
        payload = {
            "difficulty": 0.5,
            "total_questions": 1
        }
        
        # Act - Send requests in quick succession
        start_time = time.time()
        responses = []
        
        for _ in range(total_requests):
            response = client.post("/v1/candlestick-patterns/start", json=payload)
            responses.append(response)
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        # Assert - All requests should succeed and finish in reasonable time
        successful = sum(1 for r in responses if r.status_code == 200)
        assert successful == total_requests
        
        # Check if average time per request is reasonable (adjust threshold as needed)
        avg_time_per_request = total_duration / total_requests
        assert avg_time_per_request < 0.5  # 500ms per request is reasonable
    
    def test_partial_session_completion(self, client):
        """Test retrieving and working with partially completed sessions"""
        # Arrange - Start session and answer only some questions
        start_payload = {
            "difficulty": 0.5,
            "total_questions": 5
        }
        
        start_response = client.post("/v1/candlestick-patterns/start", json=start_payload)
        session_data = start_response.json()
        session_id = session_data["session_id"]
        
        # Answer only 2 questions in a 5-question session
        current_question = {
            "question_id": session_data["question_id"],
            "options": session_data["options"]
        }
        
        for i in range(2):
            answer_payload = {
                "session_id": session_id,
                "question_id": current_question["question_id"],
                "selected_option": current_question["options"][0],
                "response_time_ms": 1500
            }
            
            response = client.post("/v1/candlestick-patterns/submit_answer", json=answer_payload)
            assert response.status_code == 200
            
            response_data = response.json()
            if "next_question" in response_data:
                current_question = response_data["next_question"]
        
        # Act - Get the partial session stats
        response = client.get(f"/v1/candlestick-patterns/session/{session_id}")
        
        # Assert - Session should show partial completion
        assert response.status_code == 200
        data = response.json()
        assert data["completed_questions"] == 2
        assert data["total_questions"] == 5
        assert not data.get("completed", False)  # Session should not be marked as completed
    
    def test_long_running_session(self, client, monkeypatch):
        """Test handling of a long-running session with delays between answers"""
        import time
        
        # Arrange - Start a session
        start_payload = {
            "difficulty": 0.5,
            "total_questions": 3
        }
        
        start_response = client.post("/v1/candlestick-patterns/start", json=start_payload)
        session_data = start_response.json()
        session_id = session_data["session_id"]
        
        # Mock time to simulate passage of time without actually waiting
        original_time = time.time
        mock_time = [original_time()]
        
        def mock_time_func():
            return mock_time[0]
        
        monkeypatch.setattr(time, "time", mock_time_func)
        
        # Act - Answer questions with simulated long delays
        current_question = {
            "question_id": session_data["question_id"],
            "options": session_data["options"]
        }
        
        for i in range(3):
            # Simulate 5 minute delay between questions
            mock_time[0] += 300  # 5 minutes in seconds
            
            answer_payload = {
                "session_id": session_id,
                "question_id": current_question["question_id"],
                "selected_option": current_question["options"][0],
                "response_time_ms": 10000  # 10 seconds response time
            }
            
            response = client.post("/v1/candlestick-patterns/submit_answer", json=answer_payload)
            assert response.status_code == 200
            
            response_data = response.json()
            if i < 2 and "next_question" in response_data:
                current_question = response_data["next_question"]
        
        # Check final session stats
        response = client.get(f"/v1/candlestick-patterns/session/{session_id}")
        
        # Assert - Session should complete successfully despite long delays
        assert response.status_code == 200
        data = response.json()
        assert data["completed_questions"] == 3
        assert data["total_questions"] == 3


# Load testing simulation
@pytest.mark.skip(reason="Load testing should be run separately")
class TestLoadSimulation:
    """Simulated load testing using real infrastructure"""
    
    def test_concurrent_requests(self, client):
        """Test system behavior under concurrent requests"""
        # Arrange
        num_requests = 10
        payload = {
            "difficulty": 0.5,
            "total_questions": 1
        }
        
        # Act - Send multiple concurrent requests
        import threading
        responses = []
        
        def make_request():
            resp = client.post("/v1/candlestick-patterns/start", json=payload)
            responses.append(resp)
        
        threads = []
        for _ in range(num_requests):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Assert - All requests should be processed successfully
        successful_requests = sum(1 for response in responses if response.status_code == 200)
        assert successful_requests >= 0.8 * num_requests  # At least 80% success rate
        
        for response in responses:
            if response.status_code == 200:
                assert "session_id" in response.json()
    
    def test_load_simulation(self, client):
        """Simulate moderate load on the API"""
        # This test simulates multiple users accessing the API concurrently
        import threading
        import random
        import time
        
        # Configuration
        num_users = 5  # Reduced for regular testing
        requests_per_user = 5
        
        # Tracking
        errors = []
        latencies = []
        
        # User simulation function
        def user_simulation(user_id):
            try:
                # Start a session
                start_time = time.time()
                start_payload = {
                    "difficulty": random.uniform(0.3, 0.8),
                    "total_questions": random.randint(2, 3)  # Smaller tests
                }
                
                start_response = client.post("/v1/candlestick-patterns/start", json=start_payload)
                if start_response.status_code != 200:
                    errors.append(f"Failed to start session: {start_response.status_code}")
                    return
                
                session_data = start_response.json()
                session_id = session_data["session_id"]
                
                # Track latency
                latencies.append(time.time() - start_time)
                
                # Answer questions
                current_question = {
                    "question_id": session_data["question_id"],
                    "options": session_data["options"]
                }
                
                # Just answer one question to keep test duration reasonable
                    answer_payload = {
                        "session_id": session_id,
                        "question_id": current_question["question_id"],
                    "selected_option": current_question["options"][0],
                    "response_time_ms": int(random.uniform(1000, 3000))
                    }
                    
                    start_time = time.time()
                    response = client.post("/v1/candlestick-patterns/submit_answer", json=answer_payload)
                    latencies.append(time.time() - start_time)
                    
                    if response.status_code != 200:
                        errors.append(f"Failed to submit answer: {response.status_code}")
                        return
                
                # Check session status
                start_time = time.time()
                response = client.get(f"/v1/candlestick-patterns/session/{session_id}")
                latencies.append(time.time() - start_time)
                
                if response.status_code != 200:
                    errors.append(f"Failed to get session: {response.status_code}")
                
            except Exception as e:
                errors.append(f"Exception in user {user_id}: {str(e)}")
        
        # Run simulations
        threads = []
        for i in range(num_users):
            thread = threading.Thread(target=user_simulation, args=(f"user-{i}",))
            threads.append(thread)
            thread.start()
            
            # Start users with slight delay to simulate real traffic
            time.sleep(random.uniform(0.1, 0.3))
        
        # Wait for all to complete
        for thread in threads:
            thread.join()
        
        # Print results
        print(f"\nLoad Test Results:")
        print(f"Total users: {num_users}")
        print(f"Errors: {len(errors)}")
        if latencies:
        print(f"Average latency: {sum(latencies)/len(latencies):.3f}s")
        print(f"Max latency: {max(latencies):.3f}s")
        print(f"Min latency: {min(latencies):.3f}s")
        
        # Assert acceptable performance
        error_rate = len(errors) / (num_users * 3)  # Each user makes ~3 requests
        assert error_rate < 0.2  # Less than 20% error rate for real infra testing
        if latencies:
            assert sum(latencies)/len(latencies) < 2.0  # Average latency under 2 seconds


# Test cases for security and data integrity
class TestSecurityAndDataIntegrity:
    """Tests focused on security concerns and data integrity"""
    
    def test_sql_injection_attempt(self, client):
        """Test resistance to SQL injection attempts in parameters"""
        # Arrange - Prepare payload with SQL injection attempt
        malicious_session_id = "'; DROP TABLE users; --"
        
        # Act - Try to retrieve session with malicious ID
        response = client.get(f"/v1/candlestick-patterns/session/{malicious_session_id}")
        
        # Assert - System should handle it gracefully without server error
        assert response.status_code in [400, 404]  # Either invalid format or not found
        assert "detail" in response.json()
    
    def test_xss_attempt(self, client):
        """Test handling of potential XSS attacks in input"""
        # Arrange - Start with session containing potential XSS payload
        start_payload = {
            "difficulty": 0.5,
            "total_questions": 5,
            "user_data": "<script>alert('XSS')</script>"  # Additional field with XSS payload
        }
        
        # Act
        response = client.post("/v1/candlestick-patterns/start", json=start_payload)
        
        # Assert - Either reject or sanitize the input
        if response.status_code == 200:
            # If accepted, ensure no script tags in response
            response_text = response.text
            assert "<script>" not in response_text
        else:
            # Or reject invalid input
            assert response.status_code in [400, 422]
    
    def test_large_payload(self, client):
        """Test handling extremely large request payloads"""
        # Arrange - Create payload with very large data
        large_option = "A" * 10000  # 10KB string
        start_payload = {
            "difficulty": 0.5,
            "total_questions": 5,
            "user_preferences": {
                "large_field": large_option,
                "nested": {
                    "even_more": large_option
                }
            }
        }
        
        # Act
        response = client.post("/v1/candlestick-patterns/start", json=start_payload)
        
        # Assert - System should handle it without crashing
        assert response.status_code in [200, 400, 413]  # Accept, reject as invalid, or payload too large
    
    def test_session_data_consistency(self, client):
        """Test consistency of session data across multiple API calls"""
        # Arrange - Start a session
        start_payload = {
            "difficulty": 0.5,
            "total_questions": 3
        }
        
        start_response = client.post("/v1/candlestick-patterns/start", json=start_payload)
        session_data = start_response.json()
        session_id = session_data["session_id"]
        question_id = session_data["question_id"]
        
        # Act - Get session info through different API endpoints
        # Direct session info
        session_response = client.get(f"/v1/candlestick-patterns/session/{session_id}")
        assert session_response.status_code == 200
        
        # Submit answer to get session info in response
        answer_payload = {
            "session_id": session_id,
            "question_id": question_id,
            "selected_option": session_data["options"][0],
            "response_time_ms": 1500
        }
        
        answer_response = client.post("/v1/candlestick-patterns/submit_answer", json=answer_payload)
        assert answer_response.status_code == 200
        
        # Get session info again
        session_response2 = client.get(f"/v1/candlestick-patterns/session/{session_id}")
        assert session_response2.status_code == 200
        
        # Assert - Session data should be consistent across all responses
        # Initial session should have 0 completed questions
        initial_session = session_response.json()
        assert initial_session["session_id"] == session_id
        assert initial_session["completed_questions"] == 0
        
        # After answer, should have 1 completed question
        final_session = session_response2.json()
        assert final_session["session_id"] == session_id
        assert final_session["completed_questions"] == 1
        
        # Answer response should contain consistent session ID
        answer_data = answer_response.json()
        if "session_id" in answer_data:
            assert answer_data["session_id"] == session_id
    
    def test_input_sanitization(self, client):
        """Test that inputs are properly sanitized"""
        # Arrange - Prepare payload with potentially dangerous characters
        dangerous_chars = [
            "<", ">", "&", "'", '"', "/", "\\", 
            "\u0000", "\u001F", "\u2028", "\u2029"  # Control and special Unicode chars
        ]
        
        for char in dangerous_chars:
            start_payload = {
                "difficulty": 0.5,
                "total_questions": 3,
                "user_data": f"Test{char}Data"  # Insert dangerous character
            }
            
            # Act
            response = client.post("/v1/candlestick-patterns/start", json=start_payload)
            
            # Assert - System should either reject or sanitize
            if response.status_code == 200:
                session_id = response.json()["session_id"]
                
                # Check stored data via session retrieval
                session_response = client.get(f"/v1/candlestick-patterns/session/{session_id}")
                assert session_response.status_code == 200
                
                # No server errors indicates proper handling
                # For more thorough testing, we would need to inspect the database
    
    def test_data_consistency_after_error(self, client):
        """Test that data remains consistent after an error occurs"""
        # Arrange - Start a valid session
        start_payload = {
            "difficulty": 0.5,
            "total_questions": 3
        }
        
        start_response = client.post("/v1/candlestick-patterns/start", json=start_payload)
        session_data = start_response.json()
        session_id = session_data["session_id"]
        
        # Submit a valid answer
        valid_answer = {
            "session_id": session_id,
            "question_id": session_data["question_id"],
            "selected_option": session_data["options"][0],
            "response_time_ms": 1500
        }
        
        valid_response = client.post("/v1/candlestick-patterns/submit_answer", json=valid_answer)
        assert valid_response.status_code == 200
        
        # Act - Submit an invalid answer to cause an error
        invalid_answer = {
            "session_id": session_id,
            "question_id": "invalid-question-id",
            "selected_option": "Invalid Option",
            "response_time_ms": -1
        }
        
        client.post("/v1/candlestick-patterns/submit_answer", json=invalid_answer)
        
        # Get session data after error
        session_response = client.get(f"/v1/candlestick-patterns/session/{session_id}")
        
        # Assert - Session data should remain consistent
        assert session_response.status_code == 200
        session_after_error = session_response.json()
        assert session_after_error["session_id"] == session_id
        assert session_after_error["completed_questions"] == 1  # Only the valid answer should count
    
    def test_uuid_validation(self, client):
        """Test proper validation of UUID format for session and question IDs"""
        # Test various malformed UUID strings
        malformed_uuids = [
            "not-a-uuid",
            "123e4567-e89b-12d3-a456-42661417400",  # Missing a digit
            "123e4567-e89b-12d3-a456-4266141740Z",  # Invalid character
            "{123e4567-e89b-12d3-a456-426614174000}"  # Extra braces
        ]
        
        for bad_uuid in malformed_uuids:
            # Act - Try to use in session endpoint
            response = client.get(f"/v1/candlestick-patterns/session/{bad_uuid}")
            
            # Assert - Should reject with 400 or 404
            assert response.status_code in [400, 404]
            
            # Also try in answer submission
            answer_payload = {
                "session_id": bad_uuid,
                "question_id": "q123",
                "selected_option": "Option A",
                "response_time_ms": 1500
            }
            
            response = client.post("/v1/candlestick-patterns/submit_answer", json=answer_payload)
            assert response.status_code in [400, 404, 422]


# Test cases for internationalization and accessibility
class TestInternationalizationAndAccessibility:
    """Tests for handling international characters, different languages, and accessibility requirements"""
    
    def test_unicode_handling(self, client):
        """Test proper handling of Unicode characters in inputs"""
        # Arrange - Create payloads with various Unicode character sets
        unicode_test_cases = [
            "こんにちは",  # Japanese
            "你好",  # Chinese
            "안녕하세요",  # Korean
            "Привет",  # Russian
            "مرحبا",  # Arabic
            "שלום",  # Hebrew
            "Γειά σου",  # Greek
            "नमस्ते",  # Hindi
            "🚀📊📈📉💹",  # Emoji
            "Ça va bien. Español. Português. Deutsche. Français"  # European languages with accents
        ]
        
        for unicode_text in unicode_test_cases:
            # Test with unicode in option fields
            start_payload = {
                "difficulty": 0.5,
                "total_questions": 3,
                "user_name": unicode_text
            }
            
            # Act
            response = client.post("/v1/candlestick-patterns/start", json=start_payload)
            
            # Assert - Should process normally
            assert response.status_code == 200
            session_id = response.json()["session_id"]
            
            # Get created session to check if unicode was stored correctly
            session_response = client.get(f"/v1/candlestick-patterns/session/{session_id}")
            assert session_response.status_code == 200
    
    def test_right_to_left_text(self, client):
        """Test handling of right-to-left text"""
        # Arrange - Create payloads with RTL text
        rtl_texts = [
            "مرحبا بالعالم",  # Arabic: Hello World
            "שלום עולם"  # Hebrew: Hello World
        ]
        
        for rtl_text in rtl_texts:
            start_payload = {
                "difficulty": 0.5,
                "total_questions": 3,
                "user_name": rtl_text
            }
            
            # Act
            response = client.post("/v1/candlestick-patterns/start", json=start_payload)
            
            # Assert - Should process normally
            assert response.status_code == 200
    
    def test_content_language_header(self, client):
        """Test API behavior with different Accept-Language headers"""
        # Set of language headers to test
        language_headers = [
            "en-US",
            "fr-FR",
            "zh-CN",
            "ar-SA",
            "ru-RU"
        ]
        
        for lang in language_headers:
            # Act - Make request with language header
            headers = {"Accept-Language": lang}
            response = client.post(
                "/v1/candlestick-patterns/start", 
                json={"difficulty": 0.5, "total_questions": 3},
                headers=headers
            )
            
            # Assert - Should process normally regardless of language header
            assert response.status_code == 200
    
    def test_long_text_handling(self, client):
        """Test handling of very long text that might be needed for accessibility"""
        # Arrange - Create payload with long descriptive text (useful for screen readers)
        long_description = "A" * 1000  # 1000 character description
        
        start_payload = {
            "difficulty": 0.5,
            "total_questions": 3,
            "detailed_description": long_description
        }
        
        # Act
        response = client.post("/v1/candlestick-patterns/start", json=start_payload)
        
        # Assert - Should handle long text properly
        assert response.status_code in [200, 400, 413]  # Either accept it or reject with appropriate error
    
    def test_api_versioning(self, client):
        """Test API versioning for backward compatibility"""
        # Test current API version
        current_version_response = client.post(
            "/v1/candlestick-patterns/start", 
            json={"difficulty": 0.5, "total_questions": 3}
        )
        assert current_version_response.status_code == 200
        
        # Test non-existent version
        nonexistent_version_response = client.post(
            "/v999/candlestick-patterns/start", 
            json={"difficulty": 0.5, "total_questions": 3}
        )
        assert nonexistent_version_response.status_code == 404  # Should return not found
    
    def test_character_encoding(self, client):
        """Test proper handling of different character encodings"""
        # Arrange - Special characters that might cause encoding issues
        special_chars = "©®™℠℉℃№★♠♣♥♦♪♫€£¥$¢₹₽₿"
        
        start_payload = {
            "difficulty": 0.5,
            "total_questions": 3,
            "user_data": special_chars
        }
        
        # Act
        response = client.post("/v1/candlestick-patterns/start", json=start_payload)
        
        # Assert - Should handle special characters properly
        assert response.status_code == 200
        
        # Get the session to verify data was stored correctly
        session_id = response.json()["session_id"]
        session_response = client.get(f"/v1/candlestick-patterns/session/{session_id}")
        assert session_response.status_code == 200


# Test cases for error handling and recovery mechanisms
class TestErrorHandlingAndRecovery:
    """Tests for how the system handles error conditions, recovery, and graceful degradation"""
    
    def test_redis_temporary_failure(self, client, monkeypatch):
        """Test system behavior when Redis is temporarily unavailable"""
        import pytest
        from unittest.mock import patch
        import redis.exceptions
        
        # Arrange - Start a normal session
        start_payload = {
            "difficulty": 0.5,
            "total_questions": 3
        }
        
        start_response = client.post("/v1/candlestick-patterns/start", json=start_payload)
        assert start_response.status_code == 200
        session_data = start_response.json()
        session_id = session_data["session_id"]
        
        # Mock Redis to simulate temporary failure
        # We'll patch the Redis client to raise an exception
        from backend.cache.redis_client import RedisClientImpl
        original_get = RedisClientImpl.get
        
        failure_count = [0]  # Use a list to allow modification inside the function
        
        def mock_get(self, key):
            if failure_count[0] < 1:  # Fail the first time
                failure_count[0] += 1
                raise redis.exceptions.ConnectionError("Connection refused")
            # Then succeed on subsequent calls
            return original_get(self, key)
        
        # Apply the mock
        monkeypatch.setattr(RedisClientImpl, "get", mock_get)
        
        # Act - Try to get session during "failure"
        response = client.get(f"/v1/candlestick-patterns/session/{session_id}")
        
        # Assert - Should handle error gracefully
        # Acceptable responses:
        # 1. 200 OK with recovered data (if using fallback mechanism)
        # 2. 503 Service Unavailable (if properly reporting the temporary issue)
        assert response.status_code in [200, 503]
        
        # Try again - should succeed after recovering
        response2 = client.get(f"/v1/candlestick-patterns/session/{session_id}")
        assert response2.status_code == 200
    
    def test_rate_limiting(self, client):
        """Test API's response to high frequency requests (rate limiting)"""
        # Send many requests in quick succession
        num_requests = 50
        responses = []
        
        for _ in range(num_requests):
            response = client.post(
                "/v1/candlestick-patterns/start", 
                json={"difficulty": 0.5, "total_questions": 1}
            )
            responses.append(response)
        
        # Count response codes
        success = sum(1 for r in responses if r.status_code == 200)
        rate_limited = sum(1 for r in responses if r.status_code == 429)
        
        # Assert - Either all succeed (no rate limiting) or some are rate limited
        assert success > 0  # At least some should succeed
        
        # This assertion is conditional - only relevant if rate limiting is implemented
        if rate_limited > 0:
            assert 'Retry-After' in [h for r in responses if r.status_code == 429 for h in r.headers]
    
    def test_graceful_degradation(self, client, monkeypatch):
        """Test system's ability to fall back to simpler questions if advanced features fail"""
        # Arrange - Mock the question generation to simulate partial failure
        from backend.assessments.candlestick_patterns.candlestick_db import SessionManager
        original_method = SessionManager.get_next_question
        
        def mock_get_next_question(self, session_id, difficulty=None):
            # Simulate complex algorithm failure but basic functionality works
            # Return a simplified question
            import uuid
            return {
                "question_id": str(uuid.uuid4()),
                "question_text": "Fallback basic question about candlesticks",
                "options": ["Option A", "Option B", "Option C", "Option D"],
                "correct_answer": "Option A",
                "difficulty": 0.3  # Lower difficulty as fallback
            }
        
        # Apply the mock
        monkeypatch.setattr(SessionManager, "get_next_question", mock_get_next_question)
        
        # Act - Start a session with the fallback mechanism
        start_payload = {
            "difficulty": 0.8,  # Request high difficulty
            "total_questions": 3
        }
        
        response = client.post("/v1/candlestick-patterns/start", json=start_payload)
        
        # Assert - Should still provide a valid question even if not at requested difficulty
        assert response.status_code == 200
        data = response.json()
        assert "question_id" in data
        assert "question_text" in data
        assert "options" in data
        # Note: The actual difficulty might be lower than requested
    
    def test_silent_recovery(self, client, monkeypatch):
        """Test recovery from error without user impact"""
        # Arrange - Create a session
        start_payload = {
            "difficulty": 0.5,
            "total_questions": 3
        }
        
        start_response = client.post("/v1/candlestick-patterns/start", json=start_payload)
        session_data = start_response.json()
        session_id = session_data["session_id"]
        
        # Mock to simulate an internal error that should be handled silently
        from backend.assessments.candlestick_patterns.candlestick_db import CandlestickCache
        original_update_stats = CandlestickCache.update_pattern_statistics
        
        # This function normally updates global stats but we'll make it fail
        def mock_update_stats(*args, **kwargs):
            raise Exception("Internal error updating statistics")
        
        monkeypatch.setattr(CandlestickCache, "update_pattern_statistics", mock_update_stats)
        
        # Act - Submit an answer, which would normally update stats
        answer_payload = {
            "session_id": session_id,
            "question_id": session_data["question_id"],
            "selected_option": session_data["options"][0],
            "response_time_ms": 1500
        }
        
        response = client.post("/v1/candlestick-patterns/submit_answer", json=answer_payload)
        
        # Assert - Main functionality should still work despite the internal error
        assert response.status_code == 200
        assert "is_correct" in response.json()
    
    def test_invalid_state_transition(self, client):
        """Test system handling of invalid state transitions"""
        # Arrange - Create a session
        start_payload = {
            "difficulty": 0.5,
            "total_questions": 3
        }
        
        start_response = client.post("/v1/candlestick-patterns/start", json=start_payload)
        session_data = start_response.json()
        session_id = session_data["session_id"]
        question_id = session_data["question_id"]
        
        # Act - Try to skip to a future question without answering the current one
        # This is an invalid state transition
        future_question_payload = {
            "session_id": session_id,
            "question_number": 3  # Trying to jump ahead
        }
        
        # This endpoint might not exist, so we'll check for appropriate error
        response = client.post("/v1/candlestick-patterns/set_question", json=future_question_payload)
        
        # Assert - Should either reject the invalid transition or respond with 404 if endpoint doesn't exist
        assert response.status_code in [400, 404, 405]
    
    def test_concurrent_session_modification(self, client):
        """Test system handling of concurrent modifications to the same session"""
        import threading
        import time
        
        # Arrange - Create a session
        start_payload = {
            "difficulty": 0.5,
            "total_questions": 3
        }
        
        start_response = client.post("/v1/candlestick-patterns/start", json=start_payload)
        session_data = start_response.json()
        session_id = session_data["session_id"]
        question_id = session_data["question_id"]
        
        # Prepare concurrent requests to modify the same session
        answer_payload = {
            "session_id": session_id,
            "question_id": question_id,
            "selected_option": session_data["options"][0],
            "response_time_ms": 1500
        }
        
        # Set up to make multiple concurrent requests
        num_threads = 5
        responses = [None] * num_threads
        
        def make_request(thread_idx):
            # Each thread submits the same answer
            response = client.post("/v1/candlestick-patterns/submit_answer", json=answer_payload)
            responses[thread_idx] = response
        
        # Act - Launch concurrent requests
        threads = []
        for i in range(num_threads):
            thread = threading.Thread(target=make_request, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Assert - System should handle concurrent modifications
        # Either all succeed or only one succeeds and others fail
        success_count = sum(1 for r in responses if r and r.status_code == 200)
        assert 1 <= success_count <= num_threads  # At least one should succeed
        
        # Check session state after concurrent modifications
        session_response = client.get(f"/v1/candlestick-patterns/session/{session_id}")
        assert session_response.status_code == 200
        session_data = session_response.json()
        
        # Regardless of how many succeeded, the question should only be counted once
        assert session_data["completed_questions"] == 1
    
    def test_backup_recovery(self, client, monkeypatch):
        """Test recovery from backup when primary data is lost"""
        # This test simulates recovery from a backup when primary data is lost
        
        # Arrange - Create a session and answer a question
        start_payload = {
            "difficulty": 0.5,
            "total_questions": 3
        }
        
        start_response = client.post("/v1/candlestick-patterns/start", json=start_payload)
        session_data = start_response.json()
        session_id = session_data["session_id"]
        
        # Answer a question
        answer_payload = {
            "session_id": session_id,
            "question_id": session_data["question_id"],
            "selected_option": session_data["options"][0],
            "response_time_ms": 1500
        }
        
        client.post("/v1/candlestick-patterns/submit_answer", json=answer_payload)
        
        # Now mock the Redis get to simulate lost primary data but available backup
        from backend.cache.redis_client import RedisClientImpl
        original_get = RedisClientImpl.get
        
        # We'll simulate that the primary key is missing but backup exists
        def mock_get(self, key):
            if key == f"candlestick:session:{session_id}":
                # Primary data "lost"
                return None
            elif key == f"candlestick:session:backup:{session_id}":
                # But backup exists - in real implementation, this would be the actual backup data
                # For testing, we'll use the original_get to get whatever real data exists
                return original_get(self, f"candlestick:session:{session_id}")
            return original_get(self, key)
        
        monkeypatch.setattr(RedisClientImpl, "get", mock_get)
        
        # Act - Try to get the session after "data loss"
        response = client.get(f"/v1/candlestick-patterns/session/{session_id}")
        
        # Assert - Should recover from backup
        assert response.status_code == 200
        recovered_data = response.json()
        assert recovered_data["session_id"] == session_id
        assert recovered_data["completed_questions"] >= 1  # Should have the answered question
