"""
Tests for the Candlestick API module.

This module tests the API endpoints and functions in the candlestick_api.py module,
focusing specifically on the critical flow for the start_assessment function.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi import HTTPException
import time

@pytest.mark.asyncio
async def test_start_assessment_calls_rate_limiter_check():
    """Test that start_assessment correctly calls rate_limiter.check."""
    # Import here to avoid module-level import issues
    from backend.assessments.candlestick_patterns.candlestick_api import start_assessment
    from backend.assessments.candlestick_patterns.candlestick_api import ASSESSMENT_CONFIG
    from fastapi import Request
    from pydantic import BaseModel
    
    class StartAssessmentRequest(BaseModel):
        difficulty: float = 0.5
        total_questions: int = 5
    
    # Create request object
    request_obj = StartAssessmentRequest(difficulty=0.5, total_questions=5)
    
    # Create mock user_id
    user_id = "test-user-1"
    
    # Create mock request
    req = MagicMock(spec=Request)
    
    # Create mock rate_limiter with check method
    rate_limiter_mock = AsyncMock()
    rate_limiter_mock.check.return_value = (True, None)
    
    # Create mock session_manager
    session_manager_mock = AsyncMock()
    
    # Mock create_session to return a valid session
    async def mock_create_session(user_id, total_questions=None):
        if not total_questions:
            total_questions = 10
            
        return {
            "session_id": "test-session-123",
            "user_id": user_id,
            "started_at": int(time.time()),
            "total_questions": total_questions,
            "questions_asked": 0,
            "correct_answers": 0,
            "current_streak": 0,
            "max_streak": 0,
            "total_score": 0,
            "current_question": None,
            "previous_patterns": set(),
            "question_history": []
        }
    
    session_manager_mock.create_session.side_effect = mock_create_session
    
    # Mock for CandlestickAssessment.generate_question
    mock_question = {
        "question_id": "test-q-123",
        "question_number": 1,
        "total_questions": 5,
        "question_text": "What pattern is this?",
        "options": ["Hammer", "Doji", "Engulfing", "Morning Star"],
        "image_data": "base64_data",
        "time_limit_seconds": 30,
        "difficulty": 0.5
    }
    
    assessment_mock = AsyncMock()
    assessment_mock.initialize.return_value = True
    assessment_mock.generate_question.return_value = mock_question
    
    with patch("backend.assessments.candlestick_patterns.candlestick_api.rate_limiter", rate_limiter_mock), \
         patch("backend.assessments.candlestick_patterns.candlestick_api.session_manager", session_manager_mock), \
         patch("backend.assessments.candlestick_patterns.candlestick_api.CandlestickAssessment", return_value=assessment_mock):
        
        # Test the function
        result = await start_assessment(request_obj, user_id, req)
        
        # Verify rate_limiter.check was called with correct parameters
        rate_limiter_mock.check.assert_called_once_with(
            f"candlestick:start:{user_id}",
            ASSESSMENT_CONFIG["rate_limits"]["start_assessment"],
            60
        )
        
        # Verify session was initialized
        assessment_mock.initialize.assert_called_once()
        
        # Verify question was generated
        assessment_mock.generate_question.assert_called_once_with(difficulty=0.5)

@pytest.mark.asyncio
async def test_start_assessment_rate_limited():
    """Test that start_assessment raises 429 when rate limited."""
    # Import here to avoid module-level import issues
    from backend.assessments.candlestick_patterns.candlestick_api import start_assessment
    from fastapi import Request
    from pydantic import BaseModel
    
    class StartAssessmentRequest(BaseModel):
        difficulty: float = 0.5
        total_questions: int = 5
    
    # Create request object
    request_obj = StartAssessmentRequest(difficulty=0.5, total_questions=5)
    
    # Create mock user_id
    user_id = "test-user-1"
    
    # Create mock request
    req = MagicMock(spec=Request)
    
    # Create mock rate_limiter with check method that returns False (rate limited)
    rate_limiter_mock = AsyncMock()
    rate_limiter_mock.check.return_value = (False, 30)  # Not allowed, reset in 30 seconds
    
    with patch("backend.assessments.candlestick_patterns.candlestick_api.rate_limiter", rate_limiter_mock):
        # Test the function should raise HTTPException
        with pytest.raises(HTTPException) as excinfo:
            await start_assessment(request_obj, user_id, req)
        
        # Verify the status code is 429
        assert excinfo.value.status_code == 429
        assert "Rate limit exceeded" in excinfo.value.detail 