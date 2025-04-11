"""
Tests for the candlestick pattern assessment controller.

This module tests the refactored controller implementation to ensure
it handles requests correctly and produces expected responses.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
import json
import uuid
from unittest.mock import MagicMock, patch

from backend.assessments.base.models import (
    AssessmentType,
    QuestionDifficulty,
    SessionStatus,
    AnswerEvaluation
)
from backend.assessments.base.test_utils import create_test_app, get_test_user_id
from backend.assessments.candlestick_patterns.candlestick_controller import (
    CandlestickPatternController,
    router
)
from backend.assessments.candlestick_patterns.candlestick_models import (
    CandlestickQuestion,
    CandlestickSession
)


# Create a test app with the router
app = FastAPI()
app.include_router(router, prefix="/api/assessments/candlestick")


# Mock user ID for tests
@app.get("/user")
def get_current_user():
    return {"user_id": "test-user-123"}


# Configure the test client
client = TestClient(app)


# Use pytest fixtures to set up and tear down test data
@pytest.fixture
def mock_service():
    """Create a mock service for testing."""
    with patch("backend.assessments.candlestick_patterns.candlestick_controller.CandlestickAssessmentService") as mock:
        # Create a mock service instance
        service_instance = mock.return_value
        
        # Configure the mock service to return test data
        session_id = str(uuid.uuid4())
        question_id = str(uuid.uuid4())
        
        # Mock create_session
        mock_session = CandlestickSession(
            session_id=session_id,
            user_id="test-user-123",
            assessment_type=AssessmentType.CANDLESTICK_PATTERNS,
            target_question_count=10,
            questions=[],
            answers={},
        )
        service_instance.create_session.return_value = mock_session
        
        # Mock get_next_question
        mock_question = CandlestickQuestion(
            id=question_id,
            question_type="candlestick_pattern",
            question_text="What pattern is shown in this chart?",
            difficulty=QuestionDifficulty.INTERMEDIATE,
            topics=["candlestick", "patterns"],
            answer_options={
                "options": ["Hammer", "Shooting Star", "Doji", "Engulfing"],
                "correct_option": "Hammer"
            }
        )
        service_instance.get_next_question.return_value = mock_question
        service_instance.get_question.return_value = mock_question
        
        # Mock submit_answer
        mock_evaluation = AnswerEvaluation(
            is_correct=True,
            score=10,
            feedback="Correct! That's a Hammer pattern.",
            explanation="The Hammer pattern is characterized by a small body at the top of the range with a long lower shadow. It often indicates a potential reversal from a downtrend."
        )
        service_instance.submit_answer.return_value = mock_evaluation
        
        # Mock get_session_results
        service_instance.get_session_results.return_value = {
            "session_id": session_id,
            "status": SessionStatus.COMPLETED,
            "score": 85,
            "accuracy": 85.0,
            "questions_answered": 10,
            "correct_answers": 8,
            "completion_time_ms": 300000
        }
        
        yield service_instance


# Tests for the controller methods
def test_start_assessment(mock_service):
    """Test starting a new assessment."""
    # Override the dependency
    app.dependency_overrides = {
        "backend.common.auth.dependencies.get_current_user_id": get_test_user_id
    }
    
    # Make the request
    response = client.post(
        "/api/assessments/candlestick/start",
        params={"question_count": 5, "difficulty": "medium"}
    )
    
    # Check the response
    assert response.status_code == 200
    data = response.json()
    assert "session_id" in data
    assert data["question_count"] == 10  # Value from mock, not request
    
    # Verify service calls
    mock_service.create_session.assert_called_once()
    assert mock_service.create_session.call_args[1]["question_count"] == 5
    assert mock_service.create_session.call_args[1]["difficulty"] == "medium"


def test_get_question(mock_service):
    """Test getting a question."""
    # Override the dependency
    app.dependency_overrides = {
        "backend.common.auth.dependencies.get_current_user_id": get_test_user_id
    }
    
    # Create a session first
    session_response = client.post(
        "/api/assessments/candlestick/start",
        params={"question_count": 5, "difficulty": "medium"}
    )
    session_data = session_response.json()
    session_id = session_data["session_id"]
    
    # Mock question ID
    question_id = str(uuid.uuid4())
    
    # Make the request
    response = client.get(
        f"/api/assessments/candlestick/sessions/{session_id}/questions/{question_id}"
    )
    
    # Check the response
    assert response.status_code == 200
    data = response.json()
    assert "question_text" in data
    assert data["question_text"] == "What pattern is shown in this chart?"
    assert data["difficulty"] == "INTERMEDIATE"
    assert "options" in data
    
    # Verify service calls
    mock_service.get_question.assert_called_once()
    assert mock_service.get_question.call_args[0][0] == question_id


def test_submit_answer(mock_service):
    """Test submitting an answer."""
    # Override the dependency
    app.dependency_overrides = {
        "backend.common.auth.dependencies.get_current_user_id": get_test_user_id
    }
    
    # Create a session first
    session_response = client.post(
        "/api/assessments/candlestick/start",
        params={"question_count": 5, "difficulty": "medium"}
    )
    session_data = session_response.json()
    session_id = session_data["session_id"]
    
    # Mock question ID
    question_id = str(uuid.uuid4())
    
    # Prepare answer data
    answer_data = {
        "selected_option": "Hammer",
        "time_taken_ms": 5000
    }
    
    # Make the request
    response = client.post(
        f"/api/assessments/candlestick/sessions/{session_id}/questions/{question_id}/answer",
        json=answer_data
    )
    
    # Check the response
    assert response.status_code == 200
    data = response.json()
    assert "is_correct" in data
    assert data["is_correct"] is True
    assert "score" in data
    assert data["score"] == 10
    assert "feedback" in data
    
    # Verify service calls
    mock_service.submit_answer.assert_called_once()
    assert mock_service.submit_answer.call_args[0][0] == session_id
    assert mock_service.submit_answer.call_args[0][1] == question_id
    assert mock_service.submit_answer.call_args[0][2] == answer_data


def test_get_session_results(mock_service):
    """Test getting session results."""
    # Override the dependency
    app.dependency_overrides = {
        "backend.common.auth.dependencies.get_current_user_id": get_test_user_id
    }
    
    # Create a session first
    session_response = client.post(
        "/api/assessments/candlestick/start",
        params={"question_count": 5, "difficulty": "medium"}
    )
    session_data = session_response.json()
    session_id = session_data["session_id"]
    
    # Make the request
    response = client.get(
        f"/api/assessments/candlestick/sessions/{session_id}/results"
    )
    
    # Check the response
    assert response.status_code == 200
    data = response.json()
    assert "session_id" in data
    assert data["session_id"] == session_id
    assert "score" in data
    assert data["score"] == 85
    assert "accuracy" in data
    assert data["accuracy"] == 85.0
    
    # Verify service calls
    mock_service.get_session_results.assert_called_once()
    assert mock_service.get_session_results.call_args[0][0] == session_id


def test_get_next_question(mock_service):
    """Test getting the next question."""
    # Override the dependency
    app.dependency_overrides = {
        "backend.common.auth.dependencies.get_current_user_id": get_test_user_id
    }
    
    # Create a session first
    session_response = client.post(
        "/api/assessments/candlestick/start",
        params={"question_count": 5, "difficulty": "medium"}
    )
    session_data = session_response.json()
    session_id = session_data["session_id"]
    
    # Make the request
    response = client.get(
        f"/api/assessments/candlestick/sessions/{session_id}/next-question"
    )
    
    # Check the response
    assert response.status_code == 200
    data = response.json()
    assert "id" in data
    assert "question_text" in data
    assert data["question_text"] == "What pattern is shown in this chart?"
    assert "difficulty" in data
    assert data["difficulty"] == "INTERMEDIATE"
    
    # Verify service calls
    mock_service.get_next_question.assert_called_once()
    assert mock_service.get_next_question.call_args[0][0] == session_id


def test_complete_session(mock_service):
    """Test completing a session."""
    # Override the dependency
    app.dependency_overrides = {
        "backend.common.auth.dependencies.get_current_user_id": get_test_user_id
    }
    
    # Create a session first
    session_response = client.post(
        "/api/assessments/candlestick/start",
        params={"question_count": 5, "difficulty": "medium"}
    )
    session_data = session_response.json()
    session_id = session_data["session_id"]
    
    # Prepare the mock
    mock_session = CandlestickSession(
        session_id=session_id,
        user_id="test-user-123",
        assessment_type=AssessmentType.CANDLESTICK_PATTERNS,
        target_question_count=5,
        questions=[],
        answers={},
        completed_at=pytest.approx(pytest.freeze_time())
    )
    mock_service.complete_session.return_value = mock_session
    
    # Make the request
    response = client.post(
        f"/api/assessments/candlestick/sessions/{session_id}/complete"
    )
    
    # Check the response
    assert response.status_code == 200
    data = response.json()
    assert "session_id" in data
    assert data["session_id"] == session_id
    assert "status" in data
    assert data["status"] == "completed"
    
    # Verify service calls
    mock_service.complete_session.assert_called_once()
    assert mock_service.complete_session.call_args[0][0] == session_id 