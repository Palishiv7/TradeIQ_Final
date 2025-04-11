"""
Test suite for the CandlestickRepository.

These tests validate database operations, caching, and transaction management
for candlestick pattern operations.
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import patch, MagicMock

from backend.assessments.candlestick_patterns.database_models import (
    CandlestickQuestion as CandlestickQuestionORM,
    CandlestickSession as CandlestickSessionORM,
    CandlestickAttempt as CandlestickAttemptORM
)
from backend.assessments.candlestick_patterns.candlestick_repository import (
    CandlestickQuestionRepository,
    CandlestickSessionRepository,
    CandlestickAssessmentRepositoryImpl
)
from backend.database.init_db import get_session


@pytest.fixture(scope="function")
def session():
    """Create a new database session for testing."""
    with get_session() as session:
        yield session
        session.rollback()


@pytest.fixture
def repository(session):
    """Create a repository instance for testing."""
    return CandlestickAssessmentRepositoryImpl()


@pytest.fixture
def mock_redis():
    """Mock Redis client for testing."""
    with patch("backend.common.redis.redis_client") as mock:
        mock_redis = MagicMock()
        mock.return_value = mock_redis
        yield mock_redis


@pytest.fixture
def test_question():
    """Create a sample question for testing."""
    return CandlestickQuestionORM(
        question_id="test-q-1",
        difficulty="medium",
        pattern_type="hammer",
        content={
            "text": "What pattern is this?",
            "options": ["Hammer", "Doji", "Engulfing", "Morning Star"],
            "chart_data": "base64_data_placeholder",
            "topics": ["hammer", "reversal_patterns"]
        }
    )


@pytest.fixture
def test_session():
    """Create a sample session for testing."""
    return CandlestickSessionORM(
        session_id="test-session-1",
        user_id="test-user-123",
        status="in_progress",
        questions=["test-q-1"],
        current_question_index=0,
        settings={
            "difficulty": "medium",
            "total_questions": 5
        }
    )


class TestCandlestickRepository:
    """Test suite for CandlestickRepository functionality."""

    async def test_get_question(self, repository, session, test_question):
        """Test retrieving a question."""
        # Arrange
        session.add(test_question)
        await session.commit()
        
        # Act
        result = await repository.question_repository.get_by_id("test-q-1")
        
        # Assert
        assert result is not None
        assert result.id == "test-q-1"
        assert result.difficulty.value == "medium"
        assert result.pattern == "hammer"

    async def test_get_session(self, repository, session, test_session):
        """Test retrieving a session."""
        # Arrange
        session.add(test_session)
        await session.commit()
        
        # Act
        result = await repository.session_repository.get_by_id("test-session-1")
        
        # Assert
        assert result is not None
        assert result.id == "test-session-1"
        assert result.user_id == "test-user-123"
        assert result.status == "in_progress"

    async def test_get_questions_for_session(self, repository, session, test_question):
        """Test getting questions for a new session."""
        # Arrange
        session.add(test_question)
        await session.commit()
        
        # Act
        questions = await repository.get_questions_for_session(
            difficulty="medium",
            topics=["hammer"],
            count=1
        )
        
        # Assert
        assert len(questions) == 1
        assert questions[0].pattern == "hammer"
        assert questions[0].difficulty.value == "medium"

    async def test_get_user_performance(self, repository, session, test_session):
        """Test getting user performance metrics."""
        # Arrange
        session.add(test_session)
        await session.commit()
        
        # Act
        performance = await repository.get_user_performance("test-user-123")
        
        # Assert
        assert performance is not None
        assert "total_sessions" in performance
        assert "total_questions" in performance

    async def test_get_topic_performance(self, repository, session, test_session):
        """Test getting topic-specific performance metrics."""
        # Arrange
        session.add(test_session)
        await session.commit()
        
        # Act
        performance = await repository.get_topic_performance("test-user-123", "hammer")
        
        # Assert
        assert performance is not None
        assert "accuracy" in performance
        assert "attempts" in performance

    async def test_get_difficulty_distribution(self, repository, session, test_session):
        """Test getting difficulty distribution for a user."""
        # Arrange
        session.add(test_session)
        await session.commit()
        
        # Act
        distribution = await repository.get_difficulty_distribution("test-user-123")
        
        # Assert
        assert distribution is not None
        assert isinstance(distribution, dict)
        assert all(0 <= v <= 1 for v in distribution.values()) 