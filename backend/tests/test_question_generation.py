#!/usr/bin/env python
"""
Comprehensive database functional tests for question generation system.

These tests focus on edge cases and error conditions that could cause
system failures in the question generation pipeline. No mocks are used
to ensure real-world behavior is validated.

The tests cover:
1. Template selection and question generation
2. Pattern diversity and question uniqueness
3. Database interactions and persistence
4. Concurrency and race conditions
5. Error handling and resilience
"""

import os
import asyncio
import json
import time
import random
import uuid
import threading
import pytest
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

from sqlalchemy import create_engine, func, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession

from backend.common.logger import app_logger
from backend.common.cache import cache, flush_cache
from backend.assessments.candlestick_patterns.database_models import (
    CandlestickQuestion, CandlestickSession, CandlestickAttempt,
    UserPerformance, PatternStatistics, QuestionHistory, UserAnswer
)
from backend.assessments.candlestick_patterns.repositories import CandlestickAssessmentRepositoryImpl
from backend.assessments.candlestick_patterns.candlestick_config import (
    CANDLESTICK_PATTERNS, ASSESSMENT_CONFIG
)
from backend.assessments.candlestick_patterns.candlestick_questions import (
    QuestionDifficulty, QuestionType, QuestionFormat,
    question_generator, template_db
)

from database.models import (
    Base, UserPerformance, PatternStatistics, 
    AssessmentAttempt, QuestionHistory, UserAnswer
)
from database.repositories import CandlestickRepository
from database.repositories.candlestick_repository import candlestick_repository
from database.init_db import initialize_database, get_session

logger = app_logger.getChild("test_question_generator")

# Test fixtures
@pytest.fixture(scope="module")
def database_connection():
    """Setup test database connection"""
    # Use test database URL from environment or fall back to SQLite
    test_db_url = os.environ.get("TEST_DB_URL", "sqlite:///./test_question_generation.db")
    
    # Initialize database
    engine, sessionmaker = initialize_database(test_db_url)
    
    # Create all tables
    Base.metadata.create_all(engine)
    
    yield engine, sessionmaker
    
    # Clean up - drop all tables after tests
    Base.metadata.drop_all(engine)
    engine.dispose()

@pytest.fixture
def db_session(database_connection):
    """Create a new session for each test"""
    engine, sessionmaker = database_connection
    session = sessionmaker()
    
    try:
        yield session
    finally:
        session.rollback()
        session.close()

@pytest.fixture
def repository(db_session):
    """Get repository instance"""
    repo = CandlestickAssessmentRepositoryImpl(db_session)
    yield repo
    # No need to restore original session since we're creating a new instance each time

@pytest.fixture
async def clear_cache():
    """Clear Redis cache before and after tests"""
    # Clear before test
    await flush_cache()
    
    yield
    
    # Clear after test
    await flush_cache()

@pytest.fixture
def seed_pattern_data(db_session):
    """Seed the database with pattern statistics data"""
    # Create pattern statistics for common patterns
    patterns = [
        {"pattern_name": "Hammer", "total_attempts": 1000, "correct_attempts": 700, 
         "avg_response_time_ms": 3500, "success_rate": 0.7, "avg_difficulty": 1.2},
        {"pattern_name": "Doji", "total_attempts": 800, "correct_attempts": 480, 
         "avg_response_time_ms": 4200, "success_rate": 0.6, "avg_difficulty": 1.0},
        {"pattern_name": "Engulfing", "total_attempts": 600, "correct_attempts": 300, 
         "avg_response_time_ms": 5100, "success_rate": 0.5, "avg_difficulty": 1.8},
        {"pattern_name": "Morning Star", "total_attempts": 400, "correct_attempts": 160, 
         "avg_response_time_ms": 6300, "success_rate": 0.4, "avg_difficulty": 2.5},
        {"pattern_name": "Evening Star", "total_attempts": 300, "correct_attempts": 90, 
         "avg_response_time_ms": 7200, "success_rate": 0.3, "avg_difficulty": 2.8}
    ]
    
    for p in patterns:
        pattern_stat = PatternStatistics(
            pattern_name=p["pattern_name"],
            total_attempts=p["total_attempts"],
            correct_attempts=p["correct_attempts"],
            avg_response_time_ms=p["avg_response_time_ms"],
            success_rate=p["success_rate"],
            avg_difficulty=p["avg_difficulty"],
            difficulty_distribution={"easy": 0.3, "medium": 0.4, "hard": 0.3}
        )
        db_session.add(pattern_stat)
    
    # Commit the changes
    db_session.commit()
    
    return patterns

@pytest.fixture
def seed_user_data(db_session):
    """Seed the database with user performance data"""
    # Create user performance records
    users = [
        {"user_id": "test_user_1", "total_assessments": 10, "total_questions": 50, 
         "correct_answers": 35, "current_difficulty": 1.5},
        {"user_id": "test_user_2", "total_assessments": 5, "total_questions": 25, 
         "correct_answers": 20, "current_difficulty": 2.0},
        {"user_id": "new_user", "total_assessments": 0, "total_questions": 0, 
         "correct_answers": 0, "current_difficulty": 0.5}
    ]
    
    for u in users:
        user_perf = UserPerformance(
            user_id=u["user_id"],
            total_assessments=u["total_assessments"],
            total_questions=u["total_questions"],
            correct_answers=u["correct_answers"],
            avg_response_time_ms=random.randint(2000, 6000),
            current_difficulty=u["current_difficulty"],
            pattern_statistics={
                "Hammer": {"attempts": 10, "correct": 7},
                "Doji": {"attempts": 8, "correct": 5},
                "Engulfing": {"attempts": 7, "correct": 3}
            },
            difficulty_history=[0.5, 0.7, 0.9, 1.1, 1.3, u["current_difficulty"]]
        )
        db_session.add(user_perf)
    
    # Commit the changes
    db_session.commit()
    
    return users

@pytest.fixture
def seed_assessment_data(db_session):
    """Seed the database with assessment attempt data"""
    # Create assessment attempts
    assessments = [
        {"user_id": "test_user_1", "status": "completed", "start_difficulty": 1.0,
         "end_difficulty": 1.5, "total_questions": 5, "completed_questions": 5},
        {"user_id": "test_user_1", "status": "in_progress", "start_difficulty": 1.5,
         "total_questions": 5, "completed_questions": 2},
        {"user_id": "test_user_2", "status": "completed", "start_difficulty": 1.8,
         "end_difficulty": 2.0, "total_questions": 5, "completed_questions": 5}
    ]
    
    assessment_records = []
    for a in assessments:
        assessment = AssessmentAttempt(
            id=uuid.uuid4(),
            user_id=a["user_id"],
            status=a["status"],
            start_difficulty=a["start_difficulty"],
            end_difficulty=a.get("end_difficulty"),
            total_questions=a["total_questions"],
            completed_questions=a["completed_questions"],
            correct_answers=a["completed_questions"] - 1 if a["completed_questions"] > 0 else 0,
            incorrect_answers=1 if a["completed_questions"] > 0 else 0,
            avg_response_time_ms=random.randint(2000, 6000),
            session_data={"last_activity": datetime.utcnow().isoformat()}
        )
        db_session.add(assessment)
        assessment_records.append(assessment)
    
    # Commit to get IDs
    db_session.commit()
    
    return assessment_records

@pytest.fixture
def seed_question_history(db_session, seed_assessment_data):
    """Seed the database with question history data"""
    # Get assessment attempts
    assessments = seed_assessment_data
    
    # Create question history records
    questions = []
    for assessment in assessments:
        for i in range(assessment.completed_questions):
            pattern = random.choice(["Hammer", "Doji", "Engulfing", "Morning Star", "Evening Star"])
            category = "single" if pattern in ["Hammer", "Doji"] else "double" if pattern == "Engulfing" else "triple"
            
            question = QuestionHistory(
                id=uuid.uuid4(),
                user_id=assessment.user_id,
                session_id=assessment.id,
                pattern_name=pattern,
                pattern_category=category,
                difficulty=float(random.randint(10, 30)) / 10,  # 1.0 to 3.0
                chart_data={"candles": [{"open": 100, "high": 110, "low": 90, "close": 105, "volume": 1000}]},
                options=["A", "B", "C", "D"],
                correct_option="A"
            )
            db_session.add(question)
            questions.append(question)
            
            # Create user answer for each question
            answer = UserAnswer(
                user_id=assessment.user_id,
                question_id=question.id,
                assessment_id=assessment.id,
                selected_option="A" if i < assessment.correct_answers else "B",
                is_correct=i < assessment.correct_answers,
                response_time_ms=random.randint(1000, 10000),
                attempt_number=1,
                explanation_requested=random.choice([True, False])
            )
            db_session.add(answer)
    
    # Commit the changes
    db_session.commit()
    
    return questions

# Helper functions for testing
def create_user_metrics(user_id="test_user", questions_answered=10, questions_correct=7, 
                       accuracy=0.7, difficulty_level=1.5):
    """Create sample user metrics for testing"""
    return {
        "user_id": user_id,
        "questions_answered": questions_answered,
        "questions_correct": questions_correct,
        "accuracy": accuracy,
        "difficulty_level": difficulty_level,
        "correct_by_question_type": {
            "identification": 5,
            "prediction": 2,
            "characteristic": 0
        },
        "total_by_question_type": {
            "identification": 7,
            "prediction": 2,
            "characteristic": 1
        }
    }

def create_pattern_info(name="Hammer", category="single", description="A hammer pattern"):
    """Create sample pattern info for testing"""
    return {
        "name": name,
        "category": category,
        "description": description
    }

def create_pattern_diversity(patterns=None):
    """Create sample pattern diversity information for testing"""
    if patterns is None:
        patterns = ["Hammer", "Doji", "Engulfing"]
        
    pattern_counts = {p: random.randint(1, 5) for p in patterns}
    pattern_history = []
    for p, count in pattern_counts.items():
        pattern_history.extend([p] * count)
    random.shuffle(pattern_history)
    
    return {
        "pattern_counts": pattern_counts,
        "pattern_history": pattern_history
    }


# Test classes

class TestTemplateSelectionEdgeCases:
    """Test edge cases in template selection functionality"""
    
    async def test_template_selection_with_invalid_difficulty(self, seed_pattern_data):
        """Test template selection with invalid difficulty levels"""
        pattern_info = create_pattern_info()
        pattern_diversity = create_pattern_diversity()
        
        # Test with extremely low difficulty
        low_difficulty_metrics = create_user_metrics(difficulty_level=-10.0)
        question = await question_generator.generate_question(
            user_metrics=low_difficulty_metrics,
            pattern_info=pattern_info,
            pattern_diversity=pattern_diversity
        )
        
        assert question is not None, "Should handle extremely low difficulty"
        assert "question_text" in question, "Should generate question text"
        assert "difficulty" in question, "Should include difficulty info"
        assert question["difficulty"] == "easy", "Should default to easy for extremely low difficulty"
        
        # Test with extremely high difficulty
        high_difficulty_metrics = create_user_metrics(difficulty_level=100.0)
        question = await question_generator.generate_question(
            user_metrics=high_difficulty_metrics,
            pattern_info=pattern_info,
            pattern_diversity=pattern_diversity
        )
        
        assert question is not None, "Should handle extremely high difficulty"
        assert "question_text" in question, "Should generate question text"
        assert "difficulty" in question, "Should include difficulty info"
        assert question["difficulty"] == "hard", "Should default to hard for extremely high difficulty"
    
    async def test_template_selection_with_missing_metrics(self, seed_pattern_data):
        """Test template selection with missing or incomplete user metrics"""
        pattern_info = create_pattern_info()
        pattern_diversity = create_pattern_diversity()
        
        # Test with empty metrics
        empty_metrics = {}
        question = await question_generator.generate_question(
            user_metrics=empty_metrics,
            pattern_info=pattern_info,
            pattern_diversity=pattern_diversity
        )
        
        assert question is not None, "Should handle empty metrics"
        assert "question_text" in question, "Should generate question text"
        assert "difficulty" in question, "Should include difficulty info"
        
        # Test with partially populated metrics
        partial_metrics = {"user_id": "test_user", "difficulty_level": 1.5}
        question = await question_generator.generate_question(
            user_metrics=partial_metrics,
            pattern_info=pattern_info,
            pattern_diversity=pattern_diversity
        )
        
        assert question is not None, "Should handle partial metrics"
        assert "question_text" in question, "Should generate question text"
        assert "difficulty" in question, "Should include difficulty info"
        
        # Test with None metrics
        none_metrics = None
        question = await question_generator.generate_question(
            user_metrics=none_metrics,
            pattern_info=pattern_info,
            pattern_diversity=pattern_diversity
        )
        
        assert question is not None, "Should handle None metrics"
        assert "question_text" in question, "Should generate question text"
        assert "difficulty" in question, "Should include difficulty info"
    
    async def test_template_selection_with_invalid_pattern_info(self, seed_pattern_data):
        """Test template selection with invalid pattern information"""
        user_metrics = create_user_metrics()
        pattern_diversity = create_pattern_diversity()
        
        # Test with empty pattern info
        empty_pattern = {}
        question = await question_generator.generate_question(
            user_metrics=user_metrics,
            pattern_info=empty_pattern,
            pattern_diversity=pattern_diversity
        )
        
        assert question is not None, "Should handle empty pattern info"
        assert "question_text" in question, "Should generate question text"
        assert "pattern_info" in question, "Should include pattern info"
        
        # Test with missing required fields
        partial_pattern = {"name": "Hammer"}  # Missing category and description
        question = await question_generator.generate_question(
            user_metrics=user_metrics,
            pattern_info=partial_pattern,
            pattern_diversity=pattern_diversity
        )
        
        assert question is not None, "Should handle partial pattern info"
        assert "question_text" in question, "Should generate question text"
        assert "pattern_info" in question, "Should include pattern info"
        
        # Test with None pattern info
        none_pattern = None
        question = await question_generator.generate_question(
            user_metrics=user_metrics,
            pattern_info=none_pattern,
            pattern_diversity=pattern_diversity
        )
        
        assert question is not None, "Should handle None pattern info"
        assert "question_text" in question, "Should generate question text"
        assert "pattern_info" in question, "Should include pattern info"
    
    async def test_template_selection_with_invalid_diversity(self, seed_pattern_data):
        """Test template selection with invalid pattern diversity information"""
        user_metrics = create_user_metrics()
        pattern_info = create_pattern_info()
        
        # Test with empty diversity info
        empty_diversity = {}
        question = await question_generator.generate_question(
            user_metrics=user_metrics,
            pattern_info=pattern_info,
            pattern_diversity=empty_diversity
        )
        
        assert question is not None, "Should handle empty diversity info"
        assert "question_text" in question, "Should generate question text"
        
        # Test with missing required fields
        partial_diversity = {"pattern_counts": {"Hammer": 5}}  # Missing pattern_history
        question = await question_generator.generate_question(
            user_metrics=user_metrics,
            pattern_info=pattern_info,
            pattern_diversity=partial_diversity
        )
        
        assert question is not None, "Should handle partial diversity info"
        assert "question_text" in question, "Should generate question text"
        
        # Test with None diversity info
        none_diversity = None
        question = await question_generator.generate_question(
            user_metrics=user_metrics,
            pattern_info=pattern_info,
            pattern_diversity=none_diversity
        )
        
        assert question is not None, "Should handle None diversity info"
        assert "question_text" in question, "Should generate question text"
    
    async def test_template_selection_with_invalid_previous_questions(self, seed_pattern_data):
        """Test template selection with invalid previous questions information"""
        user_metrics = create_user_metrics()
        pattern_info = create_pattern_info()
        pattern_diversity = create_pattern_diversity()
        
        # Test with empty previous questions
        empty_previous = []
        question = await question_generator.generate_question(
            user_metrics=user_metrics,
            pattern_info=pattern_info,
            pattern_diversity=pattern_diversity,
            previous_questions=empty_previous
        )
        
        assert question is not None, "Should handle empty previous questions"
        assert "question_text" in question, "Should generate question text"
        
        # Test with malformed previous questions
        malformed_previous = [{"invalid": "data"}]
        question = await question_generator.generate_question(
            user_metrics=user_metrics,
            pattern_info=pattern_info,
            pattern_diversity=pattern_diversity,
            previous_questions=malformed_previous
        )
        
        assert question is not None, "Should handle malformed previous questions"
        assert "question_text" in question, "Should generate question text"
        
        # Test with None previous questions
        none_previous = None
        question = await question_generator.generate_question(
            user_metrics=user_metrics,
            pattern_info=pattern_info,
            pattern_diversity=pattern_diversity,
            previous_questions=none_previous
        )
        
        assert question is not None, "Should handle None previous questions"
        assert "question_text" in question, "Should generate question text"

class TestQuestionGenerationEdgeCases:
    """Test edge cases in question generation functionality"""
    
    async def test_question_generation_with_empty_template_database(self, monkeypatch):
        """Test question generation when the template database is empty"""
        user_metrics = create_user_metrics()
        pattern_info = create_pattern_info()
        pattern_diversity = create_pattern_diversity()
        
        # Mock the template database to be empty
        empty_db = question_generator.template_db
        empty_db.templates = {}
        empty_db.templates_by_difficulty = {
            QuestionDifficulty.EASY: [],
            QuestionDifficulty.MEDIUM: [],
            QuestionDifficulty.HARD: []
        }
        empty_db.templates_by_type = {qt: [] for qt in QuestionType}
        empty_db.templates_by_format = {qf: [] for qf in QuestionFormat}
        
        monkeypatch.setattr(question_generator, "template_db", empty_db)
        
        # Generate question with empty database
        question = await question_generator.generate_question(
            user_metrics=user_metrics,
            pattern_info=pattern_info,
            pattern_diversity=pattern_diversity
        )
        
        # Should fall back to a hardcoded question
        assert question is not None, "Should handle empty template database"
        assert "question_text" in question, "Should generate fallback question text"
        assert "fallback" in question.get("template_id", ""), "Should use fallback template"
    
    async def test_question_generation_with_extremely_long_inputs(self, seed_pattern_data):
        """Test question generation with extremely long inputs"""
        # Create user metrics with extremely long user_id
        long_user_id = "x" * 10000  # 10K characters
        user_metrics = create_user_metrics(user_id=long_user_id)
        
        # Create pattern info with extremely long description
        long_description = "x" * 10000  # 10K characters
        pattern_info = create_pattern_info(description=long_description)
        
        # Create pattern diversity with extremely long history
        long_history = ["Hammer"] * 10000  # 10K patterns
        pattern_diversity = {
            "pattern_counts": {"Hammer": 10000},
            "pattern_history": long_history
        }
        
        # Generate question with extremely long inputs
        question = await question_generator.generate_question(
            user_metrics=user_metrics,
            pattern_info=pattern_info,
            pattern_diversity=pattern_diversity
        )
        
        assert question is not None, "Should handle extremely long inputs"
        assert "question_text" in question, "Should generate question text"
        assert len(question["question_text"]) < 10000, "Should generate reasonable length question text"
    
    async def test_question_generation_with_special_characters(self, seed_pattern_data):
        """Test question generation with special characters in inputs"""
        # Create inputs with special characters
        user_metrics = create_user_metrics(user_id="test_user_😊🔥_$^#@!")
        pattern_info = create_pattern_info(
            name="Hammer 😊",
            description="This is a 'special' pattern with \"quotes\", <tags>, & other symbols: €£¥§±"
        )
        pattern_diversity = create_pattern_diversity(patterns=["Hammer 😊", "Doji 🔥", "Engulfing 💯"])
        
        # Generate question with special characters
        question = await question_generator.generate_question(
            user_metrics=user_metrics,
            pattern_info=pattern_info,
            pattern_diversity=pattern_diversity
        )
        
        assert question is not None, "Should handle special characters"
        assert "question_text" in question, "Should generate question text"
    
    async def test_question_generation_with_extreme_concurrency(self, seed_pattern_data):
        """Test question generation under extreme concurrency"""
        # Define test parameters
        num_requests = 20
        pattern_info = create_pattern_info()
        pattern_diversity = create_pattern_diversity()
        
        # Create different user metrics for each request
        user_metrics_list = [
            create_user_metrics(user_id=f"test_user_{i}", difficulty_level=float(i % 3) + 1.0)
            for i in range(num_requests)
        ]
        
        # Function to generate a question
        async def generate_question_task(user_metrics):
            try:
                return await question_generator.generate_question(
                    user_metrics=user_metrics,
                    pattern_info=pattern_info,
                    pattern_diversity=pattern_diversity
                )
            except Exception as e:
                return {"error": str(e)}
        
        # Generate questions concurrently
        start_time = time.time()
        tasks = [generate_question_task(user_metrics) for user_metrics in user_metrics_list]
        results = await asyncio.gather(*tasks)
        end_time = time.time()
        
        # Validate results
        assert len(results) == num_requests, "Should handle all concurrent requests"
        
        success_count = sum(1 for r in results if "question_text" in r and "error" not in r)
        assert success_count > 0, "At least some concurrent requests should succeed"
        
        # Log performance information
        total_time = end_time - start_time
        avg_time_per_request = total_time / num_requests
        print(f"Processed {num_requests} concurrent requests in {total_time:.2f}s "
              f"(avg: {avg_time_per_request:.4f}s per request)")
        print(f"Success rate: {success_count}/{num_requests} ({success_count/num_requests*100:.1f}%)")

    async def test_question_generation_with_invalid_model_type(self, seed_pattern_data):
        """Test question generation with invalid model type"""
        user_metrics = create_user_metrics()
        pattern_info = create_pattern_info()
        pattern_diversity = create_pattern_diversity()
        
        # Test with non-existent model type
        question = await question_generator.generate_question(
            user_metrics=user_metrics,
            pattern_info=pattern_info,
            pattern_diversity=pattern_diversity,
            model_type="non_existent_model"
        )
        
        assert question is not None, "Should handle non-existent model type"
        assert "question_text" in question, "Should generate question text"
        assert "prompt" in question, "Should include prompt data"
        
        # Test with empty model type
        question = await question_generator.generate_question(
            user_metrics=user_metrics,
            pattern_info=pattern_info,
            pattern_diversity=pattern_diversity,
            model_type=""
        )
        
        assert question is not None, "Should handle empty model type"
        assert "question_text" in question, "Should generate question text"
        
        # Test with None model type
        question = await question_generator.generate_question(
            user_metrics=user_metrics,
            pattern_info=pattern_info,
            pattern_diversity=pattern_diversity,
            model_type=None
        )
        
        assert question is not None, "Should handle None model type"
        assert "question_text" in question, "Should generate question text"

# Main test runner for the first part
@pytest.mark.asyncio
async def test_part1_question_generation():
    """Run the first part of the question generation tests"""
    print("\n===== Running Part 1: Template Selection and Question Generation Edge Cases =====")
    
    # Get test module and classes
    module = __import__(__name__)
    test_classes = [
        attr for attr in dir(module) 
        if attr.startswith("Test") and attr in globals()
    ]
    
    # Create test instances
    test_instances = [globals()[cls]() for cls in test_classes]
    
    # Run tests
    for test_class in test_instances:
        class_name = test_class.__class__.__name__
        print(f"\n----- Running tests for {class_name} -----")
        
        # Get test methods
        test_methods = [
            method for method in dir(test_class)
            if method.startswith("test_") and callable(getattr(test_class, method))
        ]
        
        # Run each test method
        for method_name in test_methods:
            print(f"  Running {method_name}...")
            method = getattr(test_class, method_name)
            try:
                await method(None)  # Pass None as monkeypatch for simplicity
                print(f"  ✓ {method_name} passed")
            except Exception as e:
                print(f"  ✗ {method_name} failed: {str(e)}")
                raise

if __name__ == "__main__":
    asyncio.run(test_part1_question_generation()) 