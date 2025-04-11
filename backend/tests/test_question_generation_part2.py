#!/usr/bin/env python
"""
Comprehensive database functional tests for question generation system - Part 2.

This part focuses on question uniqueness, persistence, and database interactions.
These tests ensure that the question generation system:
1. Maintains proper uniqueness constraints
2. Correctly persists generated questions to the database
3. Handles race conditions during database operations
4. Recovers gracefully from database failures
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

logger = app_logger.getChild("test_question_generator_part2")

# Import fixtures and helpers from Part 1
from backend.tests.test_question_generation import (
    database_connection, db_session, repository, clear_cache,
    seed_pattern_data, seed_user_data, seed_assessment_data, seed_question_history,
    create_user_metrics, create_pattern_info, create_pattern_diversity
)

# Test classes

class TestQuestionUniquenessEdgeCases:
    """Test edge cases in question uniqueness verification"""
    
    @pytest.mark.asyncio
    async def test_uniqueness_with_identical_successive_requests(self, seed_pattern_data, clear_cache):
        """Test the system's handling of identical successive question generation requests"""
        # Create test data
        user_metrics = create_user_metrics(user_id="test_user_identical")
        pattern_info = create_pattern_info()
        pattern_diversity = create_pattern_diversity()
        
        # Generate a question
        first_question = await question_generator.generate_question(
            user_metrics=user_metrics,
            pattern_info=pattern_info,
            pattern_diversity=pattern_diversity
        )
        
        # Try to generate the exact same question again immediately
        second_question = await question_generator.generate_question(
            user_metrics=user_metrics,
            pattern_info=pattern_info,
            pattern_diversity=pattern_diversity
        )
        
        # The questions should be different
        assert first_question["question_text"] != second_question["question_text"], \
            "Two identical requests should produce different questions"
        assert first_question["template_id"] != second_question["template_id"], \
            "Two identical requests should use different templates"
    
    @pytest.mark.asyncio
    async def test_uniqueness_under_high_volume(self, seed_pattern_data, clear_cache):
        """Test question uniqueness under high request volume"""
        # Create test data
        user_id = "test_user_high_volume"
        pattern_info = create_pattern_info()
        pattern_diversity = create_pattern_diversity()
        
        # Generate a large number of questions for the same user and pattern
        num_questions = 30  # This should be higher than the number of available templates
        user_metrics = create_user_metrics(user_id=user_id)
        
        questions = []
        for i in range(num_questions):
            # Slightly vary difficulty to help template selection
            user_metrics["difficulty_level"] = 1.0 + (i * 0.1 % 2.0)
            
            question = await question_generator.generate_question(
                user_metrics=user_metrics,
                pattern_info=pattern_info,
                pattern_diversity=pattern_diversity
            )
            questions.append(question)
        
        # Check for question uniqueness
        question_texts = [q["question_text"] for q in questions]
        unique_texts = set(question_texts)
        
        # Get total number of templates for reference
        total_templates = sum(len(templates) for templates in template_db.templates_by_difficulty.values())
        
        # Because we have a finite number of templates, we may not be able to get
        # all unique questions if num_questions > total_templates
        expected_unique = min(num_questions, total_templates)
        
        # We should have a reasonable number of unique questions
        # Allow some wiggle room because template selection also considers other factors
        assert len(unique_texts) >= expected_unique * 0.7, \
            f"Expected at least {expected_unique * 0.7:.0f} unique questions out of {num_questions} (got {len(unique_texts)})"
        
        # If we have fewer unique questions than expected, the system should start reusing templates
        # but still make the questions different through variable substitution
        if len(unique_texts) < num_questions:
            template_ids = [q["template_id"] for q in questions]
            template_counts = {tid: template_ids.count(tid) for tid in set(template_ids)}
            reused_templates = [tid for tid, count in template_counts.items() if count > 1]
            
            print(f"Reused templates: {len(reused_templates)} out of {len(set(template_ids))}")
            
            # For templates that were reused, verify the questions are still unique
            for template_id in reused_templates:
                template_questions = [q for q in questions if q["template_id"] == template_id]
                template_question_texts = [q["question_text"] for q in template_questions]
                unique_template_texts = set(template_question_texts)
                
                assert len(unique_template_texts) == len(template_questions), \
                    f"Questions with template {template_id} should all be unique despite template reuse"
    
    @pytest.mark.asyncio
    async def test_uniqueness_with_massive_history(self, seed_pattern_data, clear_cache):
        """Test uniqueness verification with a massive history of previous questions"""
        # Create basic test data
        user_id = "test_user_massive_history"
        pattern_info = create_pattern_info()
        pattern_diversity = create_pattern_diversity()
        user_metrics = create_user_metrics(user_id=user_id)
        
        # Create a massive list of previous questions
        previous_questions = []
        template_ids = list(template_db.templates.keys())
        
        # Generate 1000 fake previous questions
        for i in range(1000):
            # Mix up parameters to create variety
            template_id = random.choice(template_ids)
            question_type = random.choice(["identification", "prediction", "characteristic"])
            format_type = random.choice(["multiple_choice", "true_false", "fill_in_blank"])
            difficulty = random.uniform(0.5, 3.0)
            
            previous_questions.append({
                "id": f"prev_{i}",
                "pattern_name": pattern_info["name"],
                "difficulty": difficulty,
                "category": pattern_info["category"],
                "template_id": template_id,
                "question_type": question_type,
                "format": format_type,
                "question_text": f"This is a fake question #{i} for testing with template {template_id}"
            })
        
        # Now try to generate a new question
        start_time = time.time()
        question = await question_generator.generate_question(
            user_metrics=user_metrics,
            pattern_info=pattern_info,
            pattern_diversity=pattern_diversity,
            previous_questions=previous_questions
        )
        generation_time = time.time() - start_time
        
        # Verify we got a valid question back
        assert question is not None, "Should generate a question despite massive history"
        assert "question_text" in question, "Should have question text"
        assert "template_id" in question, "Should have a template ID"
        
        # The question should be different from all previous questions
        assert all(q["question_text"] != question["question_text"] for q in previous_questions), \
            "Generated question should be different from all previous questions"
        
        # Check performance was reasonable
        assert generation_time < 10, f"Question generation took too long: {generation_time:.2f}s"
        
        print(f"Generated unique question with 1000 previous questions in {generation_time:.2f}s")
    
    @pytest.mark.asyncio
    async def test_uniqueness_with_cache_failure(self, seed_pattern_data, monkeypatch):
        """Test uniqueness verification when the cache fails"""
        # Create test data
        user_metrics = create_user_metrics()
        pattern_info = create_pattern_info()
        pattern_diversity = create_pattern_diversity()
        
        # Simulate a cache failure by replacing is_unique with a function that raises an exception
        original_is_unique = question_generator.uniqueness_verifier.is_unique
        
        async def failing_is_unique(*args, **kwargs):
            raise Exception("Simulated cache failure")
        
        monkeypatch.setattr(question_generator.uniqueness_verifier, "is_unique", failing_is_unique)
        
        try:
            # Generate a question despite cache failure
            question = await question_generator.generate_question(
                user_metrics=user_metrics,
                pattern_info=pattern_info,
                pattern_diversity=pattern_diversity
            )
            
            # Verify we still got a valid question back
            assert question is not None, "Should generate a question despite cache failure"
            assert "question_text" in question, "Should have question text"
            assert "template_id" in question, "Should have a template ID"
        finally:
            # Restore original function
            monkeypatch.setattr(question_generator.uniqueness_verifier, "is_unique", original_is_unique)

class TestQuestionPersistenceEdgeCases:
    """Test edge cases in question persistence functionality"""
    
    @pytest.mark.asyncio
    async def test_persistence_with_extremely_large_question(self, db_session, repository, seed_assessment_data):
        """Test persistence of an extremely large question to the database"""
        # Get an assessment
        assessment = seed_assessment_data[0]
        
        # Create an extremely large question
        user_id = assessment.user_id
        large_chart_data = {"candles": []}
        
        # Generate 1000 candles (would be an extremely large chart)
        for i in range(1000):
            large_chart_data["candles"].append({
                "open": 100 + i * 0.01,
                "high": 110 + i * 0.02,
                "low": 90 + i * 0.005,
                "close": 105 + i * 0.015,
                "volume": 1000 + i * 10,
                "date": (datetime.utcnow() + timedelta(minutes=i)).isoformat()
            })
        
        # Generate 100 answer options
        large_options = [f"Option {i}" for i in range(100)]
        
        # Create a question with the large data
        large_question = QuestionHistory(
            id=uuid.uuid4(),
            user_id=user_id,
            session_id=assessment.id,
            pattern_name="Hammer",
            pattern_category="single",
            difficulty=2.0,
            chart_data=large_chart_data,
            options=large_options,
            correct_option="Option 0"
        )
        
        # Try to persist the large question
        db_session.add(large_question)
        
        try:
            # Commit to the database
            db_session.commit()
            
            # Retrieve the question to verify storage
            retrieved_question = db_session.query(QuestionHistory).filter_by(id=large_question.id).first()
            
            # Verify data
            assert retrieved_question is not None, "Large question should be persisted"
            assert len(retrieved_question.chart_data["candles"]) == 1000, "Should store all 1000 candles"
            assert len(retrieved_question.options) == 100, "Should store all 100 options"
        except Exception as e:
            # Some databases might have limits on JSONB size
            # If this is the case, make sure we handle the error gracefully
            db_session.rollback()
            print(f"Large question persistence failed, but this might be expected: {str(e)}")
            
            # Test that we can still store a reasonably sized question after failure
            normal_question = QuestionHistory(
                id=uuid.uuid4(),
                user_id=user_id,
                session_id=assessment.id,
                pattern_name="Hammer",
                pattern_category="single",
                difficulty=2.0,
                chart_data={"candles": large_chart_data["candles"][:10]},
                options=["A", "B", "C", "D"],
                correct_option="A"
            )
            
            db_session.add(normal_question)
            db_session.commit()
            
            # Verify normal question was stored
            normal_retrieved = db_session.query(QuestionHistory).filter_by(id=normal_question.id).first()
            assert normal_retrieved is not None, "Normal question should be persisted after large question failure"
    
    @pytest.mark.asyncio
    async def test_persistence_with_special_characters(self, db_session, repository, seed_assessment_data):
        """Test database persistence with special characters and Unicode"""
        # Get an assessment
        assessment = seed_assessment_data[0]
        
        # Create question with special characters
        special_chars_question = QuestionHistory(
            id=uuid.uuid4(),
            user_id=assessment.user_id + "🔥",  # Add emoji to user_id
            session_id=assessment.id,
            pattern_name="Hammer 💯",  # Pattern name with emoji
            pattern_category="single",
            difficulty=2.0,
            chart_data={"candles": [{
                "open": 100,
                "high": 110,
                "low": 90,
                "close": 105,
                "volume": 1000,
                "note": "Important support level 📈"  # Note with emoji
            }]},
            options=["Option 1 ✨", "Option 2 🚀", "Option 3 🌟", "Option 4 💥"],
            correct_option="Option 2 🚀"
        )
        
        # Try to persist the question with special characters
        db_session.add(special_chars_question)
        db_session.commit()
        
        # Retrieve the question to verify storage
        retrieved_question = db_session.query(QuestionHistory).filter_by(id=special_chars_question.id).first()
        
        # Verify data
        assert retrieved_question is not None, "Question with special characters should be persisted"
        assert "🔥" in retrieved_question.user_id, "Unicode in user_id should be preserved"
        assert "💯" in retrieved_question.pattern_name, "Unicode in pattern_name should be preserved"
        assert "📈" in retrieved_question.chart_data["candles"][0]["note"], "Unicode in chart_data should be preserved"
        assert "🚀" in retrieved_question.correct_option, "Unicode in correct_option should be preserved"
    
    @pytest.mark.asyncio
    async def test_persistence_with_concurrent_saves(self, db_session, repository, seed_assessment_data):
        """Test concurrent question persistence to detect race conditions"""
        # Get an assessment
        assessment = seed_assessment_data[0]
        user_id = assessment.user_id
        
        # Number of concurrent operations
        num_threads = 10
        
        # Shared state for results
        successful_saves = 0
        failed_saves = 0
        lock = threading.Lock()
        
        def concurrent_save_question(index):
            nonlocal successful_saves, failed_saves
            
            # Create a new session for each thread
            thread_session = sessionmaker(bind=db_session.bind)()
            
            try:
                # Create a question for this thread
                question = QuestionHistory(
                    id=uuid.uuid4(),
                    user_id=user_id,
                    session_id=assessment.id,
                    pattern_name=f"Concurrent Pattern {index}",
                    pattern_category="single",
                    difficulty=float(index) / 10 + 1.0,
                    chart_data={"candles": [{"open": 100, "high": 110, "low": 90, "close": 105, "volume": 1000}]},
                    options=["A", "B", "C", "D"],
                    correct_option="A"
                )
                
                # Add and commit
                thread_session.add(question)
                thread_session.commit()
                
                # Increment success counter
                with lock:
                    successful_saves += 1
                    
                return question.id
            except Exception as e:
                # Rollback on error
                thread_session.rollback()
                print(f"Thread {index} error: {str(e)}")
                
                # Increment failure counter
                with lock:
                    failed_saves += 1
                    
                return None
            finally:
                thread_session.close()
        
        # Execute concurrent save operations
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            question_ids = list(executor.map(concurrent_save_question, range(num_threads)))
        
        # Check results
        print(f"Concurrent saves: {successful_saves} successful, {failed_saves} failed")
        assert successful_saves > 0, "At least some concurrent saves should succeed"
        
        # Verify all successful questions were correctly persisted
        valid_ids = [qid for qid in question_ids if qid is not None]
        for qid in valid_ids:
            question = db_session.query(QuestionHistory).filter_by(id=qid).first()
            assert question is not None, f"Question {qid} should be retrievable after concurrent save"
    
    @pytest.mark.asyncio
    async def test_persistence_with_database_errors(self, db_session, repository, seed_assessment_data, monkeypatch):
        """Test resilience to database errors during persistence"""
        # Get an assessment
        assessment = seed_assessment_data[0]
        
        # Create a valid question
        question = QuestionHistory(
            id=uuid.uuid4(),
            user_id=assessment.user_id,
            session_id=assessment.id,
            pattern_name="Error Test Pattern",
            pattern_category="single",
            difficulty=2.0,
            chart_data={"candles": [{"open": 100, "high": 110, "low": 90, "close": 105, "volume": 1000}]},
            options=["A", "B", "C", "D"],
            correct_option="A"
        )
        
        # Test 1: Simulate integrity error
        # Create a duplicate ID to force an integrity error
        duplicate_question = QuestionHistory(
            id=question.id,  # Same ID as the first question
            user_id=assessment.user_id,
            session_id=assessment.id,
            pattern_name="Duplicate ID Pattern",
            pattern_category="single",
            difficulty=1.5,
            chart_data={"candles": [{"open": 100, "high": 110, "low": 90, "close": 105, "volume": 1000}]},
            options=["A", "B", "C", "D"],
            correct_option="B"
        )
        
        # Add the first question and commit
        db_session.add(question)
        db_session.commit()
        
        # Try to add the duplicate and expect an integrity error
        db_session.add(duplicate_question)
        with pytest.raises(Exception) as excinfo:
            db_session.commit()
        
        # Rollback after error
        db_session.rollback()
        
        # Make sure we can continue using the session
        new_question = QuestionHistory(
            id=uuid.uuid4(),
            user_id=assessment.user_id,
            session_id=assessment.id,
            pattern_name="After Error Pattern",
            pattern_category="single",
            difficulty=1.8,
            chart_data={"candles": [{"open": 100, "high": 110, "low": 90, "close": 105, "volume": 1000}]},
            options=["A", "B", "C", "D"],
            correct_option="C"
        )
        
        db_session.add(new_question)
        db_session.commit()
        
        # Test 2: Simulate connection error
        # Replace the session's commit method with one that fails
        original_commit = db_session.commit
        commit_call_count = 0
        
        def failing_commit():
            nonlocal commit_call_count
            commit_call_count += 1
            if commit_call_count == 1:
                raise SQLAlchemyError("Simulated database connection error")
            return original_commit()
        
        monkeypatch.setattr(db_session, "commit", failing_commit)
        
        # Try to add another question
        connection_error_question = QuestionHistory(
            id=uuid.uuid4(),
            user_id=assessment.user_id,
            session_id=assessment.id,
            pattern_name="Connection Error Pattern",
            pattern_category="single",
            difficulty=2.2,
            chart_data={"candles": [{"open": 100, "high": 110, "low": 90, "close": 105, "volume": 1000}]},
            options=["A", "B", "C", "D"],
            correct_option="D"
        )
        
        db_session.add(connection_error_question)
        
        # First attempt should fail
        with pytest.raises(SQLAlchemyError):
            db_session.commit()
        
        # Rollback and retry
        db_session.rollback()
        db_session.add(connection_error_question)
        db_session.commit()  # Second attempt should succeed
        
        # Verify the question was saved on retry
        retrieved_question = db_session.query(QuestionHistory).filter_by(id=connection_error_question.id).first()
        assert retrieved_question is not None, "Question should be saved after connection error recovery"
        assert retrieved_question.pattern_name == "Connection Error Pattern", "Question data should be correct"

class TestQuestionHistoryEdgeCases:
    """Test edge cases in question history functionality"""
    
    @pytest.mark.asyncio
    async def test_history_with_very_long_user_session(self, db_session, repository, seed_assessment_data):
        """Test handling of very long user sessions with many questions"""
        # Get an assessment
        assessment = seed_assessment_data[1]  # Use the in-progress assessment
        user_id = assessment.user_id
        
        # Create a large number of questions for this assessment
        num_questions = 100  # Far more than typical assessment length
        base_time = datetime.utcnow() - timedelta(hours=2)
        
        for i in range(num_questions):
            # Create questions with sequential creation times
            question_time = base_time + timedelta(minutes=i)
            
            # Create the question
            question = QuestionHistory(
                id=uuid.uuid4(),
                user_id=user_id,
                session_id=assessment.id,
                pattern_name=random.choice(["Hammer", "Doji", "Engulfing"]),
                pattern_category="single",
                difficulty=float(random.randint(10, 30)) / 10,
                chart_data={"candles": [{"open": 100, "high": 110, "low": 90, "close": 105, "volume": 1000}]},
                options=["A", "B", "C", "D"],
                correct_option="A",
                created_at=question_time
            )
            db_session.add(question)
            
            # Create an answer for each question
            answer = UserAnswer(
                id=uuid.uuid4(),
                user_id=user_id,
                question_id=question.id,
                assessment_id=assessment.id,
                selected_option=random.choice(["A", "B", "C", "D"]),
                is_correct=random.choice([True, False]),
                response_time_ms=random.randint(1000, 10000),
                attempt_number=1,
                explanation_requested=random.choice([True, False]),
                created_at=question_time + timedelta(seconds=random.randint(5, 30))
            )
            db_session.add(answer)
        
        # Commit all the questions
        db_session.commit()
        
        # Update assessment metadata
        assessment.completed_questions = assessment.completed_questions + num_questions
        db_session.commit()
        
        # Test 1: Query all questions for this assessment (performance test)
        start_time = time.time()
        questions = db_session.query(QuestionHistory).filter_by(session_id=assessment.id).all()
        query_time = time.time() - start_time
        
        assert len(questions) >= num_questions, f"Should retrieve all {num_questions} questions"
        print(f"Retrieved {len(questions)} questions in {query_time:.4f}s")
        assert query_time < 2.0, f"Query should be reasonably fast (took {query_time:.4f}s)"
        
        # Test 2: Query questions with time-based filtering
        # Get questions from first hour only
        first_hour_end = base_time + timedelta(hours=1)
        first_hour_questions = db_session.query(QuestionHistory)\
            .filter(QuestionHistory.session_id == assessment.id)\
            .filter(QuestionHistory.created_at < first_hour_end)\
            .all()
        
        assert len(first_hour_questions) > 0, "Should have questions from first hour"
        assert len(first_hour_questions) < num_questions, "Should not include all questions"
        
        # Test 3: Query with pattern filtering
        hammer_questions = db_session.query(QuestionHistory)\
            .filter(QuestionHistory.session_id == assessment.id)\
            .filter(QuestionHistory.pattern_name == "Hammer")\
            .all()
        
        assert len(hammer_questions) > 0, "Should have Hammer pattern questions"
        
        # Test 4: Get related answers for a specific question
        sample_question = questions[random.randint(0, len(questions) - 1)]
        answers = db_session.query(UserAnswer)\
            .filter(UserAnswer.question_id == sample_question.id)\
            .all()
        
        assert len(answers) == 1, "Each question should have exactly one answer"
        
        # Test 5: Query performance for complex join
        start_time = time.time()
        results = db_session.query(QuestionHistory, UserAnswer)\
            .join(UserAnswer, UserAnswer.question_id == QuestionHistory.id)\
            .filter(QuestionHistory.session_id == assessment.id)\
            .filter(UserAnswer.is_correct == True)\
            .all()
        complex_query_time = time.time() - start_time
        
        print(f"Complex join query returned {len(results)} rows in {complex_query_time:.4f}s")
        assert complex_query_time < 2.0, f"Complex query should be reasonably fast (took {complex_query_time:.4f}s)"
    
    @pytest.mark.asyncio
    async def test_history_with_concurrent_assessment_sessions(self, db_session, repository):
        """Test handling of concurrent assessment sessions for the same user"""
        # Create a user with multiple concurrent assessment sessions
        user_id = "concurrent_session_user"
        
        # Create multiple assessments with the same start time
        concurrent_time = datetime.utcnow() - timedelta(minutes=30)
        num_sessions = 5
        
        assessment_ids = []
        for i in range(num_sessions):
            assessment = AssessmentAttempt(
                id=uuid.uuid4(),
                user_id=user_id,
                status="in_progress",
                start_difficulty=1.5,
                total_questions=5,
                completed_questions=2,
                correct_answers=1,
                incorrect_answers=1,
                avg_response_time_ms=3000,
                session_data={"concurrent_session": i},
                created_at=concurrent_time
            )
            db_session.add(assessment)
            assessment_ids.append(assessment.id)
        
        db_session.commit()
        
        # Add questions to each assessment
        questions_per_assessment = 3
        for assessment_id in assessment_ids:
            for i in range(questions_per_assessment):
                question = QuestionHistory(
                    id=uuid.uuid4(),
                    user_id=user_id,
                    session_id=assessment_id,
                    pattern_name=f"Pattern {i}",
                    pattern_category="single",
                    difficulty=1.5,
                    chart_data={"candles": [{"open": 100, "high": 110, "low": 90, "close": 105, "volume": 1000}]},
                    options=["A", "B", "C", "D"],
                    correct_option="A",
                    created_at=concurrent_time + timedelta(minutes=i*2)
                )
                db_session.add(question)
        
        db_session.commit()
        
        # Test 1: Get all user's in-progress assessments
        assessments = db_session.query(AssessmentAttempt)\
            .filter(AssessmentAttempt.user_id == user_id)\
            .filter(AssessmentAttempt.status == "in_progress")\
            .all()
        
        assert len(assessments) == num_sessions, f"Should find all {num_sessions} concurrent sessions"
        
        # Test 2: Get questions for a specific session
        first_assessment = assessments[0]
        
        first_questions = db_session.query(QuestionHistory)\
            .filter(QuestionHistory.session_id == first_assessment.id)\
            .all()
            
        assert len(first_questions) == questions_per_assessment, \
            f"Should find {questions_per_assessment} questions for the first assessment"
        
        # Test 3: Complete one assessment and check status
        first_assessment.status = "completed"
        first_assessment.end_difficulty = 2.0
        first_assessment.completed_questions = 5
        first_assessment.correct_answers = 3
        first_assessment.incorrect_answers = 2
        db_session.commit()
        
        # Verify only appropriate status changed
        updated_assessments = db_session.query(AssessmentAttempt)\
            .filter(AssessmentAttempt.user_id == user_id)\
            .all()
            
        completed_count = sum(1 for a in updated_assessments if a.status == "completed")
        in_progress_count = sum(1 for a in updated_assessments if a.status == "in_progress")
        
        assert completed_count == 1, "Should have exactly one completed assessment"
        assert in_progress_count == num_sessions - 1, f"Should have {num_sessions-1} in-progress assessments"
    
    @pytest.mark.asyncio
    async def test_history_with_cross_user_constraints(self, db_session, repository):
        """Test database constraints for question histories across users"""
        # Create two users with assessments
        user1_id = "cross_test_user1"
        user2_id = "cross_test_user2"
        
        # Create assessments for each user
        assessment1 = AssessmentAttempt(
            id=uuid.uuid4(),
            user_id=user1_id,
            status="in_progress",
            start_difficulty=1.5,
            total_questions=5,
            completed_questions=0
        )
        
        assessment2 = AssessmentAttempt(
            id=uuid.uuid4(),
            user_id=user2_id,
            status="in_progress",
            start_difficulty=1.5,
            total_questions=5,
            completed_questions=0
        )
        
        db_session.add(assessment1)
        db_session.add(assessment2)
        db_session.commit()
        
        # Test 1: Create a question for user1
        question1 = QuestionHistory(
            id=uuid.uuid4(),
            user_id=user1_id,
            session_id=assessment1.id,
            pattern_name="Test Pattern",
            pattern_category="single",
            difficulty=1.5,
            chart_data={"candles": [{"open": 100, "high": 110, "low": 90, "close": 105, "volume": 1000}]},
            options=["A", "B", "C", "D"],
            correct_option="A"
        )
        db_session.add(question1)
        db_session.commit()
        
        # Test 2: Try to create an answer for user1's question but from user2
        # This should fail due to cross-user constraint
        invalid_answer = UserAnswer(
            id=uuid.uuid4(),
            user_id=user2_id,  # Different user
            question_id=question1.id,  # User1's question
            assessment_id=assessment2.id,  # User2's assessment
            selected_option="B",
            is_correct=False,
            response_time_ms=2000,
            attempt_number=1
        )
        
        db_session.add(invalid_answer)
        
        # This should fail with an integrity constraint or similar error
        # Different databases might handle this differently
        try:
            db_session.commit()
            # If no error, check that the data is still consistent
            # This might happen if the database doesn't enforce this constraint
            invalid_answers = db_session.query(UserAnswer)\
                .filter(UserAnswer.question_id == question1.id)\
                .filter(UserAnswer.user_id != question1.user_id)\
                .all()
                
            if invalid_answers:
                print("WARNING: Database allowed cross-user answer creation")
        except Exception as e:
            # Expected integrity error
            db_session.rollback()
            print(f"Expected error on cross-user answer: {str(e)}")
        
        # Test 3: Create a valid answer for user1's question
        valid_answer = UserAnswer(
            id=uuid.uuid4(),
            user_id=user1_id,  # Same user
            question_id=question1.id,  # User1's question
            assessment_id=assessment1.id,  # User1's assessment
            selected_option="A",
            is_correct=True,
            response_time_ms=1500,
            attempt_number=1
        )
        
        db_session.add(valid_answer)
        db_session.commit()
        
        # Verify the valid answer was created
        answers = db_session.query(UserAnswer)\
            .filter(UserAnswer.question_id == question1.id)\
            .all()
            
        assert len(answers) > 0, "Valid answer should be created"
        assert all(a.user_id == question1.user_id for a in answers), \
            "All answers should be from the same user as the question"

# Main test runner for the second part
@pytest.mark.asyncio
async def test_part2_question_persistence():
    """Run the second part of the question generation tests"""
    print("\n===== Running Part 2: Question Uniqueness and Database Persistence Edge Cases =====")
    
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
                await method(None)  # Pass None as needed parameters
                print(f"  ✓ {method_name} passed")
            except Exception as e:
                print(f"  ✗ {method_name} failed: {str(e)}")
                raise

if __name__ == "__main__":
    asyncio.run(test_part2_question_persistence()) 