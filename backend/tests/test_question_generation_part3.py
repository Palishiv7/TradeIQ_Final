#!/usr/bin/env python
"""
Comprehensive database functional tests for question generation system - Part 3.

This part focuses on error handling and recovery mechanisms.
These tests ensure that the question generation system:
1. Gracefully handles database connectivity issues
2. Recovers from cache failures
3. Maintains data integrity during failures
4. Properly handles timeout and performance degradation
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
from sqlalchemy.exc import IntegrityError, SQLAlchemyError, OperationalError
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

logger = app_logger.getChild("test_question_generator_part3")

# Import fixtures and helpers from Part 1
from backend.tests.test_question_generation import (
    database_connection, db_session, repository, clear_cache,
    seed_pattern_data, seed_user_data, seed_assessment_data, seed_question_history,
    create_user_metrics, create_pattern_info, create_pattern_diversity
)

class TestDatabaseFailureHandling:
    """Test error handling with database failures"""
    
    @pytest.mark.asyncio
    async def test_repository_with_connection_failure(self, db_session, repository, monkeypatch):
        """Test repository functions when database connection fails"""
        # Create a function that simulates a database connection failure
        original_execute = db_session.execute
        
        def failing_execute(*args, **kwargs):
            raise OperationalError("Connection refused", None, None)
        
        # Patch the session's execute method to simulate connection failure
        monkeypatch.setattr(db_session, "execute", failing_execute)
        
        try:
            # Test the get_pattern_statistics method
            with pytest.raises(Exception) as excinfo:
                stats = await repository.get_pattern_statistics("hammer")
                
            # Ensure the error is properly propagated
            assert "Connection refused" in str(excinfo.value) or "database error" in str(excinfo.value).lower(), \
                "Database connection error should be properly propagated"
                
            # Test get_user_performance method
            with pytest.raises(Exception) as excinfo:
                user_perf = await repository.get_user_performance("test_user", "hammer")
                
            # Ensure the error is properly propagated
            assert "Connection refused" in str(excinfo.value) or "database error" in str(excinfo.value).lower(), \
                "Database connection error should be properly propagated"
                
        finally:
            # Restore the original execute method
            monkeypatch.setattr(db_session, "execute", original_execute)
    
    @pytest.mark.asyncio
    async def test_generator_with_database_failure(self, db_session, repository, seed_pattern_data, monkeypatch):
        """Test question generator when database operations fail"""
        # Create test data
        user_metrics = create_user_metrics()
        pattern_info = create_pattern_info()
        pattern_diversity = create_pattern_diversity()
        
        # Create a function that simulates a database query failure
        original_get_pattern_stats = repository.get_pattern_statistics
        
        async def failing_get_pattern_stats(*args, **kwargs):
            raise SQLAlchemyError("Database query failed")
        
        # Patch the repository method to simulate database failure
        monkeypatch.setattr(repository, "get_pattern_statistics", failing_get_pattern_stats)
        
        try:
            # Generate a question despite database failure
            # Since we're patching get_pattern_statistics, this should trigger fallback behavior
            question = await question_generator.generate_question(
                user_metrics=user_metrics,
                pattern_info=pattern_info,
                pattern_diversity=pattern_diversity
            )
            
            # Verify we still got a valid question back
            # The generator should fallback to using pattern_info directly
            assert question is not None, "Should generate a question despite database failure"
            assert "question_text" in question, "Should have question text"
            assert "template_id" in question, "Should have a template ID"
            
        finally:
            # Restore the original method
            monkeypatch.setattr(repository, "get_pattern_statistics", original_get_pattern_stats)
    
    @pytest.mark.asyncio
    async def test_retry_mechanism(self, db_session, repository, seed_pattern_data, monkeypatch):
        """Test retry mechanism for transient database errors"""
        # Create test data
        user_metrics = create_user_metrics()
        pattern_info = create_pattern_info()
        
        # Create a function that fails on first call but succeeds on subsequent calls
        original_save_question = repository.save_question_history
        call_count = 0
        
        async def intermittent_save_question(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise OperationalError("Connection lost", None, None)
            return await original_save_question(*args, **kwargs)
        
        # Patch the repository method
        monkeypatch.setattr(repository, "save_question_history", intermittent_save_question)
        
        try:
            # Get a fresh session_id
            assessment = (await seed_assessment_data)[0]
            
            # Call the save_question_history directly through question_generator
            # This would typically be called during generate_question
            question_data = {
                "id": str(uuid.uuid4()),
                "user_id": user_metrics["user_id"],
                "session_id": str(assessment.id),
                "pattern_name": pattern_info["name"],
                "pattern_category": pattern_info["category"],
                "difficulty": user_metrics["difficulty_level"],
                "chart_data": {"candles": [{"open": 100, "high": 110, "low": 90, "close": 105}]},
                "options": ["A", "B", "C", "D"],
                "correct_option": "A",
                "question_text": "Test question with retry",
                "template_id": "template_1"
            }
            
            # The method should retry after the first failure
            saved_question = await repository.save_question_history(question_data)
            
            assert call_count > 1, "Method should have been retried"
            assert saved_question is not None, "Question should be saved after retry"
            
            # Verify it was saved to the database
            retrieved = await repository.get_question_history(question_data["id"])
            assert retrieved is not None, "Question should be retrievable after retry save"
            
        finally:
            # Restore the original method
            monkeypatch.setattr(repository, "save_question_history", original_save_question)
            
    @pytest.mark.asyncio
    async def test_transaction_isolation(self, db_session, repository):
        """Test transaction isolation during failures"""
        # Create two questions with the same ID to force a conflict
        question_id = uuid.uuid4()
        assessment = (await seed_assessment_data)[0]
        
        # First question (will be saved)
        question1 = QuestionHistory(
            id=question_id,
            user_id=assessment.user_id,
            session_id=assessment.id,
            pattern_name="Isolation Test 1",
            pattern_category="single",
            difficulty=1.5,
            chart_data={"candles": [{"open": 100, "high": 110, "low": 90, "close": 105}]},
            options=["A", "B", "C", "D"],
            correct_option="A"
        )
        
        # Second question with same ID (should fail)
        question2 = QuestionHistory(
            id=question_id,  # Same ID
            user_id=assessment.user_id,
            session_id=assessment.id,
            pattern_name="Isolation Test 2",
            pattern_category="single",
            difficulty=2.0,
            chart_data={"candles": [{"open": 200, "high": 210, "low": 190, "close": 205}]},
            options=["W", "X", "Y", "Z"],
            correct_option="Z"
        )
        
        # Create a valid question with different ID (should not be affected by rollback)
        valid_question = QuestionHistory(
            id=uuid.uuid4(),
            user_id=assessment.user_id,
            session_id=assessment.id,
            pattern_name="Isolation Valid",
            pattern_category="single",
            difficulty=1.8,
            chart_data={"candles": [{"open": 150, "high": 160, "low": 140, "close": 155}]},
            options=["A", "B", "C", "D"],
            correct_option="B"
        )
        
        # Save the first question
        db_session.add(question1)
        db_session.commit()
        
        # Try to save the valid question and the conflicting question in the same transaction
        db_session.add(valid_question)
        db_session.add(question2)  # This should cause a conflict
        
        # This commit should fail
        with pytest.raises(Exception) as excinfo:
            db_session.commit()
        
        # Rollback after the error
        db_session.rollback()
        
        # Now try to save just the valid question
        db_session.add(valid_question)
        db_session.commit()
        
        # Verify that we can still retrieve the first question
        first_question = db_session.query(QuestionHistory).filter_by(id=question_id).first()
        assert first_question is not None, "Original question should still exist"
        assert first_question.pattern_name == "Isolation Test 1", "Original question should be unchanged"
        
        # Verify that the valid question was saved after the rollback
        saved_valid = db_session.query(QuestionHistory).filter_by(id=valid_question.id).first()
        assert saved_valid is not None, "Valid question should be saved after rollback"
        assert saved_valid.pattern_name == "Isolation Valid", "Valid question should have correct data"

class TestCacheFailureHandling:
    """Test error handling with cache failures"""
    
    @pytest.mark.asyncio
    async def test_cache_service_unavailable(self, seed_pattern_data, monkeypatch):
        """Test behavior when the cache service is unavailable"""
        # Create test data
        user_metrics = create_user_metrics()
        pattern_info = create_pattern_info()
        pattern_diversity = create_pattern_diversity()
        
        # Patch the cache get function to simulate service unavailability
        original_get = cache.get
        
        async def failing_get(*args, **kwargs):
            raise Exception("Redis connection error")
        
        # Apply the patch
        monkeypatch.setattr(cache, "get", failing_get)
        
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
            # Restore the original method
            monkeypatch.setattr(cache, "get", original_get)
    
    @pytest.mark.asyncio
    async def test_cache_coherence_after_failure(self, db_session, repository, seed_pattern_data, monkeypatch):
        """Test cache coherence after cache failures"""
        # Create test data
        pattern_name = "hammer"
        
        # Get initial pattern statistics from database
        initial_stats = await repository.get_pattern_statistics(pattern_name)
        
        # Update statistics in the database directly
        if initial_stats:
            # Increment the usage count
            initial_stats.usage_count += 10
            initial_stats.correct_identifications += 5
            initial_stats.incorrect_identifications += 5
            db_session.commit()
        else:
            # Create new statistics if none exist
            new_stats = PatternStatistics(
                pattern_id=pattern_name,
                usage_count=10,
                correct_identifications=5,
                incorrect_identifications=5
            )
            db_session.add(new_stats)
            db_session.commit()
        
        # Make sure cache is clear
        await flush_cache()
        
        # The first get should cache the data from the database
        db_stats = await repository.get_pattern_statistics(pattern_name)
        
        # Patch cache get to simulate failure
        original_get = cache.get
        
        async def failing_get(*args, **kwargs):
            raise Exception("Cache read failure")
        
        # Apply the patch
        monkeypatch.setattr(cache, "get", failing_get)
        
        try:
            # This should read from database due to cache failure
            fallback_stats = await repository.get_pattern_statistics(pattern_name)
            
            # Verify data consistency despite cache failure
            assert fallback_stats is not None, "Should get stats despite cache failure"
            assert fallback_stats.usage_count == db_stats.usage_count, "Stats should match database values"
            assert fallback_stats.correct_identifications == db_stats.correct_identifications, "Stats should match database values"
            
        finally:
            # Restore the original method
            monkeypatch.setattr(cache, "get", original_get)
        
        # Now update the database again
        db_stats.usage_count += 5
        db_session.commit()
        
        # Patch cache set to simulate failure
        original_set = cache.set
        
        async def failing_set(*args, **kwargs):
            raise Exception("Cache write failure")
        
        # Apply the patch
        monkeypatch.setattr(cache, "set", failing_set)
        
        try:
            # This should update database but fail to update cache
            updated_stats = await repository.get_pattern_statistics(pattern_name)
            
            # Verify database update worked despite cache failure
            assert updated_stats.usage_count == db_stats.usage_count, "Stats should reflect latest database update"
            
        finally:
            # Restore the original method
            monkeypatch.setattr(cache, "set", original_set)
            
        # Final check: cache coherence should be restored after fixing cache
        final_stats = await repository.get_pattern_statistics(pattern_name)
        assert final_stats.usage_count == db_stats.usage_count, "After cache is fixed, data should be coherent"

class TestPerformanceRecovery:
    """Test recovery from performance issues"""
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self, seed_pattern_data, monkeypatch):
        """Test handling of slow template selection"""
        # Create test data
        user_metrics = create_user_metrics()
        pattern_info = create_pattern_info()
        pattern_diversity = create_pattern_diversity()
        
        # Patch the template selector to be extremely slow
        original_select = question_generator.template_selector.select_template
        
        async def slow_select(*args, **kwargs):
            # Sleep to simulate a very slow operation
            await asyncio.sleep(2)
            return await original_select(*args, **kwargs)
        
        # Apply the patch
        monkeypatch.setattr(question_generator.template_selector, "select_template", slow_select)
        
        try:
            # Start time
            start_time = time.time()
            
            # Generate a question despite slow selection
            # The system should have some internal timeout mechanism
            question = await question_generator.generate_question(
                user_metrics=user_metrics,
                pattern_info=pattern_info,
                pattern_diversity=pattern_diversity,
                timeout_ms=500  # Set a short timeout
            )
            
            # End time
            end_time = time.time()
            
            # Check if the system used a fallback mechanism
            # or if it correctly timed out and returned a default question
            if question is not None:
                print(f"System produced a fallback question after timeout in {end_time - start_time:.2f}s")
                assert "question_text" in question, "Should have question text even after timeout"
            else:
                assert False, "System should produce a fallback question even when template selection times out"
            
        finally:
            # Restore the original method
            monkeypatch.setattr(question_generator.template_selector, "select_template", original_select)
    
    @pytest.mark.asyncio
    async def test_degraded_performance_recovery(self, seed_pattern_data, monkeypatch):
        """Test recovery from degraded performance"""
        # Create test data
        user_metrics = create_user_metrics()
        pattern_info = create_pattern_info()
        pattern_diversity = create_pattern_diversity()
        
        # Create patched functions with progressively improving performance
        original_generate = question_generator.prompt_engineering.generate_from_template
        call_count = 0
        
        async def variable_performance(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            
            if call_count == 1:
                # First call is very slow
                await asyncio.sleep(3)
                raise TimeoutError("Generation timed out")
            elif call_count == 2:
                # Second call is moderately slow
                await asyncio.sleep(1)
                return "Fallback question generated with moderate delay"
            else:
                # Subsequent calls return to normal
                return await original_generate(*args, **kwargs)
        
        # Apply the patch
        monkeypatch.setattr(question_generator.prompt_engineering, "generate_from_template", variable_performance)
        
        try:
            # Generate first question with simulated timeout
            question1 = await question_generator.generate_question(
                user_metrics=user_metrics,
                pattern_info=pattern_info,
                pattern_diversity=pattern_diversity
            )
            
            # The system should fallback to a simpler method or default question
            assert question1 is not None, "System should provide a fallback question on timeout"
            
            # Generate second question with improved but still degraded performance
            question2 = await question_generator.generate_question(
                user_metrics=user_metrics,
                pattern_info=pattern_info,
                pattern_diversity=pattern_diversity
            )
            
            # Should get a valid question
            assert question2 is not None, "Second question should be generated despite degraded performance"
            
            # Generate third question with normal performance
            question3 = await question_generator.generate_question(
                user_metrics=user_metrics,
                pattern_info=pattern_info,
                pattern_diversity=pattern_diversity
            )
            
            # Should get a valid question
            assert question3 is not None, "Third question should be generated with normal performance"
            assert call_count >= 3, "Should have made at least 3 calls to the function"
            
        finally:
            # Restore the original method
            monkeypatch.setattr(question_generator.prompt_engineering, "generate_from_template", original_generate)

class TestDataIntegrityRecovery:
    """Test recovery from data integrity issues"""
    
    @pytest.mark.asyncio
    async def test_corrupt_pattern_data_handling(self, db_session, repository, seed_pattern_data):
        """Test handling of corrupt pattern data"""
        # Create corrupt pattern statistics
        corrupt_pattern = PatternStatistics(
            pattern_id="corrupt_pattern",
            usage_count=-1,  # Invalid negative count
            correct_identifications=100,
            incorrect_identifications=50,
            average_confidence=2.5,  # Over maximum value
            # Missing other expected fields
        )
        
        # Add corrupt data to database
        db_session.add(corrupt_pattern)
        db_session.commit()
        
        # Create test data
        user_metrics = create_user_metrics()
        pattern_info = {
            "name": "corrupt_pattern",
            "category": "single",
            "description": "A test pattern with corrupt data",
            "confidence_threshold": 0.65
        }
        pattern_diversity = create_pattern_diversity()
        
        # Try to generate a question with corrupt pattern data
        question = await question_generator.generate_question(
            user_metrics=user_metrics,
            pattern_info=pattern_info,
            pattern_diversity=pattern_diversity
        )
        
        # Verify the system can still generate a question
        assert question is not None, "Should generate a question despite corrupt pattern data"
        assert "question_text" in question, "Question should have text"
        assert "template_id" in question, "Question should have a template ID"
    
    @pytest.mark.asyncio
    async def test_invalid_user_data_handling(self, db_session, repository, seed_pattern_data):
        """Test handling of invalid user data"""
        # Create extremely abnormal user performance data
        invalid_user_perf = UserPerformance(
            user_id="invalid_user",
            pattern_id="hammer",
            correct_count=10000,  # Extremely high count
            incorrect_count=-5,   # Invalid negative count
            average_response_time_ms=0,  # Invalid zero time
            last_seen_at=datetime.utcnow() + timedelta(days=365)  # Future date
        )
        
        # Add invalid data to database
        db_session.add(invalid_user_perf)
        db_session.commit()
        
        # Create test data with the invalid user
        user_metrics = create_user_metrics(user_id="invalid_user")
        pattern_info = create_pattern_info()
        pattern_diversity = create_pattern_diversity()
        
        # Try to generate a question with invalid user data
        question = await question_generator.generate_question(
            user_metrics=user_metrics,
            pattern_info=pattern_info,
            pattern_diversity=pattern_diversity
        )
        
        # Verify the system can still generate a question
        assert question is not None, "Should generate a question despite invalid user data"
        assert "question_text" in question, "Question should have text"
        assert "template_id" in question, "Question should have a template ID"
        
        # The difficulty should be reasonable despite the corrupted user performance
        assert "difficulty" in question, "Question should have difficulty"
        assert 0.5 <= float(question["difficulty"]) <= 3.0, \
            f"Difficulty should be in valid range (got {question['difficulty']})"
    
    @pytest.mark.asyncio
    async def test_missing_template_handling(self, seed_pattern_data, monkeypatch):
        """Test handling of missing templates"""
        # Create test data
        user_metrics = create_user_metrics()
        pattern_info = create_pattern_info()
        pattern_diversity = create_pattern_diversity()
        
        # Backup original templates
        original_templates = template_db.templates.copy()
        
        try:
            # Remove all templates to simulate missing data
            template_db.templates.clear()
            template_db.templates_by_difficulty.clear()
            
            # Try to generate a question with no templates
            question = await question_generator.generate_question(
                user_metrics=user_metrics,
                pattern_info=pattern_info,
                pattern_diversity=pattern_diversity
            )
            
            # System should generate a fallback question or use a default template
            assert question is not None, "Should generate a question despite missing templates"
            assert "question_text" in question, "Question should have text"
            
        finally:
            # Restore original templates
            template_db.templates = original_templates.copy()
            # Rebuild difficulty-based index
            template_db.templates_by_difficulty = {}
            for tid, template in template_db.templates.items():
                difficulty = template.difficulty
                if difficulty not in template_db.templates_by_difficulty:
                    template_db.templates_by_difficulty[difficulty] = []
                template_db.templates_by_difficulty[difficulty].append(template)

# Main test runner for the third part
@pytest.mark.asyncio
async def test_part3_error_handling():
    """Run the third part of the question generation tests"""
    print("\n===== Running Part 3: Error Handling and Recovery Mechanisms =====")
    
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
    asyncio.run(test_part3_error_handling()) 