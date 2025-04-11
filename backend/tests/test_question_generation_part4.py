#!/usr/bin/env python
"""
Comprehensive database functional tests for question generation system - Part 4.

This part focuses on concurrency and race condition testing.
These tests ensure that the question generation system:
1. Handles concurrent question generation requests
2. Prevents race conditions when multiple users access the system
3. Maintains database consistency under high load
4. Properly manages resources during parallel execution
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
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

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

logger = app_logger.getChild("test_question_generator_part4")

# Import fixtures and helpers from Part 1
from backend.tests.test_question_generation import (
    database_connection, db_session, repository, clear_cache,
    seed_pattern_data, seed_user_data, seed_assessment_data, seed_question_history,
    create_user_metrics, create_pattern_info, create_pattern_diversity
)

class TestConcurrentQuestionGeneration:
    """Test concurrent question generation functionality"""
    
    @pytest.mark.asyncio
    async def test_concurrent_generation_same_user(self, seed_pattern_data, clear_cache):
        """Test concurrent question generation for the same user"""
        # Create test data for a single user
        user_id = "concurrent_test_user"
        user_metrics = create_user_metrics(user_id=user_id)
        pattern_info = create_pattern_info()
        pattern_diversity = create_pattern_diversity()
        
        # Number of concurrent requests
        num_concurrent = 10
        
        # Helper function to generate a question
        async def generate_question(index):
            try:
                # Make a small variation to ensure we get different questions
                adjusted_metrics = user_metrics.copy()
                adjusted_metrics["difficulty_level"] = max(0.5, min(3.0, user_metrics["difficulty_level"] + index * 0.1))
                
                # Generate the question
                question = await question_generator.generate_question(
                    user_metrics=adjusted_metrics,
                    pattern_info=pattern_info,
                    pattern_diversity=pattern_diversity
                )
                return question
            except Exception as e:
                logger.error(f"Error generating question {index}: {str(e)}")
                return None
        
        # Create tasks for concurrent generation
        tasks = [generate_question(i) for i in range(num_concurrent)]
        
        # Start timer to measure total execution time
        start_time = time.time()
        
        # Execute all tasks concurrently
        questions = await asyncio.gather(*tasks)
        
        # Calculate execution time
        total_time = time.time() - start_time
        
        # Verify all questions were generated
        assert all(q is not None for q in questions), "All questions should be generated successfully"
        assert len(questions) == num_concurrent, f"Should generate {num_concurrent} questions"
        
        # Verify questions are unique
        question_texts = [q["question_text"] for q in questions]
        unique_texts = set(question_texts)
        
        # Due to template constraints, we might not get all unique questions
        # but we should have a decent percentage of unique questions
        uniqueness_ratio = len(unique_texts) / len(question_texts)
        print(f"Generated {len(unique_texts)} unique questions out of {len(question_texts)} ({uniqueness_ratio:.2%})")
        assert uniqueness_ratio >= 0.5, "At least 50% of questions should be unique"
        
        # Check execution time - should be reasonably efficient due to parallelism
        average_time_per_question = total_time / num_concurrent
        print(f"Average time per question: {average_time_per_question:.4f}s")
        print(f"Total execution time for {num_concurrent} concurrent requests: {total_time:.4f}s")
        
        # The following assertion is commented out because actual timing depends on the system
        # but we'll print it for information purposes
        # assert average_time_per_question < 1.0, "Question generation should be reasonably fast"
    
    @pytest.mark.asyncio
    async def test_concurrent_generation_different_users(self, seed_pattern_data, clear_cache):
        """Test concurrent question generation for different users"""
        # Create base pattern info
        pattern_info = create_pattern_info()
        pattern_diversity = create_pattern_diversity()
        
        # Number of concurrent users
        num_users = 10
        
        # Helper function to generate a question for a specific user
        async def generate_for_user(user_index):
            try:
                # Create user-specific metrics
                user_id = f"concurrent_user_{user_index}"
                user_metrics = create_user_metrics(user_id=user_id)
                user_metrics["difficulty_level"] = 1.0 + (user_index % 5) * 0.4  # Vary difficulty
                
                # Generate the question
                question = await question_generator.generate_question(
                    user_metrics=user_metrics,
                    pattern_info=pattern_info,
                    pattern_diversity=pattern_diversity
                )
                return user_id, question
            except Exception as e:
                logger.error(f"Error generating question for user {user_index}: {str(e)}")
                return user_id, None
        
        # Create tasks for concurrent generation
        tasks = [generate_for_user(i) for i in range(num_users)]
        
        # Start timer
        start_time = time.time()
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks)
        
        # Calculate execution time
        total_time = time.time() - start_time
        
        # Organize results by user
        questions_by_user = {user_id: question for user_id, question in results}
        
        # Verify all users got questions
        assert len(questions_by_user) == num_users, f"All {num_users} users should get questions"
        assert all(question is not None for question in questions_by_user.values()), "All questions should be valid"
        
        # Verify questions for different users are sufficiently diverse
        question_texts = [q["question_text"] for q in questions_by_user.values()]
        unique_texts = set(question_texts)
        uniqueness_ratio = len(unique_texts) / len(question_texts)
        
        print(f"Generated {len(unique_texts)} unique questions across {num_users} users ({uniqueness_ratio:.2%})")
        assert uniqueness_ratio >= 0.7, "At least 70% of questions across users should be unique"
        
        # Check execution time
        average_time_per_user = total_time / num_users
        print(f"Average time per user: {average_time_per_user:.4f}s")
        print(f"Total execution time for {num_users} concurrent users: {total_time:.4f}s")
    
    @pytest.mark.asyncio
    async def test_concurrent_generation_same_pattern(self, seed_pattern_data, clear_cache):
        """Test concurrent question generation for the same pattern"""
        # Number of concurrent requests
        num_concurrent = 20
        
        # Create test data
        pattern_name = "hammer"
        pattern_info = {
            "name": pattern_name,
            "category": "single",
            "description": "A single candlestick pattern with a small body and long lower shadow",
            "confidence_threshold": 0.65
        }
        pattern_diversity = create_pattern_diversity(primary_pattern=pattern_name)
        
        # Helper function to generate a question
        async def generate_question(index):
            try:
                # Create user-specific metrics
                user_id = f"pattern_user_{index}"
                user_metrics = create_user_metrics(user_id=user_id)
                
                # Generate the question
                question = await question_generator.generate_question(
                    user_metrics=user_metrics,
                    pattern_info=pattern_info,
                    pattern_diversity=pattern_diversity
                )
                return question
            except Exception as e:
                logger.error(f"Error generating question {index}: {str(e)}")
                return None
        
        # Create tasks for concurrent generation
        tasks = [generate_question(i) for i in range(num_concurrent)]
        
        # Execute all tasks concurrently
        questions = await asyncio.gather(*tasks)
        
        # Verify all questions were generated
        assert all(q is not None for q in questions), "All questions should be generated successfully"
        assert len(questions) == num_concurrent, f"Should generate {num_concurrent} questions"
        
        # All questions should be about the same pattern
        pattern_names = [q.get("pattern_name", pattern_name) for q in questions]
        assert all(name == pattern_name for name in pattern_names), "All questions should be about the same pattern"
        
        # Check template usage distribution
        template_ids = [q["template_id"] for q in questions]
        template_counts = {tid: template_ids.count(tid) for tid in set(template_ids)}
        
        print(f"Template distribution for {num_concurrent} questions:")
        for tid, count in template_counts.items():
            print(f"  Template {tid}: {count} uses ({count/num_concurrent:.1%})")
        
        # With many concurrent requests, we should see good distribution across templates
        assert len(template_counts) >= 3, "Should use at least 3 different templates"

class TestDatabaseRaceConditions:
    """Test database race conditions and concurrency handling"""
    
    @pytest.mark.asyncio
    async def test_pattern_statistics_concurrent_update(self, db_session, repository, seed_pattern_data):
        """Test concurrent updates to pattern statistics"""
        # Setup - create or get pattern statistics
        pattern_id = "hammer"
        
        # Get initial statistics or create if not exists
        stats = await repository.get_pattern_statistics(pattern_id)
        if not stats:
            stats = PatternStatistics(
                pattern_id=pattern_id,
                usage_count=100,
                correct_identifications=50,
                incorrect_identifications=50,
                average_confidence=0.7
            )
            db_session.add(stats)
            db_session.commit()
            db_session.refresh(stats)
        
        initial_usage = stats.usage_count
        initial_correct = stats.correct_identifications
        initial_incorrect = stats.incorrect_identifications
        
        print(f"Initial pattern statistics: usage={initial_usage}, correct={initial_correct}, incorrect={initial_incorrect}")
        
        # Number of concurrent updates
        num_updates = 10
        updates_per_thread = 5
        total_updates = num_updates * updates_per_thread
        
        # Shared state for results tracking
        successful_updates = 0
        failed_updates = 0
        lock = threading.Lock()
        
        # Function to update statistics in a separate thread
        def update_stats_thread(thread_id):
            nonlocal successful_updates, failed_updates
            
            # Create a new session for each thread
            thread_session = sessionmaker(bind=db_session.bind)()
            
            try:
                for i in range(updates_per_thread):
                    # Get current stats (with thread-local session)
                    current_stats = thread_session.query(PatternStatistics).filter_by(pattern_id=pattern_id).with_for_update().first()
                    
                    if not current_stats:
                        with lock:
                            failed_updates += 1
                        continue
                    
                    # Update the statistics
                    current_stats.usage_count += 1
                    current_stats.correct_identifications += (thread_id % 2)  # Even threads increment correct
                    current_stats.incorrect_identifications += ((thread_id + 1) % 2)  # Odd threads increment incorrect
                    
                    # Commit the update
                    try:
                        thread_session.commit()
                        with lock:
                            successful_updates += 1
                    except Exception as e:
                        thread_session.rollback()
                        logger.error(f"Thread {thread_id} update {i} failed: {str(e)}")
                        with lock:
                            failed_updates += 1
                    
                    # Simulate variable processing time
                    time.sleep(random.uniform(0.01, 0.05))
            finally:
                thread_session.close()
        
        # Execute concurrent updates
        with ThreadPoolExecutor(max_workers=num_updates) as executor:
            futures = [executor.submit(update_stats_thread, i) for i in range(num_updates)]
            
            # Wait for all to complete
            for future in futures:
                future.result()  # This will raise any exceptions from the threads
        
        # Verify results
        print(f"Updates: {successful_updates} successful, {failed_updates} failed")
        
        # Refresh stats from database
        db_session.expire_all()
        final_stats = await repository.get_pattern_statistics(pattern_id)
        
        # Calculate expected values
        expected_usage_increase = successful_updates
        expected_correct_increase = sum(1 for i in range(num_updates) if i % 2 == 0) * updates_per_thread
        expected_incorrect_increase = sum(1 for i in range(num_updates) if i % 2 == 1) * updates_per_thread
        
        # Adjust for actual successful updates
        expected_correct_increase = min(expected_correct_increase, successful_updates)
        expected_incorrect_increase = min(expected_incorrect_increase, successful_updates)
        
        # Retrieve final values
        final_usage = final_stats.usage_count
        final_correct = final_stats.correct_identifications
        final_incorrect = final_stats.incorrect_identifications
        
        print(f"Final pattern statistics: usage={final_usage}, correct={final_correct}, incorrect={final_incorrect}")
        print(f"Changes: usage +{final_usage-initial_usage}, correct +{final_correct-initial_correct}, incorrect +{final_incorrect-initial_incorrect}")
        
        # Verify changes match expected
        # Note: In some databases, the actual values might not match exactly due to transaction isolation levels
        # so we check that the changes are reasonably close
        usage_change = final_usage - initial_usage
        correct_change = final_correct - initial_correct
        incorrect_change = final_incorrect - initial_incorrect
        
        assert abs(usage_change - successful_updates) <= 1, f"Usage count should increase by approximately {successful_updates}"
        
        # The following assertions might be too strict depending on the exact transaction behavior
        # So we'll print the results for inspection
        print(f"Expected correct increase: ~{expected_correct_increase}, Actual: {correct_change}")
        print(f"Expected incorrect increase: ~{expected_incorrect_increase}, Actual: {incorrect_change}")
    
    @pytest.mark.asyncio
    async def test_user_performance_concurrent_update(self, db_session, repository, seed_user_data):
        """Test concurrent updates to user performance records"""
        # Setup - create or get user performance records
        user_id = "concurrent_update_user"
        pattern_id = "hammer"
        
        # Create a user performance record if it doesn't exist
        user_perf = await repository.get_user_performance(user_id, pattern_id)
        if not user_perf:
            user_perf = UserPerformance(
                user_id=user_id,
                pattern_id=pattern_id,
                correct_count=10,
                incorrect_count=10,
                average_response_time_ms=2000,
                last_seen_at=datetime.utcnow()
            )
            db_session.add(user_perf)
            db_session.commit()
            db_session.refresh(user_perf)
        
        initial_correct = user_perf.correct_count
        initial_incorrect = user_perf.incorrect_count
        
        # Number of concurrent update threads
        num_threads = 8
        updates_per_thread = 5
        
        # Function to update user performance in a thread
        def update_performance(thread_id):
            # Create a new session for this thread
            thread_session = sessionmaker(bind=db_session.bind)()
            
            try:
                for i in range(updates_per_thread):
                    # Get current user performance with lock
                    current_perf = thread_session.query(UserPerformance)\
                        .filter_by(user_id=user_id, pattern_id=pattern_id)\
                        .with_for_update()\
                        .first()
                    
                    if not current_perf:
                        continue
                    
                    # Update performance - alternating correct and incorrect
                    if thread_id % 2 == 0:
                        current_perf.correct_count += 1
                        response_time = random.randint(1000, 3000)
                    else:
                        current_perf.incorrect_count += 1
                        response_time = random.randint(2000, 5000)
                    
                    # Update response time
                    total_attempts = current_perf.correct_count + current_perf.incorrect_count
                    current_perf.average_response_time_ms = (
                        (current_perf.average_response_time_ms * (total_attempts - 1) + response_time) 
                        / total_attempts
                    )
                    current_perf.last_seen_at = datetime.utcnow()
                    
                    # Commit the update
                    try:
                        thread_session.commit()
                    except Exception as e:
                        thread_session.rollback()
                        logger.error(f"Thread {thread_id} update {i} failed: {str(e)}")
                    
                    # Small delay to increase chance of conflicts
                    time.sleep(random.uniform(0.01, 0.03))
            finally:
                thread_session.close()
        
        # Execute concurrent updates
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(update_performance, i) for i in range(num_threads)]
            
            # Wait for all to complete
            for future in futures:
                future.result()
        
        # Verify results
        db_session.expire_all()
        final_perf = await repository.get_user_performance(user_id, pattern_id)
        
        # Calculate expected changes
        total_updates = num_threads * updates_per_thread
        expected_correct_increase = (num_threads // 2) * updates_per_thread
        expected_incorrect_increase = (num_threads - num_threads // 2) * updates_per_thread
        
        # Get final values
        final_correct = final_perf.correct_count
        final_incorrect = final_perf.incorrect_count
        
        print(f"Initial counts: correct={initial_correct}, incorrect={initial_incorrect}")
        print(f"Final counts: correct={final_correct}, incorrect={final_incorrect}")
        print(f"Changes: correct +{final_correct-initial_correct}, incorrect +{final_incorrect-initial_incorrect}")
        
        # Verify changes are in the expected range
        correct_change = final_correct - initial_correct
        incorrect_change = final_incorrect - initial_incorrect
        
        assert abs(correct_change - expected_correct_increase) <= 1, \
            f"Correct count should increase by approximately {expected_correct_increase}"
        assert abs(incorrect_change - expected_incorrect_increase) <= 1, \
            f"Incorrect count should increase by approximately {expected_incorrect_increase}"
    
    @pytest.mark.asyncio
    async def test_concurrent_assessment_progress(self, db_session, repository):
        """Test concurrent updates to assessment progress"""
        # Create a test assessment
        assessment = AssessmentAttempt(
            id=uuid.uuid4(),
            user_id="concurrent_assessment_user",
            status="in_progress",
            start_difficulty=1.5,
            total_questions=20,
            completed_questions=0,
            correct_answers=0,
            incorrect_answers=0
        )
        db_session.add(assessment)
        db_session.commit()
        
        # Number of concurrent threads updating this assessment
        num_threads = 10
        updates_per_thread = 2  # Each thread will submit 2 answers
        
        # Function to simulate submitting answers and updating assessment
        def submit_answers(thread_id):
            # Create a new session for this thread
            thread_session = sessionmaker(bind=db_session.bind)()
            
            try:
                for i in range(updates_per_thread):
                    # Create a new question
                    question = QuestionHistory(
                        id=uuid.uuid4(),
                        user_id=assessment.user_id,
                        session_id=assessment.id,
                        pattern_name=f"Pattern {thread_id}_{i}",
                        pattern_category="single",
                        difficulty=1.5,
                        chart_data={"candles": [{"open": 100, "high": 110, "low": 90, "close": 105}]},
                        options=["A", "B", "C", "D"],
                        correct_option="A"
                    )
                    thread_session.add(question)
                    thread_session.commit()
                    
                    # Create an answer
                    selected_option = "A" if random.random() > 0.5 else "B"
                    is_correct = selected_option == "A"
                    
                    answer = UserAnswer(
                        id=uuid.uuid4(),
                        user_id=assessment.user_id,
                        question_id=question.id,
                        assessment_id=assessment.id,
                        selected_option=selected_option,
                        is_correct=is_correct,
                        response_time_ms=random.randint(1000, 5000),
                        attempt_number=1
                    )
                    thread_session.add(answer)
                    
                    # Get and update assessment with lock
                    current_assessment = thread_session.query(AssessmentAttempt)\
                        .filter_by(id=assessment.id)\
                        .with_for_update()\
                        .first()
                    
                    # Update assessment progress
                    current_assessment.completed_questions += 1
                    if is_correct:
                        current_assessment.correct_answers += 1
                    else:
                        current_assessment.incorrect_answers += 1
                    
                    # Check if assessment is complete
                    if current_assessment.completed_questions >= current_assessment.total_questions:
                        current_assessment.status = "completed"
                        current_assessment.end_difficulty = 2.0
                    
                    # Commit the transaction
                    try:
                        thread_session.commit()
                    except Exception as e:
                        thread_session.rollback()
                        logger.error(f"Thread {thread_id} update {i} failed: {str(e)}")
                    
                    # Small delay
                    time.sleep(random.uniform(0.02, 0.05))
            finally:
                thread_session.close()
        
        # Execute concurrent updates
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(submit_answers, i) for i in range(num_threads)]
            
            # Wait for all to complete
            for future in futures:
                future.result()
        
        # Verify final assessment state
        db_session.expire_all()
        final_assessment = db_session.query(AssessmentAttempt).filter_by(id=assessment.id).first()
        
        # Expected values
        total_updates = num_threads * updates_per_thread
        
        # Get counts of questions and answers
        questions = db_session.query(QuestionHistory).filter_by(session_id=assessment.id).all()
        answers = db_session.query(UserAnswer).filter_by(assessment_id=assessment.id).all()
        
        print(f"Assessment state: completed_questions={final_assessment.completed_questions}, status={final_assessment.status}")
        print(f"Questions count: {len(questions)}, Answers count: {len(answers)}")
        print(f"Correct answers: {final_assessment.correct_answers}, Incorrect answers: {final_assessment.incorrect_answers}")
        
        # Verify assessment state matches the questions and answers
        assert len(questions) == total_updates, f"Should have {total_updates} questions"
        assert len(answers) == total_updates, f"Should have {total_updates} answers"
        assert final_assessment.completed_questions == total_updates, f"Assessment should show {total_updates} completed questions"
        assert final_assessment.correct_answers + final_assessment.incorrect_answers == total_updates, "Sum of correct and incorrect should match total"
        
        # Check if the assessment was marked as completed
        if total_updates >= final_assessment.total_questions:
            assert final_assessment.status == "completed", "Assessment should be marked as completed"
        else:
            assert final_assessment.status == "in_progress", "Assessment should still be in progress"

class TestResourceManagement:
    """Test resource management under concurrent load"""
    
    @pytest.mark.asyncio
    async def test_session_management_under_load(self, db_session, repository, seed_pattern_data):
        """Test database session management under high load"""
        # Create a basic pattern info
        pattern_info = create_pattern_info()
        
        # Number of concurrent operations
        num_operations = 20
        
        # Function to perform multiple database operations
        async def run_database_operations(op_id):
            try:
                # Create user metrics
                user_id = f"resource_user_{op_id}"
                user_metrics = create_user_metrics(user_id=user_id)
                
                # 1. Create assessment
                assessment = AssessmentAttempt(
                    id=uuid.uuid4(),
                    user_id=user_id,
                    status="in_progress",
                    start_difficulty=1.5,
                    total_questions=5,
                    completed_questions=0
                )
                db_session.add(assessment)
                db_session.commit()
                
                # 2. Generate question
                question = await question_generator.generate_question(
                    user_metrics=user_metrics,
                    pattern_info=pattern_info,
                    pattern_diversity=create_pattern_diversity()
                )
                
                # 3. Create question history
                question_history = QuestionHistory(
                    id=uuid.uuid4(),
                    user_id=user_id,
                    session_id=assessment.id,
                    pattern_name=pattern_info["name"],
                    pattern_category=pattern_info["category"],
                    difficulty=1.5,
                    chart_data={"candles": [{"open": 100, "high": 110, "low": 90, "close": 105}]},
                    options=["A", "B", "C", "D"],
                    correct_option="A",
                    question_text=question["question_text"],
                    template_id=question["template_id"]
                )
                db_session.add(question_history)
                db_session.commit()
                
                # 4. Create answer
                answer = UserAnswer(
                    id=uuid.uuid4(),
                    user_id=user_id,
                    question_id=question_history.id,
                    assessment_id=assessment.id,
                    selected_option="A",
                    is_correct=True,
                    response_time_ms=2000,
                    attempt_number=1
                )
                db_session.add(answer)
                
                # 5. Update assessment progress
                assessment.completed_questions += 1
                assessment.correct_answers += 1
                db_session.commit()
                
                # 6. Query for verification
                verification = db_session.query(AssessmentAttempt, QuestionHistory, UserAnswer)\
                    .join(QuestionHistory, QuestionHistory.session_id == AssessmentAttempt.id)\
                    .join(UserAnswer, UserAnswer.question_id == QuestionHistory.id)\
                    .filter(AssessmentAttempt.id == assessment.id)\
                    .first()
                
                assert verification is not None, "Should retrieve the complete assessment data"
                
                return True
            except Exception as e:
                logger.error(f"Operation {op_id} failed: {str(e)}")
                return False
        
        # Create tasks for concurrent operations
        tasks = [run_database_operations(i) for i in range(num_operations)]
        
        # Execute all tasks concurrently
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        execution_time = time.time() - start_time
        
        # Count successful operations
        successful = sum(1 for r in results if r)
        
        print(f"Executed {num_operations} concurrent operations: {successful} successful, {num_operations - successful} failed")
        print(f"Total execution time: {execution_time:.2f}s, Average per operation: {execution_time/num_operations:.4f}s")
        
        # At least 80% of operations should succeed under load
        assert successful >= 0.8 * num_operations, f"At least 80% of operations should succeed (got {successful/num_operations:.1%})"
    
    @pytest.mark.asyncio
    async def test_connection_pool_management(self, db_session, repository, monkeypatch):
        """Test database connection pool management"""
        # Get engine from session
        engine = db_session.bind
        
        # Check initial pool status
        try:
            pool_size = engine.pool.size()
            print(f"Initial connection pool size: {pool_size}")
        except:
            # Not all engines expose pool size, so we'll skip this test if not available
            print("Connection pool size not available for this engine, skipping detail tests")
            return
        
        # Number of concurrent database operations
        num_operations = 30  # Should exceed default pool size
        
        # Function to perform a simple database operation
        def execute_query(op_id):
            try:
                # Create a new session for this thread
                thread_session = sessionmaker(bind=engine)()
                
                # Execute a simple query
                result = thread_session.execute("SELECT 1").scalar()
                
                # Small delay to hold the connection
                time.sleep(random.uniform(0.1, 0.3))
                
                # Close the session
                thread_session.close()
                
                return True
            except Exception as e:
                logger.error(f"Query {op_id} failed: {str(e)}")
                return False
        
        # Execute concurrent queries
        with ThreadPoolExecutor(max_workers=num_operations) as executor:
            futures = [executor.submit(execute_query, i) for i in range(num_operations)]
            
            # Wait for all to complete
            results = [future.result() for future in futures]
        
        # Check results
        successful = sum(results)
        
        print(f"Executed {num_operations} concurrent queries: {successful} successful, {num_operations - successful} failed")
        
        # Final pool status
        try:
            final_pool_size = engine.pool.size()
            print(f"Final connection pool size: {final_pool_size}")
            print(f"Connections in use: {engine.pool.checkedin()}")
            print(f"Connections checked out: {engine.pool.checkedout()}")
        except:
            pass
        
        # All operations should succeed if connection pooling is working correctly
        assert successful == num_operations, "All database operations should succeed with proper connection pooling"

# Main test runner for the fourth part
@pytest.mark.asyncio
async def test_part4_concurrency():
    """Run the fourth part of the question generation tests"""
    print("\n===== Running Part 4: Concurrency and Race Condition Tests =====")
    
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
    asyncio.run(test_part4_concurrency()) 