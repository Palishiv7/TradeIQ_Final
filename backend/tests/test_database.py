"""
Comprehensive database functional tests for the candlestick pattern assessment system.

These tests focus on real database interactions without mocks to identify edge cases
and potential failure points in:
- Transaction management
- Concurrency handling
- Data integrity
- Error recovery
- Performance under load
"""

import os
import time
import pytest
import random
import threading
import multiprocessing
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text, exc
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

from database.init_db import initialize_database, get_session
from database.models import (
    Base, 
    PatternStatistics, 
    UserPerformance, 
    QuestionHistory,
    AssessmentAttempt,
    UserAnswer
)
from database.repositories import CandlestickRepository
from database.repositories.candlestick_repository import candlestick_repository


# Test database setup
@pytest.fixture(scope="module")
def test_db_url():
    """Get test database URL - prefer PostgreSQL for real-world testing"""
    # Try to use PostgreSQL for more realistic tests
    pg_url = os.environ.get("TEST_DB_URL")
    if pg_url:
        return pg_url
    
    # Fall back to SQLite for CI environments
    return "sqlite:///./test_database.db"


@pytest.fixture(scope="module")
def engine(test_db_url):
    """Create and configure the database engine"""
    # Use NullPool to ensure connections are closed properly
    test_engine = create_engine(
        test_db_url,
        # Echo SQL for debugging - consider setting to False for performance tests
        echo=False,
        # Enforce immediate transactions to catch race conditions
        isolation_level="READ COMMITTED",
        # Disable connection pooling for clearer test boundaries
        poolclass=NullPool
    )
    
    # Create all tables
    Base.metadata.create_all(test_engine)
    
    yield test_engine
    
    # Clean up - drop all tables
    Base.metadata.drop_all(test_engine)
    
    # Close engine connections
    test_engine.dispose()


@pytest.fixture
def db_session(engine):
    """Create a new database session for each test"""
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def repository(db_session):
    """Create a repository instance for testing"""
    return CandlestickRepository(db_session)


# Basic seed data for tests
@pytest.fixture
def seed_data(db_session):
    """Seed database with initial test data"""
    # Create pattern statistics
    patterns = [
        PatternStatistics(
            pattern_id="hammer",
            total_attempts=100,
            correct_attempts=75,
            avg_response_time=2.5,
            difficulty_rating=0.6,
            last_updated=datetime.utcnow()
        ),
        PatternStatistics(
            pattern_id="doji",
            total_attempts=80,
            correct_attempts=40,
            avg_response_time=4.1,
            difficulty_rating=0.8,
            last_updated=datetime.utcnow()
        ),
        PatternStatistics(
            pattern_id="engulfing",
            total_attempts=150,
            correct_attempts=90,
            avg_response_time=3.2,
            difficulty_rating=0.7,
            last_updated=datetime.utcnow()
        )
    ]
    db_session.add_all(patterns)
    
    # Create user performance records
    user_performance = [
        UserPerformance(
            user_id="test-user-1",
            pattern_id="hammer",
            total_attempts=10,
            correct_attempts=7,
            avg_response_time=2.8,
            last_attempt_at=datetime.utcnow() - timedelta(days=1)
        ),
        UserPerformance(
            user_id="test-user-1",
            pattern_id="engulfing",
            total_attempts=15,
            correct_attempts=9,
            avg_response_time=3.5,
            last_attempt_at=datetime.utcnow() - timedelta(hours=5)
        ),
        UserPerformance(
            user_id="test-user-2",
            pattern_id="doji",
            total_attempts=8,
            correct_attempts=3,
            avg_response_time=4.7,
            last_attempt_at=datetime.utcnow() - timedelta(hours=2)
        )
    ]
    db_session.add_all(user_performance)
    
    # Create assessment attempts
    attempts = [
        AssessmentAttempt(
            id="test-session-1",
            user_id="test-user-1",
            started_at=datetime.utcnow() - timedelta(days=1),
            completed_at=datetime.utcnow() - timedelta(days=1) + timedelta(minutes=15),
            questions_total=10,
            questions_completed=10,
            correct_answers=7,
            avg_response_time=3.2,
            difficulty_level=0.5,
            score=70,
            is_completed=True
        ),
        AssessmentAttempt(
            id="test-session-2",
            user_id="test-user-1",
            started_at=datetime.utcnow() - timedelta(hours=5),
            completed_at=None,
            questions_total=10,
            questions_completed=3,
            correct_answers=2,
            avg_response_time=2.9,
            difficulty_level=0.7,
            score=None,
            is_completed=False
        )
    ]
    db_session.add_all(attempts)
    
    db_session.commit()
    
    yield
    
    # Clean up data
    db_session.query(UserPerformance).delete()
    db_session.query(PatternStatistics).delete()
    db_session.query(AssessmentAttempt).delete()
    db_session.query(QuestionHistory).delete()
    db_session.query(UserAnswer).delete()
    db_session.commit()


# Basic functionality tests
class TestBasicDatabaseFunctionality:
    """Test basic database CRUD operations and constraints"""
    
    def test_create_pattern_statistics(self, db_session):
        """Test creating new pattern statistics"""
        # Create a new pattern statistic
        new_pattern = PatternStatistics(
            pattern_id="morning_star",
            total_attempts=50,
            correct_attempts=25,
            avg_response_time=5.1,
            difficulty_rating=0.9,
            last_updated=datetime.utcnow()
        )
        db_session.add(new_pattern)
        db_session.commit()
        
        # Verify it was created
        result = db_session.query(PatternStatistics).filter_by(pattern_id="morning_star").first()
        assert result is not None
        assert result.total_attempts == 50
        assert result.correct_attempts == 25
    
    def test_unique_constraint_pattern_id(self, db_session, seed_data):
        """Test unique constraint on pattern_id"""
        # Attempt to create duplicate pattern
        duplicate_pattern = PatternStatistics(
            pattern_id="hammer",  # Already exists
            total_attempts=200,
            correct_attempts=100,
            avg_response_time=3.0,
            difficulty_rating=0.7,
            last_updated=datetime.utcnow()
        )
        db_session.add(duplicate_pattern)
        
        # Should raise an integrity error
        with pytest.raises(exc.IntegrityError):
            db_session.commit()
        
        # Rollback to clean state
        db_session.rollback()
    
    def test_update_pattern_statistics(self, db_session, seed_data):
        """Test updating pattern statistics"""
        # Get existing pattern
        pattern = db_session.query(PatternStatistics).filter_by(pattern_id="hammer").first()
        original_attempts = pattern.total_attempts
        
        # Update it
        pattern.total_attempts += 1
        pattern.correct_attempts += 1
        pattern.last_updated = datetime.utcnow()
        db_session.commit()
        
        # Verify update
        updated = db_session.query(PatternStatistics).filter_by(pattern_id="hammer").first()
        assert updated.total_attempts == original_attempts + 1
    
    def test_delete_pattern_statistics(self, db_session, seed_data):
        """Test deleting pattern statistics"""
        # Delete a pattern
        db_session.query(PatternStatistics).filter_by(pattern_id="doji").delete()
        db_session.commit()
        
        # Verify it's gone
        result = db_session.query(PatternStatistics).filter_by(pattern_id="doji").first()
        assert result is None
    
    def test_composite_key_user_performance(self, db_session, seed_data):
        """Test composite primary key in UserPerformance"""
        # Create a new user performance record
        new_performance = UserPerformance(
            user_id="test-user-1",
            pattern_id="doji",  # New combination with existing user
            total_attempts=5,
            correct_attempts=2,
            avg_response_time=3.5,
            last_attempt_at=datetime.utcnow()
        )
        db_session.add(new_performance)
        db_session.commit()
        
        # This should work because the combination is unique
        saved = db_session.query(UserPerformance).filter_by(
            user_id="test-user-1", 
            pattern_id="doji"
        ).first()
        assert saved is not None
        
        # Try to add duplicate
        duplicate = UserPerformance(
            user_id="test-user-1",
            pattern_id="doji",  # Already exists
            total_attempts=10,
            correct_attempts=5,
            avg_response_time=2.5,
            last_attempt_at=datetime.utcnow()
        )
        db_session.add(duplicate)
        
        # Should raise integrity error
        with pytest.raises(exc.IntegrityError):
            db_session.commit()
        
        db_session.rollback()


# Transaction tests
class TestTransactionManagement:
    """Test database transaction behavior"""
    
    def test_successful_transaction(self, db_session):
        """Test successful transaction with multiple operations"""
        # Start transaction
        try:
            # Create two related objects
            pattern = PatternStatistics(
                pattern_id="transaction_test",
                total_attempts=10,
                correct_attempts=5,
                avg_response_time=2.0,
                difficulty_rating=0.5,
                last_updated=datetime.utcnow()
            )
            db_session.add(pattern)
            
            performance = UserPerformance(
                user_id="transaction-user",
                pattern_id="transaction_test",
                total_attempts=1,
                correct_attempts=1,
                avg_response_time=1.5,
                last_attempt_at=datetime.utcnow()
            )
            db_session.add(performance)
            
            # Commit transaction
            db_session.commit()
            
            # Verify both objects were created
            pattern_result = db_session.query(PatternStatistics).filter_by(pattern_id="transaction_test").first()
            perf_result = db_session.query(UserPerformance).filter_by(
                user_id="transaction-user", 
                pattern_id="transaction_test"
            ).first()
            
            assert pattern_result is not None
            assert perf_result is not None
            
        except Exception:
            db_session.rollback()
            raise
    
    def test_transaction_rollback(self, db_session):
        """Test transaction rollback on error"""
        # Start transaction
        try:
            # Add first object
            pattern = PatternStatistics(
                pattern_id="rollback_test",
                total_attempts=10,
                correct_attempts=5,
                avg_response_time=2.0,
                difficulty_rating=0.5,
                last_updated=datetime.utcnow()
            )
            db_session.add(pattern)
            
            # This will cause an error - missing required fields
            broken_record = UserPerformance(
                user_id="rollback-user",
                pattern_id="rollback_test"
                # Missing required fields
            )
            db_session.add(broken_record)
            
            # This should fail and trigger rollback
            db_session.commit()
            
        except Exception:
            db_session.rollback()
            
        # Verify neither object was saved
        pattern_result = db_session.query(PatternStatistics).filter_by(pattern_id="rollback_test").first()
        assert pattern_result is None
    
    def test_nested_transaction(self, db_session):
        """Test nested transaction behavior"""
        # Outer transaction
        try:
            # Create pattern in outer transaction
            pattern = PatternStatistics(
                pattern_id="nested_test",
                total_attempts=10,
                correct_attempts=5,
                avg_response_time=2.0,
                difficulty_rating=0.5,
                last_updated=datetime.utcnow()
            )
            db_session.add(pattern)
            
            # Create nested transaction with nested_transaction context
            with db_session.begin_nested():
                # Add first performance in nested transaction
                perf1 = UserPerformance(
                    user_id="nested-user",
                    pattern_id="nested_test",
                    total_attempts=1,
                    correct_attempts=1,
                    avg_response_time=1.5,
                    last_attempt_at=datetime.utcnow()
                )
                db_session.add(perf1)
                
                # This will cause the nested transaction to fail
                broken_record = UserPerformance(
                    user_id="nested-user",
                    pattern_id="nested_test"
                    # Missing required fields
                )
                db_session.add(broken_record)
                
                # This will raise an exception and roll back the nested transaction
                # but the outer transaction can still continue
        
        except Exception:
            # Continue with outer transaction
            pass
        
        # Add another valid record to outer transaction
        perf2 = UserPerformance(
            user_id="nested-user-2",
            pattern_id="nested_test",
            total_attempts=2,
            correct_attempts=1,
            avg_response_time=2.5,
            last_attempt_at=datetime.utcnow()
        )
        db_session.add(perf2)
        
        # Commit outer transaction
        db_session.commit()
        
        # Verify that outer transaction objects were saved
        # but nested transaction objects were not
        pattern_result = db_session.query(PatternStatistics).filter_by(pattern_id="nested_test").first()
        perf1_result = db_session.query(UserPerformance).filter_by(
            user_id="nested-user", 
            pattern_id="nested_test"
        ).first()
        perf2_result = db_session.query(UserPerformance).filter_by(
            user_id="nested-user-2", 
            pattern_id="nested_test"
        ).first()
        
        assert pattern_result is not None
        assert perf1_result is None  # Should not exist - rolled back in nested transaction
        assert perf2_result is not None  # Should exist - part of outer transaction


# Concurrency tests
class TestConcurrency:
    """Tests for concurrent database access patterns and race conditions"""
    
    @pytest.mark.skipif(os.environ.get("TEST_DB_URL") is None,
                       reason="Concurrency tests require PostgreSQL")
    def test_concurrent_reads(self, engine):
        """Test concurrent read operations"""
        # Create sessions
        Session = sessionmaker(bind=engine)
        read_results = []
        errors = []
        
        def read_worker(worker_id):
            """Worker function for concurrent reads"""
            try:
                # Each worker gets its own session
                session = Session()
                
                try:
                    # Read a pattern randomly to simulate concurrent read traffic
                    patterns = ["hammer", "doji", "engulfing"]
                    pattern_id = random.choice(patterns)
                    
                    result = session.query(PatternStatistics).filter_by(
                        pattern_id=pattern_id
                    ).first()
                    
                    if result:
                        read_results.append({
                            "worker_id": worker_id,
                            "pattern_id": pattern_id,
                            "total_attempts": result.total_attempts
                        })
                finally:
                    session.close()
                    
            except Exception as e:
                errors.append(f"Worker {worker_id} error: {str(e)}")
        
        # Create and start threads
        threads = []
        for i in range(10):  # 10 concurrent readers
            thread = threading.Thread(target=read_worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify no errors occurred
        assert len(errors) == 0, f"Errors during concurrent reads: {errors}"
        
        # All reads should have succeeded
        assert len(read_results) == 10
    
    @pytest.mark.skipif(os.environ.get("TEST_DB_URL") is None,
                       reason="Concurrency tests require PostgreSQL")
    def test_concurrent_writes(self, engine, seed_data):
        """Test concurrent write operations to same record - potential race condition"""
        # Create sessions
        Session = sessionmaker(bind=engine)
        errors = []
        
        # Use these for verification
        initial_value = None
        expected_increment = 10  # We'll have 10 workers each incrementing by 1
        
        # Get initial value
        with Session() as session:
            pattern = session.query(PatternStatistics).filter_by(pattern_id="hammer").first()
            initial_value = pattern.total_attempts
        
        def update_worker(worker_id):
            """Worker function for concurrent updates"""
            try:
                # Each worker gets its own session
                session = Session()
                
                try:
                    # Get the pattern
                    pattern = session.query(PatternStatistics).filter_by(pattern_id="hammer").first()
                    
                    # Increment attempts - this is where race conditions can occur
                    pattern.total_attempts += 1
                    
                    # Simulate some work that takes time (makes race conditions more likely)
                    time.sleep(random.uniform(0.01, 0.05))
                    
                    # Commit update
                    session.commit()
                except Exception as e:
                    session.rollback()
                    errors.append(f"Worker {worker_id} database error: {str(e)}")
                finally:
                    session.close()
                    
            except Exception as e:
                errors.append(f"Worker {worker_id} error: {str(e)}")
        
        # Create and start threads
        threads = []
        for i in range(expected_increment):  # 10 concurrent writers
            thread = threading.Thread(target=update_worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify no errors occurred
        assert len(errors) == 0, f"Errors during concurrent writes: {errors}"
        
        # Check final value - if there were race conditions, it might be less than expected
        with Session() as session:
            pattern = session.query(PatternStatistics).filter_by(pattern_id="hammer").first()
            final_value = pattern.total_attempts
        
        # This might fail if database isolation level allows for race conditions
        # With proper isolation, this should pass (READ COMMITTED or higher)
        assert final_value == initial_value + expected_increment, \
            f"Race condition detected: expected {initial_value + expected_increment}, got {final_value}"
    
    @pytest.mark.skipif(os.environ.get("TEST_DB_URL") is None,
                       reason="Concurrency tests require PostgreSQL")
    def test_mixed_read_write_workload(self, engine, seed_data):
        """Test mixed read and write operations - realistic workload"""
        # Create sessions
        Session = sessionmaker(bind=engine)
        results = {
            "reads": 0,
            "writes": 0,
            "errors": []
        }
        
        def worker(worker_id, operation_count=5):
            """Worker function for mixed operations"""
            try:
                # Each worker gets its own session
                session = Session()
                
                try:
                    for i in range(operation_count):
                        # 70% chance of read, 30% chance of write
                        if random.random() < 0.7:
                            # Read operation
                            patterns = ["hammer", "doji", "engulfing"]
                            pattern_id = random.choice(patterns)
                            
                            session.query(PatternStatistics).filter_by(
                                pattern_id=pattern_id
                            ).first()
                            
                            results["reads"] += 1
                        else:
                            # Write operation - update user performance
                            pattern_id = random.choice(["hammer", "doji", "engulfing"])
                            user_id = f"test-user-{worker_id}"
                            
                            # Try to find existing record
                            performance = session.query(UserPerformance).filter_by(
                                user_id=user_id,
                                pattern_id=pattern_id
                            ).first()
                            
                            if performance:
                                # Update existing
                                performance.total_attempts += 1
                                performance.avg_response_time = (
                                    (performance.avg_response_time * performance.total_attempts + random.uniform(1, 5)) /
                                    (performance.total_attempts + 1)
                                )
                            else:
                                # Create new
                                new_perf = UserPerformance(
                                    user_id=user_id,
                                    pattern_id=pattern_id,
                                    total_attempts=1,
                                    correct_attempts=int(random.random() > 0.5),  # 50% correct
                                    avg_response_time=random.uniform(1, 5),
                                    last_attempt_at=datetime.utcnow()
                                )
                                session.add(new_perf)
                            
                            # Simulate some work
                            time.sleep(random.uniform(0.001, 0.01))
                            
                            # Commit each write separately
                            session.commit()
                            results["writes"] += 1
                finally:
                    session.close()
                    
            except Exception as e:
                results["errors"].append(f"Worker {worker_id} error: {str(e)}")
        
        # Create and start threads
        threads = []
        worker_count = 5
        for i in range(worker_count):
            thread = threading.Thread(target=worker, args=(i, 10))  # Each worker does 10 operations
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify results
        assert len(results["errors"]) == 0, f"Errors during mixed workload: {results['errors']}"
        assert results["reads"] > 0, "No read operations performed"
        assert results["writes"] > 0, "No write operations performed"
        
        print(f"Mixed workload results: {results['reads']} reads, {results['writes']} writes")


# Extreme data tests
class TestExtremeData:
    """Tests for handling extreme data cases"""
    
    def test_extremely_large_values(self, db_session):
        """Test handling of extremely large numeric values"""
        # Create pattern with extremely large values
        extreme_pattern = PatternStatistics(
            pattern_id="extreme_values",
            total_attempts=2147483647,  # Max 32-bit int
            correct_attempts=2147483646,
            avg_response_time=999999.999,
            difficulty_rating=0.99999,
            last_updated=datetime.utcnow()
        )
        db_session.add(extreme_pattern)
        db_session.commit()
        
        # Retrieve and verify
        result = db_session.query(PatternStatistics).filter_by(pattern_id="extreme_values").first()
        
        assert result is not None
        assert result.total_attempts == 2147483647
        assert result.correct_attempts == 2147483646
        assert result.avg_response_time == 999999.999
        assert result.difficulty_rating == 0.99999
    
    def test_unicode_characters(self, db_session):
        """Test handling of Unicode characters in string fields"""
        # Create pattern with Unicode characters in ID
        unicode_pattern = PatternStatistics(
            pattern_id="unicode_test_🔥💯🚀",  # Emojis and Unicode
            total_attempts=10,
            correct_attempts=5,
            avg_response_time=2.5,
            difficulty_rating=0.5,
            last_updated=datetime.utcnow()
        )
        db_session.add(unicode_pattern)
        db_session.commit()
        
        # Retrieve and verify
        result = db_session.query(PatternStatistics).filter_by(pattern_id="unicode_test_🔥💯🚀").first()
        
        assert result is not None
        assert result.pattern_id == "unicode_test_🔥💯🚀"
    
    def test_very_long_string(self, db_session):
        """Test handling of very long string values"""
        # Generate a very long string to test string field limits
        long_user_id = "a" * 1000  # 1000 character string
        
        # Create record with very long string
        long_string_record = UserPerformance(
            user_id=long_user_id,
            pattern_id="long_string_test",
            total_attempts=1,
            correct_attempts=1,
            avg_response_time=1.0,
            last_attempt_at=datetime.utcnow()
        )
        
        # This might fail if the database has column size limits
        # The test verifies proper validation or error handling
        try:
            db_session.add(long_string_record)
            db_session.commit()
            
            # If it succeeded, verify the data was stored correctly
            result = db_session.query(UserPerformance).filter_by(
                user_id=long_user_id
            ).first()
            
            assert result is not None
            assert result.user_id == long_user_id
        except exc.DataError:
            # This is expected if the database enforces column size limits
            db_session.rollback()
            
            # Verify we can create a record with a reasonable length
            reasonable_user_id = "a" * 50
            reasonable_record = UserPerformance(
                user_id=reasonable_user_id,
                pattern_id="long_string_test",
                total_attempts=1,
                correct_attempts=1,
                avg_response_time=1.0,
                last_attempt_at=datetime.utcnow()
            )
            
            db_session.add(reasonable_record)
            db_session.commit()
            
            result = db_session.query(UserPerformance).filter_by(
                user_id=reasonable_user_id
            ).first()
            
            assert result is not None
            assert result.user_id == reasonable_user_id
    
    def test_zero_values(self, db_session):
        """Test zero values in numeric fields"""
        # Create a pattern with zero values
        zero_pattern = PatternStatistics(
            pattern_id="zero_values",
            total_attempts=0,
            correct_attempts=0,
            avg_response_time=0.0,
            difficulty_rating=0.0,
            last_updated=datetime.utcnow()
        )
        db_session.add(zero_pattern)
        db_session.commit()
        
        # Retrieve and verify
        result = db_session.query(PatternStatistics).filter_by(pattern_id="zero_values").first()
        
        assert result is not None
        assert result.total_attempts == 0
        assert result.correct_attempts == 0
        assert result.avg_response_time == 0.0
        assert result.difficulty_rating == 0.0
    
    def test_negative_values(self, db_session):
        """Test handling of negative values in numeric fields"""
        # Create pattern with negative values - might be handled as validation error
        negative_pattern = PatternStatistics(
            pattern_id="negative_values",
            total_attempts=-10,  # Should be non-negative
            correct_attempts=-5,  # Should be non-negative
            avg_response_time=-2.5,  # Should be non-negative
            difficulty_rating=-0.5,  # Should be between 0 and 1
            last_updated=datetime.utcnow()
        )
        
        # This should ideally fail validation or raise an exception
        # But we want to test the behavior regardless
        db_session.add(negative_pattern)
        
        try:
            db_session.commit()
            
            # If it committed successfully, verify the values
            result = db_session.query(PatternStatistics).filter_by(pattern_id="negative_values").first()
            
            assert result is not None
            # In a real system, we would expect validations to prevent negative values
            # This checks actual behavior, which might allow negative values
            assert result.total_attempts == -10
            assert result.correct_attempts == -5
            assert result.avg_response_time == -2.5
            assert result.difficulty_rating == -0.5
            
            # Document the issue if validation is expected
            print("WARNING: Database allowed negative values in fields that should be non-negative")
        except Exception as e:
            # If an exception was raised, that's good - values were invalid
            db_session.rollback()
            print(f"Properly rejected negative values with error: {str(e)}")


# Error handling tests
class TestErrorHandling:
    """Tests for database error handling"""
    
    def test_duplicate_id_error_handling(self, db_session, seed_data):
        """Test error handling for duplicate ID insertion"""
        # Attempt to create duplicate pattern
        duplicate = PatternStatistics(
            pattern_id="hammer",  # Already exists
            total_attempts=50,
            correct_attempts=25,
            avg_response_time=3.0,
            difficulty_rating=0.7,
            last_updated=datetime.utcnow()
        )
        db_session.add(duplicate)
        
        # This should fail with integrity error
        with pytest.raises(exc.IntegrityError):
            db_session.commit()
        
        # Important: Test proper rollback handling
        db_session.rollback()
        
        # Verify we can still use the session after rollback
        new_valid_pattern = PatternStatistics(
            pattern_id="after_rollback",
            total_attempts=10,
            correct_attempts=5,
            avg_response_time=2.0,
            difficulty_rating=0.5,
            last_updated=datetime.utcnow()
        )
        db_session.add(new_valid_pattern)
        db_session.commit()
        
        # Verify it was created successfully
        result = db_session.query(PatternStatistics).filter_by(pattern_id="after_rollback").first()
        assert result is not None
    
    def test_constraint_violation_handling(self, db_session):
        """Test handling of constraint violations"""
        # Try to create a record with invalid data (missing required field)
        invalid_attempt = AssessmentAttempt(
            id="missing_fields_test",
            # Missing user_id which should be required
            started_at=datetime.utcnow(),
            questions_total=10
            # Missing other required fields
        )
        db_session.add(invalid_attempt)
        
        # Should raise some form of constraint error
        with pytest.raises(Exception) as exc_info:
            db_session.commit()
        
        # Check that we got an appropriate error
        error = str(exc_info.value)
        assert "constraint" in error.lower() or "null" in error.lower() or "not null" in error.lower()
        
        # Rollback and verify session is still usable
        db_session.rollback()
        
        # Create valid record
        valid_attempt = AssessmentAttempt(
            id="valid_after_constraint_error",
            user_id="test-user-error",
            started_at=datetime.utcnow(),
            completed_at=None,
            questions_total=10,
            questions_completed=0,
            correct_answers=0,
            avg_response_time=0,
            difficulty_level=0.5,
            score=None,
            is_completed=False
        )
        db_session.add(valid_attempt)
        db_session.commit()
        
        # Verify it was created
        result = db_session.query(AssessmentAttempt).filter_by(id="valid_after_constraint_error").first()
        assert result is not None
    
    def test_invalid_foreign_key(self, db_session):
        """Test handling of invalid foreign key references"""
        # This assumes there's a foreign key relationship
        # For example, if UserAnswer has foreign key to AssessmentAttempt
        
        # Try to create a record with non-existent foreign key
        user_answer = UserAnswer(
            id="invalid_fk_test",
            assessment_attempt_id="non_existent_session",  # This doesn't exist
            question_id="q1",
            selected_option="hammer",
            is_correct=True,
            response_time=1.5,
            submitted_at=datetime.utcnow()
        )
        db_session.add(user_answer)
        
        # Should fail with foreign key constraint error if FKs are enforced
        try:
            db_session.commit()
            # If commit succeeds, foreign keys might not be enforced
            # which is a potential issue to investigate
            print("WARNING: Foreign key constraints may not be enforced!")
        except exc.IntegrityError as e:
            # Expected behavior if foreign keys are enforced
            error = str(e)
            assert "foreign key" in error.lower() or "fk" in error.lower() or "referenced" in error.lower()
            db_session.rollback()
            
            # Create a valid assessment attempt first
            valid_attempt = AssessmentAttempt(
                id="for_user_answer",
                user_id="test-user-fk",
                started_at=datetime.utcnow(),
                completed_at=None,
                questions_total=1,
                questions_completed=0,
                correct_answers=0,
                avg_response_time=0,
                difficulty_level=0.5,
                score=None,
                is_completed=False
            )
            db_session.add(valid_attempt)
            db_session.commit()
            
            # Now create user answer with valid foreign key
            valid_user_answer = UserAnswer(
                id="valid_fk_test",
                assessment_attempt_id="for_user_answer",  # Valid FK
                question_id="q1",
                selected_option="hammer",
                is_correct=True,
                response_time=1.5,
                submitted_at=datetime.utcnow()
            )
            db_session.add(valid_user_answer)
            db_session.commit()
            
            # Verify it was created
            result = db_session.query(UserAnswer).filter_by(id="valid_fk_test").first()
            assert result is not None
    
    def test_connection_error_recovery(self, test_db_url):
        """Test recovery from connection errors"""
        # Create a bad connection string to force error
        bad_url = test_db_url.replace("localhost", "non-existent-host")
        
        # Try to connect with bad URL - should fail
        try:
            bad_engine = create_engine(bad_url, connect_args={"connect_timeout": 2})
            bad_engine.connect()
            
            # If this succeeds, the test can't proceed as expected
            pytest.skip("Expected connection error did not occur")
        except Exception:
            # Expected - connection should fail
            pass
        
        # Now use the valid connection to verify recovery
        good_engine = create_engine(test_db_url)
        try:
            # This should succeed
            conn = good_engine.connect()
            conn.close()
            good_engine.dispose()
        except Exception as e:
            pytest.fail(f"Failed to recover with valid connection: {str(e)}")


# Repository edge case tests
class TestRepositoryEdgeCases:
    """Tests for edge cases in the repository layer"""
    
    def test_update_nonexistent_pattern(self, repository):
        """Test updating statistics for a non-existent pattern"""
        # Try to update a pattern that doesn't exist
        stats = repository.update_pattern_statistics(
            pattern_id="nonexistent_pattern",
            is_correct=True,
            response_time=1.5
        )
        
        # Depending on implementation, this might create the pattern or fail
        # Verify behavior matches expectations
        if stats is not None:
            # If it created a new pattern, verify it's correct
            assert stats.pattern_id == "nonexistent_pattern"
            assert stats.total_attempts == 1
            assert stats.correct_attempts == 1
            assert stats.avg_response_time == 1.5
        else:
            # If it returns None, make sure pattern wasn't created
            result = repository.session.query(PatternStatistics).filter_by(
                pattern_id="nonexistent_pattern"
            ).first()
            assert result is None
    
    def test_get_nonexistent_user_performance(self, repository):
        """Test retrieving non-existent user performance"""
        # Try to get performance for non-existent user/pattern
        result = repository.get_user_performance(
            user_id="nonexistent_user",
            pattern_id="nonexistent_pattern"
        )
        
        # Should not raise exception, should return None
        assert result is None
    
    def test_update_user_performance_edge_cases(self, repository, db_session):
        """Test edge cases in user performance updates"""
        # First create a test performance record
        test_perf = UserPerformance(
            user_id="edge-case-user",
            pattern_id="edge-case-pattern",
            total_attempts=100,
            correct_attempts=50,
            avg_response_time=2.0,
            last_attempt_at=datetime.utcnow() - timedelta(days=1)
        )
        db_session.add(test_perf)
        db_session.commit()
        
        # Test extremes: 0 response time
        repository.update_user_performance(
            user_id="edge-case-user",
            pattern_id="edge-case-pattern",
            is_correct=True,
            response_time=0
        )
        
        result = repository.get_user_performance(
            user_id="edge-case-user", 
            pattern_id="edge-case-pattern"
        )
        
        assert result is not None
        assert result.total_attempts == 101
        assert result.correct_attempts == 51
        # Avg response time should now be lower due to 0 value
        assert result.avg_response_time < 2.0
        
        # Test invalid response time - negative
        try:
            repository.update_user_performance(
                user_id="edge-case-user",
                pattern_id="edge-case-pattern",
                is_correct=True,
                response_time=-1.0  # Negative - should be validated
            )
            
            # Get updated record
            result = repository.get_user_performance(
                user_id="edge-case-user", 
                pattern_id="edge-case-pattern"
            )
            
            # Check if validation occurred
            if result.avg_response_time < 0:
                print("WARNING: Repository allowed negative response time")
        except Exception as e:
            # Exception is expected if validation is in place
            print(f"Repository properly rejected negative time: {str(e)}")
    
    def test_transaction_in_repository(self, repository, db_session):
        """Test repository's transaction management"""
        # Test transaction context manager if implemented
        if hasattr(repository, 'transaction'):
            try:
                with repository.transaction():
                    # Create new pattern
                    repository.update_pattern_statistics(
                        pattern_id="transaction_ctx_test",
                        is_correct=True,
                        response_time=1.5
                    )
                    
                    # Force an error inside transaction
                    raise ValueError("Test error to force rollback")
            except ValueError:
                pass
            
            # Verify rollback occurred - pattern should not exist
            result = db_session.query(PatternStatistics).filter_by(
                pattern_id="transaction_ctx_test"
            ).first()
            
            assert result is None, "Transaction did not roll back properly"
            
            # Try successful transaction
            with repository.transaction():
                repository.update_pattern_statistics(
                    pattern_id="successful_transaction",
                    is_correct=True,
                    response_time=1.5
                )
            
            # Verify commit occurred
            result = db_session.query(PatternStatistics).filter_by(
                pattern_id="successful_transaction"
            ).first()
            
            assert result is not None, "Successful transaction did not commit"
        else:
            pytest.skip("Repository does not implement transaction context manager")


# Performance and load testing
class TestDatabasePerformance:
    """Tests for database performance and behavior under load"""
    
    @pytest.mark.skipif(os.environ.get("TEST_DB_URL") is None,
                       reason="Performance tests require PostgreSQL")
    def test_bulk_insert_performance(self, engine):
        """Test performance of bulk insertions"""
        Session = sessionmaker(bind=engine)
        session = Session()
        
        try:
            # Record start time
            start_time = time.time()
            
            # Create large batch of assessment attempts
            batch_size = 100
            attempts = []
            
            for i in range(batch_size):
                attempts.append(AssessmentAttempt(
                    id=f"bulk-insert-{i}",
                    user_id=f"bulk-user-{i % 10}",  # 10 different users
                    started_at=datetime.utcnow() - timedelta(hours=i % 24),
                    completed_at=datetime.utcnow() - timedelta(hours=i % 24) + timedelta(minutes=15),
                    questions_total=10,
                    questions_completed=10,
                    correct_answers=i % 11,  # 0-10 correct answers
                    avg_response_time=random.uniform(1.0, 10.0),
                    difficulty_level=random.uniform(0.1, 1.0),
                    score=i % 101,  # 0-100 score
                    is_completed=True
                ))
            
            # Use bulk_save_objects for better performance
            session.bulk_save_objects(attempts)
            session.commit()
            
            # Record end time
            end_time = time.time()
            bulk_insert_time = end_time - start_time
            
            # Now measure individual inserts
            start_time = time.time()
            
            for i in range(10):  # Only do 10 for comparison, then multiply
                attempt = AssessmentAttempt(
                    id=f"individual-insert-{i}",
                    user_id=f"individual-user-{i}",
                    started_at=datetime.utcnow(),
                    completed_at=datetime.utcnow() + timedelta(minutes=15),
                    questions_total=10,
                    questions_completed=10,
                    correct_answers=i,
                    avg_response_time=random.uniform(1.0, 10.0),
                    difficulty_level=random.uniform(0.1, 1.0),
                    score=i * 10,
                    is_completed=True
                )
                session.add(attempt)
                session.commit()
            
            end_time = time.time()
            individual_insert_time = (end_time - start_time) * (batch_size / 10)
            
            # Print and verify performance difference
            print(f"\nBulk insert of {batch_size} records: {bulk_insert_time:.4f}s")
            print(f"Estimated individual inserts: {individual_insert_time:.4f}s")
            print(f"Speedup factor: {individual_insert_time / bulk_insert_time:.2f}x")
            
            # Bulk operations should be significantly faster
            assert bulk_insert_time < individual_insert_time, "Bulk insert wasn't faster than individual inserts"
            
        finally:
            # Clean up - delete all test records
            session.query(AssessmentAttempt).filter(
                AssessmentAttempt.id.like('bulk-insert-%')
            ).delete(synchronize_session=False)
            
            session.query(AssessmentAttempt).filter(
                AssessmentAttempt.id.like('individual-insert-%')
            ).delete(synchronize_session=False)
            
            session.commit()
            session.close()
    
    @pytest.mark.skipif(os.environ.get("TEST_DB_URL") is None,
                       reason="Performance tests require PostgreSQL")
    def test_query_performance(self, engine, seed_data):
        """Test query performance with different techniques"""
        Session = sessionmaker(bind=engine)
        session = Session()
        
        try:
            # First seed a larger dataset
            batch_size = 100
            patterns = ["hammer", "doji", "engulfing", "morning_star", "evening_star"]
            
            # Create user performance records
            user_perfs = []
            for i in range(batch_size):
                user_id = f"perf-user-{i % 20}"  # 20 different users
                for pattern_id in patterns:
                    user_perfs.append(UserPerformance(
                        user_id=user_id,
                        pattern_id=pattern_id,
                        total_attempts=random.randint(1, 50),
                        correct_attempts=random.randint(0, 30),
                        avg_response_time=random.uniform(1.0, 10.0),
                        last_attempt_at=datetime.utcnow() - timedelta(days=random.randint(0, 30))
                    ))
            
            session.bulk_save_objects(user_perfs)
            session.commit()
            
            # Test 1: Individual queries for each user (N+1 query problem)
            start_time = time.time()
            
            n_plus_1_results = {}
            for i in range(20):  # Query for each user
                user_id = f"perf-user-{i}"
                perf_records = session.query(UserPerformance).filter_by(user_id=user_id).all()
                n_plus_1_results[user_id] = len(perf_records)
            
            n_plus_1_time = time.time() - start_time
            
            # Test 2: Single query with filter
            start_time = time.time()
            
            single_query_results = {}
            # Get all user_ids
            user_ids = [f"perf-user-{i}" for i in range(20)]
            # Make one query to get all performance records
            all_records = session.query(UserPerformance).filter(
                UserPerformance.user_id.in_(user_ids)
            ).all()
            
            # Group by user_id
            for record in all_records:
                user_id = record.user_id
                if user_id not in single_query_results:
                    single_query_results[user_id] = 0
                single_query_results[user_id] += 1
            
            single_query_time = time.time() - start_time
            
            # Print and compare
            print(f"\nN+1 queries time: {n_plus_1_time:.4f}s")
            print(f"Single query time: {single_query_time:.4f}s")
            print(f"Speedup factor: {n_plus_1_time / single_query_time:.2f}x")
            
            # Verify both strategies got the same results
            assert n_plus_1_results == single_query_results, "Query strategies returned different results"
            
            # The single query should be faster
            assert single_query_time < n_plus_1_time, "Single query wasn't faster than N+1 queries"
            
        finally:
            # Clean up test data
            session.query(UserPerformance).filter(
                UserPerformance.user_id.like('perf-user-%')
            ).delete(synchronize_session=False)
            
            session.commit()
            session.close()
    
    @pytest.mark.skipif(os.environ.get("TEST_DB_URL") is None,
                       reason="Performance tests require PostgreSQL")
    def test_connection_pool_behavior(self, test_db_url):
        """Test database connection pool behavior under load"""
        # Create engine with explicit pool settings
        pool_size = 5
        max_overflow = 5
        test_engine = create_engine(
            test_db_url,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_timeout=3,
            pool_recycle=300
        )
        
        Session = sessionmaker(bind=test_engine)
        
        # Track concurrent connections
        concurrent_connections = 0
        max_concurrent = 0
        connection_lock = threading.Lock()
        
        def worker():
            """Worker function that obtains a connection"""
            nonlocal concurrent_connections, max_concurrent
            
            try:
                # Get session from pool
                session = Session()
                
                with connection_lock:
                    concurrent_connections += 1
                    max_concurrent = max(max_concurrent, concurrent_connections)
                
                try:
                    # Perform a simple query
                    session.execute(text("SELECT 1"))
                    
                    # Hold connection for a short time
                    time.sleep(random.uniform(0.1, 0.5))
                    
                    # Execute another query
                    session.execute(text("SELECT 2"))
                finally:
                    session.close()
                    with connection_lock:
                        concurrent_connections -= 1
            except Exception as e:
                with connection_lock:
                    print(f"\nWorker error: {str(e)}")
        
        # Run more threads than pool size to test overflow
        thread_count = pool_size + max_overflow + 5  # 5 more than limit
        threads = []
        
        for i in range(thread_count):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()
            # Stagger thread starts slightly
            time.sleep(0.05)
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Print results
        print(f"\nPool settings: size={pool_size}, max_overflow={max_overflow}")
        print(f"Thread count: {thread_count}")
        print(f"Maximum concurrent connections: {max_concurrent}")
        
        # Pool limit should be enforced - max concurrent connections should be <= pool_size + max_overflow
        assert max_concurrent <= pool_size + max_overflow, "Connection pool limits were exceeded"
        
        # Dispose engine
        test_engine.dispose()
    
    @pytest.mark.skipif(os.environ.get("TEST_DB_URL") is None,
                       reason="Performance tests require PostgreSQL")
    @pytest.mark.skip(reason="Long-running test, run manually")
    def test_stress_test_multi_process(self, test_db_url):
        """Full stress test with multiple processes"""
        # This test simulates high database load with multiple processes
        
        def process_worker(process_id, success_count, error_count):
            """Worker process function"""
            # Create engine for this process
            engine = create_engine(test_db_url, pool_size=3, max_overflow=5)
            Session = sessionmaker(bind=engine)
            
            operations_per_process = 50
            
            try:
                for i in range(operations_per_process):
                    session = Session()
                    try:
                        # Randomly choose read or write operation
                        if random.random() < 0.8:  # 80% reads
                            # Read operation
                            session.query(PatternStatistics).all()
                            with success_count.get_lock():
                                success_count.value += 1
                        else:
                            # Write operation
                            pattern_id = f"stress-test-{process_id}-{i}"
                            stats = PatternStatistics(
                                pattern_id=pattern_id,
                                total_attempts=random.randint(1, 1000),
                                correct_attempts=random.randint(1, 500),
                                avg_response_time=random.uniform(1.0, 10.0),
                                difficulty_rating=random.uniform(0.1, 1.0),
                                last_updated=datetime.utcnow()
                            )
                            session.add(stats)
                            session.commit()
                            with success_count.get_lock():
                                success_count.value += 1
                    except Exception as e:
                        with error_count.get_lock():
                            error_count.value += 1
                        print(f"Process {process_id} error: {str(e)}")
                    finally:
                        session.close()
                
                    # Random sleep to simulate variable workload
                    time.sleep(random.uniform(0.01, 0.1))
            finally:
                engine.dispose()
        
        # Use multiprocessing to create real separate processes
        process_count = 4  # Use a reasonable number based on your CPU cores
        
        # Shared counters for tracking results
        success_count = multiprocessing.Value('i', 0)
        error_count = multiprocessing.Value('i', 0)
        
        # Create and start processes
        processes = []
        start_time = time.time()
        
        for i in range(process_count):
            p = multiprocessing.Process(
                target=process_worker,
                args=(i, success_count, error_count)
            )
            processes.append(p)
            p.start()
        
        # Wait for all processes to complete
        for p in processes:
            p.join()
        
        # Calculate results
        end_time = time.time()
        total_time = end_time - start_time
        total_operations = success_count.value + error_count.value
        ops_per_second = total_operations / total_time
        
        # Print results
        print(f"\nStress test completed:")
        print(f"Processes: {process_count}")
        print(f"Total operations: {total_operations}")
        print(f"Successful operations: {success_count.value}")
        print(f"Failed operations: {error_count.value}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Operations per second: {ops_per_second:.2f}")
        print(f"Error rate: {error_count.value / total_operations * 100:.2f}%")
        
        # Clean up - remove test data
        engine = create_engine(test_db_url)
        with engine.connect() as conn:
            conn.execute(text("DELETE FROM pattern_statistics WHERE pattern_id LIKE 'stress-test-%'"))
            conn.commit()
        engine.dispose()
        
        # Assert reasonable success rate
        assert success_count.value > 0, "No successful operations"
        assert error_count.value / total_operations < 0.05, "Error rate too high (>5%)"


# Data consistency tests
class TestDataConsistency:
    """Tests for data consistency and integrity across the database"""
    
    def test_repository_calculation_consistency(self, repository, db_session):
        """Test consistency of calculations in repository layer"""
        # Create test data
        pattern = PatternStatistics(
            pattern_id="consistency-test",
            total_attempts=100,
            correct_attempts=50,
            avg_response_time=3.0,
            difficulty_rating=0.5,
            last_updated=datetime.utcnow() - timedelta(days=1)
        )
        db_session.add(pattern)
        db_session.commit()
        
        # Update via repository multiple times with different data
        updates = [
            {'is_correct': True, 'response_time': 1.0},
            {'is_correct': False, 'response_time': 2.0},
            {'is_correct': True, 'response_time': 3.0},
            {'is_correct': True, 'response_time': 4.0},
            {'is_correct': False, 'response_time': 5.0}
        ]
        
        for update in updates:
            repository.update_pattern_statistics(
                pattern_id="consistency-test",
                is_correct=update['is_correct'],
                response_time=update['response_time']
            )
        
        # Calculate expected values
        expected_total = 100 + len(updates)
        expected_correct = 50 + sum(1 for u in updates if u['is_correct'])
        
        # Complex calculation for expected average response time
        original_sum = 100 * 3.0  # original avg * count
        additional_sum = sum(u['response_time'] for u in updates)
        expected_avg = (original_sum + additional_sum) / expected_total
        
        # Get actual values from database
        result = db_session.query(PatternStatistics).filter_by(pattern_id="consistency-test").first()
        
        # Verify calculations are consistent
        assert result.total_attempts == expected_total, "Total attempts calculation inconsistent"
        assert result.correct_attempts == expected_correct, "Correct attempts calculation inconsistent"
        
        # For average, allow small floating point difference
        avg_diff = abs(result.avg_response_time - expected_avg)
        assert avg_diff < 0.001, f"Average response time calculation inconsistent. Expected: {expected_avg}, Got: {result.avg_response_time}"
        
        # Verify last_updated was updated
        assert result.last_updated > pattern.last_updated, "last_updated field was not updated"
    
    def test_concurrent_update_consistency(self, engine, seed_data):
        """Test data consistency when updates occur concurrently"""
        if os.environ.get("TEST_DB_URL") is None:
            pytest.skip("Concurrency tests require PostgreSQL")
            
        Session = sessionmaker(bind=engine)
        
        # Create test pattern
        with Session() as session:
            pattern = PatternStatistics(
                pattern_id="concurrent-consistency",
                total_attempts=0,
                correct_attempts=0,
                avg_response_time=0,
                difficulty_rating=0.5,
                last_updated=datetime.utcnow()
            )
            session.add(pattern)
            session.commit()
        
        # Track success and failure counts
        success_count = 0
        failure_count = 0
        lock = threading.Lock()
        
        def worker(is_correct, response_time):
            """Worker function for concurrent updates"""
            nonlocal success_count, failure_count
            
            session = Session()
            try:
                # Select for update with a timeout to avoid deadlocks
                pattern = session.query(PatternStatistics).filter_by(
                    pattern_id="concurrent-consistency"
                ).with_for_update(nowait=True).first()
                
                if pattern is None:
                    with lock:
                        failure_count += 1
                    return
                
                # Get original values
                total = pattern.total_attempts
                correct = pattern.correct_attempts
                avg_time = pattern.avg_response_time
                
                # Update values
                pattern.total_attempts += 1
                if is_correct:
                    pattern.correct_attempts += 1
                
                # Update average
                if pattern.total_attempts == 1:
                    pattern.avg_response_time = response_time
                else:
                    pattern.avg_response_time = (
                        (avg_time * total + response_time) / 
                        pattern.total_attempts
                    )
                
                pattern.last_updated = datetime.utcnow()
                
                # Commit changes
                session.commit()
                
                with lock:
                    success_count += 1
            except Exception:
                session.rollback()
                with lock:
                    failure_count += 1
            finally:
                session.close()
        
        # Create and start threads
        threads = []
        update_count = 20
        
        correct_count = 0
        for i in range(update_count):
            is_correct = i % 2 == 0  # Alternate true/false
            if is_correct:
                correct_count += 1
                
            thread = threading.Thread(
                target=worker, 
                args=(is_correct, float(i))
            )
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check final state
        with Session() as session:
            result = session.query(PatternStatistics).filter_by(
                pattern_id="concurrent-consistency"
            ).first()
            
            # Print results
            print(f"\nConcurrent updates:")
            print(f"Successful updates: {success_count}")
            print(f"Failed updates: {failure_count}")
            print(f"Final total_attempts: {result.total_attempts}")
            print(f"Final correct_attempts: {result.correct_attempts}")
            
            # Verify consistency - all successful operations should be reflected
            assert result.total_attempts == success_count, "Final total_attempts inconsistent with successful operations"
            
            # Confirm correct_attempts is consistent with the correct updates that succeeded
            # This is trickier to verify exactly since we don't know which specific threads succeeded/failed
            # We can only verify it's a reasonable value between 0 and success_count
            assert 0 <= result.correct_attempts <= success_count, "Final correct_attempts outside valid range"
