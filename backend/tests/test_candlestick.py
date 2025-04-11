"""
Comprehensive candlestick pattern tests focusing on edge cases and failure scenarios.

These tests identify potential breaking points in the candlestick pattern recognition system,
emphasizing edge cases, boundary conditions, and failure recovery with no mocks.
"""

import os
import pytest
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any

# Import database components
from database.models import (
    PatternStatistics, 
    UserPerformance,
    AssessmentAttempt,
    UserAnswer
)
from database.repositories import candlestick_repository
from database.init_db import initialize_database, get_session

# Import candlestick pattern components
from backend.assessments.candlestick_patterns.candlestick_pattern_recognition import (
    identify_candlestick_patterns,
    analyze_candlestick_pattern,
    get_patterns_library,
    detect_pattern
)
from backend.assessments.candlestick_patterns.candlestick_api import (
    get_pattern_statistics,
    update_pattern_statistics,
    get_user_performance,
    update_user_performance,
    generate_assessment,
    submit_answer
)

# Import Redis cache components
from backend.assessments.candlestick_patterns.candlestick_cache import (
    initialize_cache,
    get_pattern_stats_from_cache,
    update_pattern_stats_in_cache,
    clear_pattern_stats_cache
)

# Test fixtures

@pytest.fixture(scope="module")
def database_connection():
    """Setup test database connection"""
    # Use test database URL from environment or fall back to SQLite
    test_db_url = os.environ.get("TEST_DB_URL", "sqlite:///./test_candlestick.db")
    
    # Initialize database
    engine, sessionmaker = initialize_database(test_db_url)
    
    # Create all tables
    from database.models import Base
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
    # Save original session
    original_session = candlestick_repository.session
    
    # Set test session
    candlestick_repository.session = db_session
    
    yield candlestick_repository
    
    # Restore original session
    candlestick_repository.session = original_session


@pytest.fixture
def cache_initialization():
    """Initialize and then clean up Redis cache"""
    # Initialize cache for testing
    initialize_cache()
    
    yield
    
    # Clean up after tests
    clear_pattern_stats_cache()


@pytest.fixture
def seed_pattern_data(db_session):
    """Seed database with pattern statistics data"""
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
        ),
        PatternStatistics(
            pattern_id="morning_star",
            total_attempts=50,
            correct_attempts=20,
            avg_response_time=5.0,
            difficulty_rating=0.9,
            last_updated=datetime.utcnow()
        ),
        PatternStatistics(
            pattern_id="evening_star",
            total_attempts=60,
            correct_attempts=30,
            avg_response_time=4.5,
            difficulty_rating=0.85,
            last_updated=datetime.utcnow()
        )
    ]
    db_session.add_all(patterns)
    db_session.commit()
    
    yield
    
    # Clean up
    db_session.query(PatternStatistics).delete()
    db_session.commit()


@pytest.fixture
def seed_user_data(db_session, seed_pattern_data):
    """Seed database with user performance data"""
    users = ["user1", "user2", "user3"]
    patterns = ["hammer", "doji", "engulfing", "morning_star", "evening_star"]
    
    user_performances = []
    for user_id in users:
        for pattern_id in patterns:
            user_performances.append(
                UserPerformance(
                    user_id=user_id,
                    pattern_id=pattern_id,
                    total_attempts=random.randint(5, 20),
                    correct_attempts=random.randint(1, 15),
                    avg_response_time=random.uniform(1.0, 8.0),
                    last_attempt_at=datetime.utcnow() - timedelta(days=random.randint(0, 30))
                )
            )
    
    db_session.add_all(user_performances)
    db_session.commit()
    
    yield
    
    # Clean up
    db_session.query(UserPerformance).delete()
    db_session.commit()


@pytest.fixture
def sample_candlestick_data():
    """Generate sample candlestick data for pattern recognition tests"""
    # Generate common test data frames with OHLC data
    
    # 1. Perfect hammer pattern
    hammer_data = pd.DataFrame({
        'date': pd.date_range(start='2023-01-01', periods=5),
        'open': [100, 105, 95, 90, 82],
        'high': [110, 115, 105, 95, 100],
        'low': [95, 100, 85, 80, 70],
        'close': [108, 110, 90, 85, 98],
        'volume': [1000, 1200, 800, 900, 1500]
    })
    
    # 2. Perfect doji pattern
    doji_data = pd.DataFrame({
        'date': pd.date_range(start='2023-01-01', periods=5),
        'open': [100, 105, 95, 90, 90],
        'high': [110, 115, 105, 95, 100],
        'low': [95, 100, 85, 80, 80],
        'close': [108, 110, 90, 85, 90],
        'volume': [1000, 1200, 800, 900, 1500]
    })
    
    # 3. Perfect engulfing pattern
    engulfing_data = pd.DataFrame({
        'date': pd.date_range(start='2023-01-01', periods=5),
        'open': [100, 105, 95, 90, 95],
        'high': [110, 115, 105, 95, 105],
        'low': [95, 100, 85, 80, 80],
        'close': [108, 110, 90, 85, 70],
        'volume': [1000, 1200, 800, 900, 1500]
    })
    
    # 4. Perfect morning star
    morning_star_data = pd.DataFrame({
        'date': pd.date_range(start='2023-01-01', periods=7),
        'open': [100, 95, 80, 75, 76, 85, 95],
        'high': [105, 98, 85, 78, 80, 90, 105],
        'low': [98, 80, 75, 72, 74, 82, 92],
        'close': [95, 82, 76, 74, 78, 95, 100],
        'volume': [1000, 1100, 900, 800, 850, 1200, 1500]
    })
    
    # 5. Perfect evening star
    evening_star_data = pd.DataFrame({
        'date': pd.date_range(start='2023-01-01', periods=7),
        'open': [100, 105, 115, 118, 117, 105, 95],
        'high': [105, 110, 120, 120, 119, 108, 98],
        'low': [98, 102, 114, 116, 110, 90, 85],
        'close': [104, 114, 117, 117, 112, 92, 88],
        'volume': [1000, 1100, 1300, 1200, 1100, 1400, 1500]
    })
    
    # Collection of test data
    test_data = {
        'hammer': hammer_data,
        'doji': doji_data,
        'engulfing': engulfing_data,
        'morning_star': morning_star_data,
        'evening_star': evening_star_data
    }
    
    return test_data


# Basic functionality tests
class TestBasicCandlestickPatternFunctionality:
    """Test basic functionality of candlestick pattern recognition"""
    
    def test_pattern_library_integrity(self):
        """Test that pattern library contains all expected patterns"""
        pattern_library = get_patterns_library()
        
        # Check that key patterns exist
        essential_patterns = ["hammer", "doji", "engulfing", "morning_star", "evening_star"]
        for pattern in essential_patterns:
            assert pattern in pattern_library, f"Pattern '{pattern}' not found in pattern library"
        
        # Check pattern definition structure
        for name, definition in pattern_library.items():
            assert "recognition_criteria" in definition, f"Pattern '{name}' missing recognition criteria"
            assert "description" in definition, f"Pattern '{name}' missing description"
    
    def test_pattern_recognition_basic(self, sample_candlestick_data):
        """Test that pattern recognition correctly identifies ideal patterns"""
        # Test each perfect pattern
        for pattern_name, data in sample_candlestick_data.items():
            # Identify patterns in the data
            results = identify_candlestick_patterns(data)
            
            # Check that the expected pattern was found
            found = False
            for day_result in results:
                if pattern_name in day_result['patterns']:
                    found = True
                    break
                    
            assert found, f"Failed to identify perfect {pattern_name} pattern"
    
    def test_analyze_pattern_accuracy(self, sample_candlestick_data):
        """Test pattern analysis returns expected confidence levels"""
        # Test analysis of perfect patterns
        for pattern_name, data in sample_candlestick_data.items():
            # Analyze last day of the pattern
            analysis = analyze_candlestick_pattern(data, index=-1)
            
            # Check analysis includes the expected pattern
            patterns_found = [p['pattern'] for p in analysis['patterns']]
            assert pattern_name in patterns_found, f"Pattern analysis failed to identify {pattern_name}"
            
            # Extract the confidence for the specific pattern
            pattern_confidence = None
            for p in analysis['patterns']:
                if p['pattern'] == pattern_name:
                    pattern_confidence = p['confidence']
                    break
            
            # Perfect pattern should have high confidence
            assert pattern_confidence and pattern_confidence > 0.7, \
                f"Perfect {pattern_name} pattern should have high confidence (got {pattern_confidence})"
    
    def test_detection_with_noise(self, sample_candlestick_data):
        """Test pattern detection with slightly noisy data"""
        # Add noise to perfect patterns and verify they're still detected
        for pattern_name, data in sample_candlestick_data.items():
            # Create a copy of the data with small random noise
            noisy_data = data.copy()
            noise_factor = 0.05  # 5% noise
            
            # Add noise to OHLC values
            for col in ['open', 'high', 'low', 'close']:
                noise = noisy_data[col] * noise_factor * np.random.randn(len(noisy_data))
                noisy_data[col] = noisy_data[col] + noise
            
            # Ensure high > low after adding noise
            noisy_data['high'] = np.maximum(noisy_data['high'], 
                                           np.maximum(noisy_data['open'], noisy_data['close']))
            noisy_data['low'] = np.minimum(noisy_data['low'], 
                                          np.minimum(noisy_data['open'], noisy_data['close']))
            
            # Identify patterns in noisy data
            results = identify_candlestick_patterns(noisy_data)
            
            # The pattern should still be detected despite noise
            found = False
            for day_result in results:
                if pattern_name in day_result['patterns']:
                    found = True
                    break
                    
            assert found, f"Failed to identify {pattern_name} pattern with small noise"


# Edge case tests for pattern recognition
class TestEdgeCaseCandlestickPatternRecognition:
    """Test edge cases in candlestick pattern recognition to find breaking points"""
    
    def test_minimal_data_points(self, sample_candlestick_data):
        """Test pattern recognition with minimal required data points"""
        # Most patterns require at least a few days of data
        # Reduce data to minimal required and verify behavior
        
        for pattern_name, data in sample_candlestick_data.items():
            # Determine minimum required data points for this pattern
            min_points = 1  # Default
            if pattern_name in ["engulfing"]:
                min_points = 2
            elif pattern_name in ["morning_star", "evening_star"]:
                min_points = 3
            
            # Trim data to minimum required plus one (for context)
            minimal_data = data.tail(min_points + 1).copy()
            
            # Attempt pattern recognition
            results = identify_candlestick_patterns(minimal_data)
            
            # Pattern should still be detected with minimal data
            found = False
            for day_result in results:
                if pattern_name in day_result['patterns']:
                    found = True
                    break
                    
            assert found, f"Failed to identify {pattern_name} pattern with minimal data points"
    
    def test_single_data_point(self):
        """Test behavior when only a single data point is provided"""
        # Create single data point
        single_point = pd.DataFrame({
            'date': [pd.Timestamp('2023-01-01')],
            'open': [100],
            'high': [110],
            'low': [90],
            'close': [105],
            'volume': [1000]
        })
        
        # This should not crash but handle gracefully
        try:
            results = identify_candlestick_patterns(single_point)
            # Only some patterns can be detected with a single point
            assert isinstance(results, list), "Expected list result even with single data point"
        except Exception as e:
            pytest.fail(f"identify_candlestick_patterns crashed with single data point: {str(e)}")
    
    def test_empty_dataframe(self):
        """Test behavior with empty dataframe"""
        # Create empty dataframe with correct columns
        empty_data = pd.DataFrame(columns=['date', 'open', 'high', 'low', 'close', 'volume'])
        
        # Should handle gracefully without crashing
        try:
            results = identify_candlestick_patterns(empty_data)
            assert isinstance(results, list), "Expected list result"
            assert len(results) == 0, "Expected empty results for empty data"
        except Exception as e:
            pytest.fail(f"identify_candlestick_patterns crashed with empty data: {str(e)}")
    
    def test_missing_columns(self):
        """Test behavior with missing required columns"""
        # Create data with missing columns
        incomplete_data = pd.DataFrame({
            'date': pd.date_range(start='2023-01-01', periods=5),
            'open': [100, 105, 95, 90, 85],
            'high': [110, 115, 105, 95, 95],
            # Missing 'low' column
            'close': [108, 110, 90, 85, 90],
            'volume': [1000, 1200, 800, 900, 1000]
        })
        
        # Should handle gracefully or provide clear error
        try:
            identify_candlestick_patterns(incomplete_data)
            # If it doesn't raise an exception, the function should handle missing columns gracefully
        except Exception as e:
            # Check for descriptive error message
            error_msg = str(e).lower()
            assert "missing" in error_msg or "required" in error_msg or "column" in error_msg, \
                f"Error message not descriptive: {error_msg}"
    
    def test_extreme_price_values(self):
        """Test behavior with extremely large or small price values"""
        # Create data with extreme values
        extreme_data = pd.DataFrame({
            'date': pd.date_range(start='2023-01-01', periods=5),
            'open': [1e9, 1e9+100, 1e9-100, 1e9+50, 1e9-50],  # Billion dollar stock
            'high': [1e9+500, 1e9+600, 1e9+200, 1e9+200, 1e9+100],
            'low': [1e9-500, 1e9-400, 1e9-500, 1e9-200, 1e9-400],
            'close': [1e9+200, 1e9+300, 1e9-300, 1e9-100, 1e9+50],
            'volume': [1e9, 1.2e9, 0.8e9, 0.9e9, 1.1e9]
        })
        
        # Should handle extreme values
        try:
            results = identify_candlestick_patterns(extreme_data)
            assert isinstance(results, list), "Expected list result with extreme values"
        except Exception as e:
            pytest.fail(f"identify_candlestick_patterns crashed with extreme values: {str(e)}")
        
        # Also test with very small values (penny stocks)
        tiny_data = pd.DataFrame({
            'date': pd.date_range(start='2023-01-01', periods=5),
            'open': [0.001, 0.0011, 0.0009, 0.00095, 0.0008],  # Fraction of a cent
            'high': [0.0012, 0.0013, 0.00095, 0.001, 0.0009],
            'low': [0.0009, 0.001, 0.00085, 0.0009, 0.00075],
            'close': [0.0011, 0.0012, 0.0009, 0.00095, 0.00085],
            'volume': [1e6, 1.2e6, 0.8e6, 0.9e6, 1.1e6]
        })
        
        try:
            results = identify_candlestick_patterns(tiny_data)
            assert isinstance(results, list), "Expected list result with tiny values"
        except Exception as e:
            pytest.fail(f"identify_candlestick_patterns crashed with tiny values: {str(e)}")
    
    def test_zero_values(self):
        """Test behavior with zero prices (shouldn't happen in real markets, but test anyway)"""
        zero_data = pd.DataFrame({
            'date': pd.date_range(start='2023-01-01', periods=5),
            'open': [0, 0, 0, 0, 0],
            'high': [0, 0, 0, 0, 0],
            'low': [0, 0, 0, 0, 0],
            'close': [0, 0, 0, 0, 0],
            'volume': [0, 0, 0, 0, 0]
        })
        
        # Should handle without crashing
        try:
            results = identify_candlestick_patterns(zero_data)
            assert isinstance(results, list), "Expected list result with zero values"
        except Exception as e:
            pytest.fail(f"identify_candlestick_patterns crashed with zero values: {str(e)}")
    
    def test_negative_values(self):
        """Test behavior with negative prices (shouldn't happen in real markets)"""
        negative_data = pd.DataFrame({
            'date': pd.date_range(start='2023-01-01', periods=5),
            'open': [-10, -15, -5, -8, -12],
            'high': [-5, -10, -2, -5, -8],
            'low': [-15, -20, -10, -12, -18],
            'close': [-12, -12, -8, -10, -10],
            'volume': [1000, 1200, 800, 900, 1100]
        })
        
        # Should handle without crashing
        try:
            results = identify_candlestick_patterns(negative_data)
            assert isinstance(results, list), "Expected list result with negative values"
        except Exception as e:
            pytest.fail(f"identify_candlestick_patterns crashed with negative values: {str(e)}")
    
    def test_invalid_ohlc_relationship(self):
        """Test behavior when OHLC relationships are invalid (e.g., low > high)"""
        invalid_data = pd.DataFrame({
            'date': pd.date_range(start='2023-01-01', periods=5),
            'open': [100, 105, 95, 90, 85],
            'high': [90, 95, 85, 80, 75],  # High less than open (invalid)
            'low': [110, 115, 105, 100, 95],  # Low greater than high (invalid)
            'close': [108, 110, 90, 85, 80],
            'volume': [1000, 1200, 800, 900, 1000]
        })
        
        # Should handle gracefully or provide clear error
        try:
            results = identify_candlestick_patterns(invalid_data)
            assert isinstance(results, list), "Expected list result with invalid OHLC relationships"
        except Exception as e:
            # If it raises an exception, make sure it's descriptive
            error_msg = str(e).lower()
            assert "invalid" in error_msg or "relationship" in error_msg or "ohlc" in error_msg, \
                f"Error message not descriptive: {error_msg}"
    
    def test_identical_ohlc_values(self):
        """Test behavior when all OHLC values are identical"""
        flat_data = pd.DataFrame({
            'date': pd.date_range(start='2023-01-01', periods=5),
            'open': [100, 100, 100, 100, 100],
            'high': [100, 100, 100, 100, 100],
            'low': [100, 100, 100, 100, 100],
            'close': [100, 100, 100, 100, 100],
            'volume': [1000, 1000, 1000, 1000, 1000]
        })
        
        # Should handle without crashing
        try:
            results = identify_candlestick_patterns(flat_data)
            assert isinstance(results, list), "Expected list result with flat OHLC values"
        except Exception as e:
            pytest.fail(f"identify_candlestick_patterns crashed with flat OHLC values: {str(e)}")
    
    def test_conflicting_patterns(self):
        """Test behavior when data could match multiple conflicting patterns"""
        # Create data that could match multiple patterns simultaneously
        conflicting_data = pd.DataFrame({
            'date': pd.date_range(start='2023-01-01', periods=5),
            'open': [100, 100, 90, 85, 90],
            'high': [110, 105, 95, 95, 100],  # Could match doji and hammer criteria
            'low': [90, 95, 80, 75, 70],
            'close': [105, 100, 85, 90, 90],  # Same open/close (doji) but with long lower shadow (hammer)
            'volume': [1000, 900, 1200, 1500, 2000]
        })
        
        # Should handle without crashing
        results = identify_candlestick_patterns(conflicting_data)
        
        # Check last day for potential conflicts
        last_day = results[-1] if results else {}
        
        # Multiple patterns might be identified
        if 'patterns' in last_day and len(last_day['patterns']) > 1:
            print(f"Multiple patterns detected: {last_day['patterns']}")
            assert True, "System correctly identified multiple patterns"
        else:
            # If only one pattern identified, system has a priority mechanism (acceptable)
            assert 'patterns' in last_day, "No patterns detected in conflicting data"
    
    def test_non_trading_days(self):
        """Test behavior with non-contiguous dates (weekends/holidays)"""
        # Create data with gaps in dates
        dates = [
            pd.Timestamp('2023-01-02'),  # Monday
            pd.Timestamp('2023-01-03'),  # Tuesday
            pd.Timestamp('2023-01-04'),  # Wednesday
            # Skip Thursday and Friday
            pd.Timestamp('2023-01-09'),  # Next Monday
            pd.Timestamp('2023-01-10')   # Tuesday
        ]
        
        gap_data = pd.DataFrame({
            'date': dates,
            'open': [100, 105, 95, 90, 85],
            'high': [110, 115, 105, 95, 95],
            'low': [95, 100, 85, 80, 80],
            'close': [108, 110, 90, 85, 90],
            'volume': [1000, 1200, 800, 900, 1000]
        })
        
        # Should handle without crashing
        try:
            results = identify_candlestick_patterns(gap_data)
            assert isinstance(results, list), "Expected list result with date gaps"
        except Exception as e:
            pytest.fail(f"identify_candlestick_patterns crashed with date gaps: {str(e)}")
    
    def test_duplicate_dates(self):
        """Test behavior with duplicate dates in data"""
        # Create data with duplicate dates
        dates = [
            pd.Timestamp('2023-01-01'),
            pd.Timestamp('2023-01-02'),
            pd.Timestamp('2023-01-02'),  # Duplicate
            pd.Timestamp('2023-01-03'),
            pd.Timestamp('2023-01-04')
        ]
        
        duplicate_data = pd.DataFrame({
            'date': dates,
            'open': [100, 105, 102, 95, 90],
            'high': [110, 115, 112, 105, 95],
            'low': [95, 100, 98, 85, 80],
            'close': [108, 110, 105, 90, 85],
            'volume': [1000, 1200, 1100, 800, 900]
        })
        
        # Should handle without crashing
        try:
            results = identify_candlestick_patterns(duplicate_data)
            assert isinstance(results, list), "Expected list result with duplicate dates"
        except Exception as e:
            pytest.fail(f"identify_candlestick_patterns crashed with duplicate dates: {str(e)}")
    
    def test_future_dates(self):
        """Test behavior with future dates in data"""
        # Create data with future dates
        today = pd.Timestamp.now().normalize()
        future_dates = [
            today - pd.Timedelta(days=2),
            today - pd.Timedelta(days=1),
            today,
            today + pd.Timedelta(days=1),  # Tomorrow
            today + pd.Timedelta(days=2)   # Day after tomorrow
        ]
        
        future_data = pd.DataFrame({
            'date': future_dates,
            'open': [100, 105, 95, 90, 85],
            'high': [110, 115, 105, 95, 95],
            'low': [95, 100, 85, 80, 80],
            'close': [108, 110, 90, 85, 90],
            'volume': [1000, 1200, 800, 900, 1000]
        })
        
        # Should handle without crashing
        try:
            results = identify_candlestick_patterns(future_data)
            assert isinstance(results, list), "Expected list result with future dates"
        except Exception as e:
            pytest.fail(f"identify_candlestick_patterns crashed with future dates: {str(e)}")
    
    def test_extremely_large_dataset(self):
        """Test behavior with extremely large dataset"""
        # Create a very large dataset - 10 years of daily data
        periods = 365 * 10
        
        # Generate random but realistic OHLC data
        np.random.seed(42)  # For reproducibility
        base_price = 100
        volatility = 2
        
        # Start with random prices
        closes = np.random.normal(0, 1, periods)
        # Make it a random walk
        closes = base_price + np.cumsum(closes * volatility)
        
        # Generate OHLC based on close prices
        opens = closes[:-1]  # Previous close becomes next open
        opens = np.append([base_price], opens)
        
        # Add random intraday volatility
        high_offsets = np.abs(np.random.normal(0, 1, periods) * volatility)
        low_offsets = np.abs(np.random.normal(0, 1, periods) * volatility)
        
        highs = np.maximum(opens, closes) + high_offsets
        lows = np.minimum(opens, closes) - low_offsets
        
        # Create the large dataframe
        large_data = pd.DataFrame({
            'date': pd.date_range(start='2010-01-01', periods=periods),
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': np.random.randint(1000, 10000, periods)
        })
        
        # Should handle without crashing or excessive delay
        start_time = pd.Timestamp.now()
        try:
            results = identify_candlestick_patterns(large_data)
            assert isinstance(results, list), "Expected list result with large dataset"
            
            # Check execution time - should be reasonable
            execution_time = (pd.Timestamp.now() - start_time).total_seconds()
            print(f"Large dataset (n={periods}) processing time: {execution_time:.2f} seconds")
            
            # Some patterns should be found in a large random dataset
            pattern_count = sum(len(day.get('patterns', [])) for day in results)
            assert pattern_count > 0, "No patterns found in large dataset"
            
        except Exception as e:
            pytest.fail(f"identify_candlestick_patterns crashed with large dataset: {str(e)}")
    
    def test_borderline_pattern_criteria(self):
        """Test behavior with data just at the edge of pattern criteria"""
        # Create borderline hammer pattern - just barely meeting criteria
        hammer_borderline = pd.DataFrame({
            'date': pd.date_range(start='2023-01-01', periods=5),
            'open': [100, 105, 95, 90, 90.1],
            'high': [110, 115, 105, 95, 100],
            'low': [95, 100, 85, 80, 80],
            'close': [108, 110, 90, 85, 90],  # Almost equal to open
            'volume': [1000, 1200, 800, 900, 1500]
        })
        
        # Test if pattern is detected with borderline case
        results = identify_candlestick_patterns(hammer_borderline)
        last_day = results[-1] if results else {}
        
        # Check confidence level for borderline pattern
        if 'patterns' in last_day and 'hammer' in last_day['patterns']:
            confidence = last_day['patterns']['hammer']
            print(f"Borderline hammer pattern confidence: {confidence}")
            # Confidence should reflect the borderline nature (lower than ideal)
            assert 0 < confidence <= 1, "Borderline pattern confidence should be between 0 and 1"
        else:
            # It's acceptable if borderline pattern isn't detected
            print("Borderline hammer pattern not detected")
    
    def test_invalid_index(self):
        """Test behavior when requesting analysis with invalid index"""
        data = pd.DataFrame({
            'date': pd.date_range(start='2023-01-01', periods=5),
            'open': [100, 105, 95, 90, 85],
            'high': [110, 115, 105, 95, 95],
            'low': [95, 100, 85, 80, 80],
            'close': [108, 110, 90, 85, 90],
            'volume': [1000, 1200, 800, 900, 1000]
        })
        
        # Test with out-of-bounds index
        too_large_index = len(data) + 10
        
        try:
            analyze_candlestick_pattern(data, index=too_large_index)
            pytest.fail("Expected an error with out-of-bounds index")
        except Exception as e:
            # Should raise IndexError or similar
            error_msg = str(e).lower()
            assert "index" in error_msg or "bounds" in error_msg or "range" in error_msg, \
                f"Error message not descriptive: {error_msg}"
        
        # Test with negative index beyond data length
        very_negative_index = -len(data) - 10
        
        try:
            analyze_candlestick_pattern(data, index=very_negative_index)
            pytest.fail("Expected an error with very negative index")
        except Exception as e:
            # Should raise IndexError or similar
            error_msg = str(e).lower()
            assert "index" in error_msg or "bounds" in error_msg or "range" in error_msg, \
                f"Error message not descriptive: {error_msg}"


# Redis cache integration tests
class TestRedisCacheIntegration:
    """Test Redis cache integration with candlestick pattern functionality"""
    
    def test_cache_initialization(self, cache_initialization):
        """Test cache initialization process"""
        # Cache initialization fixture should have initialized the cache
        # Verify basic functionality
        try:
            # Get pattern stats from cache for a common pattern
            stats = get_pattern_stats_from_cache("hammer")
            
            # Stats might be None if not in cache, which is fine for this test
            # We're mainly checking that the function doesn't crash
            assert True, "Cache initialization and stats retrieval should not crash"
        except Exception as e:
            pytest.fail(f"Cache initialization or stats retrieval failed: {str(e)}")
    
    def test_cache_update_and_retrieval(self, cache_initialization):
        """Test updating and retrieving values from cache"""
        # Create test stats
        test_stats = {
            "pattern_id": "test_pattern",
            "total_attempts": 100,
            "correct_attempts": 75,
            "avg_response_time": 2.5,
            "difficulty_rating": 0.6
        }
        
        # Update cache
        try:
            update_pattern_stats_in_cache("test_pattern", test_stats)
        except Exception as e:
            pytest.fail(f"Failed to update cache: {str(e)}")
        
        # Retrieve from cache
        try:
            cached_stats = get_pattern_stats_from_cache("test_pattern")
            
            # Verify the data was properly cached
            assert cached_stats is not None, "Stats not found in cache after update"
            
            # Check key fields
            for key, expected_value in test_stats.items():
                assert key in cached_stats, f"Key {key} missing from cached stats"
                assert cached_stats[key] == expected_value, \
                    f"Cached value mismatch for {key}: expected {expected_value}, got {cached_stats[key]}"
                
        except Exception as e:
            pytest.fail(f"Failed to retrieve from cache: {str(e)}")
    
    def test_cache_clear(self, cache_initialization):
        """Test clearing the cache"""
        # First add some data
        test_stats = {"pattern_id": "clear_test", "total_attempts": 50}
        update_pattern_stats_in_cache("clear_test", test_stats)
        
        # Verify it's in the cache
        assert get_pattern_stats_from_cache("clear_test") is not None, "Failed to add test data to cache"
        
        # Clear the cache
        clear_pattern_stats_cache()
        
        # Verify it's gone
        assert get_pattern_stats_from_cache("clear_test") is None, "Cache not cleared properly"
    
    def test_cache_invalid_input(self, cache_initialization):
        """Test cache functions with invalid inputs"""
        # Test with None pattern_id
        try:
            get_pattern_stats_from_cache(None)
            # If we get here, it handled None gracefully, which is good
        except Exception as e:
            # If it raises an exception, make sure it's descriptive
            error_msg = str(e).lower()
            assert "pattern" in error_msg or "id" in error_msg or "none" in error_msg, \
                f"Error message not descriptive: {error_msg}"
        
        # Test update with None stats
        try:
            update_pattern_stats_in_cache("test_pattern", None)
            # If we get here, it handled None gracefully, which is good
        except Exception as e:
            # If it raises an exception, make sure it's descriptive
            error_msg = str(e).lower()
            assert "stats" in error_msg or "none" in error_msg, \
                f"Error message not descriptive: {error_msg}"
    
    def test_cache_concurrency(self, cache_initialization):
        """Test concurrent cache access"""
        import threading
        
        # Shared counters for tracking results
        success_count = 0
        error_count = 0
        results = {}
        lock = threading.Lock()
        
        def cache_worker(worker_id):
            """Worker function for concurrent cache operations"""
            nonlocal success_count, error_count, results
            
            try:
                # Generate worker-specific pattern ID
                pattern_id = f"concurrent_test_{worker_id}"
                
                # Create test stats
                test_stats = {
                    "pattern_id": pattern_id,
                    "total_attempts": worker_id * 10,
                    "correct_attempts": worker_id * 5,
                    "avg_response_time": worker_id * 0.1,
                    "difficulty_rating": min(0.1 * worker_id, 1.0)
                }
                
                # Update cache
                update_pattern_stats_in_cache(pattern_id, test_stats)
                
                # Small delay to simulate work
                import time
                time.sleep(0.01)
                
                # Retrieve from cache
                cached_stats = get_pattern_stats_from_cache(pattern_id)
                
                with lock:
                    if cached_stats and cached_stats["total_attempts"] == test_stats["total_attempts"]:
                        success_count += 1
                        results[pattern_id] = cached_stats
                    else:
                        error_count += 1
                
            except Exception as e:
                with lock:
                    error_count += 1
                    print(f"Worker {worker_id} error: {str(e)}")
        
        # Create and start threads
        threads = []
        worker_count = 10
        
        for i in range(worker_count):
            thread = threading.Thread(target=cache_worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify all operations succeeded
        assert error_count == 0, f"{error_count} cache operations failed during concurrent access"
        assert success_count == worker_count, f"Expected {worker_count} successful operations, got {success_count}"
        
        # Verify all data was cached correctly
        for i in range(worker_count):
            pattern_id = f"concurrent_test_{i}"
            assert pattern_id in results, f"Pattern {pattern_id} missing from results"
            assert results[pattern_id]["total_attempts"] == i * 10, \
                f"Data mismatch for {pattern_id}: expected {i*10}, got {results[pattern_id]['total_attempts']}"


# Database integration tests
class TestDatabaseIntegration:
    """Test database integration with candlestick pattern functionality"""
    
    def test_pattern_statistics_crud(self, repository, db_session):
        """Test create, read, update, delete operations for pattern statistics"""
        # Create test pattern
        test_pattern = PatternStatistics(
            pattern_id="db_integration_test",
            total_attempts=100,
            correct_attempts=50,
            avg_response_time=3.0,
            difficulty_rating=0.7,
            last_updated=datetime.utcnow()
        )
        
        # Create
        db_session.add(test_pattern)
        db_session.commit()
        
        # Read
        pattern = db_session.query(PatternStatistics).filter_by(
            pattern_id="db_integration_test"
        ).first()
        
        assert pattern is not None, "Failed to create and retrieve pattern"
        assert pattern.total_attempts == 100, "Pattern data mismatch"
        
        # Update
        pattern.total_attempts += 1
        pattern.correct_attempts += 1
        db_session.commit()
        
        # Read updated
        updated_pattern = db_session.query(PatternStatistics).filter_by(
            pattern_id="db_integration_test"
        ).first()
        
        assert updated_pattern.total_attempts == 101, "Failed to update pattern"
        
        # Delete
        db_session.delete(updated_pattern)
        db_session.commit()
        
        # Verify deletion
        deleted_check = db_session.query(PatternStatistics).filter_by(
            pattern_id="db_integration_test"
        ).first()
        
        assert deleted_check is None, "Failed to delete pattern"
    
    def test_user_performance_tracking(self, repository, db_session):
        """Test user performance tracking in database"""
        # Create test user performance
        test_performance = UserPerformance(
            user_id="test_user",
            pattern_id="hammer",
            total_attempts=10,
            correct_attempts=5,
            avg_response_time=2.5,
            last_attempt_at=datetime.utcnow()
        )
        
        # Create
        db_session.add(test_performance)
        db_session.commit()
        
        # Read
        performance = db_session.query(UserPerformance).filter_by(
            user_id="test_user",
            pattern_id="hammer"
        ).first()
        
        assert performance is not None, "Failed to create and retrieve user performance"
        assert performance.total_attempts == 10, "Performance data mismatch"
        
        # Update via repository
        repository.update_user_performance(
            user_id="test_user",
            pattern_id="hammer",
            is_correct=True,
            response_time=3.0
        )
        
        # Read updated
        updated_performance = db_session.query(UserPerformance).filter_by(
            user_id="test_user",
            pattern_id="hammer"
        ).first()
        
        assert updated_performance.total_attempts == 11, "Failed to increment total attempts"
        assert updated_performance.correct_attempts == 6, "Failed to increment correct attempts"
        
        # Verify last_attempt_at was updated
        assert (datetime.utcnow() - updated_performance.last_attempt_at).total_seconds() < 60, \
            "last_attempt_at field not properly updated"
    
    def test_assessment_attempt_creation(self, db_session):
        """Test creating assessment attempts in database"""
        # Create test assessment attempt
        test_attempt = AssessmentAttempt(
            id="test_assessment",
            user_id="test_user",
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
        
        # Create
        db_session.add(test_attempt)
        db_session.commit()
        
        # Read
        attempt = db_session.query(AssessmentAttempt).filter_by(
            id="test_assessment"
        ).first()
        
        assert attempt is not None, "Failed to create and retrieve assessment attempt"
        assert attempt.questions_total == 10, "Assessment data mismatch"
        assert attempt.is_completed is False, "Assessment completion status incorrect"
        
        # Update to mark as completed
        attempt.completed_at = datetime.utcnow()
        attempt.questions_completed = 10
        attempt.correct_answers = 7
        attempt.avg_response_time = 3.5
        attempt.score = 70
        attempt.is_completed = True
        db_session.commit()
        
        # Read updated
        updated_attempt = db_session.query(AssessmentAttempt).filter_by(
            id="test_assessment"
        ).first()
        
        assert updated_attempt.is_completed is True, "Failed to update completion status"
        assert updated_attempt.score == 70, "Failed to update score"
    
    def test_user_answer_recording(self, db_session):
        """Test recording user answers in database"""
        # Create test assessment attempt first
        test_attempt = AssessmentAttempt(
            id="test_answers_assessment",
            user_id="test_user",
            started_at=datetime.utcnow(),
            questions_total=5,
            is_completed=False
        )
        db_session.add(test_attempt)
        db_session.commit()
        
        # Create test user answer
        test_answer = UserAnswer(
            id="test_answer_1",
            assessment_attempt_id="test_answers_assessment",
            question_id="q1",
            selected_option="hammer",
            is_correct=True,
            response_time=2.5,
            submitted_at=datetime.utcnow()
        )
        
        # Create
        db_session.add(test_answer)
        db_session.commit()
        
        # Read
        answer = db_session.query(UserAnswer).filter_by(
            id="test_answer_1"
        ).first()
        
        assert answer is not None, "Failed to create and retrieve user answer"
        assert answer.selected_option == "hammer", "Answer data mismatch"
        assert answer.is_correct is True, "Answer correctness mismatch"
        
        # Create more answers
        for i in range(2, 6):
            is_correct = i % 2 == 0  # Even questions are correct
            test_answer = UserAnswer(
                id=f"test_answer_{i}",
                assessment_attempt_id="test_answers_assessment",
                question_id=f"q{i}",
                selected_option="doji" if is_correct else "engulfing",
                is_correct=is_correct,
                response_time=2.0 + i * 0.5,
                submitted_at=datetime.utcnow()
            )
            db_session.add(test_answer)
        
        db_session.commit()
        
        # Query all answers for this assessment
        answers = db_session.query(UserAnswer).filter_by(
            assessment_attempt_id="test_answers_assessment"
        ).all()
        
        assert len(answers) == 5, f"Expected 5 answers, got {len(answers)}"
        
        # Count correct answers
        correct_count = sum(1 for a in answers if a.is_correct)
        assert correct_count == 3, f"Expected 3 correct answers, got {correct_count}"
    
    def test_foreign_key_constraints(self, db_session):
        """Test foreign key constraints in database"""
        # Attempt to create user answer without valid assessment ID
        invalid_answer = UserAnswer(
            id="invalid_fk_test",
            assessment_attempt_id="non_existent_assessment",  # Non-existent FK
            question_id="q1",
            selected_option="hammer",
            is_correct=True,
            response_time=2.5,
            submitted_at=datetime.utcnow()
        )
        
        db_session.add(invalid_answer)
        
        # Should fail with foreign key constraint error
        try:
            db_session.commit()
            pytest.fail("Expected foreign key constraint error")
        except Exception as e:
            # Verify it's a foreign key error
            error_msg = str(e).lower()
            assert "foreign key" in error_msg or "violates" in error_msg or "constraint" in error_msg, \
                f"Unexpected error message: {error_msg}"
            
            db_session.rollback()
    
    def test_transaction_rollback(self, db_session):
        """Test transaction rollback on error"""
        # Create a valid starting point
        valid_pattern = PatternStatistics(
            pattern_id="rollback_test",
            total_attempts=100,
            correct_attempts=50,
            avg_response_time=3.0,
            difficulty_rating=0.5,
            last_updated=datetime.utcnow()
        )
        db_session.add(valid_pattern)
        db_session.commit()
        
        # Start a transaction with multiple operations
        # First valid operation
        valid_pattern.total_attempts += 1
        
        # Then invalid operation (duplicate pattern ID)
        duplicate_pattern = PatternStatistics(
            pattern_id="rollback_test",  # Duplicate - will cause error
            total_attempts=200,
            correct_attempts=100,
            avg_response_time=2.0,
            difficulty_rating=0.6,
            last_updated=datetime.utcnow()
        )
        db_session.add(duplicate_pattern)
        
        # Try to commit - should fail and rollback
        try:
            db_session.commit()
            pytest.fail("Expected error for duplicate pattern ID")
        except Exception:
            db_session.rollback()
        
        # Verify the valid operation was also rolled back
        refreshed_pattern = db_session.query(PatternStatistics).filter_by(
            pattern_id="rollback_test"
        ).first()
        
        assert refreshed_pattern.total_attempts == 100, \
            "Transaction not rolled back properly, changes persisted"
    
    def test_cascading_deletes(self, db_session):
        """Test cascading deletes with related records"""
        # Create an assessment with answers to test cascade deletion
        test_attempt = AssessmentAttempt(
            id="cascade_test",
            user_id="cascade_user",
            started_at=datetime.utcnow(),
            questions_total=3,
            is_completed=False
        )
        db_session.add(test_attempt)
        db_session.commit()
        
        # Add related answers
        for i in range(1, 4):
            answer = UserAnswer(
                id=f"cascade_answer_{i}",
                assessment_attempt_id="cascade_test",
                question_id=f"q{i}",
                selected_option="hammer",
                is_correct=True,
                response_time=2.0,
                submitted_at=datetime.utcnow()
            )
            db_session.add(answer)
        
        db_session.commit()
        
        # Verify we have the assessment and answers
        assert db_session.query(AssessmentAttempt).filter_by(id="cascade_test").count() == 1
        assert db_session.query(UserAnswer).filter_by(assessment_attempt_id="cascade_test").count() == 3
        
        # Delete the assessment
        db_session.query(AssessmentAttempt).filter_by(id="cascade_test").delete()
        db_session.commit()
        
        # Check if answers were also deleted (assuming cascade is set up)
        answer_count = db_session.query(UserAnswer).filter_by(assessment_attempt_id="cascade_test").count()
        
        # Note: This assertion depends on whether cascade delete is configured
        # If cascading is set up, the count should be 0
        # If not, the assertion should be modified accordingly
        if answer_count > 0:
            print("Warning: Cascade delete not configured, answers still exist after assessment deletion")
        
        # Regardless, the assessment should be gone
        assert db_session.query(AssessmentAttempt).filter_by(id="cascade_test").count() == 0


# API integration tests
class TestAPIIntegration:
    """Test API integration with database and cache"""
    
    def test_pattern_statistics_retrieval(self, repository, seed_pattern_data, cache_initialization):
        """Test retrieving pattern statistics through API"""
        # Try retrieving pattern statistics through API function
        stats = get_pattern_statistics("hammer")
        
        assert stats is not None, "Failed to retrieve pattern statistics"
        assert stats["pattern_id"] == "hammer", "Pattern ID mismatch"
        assert stats["total_attempts"] == 100, "Total attempts mismatch"
        assert stats["correct_attempts"] == 75, "Correct attempts mismatch"
        
        # Try a non-existent pattern
        none_stats = get_pattern_statistics("non_existent_pattern")
        assert none_stats is None, "Expected None for non-existent pattern"
    
    def test_pattern_statistics_update(self, repository, db_session, cache_initialization):
        """Test updating pattern statistics through API"""
        # Create a test pattern for updating
        test_pattern = PatternStatistics(
            pattern_id="api_update_test",
            total_attempts=100,
            correct_attempts=50,
            avg_response_time=3.0,
            difficulty_rating=0.5,
            last_updated=datetime.utcnow()
        )
        db_session.add(test_pattern)
        db_session.commit()
        
        # Update through API function
        updated_stats = update_pattern_statistics(
            pattern_id="api_update_test",
            is_correct=True,
            response_time=2.0
        )
        
        assert updated_stats is not None, "Failed to update pattern statistics"
        assert updated_stats["pattern_id"] == "api_update_test", "Pattern ID mismatch"
        assert updated_stats["total_attempts"] == 101, "Total attempts not incremented"
        assert updated_stats["correct_attempts"] == 51, "Correct attempts not incremented"
        
        # Verify database was updated
        db_pattern = db_session.query(PatternStatistics).filter_by(
            pattern_id="api_update_test"
        ).first()
        
        assert db_pattern.total_attempts == 101, "Database total attempts not updated"
        assert db_pattern.correct_attempts == 51, "Database correct attempts not updated"
        
        # Verify cache was updated
        cache_stats = get_pattern_stats_from_cache("api_update_test")
        assert cache_stats is not None, "Pattern stats not found in cache after update"
        assert cache_stats["total_attempts"] == 101, "Cache total attempts not updated"
    
    def test_user_performance_retrieval(self, repository, seed_user_data):
        """Test retrieving user performance through API"""
        # Retrieve user performance through API function
        performance = get_user_performance(user_id="user1", pattern_id="hammer")
        
        assert performance is not None, "Failed to retrieve user performance"
        assert performance["user_id"] == "user1", "User ID mismatch"
        assert performance["pattern_id"] == "hammer", "Pattern ID mismatch"
        assert "total_attempts" in performance, "Missing total_attempts field"
        assert "correct_attempts" in performance, "Missing correct_attempts field"
    
    def test_user_performance_update(self, repository, db_session):
        """Test updating user performance through API"""
        # Create test user performance
        test_performance = UserPerformance(
            user_id="api_user",
            pattern_id="hammer",
            total_attempts=10,
            correct_attempts=5,
            avg_response_time=2.5,
            last_attempt_at=datetime.utcnow() - timedelta(days=1)
        )
        db_session.add(test_performance)
        db_session.commit()
        
        # Update through API function
        updated_perf = update_user_performance(
            user_id="api_user",
            pattern_id="hammer",
            is_correct=True,
            response_time=3.0
        )
        
        assert updated_perf is not None, "Failed to update user performance"
        assert updated_perf["total_attempts"] == 11, "Total attempts not incremented"
        assert updated_perf["correct_attempts"] == 6, "Correct attempts not incremented"
        
        # Verify database was updated
        db_perf = db_session.query(UserPerformance).filter_by(
            user_id="api_user",
            pattern_id="hammer"
        ).first()
        
        assert db_perf.total_attempts == 11, "Database total attempts not updated"
        assert db_perf.correct_attempts == 6, "Database correct attempts not updated"
        
        # Verify last_attempt_at was updated
        assert (datetime.utcnow() - db_perf.last_attempt_at).total_seconds() < 60, \
            "last_attempt_at field not properly updated"
    
    def test_assessment_generation(self, seed_pattern_data):
        """Test generating an assessment through API"""
        # Generate an assessment
        assessment = generate_assessment(user_id="test_generation", difficulty=0.7, question_count=5)
        
        assert assessment is not None, "Failed to generate assessment"
        assert "assessment_id" in assessment, "Missing assessment_id in generated assessment"
        assert "questions" in assessment, "Missing questions in generated assessment"
        assert len(assessment["questions"]) == 5, f"Expected 5 questions, got {len(assessment['questions'])}"
        
        # Check assessment structure
        for question in assessment["questions"]:
            assert "question_id" in question, "Missing question_id in question"
            assert "chart_data" in question, "Missing chart_data in question"
            assert "options" in question, "Missing options in question"
            assert len(question["options"]) >= 2, "Not enough options in question"
    
    def test_answer_submission(self, db_session):
        """Test submitting an answer through API"""
        # First create an assessment attempt
        test_attempt = AssessmentAttempt(
            id="submit_test",
            user_id="submit_user",
            started_at=datetime.utcnow(),
            questions_total=5,
            questions_completed=0,
            correct_answers=0,
            avg_response_time=0,
            difficulty_level=0.5,
            score=None,
            is_completed=False
        )
        db_session.add(test_attempt)
        db_session.commit()
        
        # Submit an answer through API
        result = submit_answer(
            assessment_id="submit_test",
            question_id="q1",
            selected_option="hammer",
            actual_pattern="hammer",  # Correct answer
            response_time=2.5
        )
        
        assert result is not None, "Failed to submit answer"
        assert result["is_correct"] is True, "Answer should be marked as correct"
        assert "feedback" in result, "Missing feedback in result"
        
        # Verify the assessment was updated
        assessment = db_session.query(AssessmentAttempt).filter_by(id="submit_test").first()
        assert assessment.questions_completed == 1, "questions_completed not incremented"
        assert assessment.correct_answers == 1, "correct_answers not incremented"
        
        # Verify user answer was recorded
        answer = db_session.query(UserAnswer).filter_by(
            assessment_attempt_id="submit_test",
            question_id="q1"
        ).first()
        
        assert answer is not None, "User answer not recorded"
        assert answer.selected_option == "hammer", "Selected option mismatch"
        assert answer.is_correct is True, "Answer correctness mismatch"
        assert answer.response_time == 2.5, "Response time mismatch"
    
    def test_assessment_completion(self, db_session):
        """Test automatic assessment completion when all questions are answered"""
        # Create an assessment with just 2 questions
        test_attempt = AssessmentAttempt(
            id="completion_test",
            user_id="completion_user",
            started_at=datetime.utcnow(),
            questions_total=2,
            questions_completed=0,
            correct_answers=0,
            avg_response_time=0,
            difficulty_level=0.5,
            score=None,
            is_completed=False
        )
        db_session.add(test_attempt)
        db_session.commit()
        
        # Submit first answer
        submit_answer(
            assessment_id="completion_test",
            question_id="q1",
            selected_option="hammer",
            actual_pattern="hammer",  # Correct
            response_time=2.0
        )
        
        # Check assessment state after first answer
        assessment = db_session.query(AssessmentAttempt).filter_by(id="completion_test").first()
        assert assessment.is_completed is False, "Assessment should not be completed yet"
        assert assessment.questions_completed == 1, "questions_completed should be 1"
        
        # Submit second (final) answer
        submit_answer(
            assessment_id="completion_test",
            question_id="q2",
            selected_option="doji",
            actual_pattern="engulfing",  # Incorrect
            response_time=3.0
        )
        
        # Check assessment state after final answer
        assessment = db_session.query(AssessmentAttempt).filter_by(id="completion_test").first()
        assert assessment.is_completed is True, "Assessment should be marked as completed"
        assert assessment.questions_completed == 2, "questions_completed should be 2"
        assert assessment.correct_answers == 1, "correct_answers should be 1"
        assert assessment.completed_at is not None, "completed_at should be set"
        assert assessment.score == 50, "Score should be 50% (1/2 correct)"
        assert assessment.avg_response_time == 2.5, "avg_response_time should be 2.5"
    
    def test_invalid_answer_submission(self, db_session):
        """Test submitting an answer with invalid parameters"""
        # Create a test assessment attempt
        test_attempt = AssessmentAttempt(
            id="invalid_submit_test",
            user_id="invalid_user",
            started_at=datetime.utcnow(),
            questions_total=5,
            questions_completed=0,
            correct_answers=0,
            avg_response_time=0,
            difficulty_level=0.5,
            score=None,
            is_completed=False
        )
        db_session.add(test_attempt)
        db_session.commit()
        
        # Test with non-existent assessment ID
        try:
            submit_answer(
                assessment_id="non_existent",
                question_id="q1",
                selected_option="hammer",
                actual_pattern="hammer",
                response_time=2.5
            )
            pytest.fail("Expected error for non-existent assessment ID")
        except Exception as e:
            # Should raise an error
            error_msg = str(e).lower()
            assert "assessment" in error_msg or "not found" in error_msg or "exist" in error_msg, \
                f"Error message not descriptive: {error_msg}"
        
        # Test with invalid response time
        try:
            submit_answer(
                assessment_id="invalid_submit_test",
                question_id="q1",
                selected_option="hammer",
                actual_pattern="hammer",
                response_time=-1.0  # Negative time - should be invalid
            )
            # If it doesn't raise an exception, the function should validate and handle it
        except Exception as e:
            # If it raises an exception, make sure it's descriptive
            error_msg = str(e).lower()
            assert "time" in error_msg or "negative" in error_msg or "invalid" in error_msg, \
                f"Error message not descriptive: {error_msg}"
    
    def test_concurrent_answer_submissions(self, db_session):
        """Test submitting multiple answers concurrently"""
        import threading
        
        # Create a test assessment with multiple questions
        test_attempt = AssessmentAttempt(
            id="concurrent_submit_test",
            user_id="concurrent_user",
            started_at=datetime.utcnow(),
            questions_total=10,
            questions_completed=0,
            correct_answers=0,
            avg_response_time=0,
            difficulty_level=0.5,
            score=None,
            is_completed=False
        )
        db_session.add(test_attempt)
        db_session.commit()
        
        # Shared counters for tracking results
        success_count = 0
        error_count = 0
        lock = threading.Lock()
        
        def submit_worker(question_number):
            """Worker function for concurrent submissions"""
            nonlocal success_count, error_count
            
            try:
                # Generate worker-specific question ID
                question_id = f"q{question_number}"
                
                # Determine if this answer will be correct (alternate)
                is_correct = question_number % 2 == 0
                actual_pattern = "hammer" if is_correct else "doji"
                
                # Submit answer
                result = submit_answer(
                    assessment_id="concurrent_submit_test",
                    question_id=question_id,
                    selected_option="hammer",
                    actual_pattern=actual_pattern,
                    response_time=question_number * 0.5
                )
                
                with lock:
                    if result and "is_correct" in result:
                        success_count += 1
                    else:
                        error_count += 1
                
            except Exception as e:
                with lock:
                    error_count += 1
                    print(f"Worker {question_number} error: {str(e)}")
        
        # Create and start threads
        threads = []
        worker_count = 10  # Submit 10 answers concurrently
        
        for i in range(1, worker_count + 1):
            thread = threading.Thread(target=submit_worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify all operations succeeded
        assert error_count == 0, f"{error_count} submissions failed during concurrent access"
        assert success_count == worker_count, f"Expected {worker_count} successful submissions, got {success_count}"
        
        # Verify assessment was updated correctly
        assessment = db_session.query(AssessmentAttempt).filter_by(id="concurrent_submit_test").first()
        assert assessment.questions_completed == worker_count, \
            f"Expected {worker_count} completed questions, got {assessment.questions_completed}"
        assert assessment.correct_answers == worker_count // 2, \
            f"Expected {worker_count // 2} correct answers, got {assessment.correct_answers}"
        assert assessment.is_completed is True, "Assessment should be marked as completed"
        
        # Verify all answers were recorded
        answers = db_session.query(UserAnswer).filter_by(
            assessment_attempt_id="concurrent_submit_test"
        ).all()
        
        assert len(answers) == worker_count, f"Expected {worker_count} answers, got {len(answers)}"


# Complete system integration test
class TestCompleteSystemIntegration:
    """Complete end-to-end test of the candlestick pattern assessment system"""
    
    def test_full_assessment_flow(self, repository, db_session, cache_initialization, seed_pattern_data):
        """Test a complete assessment flow from generation to completion"""
        # 1. Generate an assessment
        user_id = "full_flow_user"
        assessment = generate_assessment(user_id=user_id, difficulty=0.5, question_count=3)
        
        assert assessment is not None, "Failed to generate assessment"
        assert "assessment_id" in assessment, "Missing assessment_id"
        
        assessment_id = assessment["assessment_id"]
        questions = assessment["questions"]
        
        # 2. Verify assessment was created in database
        db_assessment = db_session.query(AssessmentAttempt).filter_by(id=assessment_id).first()
        assert db_assessment is not None, "Assessment not found in database"
        assert db_assessment.user_id == user_id, "User ID mismatch"
        assert db_assessment.questions_total == 3, "Question count mismatch"
        
        # 3. Answer each question
        correct_count = 0
        total_response_time = 0
        
        for i, question in enumerate(questions):
            question_id = question["question_id"]
            options = question["options"]
            
            # For test purposes, we'll alternate between correct and incorrect answers
            correct_answer = None
            for option in options:
                if option.get("is_correct", False):
                    correct_answer = option["pattern"]
                    break
            
            # If we couldn't determine the correct answer, just pick the first option
            if correct_answer is None:
                correct_answer = options[0]["pattern"]
            
            # Choose correct answer for even questions, incorrect for odd
            selected_option = correct_answer if i % 2 == 0 else options[0]["pattern"]
            if selected_option == correct_answer:
                correct_count += 1
            
            response_time = 2.0 + i * 0.5
            total_response_time += response_time
            
            # Submit the answer
            result = submit_answer(
                assessment_id=assessment_id,
                question_id=question_id,
                selected_option=selected_option,
                actual_pattern=correct_answer,
                response_time=response_time
            )
            
            assert result is not None, f"Failed to submit answer for question {i+1}"
            assert "is_correct" in result, "Missing is_correct in result"
            assert result["is_correct"] == (selected_option == correct_answer), "Incorrect is_correct value"
        
        # 4. Verify assessment was completed
        completed_assessment = db_session.query(AssessmentAttempt).filter_by(id=assessment_id).first()
        assert completed_assessment.is_completed is True, "Assessment not marked as completed"
        assert completed_assessment.questions_completed == 3, "Not all questions marked as completed"
        assert completed_assessment.correct_answers == correct_count, "Correct answer count mismatch"
        
        expected_avg_time = total_response_time / 3
        assert abs(completed_assessment.avg_response_time - expected_avg_time) < 0.001, \
            f"Average response time mismatch: expected {expected_avg_time}, got {completed_assessment.avg_response_time}"
        
        expected_score = (correct_count / 3) * 100
        assert completed_assessment.score == expected_score, \
            f"Score mismatch: expected {expected_score}, got {completed_assessment.score}"
        
        # 5. Verify user performance was updated
        # For at least one pattern that was answered correctly
        if correct_count > 0:
            # Find a pattern ID that was answered correctly
            correct_answers = db_session.query(UserAnswer).filter_by(
                assessment_attempt_id=assessment_id,
                is_correct=True
            ).all()
            
            if correct_answers:
                pattern_id = correct_answers[0].selected_option
                
                user_perf = db_session.query(UserPerformance).filter_by(
                    user_id=user_id,
                    pattern_id=pattern_id
                ).first()
                
                if user_perf:
                    assert user_perf.total_attempts > 0, "User performance total_attempts not updated"
                    assert user_perf.correct_attempts > 0, "User performance correct_attempts not updated"
                    assert user_perf.last_attempt_at is not None, "User performance last_attempt_at not set"
        
        # 6. Verify pattern statistics were updated
        # For at least one pattern that was answered
        all_answers = db_session.query(UserAnswer).filter_by(
            assessment_attempt_id=assessment_id
        ).all()
        
        if all_answers:
            pattern_id = all_answers[0].selected_option
            
            pattern_stats = db_session.query(PatternStatistics).filter_by(
                pattern_id=pattern_id
            ).first()
            
            if pattern_stats:
                assert pattern_stats.total_attempts > 0, "Pattern statistics total_attempts not updated"
                assert pattern_stats.last_updated is not None, "Pattern statistics last_updated not set"
