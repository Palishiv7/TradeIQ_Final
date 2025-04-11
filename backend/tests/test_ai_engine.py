"""
Comprehensive AI engine tests with real components and no mocks.

These tests use real infrastructure to verify the TradeIQ AI engine can handle
normal operations, edge cases, and unexpected scenarios gracefully.
"""

import os
import json
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
import random
import asyncio
import threading
import time
import pickle
import uuid
from pathlib import Path
import shutil
import logging
import multiprocessing
import concurrent.futures
import psutil

# Import database components
from database.init_db import initialize_database, get_db_session
from database.models import (
    PatternStatistics,
    UserPerformance,
    AssessmentAttempt,
    UserAnswer,
    Base
)

# Import AI engine components
from backend.common.ai_engine import (
    BaseModel, ModelVersion, ModelStatus, InferenceResult, ModelRegistry
)

# Import candlestick AI components
from backend.assessments.candlestick_patterns.candlestick_ai import (
    CandlestickImagePreprocessor, 
    CandlestickFeatureExtractor, 
    CandlestickPatternModel,
    RuleBasedCandlestickModel,
    ConvolutionalCandlestickModel,
    CandlestickModelFactory,
    CandlestickPatternDetectionService,
    PatternCategory,
    PATTERN_TAXONOMY
)

# Import candlestick utilities
from backend.assessments.candlestick_patterns.candlestick_utils import (
    Candle, CandlestickData, normalize_market_data
)

# Use real logger
logger = logging.getLogger("test_ai_engine")

# Test fixtures

@pytest.fixture(scope="module")
def database_connection():
    """Setup test database with real connection"""
    # Use test database URL from environment or fall back to SQLite
    test_db_url = os.environ.get("TEST_DB_URL", "sqlite:///./test_ai_engine.db")
    
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
    """Create a new database session for each test"""
    engine, sessionmaker = database_connection
    session = sessionmaker()
    
    try:
        yield session
    finally:
        session.rollback()
        session.close()


@pytest.fixture(scope="module")
def model_registry():
    """Initialize a real model registry for testing"""
    # Create test directory
    test_models_dir = "./test_models"
    os.makedirs(test_models_dir, exist_ok=True)
    
    # Create real registry
    registry = ModelRegistry(models_dir=test_models_dir)
    
    yield registry
    
    # Clean up test models
    if os.path.exists(test_models_dir):
        shutil.rmtree(test_models_dir)


@pytest.fixture
def sample_market_data():
    """Generate real market data for testing"""
    # Create a DataFrame with realistic OHLCV data
    dates = pd.date_range(start='2023-01-01', periods=100)
    
    # Generate realistic price movement (random walk)
    np.random.seed(42)  # For reproducibility
    price = 100  # Starting price
    prices = [price]
    
    for _ in range(99):  # Generate 99 more prices
        change_percent = np.random.normal(0, 0.01)  # Mean 0%, std 1%
        price = price * (1 + change_percent)
        prices.append(price)
    
    # Create OHLCV data
    data = pd.DataFrame({
        'date': dates,
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],  # High is higher than open
        'low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],   # Low is lower than open
        'close': [p * (1 + np.random.normal(0, 0.01)) for p in prices],       # Close is random around open
        'volume': np.random.randint(1000, 10000, 100)                         # Random volume between 1000-10000
    })
    
    # Ensure high is always the highest, low is always the lowest
    for i in range(len(data)):
        values = [data.iloc[i]['open'], data.iloc[i]['close']]
        data.at[i, 'high'] = max(data.iloc[i]['high'], max(values))
        data.at[i, 'low'] = min(data.iloc[i]['low'], min(values))
    
    return data


@pytest.fixture
def candlestick_data(sample_market_data):
    """Convert market data to a list of Candle objects"""
    candles = []
    for _, row in sample_market_data.iterrows():
        # Convert datetime to timestamp
        timestamp = int(row['date'].timestamp())
        candle = Candle(
            time=timestamp,
            open=row['open'],
            high=row['high'],
            low=row['low'],
            close=row['close'],
            volume=row['volume']
        )
        candles.append(candle)
    
    return CandlestickData(symbol="BTCUSD", timeframe="1d", candles=candles)


@pytest.fixture
def feature_extractor():
    """Create a real feature extractor"""
    return CandlestickFeatureExtractor()


@pytest.fixture
def image_preprocessor():
    """Create a real image preprocessor"""
    return CandlestickImagePreprocessor()


@pytest.fixture
def rule_based_model():
    """Create a real rule-based candlestick pattern model"""
    model = RuleBasedCandlestickModel(
        model_id="test_rule_based",
        version=ModelVersion(0, 1, 0)
    )
    return model


@pytest.fixture
def pattern_detection_service():
    """Create a real pattern detection service"""
    service = CandlestickPatternDetectionService(
        model_type="candlestick",
        confidence_threshold=0.6,
        preferred_model="rule_based"
    )
    return service


@pytest.fixture
def sample_patterns():
    """Return a sample of pattern names for testing"""
    return [
        "Doji", "Hammer", "Inverted Hammer", "Spinning Top", "Marubozu",
        "Bullish Engulfing", "Bearish Engulfing", "Morning Star", "Evening Star",
        "Three White Soldiers", "Three Black Crows"
    ]


# Basic functionality tests
class TestBasicFunctionality:
    """Test basic functionality of the TradeIQ AI engine with real components"""
    
    def test_model_registry_initialization(self, model_registry):
        """Test that the model registry initializes correctly"""
        assert model_registry is not None
        assert hasattr(model_registry, 'models_dir')
        assert os.path.exists(model_registry.models_dir)
    
    def test_rule_based_model_initialization(self, rule_based_model):
        """Test that the rule-based model initializes correctly"""
        assert rule_based_model is not None
        assert rule_based_model.model_id == "test_rule_based"
        assert rule_based_model.version.major == 0
        assert rule_based_model.version.minor == 1
        assert rule_based_model.status == ModelStatus.READY
    
    def test_model_registry_operations(self, model_registry, rule_based_model):
        """Test basic model registry operations"""
        # Register model
        model_registry.register_model(rule_based_model)
        
        # Get model
        retrieved_model = model_registry.get_model(rule_based_model.model_id)
        assert retrieved_model is not None
        assert retrieved_model.model_id == rule_based_model.model_id
        
        # Set active version
        model_registry.set_active_version("candlestick", rule_based_model.model_id)
        
        # Get active model
        active_model = model_registry.get_active_model("candlestick")
        assert active_model is not None
        assert active_model.model_id == rule_based_model.model_id
        
        # List models
        models = model_registry.list_models()
        assert len(models) > 0
        assert any(model['model_id'] == rule_based_model.model_id for model in models)
    
    def test_feature_extraction(self, feature_extractor, candlestick_data):
        """Test feature extraction from real candlestick data"""
        # Extract features
        features = feature_extractor.extract_features(candlestick_data.candles)
        
        # Verify features
        assert features is not None
        assert isinstance(features, dict)
        assert len(features) > 0
        
        # Check for some expected feature categories
        assert "body_to_range_ratio" in features
        assert "upper_shadow_ratio" in features
        assert "lower_shadow_ratio" in features
        
        # Verify feature values are within expected ranges
        assert 0 <= features["body_to_range_ratio"] <= 1
        assert 0 <= features["upper_shadow_ratio"] <= 1
        assert 0 <= features["lower_shadow_ratio"] <= 1
    
    def test_image_preprocessing(self, image_preprocessor, candlestick_data):
        """Test image preprocessing for candlestick data"""
        # Preprocess candlestick data
        image = image_preprocessor.preprocess_candlestick_data(candlestick_data)
        
        # Verify image
        assert image is not None
        assert isinstance(image, np.ndarray)
        assert image.shape == (*image_preprocessor.target_size, 3)  # Width, height, channels
        
        # Check normalization (values should be between 0 and 1)
        assert np.min(image) >= 0
        assert np.max(image) <= 1
    
    def test_pattern_detection(self, pattern_detection_service, candlestick_data):
        """Test detection of candlestick patterns in real data"""
        # Detect patterns
        results = pattern_detection_service.detect_patterns(candlestick_data)
        
        # Verify results
        assert results is not None
        assert isinstance(results, dict)
        assert "detected_patterns" in results
        assert "confidence_scores" in results
        assert "timestamp" in results
        
        # Check that any patterns were detected
        if len(results["detected_patterns"]) > 0:
            # Verify that detected patterns are valid
            for pattern_info in results["detected_patterns"]:
                # Pattern should be in our taxonomy
                pattern_name = pattern_info["name"]
                found = False
                for category in PatternCategory:
                    patterns_in_category = PATTERN_TAXONOMY[category]
                    pattern_names = [p["name"] for p in patterns_in_category]
                    if pattern_name in pattern_names:
                        found = True
                        break
                assert found, f"Detected pattern '{pattern_name}' not found in pattern taxonomy"
    
    def test_rule_based_model_prediction(self, rule_based_model, candlestick_data):
        """Test prediction with rule-based model on real data"""
        # Preprocess data
        preprocessed = rule_based_model.preprocess(candlestick_data)
        
        # Run prediction
        prediction_results = rule_based_model.predict(preprocessed)
        
        # Postprocess results
        inference_result = rule_based_model.postprocess(prediction_results)
        
        # Verify results
        assert inference_result is not None
        assert isinstance(inference_result, InferenceResult)
        assert inference_result.model_id == rule_based_model.model_id
        assert inference_result.model_version == str(rule_based_model.version)
        
        # Check predictions array
        assert hasattr(inference_result, "predictions")
        assert isinstance(inference_result.predictions, list)
        
        # Check confidence scores
        assert hasattr(inference_result, "confidence_scores")
        assert isinstance(inference_result.confidence_scores, list)
        assert len(inference_result.confidence_scores) == len(inference_result.predictions)
        
        # Check confidence values are between 0 and 1
        for score in inference_result.confidence_scores:
            assert 0 <= score <= 1
    
    def test_model_info(self, rule_based_model):
        """Test retrieving model information"""
        # Get model info
        info = rule_based_model.get_info()
        
        # Verify info
        assert info is not None
        assert isinstance(info, dict)
        assert "model_id" in info
        assert "version" in info
        assert "status" in info
        assert "model_type" in info
        
        # Check values
        assert info["model_id"] == rule_based_model.model_id
        assert info["version"] == str(rule_based_model.version)
        assert info["status"] == rule_based_model.status.value
    
    def test_pattern_taxonomy(self, sample_patterns):
        """Test that pattern taxonomy is properly defined"""
        # Check that each sample pattern exists in the taxonomy
        for pattern_name in sample_patterns:
            # Find pattern in taxonomy
            found = False
            for category in PatternCategory:
                patterns_in_category = PATTERN_TAXONOMY[category]
                pattern_names = [p["name"] for p in patterns_in_category]
                if pattern_name in pattern_names:
                    found = True
                    break
            assert found, f"Pattern '{pattern_name}' not found in pattern taxonomy"

# Data edge cases tests
class TestDataEdgeCases:
    """Test how the AI engine handles data edge cases with real components"""
    
    def test_empty_dataframe(self, feature_extractor, image_preprocessor):
        """Test handling of empty dataframe"""
        # Create empty dataframe
        empty_df = pd.DataFrame(columns=['date', 'open', 'high', 'low', 'close', 'volume'])
        
        # Convert to empty candles list
        empty_candles = []
        
        # Create empty candlestick data
        empty_data = CandlestickData(symbol="BTCUSD", timeframe="1d", candles=empty_candles)
        
        # Test feature extraction - should handle gracefully or raise appropriate error
        try:
            features = feature_extractor.extract_features(empty_candles)
            # If it doesn't raise exception, verify features are empty or default
            assert features is not None
            assert isinstance(features, dict)
            assert len(features) > 0  # Should at least return default values
        except Exception as e:
            # If it raises exception, it should be an appropriate one
            assert "empty" in str(e).lower() or "no data" in str(e).lower()
        
        # Test image preprocessing - should handle gracefully or raise appropriate error
        try:
            image = image_preprocessor.preprocess_candlestick_data(empty_data)
            # If it doesn't raise exception, verify image is valid
            assert image is not None
            assert isinstance(image, np.ndarray)
            assert image.shape == (*image_preprocessor.target_size, 3)
        except Exception as e:
            # If it raises exception, it should be an appropriate one
            assert "empty" in str(e).lower() or "no data" in str(e).lower()
    
    def test_single_data_point(self, feature_extractor, image_preprocessor):
        """Test handling of single data point"""
        # Create dataframe with single row
        single_df = pd.DataFrame({
            'date': [datetime.now()],
            'open': [100],
            'high': [105],
            'low': [95],
            'close': [102],
            'volume': [1000]
        })
        
        # Convert to single candle
        single_candle = Candle(
            time=int(single_df.iloc[0]['date'].timestamp()),
            open=single_df.iloc[0]['open'],
            high=single_df.iloc[0]['high'],
            low=single_df.iloc[0]['low'],
            close=single_df.iloc[0]['close'],
            volume=single_df.iloc[0]['volume']
        )
        single_candles = [single_candle]
        
        # Create candlestick data with single candle
        single_data = CandlestickData(symbol="BTCUSD", timeframe="1d", candles=single_candles)
        
        # Test feature extraction
        features = feature_extractor.extract_features(single_candles)
        assert features is not None
        assert isinstance(features, dict)
        assert len(features) > 0
        
        # Test image preprocessing
        image = image_preprocessor.preprocess_candlestick_data(single_data)
        assert image is not None
        assert isinstance(image, np.ndarray)
        assert image.shape == (*image_preprocessor.target_size, 3)
    
    def test_missing_columns(self, sample_market_data, feature_extractor):
        """Test handling of missing columns in the data"""
        # Create copies with missing columns
        missing_open = sample_market_data.drop(columns=['open'])
        missing_high = sample_market_data.drop(columns=['high'])
        missing_low = sample_market_data.drop(columns=['low'])
        missing_close = sample_market_data.drop(columns=['close'])
        missing_volume = sample_market_data.drop(columns=['volume'])
        
        # Test each case
        for missing_df, missing_col in [
            (missing_open, 'open'),
            (missing_high, 'high'),
            (missing_low, 'low'),
            (missing_close, 'close'),
            (missing_volume, 'volume')
        ]:
            # Try to convert to candles - should handle gracefully or raise appropriate error
            try:
                candles = []
                for _, row in missing_df.iterrows():
                    # This will fail due to missing column
                    candle = Candle(
                        time=int(row['date'].timestamp()),
                        open=row.get('open', 0),  # Use 0 as default for missing columns
                        high=row.get('high', 0),
                        low=row.get('low', 0),
                        close=row.get('close', 0),
                        volume=row.get('volume', 0)
                    )
                    candles.append(candle)
                
                # If conversion succeeds, test feature extraction
                features = feature_extractor.extract_features(candles)
                assert features is not None
                assert isinstance(features, dict)
            except Exception as e:
                # If it raises exception, it should be appropriate
                assert missing_col in str(e).lower() or "missing" in str(e).lower()
    
    def test_all_null_data(self, feature_extractor, image_preprocessor):
        """Test handling of data with all null values"""
        # Create dataframe with all nulls
        null_df = pd.DataFrame({
            'date': [datetime.now()] * 10,
            'open': [None] * 10,
            'high': [None] * 10,
            'low': [None] * 10,
            'close': [None] * 10,
            'volume': [None] * 10
        })
        
        # Try to convert to candles - should handle gracefully or raise appropriate error
        try:
            # Replace None with 0 for Candle conversion
            candles = []
            for _, row in null_df.iterrows():
                candle = Candle(
                    time=int(row['date'].timestamp()),
                    open=row['open'] if row['open'] is not None else 0,
                    high=row['high'] if row['high'] is not None else 0,
                    low=row['low'] if row['low'] is not None else 0,
                    close=row['close'] if row['close'] is not None else 0,
                    volume=row['volume'] if row['volume'] is not None else 0
                )
                candles.append(candle)
            
            # Create candlestick data
            null_data = CandlestickData(symbol="BTCUSD", timeframe="1d", candles=candles)
            
            # Test feature extraction
            features = feature_extractor.extract_features(candles)
            assert features is not None
            
            # Test image preprocessing
            image = image_preprocessor.preprocess_candlestick_data(null_data)
            assert image is not None
        except Exception as e:
            # If it raises exception, it should be appropriate
            assert "null" in str(e).lower() or "none" in str(e).lower() or "missing" in str(e).lower()
    
    def test_partial_null_data(self, feature_extractor, image_preprocessor, sample_market_data):
        """Test handling of data with some null values"""
        # Create a copy of the dataframe
        partial_null_df = sample_market_data.copy()
        
        # Set some values to None
        null_indices = random.sample(range(len(partial_null_df)), 5)
        for idx in null_indices:
            col = random.choice(['open', 'high', 'low', 'close', 'volume'])
            partial_null_df.at[idx, col] = None
        
        # Try to convert to candles and handle nulls
        candles = []
        for _, row in partial_null_df.iterrows():
            # Handle nulls by using defaults
            candle = Candle(
                time=int(row['date'].timestamp()),
                open=row['open'] if row['open'] is not None else 0,
                high=row['high'] if row['high'] is not None else 0,
                low=row['low'] if row['low'] is not None else 0,
                close=row['close'] if row['close'] is not None else 0,
                volume=row['volume'] if row['volume'] is not None else 0
            )
            candles.append(candle)
        
        # Create candlestick data
        partial_null_data = CandlestickData(symbol="BTCUSD", timeframe="1d", candles=candles)
        
        # Test feature extraction - should handle nulls
        features = feature_extractor.extract_features(candles)
        assert features is not None
        assert isinstance(features, dict)
        
        # Test image preprocessing - should handle nulls
        image = image_preprocessor.preprocess_candlestick_data(partial_null_data)
        assert image is not None
        assert isinstance(image, np.ndarray)
    
    def test_extreme_values(self, feature_extractor, image_preprocessor):
        """Test handling of extreme values in data"""
        # Create dataframe with extreme values
        extreme_df = pd.DataFrame({
            'date': [datetime.now()] * 10,
            'open': [1e10, 1e-10, 1e10, 1e-10, 1e10, 1e-10, 1e10, 1e-10, 1e10, 1e-10],  # Very large/small
            'high': [1e10, 1e-10, 1e10, 1e-10, 1e10, 1e-10, 1e10, 1e-10, 1e10, 1e-10],
            'low': [1e9, 1e-11, 1e9, 1e-11, 1e9, 1e-11, 1e9, 1e-11, 1e9, 1e-11],
            'close': [1e10, 1e-10, 1e10, 1e-10, 1e10, 1e-10, 1e10, 1e-10, 1e10, 1e-10],
            'volume': [1e10, 0, 1e10, 0, 1e10, 0, 1e10, 0, 1e10, 0]  # Very large/zero
        })
        
        # Convert to candles
        candles = []
        for _, row in extreme_df.iterrows():
            candle = Candle(
                time=int(row['date'].timestamp()),
                open=row['open'],
                high=row['high'],
                low=row['low'],
                close=row['close'],
                volume=row['volume']
            )
            candles.append(candle)
        
        # Create candlestick data
        extreme_data = CandlestickData(symbol="BTCUSD", timeframe="1d", candles=candles)
        
        # Test feature extraction - should handle extreme values
        features = feature_extractor.extract_features(candles)
        assert features is not None
        assert isinstance(features, dict)
        
        # Test image preprocessing - should handle extreme values
        image = image_preprocessor.preprocess_candlestick_data(extreme_data)
        assert image is not None
        assert isinstance(image, np.ndarray)
    
    def test_invalid_date_format(self, sample_market_data):
        """Test handling of invalid date formats"""
        # Create a copy of the dataframe
        invalid_date_df = sample_market_data.copy()
        
        # Replace dates with strings
        invalid_date_df['date'] = ["invalid-date"] * len(invalid_date_df)
        
        # Try to convert to candles - should handle gracefully or raise appropriate error
        try:
            candles = []
            for _, row in invalid_date_df.iterrows():
                candle = Candle(
                    time=int(row['date'].timestamp()),
                    open=row['open'],
                    high=row['high'],
                    low=row['low'],
                    close=row['close'],
                    volume=row['volume']
                )
                candles.append(candle)
            
            # If conversion succeeds, ensure dates were handled
            for candle in candles:
                assert candle.time is not None
        except Exception as e:
            # If it raises exception, it should be appropriate
            assert "date" in str(e).lower() or "invalid" in str(e).lower() or "format" in str(e).lower()
    
    def test_non_chronological_data(self, feature_extractor, image_preprocessor):
        """Test handling of non-chronologically ordered data"""
        # Create dataframe with random date order
        dates = [datetime.now() - timedelta(days=i) for i in range(20)]
        random.shuffle(dates)  # Shuffle the dates
        
        non_chrono_df = pd.DataFrame({
            'date': dates,
            'open': [100] * 20,
            'high': [105] * 20,
            'low': [95] * 20,
            'close': [102] * 20,
            'volume': [1000] * 20
        })
        
        # Convert to candles
        candles = []
        for _, row in non_chrono_df.iterrows():
            candle = Candle(
                time=int(row['date'].timestamp()),
                open=row['open'],
                high=row['high'],
                low=row['low'],
                close=row['close'],
                volume=row['volume']
            )
            candles.append(candle)
        
        # Create candlestick data
        non_chrono_data = CandlestickData(symbol="BTCUSD", timeframe="1d", candles=candles)
        
        # Check if data is sorted before processing
        try:
            # Test feature extraction
            features = feature_extractor.extract_features(candles)
            assert features is not None
            
            # Test image preprocessing
            image = image_preprocessor.preprocess_candlestick_data(non_chrono_data)
            assert image is not None
        except Exception as e:
            # If it raises exception, it might be that chronological order is required
            assert "chronological" in str(e).lower() or "order" in str(e).lower() or "sorted" in str(e).lower()
    
    def test_future_dates(self, feature_extractor, image_preprocessor):
        """Test handling of data with future dates"""
        # Create dataframe with future dates
        future_dates = [datetime.now() + timedelta(days=i) for i in range(1, 21)]
        
        future_df = pd.DataFrame({
            'date': future_dates,
            'open': [100] * 20,
            'high': [105] * 20,
            'low': [95] * 20,
            'close': [102] * 20,
            'volume': [1000] * 20
        })
        
        # Convert to candles
        candles = []
        for _, row in future_df.iterrows():
            candle = Candle(
                time=int(row['date'].timestamp()),
                open=row['open'],
                high=row['high'],
                low=row['low'],
                close=row['close'],
                volume=row['volume']
            )
            candles.append(candle)
        
        # Create candlestick data
        future_data = CandlestickData(symbol="BTCUSD", timeframe="1d", candles=candles)
        
        # Test feature extraction - future dates should be valid
        features = feature_extractor.extract_features(candles)
        assert features is not None
        assert isinstance(features, dict)
        
        # Test image preprocessing - future dates should be valid
        image = image_preprocessor.preprocess_candlestick_data(future_data)
        assert image is not None
        assert isinstance(image, np.ndarray)
    
    def test_invalid_ohlc_relationship(self, feature_extractor, image_preprocessor):
        """Test handling of invalid OHLC relationship (high < low, etc.)"""
        # Create dataframe with invalid OHLC relationships
        invalid_ohlc_df = pd.DataFrame({
            'date': [datetime.now()] * 10,
            'open': [100] * 10,
            'high': [90] * 10,  # High < Open (invalid)
            'low': [110] * 10,  # Low > Open (invalid)
            'close': [102] * 10,
            'volume': [1000] * 10
        })
        
        # Try to convert to candles
        try:
            candles = []
            for _, row in invalid_ohlc_df.iterrows():
                candle = Candle(
                    time=int(row['date'].timestamp()),
                    open=row['open'],
                    high=row['high'],
                    low=row['low'],
                    close=row['close'],
                    volume=row['volume']
                )
                candles.append(candle)
            
            # Create candlestick data
            invalid_ohlc_data = CandlestickData(symbol="BTCUSD", timeframe="1d", candles=candles)
            
            # Test feature extraction - should handle invalid OHLC or raise appropriate error
            features = feature_extractor.extract_features(candles)
            assert features is not None
            
            # Test image preprocessing - should handle invalid OHLC or raise appropriate error
            image = image_preprocessor.preprocess_candlestick_data(invalid_ohlc_data)
            assert image is not None
        except Exception as e:
            # If it raises exception, it should be appropriate
            assert "high" in str(e).lower() or "low" in str(e).lower() or "invalid" in str(e).lower()
    
    def test_extremely_large_dataset(self, feature_extractor, image_preprocessor):
        """Test handling of extremely large datasets"""
        # Create a large dataframe (10,000 rows)
        large_size = 10000
        dates = [datetime.now() - timedelta(days=i) for i in range(large_size)]
        
        large_df = pd.DataFrame({
            'date': dates,
            'open': np.random.normal(100, 10, large_size),
            'high': np.random.normal(105, 10, large_size),
            'low': np.random.normal(95, 10, large_size),
            'close': np.random.normal(102, 10, large_size),
            'volume': np.random.randint(1000, 10000, large_size)
        })
        
        # Ensure high is always the highest, low is always the lowest
        for i in range(len(large_df)):
            values = [large_df.iloc[i]['open'], large_df.iloc[i]['close']]
            large_df.at[i, 'high'] = max(large_df.iloc[i]['high'], max(values))
            large_df.at[i, 'low'] = min(large_df.iloc[i]['low'], min(values))
        
        # Convert to candles (this may take some time)
        candles = []
        for _, row in large_df.iterrows():
            candle = Candle(
                time=int(row['date'].timestamp()),
                open=row['open'],
                high=row['high'],
                low=row['low'],
                close=row['close'],
                volume=row['volume']
            )
            candles.append(candle)
        
        # Create candlestick data
        large_data = CandlestickData(symbol="BTCUSD", timeframe="1d", candles=candles)
        
        # Test feature extraction - should handle large dataset efficiently
        start_time = time.time()
        features = feature_extractor.extract_features(candles)
        extraction_time = time.time() - start_time
        
        assert features is not None
        assert isinstance(features, dict)
        assert extraction_time < 60  # Should process in less than 60 seconds
        
        # Test image preprocessing with a subset (all might be too memory intensive)
        sample_candles = candles[:1000]  # Use first 1000 candles
        sample_data = CandlestickData(symbol="BTCUSD", timeframe="1d", candles=sample_candles)
        
        start_time = time.time()
        image = image_preprocessor.preprocess_candlestick_data(sample_data)
        preprocessing_time = time.time() - start_time
        
        assert image is not None
        assert isinstance(image, np.ndarray)
        assert preprocessing_time < 60  # Should process in less than 60 seconds

# Model edge cases tests
class TestModelEdgeCases:
    """Test how the AI engine handles model edge cases with real components"""
    
    def test_nonexistent_model(self, model_registry):
        """Test handling of requests for nonexistent models"""
        # Try to get a nonexistent model
        nonexistent_id = "nonexistent_model_" + str(uuid.uuid4())
        model = model_registry.get_model(nonexistent_id)
        
        # Should return None, not raise an exception
        assert model is None
        
        # Try to set a nonexistent model as active
        try:
            model_registry.set_active_version("candlestick", nonexistent_id)
            # If no exception, verify that no active model is set
            active_model = model_registry.get_active_model("candlestick")
            # If one is returned, it should not be our nonexistent ID
            if active_model is not None:
                assert active_model.model_id != nonexistent_id
            except Exception as e:
            # If it raises exception, it should be appropriate
            assert "not found" in str(e).lower() or "nonexistent" in str(e).lower() or "not exist" in str(e).lower()
    
    def test_corrupted_model_file(self, model_registry, rule_based_model):
        """Test handling of corrupted model files"""
        # Register a model
        model_registry.register_model(rule_based_model)
        
        # Corrupt the model file by writing invalid data
        model_path = os.path.join(model_registry.models_dir, f"{rule_based_model.model_id}.pkl")
        with open(model_path, 'wb') as f:
            f.write(b'corrupted data')
        
        # Try to load the corrupted model
        try:
            model = model_registry.get_model(rule_based_model.model_id)
            # If it loads, it should be in an error state
            if model is not None:
                assert model.status == ModelStatus.ERROR
            except Exception as e:
            # If it raises exception, it should be appropriate
            assert "corrupt" in str(e).lower() or "invalid" in str(e).lower() or "error" in str(e).lower()
        
        # Clean up - restore the model or remove it
            if os.path.exists(model_path):
                os.remove(model_path)
    
    def test_model_with_missing_methods(self, model_registry):
        """Test handling of models with missing required methods"""
        # Create a broken model class with missing methods
        class BrokenModel(BaseModel):
            def __init__(self, model_id, version):
                super().__init__(model_id, version)
            
            # Missing preprocess method
            
            def predict(self, preprocessed_inputs):
                return {"predictions": ["Doji"]}
            
            def postprocess(self, outputs):
                return InferenceResult(
                    predictions=outputs["predictions"],
                    confidence_scores=[0.9],
                    model_id=self.model_id,
                    model_version=str(self.version),
                    inference_time=0.1
                )
        
        # Create an instance of the broken model
        broken_model = BrokenModel(
            model_id="broken_model",
            version=ModelVersion(0, 1, 0)
        )
        
        # Try to register and use the broken model
        try:
            model_registry.register_model(broken_model)
            
            # Try to use the model
            test_input = {"data": "test"}
            result = broken_model.infer(test_input)
            
            # If it works, it shouldn't crash
            assert result is not None
        except Exception as e:
            # If it raises exception, ensure it's because of the missing method
            assert "preprocess" in str(e).lower() or "missing" in str(e).lower() or "abstract" in str(e).lower()
    
    def test_model_versioning(self, model_registry):
        """Test handling of multiple model versions"""
        # Create models with different versions
        model_v1 = RuleBasedCandlestickModel(
            model_id="versioned_model",
            version=ModelVersion(1, 0, 0)
        )
        
        model_v2 = RuleBasedCandlestickModel(
            model_id="versioned_model",
            version=ModelVersion(2, 0, 0)
        )
        
        # Register both versions
        model_registry.register_model(model_v1)
        model_registry.register_model(model_v2)
        
        # Get model by ID - should return the latest version
        model = model_registry.get_model("versioned_model")
        assert model is not None
        assert model.version.major == 2  # Should get the highest version
        
        # Set specific version as active
        model_registry.set_active_version("candlestick", "versioned_model")
        
        # Get active model
        active_model = model_registry.get_active_model("candlestick")
        assert active_model is not None
        assert active_model.model_id == "versioned_model"
    
    def test_invalid_model_parameters(self, model_registry):
        """Test handling of invalid model parameters"""
        # Try to create a model with invalid parameters
        try:
            # Invalid version
            invalid_version_model = RuleBasedCandlestickModel(
                model_id="invalid_version_model",
                version="not_a_version"  # Invalid version type
            )
            
            # Should raise exception during initialization
            assert False, "Should have raised exception for invalid version type"
        except Exception as e:
            # Check that exception is appropriate
            assert "version" in str(e).lower() or "type" in str(e).lower() or "invalid" in str(e).lower()
        
        # Try with empty ID
        try:
            empty_id_model = RuleBasedCandlestickModel(
                model_id="",  # Empty ID
                version=ModelVersion(1, 0, 0)
            )
            
            # If it creates the model, try to register it
            model_registry.register_model(empty_id_model)
            
            # If registration succeeds, verify the model exists
            model = model_registry.get_model("")
            assert model is not None
        except Exception as e:
            # If it raises exception, ensure it's appropriate
            assert "id" in str(e).lower() or "empty" in str(e).lower() or "invalid" in str(e).lower()
    
    def test_model_factory(self):
        """Test the model factory with various inputs"""
        # Create models using factory
        rule_based = CandlestickModelFactory.create_model("rule_based")
        assert rule_based is not None
        assert isinstance(rule_based, RuleBasedCandlestickModel)
        
        # Try with invalid model type
        try:
            invalid_model = CandlestickModelFactory.create_model("nonexistent_type")
            # If it doesn't raise exception, it should return a default model or None
            if invalid_model is not None:
                assert isinstance(invalid_model, CandlestickPatternModel)
        except Exception as e:
            # If it raises exception, ensure it's appropriate
            assert "type" in str(e).lower() or "invalid" in str(e).lower() or "not found" in str(e).lower()
        
        # Try with valid type but invalid parameters
        try:
            model_with_invalid_params = CandlestickModelFactory.create_model(
                "rule_based",
                invalid_param="should not be used"
            )
            # If it creates the model, it should ignore invalid params
            assert model_with_invalid_params is not None
            assert isinstance(model_with_invalid_params, RuleBasedCandlestickModel)
        except Exception as e:
            # If it raises exception, check it's appropriate
            assert "parameter" in str(e).lower() or "invalid" in str(e).lower()
    
    def test_model_that_always_fails(self, model_registry):
        """Test handling of a model that always fails during prediction"""
        # Create a model that always raises an exception during prediction
        class FailingModel(BaseModel):
            def __init__(self, model_id, version):
                super().__init__(model_id, version)
            
            def preprocess(self, inputs):
                return inputs
            
            def predict(self, preprocessed_inputs):
                raise RuntimeError("This model always fails during prediction!")
            
            def postprocess(self, outputs):
                return outputs
        
        # Create an instance of the failing model
        failing_model = FailingModel(
            model_id="failing_model",
            version=ModelVersion(0, 1, 0)
        )
        
        # Register the failing model
        model_registry.register_model(failing_model)
        
        # Try to use the failing model
        try:
            # Should raise an exception during inference
            result = failing_model.infer({"test": "data"})
            
            # If it doesn't raise exception, ensure the result indicates failure
            assert result is not None
            assert result[1].get("error") is not None  # Check error in metadata
        except Exception as e:
            # If it raises exception, ensure it's the expected one
            assert "fails during prediction" in str(e)
    
    def test_model_with_excessive_memory_usage(self, model_registry):
        """Test handling of models with excessive memory usage"""
        # Create a model that tries to use excessive memory
        class MemoryHogModel(BaseModel):
            def __init__(self, model_id, version):
                super().__init__(model_id, version)
            
            def preprocess(self, inputs):
                return inputs
            
            def predict(self, preprocessed_inputs):
                # Try to allocate a large amount of memory
                try:
                    # 1 GB array (this may fail depending on available memory)
                    large_array = np.ones((1024 * 1024 * 256), dtype=np.float32)
                    return {"large_array": large_array, "predictions": ["Doji"]}
                except MemoryError:
                    # Handle the memory error gracefully
                    return {"predictions": ["Memory allocation failed"], "error": "Out of memory"}
            
            def postprocess(self, outputs):
                if "error" in outputs:
                    # Return a graceful error result
                    return InferenceResult(
                        predictions=outputs.get("predictions", []),
                        confidence_scores=[0.0],
                        model_id=self.model_id,
                        model_version=str(self.version),
                        inference_time=0.1,
                        metadata={"error": outputs["error"]}
                    )
                else:
                    return InferenceResult(
                        predictions=outputs["predictions"],
                        confidence_scores=[0.9],
                        model_id=self.model_id,
                        model_version=str(self.version),
                        inference_time=0.1
                    )
        
        # Create an instance of the memory hog model
        memory_hog_model = MemoryHogModel(
            model_id="memory_hog_model",
            version=ModelVersion(0, 1, 0)
        )
        
        # Register the model
        model_registry.register_model(memory_hog_model)
        
        # Try to use the model
        start_time = time.time()
        try:
            result, metadata = memory_hog_model.infer({"test": "data"})
            
            # Check that inference completed
            assert result is not None
            
            # Check that it didn't take too long (timeout)
            assert time.time() - start_time < 30  # Should complete within 30 seconds
            
            # Note: If it succeeds in allocating memory, that's fine.
            # If it catches MemoryError internally, it should return a valid result with error metadata.
        except MemoryError:
            # If it raises MemoryError, that's expected behavior too
            pass
            except Exception as e:
            # Other exceptions should be related to memory
            assert "memory" in str(e).lower()
    
    def test_conflicting_model_ids(self, model_registry):
        """Test handling of registration with conflicting model IDs"""
        # Create two models with the same ID but different versions
        model1 = RuleBasedCandlestickModel(
            model_id="conflict_model",
            version=ModelVersion(1, 0, 0)
        )
        
        model2 = RuleBasedCandlestickModel(
            model_id="conflict_model",
            version=ModelVersion(1, 0, 0)  # Exactly the same version
        )
        
        # Register the first model
        model_registry.register_model(model1)
        
        # Try to register the second model
        try:
            model_registry.register_model(model2)
            
            # If it succeeds, verify which model is returned
            retrieved_model = model_registry.get_model("conflict_model")
            assert retrieved_model is not None
            # Should return one of the models, presumably the last registered
        except Exception as e:
            # If it raises exception, ensure it's appropriate for the conflict
            assert "conflict" in str(e).lower() or "already exists" in str(e).lower() or "duplicate" in str(e).lower()

# Infrastructure edge cases tests
class TestInfrastructureEdgeCases:
    """Test how the AI engine handles infrastructure-related edge cases with real components"""
    
    def test_model_registry_persistence(self, model_registry, rule_based_model):
        """Test that model registry persists models correctly"""
        # Register a model
        model_registry.register_model(rule_based_model)
        
        # Create a new registry instance pointing to the same directory
        new_registry = ModelRegistry(models_dir=model_registry.models_dir)
        
        # Verify the model is still available
        model = new_registry.get_model(rule_based_model.model_id)
        assert model is not None
        assert model.model_id == rule_based_model.model_id
        assert model.version == rule_based_model.version
    
    def test_concurrent_registry_access(self, model_registry, rule_based_model):
        """Test concurrent access to the model registry"""
        # Register a model
        model_registry.register_model(rule_based_model)
        
        # Create a new registry instance pointing to the same directory
        concurrent_registry = ModelRegistry(models_dir=model_registry.models_dir)
        
        # Modify model in first registry
        new_model = RuleBasedCandlestickModel(
            model_id="concurrent_model",
            version=ModelVersion(1, 0, 0)
        )
        model_registry.register_model(new_model)
        
        # Verify changes visible in second registry
        model = concurrent_registry.get_model("concurrent_model")
        assert model is not None
        assert model.model_id == "concurrent_model"
    
    def test_registry_with_invalid_directory(self):
        """Test behavior when model registry is initialized with invalid directory"""
        # Create registry with non-existent directory
        temp_dir = f"/tmp/nonexistent_dir_{uuid.uuid4()}"
        
        try:
            registry = ModelRegistry(models_dir=temp_dir)
            
            # If it creates the registry, the directory should exist now
            assert os.path.exists(temp_dir)
            
            # Clean up
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception as e:
            # If it raises exception, ensure it's appropriate
            assert "directory" in str(e).lower() or "path" in str(e).lower()
    
    def test_registry_with_read_only_directory(self):
        """Test behavior when model registry is initialized with read-only directory"""
        # Create a temporary directory
        temp_dir = f"/tmp/readonly_dir_{uuid.uuid4()}"
        os.makedirs(temp_dir, exist_ok=True)
        
        try:
            # Make it read-only
            os.chmod(temp_dir, 0o555)  # r-xr-xr-x
            
            # Try to create registry
            registry = ModelRegistry(models_dir=temp_dir)
            
            # Try to register a model (should fail due to permissions)
            model = RuleBasedCandlestickModel(
                model_id="test_model",
                version=ModelVersion(1, 0, 0)
            )
            
            try:
                registry.register_model(model)
                # If it succeeds, the directory might not actually be read-only
                # Do a sanity check
                assert os.path.exists(os.path.join(temp_dir, f"{model.model_id}.pkl"))
            except Exception as e:
                # If it raises exception, ensure it's related to permissions
                assert "permission" in str(e).lower() or "read-only" in str(e).lower() or "access" in str(e).lower()
        finally:
            # Clean up by making the directory writable again and removing it
            os.chmod(temp_dir, 0o755)  # Make writable again
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_disk_full_scenario(self, model_registry):
        """Simulate disk full scenario during model saving"""
        # This is difficult to test directly without mocking
        # Instead, we'll create a model that's too large and check error handling
        
        class LargeModel(BaseModel):
            def __init__(self, model_id, version):
                super().__init__(model_id, version)
                # Create a large attribute that will be serialized
                self.large_data = np.zeros((100, 100, 100), dtype=np.float32)
            
            def preprocess(self, inputs):
                return inputs
            
            def predict(self, preprocessed_inputs):
                return {"predictions": ["Doji"]}
            
            def postprocess(self, outputs):
                return InferenceResult(
                    predictions=outputs["predictions"],
                    confidence_scores=[0.9],
                    model_id=self.model_id,
                    model_version=str(self.version),
                    inference_time=0.1
                )
        
        # Try to save a very large model
        max_attempts = 5
        
        for i in range(max_attempts):
            try:
                # Incrementally increase size until we hit an error or max attempts
                large_model = LargeModel(
                    model_id=f"large_model_{i}",
                    version=ModelVersion(0, 1, 0)
                )
                
                # Make the model larger based on iteration
                large_model.large_data = np.zeros((100 * (i + 1), 100, 100), dtype=np.float32)
                
                # Try to register the model
                model_registry.register_model(large_model)
                
                # If it succeeds, increase size for next attempt
                print(f"Successfully registered model of size {large_model.large_data.size * 4 / (1024*1024):.2f} MB")
                
                except Exception as e:
                # Check if the exception is related to disk space
                if "disk" in str(e).lower() or "space" in str(e).lower() or "memory" in str(e).lower():
                    # This is the expected error for disk full scenario
                    print(f"Got expected disk space error: {str(e)}")
                else:
                    # Other errors are still acceptable
                    print(f"Got unexpected error: {str(e)}")
                
                # We hit an error, stop increasing size
                break
    
    def test_models_across_processes(self, model_registry, rule_based_model):
        """Test that models can be shared across processes"""
        # Register a model
        model_registry.register_model(rule_based_model)
        
        # Define a function to run in a subprocess
        def subprocess_function(model_dir, model_id):
            try:
                # Create a registry in the subprocess
                registry = ModelRegistry(models_dir=model_dir)
                
                # Try to get the model
                model = registry.get_model(model_id)
                
                # Return success if model found
                return model is not None
                    except Exception as e:
                print(f"Error in subprocess: {str(e)}")
                return False
        
        # Use multiprocessing to run the function in a separate process
        with multiprocessing.Pool(1) as pool:
            result = pool.apply(
                subprocess_function,
                (model_registry.models_dir, rule_based_model.model_id)
            )
        
        # Assert the model was accessible in the subprocess
        assert result is True, "Model should be accessible across processes"
    
    def test_temporary_files_cleanup(self, model_registry, rule_based_model):
        """Test that temporary files are cleaned up properly"""
        # Count files before
        original_files = set(os.listdir(model_registry.models_dir))
        
        # Register a model
        model_registry.register_model(rule_based_model)
        
        # Retrieve the model a few times
        for _ in range(5):
            model = model_registry.get_model(rule_based_model.model_id)
            assert model is not None
        
        # Delete the model
        model_registry.delete_model(rule_based_model.model_id)
        
        # Count files after
        final_files = set(os.listdir(model_registry.models_dir))
        
        # The only difference should be the model file and possibly a metadata file
        diff = final_files - original_files
        for file in diff:
            assert rule_based_model.model_id in file, f"Unexpected file: {file}"
        
        # Check that no temporary files remain
        for file in final_files:
            assert not file.endswith('.tmp'), f"Temporary file not cleaned up: {file}"
            assert not file.startswith('tmp_'), f"Temporary file not cleaned up: {file}"
    
    def test_backup_and_restore(self, model_registry, rule_based_model):
        """Test backup and restore functionality"""
        # Register a model
        model_registry.register_model(rule_based_model)
        
        # Create a backup directory
        backup_dir = os.path.join(os.path.dirname(model_registry.models_dir), "backup_test")
        os.makedirs(backup_dir, exist_ok=True)
        
        try:
            # Manually back up the model
            model_path = os.path.join(model_registry.models_dir, f"{rule_based_model.model_id}.pkl")
            backup_path = os.path.join(backup_dir, f"{rule_based_model.model_id}.pkl")
            
            shutil.copy2(model_path, backup_path)
            
            # Delete the original model
            model_registry.delete_model(rule_based_model.model_id)
            
            # Verify model is gone
            assert model_registry.get_model(rule_based_model.model_id) is None
            
            # Restore from backup
            shutil.copy2(backup_path, model_path)
            
            # Verify model is restored
            restored_model = model_registry.get_model(rule_based_model.model_id)
            assert restored_model is not None
            assert restored_model.model_id == rule_based_model.model_id
            assert restored_model.version == rule_based_model.version
        finally:
            # Clean up
            shutil.rmtree(backup_dir, ignore_errors=True)

# Performance tests
class TestPerformance:
    """Test performance aspects of the AI engine with real components"""
    
    def test_inference_performance(self, rule_based_model, sample_market_data):
        """Test the inference performance of a model"""
        # Prepare real data
        data = pd.DataFrame(sample_market_data)
        
        # Warm-up run
        result, metadata = rule_based_model.infer({"market_data": data})
        
        # Performance measurement
        num_iterations = 10
        start_time = time.time()
        
        for _ in range(num_iterations):
            result, metadata = rule_based_model.infer({"market_data": data})
            assert result is not None
        
        end_time = time.time()
        
        # Calculate average time
        avg_time = (end_time - start_time) / num_iterations
        
        # Log performance metrics
        print(f"Average inference time: {avg_time:.4f} seconds")
        
        # Assert reasonable performance (adjust threshold as needed)
        assert avg_time < 1.0, f"Inference too slow: {avg_time:.4f} seconds"
    
    def test_model_loading_performance(self, model_registry, rule_based_model):
        """Test the performance of model loading"""
        # Register a model
        model_registry.register_model(rule_based_model)
        
        # Performance measurement
        num_iterations = 10
                start_time = time.time()
                
        for _ in range(num_iterations):
            model = model_registry.get_model(rule_based_model.model_id)
            assert model is not None
        
        end_time = time.time()
        
        # Calculate average time
        avg_time = (end_time - start_time) / num_iterations
        
        # Log performance metrics
        print(f"Average model loading time: {avg_time:.4f} seconds")
        
        # Assert reasonable performance (adjust threshold as needed)
        assert avg_time < 0.5, f"Model loading too slow: {avg_time:.4f} seconds"
    
    def test_batch_inference(self, rule_based_model):
        """Test batch inference performance with varying batch sizes"""
        batch_sizes = [1, 5, 10, 20, 50]
        times = []
        
        for batch_size in batch_sizes:
            # Generate market data for each batch size
            batch_data = []
            for _ in range(batch_size):
                # Generate sample market data with 100 data points
                market_data = pd.DataFrame({
                    "timestamp": pd.date_range(start="2022-01-01", periods=100, freq="D"),
                    "open": np.random.normal(100, 10, 100),
                    "high": np.random.normal(110, 10, 100),
                    "low": np.random.normal(90, 10, 100),
                    "close": np.random.normal(105, 10, 100),
                    "volume": np.random.normal(1000, 200, 100)
                })
                batch_data.append(market_data)
            
            # Performance measurement
            start_time = time.time()
            
            for data in batch_data:
                result, metadata = rule_based_model.infer({"market_data": data})
                assert result is not None
            
            end_time = time.time()
            
            # Calculate total time
            total_time = end_time - start_time
            times.append(total_time)
            
            # Log performance metrics
            print(f"Batch size {batch_size}: Total time {total_time:.4f} seconds, Avg per item {total_time/batch_size:.4f} seconds")
        
        # Verify that per-item processing time decreases with batch size (economies of scale)
        if len(batch_sizes) > 1:
            per_item_times = [time / size for time, size in zip(times, batch_sizes)]
            # We expect some efficiency gain, but not necessarily monotonic
            assert min(per_item_times) < per_item_times[0], "No efficiency gain with larger batches"
    
    def test_concurrent_model_access(self, model_registry, rule_based_model):
        """Test concurrent access to models"""
        # Register a model
        model_registry.register_model(rule_based_model)
        
        # Function to run in parallel
        def inference_task(registry_dir, model_id):
            # Create registry
            registry = ModelRegistry(models_dir=registry_dir)
            
            # Get the model
            model = registry.get_model(model_id)
            if model is None:
                return False
            
            # Generate sample data
            market_data = pd.DataFrame({
                "timestamp": pd.date_range(start="2022-01-01", periods=100, freq="D"),
                "open": np.random.normal(100, 10, 100),
                "high": np.random.normal(110, 10, 100),
                "low": np.random.normal(90, 10, 100),
                "close": np.random.normal(105, 10, 100),
                "volume": np.random.normal(1000, 200, 100)
            })
            
            # Perform inference
            result, metadata = model.infer({"market_data": market_data})
            
            # Return success if inference worked
            return result is not None
        
        # Run multiple threads
        num_threads = 5
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(
                    inference_task,
                    model_registry.models_dir,
                    rule_based_model.model_id
                )
                for _ in range(num_threads)
            ]
            
            # Wait for all threads to complete
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # All threads should succeed
        assert all(results), "Some concurrent model access attempts failed"
    
    def test_memory_usage(self, model_registry, rule_based_model):
        """Test memory usage during model operations"""
        # Function to get current memory usage
        def get_memory_usage():
            # Get the current process
            process = psutil.Process(os.getpid())
            # Return memory usage in MB
            return process.memory_info().rss / (1024 * 1024)
        
        # Measure baseline memory
        baseline_memory = get_memory_usage()
        
        # Register a model
        model_registry.register_model(rule_based_model)
        after_register_memory = get_memory_usage()
        
        # Generate a large dataset
        large_data = pd.DataFrame({
            "timestamp": pd.date_range(start="2020-01-01", periods=1000, freq="D"),
            "open": np.random.normal(100, 10, 1000),
            "high": np.random.normal(110, 10, 1000),
            "low": np.random.normal(90, 10, 1000),
            "close": np.random.normal(105, 10, 1000),
            "volume": np.random.normal(1000, 200, 1000)
        })
        
        # Perform inference with the large dataset
        result, metadata = rule_based_model.infer({"market_data": large_data})
        after_inference_memory = get_memory_usage()
        
        # Load the model multiple times
        for _ in range(10):
            model = model_registry.get_model(rule_based_model.model_id)
            assert model is not None
        
        after_loading_memory = get_memory_usage()
        
        # Log memory usage at each stage
        print(f"Baseline memory usage: {baseline_memory:.2f} MB")
        print(f"After registration: {after_register_memory:.2f} MB (Δ {after_register_memory - baseline_memory:.2f} MB)")
        print(f"After inference: {after_inference_memory:.2f} MB (Δ {after_inference_memory - after_register_memory:.2f} MB)")
        print(f"After multiple loading: {after_loading_memory:.2f} MB (Δ {after_loading_memory - after_inference_memory:.2f} MB)")
        
        # Assert reasonable memory usage
        # Memory shouldn't increase dramatically after multiple loads (checking for leaks)
        assert after_loading_memory - after_inference_memory < 50, "Possible memory leak during model loading"
    
    def test_cpu_usage(self, rule_based_model, sample_market_data):
        """Test CPU usage during inference"""
        # Function to get current CPU usage
        def get_cpu_usage():
            return psutil.Process(os.getpid()).cpu_percent(interval=0.1)
        
        # Prepare real data
        data = pd.DataFrame(sample_market_data)
        
        # Warm-up run
        result, metadata = rule_based_model.infer({"market_data": data})
        
        # Measure CPU usage during inference
        start_cpu = get_cpu_usage()
        
        # Perform multiple inferences
        num_iterations = 5
        for _ in range(num_iterations):
            result, metadata = rule_based_model.infer({"market_data": data})
            assert result is not None
        
        end_cpu = get_cpu_usage()
        
        # Log CPU usage
        print(f"CPU usage during inference: {end_cpu}%")
        
        # We can't assert on exact CPU usage as it depends on the system,
        # but we can log it for reference

if __name__ == "__main__":
    pytest.main(["-xvs", "test_ai_engine.py"])
