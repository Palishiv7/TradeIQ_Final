"""
Pattern Detection Example

This script demonstrates how to use the multi-strategy pattern detection system
to identify candlestick patterns in market data. It provides examples of using
individual detectors as well as ensemble approaches.
"""

import os
import sys
import time
import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from pprint import pprint

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("pattern_detection_example")

# Add project root to path if running as script
if __name__ == "__main__" and not __package__:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))

# Import necessary modules
from backend.common.finance.patterns import PatternType, PatternStrength
from backend.assessments.candlestick_patterns.candlestick_utils import (
    Candle, CandlestickData, generate_sample_data
)
from backend.assessments.candlestick_patterns.pattern_detection.interface import (
    PatternDetector, PatternMatch, DetectionStrategy
)
from backend.assessments.candlestick_patterns.pattern_detection.factory import (
    get_default_detector, create_detector
)

async def load_sample_data(
    symbol: str = "BTC/USD", 
    timeframe: str = "1h", 
    candle_count: int = 30
) -> CandlestickData:
    """
    Load sample market data for testing pattern detection.
    Attempts to load real market data if available, or generates synthetic data.
    
    Args:
        symbol: Market symbol (e.g., "BTC/USD")
        timeframe: Timeframe for candles (e.g., "1h", "1d")
        candle_count: Number of candles to include
        
    Returns:
        CandlestickData object with candles
    """
    try:
        # Try to import market data loader (if available)
        from backend.common.data.market_data import load_market_data
        
        logger.info(f"Loading real market data for {symbol}")
        data = await load_market_data(symbol, timeframe, candle_count)
        if data and len(data) >= candle_count:
            # Convert to our Candle format
            candles = []
            for i, bar in enumerate(data):
                candle = Candle(
                    time=bar['timestamp'].timestamp(),
                    open=bar['open'],
                    high=bar['high'],
                    low=bar['low'],
                    close=bar['close'],
                    volume=bar.get('volume', 0)
                )
                candles.append(candle)
            return CandlestickData(symbol=symbol, timeframe=timeframe, candles=candles)
    except (ImportError, Exception) as e:
        logger.warning(f"Could not load real market data: {e}")
    
    # Fallback to generated data
    logger.info(f"Generating sample candlestick data for {symbol}")
    candles = generate_sample_data(
        count=candle_count,
        start_price=40000.0,  # Starting price
        volatility=0.02,      # 2% volatility
        trend=0.001,          # Slight uptrend
        pattern_at=15,        # Insert pattern at this position
        pattern_type="morning_star"  # Type of pattern to insert
    )
    
    return CandlestickData(symbol=symbol, timeframe=timeframe, candles=candles)

def print_pattern_results(
    detector_name: str,
    patterns: List[PatternMatch],
    execution_time_ms: float
) -> None:
    """
    Format and print the results of detected patterns.
    
    Args:
        detector_name: Name of the detector used
        patterns: List of detected patterns
        execution_time_ms: Time taken for detection in milliseconds
    """
    print(f"\n{'=' * 50}")
    print(f"Detector: {detector_name}")
    print(f"Execution Time: {execution_time_ms:.2f} ms")
    print(f"Found {len(patterns)} patterns")
    print(f"{'-' * 50}")
    
    if not patterns:
        print("No patterns detected")
        return
    
    # Sort patterns by confidence (descending)
    patterns.sort(key=lambda p: p.confidence, reverse=True)
    
    for i, pattern in enumerate(patterns):
        direction = "Bullish" if pattern.bullish else "Bearish" if pattern.bullish is False else "Neutral"
        print(f"{i+1}. {pattern.pattern_name} ({direction}) - Confidence: {pattern.confidence:.2f}")
        print(f"   Candle indices: {pattern.candle_indices}")
        print(f"   Detection strategy: {pattern.detection_strategy.value if pattern.detection_strategy else 'Unknown'}")
        
        # Print metadata if available
        if pattern.metadata:
            print("   Metadata:")
            for key, value in pattern.metadata.items():
                if isinstance(value, float):
                    print(f"      {key}: {value:.3f}")
                else:
                    print(f"      {key}: {value}")
        
        print(f"   {pattern.description}" if pattern.description else "")
        print(f"{'-' * 30}")

async def test_individual_detectors(data: CandlestickData) -> None:
    """
    Test individual pattern detection strategies.
    
    Args:
        data: Candlestick data to analyze
    """
    print("\n\n===== TESTING INDIVIDUAL DETECTORS =====")
    
    # Test rule-based detector
    try:
        print("\nTesting Rule-Based Detector...")
        rule_based = await create_detector(
            DetectionStrategy.RULE_BASED,
            config={"min_confidence": 0.6}
        )
        
        start_time = time.time()
        patterns = await rule_based.detect_patterns(data)
        end_time = time.time()
        
        execution_time_ms = (end_time - start_time) * 1000
        print_pattern_results("Rule-Based", patterns, execution_time_ms)
    except Exception as e:
        logger.error(f"Error testing rule-based detector: {e}")
    
    # Test ML-based detector (if available)
    try:
        print("\nTesting ML-Based Detector...")
        ml_based = await create_detector(
            DetectionStrategy.ML_BASED,
            config={"min_confidence": 0.7}
        )
        
        start_time = time.time()
        patterns = await ml_based.detect_patterns(data)
        end_time = time.time()
        
        execution_time_ms = (end_time - start_time) * 1000
        print_pattern_results("ML-Based", patterns, execution_time_ms)
    except Exception as e:
        logger.error(f"Error testing ML-based detector: {e}")

async def test_ensemble_detector(data: CandlestickData) -> None:
    """
    Test ensemble pattern detection approach.
    
    Args:
        data: Candlestick data to analyze
    """
    print("\n\n===== TESTING ENSEMBLE DETECTOR =====")
    
    try:
        # Create ensemble detector with multiple strategies
        ensemble = await create_detector(
            DetectionStrategy.WEIGHTED_CONSENSUS,
            config={
                "detectors": [
                    {"strategy": DetectionStrategy.RULE_BASED, "weight": 0.6, "name": "RuleDetector"},
                    {"strategy": DetectionStrategy.STATISTICAL, "weight": 0.4, "name": "StatDetector"}
                ],
                "min_confidence": 0.65
            }
        )
        
        start_time = time.time()
        patterns = await ensemble.detect_patterns(data)
        end_time = time.time()
        
        execution_time_ms = (end_time - start_time) * 1000
        print_pattern_results("Ensemble (Weighted Consensus)", patterns, execution_time_ms)
    except Exception as e:
        logger.error(f"Error testing ensemble detector: {e}")

async def test_default_detector(data: CandlestickData) -> None:
    """
    Test the default recommended detector.
    
    Args:
        data: Candlestick data to analyze
    """
    print("\n\n===== TESTING DEFAULT DETECTOR =====")
    
    try:
        # Get the default detector (cached, optimized for production use)
        detector = await get_default_detector(config={"min_confidence": 0.7})
        
        start_time = time.time()
        patterns = await detector.detect_patterns(data)
        end_time = time.time()
        
        execution_time_ms = (end_time - start_time) * 1000
        print_pattern_results("Default Detector", patterns, execution_time_ms)
        
        # Convert to recognition result (API format)
        result = PatternDetector.to_recognition_result(
            candlestick_data=data,
            detected_patterns=patterns,
            execution_time_ms=execution_time_ms,
            strategy_type=detector.strategy,
            metadata={"detector_name": detector.name}
        )
        
        print("\nPattern Recognition Result Summary:")
        print(f"Symbol: {result.symbol}")
        print(f"Timeframe: {data.timeframe}")
        print(f"Period: {result.start_time} to {result.end_time}")
        print(f"Total Patterns: {len(result.patterns)}")
        print(f"Metadata: {result.metadata}")
    except Exception as e:
        logger.error(f"Error testing default detector: {e}")

async def run_performance_test(iterations: int = 10, candle_count: int = 100) -> None:
    """
    Run a performance test on pattern detection.
    
    Args:
        iterations: Number of test iterations
        candle_count: Number of candles in each test
    """
    print("\n\n===== PERFORMANCE TEST =====")
    print(f"Running {iterations} iterations with {candle_count} candles each")
    
    # Create detector once to reuse
    detector = await get_default_detector()
    
    total_time_ms = 0
    total_patterns = 0
    
    for i in range(iterations):
        # Generate new data for each iteration
        data = await load_sample_data(candle_count=candle_count)
        
        # Run detection
        start_time = time.time()
        patterns = await detector.detect_patterns(data)
        end_time = time.time()
        
        # Track metrics
        execution_time_ms = (end_time - start_time) * 1000
        total_time_ms += execution_time_ms
        total_patterns += len(patterns)
        
        print(f"Iteration {i+1}: {execution_time_ms:.2f} ms, {len(patterns)} patterns")
    
    # Calculate averages
    avg_time = total_time_ms / iterations
    avg_patterns = total_patterns / iterations
    
    print(f"\nAverage execution time: {avg_time:.2f} ms")
    print(f"Average patterns detected: {avg_patterns:.1f}")
    print(f"Patterns per second: {(avg_patterns / (avg_time / 1000)):.1f}")

async def main() -> None:
    """Main function to run the examples."""
    print("Candlestick Pattern Detection Example")
    print("====================================")
    
    # Load sample data
    data = await load_sample_data(symbol="BTC/USD", timeframe="1h", candle_count=30)
    print(f"Loaded {len(data.candles)} candles for {data.symbol} ({data.timeframe})")
    
    # Test individual detectors
    await test_individual_detectors(data)
    
    # Test ensemble detector
    await test_ensemble_detector(data)
    
    # Test default detector
    await test_default_detector(data)
    
    # Run performance test
    await run_performance_test(iterations=5, candle_count=50)
    
    print("\nExample complete!")

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main()) 