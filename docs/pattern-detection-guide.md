# Pattern Detection Developer Guide

This guide provides comprehensive documentation for developers working with the TradeIQ Candlestick Pattern Detection system. It covers system architecture, usage patterns, extension mechanisms, and integration guidelines.

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Getting Started](#getting-started)
4. [Basic Usage](#basic-usage)
5. [Advanced Usage](#advanced-usage)
6. [Extending the System](#extending-the-system)
7. [Performance Considerations](#performance-considerations)
8. [Error Handling](#error-handling)
9. [Testing and Validation](#testing-and-validation)
10. [Integration Examples](#integration-examples)

## Overview

The TradeIQ Candlestick Pattern Detection system provides sophisticated market pattern recognition capabilities through multiple detection strategies and consensus mechanisms. The system is designed with the following core principles:

- **Accuracy**: Multiple detection strategies combined for maximum reliability
- **Flexibility**: Easily extensible for new patterns and strategies
- **Performance**: Optimized for efficient pattern detection in real-time
- **Maintainability**: Well-structured code with clear separation of concerns
- **Robustness**: Comprehensive error handling and graceful degradation

## Architecture

The system follows a layered architecture with clean separation between components:

### Core Components

1. **Base Classes and Interfaces**
   - `PatternDetector`: Abstract base class defining the contract for all detectors
   - `PatternMatch`: Data class representing detection results
   - `DetectionStrategy`: Enum defining supported detection approaches

2. **Rule-Based Detection**
   - `RuleBasedDetector`: Base class for geometric and statistical approaches
   - `GeometricPatternDetector`: Shape-based pattern recognition
   - `StatisticalPatternDetector`: Statistical validation for patterns

3. **ML-Based Detection**
   - `ModelBasedDetector`: Base class for all ML model detectors
   - Various model implementations (CNN, ResNet, EfficientNet)

4. **Ensemble Detection**
   - `EnsembleDetector`: Base class for consensus approaches
   - `WeightedConsensusDetector`: Advanced weighted voting implementation

5. **Factory System**
   - Factory functions for detector creation and configuration
   - Dynamic loading and initialization of detectors

### Design Patterns

The system leverages several design patterns:

1. **Strategy Pattern**: Encapsulates different detection algorithms
2. **Factory Pattern**: Creates appropriate detector instances
3. **Composite Pattern**: Combines multiple detectors in ensembles
4. **Decorator Pattern**: Adds functionality to base detectors
5. **Observer Pattern**: Implements performance tracking

## Getting Started

### Prerequisites

- Python 3.8+
- Required packages (see requirements.txt)
- Optional: GPU for ML-based detectors

### Installation

```bash
# Install required packages
pip install -r requirements.txt

# Optional: Install machine learning dependencies
pip install -r ml-requirements.txt
```

## Basic Usage

The simplest way to use the pattern detection system is with the default detector:

```python
import asyncio
from backend.assessments.candlestick_patterns.pattern_detection import get_default_detector
from backend.assessments.candlestick_patterns.candlestick_utils import Candle, CandlestickData

async def detect_patterns():
    # Create candlestick data
    candles = [
        Candle(time=1621123200, open=100.0, high=105.0, low=99.0, close=103.0, volume=1000000),
        Candle(time=1621209600, open=103.0, high=110.0, low=102.0, close=108.0, volume=1200000),
        Candle(time=1621296000, open=108.0, high=112.0, low=107.0, close=109.0, volume=900000),
    ]
    data = CandlestickData(symbol="AAPL", timeframe="1d", candles=candles)
    
    # Get the default detector (cached)
    detector = await get_default_detector()
    
    # Detect patterns
    patterns = await detector.detect_patterns(data)
    
    # Process results
    for pattern in patterns:
        print(f"Found {pattern.pattern_name} with confidence {pattern.confidence:.2f}")
        print(f"  Direction: {'Bullish' if pattern.bullish else 'Bearish' if pattern.bullish is False else 'Neutral'}")
        print(f"  Candle indices: {pattern.candle_indices}")
        print(f"  Detection strategy: {pattern.detection_strategy.value}")
```

## Advanced Usage

### Creating Custom Detectors

You can create specific detectors based on your requirements:

```python
from backend.assessments.candlestick_patterns.pattern_detection import (
    DetectionStrategy, create_detector, WeightedConsensusDetector
)

async def create_custom_detector():
    # Create specific detectors
    rule_detector = await create_detector(
        DetectionStrategy.RULE_BASED,
        config={"min_confidence": 0.6}
    )
    
    statistical_detector = await create_detector(
        DetectionStrategy.STATISTICAL,
        config={"min_confidence": 0.7}
    )
    
    # Try to create ML detector with fallback
    try:
        ml_detector = await create_detector(DetectionStrategy.ML_BASED)
        detectors = [rule_detector, statistical_detector, ml_detector]
    except Exception as e:
        print(f"ML detector unavailable, using rule-based only: {e}")
        detectors = [rule_detector, statistical_detector]
    
    # Create ensemble with custom configuration
    ensemble = WeightedConsensusDetector(
        detectors=detectors,
        name="CustomEnsembleDetector",
        min_confidence=0.65,
        max_patterns=10,
        config={
            "similarity_threshold": 0.8,
            "consensus_threshold": 0.6,
            "dynamic_weighting": True
        }
    )
    
    # Initialize the ensemble
    await ensemble.initialize()
    return ensemble
```

### Safe Pattern Detection

Use error-safe detection for improved robustness:

```python
async def detect_patterns_safely(data):
    detector = await get_default_detector()
    patterns, error = await detector.detect_patterns_safe(data)
    
    if error:
        print(f"Warning: Detection completed with error: {error}")
        
    return patterns
```

### Reading Detector Status

Check the current configuration and status of any detector:

```python
async def check_detector_status():
    detector = await get_default_detector()
    status = detector.get_status()
    
    print(f"Detector: {status['name']}")
    print(f"Type: {status['type']}")
    print(f"Configuration:")
    for key, value in status['config'].items():
        print(f"  {key}: {value}")
        
    if 'sub_detectors' in status:
        print("Sub-detectors:")
        for sub in status['sub_detectors']:
            print(f"  {sub['name']} (weight: {sub['weight']})")
```

## Extending the System

### Creating a New Pattern Detector

To create a new pattern detector, extend the appropriate base class:

```python
from backend.assessments.candlestick_patterns.pattern_detection import RuleBasedDetector, PatternMatch
from typing import List, Optional, Tuple

class MyCustomDetector(RuleBasedDetector):
    """Custom pattern detector implementing specific detection logic."""
    
    def __init__(self, name="MyCustomDetector", min_confidence=0.5, max_patterns=None):
        super().__init__(name=name, min_confidence=min_confidence, max_patterns=max_patterns)
    
    async def initialize(self):
        """Initialize resources needed by this detector."""
        self._initialized = True
        return True
    
    async def _detect_patterns_internal(self, data):
        """Implement custom pattern detection logic."""
        patterns = []
        candles = data.candles
        
        # Example: Detect a custom pattern
        for i in range(len(candles) - 2):
            # Your pattern detection logic here
            if self._is_my_custom_pattern(candles, i):
                pattern = PatternMatch(
                    pattern_name="My Custom Pattern",
                    confidence=0.85,
                    candle_indices=[i, i+1, i+2],
                    bullish=True,
                    detection_strategy=self.get_strategy_type(),
                    metadata={
                        "detector_name": self.name,
                        "custom_property": "value"
                    }
                )
                patterns.append(pattern)
        
        return patterns
    
    def _is_my_custom_pattern(self, candles, start_idx):
        """Check if candles starting at start_idx form the custom pattern."""
        # Implement your pattern recognition logic
        return False  # Replace with actual implementation
```

### Registering a Custom Detector

To make your detector available through the factory system:

```python
from backend.assessments.candlestick_patterns.pattern_detection import DetectionStrategy
from backend.assessments.candlestick_patterns.pattern_detection.factory import _STRATEGY_MAP

# Define a new strategy type
DetectionStrategy.MY_CUSTOM = "my_custom"

# Register your detector with the factory
_STRATEGY_MAP[DetectionStrategy.MY_CUSTOM] = MyCustomDetector
```

## Performance Considerations

### Caching

The system uses caching to improve performance:

1. **Detector Caching**: `get_default_detector()` is cached with LRU
2. **Initialization Caching**: Resources are loaded once and reused
3. **Pattern Caching**: Intermediate results can be cached

### Async Performance

For optimal async performance:

1. Use `asyncio.gather()` to run multiple detectors concurrently
2. Consider batching pattern detection for multiple symbols
3. Implement proper cancellation handling for long-running operations

### Memory Management

To optimize memory usage:

1. Use `max_patterns` parameter to limit result size
2. Set `min_confidence` to filter low-confidence patterns
3. Release ML model resources when not needed with `detector.release()`

## Error Handling

The system provides comprehensive error handling:

### Safe Detection Methods

```python
async def detect_with_error_handling(data):
    detector = await get_default_detector()
    
    # Method 1: Use safe detection method
    patterns, error = await detector.detect_patterns_safe(data)
    if error:
        # Handle error but still use partial results
        print(f"Warning: {error}")
    
    # Method 2: Use exception handling
    try:
        patterns = await detector.detect_patterns(data)
    except Exception as e:
        print(f"Error: {e}")
        patterns = []
```

### Graceful Degradation

The system implements graceful degradation:

1. ML model unavailability falls back to rule-based detection
2. Individual detector failures in ensembles don't stop the process
3. Configuration errors resort to default values

## Testing and Validation

### Unit Testing

Use the provided test utilities for unit testing:

```python
from backend.assessments.candlestick_patterns.pattern_detection.testing import (
    generate_test_data, validate_detector_output
)

async def test_custom_detector():
    # Generate test data
    test_data = generate_test_data(num_candles=30)
    
    # Create and test your detector
    detector = MyCustomDetector()
    await detector.initialize()
    
    # Run detection
    patterns = await detector.detect_patterns(test_data)
    
    # Validate results
    validation_result = validate_detector_output(patterns, test_data)
    print(f"Validation result: {validation_result}")
```

### Performance Testing

Benchmark your detector implementation:

```python
import time
from backend.assessments.candlestick_patterns.pattern_detection.testing import generate_test_data

async def benchmark_detector(detector, num_iterations=100):
    test_data = generate_test_data(num_candles=100)
    
    start_time = time.time()
    for _ in range(num_iterations):
        await detector.detect_patterns(test_data)
    
    elapsed = time.time() - start_time
    print(f"Average detection time: {elapsed/num_iterations*1000:.2f}ms")
```

## Integration Examples

### Integration with Assessment System

```python
from backend.assessments.candlestick_patterns.pattern_detection import get_default_detector
from backend.assessments.question_generation import generate_question

async def create_pattern_question(difficulty=0.5):
    # Get detector
    detector = await get_default_detector()
    
    # Get market data
    data = await get_market_data("AAPL", "1d", limit=30)
    
    # Detect patterns
    patterns = await detector.detect_patterns(data)
    
    # Filter patterns by difficulty
    suitable_patterns = [p for p in patterns if is_suitable_difficulty(p, difficulty)]
    
    if not suitable_patterns:
        return None
    
    # Generate question using the pattern
    selected_pattern = suitable_patterns[0]
    question = await generate_question(data, selected_pattern, difficulty)
    
    return question
```

### Web API Integration

```python
from fastapi import FastAPI, HTTPException
from backend.assessments.candlestick_patterns.pattern_detection import get_default_detector
from backend.assessments.candlestick_patterns.candlestick_utils import CandlestickData

app = FastAPI()

@app.post("/api/detect_patterns")
async def api_detect_patterns(data: CandlestickData):
    try:
        detector = await get_default_detector()
        patterns = await detector.detect_patterns(data)
        
        # Convert to API response format
        return {
            "patterns": [pattern.to_dict() for pattern in patterns],
            "count": len(patterns),
            "symbol": data.symbol,
            "timeframe": data.timeframe
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### Integration with Visualization

```python
import matplotlib.pyplot as plt
import mplfinance as mpf
from backend.assessments.candlestick_patterns.pattern_detection import get_default_detector

async def visualize_detected_patterns(data):
    # Detect patterns
    detector = await get_default_detector()
    patterns = await detector.detect_patterns(data)
    
    # Convert candles to pandas DataFrame
    df = convert_candles_to_dataframe(data.candles)
    
    # Create plot
    fig, axes = mpf.plot(df, type='candle', style='yahoo', returnfig=True)
    
    # Highlight patterns
    for pattern in patterns:
        indices = pattern.candle_indices
        color = 'green' if pattern.bullish else 'red'
        highlight_pattern(axes[0], df, indices, color, pattern.pattern_name)
    
    plt.title(f"Detected Patterns for {data.symbol} ({data.timeframe})")
    plt.show()
```

## Troubleshooting

### Common Issues

1. **ML Models Not Loading**
   - Check that model files exist in the expected location
   - Verify that required ML dependencies are installed
   - Ensure GPU drivers are properly configured (if using GPU)

2. **Low Detection Confidence**
   - Adjust `min_confidence` threshold in detector configuration
   - Check if data quality is poor or contains gaps
   - Verify that enough candles are provided for the patterns

3. **Inconsistent Results**
   - Ensure consistent candle normalization
   - Check for data anomalies like gaps or extreme values
   - Verify that timeframe is appropriate for the patterns

### Logging

Enable detailed logging for troubleshooting:

```python
import logging
from backend.assessments.candlestick_patterns.pattern_detection import get_default_detector

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("pattern_detection")

async def debug_detection(data):
    detector = await get_default_detector()
    logger.debug(f"Running detection on {len(data.candles)} candles")
    
    patterns = await detector.detect_patterns(data)
    logger.debug(f"Detected {len(patterns)} patterns")
    
    return patterns
```

## Conclusion

The TradeIQ Pattern Detection system provides a robust, extensible, and high-performance solution for candlestick pattern recognition. By following this guide, developers can effectively use, extend, and integrate the system into various applications.

For more information, refer to the API reference documentation and the source code comments. 