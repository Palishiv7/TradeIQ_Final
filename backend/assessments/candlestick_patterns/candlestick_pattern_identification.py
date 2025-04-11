"""
Candlestick Pattern Identification Module

This module provides:
1. Rule-based pattern detection algorithms
2. Machine learning model integrations for pattern recognition
3. Pattern classification utilities
4. Composite detection strategy with consensus voting

The module serves as the core engine for identifying candlestick patterns
in market data, which is essential for the answer evaluation system.
"""

from typing import Dict, List, Any, Tuple, Optional, Union, Set, Callable, Type, cast
import numpy as np
from dataclasses import dataclass
import threading
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import time
import logging
import json
import os
from enum import Enum
from functools import lru_cache
from collections import Counter
from datetime import datetime
import weakref

from backend.assessments.candlestick_patterns.candlestick_utils import Candle, CandlestickData
from backend.assessments.candlestick_patterns.candlestick_config import CANDLESTICK_PATTERNS
from backend.common.logger import get_logger
from backend.common.finance.patterns import PatternType, PatternStrength, CandlestickPattern, PatternRecognitionResult
from backend.common.serialization import SerializableMixin

# Use local class-level logger
logger = get_logger(__name__)

# Define pattern detection strategy types
class DetectionStrategy(Enum):
    """Enum for pattern detection strategies."""
    RULE_BASED = "rule_based"
    ML_BASED = "ml_based"
    HYBRID = "hybrid"

@dataclass
class PatternMatch(SerializableMixin):
    """
    Represents a detected candlestick pattern.
    This class is a simplified version of CandlestickPattern for internal use,
    with methods to convert to and from the common CandlestickPattern class.
    """
    pattern_name: str
    confidence: float
    candle_indices: List[int]
    bullish: bool  # True for bullish patterns, False for bearish
    description: Optional[str] = None
    detection_strategy: Optional[DetectionStrategy] = None
    detection_time_ms: float = 0.0
    
    def __post_init__(self) -> None:
        """Validate inputs after initialization."""
        if not isinstance(self.pattern_name, str) or not self.pattern_name:
            raise ValueError("Pattern name must be a non-empty string")
        
        if not isinstance(self.confidence, (int, float)) or not (0 <= self.confidence <= 1):
            raise ValueError("Confidence must be a number between 0 and 1")
        
        if not isinstance(self.candle_indices, list):
            raise ValueError("Candle indices must be a list")
        
        # Ensure candle_indices is always a list of integers
        self.candle_indices = [int(idx) for idx in self.candle_indices]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "pattern_name": self.pattern_name,
            "confidence": self.confidence,
            "candle_indices": self.candle_indices,
            "bullish": self.bullish,
            "description": self.description,
            "detection_strategy": self.detection_strategy.value if self.detection_strategy else None,
            "detection_time_ms": self.detection_time_ms
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PatternMatch':
        """Create a PatternMatch from a dictionary."""
        try:
            # Create a copy to avoid modifying the original
            data_copy = data.copy()
            
            # Extract and process strategy
            strategy_str = data_copy.pop("detection_strategy", None)
            strategy = DetectionStrategy(strategy_str) if strategy_str else None
            
            # Create the instance
            return cls(**data_copy, detection_strategy=strategy)
        except Exception as e:
            logger.error(f"Error creating PatternMatch from dict: {e}")
            # Create a fallback pattern match with minimal data
            return cls(
                pattern_name=data.get("pattern_name", "Unknown"),
                confidence=data.get("confidence", 0.0),
                candle_indices=data.get("candle_indices", [0]),
                bullish=data.get("bullish", False),
                description="Error creating pattern: " + str(e)
            )
    
    def to_candlestick_pattern(self, 
                               symbol: str, 
                               candles: List[Candle], 
                               start_time: datetime, 
                               end_time: datetime) -> CandlestickPattern:
        """
        Convert this PatternMatch to a CandlestickPattern.
        
        Args:
            symbol: The symbol/ticker where the pattern was found
            candles: The full list of candles
            start_time: The start time of the pattern
            end_time: The end time of the pattern
            
        Returns:
            A CandlestickPattern object
        """
        try:
            # Map pattern name to PatternType
            try:
                pattern_type = PatternType(self.pattern_name.lower())
            except ValueError:
                # Use a fallback if pattern name doesn't match PatternType
                pattern_type = PatternType.DOJI
                logger.warning(f"Unknown pattern type: {self.pattern_name}, defaulting to DOJI")
            
            # Determine pattern strength from confidence
            strength = PatternStrength.from_score(self.confidence)
            
            # Get candles for this pattern using indices - with defensive bounds checking
            pattern_candles = []
            for i in self.candle_indices:
                if 0 <= i < len(candles):
                    pattern_candles.append(candles[i])
                else:
                    logger.warning(f"Candle index {i} out of bounds (max: {len(candles)-1})")
            
            # Handle the case where no valid candles were found
            if not pattern_candles and candles:
                logger.warning(f"No valid candles found for pattern {self.pattern_name}, using last candle")
                pattern_candles = [candles[-1]]
            
            # Convert Candle objects to Candlestick objects
            from backend.common.finance.candlestick import Candlestick
            candlesticks = []
            for candle in pattern_candles:
                try:
                    # Create a Candlestick from Candle
                    cs = Candlestick(
                        timestamp=datetime.fromtimestamp(candle.timestamp),
                        open=candle.open,
                        high=candle.high,
                        low=candle.low,
                        close=candle.close,
                        volume=candle.volume
                    )
                    candlesticks.append(cs)
                except Exception as e:
                    logger.error(f"Error converting candle to candlestick: {e}")
            
            # Create and return the CandlestickPattern
            return CandlestickPattern(
                pattern_type=pattern_type,
                symbol=symbol,
                start_time=start_time,
                end_time=end_time,
                candles=candlesticks,
                strength=strength,
                confidence=self.confidence,
                expected_direction="up" if self.bullish else "down",
                metadata={
                    "detection_strategy": self.detection_strategy.value if self.detection_strategy else None,
                    "detection_time_ms": self.detection_time_ms,
                    "description": self.description
                }
            )
        except Exception as e:
            logger.error(f"Error converting PatternMatch to CandlestickPattern: {e}")
            # Create a minimal fallback pattern
            from backend.common.finance.candlestick import Candlestick
            return CandlestickPattern(
                pattern_type=PatternType.DOJI,
                symbol=symbol,
                start_time=start_time,
                end_time=end_time,
                candles=[],
                strength=PatternStrength.WEAK,
                confidence=0.1,
                expected_direction="unknown",
                metadata={"error": str(e)}
            )
    
    @classmethod
    def from_candlestick_pattern(cls, pattern: CandlestickPattern, candle_indices: Optional[List[int]] = None) -> 'PatternMatch':
        """
        Create a PatternMatch from a CandlestickPattern.
        
        Args:
            pattern: The CandlestickPattern to convert
            candle_indices: The indices of the candles in the original data
            
        Returns:
            A PatternMatch object
        """
        try:
            # Use provided indices or default to range
            indices = candle_indices or list(range(len(pattern.candles)))
            
            # Determine strategy from metadata if available
            strategy_str = pattern.metadata.get("detection_strategy")
            strategy = None
            if strategy_str:
                try:
                    strategy = DetectionStrategy(strategy_str)
                except ValueError:
                    logger.warning(f"Unknown detection strategy: {strategy_str}")
            
            return cls(
                pattern_name=pattern.pattern_type.value,
                confidence=pattern.confidence,
                candle_indices=indices,
                bullish=pattern.is_bullish,
                description=pattern.metadata.get("description"),
                detection_strategy=strategy,
                detection_time_ms=pattern.metadata.get("detection_time_ms", 0.0)
            )
        except Exception as e:
            logger.error(f"Error creating PatternMatch from CandlestickPattern: {e}")
            # Create a fallback pattern match
            return cls(
                pattern_name=pattern.pattern_type.value if hasattr(pattern, 'pattern_type') else "unknown",
                confidence=0.5,
                candle_indices=[0],
                bullish=True,
                description=f"Error creating pattern match: {e}"
            )
    
    @property
    def serializable_fields(self) -> List[str]:
        """Get a list of fields that should be serialized."""
        return [
            "pattern_name", "confidence", "candle_indices", "bullish",
            "description", "detection_strategy", "detection_time_ms"
        ]

class PatternDetector:
    """
    Base class for pattern detectors.
    Implements the core interface for pattern detection algorithms.
    """
    
    def __init__(self):
        """Initialize the detector."""
        self.name = self.__class__.__name__
    
    def detect_patterns(self, data: CandlestickData) -> List[PatternMatch]:
        """
        Detect patterns in the given candlestick data.
        
        Args:
            data: Candlestick data to analyze
            
        Returns:
            List of detected patterns
        """
        raise NotImplementedError("Subclasses must implement detect_patterns")
    
    def get_strategy_type(self) -> DetectionStrategy:
        """Get the detection strategy type."""
        return DetectionStrategy.RULE_BASED  # Default
    
    def to_recognition_result(self, 
                              data: CandlestickData, 
                              patterns: List[PatternMatch],
                              execution_time: float = 0.0) -> PatternRecognitionResult:
        """
        Convert a list of PatternMatch objects to a PatternRecognitionResult.
        
        Args:
            data: The candlestick data that was analyzed
            patterns: The detected patterns
            execution_time: The time taken for detection in seconds
            
        Returns:
            A PatternRecognitionResult object
        """
        # Create start and end times
        if data.candles and len(data.candles) > 0:
            start_time = datetime.fromtimestamp(data.candles[0].timestamp)
            end_time = datetime.fromtimestamp(data.candles[-1].timestamp)
        else:
            start_time = datetime.now()
            end_time = datetime.now()
        
        # Convert patterns to CandlestickPattern objects
        candlestick_patterns = []
        for pattern in patterns:
            try:
                # Get start and end times for this specific pattern
                pattern_indices = pattern.candle_indices
                if pattern_indices and len(pattern_indices) > 0:
                    pattern_start_idx = min(pattern_indices)
                    pattern_end_idx = max(pattern_indices)
                    
                    if 0 <= pattern_start_idx < len(data.candles) and 0 <= pattern_end_idx < len(data.candles):
                        pattern_start = datetime.fromtimestamp(data.candles[pattern_start_idx].timestamp)
                        pattern_end = datetime.fromtimestamp(data.candles[pattern_end_idx].timestamp)
                    else:
                        pattern_start = start_time
                        pattern_end = end_time
                else:
                    pattern_start = start_time
                    pattern_end = end_time
                
                # Convert to CandlestickPattern
                cp = pattern.to_candlestick_pattern(
                    symbol=data.symbol,
                    candles=data.candles,
                    start_time=pattern_start,
                    end_time=pattern_end
                )
                candlestick_patterns.append(cp)
            except Exception as e:
                logger.error(f"Error converting pattern {pattern.pattern_name}: {e}")
        
        # Create and return the result
        return PatternRecognitionResult(
            symbol=data.symbol,
            patterns=candlestick_patterns,
            analyzed_period=(start_time, end_time),
            total_candles=len(data.candles),
            execution_time=execution_time,
            algorithm=self.name,
            metadata={
                "detection_strategy": self.get_strategy_type().value
            }
        )

class RuleBasedDetector(PatternDetector):
    """
    Rule-based detector that uses predefined rules to identify patterns.
    These rules are based on geometric properties of candlesticks.
    """
    
    def __init__(self, name: str = "DefaultRuleBased"):
        """
        Initialize the rule-based detector.
        
        Args:
            name: Detector instance name
        """
        super().__init__()
        self.name = name
    
    def detect_patterns(self, data: CandlestickData) -> List[PatternMatch]:
        """Detect patterns using rule-based methods."""
        start_time = time.time()
        patterns = []
        
        # Get candles
        candles = data.candles
        if len(candles) < 3:
            logger.warning(f"Not enough candles to detect patterns: {len(candles)}")
            return []
        
        # Detect single candlestick patterns
        patterns.extend(self._detect_single_patterns(candles))
        
        # Detect double candlestick patterns if we have at least 2 candles
        if len(candles) >= 2:
            patterns.extend(self._detect_double_patterns(candles))
        
        # Detect triple candlestick patterns if we have at least 3 candles
        if len(candles) >= 3:
            patterns.extend(self._detect_triple_patterns(candles))
        
        # Add detection strategy and time
        end_time = time.time()
        detection_time_ms = (end_time - start_time) * 1000
        
        for pattern in patterns:
            pattern.detection_strategy = DetectionStrategy.RULE_BASED
            pattern.detection_time_ms = detection_time_ms
        
        return patterns
    
    def _detect_single_patterns(self, candles: List[Candle]) -> List[PatternMatch]:
        """Detect single candlestick patterns."""
        patterns = []
        
        for i, candle in enumerate(candles):
            # Skip the first and last candle for better context
            if i == 0 or i == len(candles) - 1:
                continue
            
            # Get previous and next candles for context
            prev_candle = candles[i-1]
            
            # Doji pattern
            if candle.is_doji():
                # Calculate confidence based on how close the open and close are
                body_to_range_ratio = candle.body_size / candle.range_size if candle.range_size > 0 else 0
                confidence = max(0, 1.0 - (body_to_range_ratio * 10))  # Higher confidence for smaller body/range ratio
                
                # Determine if bullish or bearish based on context
                bullish = prev_candle.is_bearish()  # Doji after a bearish candle might indicate reversal
                
                patterns.append(PatternMatch(
                    pattern_name="Doji",
                    confidence=confidence,
                    candle_indices=[i],
                    bullish=bullish
                ))
            
            # Hammer pattern
            if (candle.lower_shadow > 2 * candle.body_size and 
                candle.upper_shadow < 0.2 * candle.lower_shadow and
                prev_candle.is_bearish()):
                
                confidence = min(1.0, candle.lower_shadow / (candle.body_size * 3))
                
                patterns.append(PatternMatch(
                    pattern_name="Hammer",
                    confidence=confidence,
                    candle_indices=[i],
                    bullish=True  # Hammer is a bullish reversal pattern
                ))
            
            # Shooting Star pattern
            if (candle.upper_shadow > 2 * candle.body_size and 
                candle.lower_shadow < 0.2 * candle.upper_shadow and
                prev_candle.is_bullish()):
                
                confidence = min(1.0, candle.upper_shadow / (candle.body_size * 3))
                
                patterns.append(PatternMatch(
                    pattern_name="Shooting Star",
                    confidence=confidence,
                    candle_indices=[i],
                    bullish=False  # Shooting Star is a bearish reversal pattern
                ))
            
            # Marubozu pattern (strong full-bodied candle)
            if (candle.upper_shadow < candle.body_size * 0.05 and
                candle.lower_shadow < candle.body_size * 0.05):
                
                confidence = 0.9  # High confidence for clear pattern
                
                patterns.append(PatternMatch(
                    pattern_name="Marubozu",
                    confidence=confidence,
                    candle_indices=[i],
                    bullish=candle.is_bullish()
                ))
            
            # Spinning Top
            if (candle.body_size < candle.range_size * 0.3 and
                candle.upper_shadow > candle.body_size * 0.5 and
                candle.lower_shadow > candle.body_size * 0.5):
                
                confidence = 0.7
                
                patterns.append(PatternMatch(
                    pattern_name="Spinning Top",
                    confidence=confidence,
                    candle_indices=[i],
                    bullish=None  # Neutral pattern
                ))
        
        return patterns
    
    def _detect_double_patterns(self, candles: List[Candle]) -> List[PatternMatch]:
        """Detect double candlestick patterns."""
        patterns = []
        
        for i in range(1, len(candles) - 1):
            # Get current and previous candle
            curr_candle = candles[i]
            prev_candle = candles[i-1]
            
            # Bullish Engulfing pattern
            if (prev_candle.is_bearish() and curr_candle.is_bullish() and
                curr_candle.open < prev_candle.close and
                curr_candle.close > prev_candle.open):
                
                # Calculate confidence based on how much the current candle engulfs the previous one
                engulfing_ratio = (curr_candle.body_size - prev_candle.body_size) / prev_candle.body_size if prev_candle.body_size > 0 else 0
                confidence = min(1.0, 0.7 + (engulfing_ratio * 0.3))  # Base 0.7 confidence + bonus for size
                
                patterns.append(PatternMatch(
                    pattern_name="Bullish Engulfing",
                    confidence=confidence,
                    candle_indices=[i-1, i],
                    bullish=True
                ))
            
            # Bearish Engulfing pattern
            if (prev_candle.is_bullish() and curr_candle.is_bearish() and
                curr_candle.open > prev_candle.close and
                curr_candle.close < prev_candle.open):
                
                # Calculate confidence
                engulfing_ratio = (curr_candle.body_size - prev_candle.body_size) / prev_candle.body_size if prev_candle.body_size > 0 else 0
                confidence = min(1.0, 0.7 + (engulfing_ratio * 0.3))
                
                patterns.append(PatternMatch(
                    pattern_name="Bearish Engulfing",
                    confidence=confidence,
                    candle_indices=[i-1, i],
                    bullish=False
                ))
            
            # Bullish Harami pattern
            if (prev_candle.is_bearish() and curr_candle.is_bullish() and
                curr_candle.open > prev_candle.close and
                curr_candle.close < prev_candle.open):
                
                # Calculate confidence
                harami_ratio = curr_candle.body_size / prev_candle.body_size if prev_candle.body_size > 0 else 0
                confidence = min(1.0, 0.6 + ((1 - harami_ratio) * 0.4))  # Higher confidence for smaller harami body
                
                patterns.append(PatternMatch(
                    pattern_name="Bullish Harami",
                    confidence=confidence,
                    candle_indices=[i-1, i],
                    bullish=True
                ))
            
            # Bearish Harami pattern
            if (prev_candle.is_bullish() and curr_candle.is_bearish() and
                curr_candle.open < prev_candle.close and
                curr_candle.close > prev_candle.open):
                
                # Calculate confidence
                harami_ratio = curr_candle.body_size / prev_candle.body_size if prev_candle.body_size > 0 else 0
                confidence = min(1.0, 0.6 + ((1 - harami_ratio) * 0.4))
                
                patterns.append(PatternMatch(
                    pattern_name="Bearish Harami",
                    confidence=confidence,
                    candle_indices=[i-1, i],
                    bullish=False
                ))
            
            # Tweezer Top
            if (prev_candle.is_bullish() and
                curr_candle.is_bearish() and
                abs(prev_candle.high - curr_candle.high) < prev_candle.range_size * 0.05):
                
                confidence = 0.75
                
                patterns.append(PatternMatch(
                    pattern_name="Tweezer Top",
                    confidence=confidence,
                    candle_indices=[i-1, i],
                    bullish=False
                ))
            
            # Tweezer Bottom
            if (prev_candle.is_bearish() and
                curr_candle.is_bullish() and
                abs(prev_candle.low - curr_candle.low) < prev_candle.range_size * 0.05):
                
                confidence = 0.75
                
                patterns.append(PatternMatch(
                    pattern_name="Tweezer Bottom",
                    confidence=confidence,
                    candle_indices=[i-1, i],
                    bullish=True
                ))
        
        return patterns
    
    def _detect_triple_patterns(self, candles: List[Candle]) -> List[PatternMatch]:
        """Detect triple candlestick patterns."""
        patterns = []
        
        for i in range(2, len(candles) - 1):
            # Get three consecutive candles
            first_candle = candles[i-2]
            second_candle = candles[i-1]
            third_candle = candles[i]
            
            # Morning Star pattern
            if (first_candle.is_bearish() and 
                first_candle.body_size > 0 and
                second_candle.body_size < first_candle.body_size * 0.3 and  # Small body
                third_candle.is_bullish() and
                third_candle.close > first_candle.body_mid_point):  # Closes above midpoint of first
                
                # Calculate confidence
                star_gap = min(
                    abs(first_candle.close - second_candle.open),
                    abs(second_candle.close - third_candle.open)
                )
                confidence = min(1.0, 0.7 + (star_gap / first_candle.body_size) * 0.3)
                
                patterns.append(PatternMatch(
                    pattern_name="Morning Star",
                    confidence=confidence,
                    candle_indices=[i-2, i-1, i],
                    bullish=True
                ))
            
            # Evening Star pattern
            if (first_candle.is_bullish() and 
                first_candle.body_size > 0 and
                second_candle.body_size < first_candle.body_size * 0.3 and  # Small body
                third_candle.is_bearish() and
                third_candle.close < first_candle.body_mid_point):  # Closes below midpoint of first
                
                # Calculate confidence
                star_gap = min(
                    abs(first_candle.close - second_candle.open),
                    abs(second_candle.close - third_candle.open)
                )
                confidence = min(1.0, 0.7 + (star_gap / first_candle.body_size) * 0.3)
                
                patterns.append(PatternMatch(
                    pattern_name="Evening Star",
                    confidence=confidence,
                    candle_indices=[i-2, i-1, i],
                    bullish=False
                ))
            
            # Three White Soldiers
            if (first_candle.is_bullish() and 
                second_candle.is_bullish() and 
                third_candle.is_bullish() and
                second_candle.close > first_candle.close and
                third_candle.close > second_candle.close and
                second_candle.open > first_candle.open and
                third_candle.open > second_candle.open):
                
                # Calculate confidence based on consistency of size
                avg_size = (first_candle.body_size + second_candle.body_size + third_candle.body_size) / 3
                size_variance = np.std([first_candle.body_size, second_candle.body_size, third_candle.body_size]) / avg_size
                confidence = min(1.0, 0.8 + (1 - size_variance) * 0.2)  # Higher confidence for consistent size
                
                patterns.append(PatternMatch(
                    pattern_name="Three White Soldiers",
                    confidence=confidence,
                    candle_indices=[i-2, i-1, i],
                    bullish=True
                ))
            
            # Three Black Crows
            if (first_candle.is_bearish() and 
                second_candle.is_bearish() and 
                third_candle.is_bearish() and
                second_candle.close < first_candle.close and
                third_candle.close < second_candle.close and
                second_candle.open < first_candle.open and
                third_candle.open < second_candle.open):
                
                # Calculate confidence
                avg_size = (first_candle.body_size + second_candle.body_size + third_candle.body_size) / 3
                size_variance = np.std([first_candle.body_size, second_candle.body_size, third_candle.body_size]) / avg_size
                confidence = min(1.0, 0.8 + (1 - size_variance) * 0.2)
                
                patterns.append(PatternMatch(
                    pattern_name="Three Black Crows",
                    confidence=confidence,
                    candle_indices=[i-2, i-1, i],
                    bullish=False
                ))
            
            # Three Inside Up
            if (first_candle.is_bearish() and
                second_candle.is_bullish() and
                second_candle.body_size < first_candle.body_size and
                second_candle.open > first_candle.close and
                second_candle.close < first_candle.open and
                third_candle.is_bullish() and
                third_candle.close > first_candle.open):
                
                confidence = 0.8
                
                patterns.append(PatternMatch(
                    pattern_name="Three Inside Up",
                    confidence=confidence,
                    candle_indices=[i-2, i-1, i],
                    bullish=True
                ))
            
            # Three Inside Down
            if (first_candle.is_bullish() and
                second_candle.is_bearish() and
                second_candle.body_size < first_candle.body_size and
                second_candle.open < first_candle.close and
                second_candle.close > first_candle.open and
                third_candle.is_bearish() and
                third_candle.close < first_candle.open):
                
                confidence = 0.8
                
                patterns.append(PatternMatch(
                    pattern_name="Three Inside Down",
                    confidence=confidence,
                    candle_indices=[i-2, i-1, i],
                    bullish=False
                ))
        
        return patterns
    
    def get_strategy_type(self) -> DetectionStrategy:
        """Get the detection strategy type."""
        return DetectionStrategy.RULE_BASED

class EnhancedModelBasedDetector(PatternDetector):
    """
    Enhanced pattern detector using machine learning models for pattern recognition.
    
    Features:
    1. Model loading with automatic ONNX optimization
    2. Multi-model ensemble support
    3. Confidence calibration
    4. Configurable input preprocessing
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 model_name: str = "ResNet50",
                 confidence_threshold: float = 0.6,
                 use_onnx: bool = True):
        """
        Initialize with model configuration.
        
        Args:
            model_path: Path to the trained model file
            model_name: Name of the model architecture
            confidence_threshold: Minimum confidence for pattern detection
            use_onnx: Whether to use ONNX runtime for inference
        """
        super().__init__()
        self.model_path = model_path
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.use_onnx = use_onnx
        self.model = None
        self.name = f"{model_name}Detector"
        
        # Load model if path is provided
        if model_path and os.path.exists(model_path):
            try:
                self._load_model()
            except Exception as e:
                logger.error(f"Failed to load model {model_name} from {model_path}: {e}")
    
    def _load_model(self):
        """Load the ML model for inference."""
        # This is a placeholder for model loading logic
        # In a real implementation, it would:
        # 1. Check if ONNX or original model format should be used
        # 2. Load the appropriate model
        # 3. Initialize preprocessing pipeline
        
        logger.info(f"Loading model {self.model_name} from {self.model_path}")
        # Placeholder - replace with actual model loading code
        # self.model = load_model(self.model_path, use_onnx=self.use_onnx)
        
        # For this example, we'll simulate model loading success
        self.model = True
    
    def _preprocess_data(self, data: CandlestickData) -> np.ndarray:
        """
        Preprocess candlestick data for model input.
        
        Args:
            data: Candlestick data
            
        Returns:
            Preprocessed numpy array ready for model input
        """
        # This is a placeholder for preprocessing logic
        # In a real implementation, it would:
        # 1. Convert candlestick data to images or feature vectors
        # 2. Normalize and reshape data for the model
        # 3. Apply any required augmentations
        
        # For this example, we'll create a dummy array
        return np.ones((1, 3, 224, 224), dtype=np.float32)  # Dummy input tensor
    
    def _postprocess_results(self, predictions: np.ndarray, 
                            data: CandlestickData) -> List[PatternMatch]:
        """
        Process model predictions into pattern matches.
        
        Args:
            predictions: Raw model predictions
            data: Original candlestick data
            
        Returns:
            List of pattern matches
        """
        # This is a placeholder for postprocessing logic
        # In a real implementation, it would:
        # 1. Convert raw predictions to pattern classes
        # 2. Apply confidence thresholds
        # 3. Map predictions to candle indices
        
        # For this example, we'll create dummy patterns
        patterns = []
        candles = data.candles
        
        if len(candles) < 3:
            return []
        
        # Simulate some pattern detections (in a real system these would come from the model)
        possible_patterns = [
            ("Bullish Engulfing", 0.85, [1, 2], True),
            ("Hammer", 0.78, [2], True),
            ("Evening Star", 0.72, [0, 1, 2], False)
        ]
        
        for name, conf, indices, bullish in possible_patterns:
            if conf >= self.confidence_threshold:
                patterns.append(PatternMatch(
                    pattern_name=name,
                    confidence=conf,
                    candle_indices=indices,
                    bullish=bullish,
                    detection_strategy=DetectionStrategy.ML_BASED
                ))
        
        return patterns
    
    def detect_patterns(self, data: CandlestickData) -> List[PatternMatch]:
        """
        Detect patterns using the machine learning model.
        
        Args:
            data: Candlestick data to analyze
            
        Returns:
            List of detected patterns
        """
        start_time = time.time()
        
        if not self.model:
            logger.warning(f"No model loaded for {self.name}")
            return []
        
        try:
            # Preprocess data for model input
            model_input = self._preprocess_data(data)
            
            # Run model prediction (placeholder)
            # In a real implementation, this would call the actual model
            # predictions = self.model.predict(model_input)
            
            # For this example, we'll skip the prediction and go straight to postprocessing
            patterns = self._postprocess_results(None, data)
            
            # Add detection time
            end_time = time.time()
            detection_time_ms = (end_time - start_time) * 1000
            
            for pattern in patterns:
                pattern.detection_time_ms = detection_time_ms
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error in ML-based pattern detection: {e}")
            return []
    
    def get_strategy_type(self) -> DetectionStrategy:
        """Get the detection strategy type."""
        return DetectionStrategy.ML_BASED

class AdvancedCompositeDetector(PatternDetector):
    """
    Advanced composite detector that combines multiple detection strategies
    with sophisticated consensus voting and concurrent execution.
    
    Features:
    1. Multi-threaded pattern detection for performance
    2. Weighted consensus voting system
    3. Redundancy with fallback
    4. Auto-tuning of detector weights based on historical performance
    """
    
    def __init__(self, 
                 detectors: List[PatternDetector],
                 thread_pool_size: int = 4,
                 consensus_threshold: float = 0.6,
                 timeout_seconds: float = 1.0):
        """
        Initialize the composite detector.
        
        Args:
            detectors: List of pattern detectors
            thread_pool_size: Size of thread pool for concurrent detection
            consensus_threshold: Threshold for pattern consensus
            timeout_seconds: Maximum time to wait for detectors
        """
        super().__init__()
        self.detectors = detectors
        self.thread_pool_size = thread_pool_size
        self.consensus_threshold = consensus_threshold
        self.timeout_seconds = timeout_seconds
        self.name = "AdvancedCompositeDetector"
        
        # Initialize thread pool for concurrent detector execution
        self._executor = ThreadPoolExecutor(max_workers=thread_pool_size)
        self._lock = threading.RLock()
        
        # Detector weights for consensus voting (starts with equal weights)
        self.detector_weights = {detector.name: 1.0 for detector in detectors}
        
        # Track performance for auto-tuning
        self.detector_performance = {detector.name: [] for detector in detectors}
    
    def __del__(self):
        """Cleanup resources on destruction."""
        self._executor.shutdown(wait=False)
    
    def detect_patterns(self, data: CandlestickData) -> List[PatternMatch]:
        """
        Detect patterns using all configured detectors concurrently.
        
        This implementation runs all detectors in parallel using a thread pool,
        then combines and ranks the results using a sophisticated consensus algorithm.
        
        Args:
            data: Candlestick data to analyze
            
        Returns:
            Combined and ranked list of detected patterns
        """
        start_time = time.time()
        all_patterns: List[PatternMatch] = []
        
        try:
            # Submit detector jobs to thread pool
            future_to_detector = {}
            for detector in self.detectors:
                future = self._executor.submit(detector.detect_patterns, data)
                future_to_detector[future] = detector
            
            # Collect results with timeout
            for future in concurrent.futures.as_completed(future_to_detector, timeout=self.timeout_seconds):
                detector = future_to_detector[future]
                try:
                    detector_patterns = future.result()
                    logger.debug(f"Detector {detector.name} found {len(detector_patterns)} patterns")
                    all_patterns.extend(detector_patterns)
                except Exception as e:
                    logger.error(f"Error in detector {detector.name}: {e}")
        
        except concurrent.futures.TimeoutError:
            logger.warning(f"Some detectors timed out after {self.timeout_seconds}s")
        
        # Apply consensus voting if we have multiple detectors
        if len(self.detectors) > 1:
            result_patterns = self._apply_consensus_voting(all_patterns)
        else:
            result_patterns = all_patterns
        
        # Sort by confidence (descending)
        result_patterns.sort(key=lambda p: p.confidence, reverse=True)
        
        # Add composite detection strategy and time
        detection_time_ms = (time.time() - start_time) * 1000
        for pattern in result_patterns:
            if not pattern.detection_strategy:
                pattern.detection_strategy = DetectionStrategy.HYBRID
            pattern.detection_time_ms = detection_time_ms
        
        return result_patterns
    
    def _apply_consensus_voting(self, patterns: List[PatternMatch]) -> List[PatternMatch]:
        """
        Apply a weighted consensus voting algorithm to reconcile patterns from multiple detectors.
        
        This algorithm:
        1. Groups patterns by name and candle indices
        2. Applies weighted voting based on detector performance
        3. Merges duplicate detections with confidence adjustment
        
        Args:
            patterns: All patterns from all detectors
            
        Returns:
            Consensus pattern list with duplicates merged
        """
        if not patterns:
            return []
        
        # Group patterns by name and indices
        pattern_groups = {}
        for pattern in patterns:
            # Create a key based on pattern name and candle indices
            indices_key = tuple(sorted(pattern.candle_indices))
            group_key = (pattern.pattern_name, indices_key)
            
            if group_key not in pattern_groups:
                pattern_groups[group_key] = []
            
            pattern_groups[group_key].append(pattern)
        
        # Apply consensus algorithm to each group
        consensus_patterns = []
        for (name, indices), group in pattern_groups.items():
            # If only one detector found this pattern, use its confidence
            if len(group) == 1:
                consensus_patterns.append(group[0])
                continue
            
            # Multiple detectors found this pattern - apply weighted voting
            total_weight = 0.0
            weighted_confidence = 0.0
            bullish_votes = 0
            bearish_votes = 0
            
            for pattern in group:
                detector_name = pattern.detection_strategy.value if pattern.detection_strategy else "unknown"
                weight = self.detector_weights.get(detector_name, 1.0)
                total_weight += weight
                weighted_confidence += pattern.confidence * weight
                
                if pattern.bullish:
                    bullish_votes += weight
                else:
                    bearish_votes += weight
            
            # Calculate consensus values
            consensus_confidence = weighted_confidence / total_weight if total_weight > 0 else 0.0
            consensus_bullish = bullish_votes > bearish_votes
            
            # Only include if consensus confidence meets threshold
            if consensus_confidence >= self.consensus_threshold:
                consensus_pattern = PatternMatch(
                    pattern_name=name,
                    confidence=consensus_confidence,
                    candle_indices=list(indices),
                    bullish=consensus_bullish,
                    detection_strategy=DetectionStrategy.HYBRID,
                    description=f"Consensus from {len(group)} detectors"
                )
                consensus_patterns.append(consensus_pattern)
        
        return consensus_patterns
    
    def update_detector_weights(self, performance_data: Dict[str, float]):
        """
        Update detector weights based on performance feedback.
        
        Args:
            performance_data: Dictionary mapping detector names to performance scores
        """
        with self._lock:
            # Update performance history
            for detector_name, score in performance_data.items():
                if detector_name in self.detector_performance:
                    self.detector_performance[detector_name].append(score)
                    
                    # Keep only the last 100 scores
                    if len(self.detector_performance[detector_name]) > 100:
                        self.detector_performance[detector_name] = self.detector_performance[detector_name][-100:]
            
            # Recalculate weights based on average performance
            for detector_name, scores in self.detector_performance.items():
                if scores:
                    avg_score = sum(scores) / len(scores)
                    self.detector_weights[detector_name] = max(0.1, min(2.0, avg_score))
                else:
                    self.detector_weights[detector_name] = 1.0
    
    def get_strategy_type(self) -> DetectionStrategy:
        """Get the detection strategy type."""
        return DetectionStrategy.HYBRID

@lru_cache(maxsize=8)
def create_default_detector() -> PatternDetector:
    """
    Create a default pattern detector with optimal configuration.
    This function is cached to provide efficient reuse of detector instances.
    
    Returns:
        A configured pattern detector ready to use
    """
    logger.info("Creating new instance of default pattern detector")
    
    # Create rule-based detector variants
    primary_rule_detector = RuleBasedDetector(name="PrimaryRuleBased")
    conservative_rule_detector = RuleBasedDetector(name="ConservativeRuleBased")
    
    # Create model-based detector (simulated)
    model_detector = EnhancedModelBasedDetector(
        model_name="ResNet50",
        confidence_threshold=0.65,
        use_onnx=True
    )
    
    # Create advanced composite detector with all components
    composite_detector = AdvancedCompositeDetector(
        detectors=[primary_rule_detector, conservative_rule_detector, model_detector],
        thread_pool_size=4,
        consensus_threshold=0.6,
        timeout_seconds=0.5
    )
    
    return composite_detector

# Singleton instance storage with weakref to allow for garbage collection when not in use
_default_detector_ref = None

def get_default_detector() -> PatternDetector:
    """
    Get the singleton instance of the default detector.
    This implementation uses weak references to allow proper garbage collection
    while still maintaining the benefits of a singleton pattern.
    
    Returns:
        The default pattern detector instance
    """
    global _default_detector_ref
    
    # Check if we have a valid reference
    detector = None if _default_detector_ref is None else _default_detector_ref()
    
    # If no valid reference, create a new instance and store a weak reference
    if detector is None:
        detector = create_default_detector()
        _default_detector_ref = weakref.ref(detector)
        logger.debug("Created new default detector instance")
    else:
        logger.debug("Reusing existing default detector instance")
    
    return detector

# Factory function for creating specialized detectors
def create_detector(strategy_type: DetectionStrategy = DetectionStrategy.HYBRID, 
                   **kwargs) -> PatternDetector:
    """
    Factory function to create pattern detectors based on strategy type.
    
    Args:
        strategy_type: Type of detection strategy to use
        **kwargs: Additional parameters for the detector
        
    Returns:
        Configured pattern detector
    """
    try:
        if strategy_type == DetectionStrategy.RULE_BASED:
            return RuleBasedDetector(**kwargs)
        elif strategy_type == DetectionStrategy.ML_BASED:
            return EnhancedModelBasedDetector(**kwargs)
        else:  # HYBRID or fallback
            # Create component detectors
            rule_based = RuleBasedDetector(name=kwargs.get("rule_based_name", "RuleBased"))
            ml_based = EnhancedModelBasedDetector(
                model_name=kwargs.get("model_name", "ResNet50"),
                confidence_threshold=kwargs.get("confidence_threshold", 0.65)
            )
            
            # Create composite detector
            return AdvancedCompositeDetector(
                detectors=[rule_based, ml_based],
                thread_pool_size=kwargs.get("thread_pool_size", 4),
                consensus_threshold=kwargs.get("consensus_threshold", 0.6),
                timeout_seconds=kwargs.get("timeout_seconds", 0.5)
            )
    except Exception as e:
        logger.error(f"Error creating detector with strategy {strategy_type}: {e}")
        # Return a simple rule-based detector as fallback
        return RuleBasedDetector(name="FallbackDetector") 