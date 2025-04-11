"""
Pattern Detection Interface

This module defines the core interfaces and base classes for candlestick pattern detection.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from backend.common.finance.patterns import (
    PatternType, PatternStrength, CandlestickPattern, PatternRecognitionResult
)
from backend.assessments.candlestick_patterns.candlestick_utils import (
    Candle, CandlestickData
)
from backend.common.serialization import SerializableMixin


class DetectionStrategy(Enum):
    """Enumeration of pattern detection strategy types."""
    RULE_BASED = "rule_based"
    GEOMETRIC = "geometric"
    STATISTICAL = "statistical"
    ML_BASED = "ml_based"
    CNN = "cnn"
    RESNET = "resnet"
    EFFICIENTNET = "efficientnet"
    ENSEMBLE = "ensemble"
    WEIGHTED_CONSENSUS = "weighted_consensus"
    HYBRID = "hybrid"


@dataclass
class PatternMatch(SerializableMixin):
    """
    Represents a detected candlestick pattern with metadata about the detection.
    
    Attributes:
        pattern_name: Name of the detected pattern
        confidence: Detection confidence score (0.0-1.0)
        candle_indices: Indices of candles involved in the pattern
        bullish: True for bullish patterns, False for bearish, None for neutral
        description: Optional description of the pattern
        detection_strategy: Strategy used to detect this pattern
        detection_time_ms: Time taken to detect this pattern in milliseconds
        metadata: Additional pattern-specific metadata
    """
    pattern_name: str
    confidence: float
    candle_indices: List[int]
    bullish: Optional[bool] = None
    description: Optional[str] = None
    detection_strategy: Optional[DetectionStrategy] = None
    detection_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate and normalize attributes after initialization."""
        # Validate pattern name
        if not self.pattern_name:
            raise ValueError("Pattern name cannot be empty")
            
        # Ensure confidence is between 0 and 1
        self.confidence = max(0.0, min(1.0, self.confidence))
        
        # Ensure candle_indices is a list
        if not isinstance(self.candle_indices, list):
            self.candle_indices = [int(self.candle_indices)] if self.candle_indices is not None else []
            
        # Ensure metadata is a dictionary
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.
        
        Returns:
            Dictionary representation of pattern match
        """
        return {
            "pattern_name": self.pattern_name,
            "confidence": self.confidence,
            "candle_indices": self.candle_indices,
            "bullish": self.bullish,
            "description": self.description,
            "detection_strategy": self.detection_strategy.value if self.detection_strategy else None,
            "detection_time_ms": self.detection_time_ms,
            "metadata": self.metadata.copy() if self.metadata else {}
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PatternMatch':
        """
        Create a PatternMatch from a dictionary.
        
        Args:
            data: Dictionary containing pattern match data
            
        Returns:
            New PatternMatch instance
            
        Raises:
            ValueError: If pattern_name is missing or invalid
        """
        if not data:
            raise ValueError("Cannot create PatternMatch from empty data")
            
        data_copy = data.copy()
        
        # Required field validation
        if "pattern_name" not in data_copy or not data_copy["pattern_name"]:
            raise ValueError("Pattern name is required")
        
        # Set defaults for missing fields
        if "confidence" not in data_copy:
            data_copy["confidence"] = 0.5
        if "candle_indices" not in data_copy:
            data_copy["candle_indices"] = []
            
        # Handle strategy conversion
        strategy_str = data_copy.pop("detection_strategy", None)
        strategy = None
        if strategy_str:
            try:
                strategy = DetectionStrategy(strategy_str)
            except ValueError:
                # If invalid strategy, don't raise error but log and continue with None
                import logging
                logging.warning(f"Unknown detection strategy: {strategy_str}")
        
        # Get metadata or default to empty dict
        metadata = data_copy.pop("metadata", {}) or {}
        
        # Remove any unexpected fields to avoid __init__ errors
        expected_fields = ["pattern_name", "confidence", "candle_indices", 
                           "bullish", "description", "detection_time_ms"]
        for key in list(data_copy.keys()):
            if key not in expected_fields:
                data_copy.pop(key)
                
        return cls(**data_copy, detection_strategy=strategy, metadata=metadata)
    
    def to_candlestick_pattern(
        self, 
        symbol: str, 
        candles: List[Candle], 
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> CandlestickPattern:
        """
        Convert this PatternMatch to a CandlestickPattern.
        
        Args:
            symbol: The symbol/ticker for the market data
            candles: The full list of candles
            start_time: Optional override for pattern start time
            end_time: Optional override for pattern end time
            
        Returns:
            A CandlestickPattern object
            
        Raises:
            ValueError: If candles list is empty or invalid indices
        """
        if not candles:
            raise ValueError("Cannot convert to CandlestickPattern: empty candles list")
            
        if not symbol:
            symbol = "UNKNOWN"
            
        # Map pattern name to PatternType
        try:
            pattern_type = PatternType(self.pattern_name.lower())
        except ValueError:
            # Use a fallback for custom pattern types
            pattern_type = None
            for pt in PatternType:
                if pt.value in self.pattern_name.lower():
                    pattern_type = pt
                    break
            if pattern_type is None:
                pattern_type = PatternType.DOJI  # Default fallback
        
        # Determine pattern strength from confidence
        strength = PatternStrength.from_score(self.confidence)
        
        # Get candles for this pattern using indices
        pattern_indices = []
        for idx in self.candle_indices:
            if 0 <= idx < len(candles):
                pattern_indices.append(idx)
                
        if not pattern_indices:
            # Fallback if no valid indices
            pattern_indices = [0] if candles else []
        
        # Sort indices
        pattern_indices = sorted(pattern_indices)
        
        # Get pattern candles
        pattern_candles = [candles[i] for i in pattern_indices]
        
        # Determine start and end times if not provided
        if start_time is None and pattern_candles:
            try:
                start_time = datetime.fromtimestamp(pattern_candles[0].time)
            except (ValueError, TypeError, OverflowError):
                start_time = datetime.now()
        elif start_time is None:
            start_time = datetime.now()
            
        if end_time is None and pattern_candles:
            try:
                end_time = datetime.fromtimestamp(pattern_candles[-1].time)
            except (ValueError, TypeError, OverflowError):
                end_time = datetime.now()
        elif end_time is None:
            end_time = datetime.now()
        
        # Convert Candle objects to Candlestick objects
        from backend.common.finance.candlestick import Candlestick
        candlesticks = []
        for candle in pattern_candles:
            try:
                cs = Candlestick(
                    timestamp=datetime.fromtimestamp(candle.time),
                    open=candle.open,
                    high=candle.high,
                    low=candle.low,
                    close=candle.close,
                    volume=candle.volume
                )
                candlesticks.append(cs)
            except Exception as e:
                import logging
                logging.warning(f"Error converting candle to Candlestick: {e}")
        
        # Create and return the CandlestickPattern
        return CandlestickPattern(
            pattern_type=pattern_type,
            symbol=symbol,
            start_time=start_time,
            end_time=end_time,
            candles=candlesticks,
            strength=strength,
            confidence=self.confidence,
            expected_direction="up" if self.bullish else "down" if self.bullish is False else "neutral",
            metadata={
                "detection_strategy": self.detection_strategy.value if self.detection_strategy else None,
                "detection_time_ms": self.detection_time_ms,
                "description": self.description,
                **(self.metadata or {})
            }
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
            
        Raises:
            ValueError: If pattern is None
        """
        if pattern is None:
            raise ValueError("Cannot convert from None CandlestickPattern")
            
        # Use provided indices or default to range
        indices = candle_indices or list(range(len(pattern.candles or [])))
        
        # Determine strategy from metadata if available
        strategy_str = pattern.metadata.get("detection_strategy") if pattern.metadata else None
        strategy = None
        if strategy_str:
            try:
                strategy = DetectionStrategy(strategy_str)
            except ValueError:
                # If invalid strategy, leave as None
                pass
        
        # Determine bullish/bearish direction
        bullish = None
        if hasattr(pattern, 'expected_direction'):
            if pattern.expected_direction == "up":
                bullish = True
            elif pattern.expected_direction == "down":
                bullish = False
        
        # Extract other metadata
        metadata = {}
        if pattern.metadata:
            metadata = pattern.metadata.copy()
            # Remove fields we handle explicitly
            for key in ["detection_strategy", "detection_time_ms", "description"]:
                if key in metadata:
                    metadata.pop(key)
        
        return cls(
            pattern_name=pattern.pattern_type.value,
            confidence=getattr(pattern, 'confidence', 0.5),
            candle_indices=indices,
            bullish=bullish,
            description=pattern.metadata.get("description") if pattern.metadata else None,
            detection_strategy=strategy,
            detection_time_ms=pattern.metadata.get("detection_time_ms", 0.0) if pattern.metadata else 0.0,
            metadata=metadata
        )


class PatternDetector(ABC):
    """
    Abstract base class for all candlestick pattern detectors.
    
    This class defines the interface that all concrete pattern detector 
    implementations must implement. It provides the foundation for 
    implementing different detection strategies.
    """
    
    def __init__(
        self, 
        name: str,
        strategy: DetectionStrategy,
        min_confidence: float = 0.5,
        max_patterns: Optional[int] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the pattern detector.
        
        Args:
            name: Unique name for this detector
            strategy: Detection strategy used by this detector
            min_confidence: Minimum confidence threshold for patterns (0.0-1.0)
            max_patterns: Maximum number of patterns to return (None for unlimited)
            config: Additional configuration parameters
        
        Raises:
            ValueError: If name is empty or min_confidence is not between 0 and 1
        """
        if not name:
            raise ValueError("Detector name cannot be empty")
            
        self.name = name
        self.strategy = strategy
        self.min_confidence = max(0.0, min(1.0, min_confidence))
        self.max_patterns = max_patterns
        self.config = config or {}
        self._initialized = False
        self._error_count = 0
        self._last_error = None
    
    async def initialize(self) -> bool:
        """
        Initialize the detector with any required resources.
        
        This method should be called before using the detector to ensure
        all required resources are loaded and the detector is ready to use.
        Default implementation marks detector as initialized.
        
        Returns:
            True if initialization succeeded, False otherwise
        """
        self._initialized = True
        return True
    
    def is_initialized(self) -> bool:
        """
        Check if the detector has been initialized.
        
        Returns:
            True if detector is initialized, False otherwise
        """
        return self._initialized
    
    @abstractmethod
    async def detect_patterns(
        self, 
        candlestick_data: CandlestickData
    ) -> List[PatternMatch]:
        """
        Detect patterns in the provided candlestick data.
        
        Args:
            candlestick_data: The candlestick data to analyze
            
        Returns:
            List of detected patterns as PatternMatch objects
            
        Raises:
            ValueError: If candlestick_data is invalid
            NotInitializedError: If detector has not been initialized
        """
        pass
    
    async def detect_patterns_safe(
        self, 
        candlestick_data: CandlestickData
    ) -> Tuple[List[PatternMatch], Optional[Exception]]:
        """
        Safely detect patterns with error handling.
        
        This method wraps the detect_patterns method with error handling,
        ensuring that errors don't propagate upward and providing a 
        consistent return value even in error cases.
        
        Args:
            candlestick_data: The candlestick data to analyze
            
        Returns:
            Tuple of (patterns list, exception if any occurred)
        """
        try:
            if not self.is_initialized():
                await self.initialize()
                
            if not candlestick_data or not candlestick_data.candles:
                return [], ValueError("Empty candlestick data")
                
            return await self.detect_patterns(candlestick_data), None
        except Exception as e:
            import logging
            self._error_count += 1
            self._last_error = e
            logging.exception(f"Error in {self.name} detector: {str(e)}")
            return [], e
    
    def filter_patterns(
        self, 
        patterns: List[PatternMatch]
    ) -> List[PatternMatch]:
        """
        Filter patterns based on confidence threshold and max patterns limit.
        
        Args:
            patterns: List of detected patterns
            
        Returns:
            Filtered list of patterns
        """
        # Filter by confidence
        filtered = [p for p in patterns if p.confidence >= self.min_confidence]
        
        # Sort by confidence (descending)
        filtered.sort(key=lambda p: p.confidence, reverse=True)
        
        # Limit to max_patterns if specified
        if self.max_patterns is not None and self.max_patterns > 0:
            filtered = filtered[:self.max_patterns]
            
        return filtered
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get detector status information.
        
        Returns:
            Dictionary with detector status
        """
        return {
            "name": self.name,
            "strategy": self.strategy.value,
            "initialized": self._initialized,
            "min_confidence": self.min_confidence,
            "max_patterns": self.max_patterns,
            "error_count": self._error_count,
            "last_error": str(self._last_error) if self._last_error else None
        }
        
    def update_configuration(self, config_updates: Dict[str, Any]) -> None:
        """
        Update detector configuration parameters.
        
        Args:
            config_updates: Dictionary of configuration updates
            
        Raises:
            ValueError: If an invalid configuration parameter is provided
        """
        if "min_confidence" in config_updates:
            value = config_updates.pop("min_confidence")
            if not isinstance(value, (int, float)) or value < 0 or value > 1:
                raise ValueError("min_confidence must be between 0 and 1")
            self.min_confidence = float(value)
            
        if "max_patterns" in config_updates:
            value = config_updates.pop("max_patterns")
            if value is not None and (not isinstance(value, int) or value < 0):
                raise ValueError("max_patterns must be a positive integer or None")
            self.max_patterns = value
            
        # Update remaining configuration
        self.config.update(config_updates)
        
    def __str__(self) -> str:
        """String representation of the detector."""
        return f"{self.name} ({self.strategy.value})"
        
    def __repr__(self) -> str:
        """Detailed string representation of the detector."""
        return f"PatternDetector(name='{self.name}', strategy={self.strategy}, min_confidence={self.min_confidence})"

    def get_strategy_type(self) -> DetectionStrategy:
        """
        Get the detection strategy type for this detector.
        
        Returns:
            The strategy type enum value
        """
        return self.strategy
    
    @staticmethod
    def to_recognition_result(
        candlestick_data: CandlestickData,
        detected_patterns: List[PatternMatch],
        execution_time_ms: float = 0.0,
        strategy_type: Optional[DetectionStrategy] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> PatternRecognitionResult:
        """
        Converts a list of PatternMatch objects to a PatternRecognitionResult.
        
        Creates a standardized result object from detection results that can be
        used for further processing or API responses.
        
        Args:
            candlestick_data: The candlestick data that was analyzed
            detected_patterns: List of detected patterns
            execution_time_ms: Total execution time in milliseconds
            strategy_type: Strategy type used for detection (optional)
            metadata: Additional metadata to include in result (optional)
            
        Returns:
            A PatternRecognitionResult object
            
        Raises:
            ValueError: If candlestick_data is None
        """
        if candlestick_data is None:
            raise ValueError("Cannot create recognition result: candlestick_data is None")
        
        # Handle empty candlestick data gracefully
        candles = candlestick_data.candles or []
        
        # Determine start and end times
        start_time = None
        end_time = None
        
        if candles:
            try:
                start_time = datetime.fromtimestamp(candles[0].time)
                end_time = datetime.fromtimestamp(candles[-1].time)
            except (ValueError, TypeError, IndexError, OverflowError) as e:
                import logging
                logging.warning(f"Error determining time range from candles: {e}")
                # Fallback to current time
                start_time = datetime.now()
                end_time = datetime.now()
        else:
            # Default to current time if no candles available
            start_time = datetime.now()
            end_time = datetime.now()
        
        # Ensure we have a symbol
        symbol = candlestick_data.symbol or "UNKNOWN"
        
        # Determine timeframe
        timeframe = candlestick_data.timeframe or "unknown"
        
        # Convert PatternMatch objects to CandlestickPattern objects
        patterns = []
        for pattern_match in (detected_patterns or []):
            try:
                candlestick_pattern = pattern_match.to_candlestick_pattern(
                    symbol=symbol,
                    candles=candles,
                    start_time=start_time,
                    end_time=end_time
                )
                patterns.append(candlestick_pattern)
            except Exception as e:
                import logging
                logging.exception(f"Error converting pattern match to candlestick pattern: {e}")
                # Continue with next pattern
                continue
        
        # Create composite metadata
        result_metadata = {
            "execution_time_ms": execution_time_ms,
            "candle_count": len(candles),
            "pattern_count": len(patterns),
            "detection_time": datetime.now().isoformat(),
            "timeframe": timeframe,
        }
        
        # Add strategy type if provided
        if strategy_type is not None:
            result_metadata["strategy_type"] = strategy_type.value
        
        # Add user metadata if provided
        if metadata:
            result_metadata.update(metadata)
        
        # Create and return the result
        return PatternRecognitionResult(
            symbol=symbol,
            patterns=patterns,
            start_time=start_time,
            end_time=end_time,
            metadata=result_metadata
        ) 