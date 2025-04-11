"""
Candlestick Pattern Models

This module defines data models for candlestick patterns and pattern recognition,
enabling the identification and analysis of trading patterns from candlestick data.
"""

import datetime
import enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Union

from backend.common.serialization import SerializableMixin
from backend.common.finance.candlestick import Candlestick, CandlestickSeries


class PatternType(enum.Enum):
    """Types of candlestick patterns."""
    
    # Single candlestick patterns
    DOJI = "doji"
    HAMMER = "hammer"
    INVERTED_HAMMER = "inverted_hammer"
    HANGING_MAN = "hanging_man"
    SHOOTING_STAR = "shooting_star"
    SPINNING_TOP = "spinning_top"
    MARUBOZU = "marubozu"
    
    # Double candlestick patterns
    BULLISH_ENGULFING = "bullish_engulfing"
    BEARISH_ENGULFING = "bearish_engulfing"
    TWEEZER_TOP = "tweezer_top"
    TWEEZER_BOTTOM = "tweezer_bottom"
    PIERCING_LINE = "piercing_line"
    DARK_CLOUD_COVER = "dark_cloud_cover"
    
    # Triple candlestick patterns
    MORNING_STAR = "morning_star"
    EVENING_STAR = "evening_star"
    THREE_WHITE_SOLDIERS = "three_white_soldiers"
    THREE_BLACK_CROWS = "three_black_crows"
    THREE_INSIDE_UP = "three_inside_up"
    THREE_INSIDE_DOWN = "three_inside_down"
    
    # Complex patterns
    HEAD_AND_SHOULDERS = "head_and_shoulders"
    INVERSE_HEAD_AND_SHOULDERS = "inverse_head_and_shoulders"
    DOUBLE_TOP = "double_top"
    DOUBLE_BOTTOM = "double_bottom"
    TRIANGLE = "triangle"
    WEDGE = "wedge"
    FLAG = "flag"
    PENNANT = "pennant"
    
    @property
    def is_bullish(self) -> bool:
        """Check if this pattern is typically bullish."""
        bullish_patterns = {
            PatternType.HAMMER,
            PatternType.INVERTED_HAMMER,
            PatternType.BULLISH_ENGULFING,
            PatternType.PIERCING_LINE,
            PatternType.MORNING_STAR,
            PatternType.THREE_WHITE_SOLDIERS,
            PatternType.THREE_INSIDE_UP,
            PatternType.INVERSE_HEAD_AND_SHOULDERS,
            PatternType.DOUBLE_BOTTOM
        }
        return self in bullish_patterns
    
    @property
    def is_bearish(self) -> bool:
        """Check if this pattern is typically bearish."""
        bearish_patterns = {
            PatternType.HANGING_MAN,
            PatternType.SHOOTING_STAR,
            PatternType.BEARISH_ENGULFING,
            PatternType.DARK_CLOUD_COVER,
            PatternType.EVENING_STAR,
            PatternType.THREE_BLACK_CROWS,
            PatternType.THREE_INSIDE_DOWN,
            PatternType.HEAD_AND_SHOULDERS,
            PatternType.DOUBLE_TOP
        }
        return self in bearish_patterns
    
    @property
    def is_neutral(self) -> bool:
        """Check if this pattern is typically neutral."""
        return not (self.is_bullish or self.is_bearish)
    
    @property
    def min_candles(self) -> int:
        """Get the minimum number of candlesticks required for this pattern."""
        single_patterns = {
            PatternType.DOJI,
            PatternType.HAMMER,
            PatternType.INVERTED_HAMMER,
            PatternType.HANGING_MAN,
            PatternType.SHOOTING_STAR,
            PatternType.SPINNING_TOP,
            PatternType.MARUBOZU
        }
        
        double_patterns = {
            PatternType.BULLISH_ENGULFING,
            PatternType.BEARISH_ENGULFING,
            PatternType.TWEEZER_TOP,
            PatternType.TWEEZER_BOTTOM,
            PatternType.PIERCING_LINE,
            PatternType.DARK_CLOUD_COVER
        }
        
        triple_patterns = {
            PatternType.MORNING_STAR,
            PatternType.EVENING_STAR,
            PatternType.THREE_WHITE_SOLDIERS,
            PatternType.THREE_BLACK_CROWS,
            PatternType.THREE_INSIDE_UP,
            PatternType.THREE_INSIDE_DOWN
        }
        
        if self in single_patterns:
            return 1
        elif self in double_patterns:
            return 2
        elif self in triple_patterns:
            return 3
        else:
            # Complex patterns need more candlesticks
            return 5


class PatternStrength(enum.Enum):
    """Strength of a pattern signal."""
    
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"
    
    @property
    def score(self) -> float:
        """Get a numeric score for this strength level."""
        if self == PatternStrength.WEAK:
            return 0.25
        elif self == PatternStrength.MODERATE:
            return 0.5
        elif self == PatternStrength.STRONG:
            return 0.75
        elif self == PatternStrength.VERY_STRONG:
            return 1.0
        else:
            return 0.0
    
    @classmethod
    def from_score(cls, score: float) -> 'PatternStrength':
        """Get a strength level from a numeric score."""
        if score < 0.3:
            return cls.WEAK
        elif score < 0.6:
            return cls.MODERATE
        elif score < 0.8:
            return cls.STRONG
        else:
            return cls.VERY_STRONG


@dataclass
class CandlestickPattern(SerializableMixin):
    """
    A candlestick pattern identified in market data.
    
    Attributes:
        pattern_type: The type of pattern
        symbol: The symbol/ticker where the pattern was found
        start_time: The start time of the pattern
        end_time: The end time of the pattern
        candles: The candlesticks forming the pattern
        strength: The strength of the pattern signal
        confidence: Confidence score for the pattern (0-1)
        trend_before: The trend before the pattern (up, down, sideways)
        expected_direction: The expected price direction after the pattern
        metadata: Additional metadata about the pattern
    """
    pattern_type: PatternType
    symbol: str
    start_time: datetime.datetime
    end_time: datetime.datetime
    candles: List[Candlestick]
    strength: PatternStrength = PatternStrength.MODERATE
    confidence: float = 0.5
    trend_before: Optional[str] = None
    expected_direction: Optional[str] = None
    metadata: Dict = field(default_factory=dict)
    
    @property
    def duration(self) -> datetime.timedelta:
        """Get the duration of the pattern."""
        return self.end_time - self.start_time
    
    @property
    def candle_count(self) -> int:
        """Get the number of candlesticks in the pattern."""
        return len(self.candles)
    
    @property
    def is_bullish(self) -> bool:
        """Check if this is a bullish pattern."""
        if self.expected_direction:
            return self.expected_direction.lower() == "up"
        return self.pattern_type.is_bullish
    
    @property
    def is_bearish(self) -> bool:
        """Check if this is a bearish pattern."""
        if self.expected_direction:
            return self.expected_direction.lower() == "down"
        return self.pattern_type.is_bearish
    
    def to_dict(self) -> Dict:
        """Convert the pattern to a dictionary."""
        return {
            "pattern_type": self.pattern_type.value,
            "symbol": self.symbol,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "candles": [c.to_dict() for c in self.candles],
            "strength": self.strength.value,
            "confidence": self.confidence,
            "trend_before": self.trend_before,
            "expected_direction": self.expected_direction,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'CandlestickPattern':
        """Create a pattern from a dictionary."""
        # Convert pattern type and strength
        pattern_type = PatternType(data["pattern_type"])
        strength = PatternStrength(data["strength"])
        
        # Convert timestamps
        start_time = datetime.datetime.fromisoformat(data["start_time"])
        end_time = datetime.datetime.fromisoformat(data["end_time"])
        
        # Convert candlesticks
        candles = [Candlestick.from_dict(c) for c in data["candles"]]
        
        return cls(
            pattern_type=pattern_type,
            symbol=data["symbol"],
            start_time=start_time,
            end_time=end_time,
            candles=candles,
            strength=strength,
            confidence=data["confidence"],
            trend_before=data.get("trend_before"),
            expected_direction=data.get("expected_direction"),
            metadata=data.get("metadata", {})
        )
    
    @property
    def serializable_fields(self) -> List[str]:
        """Get a list of fields that should be serialized."""
        return [
            "pattern_type", "symbol", "start_time", "end_time", "candles",
            "strength", "confidence", "trend_before", "expected_direction", "metadata"
        ]


@dataclass
class PatternRecognitionResult(SerializableMixin):
    """
    Result of a pattern recognition analysis on a candlestick series.
    
    Attributes:
        symbol: The symbol/ticker that was analyzed
        patterns: List of patterns found
        analyzed_period: The time period that was analyzed
        total_candles: Total number of candlesticks analyzed
        execution_time: Time taken to perform the analysis (in seconds)
        algorithm: The algorithm used for pattern recognition
        metadata: Additional metadata about the analysis
    """
    symbol: str
    patterns: List[CandlestickPattern] = field(default_factory=list)
    analyzed_period: Tuple[datetime.datetime, datetime.datetime] = field(default_factory=tuple)
    total_candles: int = 0
    execution_time: float = 0.0
    algorithm: str = "default"
    metadata: Dict = field(default_factory=dict)
    
    @property
    def pattern_count(self) -> int:
        """Get the number of patterns found."""
        return len(self.patterns)
    
    @property
    def pattern_types(self) -> Set[PatternType]:
        """Get the set of pattern types found."""
        return {p.pattern_type for p in self.patterns}
    
    @property
    def pattern_density(self) -> float:
        """Get the pattern density (patterns per candle)."""
        if self.total_candles == 0:
            return 0.0
        return self.pattern_count / self.total_candles
    
    def get_patterns_by_type(self, pattern_type: Union[PatternType, str]) -> List[CandlestickPattern]:
        """
        Get all patterns of a specific type.
        
        Args:
            pattern_type: The pattern type to filter by
            
        Returns:
            List of patterns of the specified type
        """
        # Convert string to enum if needed
        if isinstance(pattern_type, str):
            pattern_type = PatternType(pattern_type)
            
        return [p for p in self.patterns if p.pattern_type == pattern_type]
    
    def get_patterns_by_time(
        self,
        start_time: Optional[datetime.datetime] = None,
        end_time: Optional[datetime.datetime] = None
    ) -> List[CandlestickPattern]:
        """
        Get all patterns within a specific time range.
        
        Args:
            start_time: The start time to filter by (inclusive)
            end_time: The end time to filter by (inclusive)
            
        Returns:
            List of patterns within the time range
        """
        filtered = self.patterns
        
        if start_time:
            filtered = [p for p in filtered if p.end_time >= start_time]
        
        if end_time:
            filtered = [p for p in filtered if p.start_time <= end_time]
            
        return filtered
    
    def get_strongest_patterns(self, count: int = 5) -> List[CandlestickPattern]:
        """
        Get the strongest patterns found.
        
        Args:
            count: The number of patterns to return
            
        Returns:
            List of the strongest patterns
        """
        # Sort by confidence and strength
        sorted_patterns = sorted(
            self.patterns,
            key=lambda p: (p.confidence, p.strength.score),
            reverse=True
        )
        
        return sorted_patterns[:count]
    
    def to_dict(self) -> Dict:
        """Convert the result to a dictionary."""
        return {
            "symbol": self.symbol,
            "patterns": [p.to_dict() for p in self.patterns],
            "analyzed_period": [
                self.analyzed_period[0].isoformat() if self.analyzed_period and len(self.analyzed_period) > 0 else None,
                self.analyzed_period[1].isoformat() if self.analyzed_period and len(self.analyzed_period) > 1 else None
            ],
            "total_candles": self.total_candles,
            "execution_time": self.execution_time,
            "algorithm": self.algorithm,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'PatternRecognitionResult':
        """Create a result from a dictionary."""
        # Convert patterns
        patterns = [CandlestickPattern.from_dict(p) for p in data.get("patterns", [])]
        
        # Convert analyzed period
        analyzed_period = data.get("analyzed_period", [])
        period_tuple = tuple(
            datetime.datetime.fromisoformat(timestamp) if timestamp else None
            for timestamp in analyzed_period
        ) if analyzed_period else tuple()
        
        return cls(
            symbol=data["symbol"],
            patterns=patterns,
            analyzed_period=period_tuple,
            total_candles=data.get("total_candles", 0),
            execution_time=data.get("execution_time", 0.0),
            algorithm=data.get("algorithm", "default"),
            metadata=data.get("metadata", {})
        )
    
    @property
    def serializable_fields(self) -> List[str]:
        """Get a list of fields that should be serialized."""
        return [
            "symbol", "patterns", "analyzed_period", "total_candles",
            "execution_time", "algorithm", "metadata"
        ] 