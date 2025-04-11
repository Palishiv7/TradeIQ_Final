"""
Rule-Based Pattern Detectors

This module provides pattern detectors based on geometric rules and pattern definitions.
"""

import time
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set, Callable
from functools import lru_cache

from backend.common.logger import get_logger
from backend.common.finance.patterns import PatternType
from backend.assessments.candlestick_patterns.candlestick_utils import Candle, CandlestickData
from backend.assessments.candlestick_patterns.pattern_detection.interface import (
    PatternDetector, PatternMatch, DetectionStrategy
)

# Configure logger
logger = get_logger(__name__)


class RuleBasedDetector(PatternDetector):
    """
    Base rule-based detector that uses predefined rules to identify patterns.
    
    This class serves as a foundation for specific rule-based detection strategies
    and implements common functionality.
    """
    
    def __init__(
        self, 
        name: str = "RuleBasedDetector", 
        min_confidence: float = 0.5,
        max_patterns: Optional[int] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the rule-based detector.
        
        Args:
            name: Custom name for this detector
            min_confidence: Minimum confidence threshold for patterns (0.0-1.0)
            max_patterns: Maximum number of patterns to return (None for unlimited)
            config: Additional configuration parameters
        """
        strategy = DetectionStrategy.RULE_BASED
        super().__init__(
            name=name, 
            strategy=strategy,
            min_confidence=min_confidence,
            max_patterns=max_patterns,
            config=config or {}
        )
        # Store enabled patterns if specified in config
        self.enabled_patterns: Set[str] = set(self.config.get("enabled_patterns", []))
        self.min_candles = self.config.get("min_candles", 3)
        
    async def initialize(self) -> bool:
        """
        Initialize the detector. For rule-based detectors this is simple.
        
        Returns:
            True as initialization always succeeds for rule-based detectors
        """
        self._initialized = True
        return True
    
    async def detect_patterns(self, candlestick_data: CandlestickData) -> List[PatternMatch]:
        """
        Detect patterns using rule-based methods.
        
        Args:
            candlestick_data: Candlestick data to analyze
            
        Returns:
            List of detected patterns
            
        Raises:
            ValueError: If candlestick_data is invalid or insufficient
        """
        if not self.is_initialized():
            await self.initialize()
            
        if not candlestick_data or not candlestick_data.candles:
            raise ValueError("Cannot detect patterns: empty candlestick data")
            
        start_time = time.time()
        patterns: List[PatternMatch] = []
        
        # Get candles
        candles = candlestick_data.candles
        if len(candles) < self.min_candles:
            logger.warning(f"Not enough candles to detect patterns: {len(candles)} < {self.min_candles}")
            return []
        
        try:
            # Detect single candlestick patterns
            single_patterns = self._detect_single_patterns(candles)
            patterns.extend(single_patterns)
            
            # Detect double candlestick patterns if we have at least 2 candles
            if len(candles) >= 2:
                double_patterns = self._detect_double_patterns(candles)
                patterns.extend(double_patterns)
            
            # Detect triple candlestick patterns if we have at least 3 candles
            if len(candles) >= 3:
                triple_patterns = self._detect_triple_patterns(candles)
                patterns.extend(triple_patterns)
            
            # Add detection strategy and time
            end_time = time.time()
            detection_time_ms = (end_time - start_time) * 1000
            
            for pattern in patterns:
                pattern.detection_strategy = self.strategy
                pattern.detection_time_ms = detection_time_ms
            
            # Filter patterns based on enabled pattern types
            if self.enabled_patterns:
                patterns = [p for p in patterns if p.pattern_name in self.enabled_patterns]
            
            # Apply confidence filtering and limits
            return self.filter_patterns(patterns)
            
        except Exception as e:
            logger.exception(f"Error detecting patterns: {str(e)}")
            self._error_count += 1
            self._last_error = e
            raise ValueError(f"Pattern detection failed: {str(e)}") from e
    
    def _detect_single_patterns(self, candles: List[Candle]) -> List[PatternMatch]:
        """
        Detect single candlestick patterns.
        
        Args:
            candles: List of candlesticks to analyze
            
        Returns:
            List of detected single-candle patterns
        """
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
                confidence = max(0, min(1.0, 1.0 - (body_to_range_ratio * 10)))  # Higher confidence for smaller body/range ratio
                
                # Determine if bullish or bearish based on context
                bullish = prev_candle.is_bearish()  # Doji after a bearish candle might indicate reversal
                
                patterns.append(PatternMatch(
                    pattern_name="Doji",
                    confidence=confidence,
                    candle_indices=[i],
                    bullish=bullish,
                    description="A candle with almost equal open and close prices"
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
                    bullish=True,  # Hammer is a bullish reversal pattern
                    description="Bullish reversal pattern with a small body and long lower shadow"
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
                    bullish=False,  # Shooting Star is a bearish reversal pattern
                    description="Bearish reversal pattern with a small body and long upper shadow"
                ))
            
            # Marubozu pattern (strong full-bodied candle)
            if (candle.upper_shadow < candle.body_size * 0.05 and
                candle.lower_shadow < candle.body_size * 0.05 and
                candle.body_size > 0):
                
                confidence = 0.9  # High confidence for clear pattern
                
                bullish = candle.is_bullish()
                direction = "bullish" if bullish else "bearish"
                
                patterns.append(PatternMatch(
                    pattern_name=f"{direction.capitalize()} Marubozu",
                    confidence=confidence,
                    candle_indices=[i],
                    bullish=bullish,
                    description=f"Strong {direction} candle with minimal or no shadows"
                ))
        
        return patterns
    
    def _detect_double_patterns(self, candles: List[Candle]) -> List[PatternMatch]:
        """
        Detect double candlestick patterns.
        
        Args:
            candles: List of candlesticks to analyze
            
        Returns:
            List of detected two-candle patterns
        """
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
                    bullish=True,
                    description="Bullish reversal pattern where a large bullish candle completely engulfs the previous bearish candle"
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
                    bullish=False,
                    description="Bearish reversal pattern where a large bearish candle completely engulfs the previous bullish candle"
                ))
            
            # Bullish Harami pattern
            if (prev_candle.is_bearish() and curr_candle.is_bullish() and
                curr_candle.open > prev_candle.close and
                curr_candle.close < prev_candle.open and
                curr_candle.body_size < prev_candle.body_size):
                
                # Calculate confidence
                size_ratio = 1 - (curr_candle.body_size / prev_candle.body_size)
                confidence = min(1.0, 0.6 + (size_ratio * 0.3))
                
                patterns.append(PatternMatch(
                    pattern_name="Bullish Harami",
                    confidence=confidence,
                    candle_indices=[i-1, i],
                    bullish=True,
                    description="Bullish reversal pattern where a small bullish candle is contained within the body of the previous bearish candle"
                ))
            
            # Bearish Harami pattern
            if (prev_candle.is_bullish() and curr_candle.is_bearish() and
                curr_candle.open < prev_candle.close and
                curr_candle.close > prev_candle.open and
                curr_candle.body_size < prev_candle.body_size):
                
                # Calculate confidence
                size_ratio = 1 - (curr_candle.body_size / prev_candle.body_size)
                confidence = min(1.0, 0.6 + (size_ratio * 0.3))
                
                patterns.append(PatternMatch(
                    pattern_name="Bearish Harami",
                    confidence=confidence,
                    candle_indices=[i-1, i],
                    bullish=False,
                    description="Bearish reversal pattern where a small bearish candle is contained within the body of the previous bullish candle"
                ))
            
            # Tweezer Bottom
            if (prev_candle.is_bearish() and 
                curr_candle.is_bullish() and
                abs(prev_candle.low - curr_candle.low) / prev_candle.range_size < 0.05):
                
                confidence = 0.7
                
                patterns.append(PatternMatch(
                    pattern_name="Tweezer Bottom",
                    confidence=confidence,
                    candle_indices=[i-1, i],
                    bullish=True,
                    description="Bullish reversal pattern with two candles having virtually identical lows"
                ))
                
            # Tweezer Top
            if (prev_candle.is_bullish() and 
                curr_candle.is_bearish() and
                abs(prev_candle.high - curr_candle.high) / prev_candle.range_size < 0.05):
                
                confidence = 0.7
                
                patterns.append(PatternMatch(
                    pattern_name="Tweezer Top",
                    confidence=confidence,
                    candle_indices=[i-1, i],
                    bullish=False,
                    description="Bearish reversal pattern with two candles having virtually identical highs"
                ))
        
        return patterns
    
    def _detect_triple_patterns(self, candles: List[Candle]) -> List[PatternMatch]:
        """
        Detect triple candlestick patterns.
        
        Args:
            candles: List of candlesticks to analyze
            
        Returns:
            List of detected three-candle patterns
        """
        patterns = []
        
        for i in range(2, len(candles) - 1):
            # Get three consecutive candles
            first_candle = candles[i-2]
            middle_candle = candles[i-1]
            last_candle = candles[i]
            
            # Morning Star pattern
            if (first_candle.is_bearish() and
                last_candle.is_bullish() and
                middle_candle.body_size < first_candle.body_size * 0.5 and
                middle_candle.body_size < last_candle.body_size * 0.5 and
                last_candle.close > first_candle.close - (first_candle.body_size * 0.5)):
                
                # Calculate confidence based on middle candle size and third candle strength
                small_body_factor = 1 - (middle_candle.body_size / max(first_candle.body_size, last_candle.body_size))
                third_candle_strength = last_candle.body_size / first_candle.body_size
                confidence = min(1.0, 0.6 + (small_body_factor * 0.2) + (third_candle_strength * 0.2))
                
                # Add statistical validation
                # A stronger third candle increases confidence
                if last_candle.body_size > first_candle.body_size:
                    confidence += 0.1
                
                # If middle candle is a doji, add confidence
                if middle_candle.is_doji():
                    confidence += 0.1
                    
                confidence = min(1.0, confidence)
                
                patterns.append(PatternMatch(
                    pattern_name="Morning Star",
                    confidence=confidence,
                    candle_indices=[i-2, i-1, i],
                    bullish=True,
                    description="Bullish reversal pattern with a strong bearish candle, a small middle candle, and a strong bullish candle",
                    metadata={
                        "first_body_size": first_candle.body_size,
                        "middle_body_size": middle_candle.body_size,
                        "last_body_size": last_candle.body_size,
                        "middle_is_doji": middle_candle.is_doji()
                    }
                ))
            
            # Evening Star pattern
            if (first_candle.is_bullish() and
                last_candle.is_bearish() and
                middle_candle.body_size < first_candle.body_size * 0.5 and
                middle_candle.body_size < last_candle.body_size * 0.5 and
                last_candle.close < first_candle.close + (first_candle.body_size * 0.5)):
                
                # Calculate confidence based on middle candle size and third candle strength
                small_body_factor = 1 - (middle_candle.body_size / max(first_candle.body_size, last_candle.body_size))
                third_candle_strength = last_candle.body_size / first_candle.body_size
                confidence = min(1.0, 0.6 + (small_body_factor * 0.2) + (third_candle_strength * 0.2))
                
                # Add statistical validation
                # A stronger third candle increases confidence
                if last_candle.body_size > first_candle.body_size:
                    confidence += 0.1
                
                # If middle candle is a doji, add confidence
                if middle_candle.is_doji():
                    confidence += 0.1
                    
                confidence = min(1.0, confidence)
                
                patterns.append(PatternMatch(
                    pattern_name="Evening Star",
                    confidence=confidence,
                    candle_indices=[i-2, i-1, i],
                    bullish=False,
                    description="Bearish reversal pattern with a strong bullish candle, a small middle candle, and a strong bearish candle",
                    metadata={
                        "first_body_size": first_candle.body_size,
                        "middle_body_size": middle_candle.body_size,
                        "last_body_size": last_candle.body_size,
                        "middle_is_doji": middle_candle.is_doji()
                    }
                ))
            
            # Three White Soldiers
            if (first_candle.is_bullish() and
                middle_candle.is_bullish() and
                last_candle.is_bullish() and
                middle_candle.open > first_candle.open and
                middle_candle.close > first_candle.close and
                last_candle.open > middle_candle.open and
                last_candle.close > middle_candle.close and
                first_candle.upper_shadow < first_candle.body_size * 0.3 and
                middle_candle.upper_shadow < middle_candle.body_size * 0.3 and
                last_candle.upper_shadow < last_candle.body_size * 0.3):
                
                # Calculate confidence based on consistent body sizes and minimal shadows
                body_size_consistency = 1 - (max(abs(first_candle.body_size - middle_candle.body_size), 
                                            abs(middle_candle.body_size - last_candle.body_size)) / 
                                       max(first_candle.body_size, middle_candle.body_size, last_candle.body_size))
                
                shadow_factor = 1 - ((first_candle.upper_shadow + middle_candle.upper_shadow + last_candle.upper_shadow) / 
                                 (first_candle.body_size + middle_candle.body_size + last_candle.body_size))
                
                confidence = min(1.0, 0.7 + (body_size_consistency * 0.15) + (shadow_factor * 0.15))
                
                # Add statistical validation
                # If each candle is stronger than the previous one, add confidence
                if last_candle.body_size > middle_candle.body_size > first_candle.body_size:
                    confidence += 0.1
                    
                confidence = min(1.0, confidence)
                
                patterns.append(PatternMatch(
                    pattern_name="Three White Soldiers",
                    confidence=confidence,
                    candle_indices=[i-2, i-1, i],
                    bullish=True,
                    description="Bullish continuation pattern with three consecutive bullish candles, each closing higher than the previous one",
                    metadata={
                        "body_size_consistency": body_size_consistency,
                        "shadow_factor": shadow_factor,
                        "progressive_strength": last_candle.body_size > middle_candle.body_size > first_candle.body_size
                    }
                ))
            
            # Three Black Crows
            if (first_candle.is_bearish() and
                middle_candle.is_bearish() and
                last_candle.is_bearish() and
                middle_candle.open < first_candle.open and
                middle_candle.close < first_candle.close and
                last_candle.open < middle_candle.open and
                last_candle.close < middle_candle.close and
                first_candle.lower_shadow < first_candle.body_size * 0.3 and
                middle_candle.lower_shadow < middle_candle.body_size * 0.3 and
                last_candle.lower_shadow < last_candle.body_size * 0.3):
                
                # Calculate confidence based on consistent body sizes and minimal shadows
                body_size_consistency = 1 - (max(abs(first_candle.body_size - middle_candle.body_size), 
                                            abs(middle_candle.body_size - last_candle.body_size)) / 
                                       max(first_candle.body_size, middle_candle.body_size, last_candle.body_size))
                
                shadow_factor = 1 - ((first_candle.lower_shadow + middle_candle.lower_shadow + last_candle.lower_shadow) / 
                                 (first_candle.body_size + middle_candle.body_size + last_candle.body_size))
                
                confidence = min(1.0, 0.7 + (body_size_consistency * 0.15) + (shadow_factor * 0.15))
                
                # Add statistical validation
                # If each candle is stronger than the previous one, add confidence
                if last_candle.body_size > middle_candle.body_size > first_candle.body_size:
                    confidence += 0.1
                    
                confidence = min(1.0, confidence)
                
                patterns.append(PatternMatch(
                    pattern_name="Three Black Crows",
                    confidence=confidence,
                    candle_indices=[i-2, i-1, i],
                    bullish=False,
                    description="Bearish continuation pattern with three consecutive bearish candles, each closing lower than the previous one",
                    metadata={
                        "body_size_consistency": body_size_consistency,
                        "shadow_factor": shadow_factor,
                        "progressive_strength": last_candle.body_size > middle_candle.body_size > first_candle.body_size
                    }
                ))
                
            # Abandoned Baby Bottom (rare but powerful)
            if (first_candle.is_bearish() and
                last_candle.is_bullish() and
                middle_candle.is_doji() and
                middle_candle.high < first_candle.close and
                middle_candle.high < last_candle.open and
                last_candle.body_size > first_candle.body_size * 0.5):
                
                confidence = 0.9  # Very high confidence for this rare pattern
                
                patterns.append(PatternMatch(
                    pattern_name="Abandoned Baby Bottom",
                    confidence=confidence,
                    candle_indices=[i-2, i-1, i],
                    bullish=True,
                    description="Strong bullish reversal pattern with a bearish candle, a gapped doji, and a bullish candle gapping up",
                    metadata={
                        "doji_gap_down": first_candle.close - middle_candle.high,
                        "doji_gap_up": last_candle.open - middle_candle.high,
                        "rare_pattern": True
                    }
                ))
                
            # Abandoned Baby Top (rare but powerful)
            if (first_candle.is_bullish() and
                last_candle.is_bearish() and
                middle_candle.is_doji() and
                middle_candle.low > first_candle.close and
                middle_candle.low > last_candle.open and
                last_candle.body_size > first_candle.body_size * 0.5):
                
                confidence = 0.9  # Very high confidence for this rare pattern
                
                patterns.append(PatternMatch(
                    pattern_name="Abandoned Baby Top",
                    confidence=confidence,
                    candle_indices=[i-2, i-1, i],
                    bullish=False,
                    description="Strong bearish reversal pattern with a bullish candle, a gapped doji, and a bearish candle gapping down",
                    metadata={
                        "doji_gap_up": middle_candle.low - first_candle.close,
                        "doji_gap_down": middle_candle.low - last_candle.open,
                        "rare_pattern": True
                    }
                ))
                
            # Three Inside Up (Bullish)
            if (first_candle.is_bearish() and
                middle_candle.is_bullish() and
                last_candle.is_bullish() and
                middle_candle.open > first_candle.close and
                middle_candle.close < first_candle.open and
                last_candle.close > first_candle.open):
                
                # Calculate confidence based on third candle strength
                third_candle_strength = (last_candle.close - first_candle.open) / first_candle.body_size
                confidence = min(1.0, 0.65 + (third_candle_strength * 0.3))
                
                patterns.append(PatternMatch(
                    pattern_name="Three Inside Up",
                    confidence=confidence,
                    candle_indices=[i-2, i-1, i],
                    bullish=True,
                    description="Bullish reversal pattern starting with a Bullish Harami and confirmed by a third bullish candle",
                    metadata={
                        "harami_part": True,
                        "confirmation_strength": third_candle_strength
                    }
                ))
                
            # Three Inside Down (Bearish)
            if (first_candle.is_bullish() and
                middle_candle.is_bearish() and
                last_candle.is_bearish() and
                middle_candle.open < first_candle.close and
                middle_candle.close > first_candle.open and
                last_candle.close < first_candle.open):
                
                # Calculate confidence based on third candle strength
                third_candle_strength = (first_candle.open - last_candle.close) / first_candle.body_size
                confidence = min(1.0, 0.65 + (third_candle_strength * 0.3))
                
                patterns.append(PatternMatch(
                    pattern_name="Three Inside Down",
                    confidence=confidence,
                    candle_indices=[i-2, i-1, i],
                    bullish=False,
                    description="Bearish reversal pattern starting with a Bearish Harami and confirmed by a third bearish candle",
                    metadata={
                        "harami_part": True,
                        "confirmation_strength": third_candle_strength
                    }
                ))
        
        return patterns
    
    def get_strategy_type(self) -> DetectionStrategy:
        """Get the detection strategy type."""
        return DetectionStrategy.RULE_BASED


class GeometricPatternDetector(RuleBasedDetector):
    """
    Pattern detector that uses geometric rules to identify patterns.
    """
    def __init__(self, name: Optional[str] = None):
        super().__init__(name=name or "GeometricPatternDetector")
    
    def get_strategy_type(self) -> DetectionStrategy:
        """Get the detection strategy type."""
        return DetectionStrategy.GEOMETRIC


class StatisticalPatternDetector(RuleBasedDetector):
    """
    Pattern detector that uses statistical analysis to identify patterns.
    """
    def __init__(
        self, 
        name: Optional[str] = None, 
        min_confidence: float = 0.65,
        use_trend_analysis: bool = True
    ):
        super().__init__(
            name=name or "StatisticalPatternDetector",
            min_confidence=min_confidence,
            config={"use_trend_analysis": use_trend_analysis}
        )
    
    def detect_patterns(self, data: CandlestickData) -> List[PatternMatch]:
        """
        Detect patterns with statistical validation.
        
        Args:
            data: Candlestick data to analyze
            
        Returns:
            List of detected patterns
        """
        # First, get base patterns using the parent detector
        patterns = super().detect_patterns(data)
        
        # Filter by confidence
        patterns = [p for p in patterns if p.confidence >= self.min_confidence]
        
        # Add trend context if enabled
        if self.config["use_trend_analysis"]:
            patterns = [self._add_trend_context(p, data.candles) for p in patterns]
        
        # Add statistical validation
        patterns = [self._add_statistical_validation(p, data.candles) for p in patterns]
        
        return patterns
    
    def _add_trend_context(self, pattern: PatternMatch, candles: List[Candle]) -> PatternMatch:
        """
        Add trend context to the pattern for more accurate signal interpretation.
        
        Args:
            pattern: The detected pattern
            candles: Full list of candles
            
        Returns:
            Pattern with updated confidence and metadata
        """
        # Skip if not enough candles
        if len(candles) < 10:
            return pattern
        
        # Get pattern candles and determine relevant range
        pattern_indices = sorted(pattern.candle_indices)
        
        # Look back up to 7 candles before the pattern
        start_idx = max(0, pattern_indices[0] - 7)
        end_idx = pattern_indices[-1]
        
        # Calculate trend before the pattern
        trend_candles = candles[start_idx:end_idx]
        if not trend_candles:
            return pattern
        
        # Calculate simple trend: positive if closing prices are increasing
        closes = [c.close for c in trend_candles]
        if len(closes) >= 3:
            # Use linear regression slope to determine trend
            x = np.arange(len(closes))
            slope, _, _, _, _ = np.polyfit(x, closes, 1, full=True)
            
            trend_strength = min(1.0, abs(slope[0]) * 10 / np.mean(closes))
            is_uptrend = slope[0] > 0
            
            # Update confidence based on trend context
            if pattern.bullish and not is_uptrend:
                # Bullish pattern in a downtrend (potential reversal) - increase confidence
                new_confidence = min(1.0, pattern.confidence + (trend_strength * 0.1))
                pattern.confidence = new_confidence
                
                # Add metadata
                pattern.metadata["trend_context"] = "potential_reversal"
                pattern.metadata["trend_strength"] = trend_strength
                pattern.metadata["pre_pattern_trend"] = "downtrend"
                
            elif not pattern.bullish and is_uptrend:
                # Bearish pattern in an uptrend (potential reversal) - increase confidence
                new_confidence = min(1.0, pattern.confidence + (trend_strength * 0.1))
                pattern.confidence = new_confidence
                
                # Add metadata
                pattern.metadata["trend_context"] = "potential_reversal"
                pattern.metadata["trend_strength"] = trend_strength
                pattern.metadata["pre_pattern_trend"] = "uptrend"
                
            else:
                # Pattern aligned with trend (continuation) - slightly increase confidence
                new_confidence = min(1.0, pattern.confidence + (trend_strength * 0.05))
                pattern.confidence = new_confidence
                
                # Add metadata
                pattern.metadata["trend_context"] = "continuation"
                pattern.metadata["trend_strength"] = trend_strength
                pattern.metadata["pre_pattern_trend"] = "uptrend" if is_uptrend else "downtrend"
        
        return pattern
    
    def _add_statistical_validation(self, pattern: PatternMatch, candles: List[Candle]) -> PatternMatch:
        """
        Add statistical validation to the pattern to improve confidence scoring.
        
        Args:
            pattern: The detected pattern
            candles: Full list of candles
            
        Returns:
            Pattern with updated confidence and metadata
        """
        # Get pattern indices and calculate volume context
        pattern_indices = sorted(pattern.candle_indices)
        
        # Skip if we can't calculate statistics
        if not pattern_indices or len(candles) < 10:
            return pattern
        
        # Look at volumes for pattern candles
        pattern_volumes = [candles[i].volume for i in pattern_indices if 0 <= i < len(candles)]
        if not pattern_volumes:
            return pattern
            
        # Calculate average volume for the dataset
        all_volumes = [c.volume for c in candles if c.volume > 0]
        avg_volume = np.mean(all_volumes) if all_volumes else 1.0
        
        # Calculate volume ratio (pattern volume vs average)
        volume_ratio = np.mean(pattern_volumes) / avg_volume if avg_volume > 0 else 1.0
        
        # Adjust confidence based on volume confirmation
        volume_confirmation = min(1.5, volume_ratio)  # Cap at 1.5x average
        
        # Patterns with above-average volume get a confidence boost
        if volume_confirmation > 1.1:
            confidence_boost = min(0.15, (volume_confirmation - 1.0) * 0.15)
            pattern.confidence = min(1.0, pattern.confidence + confidence_boost)
            pattern.metadata["volume_confirmation"] = True
            pattern.metadata["volume_ratio"] = volume_ratio
        
        # Check for volatility context
        if len(candles) >= 20:
            # Calculate recent volatility (ATR-like)
            recent_ranges = [c.range_size for c in candles[-20:]]
            avg_range = np.mean(recent_ranges) if recent_ranges else 0
            
            pattern_ranges = [candles[i].range_size for i in pattern_indices if 0 <= i < len(candles)]
            pattern_avg_range = np.mean(pattern_ranges) if pattern_ranges else 0
            
            if avg_range > 0:
                volatility_ratio = pattern_avg_range / avg_range
                
                # Patterns with above-average range get a confidence boost
                if volatility_ratio > 1.2:
                    volatility_boost = min(0.1, (volatility_ratio - 1.0) * 0.1)
                    pattern.confidence = min(1.0, pattern.confidence + volatility_boost)
                    pattern.metadata["volatility_confirmation"] = True
                    pattern.metadata["volatility_ratio"] = volatility_ratio
        
        return pattern
    
    def get_strategy_type(self) -> DetectionStrategy:
        """Get the detection strategy type."""
        return DetectionStrategy.STATISTICAL 