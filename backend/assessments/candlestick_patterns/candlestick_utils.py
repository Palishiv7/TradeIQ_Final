"""
Utility functions and classes for candlestick pattern assessments.

This module provides:
1. Helper classes for representing candlestick data
2. Utility functions for pattern analysis
3. Common operations used across the assessment
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union, Type, TypeVar, Set
from datetime import datetime, timedelta
import time
import random
import uuid
from enum import Enum
import json
import io
import base64
from dataclasses import dataclass, field
import logging

from backend.assessments.candlestick_patterns.candlestick_config import (
    CANDLESTICK_PATTERNS, DIFFICULTY_LEVELS, PATTERN_DESCRIPTIONS
)
from backend.assessments.base.models import QuestionDifficulty, BaseQuestion
from backend.common.finance.patterns import PatternType, PatternStrength, CandlestickPattern
from backend.common.serialization import SerializableMixin
from backend.common.visualization import generate_chart_image, base64_encode_image

# For backward compatibility with code expecting the Difficulty enum
class Difficulty(Enum):
    """Enum for backward compatibility with code expecting this name."""
    BEGINNER = QuestionDifficulty.VERY_EASY.value
    INTERMEDIATE = QuestionDifficulty.MEDIUM.value
    ADVANCED = QuestionDifficulty.HARD.value
    
    @classmethod
    def from_question_difficulty(cls, difficulty: QuestionDifficulty) -> 'Difficulty':
        """Convert QuestionDifficulty to Difficulty."""
        mapping = {
            QuestionDifficulty.VERY_EASY: cls.BEGINNER,
            QuestionDifficulty.EASY: cls.BEGINNER,
            QuestionDifficulty.MEDIUM: cls.INTERMEDIATE,
            QuestionDifficulty.HARD: cls.ADVANCED,
            QuestionDifficulty.VERY_HARD: cls.ADVANCED
        }
        return mapping.get(difficulty, cls.INTERMEDIATE)
    
    def to_question_difficulty(self) -> QuestionDifficulty:
        """Convert to QuestionDifficulty."""
        mapping = {
            self.BEGINNER: QuestionDifficulty.EASY,
            self.INTERMEDIATE: QuestionDifficulty.MEDIUM,
            self.ADVANCED: QuestionDifficulty.HARD
        }
        return mapping.get(self, QuestionDifficulty.MEDIUM)

# Set up logger
logger = logging.getLogger(__name__)

# Define data classes for candlestick data
@dataclass
class Candle(SerializableMixin):
    """Representation of a single candlestick."""
    time: int
    open: float
    high: float
    low: float
    close: float
    volume: float = 0
    
    __serializable_fields__ = [
        "time", "open", "high", "low", "close", "volume"
    ]
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Candle':
        """Create a Candle from a dictionary."""
        return cls(
            time=data.get("time", 0),
            open=data.get("open", 0),
            high=data.get("high", 0),
            low=data.get("low", 0),
            close=data.get("close", 0),
            volume=data.get("volume", 0)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert candle to dictionary."""
        return {
            "time": self.time,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume
        }
    
    @property
    def is_bullish(self) -> bool:
        """Check if candle is bullish (close > open)."""
        return self.close > self.open
    
    @property
    def is_bearish(self) -> bool:
        """Check if candle is bearish (close < open)."""
        return self.close < self.open
    
    @property
    def is_doji(self) -> bool:
        """Check if candle is a doji (open ≈ close)."""
        range_size = self.high - self.low
        if range_size == 0:
            return True
        body_size = abs(self.close - self.open)
        return body_size / range_size < 0.1  # Body is less than 10% of range
    
    @property
    def body_size(self) -> float:
        """Get the size of the candle's body."""
        return abs(self.close - self.open)
    
    @property
    def range_size(self) -> float:
        """Get the size of the candle's range (high - low)."""
        return self.high - self.low
    
    @property
    def upper_shadow(self) -> float:
        """Get the size of the upper shadow."""
        return self.high - max(self.open, self.close)
    
    @property
    def lower_shadow(self) -> float:
        """Get the size of the lower shadow."""
        return min(self.open, self.close) - self.low
    
    @property
    def body_mid_point(self) -> float:
        """Get the mid point of the candle's body."""
        return (self.open + self.close) / 2
    
    @property
    def datetime(self) -> datetime:
        """Get the candle's time as a datetime object."""
        return datetime.fromtimestamp(self.time)

class CandlestickData(SerializableMixin):
    """Class for handling candlestick data."""
    
    __serializable_fields__ = [
        "symbol", "timeframe", "candles"
    ]
    
    def __init__(self, symbol: str, timeframe: str, candles: List[Candle]):
        """
        Initialize candlestick data.
        
        Args:
            symbol: The market symbol
            timeframe: The timeframe of the candles
            candles: List of candles
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.candles = candles
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CandlestickData':
        """Create a CandlestickData from a dictionary."""
        symbol = data.get("symbol", "UNKNOWN")
        timeframe = data.get("timeframe", "1d")
        candles = [Candle.from_dict(c) for c in data.get("candles", [])]
        return cls(symbol, timeframe, candles)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "candles": [c.to_dict() for c in self.candles]
        }
    
    def get_normalized_data(self) -> Dict[str, Any]:
        """Get normalized data for ML models."""
        if not self.candles:
            return {}
        
        # Find min and max values for normalization
        min_low = min(c.low for c in self.candles)
        max_high = max(c.high for c in self.candles)
        price_range = max_high - min_low
        
        if price_range == 0:
            price_range = 1  # Prevent division by zero
        
        # Normalize candles
        normalized_candles = []
        for candle in self.candles:
            normalized_candles.append({
                "open": (candle.open - min_low) / price_range,
                "high": (candle.high - min_low) / price_range,
                "low": (candle.low - min_low) / price_range,
                "close": (candle.close - min_low) / price_range,
                "volume": candle.volume  # Volume is typically normalized separately
            })
        
        return {
            "candles": normalized_candles,
            "min_low": min_low,
            "max_high": max_high,
            "price_range": price_range
        }
    
    def get_pattern_segment(self, pattern_indices: List[int]) -> 'CandlestickData':
        """
        Extract a segment of candles for a specific pattern.
        
        Args:
            pattern_indices: List of indices for the pattern
            
        Returns:
            A new CandlestickData containing only the pattern candles
        """
        pattern_candles = [self.candles[i] for i in pattern_indices if 0 <= i < len(self.candles)]
        return CandlestickData(self.symbol, self.timeframe, pattern_candles)
    
    def to_candlestick_pattern(self, pattern_type: PatternType, start_idx: int, end_idx: int) -> CandlestickPattern:
        """
        Convert a segment of this data to a CandlestickPattern object.
        
        Args:
            pattern_type: The type of pattern
            start_idx: Start index of the pattern
            end_idx: End index of the pattern
            
        Returns:
            CandlestickPattern instance
        """
        if start_idx < 0 or end_idx >= len(self.candles) or start_idx > end_idx:
            raise ValueError(f"Invalid pattern indices: {start_idx} to {end_idx}")
            
        # Extract pattern candles
        pattern_candles = self.candles[start_idx:end_idx+1]
        
        # Determine pattern strength based on candle characteristics
        strength = _calculate_pattern_strength(pattern_candles, pattern_type)
        
        # Create the pattern object
        return CandlestickPattern(
            pattern_id=str(uuid.uuid4()),
            pattern_type=pattern_type,
            symbol=self.symbol,
            timeframe=self.timeframe,
            candle_data=[c.to_dict() for c in pattern_candles],
            strength=strength,
            timestamp=int(time.time())
        )
        
    def plot_candlestick_chart(self, highlight_indices: Optional[List[int]] = None) -> str:
        """
        Generate a candlestick chart and return the image as a string.
        
        Args:
            highlight_indices: Optional list of indices to highlight
            
        Returns:
            Base64 encoded string representation of the candlestick chart
        """
        # Convert candles to the format expected by generate_chart_image
        data = []
        for candle in self.candles:
            data.append({
                'date': candle.datetime,
                'open': candle.open,
                'high': candle.high,
                'low': candle.low,
                'close': candle.close,
                'volume': candle.volume
            })
        
        # Create pattern regions if highlight indices are provided
        pattern_regions = None
        if highlight_indices and len(highlight_indices) > 0:
            pattern_regions = []
            for idx in highlight_indices:
                if 0 <= idx < len(self.candles):
                    pattern_regions.append({
                        'start_date': self.candles[idx].datetime,
                        'end_date': self.candles[idx].datetime,
                        'color': 'yellow'
                    })
        
        # Generate the chart image
        image_data = generate_chart_image(
            data=data,
            pattern_regions=pattern_regions,
            width=10,
            height=6,
            style='charles'
        )
        
        # Encode the image as base64
        return base64_encode_image(image_data)


def get_patterns_by_difficulty(
    difficulty: Union[QuestionDifficulty, Difficulty, str]
) -> List[str]:
    """
    Get a list of patterns appropriate for a given difficulty level.
    
    Args:
        difficulty: Difficulty level (QuestionDifficulty, Difficulty enum, or string)
        
    Returns:
        List of pattern names suitable for the difficulty
    """
    # Handle different difficulty types
    if isinstance(difficulty, QuestionDifficulty):
        # Map QuestionDifficulty to level names
        if difficulty in [QuestionDifficulty.VERY_EASY, QuestionDifficulty.EASY]:
            level_name = "beginner"
        elif difficulty == QuestionDifficulty.MEDIUM:
            level_name = "medium"
        elif difficulty in [QuestionDifficulty.HARD, QuestionDifficulty.VERY_HARD]:
            level_name = "hard"
        else:
            level_name = "medium"  # Default to medium if unknown
    elif isinstance(difficulty, Difficulty):
        # Map Difficulty to level names
        if difficulty == Difficulty.BEGINNER:
            level_name = "beginner"
        elif difficulty == Difficulty.INTERMEDIATE:
            level_name = "medium"
        elif difficulty == Difficulty.ADVANCED:
            level_name = "hard"
        else:
            level_name = "medium"  # Default to medium if unknown
    elif isinstance(difficulty, str):
        # Map string difficulty to level names
        difficulty_lower = difficulty.lower()
        if difficulty_lower in ["very_easy", "easy", "beginner"]:
            level_name = "beginner"
        elif difficulty_lower in ["medium", "intermediate"]:
            level_name = "medium"
        elif difficulty_lower in ["hard", "very_hard", "advanced"]:
            level_name = "hard"
        else:
            level_name = "medium"  # Default to medium if unknown
    else:
        level_name = "medium"  # Default for unexpected types
    
    # Validate configuration
    if not DIFFICULTY_LEVELS:
        logger.error("DIFFICULTY_LEVELS configuration is empty or not loaded")
        # Return a minimal default list as fallback
        return ["Hammer", "Doji", "Engulfing"]
    
    # Get patterns for this level
    patterns = DIFFICULTY_LEVELS.get(level_name, [])
    
    # Validate we have valid patterns
    if not patterns:
        logger.warning(f"No patterns found for difficulty level '{level_name}', using fallback patterns")
        # Use patterns from adjacent levels as fallback
        all_levels = ["beginner", "easy", "medium", "hard", "expert"]
        fallback_patterns = []
        for level in all_levels:
            if level_patterns := DIFFICULTY_LEVELS.get(level, []):
                fallback_patterns.extend(level_patterns)
                if len(fallback_patterns) >= 3:
                    break
        
        # If we still have no patterns, use a hardcoded minimum set
        if not fallback_patterns:
            logger.error("No patterns found in any difficulty level, using hardcoded patterns")
            fallback_patterns = ["Hammer", "Doji", "Engulfing"]
        
        return fallback_patterns[:5]  # Limit to 5 patterns
        
    # If we have too few patterns, include some from adjacent levels
    if len(patterns) < 3:
        # Add patterns from adjacent levels
        all_levels = ["beginner", "easy", "medium", "hard", "expert"]
        level_idx = all_levels.index(level_name) if level_name in all_levels else 1
        
        # Try to get patterns from adjacent levels
        adjacent_levels = []
        if level_idx > 0:
            adjacent_levels.append(all_levels[level_idx - 1])
        if level_idx < len(all_levels) - 1:
            adjacent_levels.append(all_levels[level_idx + 1])
            
        for adj_level in adjacent_levels:
            adj_patterns = DIFFICULTY_LEVELS.get(adj_level, [])
            needed = max(0, 3 - len(patterns))
            if needed > 0 and adj_patterns:
                patterns.extend(random.sample(adj_patterns, min(needed, len(adj_patterns))))
    
    return patterns


def get_pattern_category(pattern_name: str) -> str:
    """
    Get the category of a pattern.
    
    Args:
        pattern_name: Name of the pattern
        
    Returns:
        Category name
    """
    for category, patterns in CANDLESTICK_PATTERNS.items():
        if pattern_name in patterns:
            return category
    return "unknown"


def get_pattern_description(pattern_name: str) -> str:
    """
    Get the description of a pattern.
    
    Args:
        pattern_name: Name of the pattern
        
    Returns:
        Description text
    """
    return PATTERN_DESCRIPTIONS.get(pattern_name, f"No description available for {pattern_name}")


def _calculate_pattern_strength(candles: List[Candle], pattern_type: PatternType) -> PatternStrength:
    """
    Calculate the strength of a candlestick pattern.
    
    Args:
        candles: List of candles forming the pattern
        pattern_type: Type of pattern
        
    Returns:
        Pattern strength
    """
    # This is a simplified strength calculation
    # In a real implementation, this would use more sophisticated heuristics
    
    # Calculate average body size
    avg_body_size = sum(c.body_size for c in candles) / len(candles)
    
    # Calculate average range size
    avg_range_size = sum(c.range_size for c in candles) / len(candles)
    
    # Body to range ratio
    body_range_ratio = avg_body_size / avg_range_size if avg_range_size > 0 else 0
    
    # Strength heuristic based on pattern type and characteristics
    if pattern_type in [PatternType.HAMMER, PatternType.SHOOTING_STAR]:
        # For single candle patterns, look at shadow lengths
        candle = candles[0]
        if pattern_type == PatternType.HAMMER:
            # Strong hammer has long lower shadow
            strength_score = candle.lower_shadow / candle.range_size if candle.range_size > 0 else 0
        else:
            # Strong shooting star has long upper shadow
            strength_score = candle.upper_shadow / candle.range_size if candle.range_size > 0 else 0
    elif pattern_type in [PatternType.ENGULFING, PatternType.HARAMI]:
        # For two-candle patterns, look at relationship between candles
        if len(candles) >= 2:
            first, second = candles[0], candles[1]
            if pattern_type == PatternType.ENGULFING:
                # Strong engulfing has second candle much larger than first
                strength_score = second.body_size / first.body_size if first.body_size > 0 else 0
            else:
                # Strong harami has second candle much smaller than first
                strength_score = 1 - (second.body_size / first.body_size if first.body_size > 0 else 0)
        else:
            strength_score = 0.5
    else:
        # Default strength heuristic
        strength_score = body_range_ratio
    
    # Convert score to enum
    if strength_score > 0.8:
        return PatternStrength.STRONG
    elif strength_score > 0.5:
        return PatternStrength.MODERATE
    else:
        return PatternStrength.WEAK


def generate_options(correct_pattern: str, all_patterns: List[str], num_options: int = 4, difficulty: float = 0.5) -> List[str]:
    """
    Generate question options including the correct pattern and distractors.
    
    Args:
        correct_pattern: The correct pattern
        all_patterns: List of all available patterns
        num_options: Number of options to generate
        difficulty: Difficulty level (0.0-1.0)
        
    Returns:
        List of pattern options
    """
    # Make sure correct pattern is in the list
    if correct_pattern not in all_patterns:
        all_patterns = all_patterns + [correct_pattern]
    
    # Get the category of the correct pattern
    correct_category = get_pattern_category(correct_pattern)
    
    # Determine number of distractors from the same category
    # Higher difficulty = more distractors from the same category
    same_category_count = int((num_options - 1) * min(difficulty, 0.8))
    other_category_count = num_options - 1 - same_category_count
    
    # Optimize pattern filtering - do a single pass through the patterns
    same_category_patterns = []
    other_category_patterns = []
    
    for pattern in all_patterns:
        if pattern != correct_pattern:
            if get_pattern_category(pattern) == correct_category:
                same_category_patterns.append(pattern)
            else:
                other_category_patterns.append(pattern)
    
    # Adjust counts if we don't have enough patterns
    if len(same_category_patterns) < same_category_count:
        other_category_count += same_category_count - len(same_category_patterns)
        same_category_count = len(same_category_patterns)
        
    if len(other_category_patterns) < other_category_count:
        other_category_count = len(other_category_patterns)
    
    # Select distractors
    distractors = []
    if same_category_count > 0:
        distractors.extend(random.sample(same_category_patterns, same_category_count))
    if other_category_count > 0:
        distractors.extend(random.sample(other_category_patterns, other_category_count))
    
    # Combine and shuffle
    options = [correct_pattern] + distractors
    random.shuffle(options)
    
    return options


def format_pattern_name(pattern: str) -> str:
    """
    Format a pattern name for display.
    
    Args:
        pattern: Pattern name (usually in UPPER_SNAKE_CASE)
        
    Returns:
        Formatted pattern name (Title Case with spaces)
    """
    return pattern.replace('_', ' ').title()


def convert_to_candlestick_pattern(pattern_data: Dict[str, Any]) -> CandlestickPattern:
    """
    Convert a dictionary to a CandlestickPattern.
    
    Args:
        pattern_data: Dictionary representation of a pattern
        
    Returns:
        CandlestickPattern instance
    """
    try:
        pattern_type = PatternType[pattern_data.get("pattern_type", "UNKNOWN")]
    except (KeyError, ValueError):
        pattern_type = PatternType.UNKNOWN
    
    try:
        strength = PatternStrength[pattern_data.get("strength", "MODERATE")]
    except (KeyError, ValueError):
        strength = PatternStrength.MODERATE
    
    return CandlestickPattern(
        pattern_id=pattern_data.get("pattern_id", str(uuid.uuid4())),
        pattern_type=pattern_type,
        symbol=pattern_data.get("symbol", "UNKNOWN"),
        timeframe=pattern_data.get("timeframe", "1d"),
        candle_data=pattern_data.get("candle_data", []),
        strength=strength,
        timestamp=pattern_data.get("timestamp", int(time.time()))
    )

def plot_candlestick_chart(
    candles: List[Candle],
    width: int = 640,
    height: int = 480,
    highlight_indices: Optional[List[int]] = None,
    as_array: bool = False,
    symbol: str = "UNKNOWN",
    timeframe: str = "1d"
) -> Union[str, np.ndarray, None]:
    """
    Generate a candlestick chart image from candle data.
    
    Args:
        candles: List of candles to plot
        width: Width of the chart in pixels
        height: Height of the chart in pixels
        highlight_indices: Indices of candles to highlight
        as_array: Return as numpy array instead of base64 string
        symbol: Market symbol to display in title
        timeframe: Timeframe to display in title
        
    Returns:
        Base64 encoded PNG image string, numpy array (if as_array=True), or None if failed
    """
    # Return placeholder image if no candles
    if not candles:
        logger.warning("No candles to plot")
        return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
    
    # Check if we have this chart cached
    cache_key = f"{symbol}:{timeframe}:{len(candles)}:{highlight_indices}"
    
    # Try to get from cache
    try:
        from backend.common.cache import get_cache_service
        cache = get_cache_service()
        cached_image = cache.get(cache_key, None)
        if cached_image:
            return cached_image
    except Exception as cache_err:
        logger.warning(f"Cache error when checking for cached chart: {cache_err}")
    
    # Import matplotlib and related libraries only when needed
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        
        try:
            import mplfinance as mpf
            HAS_PLOTTING = True
        except ImportError:
            HAS_PLOTTING = False
            logger.warning("mplfinance not available. Using basic matplotlib for chart generation.")
    except ImportError as e:
        logger.error(f"Required plotting libraries not available: {e}")
        return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
    
    # Create figure and plot
    fig = None
    try:
        # Convert candles to pandas DataFrame
        data = []
        for candle in candles:
            timestamp = datetime.fromtimestamp(candle.time)
            data.append({
                "Date": timestamp,
                "Open": candle.open,
                "High": candle.high,
                "Low": candle.low,
                "Close": candle.close,
                "Volume": candle.volume
            })
        
        df = pd.DataFrame(data)
        df.set_index("Date", inplace=True)
        
        # Create figure with specific size
        fig = plt.figure(figsize=(width/100, height/100), dpi=100)
        
        if HAS_PLOTTING:
            # Use mplfinance for better candlestick charts
            mc = mpf.make_marketcolors(
                up='green', down='red',
                wick={'up': 'green', 'down': 'red'},
                edge={'up': 'green', 'down': 'red'},
                volume={'up': 'green', 'down': 'red'},
            )
            style = mpf.make_mpf_style(marketcolors=mc, gridstyle=':', y_on_right=False)
            
            # Prepare highlighted indices if needed
            highlight = None
            if highlight_indices:
                valid_indices = [i for i in highlight_indices if 0 <= i < len(candles)]
                if valid_indices:
                    highlight = []
                    for i in valid_indices:
                        # Convert index to datetime for proper highlighting
                        try:
                            dt = df.index[i]
                            highlight.append(dt)
                        except IndexError:
                            pass
            
            # Plot the candlestick chart
            kwargs = {
                'type': 'candle',
                'figsize': (width/100, height/100),
                'style': style,
                'volume': True,
                'panel_ratios': (4, 1),
                'returnfig': True,
                'warn_too_much_data': 10000  # Avoid warnings for large datasets
            }
            
            if highlight:
                kwargs['vlines'] = {'vlines': highlight, 'linewidths': 1, 'linestyle': '--', 'colors': 'blue'}
            
            fig, axes = mpf.plot(df, **kwargs)
            
            # Add title
            axes[0].set_title(f"{symbol} {timeframe} Candlestick Chart")
        else:
            # Fallback to basic matplotlib
            ax1 = fig.add_subplot(111)
            
            # Plot candlesticks manually
            for i, candle in enumerate(candles):
                dt = candle.datetime
                
                # Draw body
                color = 'green' if candle.is_bullish else 'red'
                ax1.plot([dt, dt], [candle.low, candle.high], color='black')
                ax1.plot([dt, dt], [candle.open, candle.close], linewidth=3, color=color)
                
                # Highlight if requested
                if highlight_indices and i in highlight_indices:
                    ax1.axvspan(dt - timedelta(hours=12), dt + timedelta(hours=12), 
                               alpha=0.2, color='blue')
            
            # Add title and formatting
            ax1.set_title(f"{symbol} {timeframe} Candlestick Chart")
            ax1.grid(True)
            ax1.set_xlabel("Date")
            ax1.set_ylabel("Price")
        
        # Convert to output format
        if as_array:
            # Convert to numpy array for model input
            canvas = FigureCanvas(fig)
            canvas.draw()
            
            # Get the RGBA buffer
            w, h = canvas.get_width_height()
            buf = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
            buf.shape = (h, w, 3)
            
            result = buf
        else:
            # Convert to base64 string
            buffer = io.BytesIO()
            fig.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0.1)
            buffer.seek(0)
            image_data = base64.b64encode(buffer.read()).decode('utf-8')
            
            result = image_data
            
            # Cache the result for 30 minutes
            try:
                if 'cache' in locals():
                    cache.set(cache_key, result, ttl=1800)
            except Exception as cache_err:
                logger.warning(f"Cache error when storing chart: {cache_err}")
        
        return result
    
    except Exception as e:
        logger.error(f"Error plotting candlestick chart: {e}", exc_info=True)
        return None
    finally:
        # Always close the figure to free memory
        if fig is not None:
            plt.close(fig)

def get_all_patterns_with_metadata() -> Dict[str, List[Dict[str, str]]]:
    """
    Get all candlestick patterns organized by difficulty level with metadata.
    
    Returns:
        Dictionary mapping difficulty levels to lists of pattern dictionaries
    """
    patterns_by_difficulty = {}
    
    # Get patterns for each difficulty level
    for difficulty in ["beginner", "medium", "hard"]:
        patterns = DIFFICULTY_LEVELS.get(difficulty, [])
        pattern_list = []
        
        for pattern_name in patterns:
            pattern_info = {
                "name": pattern_name,
                "description": get_pattern_description(pattern_name),
                "category": get_pattern_category(pattern_name),
                "reliability": "High" if difficulty == "beginner" else "Medium" if difficulty == "medium" else "Low",
                "example_image_url": ""  # Can be populated if example images are available
            }
            pattern_list.append(pattern_info)
        
        patterns_by_difficulty[difficulty] = pattern_list
    
    return patterns_by_difficulty
