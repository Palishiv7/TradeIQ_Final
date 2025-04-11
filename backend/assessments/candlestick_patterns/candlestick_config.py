"""
Candlestick Patterns Configuration Module

This module provides structured configuration for the candlestick pattern assessment system,
including pattern definitions, difficulty levels, market data sources, and assessment parameters.
"""

from typing import Dict, List, Any, Optional, TypedDict, Union, Final, cast, Literal, Mapping, Sequence
from enum import Enum, auto
import os
from dataclasses import dataclass, field, InitVar, asdict
from functools import lru_cache
import json

# Module version for tracking configuration changes
__version__: Final[str] = "1.1.0"

# ============================================================================
# Enums & Type Definitions
# ============================================================================

class PatternCategory(str, Enum):
    """Enum defining candlestick pattern categories."""
    SINGLE = "single"
    DOUBLE = "double"
    TRIPLE = "triple"
    COMPLEX = "complex"
    
    @classmethod
    def from_string(cls, value: str) -> "PatternCategory":
        """Convert string to enum value with validation."""
        try:
            return cls(value.lower())
        except ValueError:
            valid_values = ", ".join([m.value for m in cls])
            raise ValueError(f"Invalid pattern category: {value}. Valid values: {valid_values}")


class DifficultyLevel(str, Enum):
    """Enum defining difficulty levels for assessment questions."""
    BEGINNER = "beginner"
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXPERT = "expert"

    @classmethod
    def from_string(cls, value: str) -> "DifficultyLevel":
        """Convert string to enum value with validation."""
        try:
            return cls(value.lower())
        except ValueError:
            valid_values = ", ".join([m.value for m in cls])
            raise ValueError(f"Invalid difficulty level: {value}. Valid values: {valid_values}")
    
    def to_numeric(self) -> float:
        """Convert difficulty level to numeric value between 0 and 1."""
        mapping = {
            self.BEGINNER: 0.0,
            self.EASY: 0.25,
            self.MEDIUM: 0.5,
            self.HARD: 0.75,
            self.EXPERT: 1.0
        }
        return mapping[self]


# ============================================================================
# API Provider Configurations
# ============================================================================

class RateLimitSettings(TypedDict):
    """Rate limit settings for API providers."""
    requests_per_minute: int
    max_daily_requests: int


class DataProviderConfig(TypedDict):
    """Configuration for market data providers."""
    provider: str
    api_key: str
    base_url: str
    timeout: int
    rate_limit: RateLimitSettings


class MarketDataCommonConfig(TypedDict):
    """Common settings for market data providers."""
    default_symbols: List[str]
    default_timeframes: List[str]
    data_ttl: int
    max_candles_per_request: int


# ============================================================================
# Cache Configuration
# ============================================================================

@dataclass(frozen=True)
class CacheConfig:
    """Configuration for Redis cache settings."""
    # Cache key prefixes
    market_data_prefix: str = "candlestick:market_data:"
    session_prefix: str = "candlestick:session:"
    question_prefix: str = "candlestick:question:"
    user_stats_prefix: str = "candlestick:user:stats:"
    pattern_stats_prefix: str = "candlestick:pattern:stats:"
    rate_limit_prefix: str = "candlestick:rate_limit:"
    
    # Time-to-live settings (in seconds)
    default_ttl: int = 3600  # 1 hour
    session_ttl: int = 3600  # 1 hour
    backup_ttl: int = 86400  # 24 hours
    question_ttl: int = 86400  # 24 hours
    market_data_ttl: int = 21600  # 6 hours
    
    # Recovery settings
    max_recovery_attempts: int = 3
    recovery_retry_delay: int = 5  # seconds
    
    # Stream settings
    stream_max_length: int = 10000  # Maximum number of entries per stream
    stream_group_prefix: str = "candlestick:group:"
    
    # Rate limiting
    rate_limit_window: int = 60  # 1 minute window for rate limits
    
    # Event processing
    event_consumer_group: str = "candlestick_processors"
    event_batch_size: int = 10
    event_poll_interval: int = 1000  # ms

    def get_full_key(self, prefix: str, key: str) -> str:
        """
        Create a full Redis key with the appropriate prefix.
        
        Args:
            prefix: Prefix type ('market_data', 'session', etc.)
            key: The specific key to append
            
        Returns:
            Full Redis key with prefix
        """
        prefix_attr = f"{prefix}_prefix"
        if hasattr(self, prefix_attr):
            return f"{getattr(self, prefix_attr)}{key}"
        return f"candlestick:{prefix}:{key}"


# ============================================================================
# Assessment Configuration
# ============================================================================

@dataclass(frozen=True)
class TimeLimits:
    """Time limits for assessment questions."""
    base_seconds: int = 30
    min_seconds: int = 15
    max_seconds: int = 60

    def get_adjusted_time(self, difficulty: float) -> int:
        """
        Calculate time limit adjusted for difficulty.
        
        Args:
            difficulty: Difficulty value between 0 and 1
            
        Returns:
            Time limit in seconds
        """
        adjusted = self.base_seconds - int((difficulty * (self.base_seconds - self.min_seconds)))
        return max(self.min_seconds, min(adjusted, self.max_seconds))


@dataclass(frozen=True)
class PassingScore:
    """Minimum score requirements for passing an assessment."""
    min_score: int = 35
    min_accuracy: float = 0.7

    def is_passing(self, score: int, accuracy: float) -> bool:
        """
        Check if the score and accuracy meet passing requirements.
        
        Args:
            score: User's score
            accuracy: User's accuracy (0-1)
            
        Returns:
            True if the score meets passing requirements
        """
        return score >= self.min_score and accuracy >= self.min_accuracy


@dataclass(frozen=True)
class RateLimits:
    """Rate limits for assessment actions."""
    start_assessment: int = 5
    submit_answer: int = 20
    get_explanation: int = 30


@dataclass(frozen=True)
class AdaptiveDifficulty:
    """Settings for adaptive difficulty adjustment."""
    enabled: bool = True
    performance_weight: float = 0.3
    adaptive_ratio: float = 0.6
    min_difficulty: float = 0.1
    max_difficulty: float = 0.9

    def calculate_difficulty(self, base_difficulty: float, user_performance: float) -> float:
        """
        Calculate adjusted difficulty based on user performance.
        
        Args:
            base_difficulty: Base difficulty level (0-1)
            user_performance: User performance metric (0-1)
            
        Returns:
            Adjusted difficulty level
        """
        if not self.enabled:
            return base_difficulty
            
        # Performance adjustment: higher performance increases difficulty
        performance_factor = (user_performance - 0.5) * 2 * self.performance_weight
        adjusted = base_difficulty + performance_factor
        
        # Ensure within bounds
        return max(self.min_difficulty, min(adjusted, self.max_difficulty))


@dataclass(frozen=True)
class ScoringSettings:
    """Settings for assessment scoring."""
    base_multiplier: float = 1.0
    difficulty_weight: float = 4.0
    time_bonus_weight: float = 0.5
    streak_bonus_weight: float = 0.3
    max_streak_bonus: int = 5

    def calculate_score(
        self, 
        is_correct: bool, 
        difficulty: float, 
        time_ratio: float, 
        streak: int
    ) -> int:
        """
        Calculate score for a question based on correctness and factors.
        
        Args:
            is_correct: Whether the answer was correct
            difficulty: Question difficulty (0-1)
            time_ratio: Ratio of time taken to time allowed (0-1), lower is better
            streak: Current correct answer streak
            
        Returns:
            Calculated score
        """
        if not is_correct:
            return 0
            
        # Base score calculation
        base_score = 100 * self.base_multiplier
        
        # Apply difficulty bonus
        difficulty_bonus = difficulty * self.difficulty_weight * base_score
        
        # Apply time bonus (faster answers get higher bonus)
        time_bonus = max(0, (1 - time_ratio)) * self.time_bonus_weight * base_score
        
        # Apply streak bonus (capped)
        effective_streak = min(streak, self.max_streak_bonus)
        streak_bonus = effective_streak * self.streak_bonus_weight * base_score / self.max_streak_bonus
        
        # Calculate total score
        total = base_score + difficulty_bonus + time_bonus + streak_bonus
        
        return int(total)


@dataclass(frozen=True)
class AssessmentConfig:
    """Configuration for candlestick pattern assessments."""
    questions_per_session: int = 10
    max_questions_per_session: int = 20
    time_limits: TimeLimits = field(default_factory=TimeLimits)
    passing_score: PassingScore = field(default_factory=PassingScore)
    rate_limits: RateLimits = field(default_factory=RateLimits)
    adaptive_difficulty: AdaptiveDifficulty = field(default_factory=AdaptiveDifficulty)
    scoring: ScoringSettings = field(default_factory=ScoringSettings)
    session_expiry_seconds: int = 7200  # 2 hours
    default_questions_per_session: int = 10

    def validate_question_count(self, count: int) -> int:
        """
        Validate and adjust the question count to be within allowed limits.
        
        Args:
            count: Requested question count
            
        Returns:
            Adjusted question count
        """
        if count < 1:
            return self.default_questions_per_session
        return min(count, self.max_questions_per_session)


# ============================================================================
# Data Source Configurations
# ============================================================================

# Market data sources configuration
MARKET_DATA: Final[Dict[str, Union[DataProviderConfig, MarketDataCommonConfig]]] = {
    # Primary data source
    "primary": {
        "provider": "alphavantage",
        "api_key": os.environ.get("ALPHAVANTAGE_API_KEY", "demo"),
        "base_url": "https://www.alphavantage.co/query",
        "timeout": 10,  # seconds
        "rate_limit": {
            "requests_per_minute": 5,
            "max_daily_requests": 500
        }
    },
    # Fallback data source
    "fallback": {
        "provider": "finnhub",
        "api_key": os.environ.get("FINNHUB_API_KEY", "demo"),
        "base_url": "https://finnhub.io/api/v1",
        "timeout": 8,  # seconds
        "rate_limit": {
            "requests_per_minute": 10,
            "max_daily_requests": 60
        }
    },
    # Common settings
    "common": {
        "default_symbols": ["AAPL", "MSFT", "AMZN", "GOOG", "META", "TSLA", "NVDA", "JPM", "BAC", "WMT"],
        "default_timeframes": ["1d", "4h", "1h", "30min"],
        "data_ttl": 3600,  # Cache TTL in seconds (1 hour)
        "max_candles_per_request": 100
    }
}


# ============================================================================
# Pattern Definitions
# ============================================================================

# Candlestick patterns organized by category
CANDLESTICK_PATTERNS: Final[Mapping[PatternCategory, Sequence[str]]] = {
    PatternCategory.SINGLE: [
        "Doji",
        "Hammer",
        "Inverted Hammer",
        "Shooting Star",
        "Spinning Top",
        "Marubozu",
        "Dragonfly Doji",
        "Gravestone Doji",
        "Long-Legged Doji"
    ],
    PatternCategory.DOUBLE: [
        "Bullish Engulfing",
        "Bearish Engulfing",
        "Bullish Harami",
        "Bearish Harami",
        "Tweezer Top",
        "Tweezer Bottom",
        "Piercing Line",
        "Dark Cloud Cover",
        "Meeting Lines"
    ],
    PatternCategory.TRIPLE: [
        "Morning Star",
        "Evening Star",
        "Three White Soldiers",
        "Three Black Crows",
        "Three Inside Up",
        "Three Inside Down",
        "Three Outside Up",
        "Three Outside Down",
        "Abandoned Baby"
    ],
    PatternCategory.COMPLEX: [
        "Rising Three Methods",
        "Falling Three Methods",
        "Upside Gap Three Methods",
        "Downside Gap Three Methods",
        "Mat Hold",
        "Stick Sandwich",
        "Ladder Bottom",
        "Ladder Top",
        "Eight New Price Lines"
    ]
}


# Pattern descriptions
PATTERN_DESCRIPTIONS: Final[Mapping[str, str]] = {
    "Doji": "A Doji forms when the opening and closing prices are virtually equal. It represents market indecision, with its various forms (standard, dragonfly, gravestone, long-legged) indicating different nuances of market sentiment.",
    
    "Hammer": "The Hammer is a bullish reversal pattern that forms during a downtrend. It has a small real body at the upper end of the trading range and a long lower shadow at least twice the size of the body, indicating rejection of lower prices.",
    
    "Inverted Hammer": "The Inverted Hammer is a potential bullish reversal pattern that forms during a downtrend. It has a small real body at the lower end of the trading range and a long upper shadow, suggesting buying pressure after a period of selling.",
    
    "Shooting Star": "The Shooting Star is a bearish reversal pattern that forms during an uptrend. It has a small real body at the lower end of the trading range and a long upper shadow, indicating rejection of higher prices.",
    
    "Spinning Top": "A Spinning Top has a small body centered between relatively long upper and lower shadows. It indicates indecision between buyers and sellers, often signaling consolidation or an upcoming reversal if it appears after a strong trend.",
    
    "Marubozu": "A Marubozu is a candlestick with no shadows, meaning the high equals the open or close, and the low equals the open or close. It indicates strong conviction by either buyers (bullish Marubozu) or sellers (bearish Marubozu).",
    
    "Dragonfly Doji": "The Dragonfly Doji forms when the open, high, and close are equal or very close, with a long lower shadow. It often signals a bullish reversal when appearing at the bottom of a downtrend, as it shows strong rejection of lower prices.",
    
    "Gravestone Doji": "The Gravestone Doji forms when the open, low, and close are equal or very close, with a long upper shadow. It often signals a bearish reversal when appearing at the top of an uptrend, as it shows strong rejection of higher prices.",
    
    "Long-Legged Doji": "The Long-Legged Doji has a small body centered between unusually long upper and lower shadows. It represents extreme uncertainty in the market with significant price movement in both directions during the session.",
    
    "Bullish Engulfing": "A Bullish Engulfing pattern consists of a smaller bearish candle followed by a larger bullish candle that completely 'engulfs' the previous one. It signals a potential trend reversal from bearish to bullish.",
    
    "Bearish Engulfing": "A Bearish Engulfing pattern consists of a smaller bullish candle followed by a larger bearish candle that completely 'engulfs' the previous one. It signals a potential trend reversal from bullish to bearish.",
    
    "Bullish Harami": "The Bullish Harami is a two-candle reversal pattern where a small bullish candle is contained within the range of the previous larger bearish candle. It suggests the downtrend may be losing momentum.",
    
    "Bearish Harami": "The Bearish Harami is a two-candle reversal pattern where a small bearish candle is contained within the range of the previous larger bullish candle. It suggests the uptrend may be losing momentum.",
    
    "Tweezer Top": "The Tweezer Top consists of two consecutive candles with identical highs, the first bullish and the second bearish, appearing at the end of an uptrend. It signals a potential reversal to the downside.",
    
    "Tweezer Bottom": "The Tweezer Bottom consists of two consecutive candles with identical lows, the first bearish and the second bullish, appearing at the end of a downtrend. It signals a potential reversal to the upside.",
    
    "Piercing Line": "The Piercing Line is a two-candle bullish reversal pattern where a bearish candle is followed by a bullish candle that opens below the prior low but closes above the midpoint of the prior candle.",
    
    "Dark Cloud Cover": "The Dark Cloud Cover is a two-candle bearish reversal pattern where a bullish candle is followed by a bearish candle that opens above the prior high but closes below the midpoint of the prior candle.",
    
    "Morning Star": "The Morning Star is a bullish three-candle reversal pattern consisting of a large bearish candle, followed by a small-bodied candle that gaps down, followed by a large bullish candle. It signals a potential bullish reversal.",
    
    "Evening Star": "The Evening Star is a bearish three-candle reversal pattern consisting of a large bullish candle, followed by a small-bodied candle that gaps up, followed by a large bearish candle. It signals a potential bearish reversal.",
    
    "Three White Soldiers": "Three White Soldiers consist of three consecutive bullish candles, each with a higher close and opening within the previous candle's body. This pattern shows strong buying pressure and signals a potential bullish reversal or continuation.",
    
    "Three Black Crows": "Three Black Crows consist of three consecutive bearish candles, each with a lower close and opening within the previous candle's body. This pattern shows strong selling pressure and signals a potential bearish reversal or continuation.",
    
    "Rising Three Methods": "The Rising Three Methods consists of a long bullish candle followed by three small bearish candles contained within the range of the first candle, then a bullish candle that closes higher than the first. It signals bullish continuation.",
    
    "Falling Three Methods": "The Falling Three Methods consists of a long bearish candle followed by three small bullish candles contained within the range of the first candle, then a bearish candle that closes lower than the first. It signals bearish continuation."
}


# Difficulty levels with associated patterns
DIFFICULTY_LEVELS: Final[Mapping[DifficultyLevel, Sequence[str]]] = {
    DifficultyLevel.BEGINNER: [
        "Doji",
        "Hammer",
        "Shooting Star",
        "Bullish Engulfing",
        "Bearish Engulfing"
    ],
    DifficultyLevel.EASY: [
        "Inverted Hammer",
        "Spinning Top",
        "Marubozu",
        "Dragonfly Doji",
        "Gravestone Doji",
        "Bullish Harami",
        "Bearish Harami"
    ],
    DifficultyLevel.MEDIUM: [
        "Tweezer Top",
        "Tweezer Bottom",
        "Piercing Line",
        "Dark Cloud Cover",
        "Long-Legged Doji",
        "Morning Star",
        "Evening Star"
    ],
    DifficultyLevel.HARD: [
        "Three White Soldiers",
        "Three Black Crows",
        "Three Inside Up",
        "Three Inside Down",
        "Three Outside Up",
        "Three Outside Down"
    ],
    DifficultyLevel.EXPERT: [
        "Abandoned Baby",
        "Rising Three Methods",
        "Falling Three Methods",
        "Upside Gap Three Methods",
        "Downside Gap Three Methods",
        "Mat Hold",
        "Stick Sandwich",
        "Ladder Bottom",
        "Ladder Top"
    ]
}


# ============================================================================
# Environment Configuration
# ============================================================================

def _get_env_value(name: str, default: Any = None, prefix: str = "CANDLESTICK_") -> Any:
    """
    Get an environment variable value with a standard prefix.
    
    Args:
        name: Variable name without prefix
        default: Default value if not found
        prefix: Optional prefix to prepend to the name
        
    Returns:
        Environment variable value or default
    """
    return os.environ.get(f"{prefix}{name}", default)


def _get_env_bool(name: str, default: bool = False, prefix: str = "CANDLESTICK_") -> bool:
    """
    Get a boolean environment variable value.
    
    Args:
        name: Variable name without prefix
        default: Default value if not found
        prefix: Optional prefix to prepend to the name
        
    Returns:
        Boolean value from environment
    """
    value = _get_env_value(name, None, prefix)
    if value is None:
        return default
        
    value = value.lower()
    return value in ("1", "true", "yes", "y", "on")


def _get_env_int(name: str, default: int = 0, prefix: str = "CANDLESTICK_") -> int:
    """
    Get an integer environment variable value.
    
    Args:
        name: Variable name without prefix
        default: Default value if not found
        prefix: Optional prefix to prepend to the name
        
    Returns:
        Integer value from environment
    """
    value = _get_env_value(name, None, prefix)
    if value is None:
        return default
        
    try:
        return int(value)
    except ValueError:
        return default


def _get_env_float(name: str, default: float = 0.0, prefix: str = "CANDLESTICK_") -> float:
    """
    Get a float environment variable value.
    
    Args:
        name: Variable name without prefix
        default: Default value if not found
        prefix: Optional prefix to prepend to the name
        
    Returns:
        Float value from environment
    """
    value = _get_env_value(name, None, prefix)
    if value is None:
        return default
        
    try:
        return float(value)
    except ValueError:
        return default


def _get_env_list(name: str, default: List[str] = None, prefix: str = "CANDLESTICK_") -> List[str]:
    """
    Get a list environment variable value (comma-separated).
    
    Args:
        name: Variable name without prefix
        default: Default value if not found
        prefix: Optional prefix to prepend to the name
        
    Returns:
        List value from environment
    """
    if default is None:
        default = []
        
    value = _get_env_value(name, None, prefix)
    if value is None:
        return default
        
    return [item.strip() for item in value.split(",") if item.strip()]


def _get_env_json(name: str, default: Any = None, prefix: str = "CANDLESTICK_") -> Any:
    """
    Get a JSON environment variable value.
    
    Args:
        name: Variable name without prefix
        default: Default value if not found
        prefix: Optional prefix to prepend to the name
        
    Returns:
        Parsed JSON value from environment
    """
    value = _get_env_value(name, None, prefix)
    if value is None:
        return default
        
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return default


# ============================================================================
# Singleton Configuration Instances
# ============================================================================

# Create singleton instances of configurations with environment overrides
CACHE_CONFIG: Final[CacheConfig] = CacheConfig(
    # Cache key prefixes
    market_data_prefix=_get_env_value("CACHE_MARKET_DATA_PREFIX", "candlestick:market_data:"),
    session_prefix=_get_env_value("CACHE_SESSION_PREFIX", "candlestick:session:"),
    question_prefix=_get_env_value("CACHE_QUESTION_PREFIX", "candlestick:question:"),
    
    # Time-to-live settings
    default_ttl=_get_env_int("CACHE_DEFAULT_TTL", 3600),
    session_ttl=_get_env_int("CACHE_SESSION_TTL", 3600),
    market_data_ttl=_get_env_int("CACHE_MARKET_DATA_TTL", 21600),
    
    # Recovery settings
    max_recovery_attempts=_get_env_int("CACHE_MAX_RECOVERY_ATTEMPTS", 3),
    recovery_retry_delay=_get_env_int("CACHE_RECOVERY_RETRY_DELAY", 5),
)

ASSESSMENT_CONFIG: Final[AssessmentConfig] = AssessmentConfig(
    questions_per_session=_get_env_int("ASSESSMENT_QUESTIONS_PER_SESSION", 10),
    max_questions_per_session=_get_env_int("ASSESSMENT_MAX_QUESTIONS", 20),
    session_expiry_seconds=_get_env_int("ASSESSMENT_SESSION_EXPIRY", 7200),
    
    # Nested configurations will use their defaults
)


# ============================================================================
# Validation Utilities
# ============================================================================

class PatternValidationError(ValueError):
    """Exception raised for invalid pattern operations."""
    pass


class DifficultyValidationError(ValueError):
    """Exception raised for invalid difficulty operations."""
    pass


class ConfigValidationError(ValueError):
    """Exception raised for invalid configuration operations."""
    pass


def _normalize_pattern_name(pattern_name: str) -> str:
    """
    Normalize a pattern name for consistent lookup.
    
    Args:
        pattern_name: Name of the pattern, possibly with inconsistent casing/spacing
        
    Returns:
        Normalized pattern name for lookup
    """
    if not pattern_name or not isinstance(pattern_name, str):
        return ""
        
    # Handle capitalization variations
    words = pattern_name.strip().split()
    if not words:
        return ""
        
    normalized = " ".join(w.capitalize() for w in words)
    return normalized


def _find_closest_pattern(pattern_name: str, max_distance: int = 2) -> Optional[str]:
    """
    Find the closest matching pattern name using Levenshtein distance.
    
    Args:
        pattern_name: The input pattern name to match
        max_distance: Maximum edit distance to consider a match
        
    Returns:
        The closest matching pattern or None if no match within distance
    """
    if not pattern_name:
        return None
        
    import difflib
    normalized = _normalize_pattern_name(pattern_name)
    
    if not normalized:
        return None
        
    # Get all valid pattern names
    valid_patterns = list(PATTERN_DESCRIPTIONS.keys())
    
    # Find closest matches
    matches = difflib.get_close_matches(normalized, valid_patterns, n=1, cutoff=0.7)
    
    if matches:
        return matches[0]
    
    return None


# ============================================================================
# Helper Functions
# ============================================================================

# Add a registry of all patterns across categories for fast lookup
_ALL_PATTERNS: Final[set] = {
    pattern 
    for patterns in CANDLESTICK_PATTERNS.values() 
    for pattern in patterns
}

# Keep a registry of pattern to category mapping for fast lookup
_PATTERN_TO_CATEGORY: Final[Dict[str, PatternCategory]] = {
    pattern: category
    for category, patterns in CANDLESTICK_PATTERNS.items()
    for pattern in patterns
}

# Keep a registry of pattern to difficulty mapping for fast lookup
_PATTERN_TO_DIFFICULTY: Final[Dict[str, DifficultyLevel]] = {
    pattern: difficulty
    for difficulty, patterns in DIFFICULTY_LEVELS.items()
    for pattern in patterns
}

@lru_cache(maxsize=32)
def get_patterns_by_category(category: Union[PatternCategory, str]) -> List[str]:
    """
    Get patterns for a specific category.
    
    Args:
        category: PatternCategory enum or string value
    
    Returns:
        List of pattern names
    
    Raises:
        PatternValidationError: If the category is invalid
    """
    try:
        if isinstance(category, str):
            cat = PatternCategory.from_string(category)
        else:
            cat = category
            
        patterns = CANDLESTICK_PATTERNS.get(cat, [])
        return list(patterns)  # Return a copy to prevent modification
    except ValueError as e:
        raise PatternValidationError(f"Invalid pattern category: {e}")


@lru_cache(maxsize=32)
def get_patterns_by_difficulty(difficulty: Union[DifficultyLevel, str]) -> List[str]:
    """
    Get patterns for a specific difficulty level.
    
    Args:
        difficulty: DifficultyLevel enum or string value
    
    Returns:
        List of pattern names
        
    Raises:
        DifficultyValidationError: If the difficulty level is invalid
    """
    try:
        if isinstance(difficulty, str):
            diff = DifficultyLevel.from_string(difficulty)
        else:
            diff = difficulty
            
        patterns = DIFFICULTY_LEVELS.get(diff, [])
        return list(patterns)  # Return a copy to prevent modification
    except ValueError as e:
        raise DifficultyValidationError(f"Invalid difficulty level: {e}")


@lru_cache(maxsize=128)
def get_pattern_description(pattern_name: str) -> Optional[str]:
    """
    Get description for a specific pattern.
    
    Args:
        pattern_name: Name of the pattern
    
    Returns:
        Pattern description or None if not found
    """
    normalized = _normalize_pattern_name(pattern_name)
    return PATTERN_DESCRIPTIONS.get(normalized)


def difficulty_to_level(difficulty_value: float) -> DifficultyLevel:
    """
    Convert a numeric difficulty value to the corresponding difficulty level.
    
    Args:
        difficulty_value: Difficulty value between 0 and 1
        
    Returns:
        The corresponding difficulty level enum
        
    Raises:
        DifficultyValidationError: If the difficulty value is outside the valid range
    """
    if not isinstance(difficulty_value, (int, float)):
        raise DifficultyValidationError(f"Difficulty value must be a number, got {type(difficulty_value)}")
        
    # Ensure the value is in range
    bounded_value = max(0.0, min(1.0, float(difficulty_value)))
    
    if bounded_value < 0.2:
        return DifficultyLevel.BEGINNER
    elif bounded_value < 0.4:
        return DifficultyLevel.EASY
    elif bounded_value < 0.6:
        return DifficultyLevel.MEDIUM
    elif bounded_value < 0.8:
        return DifficultyLevel.HARD
    else:
        return DifficultyLevel.EXPERT


def validate_pattern_name(pattern_name: str) -> bool:
    """
    Check if a pattern name is valid and exists in our configurations.
    
    Args:
        pattern_name: Name of the pattern to validate
        
    Returns:
        True if the pattern exists, False otherwise
    """
    normalized = _normalize_pattern_name(pattern_name)
    return normalized in _ALL_PATTERNS


def get_pattern_category(pattern_name: str) -> Optional[PatternCategory]:
    """
    Get the category of a pattern.
    
    Args:
        pattern_name: Name of the pattern
        
    Returns:
        The pattern category or None if not found
    """
    normalized = _normalize_pattern_name(pattern_name)
    return _PATTERN_TO_CATEGORY.get(normalized)


def get_pattern_difficulty(pattern_name: str) -> Optional[DifficultyLevel]:
    """
    Get the difficulty level of a pattern.
    
    Args:
        pattern_name: Name of the pattern
        
    Returns:
        The pattern difficulty level or None if not found
    """
    normalized = _normalize_pattern_name(pattern_name)
    return _PATTERN_TO_DIFFICULTY.get(normalized)


def suggest_pattern(pattern_name: str) -> Optional[str]:
    """
    Suggest a valid pattern name for an invalid input.
    
    Args:
        pattern_name: The potentially invalid pattern name
        
    Returns:
        A suggested valid pattern name or None if no good suggestion
    """
    # First try direct normalization
    normalized = _normalize_pattern_name(pattern_name)
    if normalized in _ALL_PATTERNS:
        return normalized
        
    # Then try fuzzy matching
    return _find_closest_pattern(pattern_name)


def get_patterns_all() -> List[str]:
    """
    Get a list of all available patterns.
    
    Returns:
        List of all pattern names
    """
    return list(_ALL_PATTERNS)


# ============================================================================
# Configuration Validation
# ============================================================================

def validate_configuration() -> Dict[str, List[str]]:
    """
    Validate the entire configuration for consistency.
    
    This function checks for:
    - Pattern names consistency across categories
    - Difficulty levels consistency
    - Data provider configurations
    - Assessment settings validity
    
    Returns:
        Dictionary mapping section names to lists of warning messages
    """
    warnings: Dict[str, List[str]] = {
        "patterns": [],
        "difficulty": [],
        "market_data": [],
        "assessment": [],
    }
    
    # Check for pattern name uniqueness across categories
    all_patterns = []
    for category, patterns in CANDLESTICK_PATTERNS.items():
        all_patterns.extend(patterns)
    
    # Check for duplicates
    pattern_counts = {}
    for pattern in all_patterns:
        pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        
    for pattern, count in pattern_counts.items():
        if count > 1:
            warnings["patterns"].append(
                f"Pattern '{pattern}' appears in multiple categories ({count} times)"
            )
    
    # Check all patterns have descriptions
    missing_descriptions = []
    for pattern in all_patterns:
        if pattern not in PATTERN_DESCRIPTIONS:
            missing_descriptions.append(pattern)
            
    if missing_descriptions:
        warnings["patterns"].append(
            f"Missing descriptions for patterns: {', '.join(missing_descriptions)}"
        )
    
    # Check difficulty levels
    all_difficulty_patterns = []
    for difficulty, patterns in DIFFICULTY_LEVELS.items():
        all_difficulty_patterns.extend(patterns)
        
    # Check for patterns not assigned to any difficulty
    unassigned_patterns = set(all_patterns) - set(all_difficulty_patterns)
    if unassigned_patterns:
        warnings["difficulty"].append(
            f"Patterns not assigned to any difficulty level: {', '.join(unassigned_patterns)}"
        )
    
    # Check for patterns in difficulty but not in main patterns
    unknown_patterns = set(all_difficulty_patterns) - set(all_patterns)
    if unknown_patterns:
        warnings["difficulty"].append(
            f"Unknown patterns in difficulty levels: {', '.join(unknown_patterns)}"
        )
    
    # Check market data configuration
    market_data_keys = MARKET_DATA.keys()
    if "primary" not in market_data_keys:
        warnings["market_data"].append("Missing 'primary' data provider configuration")
    
    if "common" not in market_data_keys:
        warnings["market_data"].append("Missing 'common' market data configuration")
    
    # Validate common settings
    if "common" in market_data_keys:
        common = MARKET_DATA["common"]
        if not common.get("default_symbols"):
            warnings["market_data"].append("No default symbols configured")
        
        if not common.get("default_timeframes"):
            warnings["market_data"].append("No default timeframes configured")
    
    # Validate assessment configuration
    assessment = ASSESSMENT_CONFIG
    if assessment.questions_per_session > assessment.max_questions_per_session:
        warnings["assessment"].append(
            "Default questions per session exceeds maximum allowed"
        )
    
    # Check time limits
    time_limits = assessment.time_limits
    if time_limits.min_seconds <= 0:
        warnings["assessment"].append("Minimum time limit should be greater than 0")
    
    if time_limits.min_seconds >= time_limits.max_seconds:
        warnings["assessment"].append(
            "Minimum time limit should be less than maximum time limit"
        )
    
    # Return all warnings
    return warnings


# ============================================================================
# Environment Variable Documentation
# ============================================================================

"""
Environment Variable Configuration

The following environment variables can be used to customize the configuration:

Cache Configuration:
- CANDLESTICK_CACHE_MARKET_DATA_PREFIX: Prefix for market data cache keys
- CANDLESTICK_CACHE_SESSION_PREFIX: Prefix for session cache keys
- CANDLESTICK_CACHE_QUESTION_PREFIX: Prefix for question cache keys
- CANDLESTICK_CACHE_DEFAULT_TTL: Default TTL in seconds (default: 3600)
- CANDLESTICK_CACHE_SESSION_TTL: Session TTL in seconds (default: 3600)
- CANDLESTICK_CACHE_MARKET_DATA_TTL: Market data TTL in seconds (default: 21600)
- CANDLESTICK_CACHE_MAX_RECOVERY_ATTEMPTS: Maximum recovery attempts (default: 3)
- CANDLESTICK_CACHE_RECOVERY_RETRY_DELAY: Recovery retry delay in seconds (default: 5)

Assessment Configuration:
- CANDLESTICK_ASSESSMENT_QUESTIONS_PER_SESSION: Default questions per session (default: 10)
- CANDLESTICK_ASSESSMENT_MAX_QUESTIONS: Maximum questions per session (default: 20)
- CANDLESTICK_ASSESSMENT_SESSION_EXPIRY: Session expiry in seconds (default: 7200)

API Keys:
- ALPHAVANTAGE_API_KEY: API key for Alpha Vantage market data
- FINNHUB_API_KEY: API key for Finnhub market data

Feature Flags:
- CANDLESTICK_ENABLE_ADAPTIVE_DIFFICULTY: Enable/disable adaptive difficulty (default: TRUE)
- CANDLESTICK_ENABLE_PERFORMANCE_TRACKING: Enable/disable performance tracking (default: TRUE)
"""


def apply_override_from_config_file(config_file_path: str) -> None:
    """
    Apply configuration overrides from a JSON configuration file.
    
    Args:
        config_file_path: Path to the JSON configuration file
        
    Raises:
        ConfigValidationError: If the configuration file is invalid
    """
    try:
        with open(config_file_path, 'r') as f:
            config = json.load(f)
            
        # Apply overrides to environment variables
        for category, values in config.items():
            if category == "cache":
                for key, value in values.items():
                    os.environ[f"CANDLESTICK_CACHE_{key.upper()}"] = str(value)
            
            elif category == "assessment":
                for key, value in values.items():
                    os.environ[f"CANDLESTICK_ASSESSMENT_{key.upper()}"] = str(value)
            
            elif category == "api_keys":
                for provider, key in values.items():
                    os.environ[f"{provider.upper()}_API_KEY"] = key
    
    except json.JSONDecodeError as e:
        raise ConfigValidationError(f"Invalid JSON in configuration file: {e}")
    except IOError as e:
        raise ConfigValidationError(f"Error reading configuration file: {e}")
    except Exception as e:
        raise ConfigValidationError(f"Unexpected error processing configuration: {e}")


# ============================================================================
# Configuration Export
# ============================================================================

def export_config_to_json() -> Dict[str, Any]:
    """
    Export the current configuration to a JSON-serializable dictionary.
    
    Returns:
        Dictionary with all configuration values
    """
    config = {
        "version": __version__,
        "cache": asdict(CACHE_CONFIG),
        "assessment": asdict(ASSESSMENT_CONFIG),
        "market_data": dict(MARKET_DATA),
        "patterns": {
            "by_category": {
                category.value: list(patterns) 
                for category, patterns in CANDLESTICK_PATTERNS.items()
            },
            "by_difficulty": {
                difficulty.value: list(patterns) 
                for difficulty, patterns in DIFFICULTY_LEVELS.items()
            },
            "total_count": len(_ALL_PATTERNS)
        }
    }
    
    return config


# ============================================================================
# Module Exports
# ============================================================================

# Export all configurations and helper functions
__all__ = [
    # Version
    "__version__",
    
    # Enums
    "PatternCategory",
    "DifficultyLevel",
    
    # Classes
    "CacheConfig",
    "AssessmentConfig",
    "TimeLimits",
    "PassingScore",
    "RateLimits",
    "AdaptiveDifficulty",
    "ScoringSettings",
    
    # Exception classes
    "PatternValidationError",
    "DifficultyValidationError",
    "ConfigValidationError",
    
    # Constants
    "MARKET_DATA",
    "CANDLESTICK_PATTERNS",
    "PATTERN_DESCRIPTIONS",
    "DIFFICULTY_LEVELS",
    "CACHE_CONFIG",
    "ASSESSMENT_CONFIG",
    
    # Helper functions
    "get_patterns_by_category",
    "get_patterns_by_difficulty",
    "get_pattern_description",
    "difficulty_to_level",
    "validate_pattern_name",
    "get_pattern_category",
    "get_pattern_difficulty",
    "suggest_pattern",
    "get_patterns_all",
    
    # Config validation and loading
    "validate_configuration",
    "apply_override_from_config_file",
    
    # Config loading functions
    "export_config_to_json"
]
