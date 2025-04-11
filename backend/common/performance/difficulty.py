"""
Adaptive Difficulty Management

This module provides components for managing and adjusting difficulty levels
based on user performance metrics, enabling personalized learning experiences.
"""

import enum
import math
import datetime
import statistics
from typing import Dict, List, Any, Optional, Tuple, Callable

from backend.common.logger import app_logger
from backend.common.performance.tracker import SkillLevel

# Module logger
logger = app_logger.getChild("performance.difficulty")


class DifficultyLevel(enum.Enum):
    """Difficulty levels for assessment questions."""
    VERY_EASY = "very_easy"
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    VERY_HARD = "very_hard"
    
    @classmethod
    def from_numeric(cls, value: int) -> 'DifficultyLevel':
        """Convert a numeric value (1-5) to a difficulty level."""
        mapping = {
            1: cls.VERY_EASY,
            2: cls.EASY,
            3: cls.MEDIUM,
            4: cls.HARD,
            5: cls.VERY_HARD
        }
        return mapping.get(value, cls.MEDIUM)
    
    def to_numeric(self) -> int:
        """Convert difficulty level to a numeric value (1-5)."""
        return {
            DifficultyLevel.VERY_EASY: 1,
            DifficultyLevel.EASY: 2,
            DifficultyLevel.MEDIUM: 3,
            DifficultyLevel.HARD: 4,
            DifficultyLevel.VERY_HARD: 5
        }[self]


@enum.unique
class AdjustmentStrategy(enum.Enum):
    """Strategies for adjusting difficulty."""
    FIXED_STEP = "fixed_step"  # Adjust by a fixed amount
    PROPORTIONAL = "proportional"  # Adjust proportionally to performance gap
    DYNAMIC = "dynamic"  # Adjust based on learning rate and performance variability


class DifficultyAdjustment:
    """
    Represents a difficulty adjustment with metadata.
    
    Tracks when and why a difficulty adjustment was made, along with
    the previous and new difficulty levels.
    """
    
    def __init__(
        self,
        previous_level: DifficultyLevel,
        new_level: DifficultyLevel,
        reason: str,
        timestamp: Optional[datetime.datetime] = None,
        performance_metric: Optional[float] = None
    ):
        """
        Initialize a difficulty adjustment.
        
        Args:
            previous_level: Previous difficulty level
            new_level: New difficulty level
            reason: Reason for the adjustment
            timestamp: When the adjustment occurred
            performance_metric: Related performance metric that triggered the adjustment
        """
        self.previous_level = previous_level
        self.new_level = new_level
        self.reason = reason
        self.timestamp = timestamp or datetime.datetime.now()
        self.performance_metric = performance_metric
    
    @property
    def magnitude(self) -> int:
        """Get the magnitude of the adjustment."""
        return self.new_level.to_numeric() - self.previous_level.to_numeric()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "previous_level": self.previous_level.value,
            "new_level": self.new_level.value,
            "reason": self.reason,
            "timestamp": self.timestamp.isoformat(),
            "performance_metric": self.performance_metric,
            "magnitude": self.magnitude
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DifficultyAdjustment':
        """Create from dictionary."""
        return cls(
            previous_level=DifficultyLevel(data["previous_level"]),
            new_level=DifficultyLevel(data["new_level"]),
            reason=data["reason"],
            timestamp=datetime.datetime.fromisoformat(data["timestamp"]),
            performance_metric=data.get("performance_metric")
        )


class DifficultyManager:
    """
    Manages difficulty levels for different topics and question types.
    
    Provides interfaces to get appropriate difficulty levels based on
    various parameters and user performance.
    """
    
    def __init__(
        self,
        default_level: DifficultyLevel = DifficultyLevel.MEDIUM,
        min_level: DifficultyLevel = DifficultyLevel.VERY_EASY,
        max_level: DifficultyLevel = DifficultyLevel.VERY_HARD
    ):
        """
        Initialize the difficulty manager.
        
        Args:
            default_level: Default difficulty level
            min_level: Minimum allowed difficulty level
            max_level: Maximum allowed difficulty level
        """
        self.default_level = default_level
        self.min_level = min_level
        self.max_level = max_level
        
        # Topic difficulty mappings
        self.topic_levels: Dict[str, DifficultyLevel] = {}
        
        # History of difficulty adjustments
        self.adjustment_history: List[DifficultyAdjustment] = []
    
    def get_difficulty(self, topic: Optional[str] = None) -> DifficultyLevel:
        """
        Get the current difficulty level for a topic.
        
        Args:
            topic: Topic to get difficulty for
            
        Returns:
            Current difficulty level
        """
        if topic and topic in self.topic_levels:
            return self.topic_levels[topic]
        return self.default_level
    
    def set_difficulty(
        self,
        level: DifficultyLevel,
        topic: Optional[str] = None,
        reason: str = "manual",
        performance_metric: Optional[float] = None
    ) -> None:
        """
        Set the difficulty level for a topic.
        
        Args:
            level: New difficulty level
            topic: Topic to set difficulty for
            reason: Reason for the change
            performance_metric: Optional performance metric that led to the change
        """
        # Clamp to allowed range
        numeric_level = level.to_numeric()
        numeric_level = max(self.min_level.to_numeric(), min(numeric_level, self.max_level.to_numeric()))
        level = DifficultyLevel.from_numeric(numeric_level)
        
        # Record the adjustment
        previous_level = self.get_difficulty(topic)
        adjustment = DifficultyAdjustment(
            previous_level=previous_level,
            new_level=level,
            reason=reason,
            performance_metric=performance_metric
        )
        
        self.adjustment_history.append(adjustment)
        
        # Actually set the level
        if topic:
            self.topic_levels[topic] = level
        else:
            self.default_level = level
        
        logger.debug(f"Adjusted difficulty for {topic or 'default'} from {previous_level.value} to {level.value} ({reason})")
    
    def adjust_difficulty(
        self,
        adjustment: int,
        topic: Optional[str] = None,
        reason: str = "algorithm",
        performance_metric: Optional[float] = None
    ) -> DifficultyLevel:
        """
        Adjust difficulty level by a relative amount.
        
        Args:
            adjustment: Relative adjustment (-4 to +4)
            topic: Topic to adjust difficulty for
            reason: Reason for the adjustment
            performance_metric: Optional performance metric that led to the adjustment
            
        Returns:
            New difficulty level
        """
        current_level = self.get_difficulty(topic)
        numeric_level = current_level.to_numeric() + adjustment
        
        # Clamp to allowed range
        numeric_level = max(self.min_level.to_numeric(), min(numeric_level, self.max_level.to_numeric()))
        new_level = DifficultyLevel.from_numeric(numeric_level)
        
        # Set the new level
        self.set_difficulty(new_level, topic, reason, performance_metric)
        
        return new_level
    
    def get_recent_adjustments(self, count: int = 5) -> List[DifficultyAdjustment]:
        """Get the most recent difficulty adjustments."""
        return sorted(
            self.adjustment_history,
            key=lambda adj: adj.timestamp,
            reverse=True
        )[:count]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "default_level": self.default_level.value,
            "min_level": self.min_level.value,
            "max_level": self.max_level.value,
            "topic_levels": {
                topic: level.value for topic, level in self.topic_levels.items()
            },
            "adjustment_history": [
                adj.to_dict() for adj in self.adjustment_history
            ]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DifficultyManager':
        """Create from dictionary."""
        manager = cls(
            default_level=DifficultyLevel(data.get("default_level", DifficultyLevel.MEDIUM.value)),
            min_level=DifficultyLevel(data.get("min_level", DifficultyLevel.VERY_EASY.value)),
            max_level=DifficultyLevel(data.get("max_level", DifficultyLevel.VERY_HARD.value))
        )
        
        # Restore topic levels
        for topic, level_value in data.get("topic_levels", {}).items():
            manager.topic_levels[topic] = DifficultyLevel(level_value)
        
        # Restore adjustment history
        manager.adjustment_history = [
            DifficultyAdjustment.from_dict(adj_data)
            for adj_data in data.get("adjustment_history", [])
        ]
        
        return manager


class AdaptiveDifficultyEngine:
    """
    Engine for adaptive difficulty adjustment based on user performance.
    
    This implements various strategies for dynamically adjusting difficulty
    to keep users in an optimal learning zone (not too easy, not too hard).
    """
    
    def __init__(
        self,
        difficulty_manager: DifficultyManager,
        strategy: AdjustmentStrategy = AdjustmentStrategy.DYNAMIC,
        target_success_rate: float = 0.7,  # Aim for 70% correct
        adjustment_threshold: float = 0.1,  # Adjust if off by 10%
        learning_rate_weight: float = 0.3,  # How much to factor in learning rate
        minimum_attempts: int = 5  # Minimum attempts before adjusting
    ):
        """
        Initialize the adaptive difficulty engine.
        
        Args:
            difficulty_manager: Difficulty manager to use
            strategy: Strategy for adjusting difficulty
            target_success_rate: Target success rate (0-1)
            adjustment_threshold: Threshold for adjustment (0-1)
            learning_rate_weight: Weight for learning rate in adjustments
            minimum_attempts: Minimum attempts before adjusting difficulty
        """
        self.difficulty_manager = difficulty_manager
        self.strategy = strategy
        self.target_success_rate = target_success_rate
        self.adjustment_threshold = adjustment_threshold
        self.learning_rate_weight = learning_rate_weight
        self.minimum_attempts = minimum_attempts
        
        # Performance data
        self.topic_success_rates: Dict[str, List[float]] = {}
        self.topic_attempt_counts: Dict[str, int] = {}
    
    def record_attempt(
        self,
        topic: str,
        success: bool,
        current_difficulty: Optional[DifficultyLevel] = None
    ) -> None:
        """
        Record an attempt for a topic.
        
        Args:
            topic: Topic of the attempt
            success: Whether the attempt was successful
            current_difficulty: Current difficulty level used for the attempt
        """
        # Initialize if needed
        if topic not in self.topic_success_rates:
            self.topic_success_rates[topic] = []
            self.topic_attempt_counts[topic] = 0
        
        # Record the attempt
        self.topic_success_rates[topic].append(1.0 if success else 0.0)
        self.topic_attempt_counts[topic] += 1
        
        # Keep only the last 20 attempts
        if len(self.topic_success_rates[topic]) > 20:
            self.topic_success_rates[topic] = self.topic_success_rates[topic][-20:]
    
    def get_recommended_difficulty(
        self, 
        topic: str,
        skill_level: Optional[SkillLevel] = None
    ) -> DifficultyLevel:
        """
        Get recommended difficulty for a topic based on performance.
        
        Args:
            topic: Topic to get difficulty for
            skill_level: Optional user skill level
            
        Returns:
            Recommended difficulty level
        """
        # If we don't have enough data, use current difficulty or map from skill level
        if (topic not in self.topic_attempt_counts or 
                self.topic_attempt_counts[topic] < self.minimum_attempts):
            if skill_level:
                # Map skill level to appropriate difficulty
                skill_to_difficulty = {
                    SkillLevel.BEGINNER: DifficultyLevel.VERY_EASY,
                    SkillLevel.NOVICE: DifficultyLevel.EASY,
                    SkillLevel.INTERMEDIATE: DifficultyLevel.MEDIUM,
                    SkillLevel.ADVANCED: DifficultyLevel.HARD,
                    SkillLevel.EXPERT: DifficultyLevel.VERY_HARD
                }
                return skill_to_difficulty.get(skill_level, self.difficulty_manager.get_difficulty(topic))
            else:
                return self.difficulty_manager.get_difficulty(topic)
        
        # Calculate current success rate (last N attempts)
        current_success_rate = statistics.mean(self.topic_success_rates[topic])
        
        # Determine if adjustment is needed
        if abs(current_success_rate - self.target_success_rate) < self.adjustment_threshold:
            # Within acceptable range, no adjustment needed
            return self.difficulty_manager.get_difficulty(topic)
        
        # Calculate adjustment based on strategy
        adjustment = self._calculate_adjustment(topic, current_success_rate)
        
        # Apply the adjustment
        return self.difficulty_manager.adjust_difficulty(
            adjustment=adjustment,
            topic=topic,
            reason="adaptive",
            performance_metric=current_success_rate
        )
    
    def _calculate_adjustment(self, topic: str, current_success_rate: float) -> int:
        """
        Calculate difficulty adjustment based on the selected strategy.
        
        Args:
            topic: Topic to calculate adjustment for
            current_success_rate: Current success rate
            
        Returns:
            Recommended adjustment (-4 to +4)
        """
        success_gap = current_success_rate - self.target_success_rate
        
        if self.strategy == AdjustmentStrategy.FIXED_STEP:
            # Simple fixed step: too easy -> harder, too hard -> easier
            if success_gap > 0:
                return 1  # Too easy, increase difficulty
            else:
                return -1  # Too hard, decrease difficulty
        
        elif self.strategy == AdjustmentStrategy.PROPORTIONAL:
            # Proportional adjustment based on how far off target
            # Scale to range -2 to +2
            return int(math.copysign(min(abs(success_gap / 0.2), 2), success_gap))
        
        elif self.strategy == AdjustmentStrategy.DYNAMIC:
            # Dynamic adjustment based on success gap and performance variability
            
            # Base adjustment from success gap
            base_adjustment = int(math.copysign(min(abs(success_gap / 0.15), 3), success_gap))
            
            # Calculate performance variability
            if len(self.topic_success_rates[topic]) >= 5:
                try:
                    variability = statistics.stdev(self.topic_success_rates[topic])
                    
                    # If high variability, make smaller adjustments
                    if variability > 0.3:  # High variability
                        base_adjustment = int(base_adjustment / 2)
                except statistics.StatisticsError:
                    pass  # Not enough data or all values are the same
            
            return base_adjustment
        
        return 0  # Fallback: no adjustment
    
    def reset_topic_data(self, topic: Optional[str] = None) -> None:
        """
        Reset performance data for a topic or all topics.
        
        Args:
            topic: Optional topic to reset, or None for all
        """
        if topic:
            if topic in self.topic_success_rates:
                self.topic_success_rates[topic] = []
                self.topic_attempt_counts[topic] = 0
        else:
            self.topic_success_rates = {}
            self.topic_attempt_counts = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "strategy": self.strategy.value,
            "target_success_rate": self.target_success_rate,
            "adjustment_threshold": self.adjustment_threshold,
            "learning_rate_weight": self.learning_rate_weight,
            "minimum_attempts": self.minimum_attempts,
            "topic_success_rates": self.topic_success_rates,
            "topic_attempt_counts": self.topic_attempt_counts,
            "difficulty_manager": self.difficulty_manager.to_dict()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AdaptiveDifficultyEngine':
        """Create from dictionary."""
        difficulty_manager = DifficultyManager.from_dict(
            data.get("difficulty_manager", {})
        )
        
        engine = cls(
            difficulty_manager=difficulty_manager,
            strategy=AdjustmentStrategy(data.get("strategy", AdjustmentStrategy.DYNAMIC.value)),
            target_success_rate=data.get("target_success_rate", 0.7),
            adjustment_threshold=data.get("adjustment_threshold", 0.1),
            learning_rate_weight=data.get("learning_rate_weight", 0.3),
            minimum_attempts=data.get("minimum_attempts", 5)
        )
        
        # Restore performance data
        engine.topic_success_rates = data.get("topic_success_rates", {})
        engine.topic_attempt_counts = data.get("topic_attempt_counts", {})
        
        return engine 