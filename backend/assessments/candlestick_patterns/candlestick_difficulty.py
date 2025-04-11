"""
Difficulty Management Module for Candlestick Pattern Assessments

This module implements:
1. Difficulty level management for patterns
2. Progression curve implementation
3. Performance tracking for dynamic difficulty adjustment
4. Integration with the adaptive difficulty engine

The module provides a smooth difficulty progression based on user performance,
implementing forgetting curve algorithms and sophisticated user profiling.
"""

import math
import random
import json
import time
from enum import Enum
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field

# Import from base assessment architecture
from backend.common.performance.difficulty import AdaptiveDifficultyEngine as BaseAdaptiveDifficultyEngine
from backend.common.performance.difficulty import DifficultyManager as BaseDifficultyManager
from backend.common.performance.difficulty import DifficultyLevel as BaseDifficultyLevel
from backend.common.performance.difficulty import AdjustmentStrategy

from backend.assessments.candlestick_patterns.candlestick_config import (
    CANDLESTICK_PATTERNS, DIFFICULTY_LEVELS, ASSESSMENT_CONFIG
)
from backend.assessments.candlestick_patterns.candlestick_utils import get_pattern_category
from backend.common.cache import get_cache_client
from backend.common.logger import get_logger

logger = get_logger(__name__)
cache_client = get_cache_client()

class DifficultyLevel(Enum):
    """Enum representing difficulty levels."""
    BEGINNER = "beginner"
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXPERT = "expert"
    
    @classmethod
    def from_base(cls, base_level: BaseDifficultyLevel) -> 'DifficultyLevel':
        """Convert base difficulty level to candlestick difficulty level."""
        mapping = {
            BaseDifficultyLevel.VERY_EASY: cls.BEGINNER,
            BaseDifficultyLevel.EASY: cls.EASY,
            BaseDifficultyLevel.MEDIUM: cls.MEDIUM,
            BaseDifficultyLevel.HARD: cls.HARD,
            BaseDifficultyLevel.VERY_HARD: cls.EXPERT
        }
        return mapping.get(base_level, cls.MEDIUM)
    
    def to_base(self) -> BaseDifficultyLevel:
        """Convert to base difficulty level."""
        mapping = {
            self.BEGINNER: BaseDifficultyLevel.VERY_EASY,
            self.EASY: BaseDifficultyLevel.EASY,
            self.MEDIUM: BaseDifficultyLevel.MEDIUM,
            self.HARD: BaseDifficultyLevel.HARD,
            self.EXPERT: BaseDifficultyLevel.VERY_HARD
        }
        return mapping.get(self, BaseDifficultyLevel.MEDIUM)

class DifficultyManager(BaseDifficultyManager):
    """
    Manages difficulty levels for candlestick pattern assessments.
    
    This class extends the base DifficultyManager and handles:
    1. Static difficulty based on a smooth progression curve
    2. Adaptive difficulty based on user performance
    3. Pattern selection based on difficulty
    4. Time pressure calculation
    5. Score calculation
    """
    
    def __init__(self, user_id: Optional[str] = None):
        """
        Initialize the difficulty manager.
        
        Args:
            user_id: Optional user ID for personalized difficulty
        """
        super().__init__(user_id)
        self.config = ASSESSMENT_CONFIG
        self.performance_tracker = UserPerformanceTracker(user_id) if user_id else None
        self.forgetting_curve = ForgettingCurveManager() if user_id else None
        
        # Cache for pattern difficulties
        self._pattern_difficulty_cache = {}
        
        # Initialize with default data - user data will be loaded later if needed
        self._user_data = None
    
    async def load_user_data(self):
        """Load user-specific difficulty data from cache."""
        if not self.user_id:
            return False
        
        try:
            # Try to get user data from Redis
            key = f"candlestick:user_difficulty:{self.user_id}"
            user_data = await redis_client.get(key)
            
            # Initialize user_data if it doesn't exist
            if user_data is None:
                logger.info(f"Initializing new user difficulty data for user {self.user_id}")
                self._user_data = {
                    "performance": {},
                    "forgetting_curve": {},
                    "pattern_difficulties": {}
                }
                return True
            else:
                # Parse JSON string if it's returned as a string
                if isinstance(user_data, str):
                    try:
                        self._user_data = json.loads(user_data)
                    except json.JSONDecodeError:
                        logger.error(f"Error parsing user data JSON for {self.user_id}")
                        self._user_data = {
                            "performance": {},
                            "forgetting_curve": {},
                            "pattern_difficulties": {}
                        }
                else:
                    # Otherwise use the data directly
                    self._user_data = user_data
            
            # Initialize if it's still None
            if self._user_data is None:
                self._user_data = {
                    "performance": {},
                    "forgetting_curve": {},
                    "pattern_difficulties": {}
                }
            
            # Update performance tracker with loaded data
            if self.performance_tracker and "performance" in self._user_data:
                self.performance_tracker.update_from_data(self._user_data.get("performance", {}))
            
            # Update forgetting curve data
            if self.forgetting_curve and "forgetting_curve" in self._user_data:
                self.forgetting_curve.update_from_data(self._user_data.get("forgetting_curve", {}))
                
            # Update pattern difficulty cache
            self._pattern_difficulty_cache = self._user_data.get("pattern_difficulties", {})
            
            return True
                
        except Exception as e:
            logger.error(f"Error loading user difficulty data: {e}")
            # Initialize with default data
            self._user_data = {
                "performance": {},
                "forgetting_curve": {},
                "pattern_difficulties": {}
            }
            return False
    
    async def _save_user_data(self):
        """Save user-specific difficulty data to cache."""
        if not self.user_id:
            return
        
        try:
            # Prepare data to save
            data = {}
            
            if self.performance_tracker:
                data["performance"] = self.performance_tracker.to_dict()
            
            if self.forgetting_curve:
                data["forgetting_curve"] = self.forgetting_curve.to_dict()
            
            # Save to Redis with TTL
            await redis_client.set(
                f"candlestick:user_difficulty:{self.user_id}",
                json.dumps(data),
                expire=86400 * 30  # 30 days TTL
            )
            
            logger.info(f"Saved difficulty data for user {self.user_id}")
            
        except Exception as e:
            logger.error(f"Error saving user difficulty data: {e}")
    
    def calculate_static_difficulty(self, question_number: int, total_questions: int) -> float:
        """
        Calculate static difficulty based on question progression.
        
        This uses a sigmoid function to create a smooth progression curve
        that starts easier and gradually increases in difficulty.
        
        Args:
            question_number: Current question number (1-based)
            total_questions: Total questions in the assessment
            
        Returns:
            Difficulty value from 0.0 to 1.0
        """
        # Use sigmoid function for smooth progression
        progress = (question_number - 1) / max(1, total_questions - 1)
        
        # Scale sigmoid to start easier and end harder
        # x ranges from -6 to 6 for a good sigmoid curve
        x = 12 * progress - 6
        
        # Calculate sigmoid: 1 / (1 + e^-x)
        sigmoid = 1 / (1 + math.exp(-x))
        
        # Scale to difficulty range (0.1 to 0.9 to avoid extremes)
        difficulty = 0.1 + (sigmoid * 0.8)
        
        return difficulty
    
    def calculate_adaptive_difficulty(self, base_difficulty: float) -> float:
        """
        Calculate adaptive difficulty based on user performance.
        
        Args:
            base_difficulty: Base difficulty value
            
        Returns:
            Adjusted difficulty value
        """
        if not self.performance_tracker:
            return base_difficulty
        
        # Get user performance metrics
        performance = self.performance_tracker.get_overall_metrics()
        
        # Calculate adaptivity weight (how much we adjust based on performance)
        adaptivity_weight = self.config.get("adaptive_difficulty", {}).get("performance_weight", 0.3)
        
        # Calculate performance factor (-1.0 to 1.0)
        # Positive means user is doing well, negative means struggling
        accuracy = performance.get("accuracy", 0.5)
        accuracy_factor = (accuracy - 0.5) * 2  # Scale to -1.0 to 1.0
        
        # Adjust difficulty
        difficulty_adjustment = accuracy_factor * adaptivity_weight
        
        # Apply adjustment with bounds
        adjusted_difficulty = max(0.1, min(0.9, base_difficulty + difficulty_adjustment))
        
        logger.debug(f"Adaptive difficulty: {base_difficulty} -> {adjusted_difficulty} (adjustment: {difficulty_adjustment})")
        
        return adjusted_difficulty
    
    def blend_difficulties(self, static_difficulty: float, adaptive_difficulty: float) -> float:
        """
        Blend static and adaptive difficulties.
        
        Args:
            static_difficulty: Difficulty based on progression
            adaptive_difficulty: Difficulty based on performance
            
        Returns:
            Blended difficulty value
        """
        # Get adaptive ratio from config
        adaptive_ratio = self.config.get("adaptive_difficulty", {}).get("adaptive_ratio", 0.6)
        
        # Blend the two difficulties
        blended = (static_difficulty * (1 - adaptive_ratio)) + (adaptive_difficulty * adaptive_ratio)
        
        return blended
    
    def select_pattern_for_question(
        self, difficulty: float, 
        excluded_patterns: Optional[List[str]] = None
    ) -> str:
        """
        Select a pattern for a question based on difficulty.
        
        Args:
            difficulty: Target difficulty level (0.0 to 1.0)
            excluded_patterns: Patterns to exclude
            
        Returns:
            Selected pattern name
        """
        # Initialize with safe values
        if excluded_patterns is None:
            excluded_patterns = []
            
        excluded = set(excluded_patterns or [])
        
        # Map difficulty to level
        level_name = self._map_difficulty_to_level(difficulty)
        
        # Get patterns for this level
        patterns = DIFFICULTY_LEVELS.get(level_name, [])
        
        # Filter out excluded patterns
        available_patterns = [p for p in patterns if p not in excluded]
        
        # If we have too few patterns, include some from adjacent levels
        if len(available_patterns) < 3:
            # Get adjacent levels
            levels = list(DIFFICULTY_LEVELS.keys())
            current_idx = levels.index(level_name) if level_name in levels else -1
            
            # Try lower level first if available
            if current_idx > 0:
                lower_level = levels[current_idx - 1]
                lower_patterns = [p for p in DIFFICULTY_LEVELS.get(lower_level, []) 
                                 if p not in excluded]
                available_patterns.extend(lower_patterns)
            
            # Then try higher level if needed
            if len(available_patterns) < 3 and current_idx < len(levels) - 1:
                higher_level = levels[current_idx + 1]
                higher_patterns = [p for p in DIFFICULTY_LEVELS.get(higher_level, []) 
                                  if p not in excluded]
                available_patterns.extend(higher_patterns)
        
        # If no patterns available, use all patterns
        if not available_patterns:
            all_patterns = []
            for patterns_list in DIFFICULTY_LEVELS.values():
                all_patterns.extend(patterns_list)
            available_patterns = [p for p in all_patterns if p not in excluded]
        
        # If still no patterns, use any pattern
        if not available_patterns:
            for category, patterns in CANDLESTICK_PATTERNS.items():
                available_patterns.extend(patterns)
            
            # If there are excluded patterns, filter them
            if excluded:
                available_patterns = [p for p in available_patterns if p not in excluded]
        
        # If we have available patterns, select one
        if available_patterns:
            # If user has performance data, prioritize patterns
            if self.performance_tracker and self.forgetting_curve:
                try:
                    # Get pattern priorities
                    priorities = {}
                    for pattern in available_patterns:
                        # Check if pattern is due for review
                        review_priority = self.forgetting_curve.get_review_priority(pattern)
                        
                        # Check performance on this pattern
                        performance = self.performance_tracker.get_pattern_metrics(pattern)
                        
                        # Higher priority for patterns due for review or with lower accuracy
                        accuracy = performance.get("accuracy", 0.5)
                        
                        # Calculate priority (higher is more likely to be selected)
                        priority = review_priority + (1 - accuracy) * 0.5
                        
                        priorities[pattern] = priority
                    
                    # Select based on priorities (weighted random)
                    if priorities:
                        return self._weighted_random_selection(priorities)
                except Exception as e:
                    logger.error(f"Error selecting pattern based on performance: {e}")
                    # Fall through to random selection
            
            # Fallback to random selection
            return random.choice(available_patterns)
        
        # Absolute fallback
        all_patterns = []
        for category, patterns in CANDLESTICK_PATTERNS.items():
            all_patterns.extend(patterns)
        return random.choice(all_patterns)
    
    def calculate_time_pressure(self, pattern: str, difficulty: float) -> int:
        """
        Calculate time pressure (seconds) for a question.
        
        Args:
            pattern: Pattern name
            difficulty: Question difficulty
            
        Returns:
            Time limit in seconds
        """
        # Base time limits from config
        base_time = self.config.get("time_limits", {}).get("base_seconds", 30)
        min_time = self.config.get("time_limits", {}).get("min_seconds", 15)
        max_time = self.config.get("time_limits", {}).get("max_seconds", 60)
        
        # Get pattern category (single, double, triple, complex)
        category = get_pattern_category(pattern)
        
        # Apply category-based adjustment
        # Complex patterns get more time, single patterns get less
        category_multipliers = {
            "single": 0.8,
            "double": 1.0,
            "triple": 1.2,
            "complex": 1.5,
            # Default
            "unknown": 1.0
        }
        
        category_mult = category_multipliers.get(category, 1.0)
        
        # Apply difficulty adjustment (harder questions get less time)
        difficulty_mult = 1.0 - (difficulty * 0.5)  # 0.5 to 1.0
        
        # Calculate final time limit
        time_limit = base_time * category_mult * difficulty_mult
        
        # Apply bounds
        time_limit = max(min_time, min(max_time, round(time_limit)))
        
        return int(time_limit)
    
    def record_performance(
        self, pattern: str, is_correct: bool, 
        response_time: float, difficulty: float
    ) -> None:
        """
        Record user performance for a pattern.
        
        Args:
            pattern: Pattern name
            is_correct: Whether the answer was correct
            response_time: Response time in seconds
            difficulty: Question difficulty
        """
        if not self.user_id:
            return
        
        if self.performance_tracker:
            # Record the attempt
            self.performance_tracker.record_attempt(
                pattern, is_correct, response_time, difficulty
            )
        
        if self.forgetting_curve and is_correct:
            # Update forgetting curve data
            self.forgetting_curve.update_memory_strength(pattern)
        
        # Instead of creating a task, just ignore the result
        # This is safer than asyncio.create_task which can cause issues
        # if the event loop isn't properly set up
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Schedule the coroutine to run soon but don't wait for it
                loop.create_task(self._save_user_data())
            else:
                # Log that we couldn't save user data
                logger.warning("Event loop not running, couldn't save user data")
        except Exception as e:
            logger.error(f"Error scheduling _save_user_data: {e}")
            # Continue execution regardless of error
    
    def calculate_score(
        self, is_correct: bool, difficulty: float, 
        response_time: float, time_limit: float, streak: int
    ) -> float:
        """
        Calculate score for an answer.
        
        Args:
            is_correct: Whether the answer was correct
            difficulty: Question difficulty
            response_time: Response time in seconds
            time_limit: Time limit in seconds
            streak: Current streak of correct answers
            
        Returns:
            Score value
        """
        if not is_correct:
            return 0.0
        
        # Base score based on difficulty (1 to 5 points)
        base_score = 1 + (difficulty * 4)
        
        # Time bonus (up to 50% of base score)
        time_ratio = min(1.0, response_time / time_limit)
        time_bonus = (1 - time_ratio) * (base_score * 0.5)
        
        # Streak bonus (up to 30% of base score)
        streak_bonus = min(5, streak) / 5 * (base_score * 0.3)
        
        # Calculate total score
        total_score = base_score + time_bonus + streak_bonus
        
        return round(total_score, 1)
    
    def _map_difficulty_to_level(self, difficulty: float) -> str:
        """
        Map numerical difficulty to a difficulty level name.
        
        Args:
            difficulty: Numerical difficulty (0.0 to 1.0)
            
        Returns:
            Difficulty level name
        """
        if difficulty < 0.2:
            return "beginner"
        elif difficulty < 0.4:
            return "easy"
        elif difficulty < 0.6:
            return "medium"
        elif difficulty < 0.8:
            return "hard"
        else:
            return "expert"
    
    def _weighted_random_selection(self, weights: Dict[str, float]) -> str:
        """
        Select an item based on weights.
        
        Args:
            weights: Dictionary mapping items to weights
            
        Returns:
            Selected item
        """
        items = list(weights.keys())
        weights_list = [max(0.1, weights[item]) for item in items]  # Ensure all weights are positive
        
        # Select based on weights
        return random.choices(items, weights=weights_list, k=1)[0]

@dataclass
class PatternPerformance:
    """Class to track performance metrics for a specific pattern."""
    pattern: str
    attempts: int = 0
    correct: int = 0
    total_response_time: float = 0.0
    last_attempt_timestamp: int = 0
    skill_level: float = 0.5  # 0.0 to 1.0
    difficulty_history: List[float] = field(default_factory=list)
    
    @property
    def accuracy(self) -> float:
        """Calculate accuracy rate."""
        return self.correct / max(1, self.attempts)
    
    @property
    def avg_response_time(self) -> float:
        """Calculate average response time."""
        return self.total_response_time / max(1, self.attempts)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "pattern": self.pattern,
            "attempts": self.attempts,
            "correct": self.correct,
            "total_response_time": self.total_response_time,
            "last_attempt_timestamp": self.last_attempt_timestamp,
            "skill_level": self.skill_level,
            "difficulty_history": self.difficulty_history
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PatternPerformance':
        """Create from dictionary."""
        return cls(
            pattern=data["pattern"],
            attempts=data["attempts"],
            correct=data["correct"],
            total_response_time=data["total_response_time"],
            last_attempt_timestamp=data["last_attempt_timestamp"],
            skill_level=data["skill_level"],
            difficulty_history=data.get("difficulty_history", [])
        )

class UserPerformanceTracker:
    """
    Tracks user performance metrics for different patterns.
    
    This class maintains performance data for each pattern a user has
    attempted, including accuracy, response times, and skill levels.
    """
    
    def __init__(self, user_id: Optional[str] = None):
        """
        Initialize user performance tracker.
        
        Args:
            user_id: Optional user ID
        """
        self.user_id = user_id
        self.pattern_metrics: Dict[str, PatternPerformance] = {}
        self.category_metrics: Dict[str, Dict[str, float]] = {}
        self.overall_attempts = 0
        self.overall_correct = 0
        self.learning_rate = 0.1  # How quickly skill improves with correct answers
        self.forgetting_rate = 0.05  # How quickly skill decreases with wrong answers
    
    def record_attempt(
        self, pattern: str, is_correct: bool, 
        response_time: float, difficulty: float
    ) -> None:
        """
        Record an attempt at identifying a pattern.
        
        Args:
            pattern: Pattern name
            is_correct: Whether the answer was correct
            response_time: Response time in seconds
            difficulty: Question difficulty
        """
        # Get or create pattern metrics
        if pattern not in self.pattern_metrics:
            self.pattern_metrics[pattern] = PatternPerformance(pattern)
        
        metrics = self.pattern_metrics[pattern]
        
        # Update metrics
        metrics.attempts += 1
        if is_correct:
            metrics.correct += 1
        metrics.total_response_time += response_time
        metrics.last_attempt_timestamp = int(time.time())
        metrics.difficulty_history.append(difficulty)
        
        # Limit history length
        if len(metrics.difficulty_history) > 10:
            metrics.difficulty_history = metrics.difficulty_history[-10:]
        
        # Update skill level
        self._update_skill_level(pattern, is_correct, difficulty)
        
        # Update category metrics
        category = get_pattern_category(pattern)
        self._update_category_metrics(category, is_correct, response_time)
        
        # Update overall metrics
        self.overall_attempts += 1
        if is_correct:
            self.overall_correct += 1
    
    def _update_skill_level(self, pattern: str, is_correct: bool, difficulty: float) -> None:
        """
        Update skill level for a pattern.
        
        Args:
            pattern: Pattern name
            is_correct: Whether the answer was correct
            difficulty: Question difficulty
        """
        metrics = self.pattern_metrics[pattern]
        
        # Calculate skill change based on correctness and difficulty
        if is_correct:
            # Correct answers increase skill more if the difficulty was high
            skill_change = self.learning_rate * difficulty
        else:
            # Incorrect answers decrease skill more if the difficulty was low
            skill_change = -self.forgetting_rate * (1 - difficulty)
        
        # Update skill level with bounds
        metrics.skill_level = max(0.0, min(1.0, metrics.skill_level + skill_change))
    
    def _update_category_metrics(self, category: str, is_correct: bool, response_time: float) -> None:
        """
        Update metrics for a pattern category.
        
        Args:
            category: Pattern category
            is_correct: Whether the answer was correct
            response_time: Response time in seconds
        """
        if category not in self.category_metrics:
            self.category_metrics[category] = {
                "attempts": 0,
                "correct": 0,
                "total_response_time": 0.0
            }
        
        metrics = self.category_metrics[category]
        metrics["attempts"] += 1
        if is_correct:
            metrics["correct"] += 1
        metrics["total_response_time"] += response_time
    
    def get_pattern_metrics(self, pattern: str) -> Dict[str, float]:
        """
        Get metrics for a specific pattern.
        
        Args:
            pattern: Pattern name
            
        Returns:
            Dictionary with metrics
        """
        if pattern not in self.pattern_metrics:
            # Return default metrics for unknown patterns
            return {
                "accuracy": 0.5,
                "skill_level": 0.5,
                "avg_response_time": 30.0,
                "attempts": 0
            }
        
        metrics = self.pattern_metrics[pattern]
        
        return {
            "accuracy": metrics.accuracy,
            "skill_level": metrics.skill_level,
            "avg_response_time": metrics.avg_response_time,
            "attempts": metrics.attempts,
            "last_attempt": metrics.last_attempt_timestamp
        }
    
    def get_category_metrics(self, category: str) -> Dict[str, float]:
        """
        Get metrics for a pattern category.
        
        Args:
            category: Pattern category
            
        Returns:
            Dictionary with metrics
        """
        if category not in self.category_metrics:
            return {
                "accuracy": 0.5,
                "avg_response_time": 30.0,
                "attempts": 0
            }
        
        metrics = self.category_metrics[category]
        accuracy = metrics["correct"] / max(1, metrics["attempts"])
        avg_time = metrics["total_response_time"] / max(1, metrics["attempts"])
        
        return {
            "accuracy": accuracy,
            "avg_response_time": avg_time,
            "attempts": metrics["attempts"]
        }
    
    def get_overall_metrics(self) -> Dict[str, float]:
        """
        Get overall performance metrics.
        
        Returns:
            Dictionary with metrics
        """
        accuracy = self.overall_correct / max(1, self.overall_attempts)
        
        # Calculate average response time across all patterns
        total_time = sum(m.total_response_time for m in self.pattern_metrics.values())
        total_attempts = sum(m.attempts for m in self.pattern_metrics.values())
        avg_time = total_time / max(1, total_attempts)
        
        # Calculate average skill level
        avg_skill = sum(m.skill_level for m in self.pattern_metrics.values()) / max(1, len(self.pattern_metrics))
        
        return {
            "accuracy": accuracy,
            "avg_response_time": avg_time,
            "total_attempts": self.overall_attempts,
            "avg_skill_level": avg_skill
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "user_id": self.user_id,
            "overall_attempts": self.overall_attempts,
            "overall_correct": self.overall_correct,
            "learning_rate": self.learning_rate,
            "forgetting_rate": self.forgetting_rate,
            "pattern_metrics": {
                p: m.to_dict() for p, m in self.pattern_metrics.items()
            },
            "category_metrics": self.category_metrics
        }
    
    def update_from_data(self, data: Dict[str, Any]) -> None:
        """Load from dictionary."""
        if not data:
            return
            
        self.overall_attempts = data.get("overall_attempts", 0)
        self.overall_correct = data.get("overall_correct", 0)
        self.learning_rate = data.get("learning_rate", 0.1)
        self.forgetting_rate = data.get("forgetting_rate", 0.05)
        
        # Load pattern metrics
        pattern_data = data.get("pattern_metrics", {})
        if pattern_data:
            self.pattern_metrics = {
                p: PatternPerformance.from_dict(m) for p, m in pattern_data.items()
            }
        
        # Load category metrics
        self.category_metrics = data.get("category_metrics", {})
    
    # Original method was named load_from_dict, but update_from_data is actually used
    def load_from_dict(self, data: Dict[str, Any]) -> None:
        """Alias for update_from_data for backward compatibility."""
        self.update_from_data(data)

class ForgettingCurveManager:
    """
    Implements forgetting curve algorithm for optimal pattern review.
    
    This class tracks memory strength for patterns and calculates
    optimal review intervals based on the forgetting curve.
    """
    
    def __init__(self):
        """Initialize the forgetting curve manager."""
        self.pattern_memory: Dict[str, Dict[str, Any]] = {}
        self.base_interval = 1.0  # Base interval in days
        self.default_strength = 0.5  # Default memory strength
    
    def update_memory_strength(self, pattern: str) -> None:
        """
        Update memory strength for a pattern after a successful recall.
        
        Args:
            pattern: Pattern name
        """
        # Get or initialize memory data
        if pattern not in self.pattern_memory:
            self.pattern_memory[pattern] = {
                "strength": self.default_strength,
                "last_review": int(time.time()),
                "review_count": 0,
                "next_review": int(time.time()) + int(self.base_interval * 86400)
            }
        
        memory = self.pattern_memory[pattern]
        
        # Increment review count
        memory["review_count"] += 1
        
        # Calculate new strength
        # Each successful review increases strength, with diminishing returns
        current_strength = memory["strength"]
        strength_increase = (1 - current_strength) * 0.2  # 20% of remaining headroom
        new_strength = current_strength + strength_increase
        
        # Update memory data
        memory["strength"] = new_strength
        memory["last_review"] = int(time.time())
        
        # Calculate next review time based on new strength
        days_until_review = self._calculate_review_interval(new_strength, memory["review_count"])
        memory["next_review"] = int(time.time()) + int(days_until_review * 86400)
    
    def get_review_priority(self, pattern: str) -> float:
        """
        Get review priority for a pattern.
        
        Higher values indicate patterns that should be reviewed soon.
        
        Args:
            pattern: Pattern name
            
        Returns:
            Priority value (higher means higher priority)
        """
        # If pattern has never been seen, high priority
        if pattern not in self.pattern_memory:
            return 1.0
        
        memory = self.pattern_memory[pattern]
        current_time = int(time.time())
        
        # If it's past review time, high priority
        if current_time >= memory["next_review"]:
            # Priority increases the longer it's overdue
            days_overdue = (current_time - memory["next_review"]) / 86400
            return min(1.0, 0.8 + (days_overdue * 0.1))
        
        # Calculate how far we are into the review interval
        interval_progress = 1.0 - ((memory["next_review"] - current_time) / 
                                 (memory["next_review"] - memory["last_review"]))
        
        # Priority increases as we approach review time
        return max(0.1, interval_progress)
    
    def _calculate_review_interval(self, strength: float, review_count: int) -> float:
        """
        Calculate optimal review interval based on memory strength.
        
        Args:
            strength: Memory strength (0.0 to 1.0)
            review_count: Number of times reviewed
            
        Returns:
            Interval in days
        """
        # Base interval (starting small and increasing with strength)
        interval = self.base_interval * (1 + (strength * 5))
        
        # Apply review count multiplier (intervals grow with more reviews)
        count_factor = min(3.0, 1.0 + (review_count * 0.2))
        interval *= count_factor
        
        return interval
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "pattern_memory": self.pattern_memory,
            "base_interval": self.base_interval,
            "default_strength": self.default_strength
        }
    
    def update_from_data(self, data: Dict[str, Any]) -> None:
        """Load data from dictionary."""
        if not data:
            return
            
        self.pattern_memory = data.get("pattern_memory", {})
        self.base_interval = data.get("base_interval", 1.0)
        self.default_strength = data.get("default_strength", 0.5)
    
    # Original method was named load_from_dict, but update_from_data is actually used
    def load_from_dict(self, data: Dict[str, Any]) -> None:
        """Alias for update_from_data for backward compatibility."""
        self.update_from_data(data)

class AdaptiveDifficultyEngine(BaseAdaptiveDifficultyEngine):
    """
    Combines user performance tracking, forgetting curve management,
    and difficulty adjustment into a unified engine.
    
    This class provides a comprehensive interface for all adaptive
    difficulty functionality.
    """
    
    def __init__(self, user_id: str):
        """
        Initialize the adaptive difficulty engine.
        
        Args:
            user_id: User ID
        """
        super().__init__(user_id)
        self.difficulty_manager = DifficultyManager(user_id)
    
    async def load_user_data(self) -> None:
        """Load user data from cache."""
        await self.difficulty_manager.load_user_data()
    
    async def save_user_data(self) -> None:
        """Save user data to cache."""
        await self.difficulty_manager._save_user_data()
    
    def get_next_patterns(self, count: int, excluded_patterns: Optional[List[str]] = None) -> List[str]:
        """
        Get the next patterns for assessment based on adaptive difficulty.
        
        Args:
            count: Number of patterns to get
            excluded_patterns: Patterns to exclude
            
        Returns:
            List of pattern names
        """
        patterns = []
        excluded = set(excluded_patterns or [])
        
        # Get overall metrics to determine base difficulty
        metrics = self.performance_tracker.get_overall_metrics()
        base_difficulty = 0.5  # Default starting point
        
        # Adjust difficulty based on accuracy
        accuracy = metrics.get("accuracy", 0.5)
        difficulty_adjustment = (accuracy - 0.5) * 0.4  # -0.2 to +0.2
        base_difficulty += difficulty_adjustment
        
        # Ensure diverse pattern categories
        categories = ["single", "double", "triple", "complex"]
        category_counts = {category: 0 for category in categories}
        target_counts = self._calculate_category_distribution(count, categories)
        
        # Select patterns for each category
        for _ in range(count):
            # Find the category that needs more patterns
            target_category = None
            for category in categories:
                if category_counts[category] < target_counts[category]:
                    target_category = category
                    break
            
            # If all categories are full, just pick any
            if target_category is None:
                target_category = random.choice(categories)
            
            # Get patterns for this category
            category_patterns = [
                p for p in CANDLESTICK_PATTERNS.get(target_category, [])
                if p not in excluded and p not in patterns
            ]
            
            # If no patterns available in this category, try any category
            if not category_patterns:
                for category in categories:
                    category_patterns = [
                        p for p in CANDLESTICK_PATTERNS.get(category, [])
                        if p not in excluded and p not in patterns
                    ]
                    if category_patterns:
                        break
            
            # If still no patterns, just use any available pattern
            if not category_patterns:
                all_patterns = []
                for cat_patterns in CANDLESTICK_PATTERNS.values():
                    all_patterns.extend(cat_patterns)
                category_patterns = [p for p in all_patterns if p not in excluded and p not in patterns]
            
            # Select a pattern
            if category_patterns:
                pattern = self.difficulty_manager.select_pattern_for_question(
                    base_difficulty, excluded_patterns=excluded
                )
                if not pattern and category_patterns:
                    pattern = random.choice(category_patterns)
                
                if pattern:
                    patterns.append(pattern)
                    excluded.add(pattern)
                    category_counts[target_category] += 1
        
        return patterns
    
    def get_assessment_config(self, total_questions: int) -> Dict[str, Any]:
        """
        Generate a complete assessment configuration based on adaptive difficulty.
        
        Args:
            total_questions: Total number of questions
            
        Returns:
            Assessment configuration
        """
        # Get patterns for the assessment
        patterns = self.get_next_patterns(total_questions)
        
        # Calculate difficulty for each pattern
        questions = []
        
        for i, pattern in enumerate(patterns):
            # Calculate question difficulty
            static_difficulty = self.difficulty_manager.calculate_static_difficulty(i + 1, total_questions)
            adaptive_difficulty = self.difficulty_manager.calculate_adaptive_difficulty(static_difficulty)
            difficulty = self.difficulty_manager.blend_difficulties(static_difficulty, adaptive_difficulty)
            
            # Calculate time pressure
            time_limit = self.difficulty_manager.calculate_time_pressure(pattern, difficulty)
            
            questions.append({
                "pattern": pattern,
                "difficulty": difficulty,
                "time_limit_seconds": time_limit
            })
        
        return {
            "total_questions": total_questions,
            "questions": questions,
            "user_id": self.user_id
        }
    
    def record_performance(
        self, pattern: str, is_correct: bool, 
        response_time: float, difficulty: float
    ) -> None:
        """
        Record performance for a pattern.
        
        Args:
            pattern: Pattern name
            is_correct: Whether the answer was correct
            response_time: Response time in seconds
            difficulty: Question difficulty
        """
        self.difficulty_manager.record_performance(
            pattern, is_correct, response_time, difficulty
        )
    
    def _calculate_category_distribution(
        self, total_count: int, categories: List[str]
    ) -> Dict[str, int]:
        """
        Calculate how many patterns to include from each category.
        
        Args:
            total_count: Total number of patterns
            categories: List of categories
            
        Returns:
            Dictionary mapping categories to counts
        """
        # Default distribution (can be adjusted based on user performance)
        distribution = {
            "single": 0.3,  # 30% single patterns
            "double": 0.4,  # 40% double patterns
            "triple": 0.2,  # 20% triple patterns
            "complex": 0.1  # 10% complex patterns
        }
        
        # Calculate counts for each category
        counts = {}
        remaining = total_count
        
        for i, category in enumerate(categories):
            # For the last category, assign all remaining slots
            if i == len(categories) - 1:
                counts[category] = remaining
            else:
                # Calculate count and round to nearest integer
                weight = distribution.get(category, 1.0 / len(categories))
                count = round(total_count * weight)
                counts[category] = count
                remaining -= count
        
        return counts 