"""
Adaptive Difficulty Logic for candlestick pattern assessments.

This module provides:
1. Sophisticated user profiling with pattern-specific performance tracking
2. Learning rate calculation and forgetting curve implementation
3. Dynamic difficulty adjustment based on user performance
4. Reinforcement learning and bandit algorithms for optimizing difficulty selection
"""

import math
import time
import random
import numpy as np
import weakref
import threading
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Set, Union, Callable
from datetime import datetime, timedelta
from enum import Enum
from functools import lru_cache
from collections import defaultdict, deque

# Import from base assessment architecture
from backend.assessments.base.models import QuestionDifficulty
from backend.assessments.base.services import PerformanceAnalyzer

# Other common imports
from backend.common.logger import app_logger, log_execution_time
from backend.common.cache import cache, async_cached
# Import common performance modules
from backend.common.performance.tracker import SkillLevel
from backend.common.performance.forgetting import ForgettingCurveModel
from backend.common.performance.difficulty import DifficultyLevel, DifficultyAdjustment

from backend.assessments.candlestick_patterns.candlestick_config import (
    CANDLESTICK_PATTERNS, DIFFICULTY_LEVELS, ASSESSMENT_CONFIG
)

# Module logger
logger = app_logger.getChild("adaptive_difficulty")

# Constants for efficiency
MAX_HISTORY_ENTRIES = 50  # Maximum history entries to keep per pattern
DEFAULT_LEARNING_RATE = 0.05  # Default learning rate
DEFAULT_MEMORY_STRENGTH = 0.3  # Default memory strength
DEFAULT_FORGETTING_RATE = 0.1  # Default forgetting rate
MIN_LEARNING_RATE = 0.01  # Minimum learning rate
MAX_LEARNING_RATE = 0.2  # Maximum learning rate

class PerformanceRating(Enum):
    """Enum for pattern performance ratings."""
    EXCELLENT = 4
    GOOD = 3
    FAIR = 2
    POOR = 1
    UNKNOWN = 0

class UserPerformanceTracker:
    """
    Tracks user performance on different candlestick patterns over time.
    
    This class maintains:
    1. Pattern-specific performance metrics
    2. Learning rate for each pattern
    3. Success rates under different time pressures
    
    Features:
    - Memory-efficient history tracking with automatic pruning
    - Thread-safe operations
    - Optimized metric calculations with caching
    """
    
    # Maximum history entries to keep per pattern
    MAX_HISTORY_ENTRIES = MAX_HISTORY_ENTRIES
    
    def __init__(self, user_id: str):
        """
        Initialize the performance tracker.
        
        Args:
            user_id: Unique identifier for the user
        """
        self.user_id = user_id
        self.pattern_metrics: Dict[str, Dict[str, Any]] = {}
        self.overall_metrics: Dict[str, Any] = {
            "total_questions": 0,
            "correct_answers": 0,
            "avg_response_time": 0,
            "last_activity": time.time(),
            "skill_level": 0.0,  # 0.0 to 1.0
            "learning_rate": DEFAULT_LEARNING_RATE,  # Default learning rate
            "streak": 0,
            "max_streak": 0,
            "xp": 0,  # Experience points for gamification
            "level": 1,  # User level
            "consecutive_correct": 0,  # Consecutive correct answers
            "consecutive_incorrect": 0  # Consecutive incorrect answers
        }
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Performance optimization
        self._pattern_cache = {}  # Cache for pattern-related calculations
        self._update_needed = set()  # Track patterns needing updates
        
        # Initialize pattern metrics for all patterns
        self._initialize_pattern_metrics()
        
        # Initialize forgetting curve model
        self.forgetting_model = ForgettingCurveModel()
        
        logger.info(f"Initialized performance tracker for user {user_id}")
    
    def _initialize_pattern_metrics(self):
        """Initialize metrics for all available candlestick patterns."""
        with self._lock:
            for category, patterns in CANDLESTICK_PATTERNS.items():
                for pattern in patterns:
                    self.pattern_metrics[pattern] = {
                        "attempts": 0,
                        "correct": 0,
                        "last_seen": 0,  # Timestamp
                        "avg_response_time": 0,
                        "fast_responses": 0,  # Responses under time pressure
                        "fast_correct": 0,    # Correct responses under time pressure
                        "confidence": 0.0,    # 0.0 to 1.0
                        "mastery": 0.0,       # 0.0 to 1.0
                        "due_factor": 1.0,    # Spaced repetition factor
                        "history": [],        # List of attempt timestamps and results
                        "learning_rate": DEFAULT_LEARNING_RATE,  # Pattern-specific learning rate
                        "forgetting_rate": DEFAULT_FORGETTING_RATE,  # Rate at which knowledge is forgotten
                        "memory_strength": DEFAULT_MEMORY_STRENGTH,  # Current memory strength (0.0-1.0)
                        "consecutive_correct": 0,  # Pattern-specific consecutive correct
                        "consecutive_incorrect": 0  # Pattern-specific consecutive incorrect
                    }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the tracker state to a dictionary for serialization."""
        with self._lock:
            # Update any patterns that need it before serialization
            self._update_pending_metrics()
            
            return {
                "user_id": self.user_id,
                "pattern_metrics": self.pattern_metrics,
                "overall_metrics": self.overall_metrics
            }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserPerformanceTracker':
        """
        Create a tracker from a dictionary.
        
        Args:
            data: Dictionary representation of tracker state
            
        Returns:
            Initialized UserPerformanceTracker instance
        """
        tracker = cls(data["user_id"])
        
        with tracker._lock:
            tracker.pattern_metrics = data["pattern_metrics"]
            tracker.overall_metrics = data["overall_metrics"]
            
            # If we're loading older data that might be missing some fields, add them
            for pattern, metrics in tracker.pattern_metrics.items():
                # Ensure all required fields exist
                if "learning_rate" not in metrics:
                    metrics["learning_rate"] = DEFAULT_LEARNING_RATE
                if "forgetting_rate" not in metrics:
                    metrics["forgetting_rate"] = DEFAULT_FORGETTING_RATE
                if "memory_strength" not in metrics:
                    metrics["memory_strength"] = DEFAULT_MEMORY_STRENGTH
                if "consecutive_correct" not in metrics:
                    metrics["consecutive_correct"] = 0
                if "consecutive_incorrect" not in metrics:
                    metrics["consecutive_incorrect"] = 0
        
        return tracker
    
    def _update_pending_metrics(self):
        """Update all patterns that have pending metric updates."""
        with self._lock:
            for pattern in self._update_needed:
                if pattern in self.pattern_metrics:
                    self._update_pattern_metrics(pattern)
                    self._update_learning_rate(pattern)
            
            # Clear the update set after processing
            self._update_needed.clear()
            
            # Update overall skill level if needed
            self._update_skill_level()
    
    @log_execution_time()
    def record_attempt(self, 
                      pattern: str, 
                      is_correct: bool, 
                      response_time: float,
                      timestamp: Optional[float] = None) -> Dict[str, Any]:
        """
        Record an attempt at identifying a pattern.
        
        Args:
            pattern: The candlestick pattern name
            is_correct: Whether the identification was correct
            response_time: Time taken to respond in seconds
            timestamp: Optional timestamp of the attempt
            
        Returns:
            Dictionary with performance updates and rewards
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Track attempt results for returning rewards
        attempt_results = {
            "xp_earned": 0,
            "streak_bonus": 0,
            "time_bonus": 0,
            "difficulty_adjustment": 0,
            "mastery_increase": 0
        }
        
        with self._lock:
            # Update overall metrics
            self._update_overall_metrics(is_correct, response_time, timestamp)
            
            # Check if this is a known pattern
            if pattern not in self.pattern_metrics:
                logger.warning(f"Attempt to record unknown pattern: {pattern}")
                # Initialize the pattern if it doesn't exist
                self.pattern_metrics[pattern] = {
                    "attempts": 0,
                    "correct": 0,
                    "last_seen": 0,
                    "avg_response_time": 0,
                    "fast_responses": 0,
                    "fast_correct": 0,
                    "confidence": 0.0,
                    "mastery": 0.0,
                    "due_factor": 1.0,
                    "history": [],
                    "learning_rate": DEFAULT_LEARNING_RATE,
                    "forgetting_rate": DEFAULT_FORGETTING_RATE,
                    "memory_strength": DEFAULT_MEMORY_STRENGTH,
                    "consecutive_correct": 0,
                    "consecutive_incorrect": 0
                }
            
            # Update pattern-specific metrics
            self._update_pattern_attempt(pattern, is_correct, response_time, timestamp)
            
            # Calculate rewards based on performance
            attempt_results = self._calculate_rewards(pattern, is_correct, response_time)
            
            # Prune history if needed
            self._prune_pattern_history(pattern)
            
            # Mark pattern for metrics update
            self._update_needed.add(pattern)
            
            # Immediately update pattern metrics if there are fewer than 5 patterns to update
            if len(self._update_needed) < 5:
                self._update_pending_metrics()
            else:
                # Otherwise, just update this pattern's metrics
                self._update_pattern_metrics(pattern)
                self._update_learning_rate(pattern)
        
        return attempt_results
        
    def _update_overall_metrics(self, is_correct: bool, response_time: float, timestamp: float):
        """Update overall user metrics based on the current attempt."""
        self.overall_metrics["total_questions"] += 1
        
        # Handle correct answers
        if is_correct:
            self.overall_metrics["correct_answers"] += 1
            self.overall_metrics["streak"] += 1
            self.overall_metrics["max_streak"] = max(
                self.overall_metrics["max_streak"], 
                self.overall_metrics["streak"]
            )
            self.overall_metrics["consecutive_correct"] += 1
            self.overall_metrics["consecutive_incorrect"] = 0
        else:
            # Reset streak for incorrect answers
            self.overall_metrics["streak"] = 0
            self.overall_metrics["consecutive_correct"] = 0
            self.overall_metrics["consecutive_incorrect"] += 1
        
        # Update running average response time with more stable formula
        total_q = self.overall_metrics["total_questions"]
        if total_q > 1:
            # Use weighted average for stability
            prev_avg = self.overall_metrics["avg_response_time"]
            alpha = 1.0 / min(100, total_q)  # Smaller alpha for larger sample size
            self.overall_metrics["avg_response_time"] = (
                (1 - alpha) * prev_avg + alpha * response_time
            )
        else:
            self.overall_metrics["avg_response_time"] = response_time
        
        self.overall_metrics["last_activity"] = timestamp
        
    def _update_pattern_attempt(self, pattern: str, is_correct: bool, response_time: float, timestamp: float):
        """Update pattern-specific metrics for a single attempt."""
        pattern_data = self.pattern_metrics[pattern]
        pattern_data["attempts"] += 1
        
        # Update correct count
        if is_correct:
            pattern_data["correct"] += 1
            pattern_data["consecutive_correct"] += 1
            pattern_data["consecutive_incorrect"] = 0
        else:
            pattern_data["consecutive_correct"] = 0
            pattern_data["consecutive_incorrect"] += 1
        
        # Update pattern-specific average response time with more stable formula
        attempts = pattern_data["attempts"]
        if attempts > 1:
            # Use weighted average for stability
            prev_avg = pattern_data["avg_response_time"]
            alpha = 1.0 / min(20, attempts)  # Smaller alpha for larger sample size
            pattern_data["avg_response_time"] = (
                (1 - alpha) * prev_avg + alpha * response_time
            )
        else:
            pattern_data["avg_response_time"] = response_time
            
        # Check if this was a fast response (below average)
        avg_time = self.overall_metrics["avg_response_time"]
        is_fast = avg_time > 0 and response_time < avg_time * 0.9
        if is_fast:
            pattern_data["fast_responses"] += 1
            if is_correct:
                pattern_data["fast_correct"] += 1
            
        # Update last seen timestamp
        pattern_data["last_seen"] = timestamp
            
        # Record in history
        attempt_data = {
            "timestamp": timestamp,
            "correct": is_correct,
            "response_time": response_time,
            "fast": is_fast
        }
        pattern_data["history"].append(attempt_data)
        
        # Invalidate cached calculations
        if pattern in self._pattern_cache:
            del self._pattern_cache[pattern]
    
    def _calculate_rewards(self, pattern: str, is_correct: bool, response_time: float) -> Dict[str, Any]:
        """Calculate rewards for this attempt."""
        rewards = {
            "xp_earned": 0,
            "streak_bonus": 0,
            "time_bonus": 0,
            "difficulty_adjustment": 0,
            "mastery_increase": 0
        }
        
        if not is_correct:
            return rewards
            
        # Base XP for correct answer
        base_xp = 10
        
        # Streak bonus
        streak = self.overall_metrics["streak"]
        streak_bonus = min(streak * 2, 20)  # Cap at +20 XP
        rewards["streak_bonus"] = streak_bonus
        
        # Time bonus for fast answers
        avg_time = self.overall_metrics["avg_response_time"] or response_time
        if response_time < avg_time * 0.8:  # 20% faster than average
            time_bonus = int(10 * (avg_time / response_time - 0.8))
            time_bonus = min(time_bonus, 15)  # Cap at +15 XP
            rewards["time_bonus"] = time_bonus
        else:
            time_bonus = 0
        
        # Mastery increase calculation
        if pattern in self.pattern_metrics:
            old_mastery = self.pattern_metrics[pattern]["mastery"]
            # We'll update the metrics later, but compute the estimated mastery increase
            consistency_factor = min(1.0, self.pattern_metrics[pattern]["attempts"] / 10.0)
            estimated_mastery_increase = 0.05 * (1.0 + consistency_factor)
            rewards["mastery_increase"] = estimated_mastery_increase
        
        # Total XP earned
        xp_earned = base_xp + streak_bonus + time_bonus
        self.overall_metrics["xp"] += xp_earned
        rewards["xp_earned"] = xp_earned
        
        # Check for level up
        if self._check_level_up():
            rewards["level_up"] = True
            rewards["new_level"] = self.overall_metrics["level"]
        
        return rewards
    
    def _prune_pattern_history(self, pattern: str):
        """Limit the history size to prevent memory issues."""
        history = self.pattern_metrics[pattern]["history"]
        if len(history) > self.MAX_HISTORY_ENTRIES:
            # Keep the most recent entries and some distributed older entries
            keep_recent = int(self.MAX_HISTORY_ENTRIES * 0.8)  # Keep 80% recent
            keep_distributed = self.MAX_HISTORY_ENTRIES - keep_recent  # Keep 20% distributed
            
            # Keep all recent entries
            recent_entries = history[-keep_recent:]
            
            # Keep some distributed older entries
            older_entries = history[:-keep_recent]
            if older_entries and keep_distributed > 0:
                indices = np.linspace(0, len(older_entries)-1, keep_distributed, dtype=int)
                distributed_entries = [older_entries[i] for i in indices]
            else:
                distributed_entries = []
            
            # Combine distributed older entries with recent entries
            self.pattern_metrics[pattern]["history"] = distributed_entries + recent_entries
    
    def _check_level_up(self) -> bool:
        """
        Check if user has reached enough XP to level up.
        
        Returns:
            True if level up occurred, False otherwise
        """
        current_level = self.overall_metrics["level"]
        current_xp = self.overall_metrics["xp"]
        
        # Simple level formula: level N requires N^2 * 100 XP
        required_xp = current_level * current_level * 100
        
        if current_xp >= required_xp:
            self.overall_metrics["level"] += 1
            logger.info(f"User {self.user_id} leveled up to {self.overall_metrics['level']}!")
            return True
        
        return False
    
    def _update_pattern_metrics(self, pattern: str) -> None:
        """
        Update derived metrics for a specific pattern.

        Args:
            pattern: The candlestick pattern name
        """
        # Skip if pattern doesn't exist
        if pattern not in self.pattern_metrics:
            return

        # Check if we have cached results
        cache_key = f"{pattern}:{self.pattern_metrics[pattern]['attempts']}:{self.pattern_metrics[pattern]['correct']}"
        if cache_key in self._pattern_cache:
            cached_metrics = self._pattern_cache[cache_key]
            for key, value in cached_metrics.items():
                self.pattern_metrics[pattern][key] = value
            return

        data = self.pattern_metrics[pattern]

        # Calculate success rate (basic confidence)
        if data["attempts"] > 0:
            success_rate = data["correct"] / data["attempts"]
        else:
            success_rate = 0.0

        # Calculate time pressure success rate
        time_pressure_rate = 0.0
        if data["fast_responses"] > 0:
            time_pressure_rate = data["fast_correct"] / data["fast_responses"]

        # Calculate recency factor (how recently the pattern was seen)
        if data["last_seen"] > 0:
            elapsed_since_last = time.time() - data["last_seen"]
            # Create a smooth decay curve for recency
            recency_factor = 1.0 / (1.0 + elapsed_since_last / (7 * 24 * 3600))  # 7-day half-life
        else:
            recency_factor = 0.0

        # Calculate memory retention using forgetting curve
        # This estimates how well the pattern is currently retained in memory
        retention = 0.0  # Initialize retention
        if data["last_seen"] > 0:
            elapsed_hours = (time.time() - data["last_seen"]) / 3600  # Convert to hours
            try:
                retention = self.forgetting_model.calculate_retention(
                    data["memory_strength"], data["forgetting_rate"], elapsed_hours
                )
            except Exception as e:
                logger.error(f"Error calculating retention for pattern {pattern}: {e}")
                retention = 0.5  # Default to mid-range if calculation fails
        # else: retention remains 0.0 as initialized

        # Combine factors with balanced weighting
        confidence = (
            success_rate * 0.6 +            # Success rate (60% weight)
            time_pressure_rate * 0.2 +      # Performance under time pressure (20% weight)
            retention * 0.2                 # Memory retention (20% weight)
        )

        # Ensure confidence is in 0-1 range
        confidence = max(0.0, min(1.0, confidence))

        # Calculate mastery level (combines confidence with consistency and recency)
        consistency_factor = min(1.0, data["attempts"] / 10.0)  # Saturates at 10 attempts

        # More nuanced mastery calculation
        mastery = (
            confidence * 0.5 +              # Base performance (50% weight)
            consistency_factor * 0.2 +      # Experience with pattern (20% weight)
            recency_factor * 0.1 +          # Recency of exposure (10% weight)
            data["memory_strength"] * 0.2   # Memory strength (20% weight)
        )

        # Ensure mastery is in 0-1 range
        mastery = max(0.0, min(1.0, mastery))

        # Update pattern metrics
        data["confidence"] = confidence
        data["mastery"] = mastery

        # Cache the computed metrics to avoid redundant calculations
        self._pattern_cache[cache_key] = {
            "confidence": confidence,
            "mastery": mastery
        }
    
    def _update_learning_rate(self, pattern: str) -> None:
        """
        Update the learning rate for a specific pattern based on performance.

        Args:
            pattern: The candlestick pattern name
        """
        # Skip if pattern doesn't exist
        if pattern not in self.pattern_metrics:
            return

        data = self.pattern_metrics[pattern]

        # Need at least 3 attempts to calculate meaningful learning rate
        if data["attempts"] < 3:
            return

        # Get the last 5 attempts (or fewer if not available)
        history = data["history"][-5:]

        if len(history) < 2:
            return

        try:
            # Calculate accuracy improvement rate
            # We look at how the accuracy has changed over recent attempts
            correct_count = sum(1 for attempt in history if attempt["correct"])
            recent_accuracy = correct_count / len(history) if history else 0

            # Get previous accuracy (before these attempts)
            older_history = data["history"][:-len(history)] if len(data["history"]) > len(history) else []
            if older_history:
                older_correct = sum(1 for attempt in older_history if attempt["correct"])
                older_accuracy = older_correct / len(older_history) if older_history else 0
                accuracy_change = recent_accuracy - older_accuracy
            else:
                accuracy_change = 0

            # Calculate response time improvement
            avg_recent_time = 0
            if history:
                recent_times = [attempt["response_time"] for attempt in history]
                avg_recent_time = sum(recent_times) / len(recent_times) if recent_times else 0

            time_change = 0
            if older_history:
                older_times = [attempt["response_time"] for attempt in older_history]
                avg_older_time = sum(older_times) / len(older_times) if older_times else 0
                # Negative value means getting faster (improvement)
                time_change = (avg_recent_time - avg_older_time) / max(avg_older_time, 0.001)
            else:
                time_change = 0


            # Combine factors to get learning rate
            # Higher accuracy improvement and faster responses = higher learning rate
            base_rate = DEFAULT_LEARNING_RATE
            accuracy_factor = accuracy_change * 0.2  # Scale by 0.2
            time_factor = -time_change * 0.1  # Negative because less time is better

            # Calculate new learning rate (bounded between MIN_LEARNING_RATE and MAX_LEARNING_RATE)
            new_rate = base_rate + accuracy_factor + time_factor
            new_rate = max(MIN_LEARNING_RATE, min(MAX_LEARNING_RATE, new_rate))

            # Smooth learning rate changes to prevent wild fluctuations
            current_rate = data["learning_rate"]
            smoothed_rate = current_rate * 0.7 + new_rate * 0.3  # 70% old, 30% new

            # Update the learning rate
            data["learning_rate"] = smoothed_rate

        except Exception as e:
            logger.error(f"Error updating learning rate for pattern {pattern}: {e}")
    
    def _update_skill_level(self) -> None:
        """Update the overall skill level based on pattern mastery and learning rate."""
        # Calculate weighted average of pattern mastery levels
        total_weight = 0.0
        weighted_sum = 0.0
        
        for pattern, data in self.pattern_metrics.items():
            # Weight by number of attempts (more attempts = more reliable data)
            # Use a logarithmic scale to prevent a few patterns from dominating
            weight = math.log(data["attempts"] + 1) / math.log(10)  # log10(attempts+1)
            weighted_sum += data["mastery"] * weight
            total_weight += weight
        
        # Calculate raw skill level
        if total_weight > 0:
            raw_skill = weighted_sum / total_weight
        else:
            raw_skill = 0.0
        
        # Apply learning rate adjustment
        avg_learning_rate = self._calculate_average_learning_rate()
        learning_boost = avg_learning_rate * 0.5  # Scale the boost (max +0.1)
        
        # Combine raw skill with learning boost
        adjusted_skill = raw_skill * (1.0 + learning_boost)
        
        # Apply a smooth learning curve function
        # This creates a sigmoid-like curve that's steeper at the beginning
        new_skill_level = 1.0 - 1.0 / (1.0 + math.exp(5.0 * adjusted_skill - 2.5))
        
        # Smooth skill level changes to prevent wild fluctuations
        current_skill = self.overall_metrics["skill_level"]
        smoothed_skill = current_skill * 0.8 + new_skill_level * 0.2  # 80% old, 20% new
        
        self.overall_metrics["skill_level"] = smoothed_skill
        
        # Update overall learning rate based on recent performance
        recent_performance = self._calculate_recent_performance()
        self.overall_metrics["learning_rate"] = 0.03 + 0.07 * recent_performance
    
    def _calculate_average_learning_rate(self) -> float:
        """
        Calculate the average learning rate across all patterns.
        
        Returns:
            Average learning rate
        """
        learning_rates = [
            data["learning_rate"] for pattern, data in self.pattern_metrics.items()
            if data["attempts"] > 0
        ]
        
        if not learning_rates:
            return DEFAULT_LEARNING_RATE  # Default
        
        return sum(learning_rates) / len(learning_rates)
    
    def _calculate_recent_performance(self) -> float:
        """
        Calculate performance over recent attempts.
        
        Returns:
            Performance rating from 0.0 to 1.0
        """
        # Get the last 10 attempts across all patterns
        recent_attempts = []
        
        for pattern, data in self.pattern_metrics.items():
            for attempt in data["history"]:
                recent_attempts.append((attempt["timestamp"], attempt["correct"]))
        
        # Sort by timestamp (newest first)
        recent_attempts.sort(key=lambda x: x[0], reverse=True)
        recent_attempts = recent_attempts[:10]
        
        if not recent_attempts:
            return 0.5  # Default to mid-range
        
        # Calculate weighted average (newer attempts count more)
        total_weight = 0.0
        weighted_sum = 0.0
        
        for i, (_, correct) in enumerate(recent_attempts):
            weight = 1.0 / (i + 1)  # Most recent has weight 1, then 1/2, 1/3, etc.
            weighted_sum += weight * (1.0 if correct else 0.0)
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.5
    
    def get_pattern_performance(self, pattern: str) -> PerformanceRating:
        """
        Get performance rating for a specific pattern.
        
        Args:
            pattern: The candlestick pattern name
            
        Returns:
            PerformanceRating indicating user's performance level
        """
        if pattern not in self.pattern_metrics:
            return PerformanceRating.UNKNOWN
        
        with self._lock:
            data = self.pattern_metrics[pattern]
            
            # Need at least 2 attempts for a meaningful rating
            if data["attempts"] < 2:
                return PerformanceRating.UNKNOWN
            
            # Ensure metrics are up to date
            if pattern in self._update_needed:
                self._update_pattern_metrics(pattern)
                self._update_learning_rate(pattern)
                self._update_needed.remove(pattern)
            
            mastery = data["mastery"]
            
            if mastery >= 0.8:
                return PerformanceRating.EXCELLENT
            elif mastery >= 0.6:
                return PerformanceRating.GOOD
            elif mastery >= 0.4:
                return PerformanceRating.FAIR
            else:
                return PerformanceRating.POOR
    
    @log_execution_time()
    def get_patterns_by_performance(self) -> Dict[PerformanceRating, List[str]]:
        """
        Group patterns by performance rating.
        
        Returns:
            Dictionary mapping performance ratings to lists of pattern names
        """
        with self._lock:
            # Update any patterns that need it
            self._update_pending_metrics()
            
        result: Dict[PerformanceRating, List[str]] = {
            rating: [] for rating in PerformanceRating
        }
        
        for pattern in self.pattern_metrics:
            rating = self.get_pattern_performance(pattern)
            result[rating].append(pattern)
        
        return result
    
    @log_execution_time()
    def get_patterns_for_practice(self, count: int = 5) -> List[str]:
        """
        Get patterns that need the most practice.
        
        Args:
            count: Number of patterns to return
            
        Returns:
            List of pattern names that need practice
        """
        with self._lock:
            # Update any patterns that need it
            self._update_pending_metrics()
            
        # Group patterns by performance
        performance_groups = self.get_patterns_by_performance()
        
        # Prioritize:
        # 1. Patterns with POOR performance
        # 2. Patterns with FAIR performance
        # 3. Patterns with UNKNOWN performance (not attempted enough)
        # 4. Patterns with GOOD performance
        priority_order = [
            PerformanceRating.POOR,
            PerformanceRating.FAIR,
            PerformanceRating.UNKNOWN,
            PerformanceRating.GOOD
        ]
        
        result = []
        for rating in priority_order:
            patterns = performance_groups[rating]
                
            if rating == PerformanceRating.UNKNOWN:
                # Sort unknown patterns by attempts (least attempts first)
                patterns.sort(key=lambda p: self.pattern_metrics[p]["attempts"])
            else:
                # Sort other patterns by mastery (lowest first)
                patterns.sort(key=lambda p: self.pattern_metrics[p]["mastery"])
            
            result.extend(patterns)
            if len(result) >= count:
                return result[:count]
        
        # If we still need more, include EXCELLENT patterns
        if len(result) < count:
            excellent_patterns = performance_groups[PerformanceRating.EXCELLENT]
            # Sort by least recently seen
            excellent_patterns.sort(key=lambda p: self.pattern_metrics[p]["last_seen"])
            result.extend(excellent_patterns)
        
        return result[:count]
    
    @log_execution_time()
    def calculate_difficulty_adjustment(self) -> float:
        """
        Calculate the difficulty adjustment based on skill level.
        
        Returns:
            Difficulty adjustment factor (0.0 to 1.0)
        """
        with self._lock:
            # Update any pending metrics
            self._update_pending_metrics()
            
        skill_level = self.overall_metrics["skill_level"]
        
        # Start with a minimum difficulty that increases with skill
        min_difficulty = 0.3 + 0.3 * skill_level
        
        # Apply learning rate - faster learners get more challenging content
        learning_rate = self.overall_metrics["learning_rate"]
        learning_adjustment = 0.1 * learning_rate
        
        # Apply streak bonus - reward streaks with slightly increased difficulty
        streak = self.overall_metrics["streak"]
        streak_bonus = min(0.15, 0.01 * streak)  # Max bonus of 0.15 at streak of 15
        
        # Track consecutive correct/incorrect streaks for dynamic adjustment
        consecutive_correct = self.overall_metrics["consecutive_correct"]
        consecutive_incorrect = self.overall_metrics["consecutive_incorrect"]
        
        dynamic_adjustment = 0.0
        
            # Apply progressive difficulty adjustments based on streaks
        if consecutive_correct >= 5:
            dynamic_adjustment = 0.2  # Big increase
        elif consecutive_correct >= 3:
            dynamic_adjustment = 0.1  # Medium increase
            
        if consecutive_incorrect >= 3:
            dynamic_adjustment = -0.2  # Big decrease
        elif consecutive_incorrect >= 2:
            dynamic_adjustment = -0.1  # Medium decrease
        
        # Combine factors
        difficulty = min_difficulty + learning_adjustment + streak_bonus + dynamic_adjustment
        
        # Ensure we stay in 0.0-1.0 range with a smooth progression
        return min(1.0, max(0.3, difficulty))

class ForgettingCurveManager:
    """
    Implements the forgetting curve algorithm for spaced repetition.
    
    This class determines:
    1. When patterns should be reviewed
    2. The optimal review intervals for each pattern
    3. The difficulty adjustment for spaced repetition
    
    Features:
    - Optimized retention calculations with caching
    - Smart scheduling of review intervals
    - Adapts to individual learning characteristics
    """
    
    def __init__(self, user_performance: UserPerformanceTracker):
        """
        Initialize the forgetting curve manager.
        
        Args:
            user_performance: User performance tracker instance
        """
        self.user_performance = user_performance
        self.base_interval = 24 * 3600  # 1 day in seconds
        self.logger = app_logger.getChild("forgetting_curve")
        
        # Integration with common forgetting curve model
        self.forgetting_model = ForgettingCurveModel()
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Performance optimization
        self._next_review_cache = {}  # Cache for next review calculations
        self._retention_cache = {}    # Cache for retention calculations
        
        # Review interval schedule (in hours) - exponential spacing
        self.review_schedule = [
            1,      # 1 hour
            4,      # 4 hours
            8,      # 8 hours
            24,     # 1 day
            72,     # 3 days
            168,    # 1 week
            336,    # 2 weeks
            672,    # 4 weeks
            1344,   # 8 weeks
        ]
    
    @log_execution_time()
    def calculate_next_review(self, pattern: str) -> float:
        """
        Calculate when a pattern should next be reviewed.
        
        Args:
            pattern: The candlestick pattern name
            
        Returns:
            Timestamp when the pattern should be reviewed next
        """
        # Define cache_key and current_time early for broader scope
        current_time = time.time()
        cache_key = f"{pattern}:{int(current_time / 3600)}"  # Cache by hour

        with self._lock:
            # Check cache first
            if cache_key in self._next_review_cache:
                return self._next_review_cache[cache_key]
                
        if pattern not in self.user_performance.pattern_metrics:
            self.logger.warning(f"Pattern '{pattern}' not found in metrics. Reviewing immediately.")
            return current_time  # Review immediately if pattern not in metrics
        
        data = self.user_performance.pattern_metrics[pattern]
        
        # If never seen or no last seen time, review immediately
        if data["attempts"] == 0 or data["last_seen"] == 0:
            self.logger.debug(f"Pattern '{pattern}' never seen or no last_seen time. Reviewing immediately.")
            return current_time
        
        try:
            # Get memory parameters
            memory_strength = data["memory_strength"]
            forgetting_rate = data["forgetting_rate"]
            
            # Calculate optimal interval based on memory strength
            # Higher memory strength = longer interval
            optimal_hours = self.forgetting_model.calculate_optimal_interval(
                memory_strength, forgetting_rate, target_retention=0.7
            )
            
            # Apply difficulty-based adjustment
            mastery = data["mastery"]
            difficulty_factor = 1.0
            
            # Apply a graduated difficulty factor
            if mastery < 0.3:
                # Struggling with pattern - review more frequently
                difficulty_factor = 0.7
            elif mastery > 0.8:
                # Mastered pattern - can wait longer
                difficulty_factor = 1.3
            
            # Apply spaced repetition factor (ease factor)
            due_factor = data.get("due_factor", 1.0) # Default ease factor
            adjusted_hours = optimal_hours * difficulty_factor * due_factor
            
            # Find the closest review schedule point
            next_review_hours = self._find_next_review_schedule(adjusted_hours)
            
            # Calculate next review time
            next_review_timestamp = data["last_seen"] + (next_review_hours * 3600)
                    
            # Apply randomization to prevent bunching of reviews
            randomization = random.uniform(0.9, 1.1)  # +/- 10% variability
            # Ensure randomization doesn't make it earlier than now if last_seen was recent
            next_review_timestamp = max(current_time, data["last_seen"] + (next_review_hours * 3600 * randomization))

            # Log scheduled review
            self.logger.debug(
                f"Scheduled review for pattern {pattern}: "
                f"memory_strength={memory_strength:.2f}, "
                f"optimal_hours={optimal_hours:.1f}, "
                f"adjusted_hours={adjusted_hours:.1f}, "
                f"next_review_hours={next_review_hours}, "
                f"next_review_timestamp={datetime.fromtimestamp(next_review_timestamp)}"
            )
                    
            # Cache the result (needs lock)
            with self._lock:
                self._next_review_cache[cache_key] = next_review_timestamp
            
            return next_review_timestamp
                
        except Exception as e:
            self.logger.error(f"Error calculating next review for pattern {pattern}: {e}", exc_info=True)
            # Return a sensible default in case of error: 1 day from last seen, but not before now
            default_next_review = data["last_seen"] + (24 * 3600)
            return max(current_time, default_next_review)
    
    def _find_next_review_schedule(self, hours: float) -> float:
        """
        Find the next review point in the schedule.
        
        Args:
            hours: Calculated optimal hours
            
        Returns:
            Hours until next review from schedule
        """
        # Find the closest next point in review schedule
        for interval in self.review_schedule:
            if interval >= hours:
                return interval
        
        # If beyond all intervals, use the largest one with a scaling factor
        # This allows for gradually increasing intervals beyond the explicit schedule
        largest_interval = self.review_schedule[-1]
        factor = hours / largest_interval
        # Cap at reasonable maximum (6 months)
        return min(largest_interval * factor, 4320)  # 180 days max
    
    @log_execution_time()
    def update_due_factor(self, pattern: str, was_correct: bool, response_time: float = 0) -> None:
        """
        Update the due factor for a pattern based on review result.

        Args:
            pattern: The candlestick pattern name
            was_correct: Whether the answer was correct
            response_time: Response time in seconds (optional)
        """
        with self._lock:
            if pattern not in self.user_performance.pattern_metrics:
                self.logger.warning(f"Attempted to update non-existent pattern: {pattern}")
                return

            # Invalidate caches for this pattern before updating metrics
            for key in list(self._next_review_cache.keys()):
                if key.startswith(f"{pattern}:"):
                    del self._next_review_cache[key]
            self._retention_cache = {k: v for k, v in self._retention_cache.items()
                                     if not k.startswith(f"{pattern}:")}

            data = self.user_performance.pattern_metrics[pattern]

            try:
                # Get current memory parameters
                memory_strength = data["memory_strength"]
                forgetting_rate = data["forgetting_rate"]

                # Calculate quality of recall (0-5 scale)
                recall_quality = self._calculate_recall_quality(was_correct, response_time, data)

                # Update memory parameters based on recall quality
                # Higher quality = greater memory strengthening
                strength_increment = 0.05 + (recall_quality / 10.0)  # 0.05 to 0.55

                # Failed recall slightly decreases memory strength
                if not was_correct:
                    memory_strength = max(0.1, memory_strength - 0.05)
                else:
                    memory_strength = min(0.95, memory_strength + strength_increment)

                # Update forgetting rate based on recall quality
                if was_correct:
                    # Correct answers mean slower forgetting (smaller forgetting rate)
                    forgetting_rate = max(0.05, forgetting_rate * 0.95)
                else:
                    # Incorrect answers mean faster forgetting (larger forgetting rate)
                    forgetting_rate = min(0.5, forgetting_rate * 1.1)

                # Update memory parameters
                data["memory_strength"] = memory_strength
                data["forgetting_rate"] = forgetting_rate

                # Update due factor based on recall quality
                if was_correct:
                    # Increase due factor if answer was correct
                    # Higher quality = larger increase
                    increase_factor = 1.0 + (recall_quality / 10.0)  # 1.0 to 1.5
                    data["due_factor"] = min(5.0, data["due_factor"] * increase_factor)
                else:
                    # Reset due factor if answer was wrong
                    data["due_factor"] = 1.0

                # Update last seen timestamp
                data["last_seen"] = time.time()

                self.logger.info(
                    f"Updated memory parameters for pattern {pattern}: "
                    f"memory_strength={memory_strength:.2f}, "
                    f"forgetting_rate={forgetting_rate:.2f}, "
                    f"due_factor={data['due_factor']:.2f}, "
                    f"recall_quality={recall_quality}"
                )

            except Exception as e:
                self.logger.error(f"Error updating due factor for pattern {pattern}: {e}", exc_info=True)
    def _calculate_recall_quality(self, was_correct: bool, response_time: float,
                                  pattern_data: Dict[str, Any]) -> int:
        """
        Calculate quality of recall on a 0-5 scale.

        Args:
            was_correct: Whether the answer was correct
            response_time: Time taken to respond in seconds
            pattern_data: Pattern-specific metrics

        Returns:
            Recall quality (0-5 scale)
        """
        if not was_correct:
            return 0  # Incorrect answer

        try:
            # Calculate average response time for the pattern
            # Use response_time as fallback if avg_response_time is None or 0
            avg_time = pattern_data.get("avg_response_time") or response_time
            # Ensure avg_time is not zero to avoid division by zero
            avg_time = max(0.1, avg_time)

            # Calculate normalized response time ratio
            # <1.0 means faster than average, >1.0 means slower
            time_ratio = response_time / avg_time

            # Calculate quality based on response time
            if time_ratio < 0.6:
                # Very fast (40% faster than average)
                return 5
            elif time_ratio < 0.8:
                # Fast (20% faster than average)
                return 4
            elif time_ratio < 1.0:
                # Slightly faster than average
                return 3
            elif time_ratio < 1.5:
                # Slower than average but still good
                return 2
            else:
                # Much slower than average
                return 1
        except Exception as e:
            self.logger.error(f"Error calculating recall quality: {e}", exc_info=True)
            return 1  # Default to low quality in case of error
    
    def calculate_retention(self, pattern: str, elapsed_hours: Optional[float] = None) -> float:
        """
        Calculate current retention probability for a pattern.
        
        Args:
            pattern: The candlestick pattern name
            elapsed_hours: Optional override for hours since last review
            
        Returns:
            Retention probability (0.0 to 1.0)
        """
        with self._lock:
            try:
                if pattern not in self.user_performance.pattern_metrics:
                    return 0.0
                
                data = self.user_performance.pattern_metrics[pattern]
                
                if data["last_seen"] == 0:
                    return 0.0
                
                if elapsed_hours is None:
                    elapsed_hours = (time.time() - data["last_seen"]) / 3600
                
                # Check cache
                cache_key = f"{pattern}:{int(elapsed_hours)}"
                if cache_key in self._retention_cache:
                    return self._retention_cache[cache_key]
                
                # Calculate retention using forgetting curve model
                retention = self.forgetting_model.calculate_retention(
                    data["memory_strength"],
                    data["forgetting_rate"],
                    elapsed_hours
                )
                
                # Cache the result
                self._retention_cache[cache_key] = retention
                
                return retention
            except Exception as e:
                self.logger.error(f"Error calculating retention for pattern {pattern}: {e}")
                return 0.5  # Return a mid-range value in case of error
    
    @log_execution_time()
    def get_due_patterns(self) -> List[Tuple[str, float]]:
        """
        Get patterns that are due for review, sorted by priority.

        Returns:
            List of (pattern, due_score) tuples sorted by due_score (highest first)
        """
        with self._lock:
            now = time.time()
            due_patterns = []

            for pattern, data in self.user_performance.pattern_metrics.items():
                try:
                    if data["attempts"] == 0:
                        # Include new patterns with a mid-range priority
                        due_patterns.append((pattern, 0.5))
                        continue

                    next_review = self.calculate_next_review(pattern)

                    # Calculate how overdue the pattern is
                    # Negative value = not due yet, positive = overdue
                    due_score = (now - next_review) / self.base_interval

                    # Calculate retention probability at current time
                    retention = self.calculate_retention(pattern)

                    # Adjust due score based on retention
                    # Lower retention = higher priority
                    retention_factor = max(0.1, 1.0 - retention)
                    adjusted_due_score = due_score * retention_factor * 2

                    # Calculate mastery-based priority
                    # Lower mastery = higher priority for review
                    mastery_priority = 1.0 - data.get("mastery", 0.5) # Use .get for safety

                    # Final priority score combines due score and mastery
                    priority_score = (adjusted_due_score * 0.7) + (mastery_priority * 0.3)

                    # Only include patterns that are due or close to due
                    if due_score > -0.5:  # Include patterns that are close to being due
                        due_patterns.append((pattern, priority_score))

                except Exception as e:
                    self.logger.error(f"Error calculating due score for pattern {pattern}: {e}")
                    # Optionally add a default entry or skip
                    # due_patterns.append((pattern, 0.0)) # Example: Add with low priority

            # Sort by priority_score (highest first)
            due_patterns.sort(key=lambda x: x[1], reverse=True)

            return due_patterns
    
    @log_execution_time()
    def get_optimal_batch(self, count: int = 5) -> List[str]:
        """
        Get an optimal batch of patterns for review.

        This combines due patterns with patterns that need practice
        to create an optimal learning session.

        Args:
            count: Number of patterns to return

        Returns:
            List of pattern names to review
        """
        with self._lock:
            try:
                # Get patterns due for review
                due_patterns = self.get_due_patterns()
                due_pattern_names = [p[0] for p in due_patterns]

                # Get patterns that need practice based on performance
                practice_patterns = self.user_performance.get_patterns_for_practice(count * 2)

                # Create a balanced batch with 70% due patterns and 30% practice patterns
                due_count = min(len(due_pattern_names), int(count * 0.7) + 1)
                selected_due_patterns = due_pattern_names[:due_count]

                # Add practice patterns that aren't already in due patterns
                remaining_slots = count - len(selected_due_patterns)
                additional_patterns = [
                    p for p in practice_patterns
                    if p not in selected_due_patterns
                ][:remaining_slots]

                # Combine the two lists
                result = selected_due_patterns + additional_patterns

                # Ensure we have enough patterns by adding random ones if needed
                if len(result) < count:
                    all_patterns = list(self.user_performance.pattern_metrics.keys())
                    # Prioritize patterns with at least one attempt
                    attempted_patterns = [p for p in all_patterns
                                          if self.user_performance.pattern_metrics[p]["attempts"] > 0
                                          and p not in result]
                    if attempted_patterns:
                        # Sort by least recently seen
                        attempted_patterns.sort(key=lambda p:
                                                self.user_performance.pattern_metrics[p]["last_seen"]
                                                or 0)
                        result.extend(attempted_patterns[:count - len(result)])

                    # If we still need more, add unseen patterns
                    if len(result) < count:
                        unseen_patterns = [p for p in all_patterns
                                           if self.user_performance.pattern_metrics[p]["attempts"] == 0
                                           and p not in result]
                        random.shuffle(unseen_patterns)
                        result.extend(unseen_patterns[:count - len(result)])

                # Final randomization to avoid predictability
                if len(result) > 1:
                    # Keep the top 1-2 most due patterns in place, shuffle the rest
                    top_patterns = result[:min(2, len(result))]
                    shuffle_patterns = result[min(2, len(result)):]
                    random.shuffle(shuffle_patterns)
                    result = top_patterns + shuffle_patterns

                return result[:count]

            except Exception as e:
                self.logger.error(f"Error generating optimal batch: {e}")
                # Fallback: return random patterns
                all_patterns = list(self.user_performance.pattern_metrics.keys())
                random.shuffle(all_patterns)
                return all_patterns[:count]

class ReinforcementLearningDifficultyOptimizer:
    """
    Reinforcement Learning-based optimizer for difficulty levels.
    
    Uses a Q-learning approach to learn optimal difficulty adjustments
    based on user performance.
    
    Features:
    - Q-learning with adjustable exploration/exploitation
    - Dynamic learning rate based on performance
    - Memory-optimized state-action storage
    """
    
    def __init__(self, user_performance: UserPerformanceTracker):
        """
        Initialize RL optimizer.
        
        Args:
            user_performance: User performance tracker
        """
        self.user_performance = user_performance
        self.logger = app_logger.getChild("rl_optimizer")
        
        # Q-learning parameters
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.exploration_rate = 0.2
        self.min_exploration_rate = 0.05
        self.exploration_decay = 0.995
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Memory-optimized Q-table using sparse representation
        # Format: {(state_tuple): {action: q_value}}
        self.q_table = defaultdict(lambda: defaultdict(float))
        
        # Performance counters for analytics
        self.update_count = 0
        self.reward_history = deque(maxlen=100)  # Sliding window of recent rewards
        
        # State discretization settings
        self.mastery_bins = 5  # Number of bins for mastery level
        self.confidence_bins = 3  # Number of bins for confidence level
        self.streak_bins = 3  # Number of bins for correct/incorrect streaks
        
        # Action space
        self.actions = [-2, -1, 0, 1, 2]  # Difficulty adjustments
    
    def _get_state(self, metrics: Dict[str, Any]) -> Tuple[int, int, int]:
        """
        Convert user performance metrics to a discretized state.
        
        Args:
            metrics: User performance metrics
            
        Returns:
            Discretized state as a tuple
        """
        try:
            # Extract relevant features
            mastery = metrics.get("mastery", 0.5)
            confidence = metrics.get("confidence", 0.5)
            correct_streak = min(5, metrics.get("correct_streak", 0))
            incorrect_streak = min(5, metrics.get("incorrect_streak", 0))
            
            # Discretize continuous values into bins
            mastery_bin = min(self.mastery_bins - 1, int(mastery * self.mastery_bins))
            confidence_bin = min(self.confidence_bins - 1, int(confidence * self.confidence_bins))
            
            # Use streak difference as a single feature
            streak_diff = correct_streak - incorrect_streak
            streak_bin = max(0, min(self.streak_bins - 1, 
                                 int((streak_diff + 5) / 10 * self.streak_bins)))
            
            return (mastery_bin, confidence_bin, streak_bin)
        except Exception as e:
            self.logger.error(f"Error getting RL state: {e}")
            # Return a default state in case of error
            return (2, 1, 1)  # Middle values for each dimension
    
    def _get_best_action(self, state: Tuple[int, int, int]) -> int:
        """
        Get the best action for a given state using Q-values.
        
        Args:
            state: The current state tuple
            
        Returns:
            Selected action (difficulty adjustment)
        """
        with self._lock:
            try:
                # Apply epsilon-greedy exploration strategy
                if random.random() < self.exploration_rate:
                    # Explore: Choose random action
                    return random.choice(self.actions)
                else:
                    # Exploit: Choose best known action
                    state_actions = self.q_table[state]
                    
                    # If no actions were taken in this state before, return default
                    if not state_actions:
                        return 0  # No change in difficulty
                    
                    # Find action with highest Q-value
                    return max(state_actions.items(), key=lambda x: x[1])[0]
            except Exception as e:
                self.logger.error(f"Error getting best action: {e}")
                return 0  # Default to no change in difficulty
    
    def _calculate_reward(self, metrics: Dict[str, Any], action: int, 
                          was_correct: bool, response_time: float) -> float:
        """
        Calculate reward based on user performance and action taken.
        
        Args:
            metrics: User performance metrics
            action: Action taken (difficulty adjustment)
            was_correct: Whether answer was correct
            response_time: Response time in seconds
            
        Returns:
            Calculated reward
        """
        try:
            # Base reward is positive for correct, negative for incorrect
            base_reward = 1.0 if was_correct else -1.0
            
            # Adjust reward based on difficulty change:
            # - If correct and difficulty was increased, give bonus
            # - If incorrect and difficulty was decreased, give bonus
            # - If correct and difficulty was decreased, give penalty
            # - If incorrect and difficulty was increased, give penalty
            difficulty_alignment = (was_correct and action > 0) or (not was_correct and action < 0)
            difficulty_misalignment = (was_correct and action < 0) or (not was_correct and action > 0)
            
            alignment_modifier = 0.5 if difficulty_alignment else 0.0
            misalignment_modifier = -0.5 if difficulty_misalignment else 0.0
            
            # Adjust reward based on response time
            # Faster correct answers are better
            # Slower incorrect answers are worse (indicates confusion)
            time_factor = 0.0
            if metrics.get("avg_response_time"):
                avg_time = max(0.1, metrics["avg_response_time"])
                time_ratio = response_time / avg_time
                
                if was_correct:
                    # Reward fast correct answers
                    time_factor = max(-0.5, min(0.5, 0.5 - (time_ratio * 0.5)))
                else:
                    # Penalize slow incorrect answers
                    time_factor = max(-0.5, min(0.0, -0.5 * min(1.0, time_ratio - 1.0)))
            
            # Calculate final reward
            reward = base_reward + alignment_modifier + misalignment_modifier + time_factor
            
            return max(-2.0, min(2.0, reward))  # Clamp reward to [-2.0, 2.0]
        except Exception as e:
            self.logger.error(f"Error calculating reward: {e}")
            return 0.0  # Default to neutral reward in case of error
    
    @log_execution_time()
    def update(self, pattern: str, difficulty_change: int, was_correct: bool,
               response_time: float) -> None:
        """
        Update Q-values based on action and outcome.

        Args:
            pattern: Candlestick pattern name
            difficulty_change: Difficulty adjustment that was applied
            was_correct: Whether the answer was correct
            response_time: Response time in seconds
        """
        with self._lock:
            try:
                if pattern not in self.user_performance.pattern_metrics:
                    self.logger.warning(f"Attempted to update RL model for non-existent pattern: {pattern}")
                    return  # Exit if pattern data doesn't exist

                metrics = self.user_performance.pattern_metrics[pattern]

                # Get current state
                current_state = self._get_state(metrics)

                # Calculate reward
                reward = self._calculate_reward(metrics, difficulty_change,
                                                was_correct, response_time)

                # Add to reward history
                self.reward_history.append(reward)

                # Update mastery-based state to calculate next state
                if was_correct:
                    # Simulate temporary mastery increase for next state
                    temp_mastery = min(1.0, metrics["mastery"] + 0.1)
                else:
                    # Simulate temporary mastery decrease for next state
                    temp_mastery = max(0.0, metrics["mastery"] - 0.1)

                # Simulate the next state
                temp_metrics = metrics.copy()
                temp_metrics["mastery"] = temp_mastery
                next_state = self._get_state(temp_metrics)

                # Q-learning update
                # Q(s,a) = Q(s,a) + alpha * [r + gamma * max(Q(s',a')) - Q(s,a)]
                current_q = self.q_table[current_state][difficulty_change]

                # Get max Q value for next state
                next_max_q = max([self.q_table[next_state][a] for a in self.actions], default=0)

                # Calculate new Q value
                new_q = current_q + self.learning_rate * (
                    reward + self.discount_factor * next_max_q - current_q
                )

                # Update Q-table
                self.q_table[current_state][difficulty_change] = new_q

                # Update exploration rate (decay over time)
                self.exploration_rate = max(
                    self.min_exploration_rate,
                    self.exploration_rate * self.exploration_decay
                )

                # Update learning rate based on update count
                # Start high, then gradually decrease for stability
                self.update_count += 1
                if self.update_count % 50 == 0:
                    self.learning_rate = max(0.01, self.learning_rate * 0.95)

                # Cleanup: remove near-zero Q-values to save memory
                if self.update_count % 100 == 0:
                    self._cleanup_q_table()

                # Log update
                self.logger.debug(
                    f"RL update for pattern '{pattern}': "
                    f"state={current_state}, action={difficulty_change}, "
                    f"reward={reward:.2f}, new_q={new_q:.2f}, "
                    f"exploration_rate={self.exploration_rate:.3f}"
                )
            except KeyError as e:
                # Handle cases where state or action might not exist in q_table yet
                self.logger.error(f"KeyError during RL update for pattern '{pattern}': {e}. State: {current_state}, Action: {difficulty_change}", exc_info=True)
                # Initialize missing entries if necessary
                if current_state not in self.q_table:
                    self.q_table[current_state] = {action: 0.0 for action in self.actions}
                if difficulty_change not in self.q_table[current_state]:
                    self.q_table[current_state][difficulty_change] = 0.0
                # Optionally retry the update or just log and continue
            except Exception as e:
                self.logger.error(f"Error updating RL model for pattern '{pattern}': {e}", exc_info=True)
    
    def _cleanup_q_table(self) -> None:
        """
        Clean up Q-table by removing near-zero entries to save memory.
        """
        try:
            threshold = 0.01
            states_to_remove = []
            
            # Find states with no significant Q-values
            for state, actions in self.q_table.items():
                # Remove near-zero actions
                actions_to_remove = [
                    action for action, q_value in actions.items() 
                    if abs(q_value) < threshold
                ]
                
                for action in actions_to_remove:
                    del actions[action]
                
                # If no actions left, mark state for removal
                if not actions:
                    states_to_remove.append(state)
            
            # Remove empty states
            for state in states_to_remove:
                del self.q_table[state]
            
            self.logger.debug(f"Cleaned up Q-table: removed {len(states_to_remove)} states")
        except Exception as e:
            self.logger.error(f"Error cleaning up Q-table: {e}")
    
    @log_execution_time()
    def get_difficulty_adjustment(self, pattern: str) -> int:
        """
        Get recommended difficulty adjustment for a pattern.
        
        Args:
            pattern: Candlestick pattern name
            
        Returns:
            Recommended difficulty adjustment (-2 to +2)
        """
        with self._lock:
            try:
                if pattern not in self.user_performance.pattern_metrics:
                    return 0  # Default: no change
                
                metrics = self.user_performance.pattern_metrics[pattern]
                state = self._get_state(metrics)
                action = self._get_best_action(state)
                
                self.logger.debug(
                    f"RL recommending adjustment {action} for pattern '{pattern}' "
                    f"with state {state}"
                )
                
                return action
            except Exception as e:
                self.logger.error(f"Error getting difficulty adjustment: {e}")
                return 0  # Default to no adjustment in case of error
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics for the RL optimizer.

        Returns:
            Dictionary with performance metrics
        """
        with self._lock:
            try:
                # Calculate average reward
                avg_reward = sum(self.reward_history) / max(1, len(self.reward_history))

                # Calculate policy statistics
                policy_stats = {}
                for state in self.q_table:
                    if self.q_table[state]:
                        best_action = max(self.q_table[state].items(), key=lambda x: x[1])[0]
                        policy_stats[str(state)] = best_action

                return {
                    "average_reward": avg_reward,
                    "exploration_rate": self.exploration_rate,
                    "learning_rate": self.learning_rate,
                    "update_count": self.update_count,
                    "q_table_size": len(self.q_table),
                    "policy_sample": dict(list(policy_stats.items())[:10])
                }
            except Exception as e:
                self.logger.error(f"Error getting RL performance stats: {e}")
                return {"error": str(e)}
                
    def serialize(self) -> Dict[str, Any]:
        """
        Serialize the RL model for storage.
            
        Returns:
            Dictionary with serialized model data
        """
        with self._lock:
            try:
                # Convert defaultdict to regular dict for serialization
                q_table_serialized = {}
                for state, actions in self.q_table.items():
                    # Convert inner defaultdict to dict
                    q_table_serialized[str(state)] = dict(actions)
                
                return {
                    "q_table": q_table_serialized,
                    "learning_rate": self.learning_rate,
                    "exploration_rate": self.exploration_rate,
                    "update_count": self.update_count
                }
            except Exception as e:
                self.logger.error(f"Error serializing RL model: {e}")
                return {}
    
    def deserialize(self, data: Dict[str, Any]) -> None:
        """
        Deserialize the RL model from storage.
        
        Args:
            data: Dictionary with serialized model data
        """
        with self._lock:
            try:
                # Restore parameters
                self.learning_rate = data.get("learning_rate", 0.1)
                self.exploration_rate = data.get("exploration_rate", 0.2)
                self.update_count = data.get("update_count", 0)
                
                # Restore Q-table
                q_table_data = data.get("q_table", {})
                for state_str, actions in q_table_data.items():
                    # Convert string state back to tuple
                    try:
                        state_tuple = eval(state_str)
                        for action, q_value in actions.items():
                            self.q_table[state_tuple][int(action)] = float(q_value)
                    except Exception as inner_e:
                        self.logger.error(f"Error deserializing state {state_str}: {inner_e}")
                
                self.logger.info(f"Deserialized RL model with {len(self.q_table)} states")
            except Exception as e:
                self.logger.error(f"Error deserializing RL model: {e}")

class MultiBanditDifficultyOptimizer:
    """
    Multi-armed bandit approach to difficulty optimization.
    
    Uses a Thompson Sampling algorithm to model uncertainty and
    balance exploration-exploitation tradeoff.
    
    Features:
    - Per-pattern bandit models for personalized difficulty selection
    - Bayesian updates with Thompson sampling
    - Memory-efficient model storage
    - Thread-safe operations
    """
    
    def __init__(self, user_performance: UserPerformanceTracker):
        """
        Initialize the multi-armed bandit optimizer.
        
        Args:
            user_performance: User performance tracker
        """
        self.user_performance = user_performance
        self.logger = app_logger.getChild("bandit_optimizer")
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Bandits storage - using defaultdict for automatic initialization
        # Maps pattern -> difficulty_adjustment -> (success, failure) counts
        self._bandits = defaultdict(lambda: defaultdict(lambda: [1.0, 1.0]))
        
        # Performance metrics
        self._total_pulls = 0
        self._performance_history = deque(maxlen=100)  # Last 100 selections
        
        # Configurable parameters
        self.explore_rate = 0.1  # Explicit exploration rate
        self.min_pulls_before_confident = 5  # Minimum pulls before confidence
        
        # Action space
        self.actions = [-2, -1, 0, 1, 2]  # Difficulty adjustments
    
    @log_execution_time()
    def update(self, pattern: str, adjustment: int, was_correct: bool) -> None:
        """
        Update bandit model based on outcome.

        Args:
            pattern: Candlestick pattern
            adjustment: Difficulty adjustment that was selected
            was_correct: Whether the user answered correctly
        """
        with self._lock:
            try:
                # Cast adjustment to int to ensure consistency
                adjustment = int(adjustment)
                if adjustment not in self.actions:
                    self.logger.warning(f"Invalid adjustment value: {adjustment}. Using 0 instead.")
                    adjustment = 0

                # Get current arm counts
                success, failure = self._bandits[pattern][adjustment]

                # Update based on outcome
                if was_correct:
                    success += 1.0
                else:
                    failure += 1.0

                # Store updated values
                self._bandits[pattern][adjustment] = [success, failure]

                # Track performance
                self._total_pulls += 1
                self._performance_history.append(1.0 if was_correct else 0.0)

                # Log update
                self.logger.debug(
                    f"Updated bandit for pattern '{pattern}', adjustment {adjustment}: "
                    f"correct={was_correct}, new counts=[{success:.1f}, {failure:.1f}]"
                )

                # Clean up rarely used arms periodically
                if self._total_pulls % 100 == 0:
                    self._cleanup_bandits()
            except Exception as e:
                self.logger.error(f"Error updating bandit model for pattern {pattern}: {e}")
    def _cleanup_bandits(self) -> None:
        """Clean up bandits with little data to save memory."""
        try:
            patterns_to_remove = []
            
            for pattern, adjustments in self._bandits.items():
                # Check if this pattern has so little data it should be removed
                total_pulls = sum(a[0] + a[1] for a in adjustments.values())
                if total_pulls <= 2.0 * len(self.actions):  # Only default priors
                    patterns_to_remove.append(pattern)
            
            # Remove patterns with no significant data
            for pattern in patterns_to_remove:
                del self._bandits[pattern]
            
            if patterns_to_remove:
                self.logger.debug(f"Cleaned up {len(patterns_to_remove)} unused bandits")
        except Exception as e:
            self.logger.error(f"Error cleaning up bandits: {e}")
    
    def _calculate_bandit_values(self, pattern: str) -> Dict[int, float]:
        """
        Calculate bandit arm values using Thompson sampling.
        
        Args:
            pattern: Candlestick pattern name
        
        Returns:
            Dictionary mapping adjustment to value
        """
        try:
            values = {}
            
            # Handle case where pattern isn't in our bandits
            if pattern not in self._bandits:
                # Return default values for all actions, slightly favoring 0 (no change)
                return {a: 0.5 + (0.1 if a == 0 else 0.0) for a in self.actions}
            
            # Apply Thompson sampling
            for adjustment in self.actions:
                success, failure = self._bandits[pattern][adjustment]
                # Sample from Beta distribution
                value = random.betavariate(success, failure)
                values[adjustment] = value
            
            return values
        except Exception as e:
            self.logger.error(f"Error calculating bandit values for pattern {pattern}: {e}")
            # Default values in case of error
            return {a: 0.5 for a in self.actions}
    
    @log_execution_time()
    def get_difficulty_adjustment(self, pattern: str) -> int:
        """
        Get the recommended difficulty adjustment for a pattern.
        
        Args:
            pattern: Candlestick pattern name
            
        Returns:
            Selected difficulty adjustment
        """
        with self._lock:
            try:
                # Calculate current mastery for context
                mastery = 0.5  # Default
                if pattern in self.user_performance.pattern_metrics:
                    mastery = self.user_performance.pattern_metrics[pattern].get("mastery", 0.5)
                
                # Pure exploration with low probability
                if random.random() < self.explore_rate:
                    return random.choice(self.actions)
                
                # Get bandit values
                values = self._calculate_bandit_values(pattern)
                
                # Adjust values based on mastery level
                # For low mastery, prefer easier questions; for high mastery, prefer harder
                adjusted_values = {}
                for adj, value in values.items():
                    # Apply a mastery-based bias to the values
                    # Lower mastery = prefer negative adjustments
                    # Higher mastery = prefer positive adjustments
                    mastery_bias = (adj * (mastery - 0.5)) * 0.2
                    adjusted_values[adj] = value + mastery_bias
                
                # Select the best adjustment
                best_adjustment = max(adjusted_values.items(), key=lambda x: x[1])[0]
                
                # Get confidence level
                confidence = self._get_confidence_level(pattern, best_adjustment)
                self.logger.debug(
                    f"Bandit selecting adjustment {best_adjustment} for pattern '{pattern}' "
                    f"with confidence {confidence:.2f}"
                )
                
                return best_adjustment
            except Exception as e:
                self.logger.error(f"Error getting difficulty adjustment for pattern {pattern}: {e}")
                return 0  # Default to no adjustment in case of error
    
    def _get_confidence_level(self, pattern: str, adjustment: int) -> float:
        """
        Calculate confidence level in the selected adjustment.
        
        Args:
            pattern: Candlestick pattern name
            adjustment: Selected adjustment
        
        Returns:
            Confidence level (0.0 to 1.0)
        """
        try:
            if pattern not in self._bandits or adjustment not in self._bandits[pattern]:
                return 0.1  # Very low confidence if no data
            
            # Get total pulls for this arm
            success, failure = self._bandits[pattern][adjustment]
            total_pulls = success + failure - 2.0  # Subtract priors
            
            # Calculate confidence based on pulls
            # More pulls = more confidence, up to a saturation point
            confidence = min(1.0, total_pulls / self.min_pulls_before_confident)
            
            # Calculate expected value from Beta distribution parameters
            expected_value = success / (success + failure)
            
            # Adjust confidence based on how close expected value is to 0.5
            # Values closer to 0.5 have higher uncertainty
            certainty_factor = abs(expected_value - 0.5) * 2.0  # 0.0 to 1.0
            
            # Combine both factors with more weight on pull count
            combined_confidence = (confidence * 0.7) + (certainty_factor * 0.3)
            
            return combined_confidence
        except Exception as e:
            self.logger.error(f"Error calculating confidence for pattern {pattern}: {e}")
            return 0.1
    
    def get_best_arm_stats(self, pattern: str) -> Dict[str, Any]:
        """
        Get statistics about the best arm for a pattern.

        Args:
            pattern: Candlestick pattern name

        Returns:
            Statistics dictionary
        """
        with self._lock:
            try:
                if pattern not in self._bandits:
                    return {
                        "best_adjustment": 0,
                        "expected_value": 0.5,
                        "confidence": 0.0,
                        "total_pulls": 0
                    }

                # Calculate values for all arms
                values = self._calculate_bandit_values(pattern)

                # Find best arm
                best_adjustment = max(values.items(), key=lambda x: x[1])[0]
                success, failure = self._bandits[pattern][best_adjustment]

                # Calculate stats
                total_pulls = success + failure - 2.0  # Subtract priors
                expected_value = success / (success + failure)
                confidence = self._get_confidence_level(pattern, best_adjustment)

                return {
                    "best_adjustment": best_adjustment,
                    "expected_value": expected_value,
                    "confidence": confidence,
                    "total_pulls": total_pulls
                }
            except Exception as e:
                self.logger.error(f"Error getting best arm stats for pattern {pattern}: {e}")
                return {"error": str(e)}

    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get overall performance statistics.

        Returns:
            Statistics dictionary
        """
        with self._lock:
            try:
                # Calculate recent performance
                recent_success_rate = 0.0
                if self._performance_history:
                    recent_success_rate = sum(self._performance_history) / len(self._performance_history)

                # Count total arms
                total_arms = sum(len(arms) for arms in self._bandits.values())

                # Get pattern coverage
                pattern_count = len(self._bandits)

                return {
                    "total_pulls": self._total_pulls,
                    "recent_success_rate": recent_success_rate,
                    "total_arms": total_arms,
                    "pattern_count": pattern_count,
                    "explore_rate": self.explore_rate
                }
            except Exception as e:
                self.logger.error(f"Error getting performance stats: {e}")
                return {"error": str(e)}
    
    def serialize(self) -> Dict[str, Any]:
        """
        Serialize the bandit model for storage.
        
        Returns:
            Dictionary with serialized model data
        """
        with self._lock:
            try:
                # Convert nested defaultdicts to regular dicts
                serialized_bandits = {}
                for pattern, adjustments in self._bandits.items():
                    serialized_bandits[pattern] = {
                        str(adj): list(counts) for adj, counts in adjustments.items()
                    }
                
                return {
                    "bandits": serialized_bandits,
                    "total_pulls": self._total_pulls,
                    "explore_rate": self.explore_rate
                }
            except Exception as e:
                self.logger.error(f"Error serializing bandit model: {e}")
                return {}
    
    def deserialize(self, data: Dict[str, Any]) -> None:
        """
        Deserialize the bandit model from storage.
        
        Args:
            data: Dictionary with serialized model data
        """
        with self._lock:
            try:
                # Restore parameters
                self._total_pulls = data.get("total_pulls", 0)
                self.explore_rate = data.get("explore_rate", 0.1)
                
                # Restore bandits
                bandit_data = data.get("bandits", {})
                for pattern, adjustments in bandit_data.items():
                    for adj_str, counts in adjustments.items():
                        try:
                            adj = int(adj_str)
                            self._bandits[pattern][adj] = counts
                        except Exception as inner_e:
                            self.logger.error(f"Error deserializing arm {adj_str}: {inner_e}")
                
                self.logger.info(f"Deserialized bandit model with {len(self._bandits)} patterns")
            except Exception as e:
                self.logger.error(f"Error deserializing bandit model: {e}")

class AdaptiveDifficultyEngine:
    """
    Main engine for dynamic difficulty adjustment.
    
    Integrates multiple difficulty optimization strategies into a unified
    decision-making framework.
    
    Features:
    - Thread-safe operations for concurrent access
    - Multiple difficulty optimization strategies with weighted ensemble
    - Performance monitoring and analytics
    - Automated data persistence and recovery
    """
    
    def __init__(self, user_id: str):
        """
        Initialize the adaptive difficulty engine.
        
        Args:
            user_id: User identifier
        """
        self.user_id = user_id
        self.logger = app_logger.getChild(f"adaptive_difficulty.{user_id}")
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Initialize user performance tracker
        self.performance_tracker = UserPerformanceTracker(user_id)
        
        # Initialize forgetting curve manager
        self.forgetting_curve = ForgettingCurveManager(self.performance_tracker)
        
        # Add optimizers
        self.rl_optimizer = ReinforcementLearningDifficultyOptimizer(self.performance_tracker)
        self.bandit_optimizer = MultiBanditDifficultyOptimizer(self.performance_tracker)
        
        # Decision weights for ensemble approach
        self.decision_weights = {
            "forgetting_curve": 0.35,
            "rl": 0.30,
            "bandit": 0.35
        }
        
        # Configure base difficulty levels
        self.difficulty_levels = {
            QuestionDifficulty.VERY_EASY.value: -2,
            QuestionDifficulty.EASY.value: -1,
            QuestionDifficulty.MEDIUM.value: 0,
            QuestionDifficulty.HARD.value: 1,
            QuestionDifficulty.VERY_HARD.value: 2
        }
        
        # Analytics data
        self.analytics = {
            "decisions": [],
            "performance": []
        }
        
        # Persistence helper
        self._last_save_time = time.time()
        self._changes_since_save = 0
        self._save_threshold = 10  # Save after 10 changes
        
        self.logger.info(f"Initialized Adaptive Difficulty Engine for user {user_id}")
    
    @log_execution_time()
    def update_performance(self, pattern: str, difficulty: str, 
                         is_correct: bool, response_time: float) -> None:
        """
        Update performance metrics and optimization models.
        
        Args:
            pattern: Candlestick pattern
            difficulty: Question difficulty level
            is_correct: Whether the answer was correct
            response_time: Response time in seconds
        """
        with self._lock:
            try:
                difficulty_value = self.difficulty_levels.get(difficulty, 0)
                
                # Update user performance tracker
                self.performance_tracker.update_metrics(
                    pattern, is_correct, response_time, difficulty_value
                )
                
                # Calculate reward for optimization models
                confidence = self.performance_tracker.get_pattern_confidence(pattern)
                mastery = self.performance_tracker.get_pattern_mastery(pattern)
                
                # Calculate skill level for pattern
                skill_level = (mastery + confidence) / 2
                
                # Update forgetting curve
                self.forgetting_curve.update_due_factor(pattern, is_correct, response_time)
        
        # Update RL optimizer
                self.rl_optimizer.update(pattern, difficulty_value, is_correct, response_time)
        
        # Update Bandit optimizer
                self.bandit_optimizer.update(pattern, difficulty_value, is_correct)
                
                # Log update
                self.logger.info(
                    f"Updated performance for pattern '{pattern}', difficulty '{difficulty}', "
                    f"correct={is_correct}, mastery={mastery:.2f}, confidence={confidence:.2f}"
                )
                
                # Track analytics
                self._track_performance(pattern, difficulty, is_correct, response_time, 
                                     mastery, confidence)
                
                # Check if we should save state
                self._changes_since_save += 1
                if self._changes_since_save >= self._save_threshold or time.time() - self._last_save_time > 300:
                    self._save_state()
            except Exception as e:
                self.logger.error(f"Error updating performance: {e}", exc_info=True)
    
    def _track_performance(self, pattern: str, difficulty: str, is_correct: bool, 
                         response_time: float, mastery: float, confidence: float) -> None:
        """Record performance data for analytics."""
        try:
            timestamp = time.time()
            
            # Limit analytics data size
            if len(self.analytics["performance"]) > 1000:
                self.analytics["performance"] = self.analytics["performance"][-1000:]
            
            self.analytics["performance"].append({
                "timestamp": timestamp,
                "pattern": pattern,
                "difficulty": difficulty,
                "correct": is_correct,
                "response_time": response_time,
                "mastery": mastery,
                "confidence": confidence
            })
        except Exception as e:
            self.logger.error(f"Error tracking performance analytics: {e}")
    
    @log_execution_time()
    def get_recommended_difficulty(self, pattern: str) -> str:
        """
        Get the recommended difficulty level for a pattern.
        
        Args:
            pattern: Candlestick pattern
            
        Returns:
            Recommended difficulty level
        """
        with self._lock:
            try:
                # Ensure pattern exists in performance metrics
                if pattern not in self.performance_tracker.pattern_metrics:
                    self.performance_tracker.initialize_pattern(pattern)
                
                # Get current metrics
                mastery = self.performance_tracker.get_pattern_mastery(pattern)
                confidence = self.performance_tracker.get_pattern_confidence(pattern)
                
                # Get recommendations from each optimizer
                recommendations = {}
        
                # 1. Forgetting curve recommendation
                # Check if pattern is due for review
                due_patterns = self.forgetting_curve.get_due_patterns()
                due_score = 0
                for p, score in due_patterns:
                    if p == pattern:
                        due_score = score
                        break
                
                # Convert due score to a difficulty adjustment
                if due_score > 1.0:
                    # Significantly overdue - make it easier
                    forgetting_adj = -1
                elif due_score > 0.5:
                    # Moderately overdue - make it slightly easier
                    forgetting_adj = 0
                elif due_score > 0:
                    # Just becoming due - keep difficulty
                    forgetting_adj = 0
                else:
                    # Not due yet - can increase difficulty
                    forgetting_adj = 1
                
                recommendations["forgetting_curve"] = forgetting_adj
        
                # 2. RL-based recommendation
                rl_adj = self.rl_optimizer.get_difficulty_adjustment(pattern)
                recommendations["rl"] = rl_adj
        
                # 3. Bandit-based recommendation
                bandit_adj = self.bandit_optimizer.get_difficulty_adjustment(pattern)
                recommendations["bandit"] = bandit_adj
                
                # Calculate weighted ensemble recommendation
                weighted_adjustment = 0.0
                weight_sum = 0.0
                
                for optimizer, adjustment in recommendations.items():
                    weight = self.decision_weights.get(optimizer, 0.0)
                    weighted_adjustment += adjustment * weight
                    weight_sum += weight
                
                # Normalize if weights don't sum to 1.0
                if weight_sum > 0:
                    weighted_adjustment /= weight_sum
                
                # Round to nearest integer adjustment
                final_adjustment = round(weighted_adjustment)
                
                # Convert the adjustment to a difficulty level
                # Start from MEDIUM and adjust
                base_level = QuestionDifficulty.MEDIUM
                adjusted_value = base_level.to_numeric() + final_adjustment
                
                # Clamp to valid difficulty range
                adjusted_value = max(1, min(5, adjusted_value))
                final_difficulty = QuestionDifficulty.from_numeric(adjusted_value).value
                
                # Log the decision
                self.logger.info(
                    f"Recommending difficulty '{final_difficulty}' for pattern '{pattern}', "
                    f"mastery={mastery:.2f}, confidence={confidence:.2f}, "
                    f"adjustment={final_adjustment}"
                )
                
                # Track analytics
                self._track_decision(pattern, recommendations, final_adjustment, final_difficulty)
                
                return final_difficulty
            except Exception as e:
                self.logger.error(f"Error getting recommended difficulty: {e}", exc_info=True)
                # Default to medium difficulty in case of error
                return QuestionDifficulty.MEDIUM.value
    
    def _track_decision(self, pattern: str, recommendations: Dict[str, int], 
                      adjustment: int, difficulty: str) -> None:
        """Record decision data for analytics."""
        try:
            timestamp = time.time()
            
            # Limit analytics data size
            if len(self.analytics["decisions"]) > 1000:
                self.analytics["decisions"] = self.analytics["decisions"][-1000:]
            
            self.analytics["decisions"].append({
                "timestamp": timestamp,
                "pattern": pattern,
                "recommendations": dict(recommendations),
                "final_adjustment": adjustment,
                "final_difficulty": difficulty
            })
        except Exception as e:
            self.logger.error(f"Error tracking decision analytics: {e}")
    
    @log_execution_time()
    def get_optimal_pattern_batch(self, count: int = 5) -> List[str]:
        """
        Get an optimal batch of patterns for review.
        
        Args:
            count: Number of patterns to include
            
        Returns:
            List of pattern names
        """
        with self._lock:
            try:
                return self.forgetting_curve.get_optimal_batch(count)
            except Exception as e:
                self.logger.error(f"Error getting optimal pattern batch: {e}")
                # Fallback: return some common patterns
                common_patterns = [
                    "Doji", "Hammer", "Shooting Star", "Engulfing", "Harami",
                    "Morning Star", "Evening Star", "Three White Soldiers", "Three Black Crows"
                ]
                return common_patterns[:count]
    
    def get_performance_analytics(self) -> Dict[str, Any]:
        """
        Get comprehensive performance analytics.
        
        Returns:
            Dictionary with analytics data
        """
        with self._lock:
            try:
                # Get overall metrics
                overall_metrics = self.performance_tracker.get_overall_metrics()
                
                # Get optimizer-specific metrics
                rl_stats = self.rl_optimizer.get_performance_stats()
                bandit_stats = self.bandit_optimizer.get_performance_stats()
                
                # Calculate recent performance trends
                recent_performance = self.analytics["performance"][-50:] if self.analytics["performance"] else []
                
                recent_accuracy = 0.0
                if recent_performance:
                    correct_count = sum(1 for p in recent_performance if p["correct"])
                    recent_accuracy = correct_count / len(recent_performance)
                
                # Return combined analytics
                return {
                    "overall_metrics": overall_metrics,
                    "recent_accuracy": recent_accuracy,
                    "optimizer_stats": {
                        "reinforcement_learning": rl_stats,
                        "bandit": bandit_stats
                    },
                    "decision_weights": self.decision_weights,
                    "recent_decisions": self.analytics["decisions"][-10:] if self.analytics["decisions"] else []
                }
            except Exception as e:
                self.logger.error(f"Error getting performance analytics: {e}")
                return {"error": str(e)}
    
    def _save_state(self) -> None:
        """Save the current state to persistent storage."""
        try:
            # Log save operation
            self.logger.info(f"Saving adaptive difficulty state for user {self.user_id}")
            
            # Update timestamps
            self._last_save_time = time.time()
            self._changes_since_save = 0
            
            # In a real implementation, this would save to a database or file
            # For now, we'll just simulate the operation
            self.logger.info(f"State saved successfully. User has mastered " 
                          f"{self.performance_tracker.get_mastered_pattern_count()} patterns.")
        except Exception as e:
            self.logger.error(f"Error saving state: {e}")
    
    def load_state(self, state_data: Dict[str, Any]) -> bool:
        """
        Load state from persistent storage.
        
        Args:
            state_data: Dictionary with serialized state data
            
        Returns:
            Whether the load was successful
        """
        with self._lock:
            try:
                # Load performance tracker state
                if "performance_tracker" in state_data:
                    self.performance_tracker.deserialize(state_data["performance_tracker"])
                
                # Load optimizer states
                if "rl_optimizer" in state_data:
                    self.rl_optimizer.deserialize(state_data["rl_optimizer"])
                
                if "bandit_optimizer" in state_data:
                    self.bandit_optimizer.deserialize(state_data["bandit_optimizer"])
                
                # Load analytics
                if "analytics" in state_data:
                    self.analytics = state_data["analytics"]
                
                # Load decision weights
                if "decision_weights" in state_data:
                    self.decision_weights = state_data["decision_weights"]
                
                self.logger.info(f"Successfully loaded state for user {self.user_id}")
                return True
            except Exception as e:
                self.logger.error(f"Error loading state: {e}", exc_info=True)
                return False
    
    def serialize(self) -> Dict[str, Any]:
        """
        Serialize the engine state for storage.
            
        Returns:
            Dictionary with serialized state
        """
        with self._lock:
            try:
                return {
                    "user_id": self.user_id,
                    "performance_tracker": self.performance_tracker.serialize(),
                    "rl_optimizer": self.rl_optimizer.serialize(),
                    "bandit_optimizer": self.bandit_optimizer.serialize(),
                    "decision_weights": self.decision_weights,
                    "analytics": {
                        "decisions": self.analytics["decisions"][-100:],  # Limit size for storage
                        "performance": self.analytics["performance"][-100:]
                    },
                    "timestamp": time.time()
                }
            except Exception as e:
                self.logger.error(f"Error serializing engine state: {e}")
                return {"error": str(e)}


class ForgettingCurveModel:
    """
    Implements the underlying mathematical model for the forgetting curve.
    
    Provides utilities for calculating retention probabilities and 
    determining optimal review intervals.
    """
    
    def __init__(self):
        # Cache for expensive calculations
        self._retention_cache = {}
        self._interval_cache = {}
        
        # Constants for the forgetting curve formula
        self.STABILITY_FACTOR = 1.2
        self.TIME_FACTOR = 0.1
    
    @lru_cache(maxsize=100)
    def calculate_retention(self, memory_strength: float, forgetting_rate: float, 
                          elapsed_time: float) -> float:
        """
        Calculate retention probability using the forgetting curve model.
        
        Args:
            memory_strength: Strength of memory (0.0 to 1.0)
            forgetting_rate: Rate of forgetting (0.0 to 1.0)
            elapsed_time: Time since last review in hours
        
        Returns:
            Retention probability (0.0 to 1.0)
        """
        # Apply Ebbinghaus forgetting curve: R = e^(-t/S)
        # where R is retention, t is time, S is stability
        
        # Calculate stability from memory strength
        # Higher memory strength = higher stability = slower forgetting
        stability = (1 + (memory_strength * self.STABILITY_FACTOR)) / (forgetting_rate + self.TIME_FACTOR)
        
        # Calculate retention using the exponential forgetting function
        retention = math.exp(-elapsed_time / (stability * 24))  # Convert stability to hours
        
        # Ensure retention is within valid range
        return max(0.0, min(1.0, retention))
    
    @lru_cache(maxsize=100)
    def calculate_optimal_interval(self, memory_strength: float, forgetting_rate: float, 
                                target_retention: float = 0.7) -> float:
        """
        Calculate optimal review interval to achieve target retention.
        
        Args:
            memory_strength: Strength of memory (0.0 to 1.0)
            forgetting_rate: Rate of forgetting (0.0 to 1.0)
            target_retention: Target retention probability (0.0 to 1.0)
            
        Returns:
            Optimal interval in hours
        """
        # Solve for t in R = e^(-t/S) given R (target retention)
        # t = -S * ln(R)
        
        # Calculate stability from memory strength
        stability = (1 + (memory_strength * self.STABILITY_FACTOR)) / (forgetting_rate + self.TIME_FACTOR)
        
        # Convert stability to hours and calculate interval
        stability_hours = stability * 24
        
        # Avoid taking log of zero
        safe_retention = max(0.01, min(0.99, target_retention))
        optimal_interval = -stability_hours * math.log(safe_retention)
        
        # Ensure interval is reasonable (at least 1 hour, at most 180 days)
        return max(1.0, min(4320.0, optimal_interval))


def get_engine_for_user(user_id: str) -> AdaptiveDifficultyEngine:
    """
    Get an adaptive difficulty engine instance for a user.
    
    This function caches engines to avoid recreating them unnecessarily.
    
    Args:
        user_id: User identifier
        
        Returns:
        AdaptiveDifficultyEngine instance
    """
    # Use a thread-safe cache for engine instances
    if not hasattr(get_engine_for_user, "_engine_cache"):
        get_engine_for_user._engine_cache = {}
        get_engine_for_user._cache_lock = threading.RLock()
    
    with get_engine_for_user._cache_lock:
        # Check if engine exists in cache
        if user_id in get_engine_for_user._engine_cache:
            return get_engine_for_user._engine_cache[user_id]
        
        # Create new engine
        engine = AdaptiveDifficultyEngine(user_id)
        
        # Load persisted state if available
        try:
            # In a real implementation, this would load from a database or file
            # Here we just initialize a new engine
            pass
        except Exception as e:
            logger.error(f"Failed to load state for user {user_id}: {e}")
        
        # Store in cache
        get_engine_for_user._engine_cache[user_id] = engine
        
        # Implement cache size management
        if len(get_engine_for_user._engine_cache) > 1000:
            # Remove 20% of least recently used engines
            # In a real implementation, this would be more sophisticated
            # Here we just remove a random 20%
            keys_to_remove = random.sample(list(get_engine_for_user._engine_cache.keys()), 
                                        k=int(len(get_engine_for_user._engine_cache) * 0.2))
            for key in keys_to_remove:
                if key != user_id:  # Don't remove the one we just added
                    try:
                        # Save state before removing
                        engine = get_engine_for_user._engine_cache[key]
                        engine._save_state()
                        del get_engine_for_user._engine_cache[key]
                    except Exception:
                        # Just remove it if saving fails
                        del get_engine_for_user._engine_cache[key]
        
        return engine


def cleanup_engines() -> None:
    """Clean up and save all engine states."""
    if hasattr(get_engine_for_user, "_engine_cache"):
        with get_engine_for_user._cache_lock:
            for user_id, engine in get_engine_for_user._engine_cache.items():
                try:
                    engine._save_state()
                except Exception as e:
                    logger.error(f"Failed to save state for user {user_id}: {e}")
                    
            # Clear cache
            get_engine_for_user._engine_cache.clear()


# Set up automatic cleanup on module exit
import atexit
atexit.register(cleanup_engines)