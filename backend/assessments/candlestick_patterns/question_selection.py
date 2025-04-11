"""
Question Selection Algorithms for Candlestick Pattern Assessments

This module provides:
1. Enhanced implementations of probabilistic data structures (Bloom Filter, Count-Min Sketch)
2. Optimized question selection algorithms that avoid repetition
3. User history-aware selection mechanisms
"""

from __future__ import annotations

import hashlib
import math
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set, Union, ClassVar, Final, TypeVar, Generic
from collections import defaultdict

import numpy as np
import mmh3  # MurmurHash3 for better hash distributions

from backend.common.logger import app_logger, log_execution_time
from backend.common.cache import cache, async_cached

# Module logger
logger = app_logger.getChild("question_selection")

# Type aliases for better code readability
QuestionId = str
UserId = str
Timestamp = datetime
Difficulty = float

# Module constants
DEFAULT_USER_CAPACITY: Final[int] = 1000
DEFAULT_QUESTION_CAPACITY: Final[int] = 5000
DEFAULT_BLOOM_FILTER_CAPACITY: Final[int] = 10000
DEFAULT_BLOOM_FILTER_FALSE_POSITIVE_RATE: Final[float] = 0.01
DEFAULT_SKETCH_WIDTH: Final[int] = 1000
DEFAULT_SKETCH_DEPTH: Final[int] = 5
DEFAULT_DECAY_DAYS: Final[int] = 30

class EnhancedBloomFilter:
    """
    Enhanced Bloom Filter implementation for question uniqueness checking.
    
    Features:
    1. Multiple hash functions for better distribution
    2. Capacity-aware design with false positive rate estimation
    3. Time-aware decay for question reuse after sufficient time
    """
    
    def __init__(
        self, 
        capacity: int = DEFAULT_BLOOM_FILTER_CAPACITY, 
        false_positive_rate: float = DEFAULT_BLOOM_FILTER_FALSE_POSITIVE_RATE,
        max_timestamps: int = 10000  # Limit the number of timestamps to store
    ):
        """
        Initialize the Bloom Filter with optimal parameters.
        
        Args:
            capacity: Expected number of items to be stored
            false_positive_rate: Target false positive rate (0.0-1.0)
            max_timestamps: Maximum number of timestamps to store
        """
        if capacity <= 0:
            raise ValueError("Capacity must be positive")
        if not 0 < false_positive_rate < 1:
            raise ValueError("False positive rate must be between 0 and 1")
            
        # Calculate optimal filter size and hash count
        self.capacity: int = capacity
        self.false_positive_rate: float = false_positive_rate
        self.max_timestamps: int = max_timestamps
        
        # Optimal bit array size formula: m = -n * ln(p) / (ln(2)^2)
        # where n is capacity and p is false positive rate
        self.size: int = max(
            8,  # Minimum size to avoid edge cases
            int(-capacity * math.log(false_positive_rate) / (math.log(2) ** 2))
        )
        
        # Optimal number of hash functions: k = m/n * ln(2)
        self.hash_count: int = max(
            1,  # At least one hash function
            int(math.ceil((self.size / capacity) * math.log(2)))
        )
        
        # Initialize bit array (use bytearray for memory efficiency)
        self.bit_array: List[bool] = [False] * self.size
        
        # Keep track of insertion timestamps for decay
        self.insertion_times: Dict[str, Timestamp] = {}
        
        logger.info(
            f"Initialized Bloom Filter with size {self.size} bits, "
            f"{self.hash_count} hash functions, target FPR {false_positive_rate:.4f}, "
            f"max_timestamps={max_timestamps}"
        )
    
    def _get_hash_positions(self, item: str) -> List[int]:
        """
        Get hash positions for an item using multiple hash functions.
        
        Args:
            item: Item to hash
            
        Returns:
            List of bit positions in the bit array
        """
        if not item:
            raise ValueError("Item cannot be empty")
            
        # For efficiency, compute two base hashes and derive others
        # This is the double hashing technique used in many Bloom filter implementations
        h1 = mmh3.hash(item, 0) % self.size
        h2 = mmh3.hash(item, 1) % self.size
        
        # First two positions come from the base hashes
        positions = [h1, h2]
        
        # Generate remaining positions using linear combination
        for i in range(2, self.hash_count):
            # Generate a new hash value by linear combination of h1 and h2
            # This avoids computing more expensive hash functions
            pos = (h1 + i * h2) % self.size
            positions.append(pos)
            
        return positions
    
    def add(self, item: str, user_id: Optional[UserId] = None) -> None:
        """
        Add an item to the filter.
        
        Args:
            item: Item to add
            user_id: Optional user ID for user-specific filtering
        """
        if not item:
            logger.warning("Attempted to add empty item to BloomFilter")
            return
            
        key = f"{user_id}:{item}" if user_id else item
        
        # Set bits for all hash positions
        try:
            for pos in self._get_hash_positions(key):
                self.bit_array[pos] = True
        except Exception as e:
            logger.error(f"Error adding item to BloomFilter: {str(e)}")
            return
        
        # Record insertion time, with capped storage
        current_time = datetime.now()
        
        # If we've reached max timestamps, remove some old ones first
        if len(self.insertion_times) >= self.max_timestamps:
            # Remove oldest 10% of timestamps to avoid frequent evictions
            self._evict_oldest(0.1)
            
        self.insertion_times[key] = current_time
    
    def check(self, item: str, user_id: Optional[UserId] = None) -> bool:
        """
        Check if an item might be in the filter.
        
        Args:
            item: Item to check
            user_id: Optional user ID for user-specific filtering
            
        Returns:
            True if the item might be in the filter, False if definitely not
        """
        if not item:
            return False
            
        key = f"{user_id}:{item}" if user_id else item
        
        try:
            # Check all hash positions - if any bit is 0, the item is definitely not in the set
            for pos in self._get_hash_positions(key):
                if not self.bit_array[pos]:
                    return False
            return True
        except Exception as e:
            logger.error(f"Error checking item in BloomFilter: {str(e)}")
            return False  # Fail safe by assuming item is not in the filter
    
    def forget_items_older_than(self, days: int = DEFAULT_DECAY_DAYS) -> int:
        """
        Remove items older than the specified number of days.
        
        Args:
            days: Number of days to consider for decay
            
        Returns:
            Number of items forgotten
        """
        if days <= 0:
            logger.warning(f"Invalid decay days value: {days}, using default {DEFAULT_DECAY_DAYS}")
            days = DEFAULT_DECAY_DAYS
            
        threshold = datetime.now() - timedelta(days=days)
        keys_to_remove = []
        
        # Identify keys to remove
        for key, insert_time in self.insertion_times.items():
            if insert_time < threshold:
                keys_to_remove.append(key)
        
        if not keys_to_remove:
            return 0

        # Create a map of bit positions to keys that set those bits
        bit_to_keys = defaultdict(set)
        
        # For all keys not being removed, record which bits they set
        for key in self.insertion_times:
            if key not in keys_to_remove:
                for pos in self._get_hash_positions(key):
                    bit_to_keys[pos].add(key)
        
        # For each key to remove, reset bits that aren't used by other keys
        for key in keys_to_remove:
            for pos in self._get_hash_positions(key):
                if pos not in bit_to_keys or len(bit_to_keys[pos]) == 0:
                    self.bit_array[pos] = False
            
            # Remove from insertion times
            del self.insertion_times[key]
        
        logger.info(f"Forgotten {len(keys_to_remove)} items older than {days} days")
        return len(keys_to_remove)
    
    def _evict_oldest(self, percentage: float = 0.1) -> None:
        """
        Evict the oldest items from the filter.
        
        Args:
            percentage: Percentage of oldest items to evict (0.0-1.0)
        """
        count = len(self.insertion_times)
        target = max(1, int(count * percentage))  # Evict at least one item
        
        if count <= 0:
            return
        
        # Sort by insertion time
        sorted_items = sorted(self.insertion_times.items(), key=lambda x: x[1])
        keys_to_remove = [key for key, _ in sorted_items[:target]]
        
        # Create a map of bit positions to keys that set those bits (excluding keys to remove)
        bit_to_keys = defaultdict(set)
        for key in self.insertion_times:
            if key not in keys_to_remove:
                for pos in self._get_hash_positions(key):
                    bit_to_keys[pos].add(key)
        
        # For each key to remove, reset bits that aren't used by other keys
        for key in keys_to_remove:
            for pos in self._get_hash_positions(key):
                if pos not in bit_to_keys or len(bit_to_keys[pos]) == 0:
                    self.bit_array[pos] = False
            
            # Remove from insertion times
            del self.insertion_times[key]
        
        logger.debug(f"Evicted {len(keys_to_remove)} oldest items from BloomFilter")
    
    def reset(self) -> None:
        """Reset the filter, removing all items."""
        self.bit_array = [False] * self.size
        self.insertion_times = {}
        logger.info("Reset BloomFilter to empty state")
    
    def estimate_current_false_positive_rate(self) -> float:
        """
        Estimate the current false positive rate based on fill ratio.
        
        Returns:
            Estimated false positive rate (0.0-1.0)
        """
        # Count filled bits
        filled_bits = sum(1 for bit in self.bit_array if bit)
        fill_ratio = filled_bits / self.size
        
        # Formula: (1 - e^(-k * n / m))^k
        # where k is hash count, n is item count, m is bit array size
        estimated_fpr = (1 - math.exp(-self.hash_count * len(self.insertion_times) / self.size)) ** self.hash_count
        
        return min(1.0, max(0.0, estimated_fpr))  # Ensure result is between 0 and 1


class CountMinSketch:
    """
    Count-Min Sketch implementation for frequency estimation.
    
    Features:
    1. Efficient frequency estimation with controlled error
    2. Support for time decay to favor recent items
    3. Memory-efficient alternative to exact counters
    """
    
    def __init__(
        self, 
        width: int = DEFAULT_SKETCH_WIDTH, 
        depth: int = DEFAULT_SKETCH_DEPTH,
        decay_period_days: int = 7
    ):
        """
        Initialize the Count-Min Sketch.
        
        Args:
            width: Width of the sketch (number of buckets per hash function)
            depth: Depth of the sketch (number of hash functions)
            decay_period_days: Period in days after which decay is applied
        """
        if width <= 0:
            raise ValueError("Width must be positive")
        if depth <= 0:
            raise ValueError("Depth must be positive")
        if decay_period_days <= 0:
            raise ValueError("Decay period must be positive")
            
        self.width: int = width
        self.depth: int = depth
        self.sketch: np.ndarray = np.zeros((depth, width), dtype=np.int32)
        
        # Use fixed seeds for predictable hashing
        # This makes the behavior consistent across restarts
        self.hash_seeds: List[int] = [
            0x123456, 0xabcdef, 0xdeadbeef, 0xfeedface, 0x7654321,
            0x55555555, 0x33333333, 0x77777777, 0x99999999, 0xaaaaaaaa
        ][:depth]  # Use at most depth seeds
        
        # If we need more seeds than provided, generate them deterministically
        if len(self.hash_seeds) < depth:
            for i in range(len(self.hash_seeds), depth):
                seed = (self.hash_seeds[0] * i + self.hash_seeds[i % len(self.hash_seeds)]) & 0x7fffffff
                self.hash_seeds.append(seed)
                
        # Time decay parameters
        self.last_decay: Timestamp = datetime.now()
        self.decay_factor: float = 0.5  # Halve counts after decay_period
        self.decay_period: timedelta = timedelta(days=decay_period_days)
        
        # Statistics
        self.total_items_added: int = 0
        self.last_error_estimate: float = 0.0
        
        logger.info(
            f"Initialized Count-Min Sketch with width={width}, depth={depth}, "
            f"decay_period={decay_period_days} days"
        )
    
    def _get_hash_positions(self, item: str) -> List[int]:
        """
        Get hash positions for an item using multiple hash functions.
        
        Args:
            item: Item to hash
            
        Returns:
            List of bucket positions in the sketch
        """
        if not item:
            raise ValueError("Item cannot be empty")
            
        # Generate hash positions using MurmurHash3 with different seeds
        try:
            positions = [mmh3.hash(item, seed) % self.width for seed in self.hash_seeds]
            return positions
        except Exception as e:
            logger.error(f"Error generating hash positions: {str(e)}")
            # Return default positions in case of error
            return [i % self.width for i in range(self.depth)]
    
    def add(self, item: str, count: int = 1) -> None:
        """
        Add an item to the sketch.
        
        Args:
            item: Item to add
            count: Count to add
        """
        if not item:
            logger.warning("Attempted to add empty item to CountMinSketch")
            return
            
        if count <= 0:
            return  # Nothing to add
            
        try:
            # Check if decay should be applied
            current_time = datetime.now()
            if current_time - self.last_decay > self.decay_period:
                self._apply_decay()
                self.last_decay = current_time
            
            # Update counts for all hash positions
            positions = self._get_hash_positions(item)
            for i, pos in enumerate(positions):
                self.sketch[i, pos] += count
                
            self.total_items_added += 1
        except Exception as e:
            logger.error(f"Error adding item to CountMinSketch: {str(e)}")
    
    def estimate_count(self, item: str) -> int:
        """
        Estimate the frequency of an item.
        
        Args:
            item: Item to estimate frequency for
            
        Returns:
            Estimated frequency (non-negative integer)
        """
        if not item:
            return 0
            
        try:
            positions = self._get_hash_positions(item)
            if not positions:
                return 0
                
            # Return the minimum count across all hash functions
            # This is the standard Count-Min Sketch estimator
            return max(0, min(self.sketch[i, pos] for i, pos in enumerate(positions)))
        except Exception as e:
            logger.error(f"Error estimating count: {str(e)}")
            return 0
            
    def estimate_error(self) -> float:
        """
        Estimate the error in frequency estimates.
        
        The error is approximately (2 * e) / w where e is the total number of items 
        and w is the width of the sketch.
        
        Returns:
            Estimated error
        """
        # Calculate total count
        total_count = np.sum(self.sketch) / self.depth
        
        # Standard error formula
        self.last_error_estimate = (2.0 * total_count) / self.width
        return self.last_error_estimate
    
    def _apply_decay(self) -> None:
        """Apply time decay to all counts in the sketch."""
        try:
            # Multiply all values by decay factor and floor to integers
            self.sketch = np.floor(self.sketch * self.decay_factor).astype(np.int32)
            logger.debug(f"Applied decay with factor {self.decay_factor}")
        except Exception as e:
            logger.error(f"Error applying decay: {str(e)}")
    
    def merge(self, other: 'CountMinSketch') -> None:
        """
        Merge another sketch into this one.
        
        Args:
            other: Another CountMinSketch instance to merge
        """
        if not isinstance(other, CountMinSketch):
            raise TypeError("Can only merge with another CountMinSketch")
            
        if self.width != other.width or self.depth != other.depth:
            raise ValueError("Cannot merge sketches of different dimensions")
            
        try:
            # Element-wise maximum preserves the Count-Min property
            self.sketch = np.maximum(self.sketch, other.sketch)
            self.total_items_added += other.total_items_added
            logger.debug(f"Merged with another CountMinSketch containing {other.total_items_added} items")
        except Exception as e:
            logger.error(f"Error merging sketches: {str(e)}")
    
    def reset(self) -> None:
        """Reset the sketch to its initial state."""
        try:
            self.sketch = np.zeros((self.depth, self.width), dtype=np.int32)
            self.last_decay = datetime.now()
            self.total_items_added = 0
            logger.info("Reset CountMinSketch to empty state")
        except Exception as e:
            logger.error(f"Error resetting sketch: {str(e)}")


class QuestionSelectionAlgorithm:
    """
    Advanced question selection algorithm with uniqueness guarantees.
    
    Features:
    1. Uses Bloom Filter to avoid repetition
    2. Uses Count-Min Sketch to track question frequency
    3. Adjusts selection based on user history
    4. Supports question decaying to allow reuse after sufficient time
    """
    
    def __init__(
        self, 
        user_capacity: int = DEFAULT_USER_CAPACITY, 
        question_capacity: int = DEFAULT_QUESTION_CAPACITY
    ):
        """
        Initialize the question selection algorithm.
        
        Args:
            user_capacity: Expected number of users
            question_capacity: Expected number of questions
        """
        if user_capacity <= 0:
            raise ValueError("User capacity must be positive")
        if question_capacity <= 0:
            raise ValueError("Question capacity must be positive")
            
        # Initialize Bloom Filter for uniqueness checking
        self.bloom_filter = EnhancedBloomFilter(capacity=user_capacity * 100)
        
        # Initialize Count-Min Sketch for frequency tracking
        self.count_min_sketch = CountMinSketch(width=question_capacity * 10)
        
        # User history tracking with timestamps
        self.user_history: Dict[UserId, Dict[QuestionId, Timestamp]] = {}
        
        # Question pool management
        self.all_question_ids: Set[QuestionId] = set()
        self.question_difficulty: Dict[QuestionId, Difficulty] = {}
        
        # Question tags for better filtering
        self.question_tags: Dict[QuestionId, Set[str]] = {}
        
        # Statistics
        self.selections_count: int = 0
        self.cache_hits: int = 0
        
        logger.info(
            f"Initialized QuestionSelectionAlgorithm with "
            f"user_capacity={user_capacity}, question_capacity={question_capacity}"
        )
    
    def register_question(
        self, 
        question_id: QuestionId, 
        difficulty: Difficulty,
        tags: Optional[List[str]] = None
    ) -> None:
        """
        Register a question in the pool.
        
        Args:
            question_id: Question identifier
            difficulty: Question difficulty (0.0-1.0)
            tags: Optional list of tags for the question
        """
        if not question_id:
            logger.warning("Attempted to register question with empty ID")
            return
            
        # Validate difficulty
        difficulty = min(1.0, max(0.0, difficulty))
        
        try:
            self.all_question_ids.add(question_id)
            self.question_difficulty[question_id] = difficulty
            
            # Store tags if provided
            if tags:
                self.question_tags[question_id] = set(tags)
        except Exception as e:
            logger.error(f"Error registering question {question_id}: {str(e)}")
    
    def record_question_seen(
        self, 
        question_id: QuestionId, 
        user_id: UserId
    ) -> None:
        """
        Record that a user has seen a question.
        
        Args:
            question_id: Question identifier
            user_id: User identifier
        """
        if not question_id or not user_id:
            logger.warning("Attempted to record question seen with empty ID")
            return
            
        try:
            # Add to Bloom Filter for fast lookups
            key = f"{user_id}:{question_id}"
            self.bloom_filter.add(key)
            
            # Update Count-Min Sketch for frequency tracking
            self.count_min_sketch.add(question_id)
            
            # Update user history with timestamp
            if user_id not in self.user_history:
                self.user_history[user_id] = {}
            self.user_history[user_id][question_id] = datetime.now()
        except Exception as e:
            logger.error(f"Error recording question seen: {str(e)}")
    
    def has_user_seen_question(
        self, 
        question_id: QuestionId, 
        user_id: UserId
    ) -> bool:
        """
        Check if a user has seen a question.
        
        Args:
            question_id: Question identifier
            user_id: User identifier
            
        Returns:
            True if the user has likely seen the question, False otherwise
        """
        if not question_id or not user_id:
            return False
            
        try:
            # Check Bloom Filter first (fast negative responses)
            key = f"{user_id}:{question_id}"
            if not self.bloom_filter.check(key):
                return False
                
            # Double-check with exact history if available
            if user_id in self.user_history and question_id in self.user_history[user_id]:
                return True
                
            # Bloom filter might have false positives, but that's acceptable
            # for this use case (slightly higher chance of not showing a question)
            return True
        except Exception as e:
            logger.error(f"Error checking if user has seen question: {str(e)}")
            return False  # Conservative approach
    
    def forget_old_questions(self, days: int = DEFAULT_DECAY_DAYS) -> int:
        """
        Allow questions older than the specified number of days to be reused.
        
        Args:
            days: Number of days to consider for decay
            
        Returns:
            Number of question records forgotten
        """
        if days <= 0:
            logger.warning(f"Invalid decay days: {days}, using default {DEFAULT_DECAY_DAYS}")
            days = DEFAULT_DECAY_DAYS
            
        try:
            # Apply bloom filter decay
            forgotten_count = self.bloom_filter.forget_items_older_than(days)
            
            # Clean up user history
            threshold = datetime.now() - timedelta(days=days)
            total_forgotten = 0
            
            for user_id in self.user_history:
                before_count = len(self.user_history[user_id])
                self.user_history[user_id] = {
                    q_id: ts for q_id, ts in self.user_history[user_id].items()
                    if ts >= threshold
                }
                total_forgotten += before_count - len(self.user_history[user_id])
                
            logger.info(f"Forgot {total_forgotten} old question records older than {days} days")
            return total_forgotten
        except Exception as e:
            logger.error(f"Error forgetting old questions: {str(e)}")
            return 0
    
    def get_question_frequency(self, question_id: QuestionId) -> int:
        """
        Get the estimated frequency of a question being shown.
        
        Args:
            question_id: Question identifier
            
        Returns:
            Estimated frequency
        """
        if not question_id:
            return 0
            
        return self.count_min_sketch.estimate_count(question_id)
    
    def get_user_question_history(
        self, 
        user_id: UserId, 
        limit: int = 100
    ) -> List[Tuple[QuestionId, Timestamp]]:
        """
        Get a user's question history, sorted by recency.
        
        Args:
            user_id: User identifier
            limit: Maximum number of records to return
            
        Returns:
            List of (question_id, timestamp) tuples, most recent first
        """
        if not user_id or user_id not in self.user_history:
            return []
            
        try:
            # Sort by timestamp (newest first)
            history = sorted(
                self.user_history[user_id].items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            return history[:limit]
        except Exception as e:
            logger.error(f"Error getting user question history: {str(e)}")
            return []
    
    @log_execution_time()
    def select_questions(
        self,
        user_id: UserId,
        count: int,
        difficulty_range: Tuple[Difficulty, Difficulty] = (0.0, 1.0),
        required_tags: Optional[List[str]] = None,
        excluded_question_ids: Optional[List[QuestionId]] = None
    ) -> List[QuestionId]:
        """
        Select questions for a user based on various criteria.
        
        Args:
            user_id: User identifier
            count: Number of questions to select
            difficulty_range: Range of acceptable difficulties (min, max)
            required_tags: Optional tags that questions must have
            excluded_question_ids: Optional question IDs to exclude
            
        Returns:
            List of selected question IDs
        """
        if count <= 0:
            logger.warning(f"Invalid count: {count}, returning empty list")
            return []
            
        try:
            # Apply regular maintenance (every ~100 selections)
            if self.selections_count % 100 == 0:
                self.forget_old_questions()
            
            self.selections_count += 1
            
            # Prepare exclusion set
            excluded_ids = set(excluded_question_ids or [])
            
            # Get valid difficulty range
            min_diff, max_diff = max(0.0, min(difficulty_range)), min(1.0, max(difficulty_range))
            
            # Filter by tags if required
            tag_filtered_questions = self._filter_by_tags(required_tags) if required_tags else self.all_question_ids
            
            # Get candidate question IDs within difficulty range and not seen by user
            candidates = self._get_candidates(user_id, tag_filtered_questions, min_diff, max_diff, excluded_ids)
            
            # Handle insufficient candidates
            if len(candidates) < count:
                candidates = self._expand_candidates(
                    user_id, 
                    candidates, 
                    count, 
                    min_diff, 
                    max_diff, 
                    excluded_ids,
                    tag_filtered_questions
                )
            
            # If we still don't have enough, just return what we have
            if len(candidates) < count:
                logger.warning(
                    f"Could not find {count} suitable questions for user {user_id}, "
                    f"returning {len(candidates)} questions"
                )
            
            # Shuffle to avoid predictable patterns
            random.shuffle(candidates)
            
            # Select the desired number of questions
            selected = candidates[:min(count, len(candidates))]
            
            # Record selected questions
            for q_id in selected:
                self.record_question_seen(q_id, user_id)
            
            return selected
        except Exception as e:
            logger.error(f"Error selecting questions: {str(e)}")
            # Fallback: return any questions that match criteria
            fallback = []
            for q_id in self.all_question_ids:
                if q_id not in excluded_ids and len(fallback) < count:
                    fallback.append(q_id)
            return fallback
    
    def _filter_by_tags(self, required_tags: List[str]) -> Set[QuestionId]:
        """
        Filter questions by tags.
        
        Args:
            required_tags: List of required tags
            
        Returns:
            Set of question IDs that have the required tags
        """
        if not required_tags:
            return self.all_question_ids
            
        try:
            # Convert to set for faster lookups
            required_tag_set = set(required_tags)
            
            # Find questions that have all required tags
            result = set()
            for q_id, tags in self.question_tags.items():
                if tags.issuperset(required_tag_set):
                    result.add(q_id)
                    
            return result
        except Exception as e:
            logger.error(f"Error filtering by tags: {str(e)}")
            return self.all_question_ids
    
    def _get_candidates(
        self,
        user_id: UserId,
        question_pool: Set[QuestionId],
        min_diff: Difficulty,
        max_diff: Difficulty,
        excluded_ids: Set[QuestionId]
    ) -> List[QuestionId]:
        """
        Get candidate questions for selection.
        
        Args:
            user_id: User identifier
            question_pool: Pool of questions to choose from
            min_diff: Minimum difficulty
            max_diff: Maximum difficulty
            excluded_ids: Set of question IDs to exclude
            
        Returns:
            List of candidate question IDs
        """
        # First, collect the questions that match basic criteria
        basic_candidates = [
            q_id for q_id in question_pool
            if (q_id not in excluded_ids and
                min_diff <= self.question_difficulty.get(q_id, 0.5) <= max_diff and
                not self.has_user_seen_question(q_id, user_id))
        ]
        
        return basic_candidates
    
    def _expand_candidates(
        self,
        user_id: UserId,
        current_candidates: List[QuestionId],
        target_count: int,
        min_diff: Difficulty,
        max_diff: Difficulty,
        excluded_ids: Set[QuestionId],
        tag_filtered_questions: Set[QuestionId]
    ) -> List[QuestionId]:
        """
        Expand candidates when not enough are available.
        
        This method implements several strategies to find more candidates:
        1. Include previously seen questions, oldest first
        2. Widen the difficulty range
        3. Include any valid questions from the pool
        
        Args:
            user_id: User identifier
            current_candidates: Current list of candidates
            target_count: Desired number of candidates
            min_diff: Minimum difficulty
            max_diff: Maximum difficulty
            excluded_ids: Set of question IDs to exclude
            tag_filtered_questions: Questions filtered by tags
            
        Returns:
            Expanded list of candidates
        """
        # Make a copy so we don't modify the original
        candidates = current_candidates.copy()
        
        # If we don't have enough candidates, include some previously seen questions
        # Choose the oldest ones first
        if len(candidates) < target_count and user_id in self.user_history:
            seen_questions = sorted(
                self.user_history[user_id].items(),
                key=lambda x: x[1]  # Sort by timestamp (oldest first)
            )
            
            for q_id, _ in seen_questions:
                if (q_id not in excluded_ids and
                    q_id in tag_filtered_questions and
                    min_diff <= self.question_difficulty.get(q_id, 0.5) <= max_diff and
                    q_id not in candidates and
                    len(candidates) < target_count):
                    candidates.append(q_id)
        
        # If we still don't have enough candidates, widen the difficulty range
        if len(candidates) < target_count:
            # Increase difficulty range by 20% in both directions
            range_width = max_diff - min_diff
            expanded_min = max(0.0, min_diff - range_width * 0.2)
            expanded_max = min(1.0, max_diff + range_width * 0.2)
            
            for q_id in tag_filtered_questions:
                diff = self.question_difficulty.get(q_id, 0.5)
                if (q_id not in excluded_ids and
                    q_id not in candidates and
                    expanded_min <= diff <= expanded_max):
                    candidates.append(q_id)
                    if len(candidates) >= target_count:
                        break
        
        # If we still don't have enough, include any valid questions
        if len(candidates) < target_count:
            for q_id in tag_filtered_questions:
                if q_id not in excluded_ids and q_id not in candidates:
                    candidates.append(q_id)
                    if len(candidates) >= target_count:
                        break
                        
        return candidates 