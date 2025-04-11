"""
Forgetting Curve Model

This module implements the Ebbinghaus forgetting curve model and spaced repetition
scheduling for optimizing long-term memory retention of learned material.
"""

import math
import enum
import random
import datetime
from typing import Dict, List, Optional, Any, Tuple, Union

from backend.common.logger import app_logger

# Module logger
logger = app_logger.getChild("performance.forgetting")


class MemoryState(enum.Enum):
    """Memory states for learned material based on retention strength."""
    NEW = "new"                    # Never seen or remembered
    LEARNING = "learning"          # In the initial learning phase
    REVIEWING = "reviewing"        # Being reviewed periodically
    MASTERED = "mastered"          # Well retained with long review intervals
    FORGOTTEN = "forgotten"        # Previously known but now forgotten
    
    @classmethod
    def from_retention(cls, retention: float) -> 'MemoryState':
        """
        Convert retention probability to memory state.
        
        Args:
            retention: Retention probability (0-1)
            
        Returns:
            Corresponding memory state
        """
        if retention < 0.2:
            return cls.FORGOTTEN
        elif retention < 0.5:
            return cls.LEARNING
        elif retention < 0.8:
            return cls.REVIEWING
        else:
            return cls.MASTERED


class ForgettingCurveModel:
    """
    Implementation of the Ebbinghaus forgetting curve model.
    
    Predicts memory retention over time and determines optimal
    review intervals based on personal learning parameters.
    """
    
    def __init__(
        self, 
        initial_strength: float = 0.3,              # Initial memory strength
        minimum_strength: float = 0.2,              # Minimum memory strength
        strength_increase: float = 0.1,             # Strength increase per successful review
        strength_penalty: float = 0.15,             # Strength decrease on forgetting
        base_decay_rate: float = 0.1,               # Base decay rate
        decay_reduction_per_review: float = 0.02,   # How much decay slows per review
        minimum_decay_rate: float = 0.015           # Slowest possible decay rate
    ):
        """
        Initialize the forgetting curve model.
        
        Args:
            initial_strength: Initial memory strength for new items
            minimum_strength: Minimum memory strength
            strength_increase: How much strength increases per successful review
            strength_penalty: How much strength decreases when item is forgotten
            base_decay_rate: Base decay rate of memory
            decay_reduction_per_review: How much decay slows after each successful review
            minimum_decay_rate: Minimum decay rate (maximum memory persistence)
        """
        self.initial_strength = initial_strength
        self.minimum_strength = minimum_strength
        self.strength_increase = strength_increase
        self.strength_penalty = strength_penalty
        self.base_decay_rate = base_decay_rate
        self.decay_reduction_per_review = decay_reduction_per_review
        self.minimum_decay_rate = minimum_decay_rate
    
    def calculate_retention(
        self, 
        strength: float, 
        decay_rate: float, 
        elapsed_hours: float
    ) -> float:
        """
        Calculate memory retention after elapsed time.
        
        Uses the exponential forgetting curve formula:
        R = e^(-d*t/s)
        
        Where:
        - R is retention (0-1)
        - d is decay rate
        - t is elapsed time
        - s is memory strength
        
        Args:
            strength: Memory strength (higher is better)
            decay_rate: Decay rate (lower is better)
            elapsed_hours: Hours since last review
            
        Returns:
            Retention probability (0-1)
        """
        # Ensure strength is at least minimum
        strength = max(strength, self.minimum_strength)
        
        # Calculate retention using exponential decay
        exponent = -decay_rate * elapsed_hours / strength
        retention = math.exp(exponent)
        
        return min(max(retention, 0.0), 1.0)  # Clamp to [0, 1]
    
    def update_memory_parameters(
        self,
        current_strength: float,
        current_decay_rate: float,
        successful_recall: bool,
        difficult_recall: bool = False
    ) -> Tuple[float, float]:
        """
        Update memory parameters after a review.
        
        Args:
            current_strength: Current memory strength
            current_decay_rate: Current decay rate
            successful_recall: Whether the recall was successful
            difficult_recall: Whether recall was difficult but successful
            
        Returns:
            (new_strength, new_decay_rate)
        """
        if successful_recall:
            # Increase strength and decrease decay rate
            new_strength = current_strength + (
                self.strength_increase if not difficult_recall 
                else self.strength_increase / 2
            )
            
            # Reduce decay rate (memory becomes more persistent)
            new_decay_rate = max(
                current_decay_rate - self.decay_reduction_per_review,
                self.minimum_decay_rate
            )
        else:
            # Failed recall - decrease strength but don't increase decay rate
            new_strength = max(
                current_strength - self.strength_penalty,
                self.minimum_strength
            )
            
            # Keep current decay rate
            new_decay_rate = current_decay_rate
        
        return (new_strength, new_decay_rate)
    
    def calculate_next_review_interval(
        self,
        strength: float,
        decay_rate: float,
        target_retention: float = 0.7
    ) -> float:
        """
        Calculate optimal interval until next review.
        
        Args:
            strength: Memory strength
            decay_rate: Decay rate
            target_retention: Target retention at review time (0-1)
            
        Returns:
            Recommended hours until next review
        """
        # Ensure strength is at least minimum
        strength = max(strength, self.minimum_strength)
        
        # Solve for time that gives target retention
        # R = e^(-d*t/s)
        # ln(R) = -d*t/s
        # t = -s*ln(R)/d
        hours = -strength * math.log(target_retention) / decay_rate
        
        # Apply some randomness (±10%) to prevent predictability
        variation = random.uniform(0.9, 1.1)
        hours *= variation
        
        return max(hours, 1.0)  # Minimum 1 hour
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model parameters to dictionary for serialization."""
        return {
            "initial_strength": self.initial_strength,
            "minimum_strength": self.minimum_strength,
            "strength_increase": self.strength_increase,
            "strength_penalty": self.strength_penalty,
            "base_decay_rate": self.base_decay_rate,
            "decay_reduction_per_review": self.decay_reduction_per_review,
            "minimum_decay_rate": self.minimum_decay_rate
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ForgettingCurveModel':
        """Create model from dictionary."""
        return cls(
            initial_strength=data.get("initial_strength", 0.3),
            minimum_strength=data.get("minimum_strength", 0.2),
            strength_increase=data.get("strength_increase", 0.1),
            strength_penalty=data.get("strength_penalty", 0.15),
            base_decay_rate=data.get("base_decay_rate", 0.1),
            decay_reduction_per_review=data.get("decay_reduction_per_review", 0.02),
            minimum_decay_rate=data.get("minimum_decay_rate", 0.015)
        )


class ReviewItem:
    """
    Represents an item to be reviewed using spaced repetition.
    
    Tracks memory parameters and review history for a specific topic or question.
    """
    
    def __init__(
        self,
        item_id: str,
        topic: str,
        subtopic: Optional[str] = None,
        initial_strength: Optional[float] = None,
        initial_decay_rate: Optional[float] = None
    ):
        """
        Initialize a review item.
        
        Args:
            item_id: Unique identifier for the item
            topic: Topic for the item
            subtopic: Optional subtopic
            initial_strength: Optional initial memory strength
            initial_decay_rate: Optional initial decay rate
        """
        self.item_id = item_id
        self.topic = topic
        self.subtopic = subtopic
        
        # Memory parameters
        self.strength = initial_strength
        self.decay_rate = initial_decay_rate
        
        # Review history
        self.reviews: List[Dict[str, Any]] = []
        self.last_review_time: Optional[datetime.datetime] = None
        self.next_review_time: Optional[datetime.datetime] = None
        self.review_count = 0
        self.successful_count = 0
        
        # State tracking
        self.created_at = datetime.datetime.now()
        self.memory_state = MemoryState.NEW
    
    def initialize_parameters(self, model: ForgettingCurveModel) -> None:
        """
        Initialize memory parameters if not already set.
        
        Args:
            model: Forgetting curve model to use for defaults
        """
        if self.strength is None:
            self.strength = model.initial_strength
        
        if self.decay_rate is None:
            self.decay_rate = model.base_decay_rate
    
    def record_review(
        self,
        successful: bool,
        difficulty: float,
        review_time: Optional[datetime.datetime] = None,
        model: Optional[ForgettingCurveModel] = None
    ) -> None:
        """
        Record a review of this item.
        
        Args:
            successful: Whether recall was successful
            difficulty: Reported difficulty (0-1, higher is more difficult)
            review_time: When the review occurred
            model: Forgetting curve model for parameter updates
        """
        review_time = review_time or datetime.datetime.now()
        
        # Create review record
        review = {
            "time": review_time,
            "successful": successful,
            "difficulty": difficulty,
            "prev_strength": self.strength,
            "prev_decay_rate": self.decay_rate
        }
        
        # Update memory parameters if model provided
        if model:
            difficult_recall = successful and difficulty > 0.6
            self.strength, self.decay_rate = model.update_memory_parameters(
                self.strength, self.decay_rate, successful, difficult_recall
            )
            
            # Calculate next review time
            hours_until_next = model.calculate_next_review_interval(
                self.strength, self.decay_rate
            )
            self.next_review_time = review_time + datetime.timedelta(hours=hours_until_next)
            
            # Update state based on current retention
            retention = self.get_current_retention(model)
            self.memory_state = MemoryState.from_retention(retention)
        
        # Update review record with new parameters
        review.update({
            "new_strength": self.strength,
            "new_decay_rate": self.decay_rate,
            "next_review_time": self.next_review_time
        })
        
        # Update review history
        self.reviews.append(review)
        self.last_review_time = review_time
        self.review_count += 1
        
        if successful:
            self.successful_count += 1
    
    def get_current_retention(self, model: ForgettingCurveModel) -> float:
        """
        Calculate current retention probability.
        
        Args:
            model: Forgetting curve model to use
            
        Returns:
            Current retention probability (0-1)
        """
        if self.last_review_time is None:
            return 0.0  # Never reviewed
        
        # Calculate hours since last review
        now = datetime.datetime.now()
        elapsed_hours = (now - self.last_review_time).total_seconds() / 3600
        
        # Calculate retention
        return model.calculate_retention(self.strength, self.decay_rate, elapsed_hours)
    
    def is_due_for_review(self) -> bool:
        """
        Check if item is due for review.
        
        Returns:
            Whether the item is due for review
        """
        if self.next_review_time is None:
            return self.memory_state == MemoryState.NEW  # New items should be reviewed
        
        return datetime.datetime.now() >= self.next_review_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "item_id": self.item_id,
            "topic": self.topic,
            "subtopic": self.subtopic,
            "strength": self.strength,
            "decay_rate": self.decay_rate,
            "review_count": self.review_count,
            "successful_count": self.successful_count,
            "last_review_time": self.last_review_time.isoformat() if self.last_review_time else None,
            "next_review_time": self.next_review_time.isoformat() if self.next_review_time else None,
            "memory_state": self.memory_state.value,
            "created_at": self.created_at.isoformat(),
            "reviews": self.reviews
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ReviewItem':
        """Create from dictionary."""
        item = cls(
            item_id=data["item_id"],
            topic=data["topic"],
            subtopic=data.get("subtopic"),
            initial_strength=data.get("strength"),
            initial_decay_rate=data.get("decay_rate")
        )
        
        item.review_count = data.get("review_count", 0)
        item.successful_count = data.get("successful_count", 0)
        
        if data.get("last_review_time"):
            item.last_review_time = datetime.datetime.fromisoformat(data["last_review_time"])
        
        if data.get("next_review_time"):
            item.next_review_time = datetime.datetime.fromisoformat(data["next_review_time"])
        
        if data.get("memory_state"):
            item.memory_state = MemoryState(data["memory_state"])
        
        if data.get("created_at"):
            item.created_at = datetime.datetime.fromisoformat(data["created_at"])
        
        item.reviews = data.get("reviews", [])
        
        return item


class SpacedRepetitionScheduler:
    """
    Manages spaced repetition scheduling for a collection of review items.
    
    Uses a forgetting curve model to determine optimal review times and
    selects items for review based on their memory states and due times.
    """
    
    def __init__(
        self,
        forgetting_model: Optional[ForgettingCurveModel] = None,
        target_retention: float = 0.7,
        max_daily_reviews: int = 20
    ):
        """
        Initialize the spaced repetition scheduler.
        
        Args:
            forgetting_model: Forgetting curve model to use
            target_retention: Target retention probability
            max_daily_reviews: Maximum number of reviews per day
        """
        self.model = forgetting_model or ForgettingCurveModel()
        self.target_retention = target_retention
        self.max_daily_reviews = max_daily_reviews
        
        # Items by ID
        self.items: Dict[str, ReviewItem] = {}
        
        # Cached due items for quicker access
        self._due_items: List[str] = []
        self._due_cache_time: Optional[datetime.datetime] = None
    
    def add_item(
        self,
        item_id: str,
        topic: str,
        subtopic: Optional[str] = None,
        initial_strength: Optional[float] = None,
        initial_decay_rate: Optional[float] = None
    ) -> ReviewItem:
        """
        Add a new item to the scheduler.
        
        Args:
            item_id: Unique identifier for the item
            topic: Topic for the item
            subtopic: Optional subtopic
            initial_strength: Optional initial memory strength
            initial_decay_rate: Optional initial decay rate
            
        Returns:
            The created review item
        """
        item = ReviewItem(
            item_id=item_id,
            topic=topic,
            subtopic=subtopic,
            initial_strength=initial_strength,
            initial_decay_rate=initial_decay_rate
        )
        
        # Initialize with model parameters
        item.initialize_parameters(self.model)
        
        # Add to collection
        self.items[item_id] = item
        
        # Invalidate cache
        self._due_cache_time = None
        
        return item
    
    def get_item(self, item_id: str) -> Optional[ReviewItem]:
        """
        Get a review item by ID.
        
        Args:
            item_id: Item ID to retrieve
            
        Returns:
            Review item or None if not found
        """
        return self.items.get(item_id)
    
    def record_review(
        self,
        item_id: str,
        successful: bool,
        difficulty: float,
        review_time: Optional[datetime.datetime] = None
    ) -> None:
        """
        Record a review for an item.
        
        Args:
            item_id: Item ID
            successful: Whether recall was successful
            difficulty: Reported difficulty (0-1)
            review_time: When the review occurred
        """
        item = self.get_item(item_id)
        if item:
            item.record_review(successful, difficulty, review_time, self.model)
            
            # Invalidate cache
            self._due_cache_time = None
            
            logger.debug(f"Recorded review for item {item_id}: success={successful}, difficulty={difficulty}")
        else:
            logger.warning(f"Attempted to record review for unknown item: {item_id}")
    
    def get_due_items(
        self,
        max_items: Optional[int] = None,
        topics: Optional[List[str]] = None,
        prioritize_new: bool = True
    ) -> List[ReviewItem]:
        """
        Get items that are due for review.
        
        Args:
            max_items: Maximum number of items to return
            topics: Optional list of topics to filter by
            prioritize_new: Whether to prioritize new items
            
        Returns:
            List of items due for review
        """
        now = datetime.datetime.now()
        
        # Check if we need to refresh the cache
        if self._due_cache_time is None or (now - self._due_cache_time).total_seconds() > 60:
            self._refresh_due_items_cache()
        
        # Get due items
        due_items = [self.items[item_id] for item_id in self._due_items]
        
        # Filter by topic if specified
        if topics:
            due_items = [item for item in due_items if item.topic in topics]
        
        # Sort items
        if prioritize_new:
            # Sort by state (new first) then by due time
            due_items.sort(key=lambda i: (
                i.memory_state != MemoryState.NEW,  # New items first
                i.next_review_time or now  # Then by due time
            ))
        else:
            # Sort by due time
            due_items.sort(key=lambda i: i.next_review_time or now)
        
        # Limit number of items
        if max_items:
            due_items = due_items[:max_items]
        
        return due_items
    
    def _refresh_due_items_cache(self) -> None:
        """Refresh the cache of due items."""
        self._due_items = [
            item_id for item_id, item in self.items.items()
            if item.is_due_for_review()
        ]
        self._due_cache_time = datetime.datetime.now()
    
    def get_item_stats(self) -> Dict[str, int]:
        """
        Get statistics about items by memory state.
        
        Returns:
            Dictionary mapping memory states to counts
        """
        stats = {state.value: 0 for state in MemoryState}
        
        for item in self.items.values():
            stats[item.memory_state.value] += 1
        
        return stats
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "model": self.model.to_dict(),
            "target_retention": self.target_retention,
            "max_daily_reviews": self.max_daily_reviews,
            "items": {
                item_id: item.to_dict() for item_id, item in self.items.items()
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SpacedRepetitionScheduler':
        """Create from dictionary."""
        model = None
        if data.get("model"):
            model = ForgettingCurveModel.from_dict(data["model"])
        
        scheduler = cls(
            forgetting_model=model,
            target_retention=data.get("target_retention", 0.7),
            max_daily_reviews=data.get("max_daily_reviews", 20)
        )
        
        # Load items
        for item_id, item_data in data.get("items", {}).items():
            scheduler.items[item_id] = ReviewItem.from_dict(item_data)
        
        return scheduler 