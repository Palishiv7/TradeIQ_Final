"""
Gamification System Models

This module defines the core data models for the gamification system including:
1. XP and leveling mechanics
2. Achievements and badges with rarity tiers
3. Leaderboards

These models are used across the application to provide engagement features.
"""

import enum
import uuid
import math
import datetime
from typing import Dict, List, Any, Optional, Set, Union
from dataclasses import dataclass, field

from backend.common.serialization import SerializableMixin


class XPSource(enum.Enum):
    """Sources of XP in the system."""
    QUESTION_COMPLETION = "question_completion"
    ASSESSMENT_COMPLETION = "assessment_completion"
    ACHIEVEMENT_UNLOCK = "achievement_unlock"
    STREAK_BONUS = "streak_bonus"
    SPEED_BONUS = "speed_bonus"
    DIFFICULTY_BONUS = "difficulty_bonus"
    FIRST_TIME_BONUS = "first_time_bonus"
    DAILY_ACTIVITY = "daily_activity"
    ADMIN_ADJUSTMENT = "admin_adjustment"


class AchievementRarity(enum.Enum):
    """Rarity tiers for achievements."""
    COMMON = "common"
    UNCOMMON = "uncommon"
    RARE = "rare"
    EPIC = "epic"
    LEGENDARY = "legendary"
    
    @property
    def xp_reward(self) -> int:
        """Get the XP reward for this rarity tier."""
        return {
            AchievementRarity.COMMON: 50,
            AchievementRarity.UNCOMMON: 100,
            AchievementRarity.RARE: 250,
            AchievementRarity.EPIC: 500,
            AchievementRarity.LEGENDARY: 1000
        }[self]
    
    @property
    def color(self) -> str:
        """Get the display color for this rarity tier."""
        return {
            AchievementRarity.COMMON: "#8E8E8E",      # Gray
            AchievementRarity.UNCOMMON: "#2ECC71",    # Green
            AchievementRarity.RARE: "#3498DB",        # Blue
            AchievementRarity.EPIC: "#9B59B6",        # Purple
            AchievementRarity.LEGENDARY: "#F1C40F"    # Gold
        }[self]


class LeaderboardType(enum.Enum):
    """Types of leaderboards in the system."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    ALL_TIME = "all_time"
    ASSESSMENT_SPECIFIC = "assessment_specific"
    ACHIEVEMENT_COUNT = "achievement_count"


@dataclass
class XPTransaction(SerializableMixin):
    """
    Represents a single XP transaction in the user's history.
    
    This tracks each instance of XP being awarded or deducted.
    """
    
    __serializable_fields__ = [
        "id", "user_id", "amount", "source", "description", 
        "timestamp", "assessment_id", "metadata"
    ]
    
    id: str
    user_id: str
    amount: int
    source: XPSource
    description: str
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.now)
    assessment_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize after creation."""
        if not self.id:
            self.id = str(uuid.uuid4())
        
        if isinstance(self.source, str):
            self.source = XPSource(self.source)


@dataclass
class LevelInfo(SerializableMixin):
    """
    Information about a specific level in the XP progression system.
    
    Defines the XP requirements and rewards for each level.
    """
    
    __serializable_fields__ = [
        "level", "min_xp", "max_xp", "title", "rewards"
    ]
    
    level: int
    min_xp: int
    max_xp: int
    title: Optional[str] = None
    rewards: List[Dict[str, Any]] = field(default_factory=list)
    
    @classmethod
    def calculate_min_xp(cls, level: int) -> int:
        """
        Calculate the minimum XP required for a level.
        
        Uses a progressive scaling formula to create an engaging progression curve:
        - Early levels require less XP to promote quick progression
        - Higher levels scale quadratically to create long-term goals
        
        Args:
            level: The level to calculate for
            
        Returns:
            Minimum XP required for this level
        """
        if level <= 1:
            return 0
        
        # Base XP for level 2 is 100
        if level == 2:
            return 100
        
        # Level 3-10: Linear growth with slight acceleration
        if level <= 10:
            return int(100 * (level - 1) * 1.5)
        
        # Level 11-50: Quadratic growth for mid-game progression
        if level <= 50:
            return int(100 * (level - 1) * 1.5 + (level - 10) ** 2 * 50)
        
        # Level 51+: Steeper quadratic growth for end-game progression
        return int(100 * (level - 1) * 1.5 + (level - 10) ** 2 * 50 + (level - 50) ** 2.2 * 100)


@dataclass
class XPSystem(SerializableMixin):
    """
    Manages the user's XP and level progression.
    
    Tracks total XP, current level, and transactions.
    """
    
    __serializable_fields__ = [
        "user_id", "total_xp", "current_level", "transactions",
        "created_at", "updated_at"
    ]
    
    user_id: str
    total_xp: int = 0
    current_level: int = 1
    transactions: List[XPTransaction] = field(default_factory=list)
    created_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    updated_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    
    def add_xp(
        self, 
        amount: int, 
        source: XPSource, 
        description: str,
        assessment_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Add XP to the user's account and create a transaction record.
        
        Args:
            amount: Amount of XP to add
            source: Source of the XP
            description: Description of the transaction
            assessment_id: Optional ID of related assessment
            metadata: Optional additional metadata
            
        Returns:
            Dict with level up information if applicable
        """
        if amount <= 0:
            raise ValueError("XP amount must be positive")
            
        # Create transaction
        transaction = XPTransaction(
            id=str(uuid.uuid4()),
            user_id=self.user_id,
            amount=amount,
            source=source,
            description=description,
            assessment_id=assessment_id,
            metadata=metadata or {}
        )
        
        # Add transaction to history
        self.transactions.append(transaction)
        
        # Update total XP
        old_xp = self.total_xp
        self.total_xp += amount
        
        # Update timestamp
        self.updated_at = datetime.datetime.now()
        
        # Check for level up
        result = {"xp_added": amount, "level_up": False}
        new_level = self._calculate_level(self.total_xp)
        
        if new_level > self.current_level:
            result["level_up"] = True
            result["old_level"] = self.current_level
            result["new_level"] = new_level
            self.current_level = new_level
        
        return result
    
    def _calculate_level(self, xp: int) -> int:
        """
        Calculate level based on total XP.
        
        Args:
            xp: Total XP to calculate level for
            
        Returns:
            Calculated level
        """
        level = 1
        while True:
            next_level_min_xp = LevelInfo.calculate_min_xp(level + 1)
            if xp < next_level_min_xp:
                break
            level += 1
        return level
    
    def get_level_progress(self) -> Dict[str, Any]:
        """
        Get information about the current level progress.
        
        Returns:
            Dict with level progress information
        """
        current_level_min = LevelInfo.calculate_min_xp(self.current_level)
        next_level_min = LevelInfo.calculate_min_xp(self.current_level + 1)
        
        xp_in_level = self.total_xp - current_level_min
        xp_needed = next_level_min - current_level_min
        progress_percent = min(100, (xp_in_level / xp_needed) * 100) if xp_needed > 0 else 100
        
        return {
            "current_level": self.current_level,
            "total_xp": self.total_xp,
            "current_level_min_xp": current_level_min,
            "next_level_min_xp": next_level_min,
            "xp_in_level": xp_in_level,
            "xp_needed_for_next": next_level_min - self.total_xp,
            "progress_percent": progress_percent
        }
    
    def get_recent_transactions(self, limit: int = 10) -> List[XPTransaction]:
        """
        Get the most recent XP transactions.
        
        Args:
            limit: Maximum number of transactions to return
            
        Returns:
            List of recent transactions
        """
        sorted_transactions = sorted(
            self.transactions, 
            key=lambda t: t.timestamp, 
            reverse=True
        )
        return sorted_transactions[:limit]


@dataclass
class Achievement(SerializableMixin):
    """
    Represents an achievement that users can unlock.
    
    Achievements provide extra engagement through goals and rewards.
    """
    
    __serializable_fields__ = [
        "id", "name", "description", "rarity", "icon", "category",
        "criteria", "xp_reward", "is_hidden", "metadata"
    ]
    
    id: str
    name: str
    description: str
    rarity: AchievementRarity
    icon: str
    category: str
    criteria: Dict[str, Any]
    xp_reward: int
    is_hidden: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize after creation."""
        if not self.id:
            self.id = str(uuid.uuid4())
        
        if isinstance(self.rarity, str):
            self.rarity = AchievementRarity(self.rarity)
            
        # Set default XP reward based on rarity if not specified
        if not self.xp_reward:
            self.xp_reward = self.rarity.xp_reward


@dataclass
class UserAchievement(SerializableMixin):
    """
    Represents an achievement a user has unlocked.
    
    Tracks when and how achievements were earned.
    """
    
    __serializable_fields__ = [
        "user_id", "achievement_id", "unlocked_at", "progress",
        "progress_max", "metadata"
    ]
    
    user_id: str
    achievement_id: str
    unlocked_at: Optional[datetime.datetime] = None
    progress: int = 0
    progress_max: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_unlocked(self) -> bool:
        """Check if the achievement is unlocked."""
        return self.unlocked_at is not None
    
    @property
    def progress_percent(self) -> float:
        """Get progress as a percentage."""
        if self.progress_max <= 0:
            return 100.0
        return min(100.0, (self.progress / self.progress_max) * 100)
    
    def update_progress(self, progress: int) -> bool:
        """
        Update progress towards achievement.
        
        Args:
            progress: New progress value
            
        Returns:
            True if achievement was unlocked by this update
        """
        if self.is_unlocked:
            return False
            
        old_progress = self.progress
        self.progress = progress
        
        # Check if achievement is now unlocked
        if not self.is_unlocked and self.progress >= self.progress_max:
            self.unlocked_at = datetime.datetime.now()
            return True
            
        return False


@dataclass
class AchievementSystem(SerializableMixin):
    """
    Manages the user's achievements.
    
    Tracks achievement progress and unlocks.
    """
    
    __serializable_fields__ = [
        "user_id", "achievements", "created_at", "updated_at"
    ]
    
    user_id: str
    achievements: Dict[str, UserAchievement] = field(default_factory=dict)
    created_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    updated_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    
    def get_achievement(self, achievement_id: str) -> Optional[UserAchievement]:
        """
        Get user achievement by ID.
        
        Args:
            achievement_id: Achievement ID to get
            
        Returns:
            UserAchievement if found, None otherwise
        """
        return self.achievements.get(achievement_id)
    
    def initialize_achievement(
        self, 
        achievement_id: str, 
        progress_max: int = 1,
        initial_progress: int = 0
    ) -> UserAchievement:
        """
        Initialize tracking for an achievement.
        
        Args:
            achievement_id: Achievement ID to initialize
            progress_max: Maximum progress value
            initial_progress: Initial progress value
            
        Returns:
            The initialized UserAchievement
        """
        if achievement_id in self.achievements:
            return self.achievements[achievement_id]
            
        achievement = UserAchievement(
            user_id=self.user_id,
            achievement_id=achievement_id,
            progress=initial_progress,
            progress_max=progress_max
        )
        
        self.achievements[achievement_id] = achievement
        self.updated_at = datetime.datetime.now()
        
        return achievement
    
    def update_achievement(
        self, 
        achievement_id: str, 
        progress: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Update progress for an achievement.
        
        Args:
            achievement_id: Achievement ID to update
            progress: New progress value
            metadata: Optional metadata to update
            
        Returns:
            Dict with update results
        """
        # Get or initialize achievement
        if achievement_id not in self.achievements:
            raise ValueError(f"Achievement {achievement_id} not initialized")
            
        achievement = self.achievements[achievement_id]
        
        # Update metadata if provided
        if metadata:
            achievement.metadata.update(metadata)
        
        # Update progress
        was_unlocked = achievement.is_unlocked
        newly_unlocked = achievement.update_progress(progress)
        
        self.updated_at = datetime.datetime.now()
        
        return {
            "achievement_id": achievement_id,
            "was_unlocked": was_unlocked,
            "newly_unlocked": newly_unlocked,
            "current_progress": achievement.progress,
            "progress_max": achievement.progress_max,
            "progress_percent": achievement.progress_percent
        }
    
    def get_unlocked_achievements(self) -> List[str]:
        """
        Get list of unlocked achievement IDs.
        
        Returns:
            List of unlocked achievement IDs
        """
        return [
            achievement_id
            for achievement_id, achievement in self.achievements.items()
            if achievement.is_unlocked
        ]
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """
        Get summary of achievement progress.
        
        Returns:
            Dict with achievement progress summary
        """
        total = len(self.achievements)
        unlocked = len(self.get_unlocked_achievements())
        
        return {
            "total": total,
            "unlocked": unlocked,
            "percent_complete": (unlocked / total * 100) if total > 0 else 0
        }


@dataclass
class LeaderboardEntry(SerializableMixin):
    """
    Represents a single entry on a leaderboard.
    
    Tracks user score and rank.
    """
    
    __serializable_fields__ = [
        "user_id", "score", "rank", "metadata", "timestamp"
    ]
    
    user_id: str
    score: int
    rank: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.now)


@dataclass
class GamificationProfile(SerializableMixin):
    """
    Comprehensive gamification profile for a user.
    
    Combines XP, achievements, and leaderboard data.
    """
    
    __serializable_fields__ = [
        "user_id", "xp_system", "achievement_system", "created_at", "updated_at"
    ]
    
    user_id: str
    xp_system: XPSystem
    achievement_system: AchievementSystem
    created_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    updated_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    
    @classmethod
    def create(cls, user_id: str) -> 'GamificationProfile':
        """
        Create a new gamification profile.
        
        Args:
            user_id: User ID to create profile for
            
        Returns:
            New GamificationProfile instance
        """
        return cls(
            user_id=user_id,
            xp_system=XPSystem(user_id=user_id),
            achievement_system=AchievementSystem(user_id=user_id)
        ) 