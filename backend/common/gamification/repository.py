"""
Gamification Repository

This module provides data persistence for gamification features:
1. XP and leveling systems
2. Achievements and badges
3. Leaderboards

It handles storage, retrieval, and efficient caching of gamification data.
"""

import json
import logging
import asyncio
import datetime
from typing import Dict, List, Any, Optional, Set, Tuple, Union, Type, TypeVar, Generic

from redis.asyncio import Redis as AsyncRedis

from backend.common.cache import async_cached
from backend.common.serialization import SerializableMixin
from backend.common.db.repository import BaseRepository, NotFoundError
from backend.common.logger import app_logger

from backend.common.gamification.models import (
    XPSystem, AchievementSystem, GamificationProfile,
    LeaderboardEntry, LeaderboardType, Achievement
)

# Set up module logger
logger = app_logger.getChild("gamification.repository")

# Type variables for generics
T = TypeVar('T', bound=SerializableMixin)


class GamificationRepository:
    """
    Repository for gamification data, handling XP, achievements, and leaderboards.
    
    This repository uses a combination of database storage for persistent data
    and Redis for high-performance leaderboards and caching.
    """
    
    def __init__(
        self,
        db_client=None,
        redis_client: Optional[AsyncRedis] = None,
        cache_ttl: int = 3600
    ):
        """
        Initialize the gamification repository.
        
        Args:
            db_client: Database client for persistent storage
            redis_client: Redis client for leaderboards and caching
            cache_ttl: Cache TTL in seconds (default: 1 hour)
        """
        self.db_client = db_client
        self.redis = redis_client
        self.cache_ttl = cache_ttl
        self.achievement_definitions: Dict[str, Achievement] = {}
    
    # Database helper methods
    async def get_document(self, collection: str, query: Dict[str, Any], model_class: Type[T]) -> T:
        """Get a document from a collection."""
        # Implementation would depend on your DB client
        # This is a placeholder implementation
        if not self.db_client:
            raise NotFoundError(model_class.__name__, str(query))
        # In a real implementation, you would query the database
        # For now we'll just raise NotFoundError for simplicity
        raise NotFoundError(model_class.__name__, str(query))
    
    async def save_document(self, collection: str, document: SerializableMixin) -> None:
        """Save a document to a collection."""
        # Implementation would depend on your DB client
        # This is a placeholder implementation
        if not self.db_client:
            return
        # In a real implementation, you would save to the database
        logger.info(f"Mock saving document to {collection}: {document}")
    
    async def get_documents(self, collection: str, query: Dict[str, Any], model_class: Type[T]) -> List[T]:
        """Get documents from a collection."""
        # Implementation would depend on your DB client
        # This is a placeholder implementation
        if not self.db_client:
            return []
        # In a real implementation, you would query the database
        return []
    
    # Helper methods for cached profile updates
    async def _update_cached_profile_xp(self, user_id: str, xp_system: XPSystem) -> None:
        """Update XP in cached profile if it exists."""
        # Implementation would depend on your caching strategy
        pass
    
    async def _update_cached_profile_achievements(self, user_id: str, achievement_system: AchievementSystem) -> None:
        """Update achievements in cached profile if it exists."""
        # Implementation would depend on your caching strategy
        pass
    
    async def _update_xp_leaderboards(self, user_id: str, total_xp: int) -> None:
        """Update XP leaderboards for a user."""
        # Implementation would depend on your leaderboard strategy
        pass
    
    async def _update_achievement_leaderboard(self, user_id: str, achievement_count: int) -> None:
        """Update achievement count leaderboard for a user."""
        # Implementation would depend on your leaderboard strategy
        pass
    
    def _get_leaderboard_key(self, leaderboard_type: LeaderboardType, id_suffix: str = "") -> str:
        """Get Redis key for a leaderboard."""
        base_key = f"leaderboard:{leaderboard_type.value}"
        if id_suffix:
            return f"{base_key}:{id_suffix}"
        return base_key
    
    def _get_leaderboard_ttl(self, leaderboard_type: LeaderboardType) -> int:
        """Get TTL for a leaderboard type."""
        # Different leaderboard types might have different retention periods
        if leaderboard_type == LeaderboardType.DAILY:
            return 60 * 60 * 24 * 2  # 2 days
        elif leaderboard_type == LeaderboardType.WEEKLY:
            return 60 * 60 * 24 * 9  # 9 days (slightly more than a week)
        elif leaderboard_type == LeaderboardType.MONTHLY:
            return 60 * 60 * 24 * 35  # 35 days (slightly more than a month)
        else:
            return 60 * 60 * 24 * 90  # 90 days
    
    async def get_xp_system(self, user_id: str) -> XPSystem:
        """
        Get a user's XP system, creating if it doesn't exist.
        
        Args:
            user_id: User identifier
            
        Returns:
            User's XP system
        """
        try:
            # Try to get existing record
            xp_system = await self.get_document(
                "gamification_xp",
                {"user_id": user_id},
                XPSystem
            )
            return xp_system
        except NotFoundError:
            # Create new XP system if not found
            xp_system = XPSystem(user_id=user_id)
            await self.save_document("gamification_xp", xp_system)
            return xp_system
    
    async def save_xp_system(self, xp_system: XPSystem) -> None:
        """
        Save a user's XP system.
        
        Args:
            xp_system: XP system to save
        """
        # Update timestamp
        xp_system.updated_at = datetime.datetime.now()
        
        # Save to database
        await self.save_document("gamification_xp", xp_system)
        
        # Update cached profile if exists
        await self._update_cached_profile_xp(xp_system.user_id, xp_system)
        
        # Update leaderboards
        await self._update_xp_leaderboards(xp_system.user_id, xp_system.total_xp)
    
    async def get_achievement_system(self, user_id: str) -> AchievementSystem:
        """
        Get a user's achievement system, creating if it doesn't exist.
        
        Args:
            user_id: User identifier
            
        Returns:
            User's achievement system
        """
        try:
            # Try to get existing record
            achievement_system = await self.get_document(
                "gamification_achievements",
                {"user_id": user_id},
                AchievementSystem
            )
            return achievement_system
        except NotFoundError:
            # Create new achievement system if not found
            achievement_system = AchievementSystem(user_id=user_id)
            await self.save_document("gamification_achievements", achievement_system)
            return achievement_system
    
    async def save_achievement_system(self, achievement_system: AchievementSystem) -> None:
        """
        Save a user's achievement system.
        
        Args:
            achievement_system: Achievement system to save
        """
        # Update timestamp
        achievement_system.updated_at = datetime.datetime.now()
        
        # Save to database
        await self.save_document("gamification_achievements", achievement_system)
        
        # Update cached profile if exists
        await self._update_cached_profile_achievements(achievement_system.user_id, achievement_system)
        
        # Update achievement count leaderboard
        unlocked_count = len(achievement_system.get_unlocked_achievements())
        await self._update_achievement_leaderboard(achievement_system.user_id, unlocked_count)
    
    @async_cached(prefix="gamification:profile:", ttl=1800)
    async def get_gamification_profile(self, user_id: str) -> GamificationProfile:
        """
        Get a user's complete gamification profile.
        
        Args:
            user_id: User identifier
            
        Returns:
            User's gamification profile
        """
        # Get XP and achievement systems in parallel
        xp_system, achievement_system = await asyncio.gather(
            self.get_xp_system(user_id),
            self.get_achievement_system(user_id)
        )
        
        # Create and return profile
        return GamificationProfile(
            user_id=user_id,
            xp_system=xp_system,
            achievement_system=achievement_system
        )
    
    async def save_gamification_profile(self, profile: GamificationProfile) -> None:
        """
        Save a user's complete gamification profile.
        
        Args:
            profile: Gamification profile to save
        """
        # Update timestamp
        profile.updated_at = datetime.datetime.now()
        
        # Save XP and achievement systems in parallel
        await asyncio.gather(
            self.save_xp_system(profile.xp_system),
            self.save_achievement_system(profile.achievement_system)
        )
    
    async def get_achievement_definition(self, achievement_id: str) -> Optional[Achievement]:
        """
        Get an achievement definition by ID.
        
        Args:
            achievement_id: Achievement identifier
            
        Returns:
            Achievement definition or None if not found
        """
        # Check cache first
        if achievement_id in self.achievement_definitions:
            return self.achievement_definitions[achievement_id]
        
        # Try to get from database
        try:
            achievement = await self.get_document(
                "achievement_definitions",
                {"id": achievement_id},
                Achievement
            )
            # Cache for future use
            self.achievement_definitions[achievement_id] = achievement
            return achievement
        except NotFoundError:
            return None
    
    async def save_achievement_definition(self, achievement: Achievement) -> None:
        """
        Save an achievement definition.
        
        Args:
            achievement: Achievement definition to save
        """
        # Save to database
        await self.save_document("achievement_definitions", achievement)
        
        # Update cache
        self.achievement_definitions[achievement.id] = achievement
    
    async def get_all_achievement_definitions(self) -> Dict[str, Achievement]:
        """
        Get all achievement definitions.
        
        Returns:
            Dictionary mapping achievement IDs to definitions
        """
        # Try to get from database
        try:
            achievements = await self.get_documents(
                "achievement_definitions",
                {},
                Achievement
            )
            
            # Cache all achievements
            self.achievement_definitions = {a.id: a for a in achievements}
            return self.achievement_definitions
        except Exception as e:
            logger.error(f"Error getting achievement definitions: {e}")
            return {}
    
    async def add_leaderboard_entry(
        self,
        leaderboard_type: LeaderboardType,
        user_id: str,
        score: int,
        metadata: Optional[Dict[str, Any]] = None,
        id_suffix: str = ""
    ) -> None:
        """
        Add an entry to a leaderboard.
        
        Args:
            leaderboard_type: Type of leaderboard
            user_id: User identifier
            score: User's score
            metadata: Optional additional data
            id_suffix: Optional suffix for assessment-specific leaderboards
        """
        if not self.redis:
            logger.warning("Redis not available, skipping leaderboard update")
            return
            
        try:
            # Create entry
            entry = LeaderboardEntry(
                user_id=user_id,
                score=score,
                metadata=metadata or {}
            )
            
            # Get leaderboard key
            leaderboard_key = self._get_leaderboard_key(leaderboard_type, id_suffix)
            
            # Add to leaderboard
            await self.redis.zadd(leaderboard_key, {user_id: score})
            
            # Store entry metadata
            entry_key = f"{leaderboard_key}:entries:{user_id}"
            await self.redis.set(
                entry_key,
                json.dumps(entry.to_dict()),
                ex=self._get_leaderboard_ttl(leaderboard_type)
            )
            
            # Update rank
            rank = await self.redis.zrevrank(leaderboard_key, user_id)
            if rank is not None:
                entry.rank = rank + 1  # Convert to 1-indexed
                
                # Update stored entry with rank
                await self.redis.set(
                    entry_key,
                    json.dumps(entry.to_dict()),
                    ex=self._get_leaderboard_ttl(leaderboard_type)
                )
        except Exception as e:
            logger.error(f"Error adding leaderboard entry: {e}")
    
    async def get_leaderboard(
        self,
        leaderboard_type: LeaderboardType,
        start: int = 0,
        count: int = 10,
        id_suffix: str = ""
    ) -> List[LeaderboardEntry]:
        """
        Get entries from a leaderboard.
        
        Args:
            leaderboard_type: Type of leaderboard
            start: Starting rank (0-indexed)
            count: Number of entries to get
            id_suffix: Optional suffix for assessment-specific leaderboards
            
        Returns:
            List of leaderboard entries
        """
        if not self.redis:
            logger.warning("Redis not available, returning empty leaderboard")
            return []
            
        try:
            # Get leaderboard key
            leaderboard_key = self._get_leaderboard_key(leaderboard_type, id_suffix)
            
            # Get top scores with rank
            leaderboard_data = await self.redis.zrevrange(
                leaderboard_key,
                start,
                start + count - 1,
                withscores=True
            )
            
            entries = []
            for user_id, score in leaderboard_data:
                # Try to get full entry with metadata
                entry_key = f"{leaderboard_key}:entries:{user_id.decode('utf-8')}"
                entry_data = await self.redis.get(entry_key)
                
                if entry_data:
                    # Parse stored entry
                    entry_dict = json.loads(entry_data)
                    entry = LeaderboardEntry(**entry_dict)
                else:
                    # Create minimal entry
                    entry = LeaderboardEntry(
                        user_id=user_id.decode('utf-8'),
                        score=int(score),
                        rank=start + len(entries) + 1  # 1-indexed rank
                    )
                
                entries.append(entry)
            
            return entries
        except Exception as e:
            logger.error(f"Error getting leaderboard: {e}")
            return []
    
    async def get_user_leaderboard_rank(
        self,
        leaderboard_type: LeaderboardType,
        user_id: str,
        id_suffix: str = ""
    ) -> Optional[int]:
        """
        Get a user's rank on a leaderboard.
        
        Args:
            leaderboard_type: Type of leaderboard
            user_id: User identifier
            id_suffix: Optional suffix for assessment-specific leaderboards
            
        Returns:
            User's rank (1-indexed) or None if not ranked
        """
        if not self.redis:
            logger.warning("Redis not available, returning None for leaderboard rank")
            return None
            
        try:
            # Get leaderboard key
            leaderboard_key = self._get_leaderboard_key(leaderboard_type, id_suffix)
            
            # Get rank
            rank = await self.redis.zrevrank(leaderboard_key, user_id)
            
            # Convert to 1-indexed if found
            return rank + 1 if rank is not None else None
        except Exception as e:
            logger.error(f"Error getting user leaderboard rank: {e}")
            return None
    
    async def get_neighboring_leaderboard_entries(
        self,
        leaderboard_type: LeaderboardType,
        user_id: str,
        count: int = 5,
        id_suffix: str = ""
    ) -> List[LeaderboardEntry]:
        """
        Get entries around a user on a leaderboard.
        
        Args:
            leaderboard_type: Type of leaderboard
            user_id: User identifier
            count: Number of entries to get on each side
            id_suffix: Optional suffix for assessment-specific leaderboards
            
        Returns:
            List of leaderboard entries centered around the user
        """
        # Get user's rank
        rank = await self.get_user_leaderboard_rank(leaderboard_type, user_id, id_suffix)
        
        if rank is None:
            return []
            
        # Get entries around user
        start = max(0, rank - count - 1)  # Convert to 0-indexed
        return await self.get_leaderboard(
            leaderboard_type, 
            start, 
            count * 2 + 1,
            id_suffix
        )
    
    async def update_assessment_leaderboard(
        self,
        assessment_id: str,
        user_id: str,
        score: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Update an assessment-specific leaderboard.
        
        Args:
            assessment_id: Assessment identifier
            user_id: User identifier
            score: User's score
            metadata: Optional additional data
        """
        await self.add_leaderboard_entry(
            LeaderboardType.ASSESSMENT_SPECIFIC,
            user_id,
            score,
            metadata,
            assessment_id
        )
    
    async def _update_xp_leaderboards(self, user_id: str, xp: int) -> None:
        """
        Update XP-based leaderboards.
        
        Args:
            user_id: User identifier
            xp: User's XP
        """
        # Update all-time leaderboard
        await self.add_leaderboard_entry(
            LeaderboardType.ALL_TIME,
            user_id,
            xp
        )
        
        # Update time-based leaderboards
        # These will be more complex in a production system with time-based segmentation
        # For now, we'll just update all leaderboards with the same XP
        await asyncio.gather(
            self.add_leaderboard_entry(LeaderboardType.DAILY, user_id, xp),
            self.add_leaderboard_entry(LeaderboardType.WEEKLY, user_id, xp),
            self.add_leaderboard_entry(LeaderboardType.MONTHLY, user_id, xp)
        )
    
    async def _update_achievement_leaderboard(self, user_id: str, count: int) -> None:
        """
        Update achievement count leaderboard.
        
        Args:
            user_id: User identifier
            count: Achievement count
        """
        await self.add_leaderboard_entry(
            LeaderboardType.ACHIEVEMENT_COUNT,
            user_id,
            count
        )
    
    async def _update_cached_profile_xp(self, user_id: str, xp_system: XPSystem) -> None:
        """
        Update XP in a cached gamification profile.
        
        Args:
            user_id: User identifier
            xp_system: Updated XP system
        """
        cache_key = f"gamification:profile:{user_id}"
        
        if not self.redis:
            return
            
        try:
            # Check if cached profile exists
            profile_data = await self.redis.get(cache_key)
            if not profile_data:
                return
                
            # Parse profile
            profile_dict = json.loads(profile_data)
            
            # Update XP system
            profile_dict["xp_system"] = xp_system.to_dict()
            
            # Update timestamp
            profile_dict["updated_at"] = datetime.datetime.now().isoformat()
            
            # Store updated profile
            await self.redis.set(
                cache_key,
                json.dumps(profile_dict),
                ex=1800  # 30 minutes
            )
        except Exception as e:
            logger.error(f"Error updating cached profile XP: {e}")
    
    async def _update_cached_profile_achievements(
        self, 
        user_id: str, 
        achievement_system: AchievementSystem
    ) -> None:
        """
        Update achievements in a cached gamification profile.
        
        Args:
            user_id: User identifier
            achievement_system: Updated achievement system
        """
        cache_key = f"gamification:profile:{user_id}"
        
        if not self.redis:
            return
            
        try:
            # Check if cached profile exists
            profile_data = await self.redis.get(cache_key)
            if not profile_data:
                return
                
            # Parse profile
            profile_dict = json.loads(profile_data)
            
            # Update achievement system
            profile_dict["achievement_system"] = achievement_system.to_dict()
            
            # Update timestamp
            profile_dict["updated_at"] = datetime.datetime.now().isoformat()
            
            # Store updated profile
            await self.redis.set(
                cache_key,
                json.dumps(profile_dict),
                ex=1800  # 30 minutes
            )
        except Exception as e:
            logger.error(f"Error updating cached profile achievements: {e}")
    
    def _get_leaderboard_key(self, leaderboard_type: LeaderboardType, id_suffix: str = "") -> str:
        """
        Get Redis key for a leaderboard.
        
        Args:
            leaderboard_type: Type of leaderboard
            id_suffix: Optional suffix for assessment-specific leaderboards
            
        Returns:
            Redis key
        """
        base_key = f"leaderboard:{leaderboard_type.value}"
        
        if id_suffix and leaderboard_type == LeaderboardType.ASSESSMENT_SPECIFIC:
            return f"{base_key}:{id_suffix}"
            
        return base_key
    
    def _get_leaderboard_ttl(self, leaderboard_type: LeaderboardType) -> int:
        """
        Get TTL for a leaderboard.
        
        Args:
            leaderboard_type: Type of leaderboard
            
        Returns:
            TTL in seconds
        """
        if leaderboard_type == LeaderboardType.DAILY:
            return 60 * 60 * 24 * 2  # 2 days
        elif leaderboard_type == LeaderboardType.WEEKLY:
            return 60 * 60 * 24 * 8  # 8 days
        elif leaderboard_type == LeaderboardType.MONTHLY:
            return 60 * 60 * 24 * 32  # 32 days
        else:
            return 60 * 60 * 24 * 90  # 90 days 