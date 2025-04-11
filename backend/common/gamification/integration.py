"""
Gamification Integration Module

This module provides integration between the assessment system and gamification features.
It handles events from assessments and triggers appropriate gamification actions.
"""

import logging
import datetime
import asyncio
from typing import Dict, Any, Optional, List, Union, Tuple

from backend.common.logger import app_logger
from backend.common.gamification.service import get_gamification_service
from backend.common.gamification.models import XPSource, LeaderboardType

from backend.assessments.base.models import (
    BaseQuestion, 
    AnswerEvaluation, 
    QuestionDifficulty,
    AssessmentSession,
    AssessmentType
)

# Set up module logger
logger = app_logger.getChild("gamification.integration")


class GamificationEventHandler:
    """
    Handles events from the assessment system and triggers gamification actions.
    
    This class serves as the bridge between the assessment system and the gamification
    features, acting as an event bus for gamification-related events.
    """
    
    def __init__(self):
        """Initialize the event handler."""
        self._service = get_gamification_service()
        
        # Track pending tasks
        self._pending_tasks = set()
        
    async def on_assessment_started(
        self,
        user_id: str,
        session_id: str,
        assessment_type: AssessmentType
    ) -> None:
        """
        Handle assessment started event.
        
        Args:
            user_id: User identifier
            session_id: Assessment session identifier
            assessment_type: Type of assessment
        """
        # No gamification actions needed at this point
        logger.info(f"Assessment started: user={user_id}, session={session_id}, type={assessment_type.value}")
    
    async def on_question_answered(
        self,
        user_id: str,
        session_id: str,
        question: BaseQuestion,
        evaluation: AnswerEvaluation,
        time_taken_ms: int
    ) -> Dict[str, Any]:
        """
        Handle question answered event.
        
        Args:
            user_id: User identifier
            session_id: Assessment session identifier
            question: The question that was answered
            evaluation: Evaluation of the answer
            time_taken_ms: Time taken to answer in milliseconds
            
        Returns:
            Dict with gamification results
        """
        try:
            # Award XP for question completion
            result = await self._service.award_question_completion_xp(
                user_id=user_id,
                question=question,
                evaluation=evaluation,
                time_taken_ms=time_taken_ms,
                session_id=session_id
            )
            
            # Check for pattern-specific achievements for candlestick patterns
            if (
                hasattr(question, 'pattern') and 
                question.pattern and 
                evaluation.is_correct
            ):
                await self._handle_pattern_identification(
                    user_id=user_id,
                    pattern_name=question.pattern,
                    difficulty=question.difficulty
                )
            
            return {
                "xp_awarded": result.get("amount", 0),
                "xp_breakdown": result.get("xp_breakdown", {}),
                "level_up": result.get("level_up", False),
                "current_level": result.get("current_level", 1),
                "next_level": result.get("next_level", 2),
                "progress_to_next": result.get("progress_to_next", 0)
            }
            
        except Exception as e:
            logger.error(f"Error handling question answered event: {e}")
            # Return minimal information on error
            return {
                "xp_awarded": 0,
                "error": str(e)
            }
    
    async def on_assessment_completed(
        self,
        user_id: str,
        session: AssessmentSession
    ) -> Dict[str, Any]:
        """
        Handle assessment completed event.
        
        Args:
            user_id: User identifier
            session: The completed assessment session
            
        Returns:
            Dict with gamification results
        """
        try:
            # Award XP for assessment completion
            result = await self._service.award_assessment_completion_xp(
                user_id=user_id,
                session=session
            )
            
            # Check for achievement eligibility
            achievements_task = asyncio.create_task(
                self._check_assessment_achievements(
                    user_id=user_id,
                    session=session
                )
            )
            self._track_task(achievements_task)
            
            # Update daily and weekly leaderboards
            leaderboard_task = asyncio.create_task(
                self._update_time_based_leaderboards(
                    user_id=user_id,
                    session=session
                )
            )
            self._track_task(leaderboard_task)
            
            return {
                "xp_awarded": result.get("amount", 0),
                "xp_breakdown": result.get("xp_breakdown", {}),
                "level_up": result.get("level_up", False),
                "current_level": result.get("current_level", 1),
                "next_level": result.get("next_level", 2),
                "progress_to_next": result.get("progress_to_next", 0)
            }
            
        except Exception as e:
            logger.error(f"Error handling assessment completed event: {e}")
            # Return minimal information on error
            return {
                "xp_awarded": 0,
                "error": str(e)
            }
    
    async def on_level_up(
        self,
        user_id: str,
        previous_level: int,
        current_level: int
    ) -> Dict[str, Any]:
        """
        Handle level up event.
        
        Args:
            user_id: User identifier
            previous_level: Previous level
            current_level: New level
            
        Returns:
            Dict with level up results
        """
        try:
            # Check for level-based achievements
            level_achievement_task = asyncio.create_task(
                self._check_level_achievements(
                    user_id=user_id,
                    level=current_level
                )
            )
            self._track_task(level_achievement_task)
            
            # Get level progress
            level_progress = await self._service.get_level_progress(user_id)
            
            return {
                "previous_level": previous_level,
                "current_level": current_level,
                "next_level": level_progress.get("next_level", current_level + 1),
                "progress_to_next": level_progress.get("progress_percent", 0),
                "xp_to_next": level_progress.get("remaining_xp", 0),
                "total_xp": level_progress.get("total_xp", 0)
            }
            
        except Exception as e:
            logger.error(f"Error handling level up event: {e}")
            # Return minimal information on error
            return {
                "previous_level": previous_level,
                "current_level": current_level,
                "error": str(e)
            }
    
    async def on_streak_update(
        self,
        user_id: str,
        streak_type: str,
        current_streak: int,
        longest_streak: int
    ) -> Dict[str, Any]:
        """
        Handle streak update event.
        
        Args:
            user_id: User identifier
            streak_type: Type of streak (e.g., 'login', 'correct_answers')
            current_streak: Current streak count
            longest_streak: Longest streak count
            
        Returns:
            Dict with streak update results
        """
        try:
            # Check for streak-based achievements
            streak_achievement_task = asyncio.create_task(
                self._check_streak_achievements(
                    user_id=user_id,
                    streak_type=streak_type,
                    streak_value=current_streak
                )
            )
            self._track_task(streak_achievement_task)
            
            return {
                "streak_type": streak_type,
                "current_streak": current_streak,
                "longest_streak": longest_streak
            }
            
        except Exception as e:
            logger.error(f"Error handling streak update event: {e}")
            # Return minimal information on error
            return {
                "streak_type": streak_type,
                "current_streak": current_streak,
                "error": str(e)
            }
    
    async def get_user_gamification_profile(
        self,
        user_id: str
    ) -> Dict[str, Any]:
        """
        Get a user's complete gamification profile.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dict with user's gamification profile
        """
        try:
            # Get basic profile from service
            profile = await self._service.get_profile(user_id)
            
            # Add extra information
            enriched_profile = {
                "user_id": profile.user_id,
                "xp": profile.xp_system.total_xp,
                "level": profile.xp_system.current_level,
                "level_progress": profile.xp_system.get_level_progress(),
                "achievements": {},
                "leaderboards": {}
            }
            
            # Get achievements with progress
            achievements = await self._service.get_achievements(
                user_id=user_id,
                include_hidden=False  # Hidden achievements not shown in profile
            )
            
            # Group by category
            achievement_categories = {}
            for achievement in achievements:
                category = achievement["category"]
                if category not in achievement_categories:
                    achievement_categories[category] = []
                achievement_categories[category].append(achievement)
            
            enriched_profile["achievements"] = {
                "total": len(achievements),
                "unlocked": sum(1 for a in achievements if a["is_unlocked"]),
                "by_category": achievement_categories
            }
            
            # Get leaderboard rankings
            leaderboard_rankings = {}
            
            # Daily leaderboard
            daily_rank = await self._service.get_user_rank(
                LeaderboardType.DAILY, user_id
            )
            if daily_rank:
                leaderboard_rankings["daily"] = {
                    "rank": daily_rank,
                    "context": await self._service.get_user_ranking_context(
                        LeaderboardType.DAILY, user_id
                    )
                }
            
            # Weekly leaderboard
            weekly_rank = await self._service.get_user_rank(
                LeaderboardType.WEEKLY, user_id
            )
            if weekly_rank:
                leaderboard_rankings["weekly"] = {
                    "rank": weekly_rank,
                    "context": await self._service.get_user_ranking_context(
                        LeaderboardType.WEEKLY, user_id
                    )
                }
            
            # All-time leaderboard
            alltime_rank = await self._service.get_user_rank(
                LeaderboardType.ALL_TIME, user_id
            )
            if alltime_rank:
                leaderboard_rankings["all_time"] = {
                    "rank": alltime_rank,
                    "context": await self._service.get_user_ranking_context(
                        LeaderboardType.ALL_TIME, user_id
                    )
                }
            
            enriched_profile["leaderboards"] = leaderboard_rankings
            
            # Get recent XP transactions
            recent_xp = await self._service.get_recent_xp_transactions(
                user_id=user_id,
                limit=10
            )
            enriched_profile["recent_xp"] = recent_xp
            
            return enriched_profile
            
        except Exception as e:
            logger.error(f"Error getting gamification profile: {e}")
            # Return minimal information on error
            return {
                "user_id": user_id,
                "error": str(e)
            }
    
    async def _check_assessment_achievements(
        self,
        user_id: str,
        session: AssessmentSession
    ) -> None:
        """
        Check for achievements related to assessment completion.
        
        Args:
            user_id: User identifier
            session: The completed assessment session
        """
        try:
            # Check for time-of-day achievement (Night Owl)
            current_hour = datetime.datetime.now().hour
            if 0 <= current_hour < 4:
                await self._service.unlock_achievement(
                    user_id=user_id,
                    achievement_id="night_owl",
                    metadata={
                        "session_id": session.id,
                        "hour": current_hour
                    }
                )
            
            # Check for weekend achievement
            current_day = datetime.datetime.now().weekday()
            if current_day >= 5:  # 5 = Saturday, 6 = Sunday
                # Weekend warrior is tracked separately via a streak tracker
                # This just ensures it's registered for the current weekend
                pass
            
            # Check for score improvement (for "comeback_king" achievement)
            # This would require previous session data, which would be handled
            # by a separate part of the system.
            
        except Exception as e:
            logger.error(f"Error checking assessment achievements: {e}")
    
    async def _check_level_achievements(
        self,
        user_id: str,
        level: int
    ) -> None:
        """
        Check for achievements related to reaching specific levels.
        
        Args:
            user_id: User identifier
            level: Current level
        """
        try:
            # Level-specific achievements
            level_achievements = {
                10: "level_10",
                25: "level_25",
                50: "level_50",
                100: "level_100"
            }
            
            # Unlock achievement if level matches
            if level in level_achievements:
                await self._service.unlock_achievement(
                    user_id=user_id,
                    achievement_id=level_achievements[level],
                    metadata={
                        "level": level,
                        "timestamp": datetime.datetime.now().isoformat()
                    }
                )
                
        except Exception as e:
            logger.error(f"Error checking level achievements: {e}")
    
    async def _check_streak_achievements(
        self,
        user_id: str,
        streak_type: str,
        streak_value: int
    ) -> None:
        """
        Check for achievements related to maintaining streaks.
        
        Args:
            user_id: User identifier
            streak_type: Type of streak
            streak_value: Current streak value
        """
        try:
            # Different streak types
            if streak_type == "login_streak":
                # Login streaks
                if streak_value >= 7:
                    await self._service.unlock_achievement(
                        user_id=user_id,
                        achievement_id="week_streak",
                        metadata={
                            "streak_value": streak_value
                        }
                    )
                
                if streak_value >= 30:
                    await self._service.unlock_achievement(
                        user_id=user_id,
                        achievement_id="month_streak",
                        metadata={
                            "streak_value": streak_value
                        }
                    )
                    
            elif streak_type == "correct_streak":
                # Correct answer streaks
                streak_achievements = {
                    5: "perfect_streak_5",
                    10: "perfect_streak_10",
                    25: "perfect_streak_25"
                }
                
                # Find the highest eligible achievement
                eligible_achievements = [
                    (threshold, achievement_id) 
                    for threshold, achievement_id in streak_achievements.items() 
                    if streak_value >= threshold
                ]
                
                # Unlock the highest eligible achievement
                if eligible_achievements:
                    threshold, achievement_id = max(eligible_achievements)
                    await self._service.unlock_achievement(
                        user_id=user_id,
                        achievement_id=achievement_id,
                        metadata={
                            "streak_value": streak_value
                        }
                    )
                    
            elif streak_type == "weekend_streak":
                # Weekend streaks
                if streak_value >= 5:
                    await self._service.unlock_achievement(
                        user_id=user_id,
                        achievement_id="weekend_warrior",
                        metadata={
                            "streak_value": streak_value
                        }
                    )
                    
        except Exception as e:
            logger.error(f"Error checking streak achievements: {e}")
    
    async def _handle_pattern_identification(
        self,
        user_id: str,
        pattern_name: str,
        difficulty: QuestionDifficulty
    ) -> None:
        """
        Handle pattern identification for achievement tracking.
        
        Args:
            user_id: User identifier
            pattern_name: Name of the pattern
            difficulty: Difficulty of the question
        """
        try:
            # Track pattern-specific achievements
            pattern_type = self._get_base_pattern_type(pattern_name.lower())
            
            if pattern_type:
                achievement_id = f"pattern_master_{pattern_type}"
                
                # Get achievement details
                achievement_system = await self._service._repository.get_achievement_system(user_id)
                user_achievement = achievement_system.get_achievement(achievement_id)
                
                if user_achievement:
                    # Update achievement progress
                    await self._service.update_achievement_progress(
                        user_id=user_id,
                        achievement_id=achievement_id,
                        progress=user_achievement.progress + 1,
                        metadata={
                            "pattern_name": pattern_name,
                            "difficulty": difficulty.value
                        }
                    )
                else:
                    # Initialize and update achievement
                    await self._service.update_achievement_progress(
                        user_id=user_id,
                        achievement_id=achievement_id,
                        progress=1,
                        metadata={
                            "pattern_name": pattern_name,
                            "difficulty": difficulty.value
                        }
                    )
                
                # Track for "all patterns" achievement (separate service would need to track this)
                
        except Exception as e:
            logger.error(f"Error handling pattern identification achievement: {e}")
    
    def _get_base_pattern_type(self, pattern_name: str) -> Optional[str]:
        """
        Get the base pattern type from a specific pattern name.
        
        Args:
            pattern_name: Full pattern name
            
        Returns:
            Base pattern type or None if not recognized
        """
        pattern_mapping = {
            # Doji patterns
            "doji": "doji",
            "dragonfly doji": "doji",
            "gravestone doji": "doji",
            "long-legged doji": "doji",
            "four price doji": "doji",
            
            # Hammer patterns
            "hammer": "hammer",
            "inverted hammer": "hammer",
            "hanging man": "hammer",
            "shooting star": "hammer",
            
            # Engulfing patterns
            "bullish engulfing": "engulfing",
            "bearish engulfing": "engulfing",
            
            # Star patterns
            "morning star": "star",
            "evening star": "star",
            "morning doji star": "star",
            "evening doji star": "star",
            
            # Harami patterns
            "bullish harami": "harami",
            "bearish harami": "harami",
            "bullish harami cross": "harami",
            "bearish harami cross": "harami",
        }
        
        # Try to find an exact match first
        if pattern_name in pattern_mapping:
            return pattern_mapping[pattern_name]
        
        # If not found, try to match based on substring
        for key, value in pattern_mapping.items():
            if key in pattern_name or pattern_name in key:
                return value
        
        return None
    
    async def _update_time_based_leaderboards(
        self,
        user_id: str,
        session: AssessmentSession
    ) -> None:
        """
        Update time-based leaderboards for a user.
        
        Args:
            user_id: User identifier
            session: The completed assessment session
        """
        try:
            score = session.score or 0
            
            # Update daily leaderboard
            await self._service.repository.update_leaderboard(
                leaderboard_type=LeaderboardType.DAILY,
                user_id=user_id,
                score=int(score),
                metadata={
                    "session_id": session.id,
                    "assessment_type": session.assessment_type.value
                }
            )
            
            # Update weekly leaderboard
            await self._service.repository.update_leaderboard(
                leaderboard_type=LeaderboardType.WEEKLY,
                user_id=user_id,
                score=int(score),
                metadata={
                    "session_id": session.id,
                    "assessment_type": session.assessment_type.value
                }
            )
            
            # Update all-time leaderboard
            await self._service.repository.update_leaderboard(
                leaderboard_type=LeaderboardType.ALL_TIME,
                user_id=user_id,
                score=int(score),
                metadata={
                    "session_id": session.id,
                    "assessment_type": session.assessment_type.value
                }
            )
            
            # Check for leaderboard achievements
            await self._check_leaderboard_achievements(user_id)
            
        except Exception as e:
            logger.error(f"Error updating time-based leaderboards: {e}")
    
    async def _check_leaderboard_achievements(self, user_id: str) -> None:
        """
        Check for achievements related to leaderboard positions.
        
        Args:
            user_id: User identifier
        """
        try:
            # Check weekly leaderboard rank
            weekly_rank = await self._service.get_user_rank(
                LeaderboardType.WEEKLY, user_id
            )
            
            if weekly_rank is not None:
                # Top position achievement
                if weekly_rank == 1:
                    await self._service.unlock_achievement(
                        user_id=user_id,
                        achievement_id="leaderboard_rank_1",
                        metadata={
                            "leaderboard_type": "weekly",
                            "rank": weekly_rank
                        }
                    )
                
                # Top 10 achievement
                if weekly_rank <= 10:
                    await self._service.unlock_achievement(
                        user_id=user_id,
                        achievement_id="leaderboard_top_10",
                        metadata={
                            "leaderboard_type": "weekly",
                            "rank": weekly_rank
                        }
                    )
                    
        except Exception as e:
            logger.error(f"Error checking leaderboard achievements: {e}")
    
    def _track_task(self, task: asyncio.Task) -> None:
        """
        Track an asyncio task to ensure completion and cleanup.
        
        Args:
            task: Task to track
        """
        self._pending_tasks.add(task)
        task.add_done_callback(self._remove_task)
    
    def _remove_task(self, task: asyncio.Task) -> None:
        """
        Remove a task from the tracking set and handle exceptions.
        
        Args:
            task: Task that has completed
        """
        self._pending_tasks.discard(task)
        
        # Handle exception if any
        if not task.cancelled() and task.exception():
            logger.error(f"Task raised exception: {task.exception()}")


# Singleton instance
_gamification_event_handler: Optional[GamificationEventHandler] = None


def get_gamification_event_handler() -> GamificationEventHandler:
    """
    Get the singleton gamification event handler instance.
    
    Returns:
        Gamification event handler instance
    """
    global _gamification_event_handler
    
    if _gamification_event_handler is None:
        _gamification_event_handler = GamificationEventHandler()
        
    return _gamification_event_handler 