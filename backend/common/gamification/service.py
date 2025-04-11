"""
Gamification Service Module

This module provides the core service layer for gamification features:
1. XP and leveling
2. Achievements and badges
3. Leaderboards

It integrates with the assessment system to provide gamification functionality.
"""

import json
import logging
import asyncio
import datetime
from typing import Dict, List, Any, Optional, Set, Tuple, Union, Type, TypeVar

from backend.common.logger import app_logger
from backend.common.redis import get_redis_client
from redis.asyncio import Redis as AsyncRedis

from backend.assessments.base.models import (
    BaseQuestion, 
    AnswerEvaluation, 
    QuestionDifficulty,
    AssessmentSession
)

from backend.common.gamification.models import (
    XPSystem, AchievementSystem, GamificationProfile,
    Achievement, UserAchievement, LeaderboardEntry,
    XPSource, AchievementRarity, LeaderboardType
)

from backend.common.gamification.repository import GamificationRepository

# Set up module logger
logger = app_logger.getChild("gamification.service")


class GamificationService:
    """
    Service for gamification features.
    
    This class provides methods for managing XP, achievements, and leaderboards,
    as well as integration with the assessment system.
    """
    
    def __init__(
        self,
        repository: Optional[GamificationRepository] = None,
        redis_client: Optional[AsyncRedis] = None
    ):
        """
        Initialize the gamification service.
        
        Args:
            repository: Repository for gamification data
            redis_client: Redis client for leaderboards and caching
        """
        # Set up repository
        if repository:
            self.repository = repository
        else:
            redis = redis_client or get_redis_client()
            self.repository = GamificationRepository(redis_client=redis)
        
        # Load achievement definitions
        self._achievement_definitions: Dict[str, Achievement] = {}
        self._achievement_initialization_task = None
    
    async def initialize(self) -> None:
        """Initialize the service, loading necessary data."""
        # Load achievement definitions
        self._achievement_initialization_task = asyncio.create_task(
            self._load_achievement_definitions()
        )
    
    async def _load_achievement_definitions(self) -> None:
        """Load achievement definitions from the repository."""
        try:
            self._achievement_definitions = await self.repository.get_all_achievement_definitions()
            logger.info(f"Loaded {len(self._achievement_definitions)} achievement definitions")
        except Exception as e:
            logger.error(f"Error loading achievement definitions: {e}")
    
    async def get_profile(self, user_id: str) -> GamificationProfile:
        """
        Get a user's gamification profile.
        
        Args:
            user_id: User identifier
            
        Returns:
            User's gamification profile
        """
        return await self.repository.get_gamification_profile(user_id)
    
    async def add_xp(
        self,
        user_id: str,
        amount: int,
        source: XPSource,
        description: str,
        assessment_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Add XP to a user's account.
        
        Args:
            user_id: User identifier
            amount: Amount of XP to add
            source: Source of the XP
            description: Description of the transaction
            assessment_id: Optional ID of related assessment
            metadata: Optional additional metadata
            
        Returns:
            Dict with transaction results
        """
        if amount <= 0:
            raise ValueError("XP amount must be positive")
        
        # Get user's XP system
        xp_system = await self.repository.get_xp_system(user_id)
        
        # Add XP
        result = xp_system.add_xp(
            amount=amount,
            source=source,
            description=description,
            assessment_id=assessment_id,
            metadata=metadata
        )
        
        # Save XP system
        await self.repository.save_xp_system(xp_system)
        
        return result
    
    async def get_level_progress(self, user_id: str) -> Dict[str, Any]:
        """
        Get a user's level progress.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dict with level progress information
        """
        # Get user's XP system
        xp_system = await self.repository.get_xp_system(user_id)
        
        # Get level progress
        return xp_system.get_level_progress()
    
    async def get_recent_xp_transactions(
        self,
        user_id: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get a user's recent XP transactions.
        
        Args:
            user_id: User identifier
            limit: Maximum number of transactions to return
            
        Returns:
            List of recent transactions
        """
        # Get user's XP system
        xp_system = await self.repository.get_xp_system(user_id)
        
        # Get recent transactions
        transactions = xp_system.get_recent_transactions(limit)
        
        # Convert to dictionaries
        return [transaction.to_dict() for transaction in transactions]
    
    async def unlock_achievement(
        self,
        user_id: str,
        achievement_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Unlock an achievement for a user.
        
        Args:
            user_id: User identifier
            achievement_id: Achievement identifier
            metadata: Optional metadata about achievement completion
            
        Returns:
            Dict with achievement unlock results
        """
        # Get achievement definition
        achievement = await self.repository.get_achievement_definition(achievement_id)
        if not achievement:
            raise ValueError(f"Achievement {achievement_id} not found")
        
        # Get user's achievement system
        achievement_system = await self.repository.get_achievement_system(user_id)
        
        # Initialize achievement if needed
        user_achievement = achievement_system.get_achievement(achievement_id)
        if not user_achievement:
            user_achievement = achievement_system.initialize_achievement(
                achievement_id=achievement_id,
                progress_max=1
            )
        
        # Check if already unlocked
        if user_achievement.is_unlocked:
            return {
                "achievement_id": achievement_id,
                "name": achievement.name,
                "already_unlocked": True,
                "xp_awarded": 0
            }
        
        # Update achievement progress to max to unlock
        update_result = achievement_system.update_achievement(
            achievement_id=achievement_id,
            progress=user_achievement.progress_max,
            metadata=metadata
        )
        
        # Save achievement system
        await self.repository.save_achievement_system(achievement_system)
        
        # Award XP for unlock
        xp_result = await self.add_xp(
            user_id=user_id,
            amount=achievement.xp_reward,
            source=XPSource.ACHIEVEMENT_UNLOCK,
            description=f"Unlocked achievement: {achievement.name}",
            metadata={
                "achievement_id": achievement_id,
                "achievement_name": achievement.name,
                "achievement_rarity": achievement.rarity.value
            }
        )
        
        return {
            "achievement_id": achievement_id,
            "name": achievement.name,
            "newly_unlocked": update_result["newly_unlocked"],
            "xp_awarded": achievement.xp_reward,
            "level_up": xp_result.get("level_up", False)
        }
    
    async def update_achievement_progress(
        self,
        user_id: str,
        achievement_id: str,
        progress: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Update progress for an achievement.
        
        Args:
            user_id: User identifier
            achievement_id: Achievement identifier
            progress: New progress value
            metadata: Optional metadata
            
        Returns:
            Dict with achievement update results
        """
        # Get achievement definition
        achievement = await self.repository.get_achievement_definition(achievement_id)
        if not achievement:
            raise ValueError(f"Achievement {achievement_id} not found")
        
        # Get user's achievement system
        achievement_system = await self.repository.get_achievement_system(user_id)
        
        # Initialize achievement if needed
        user_achievement = achievement_system.get_achievement(achievement_id)
        if not user_achievement:
            progress_max = achievement.criteria.get("target_count", 1)
            user_achievement = achievement_system.initialize_achievement(
                achievement_id=achievement_id,
                progress_max=progress_max
            )
        
        # Update achievement progress
        update_result = achievement_system.update_achievement(
            achievement_id=achievement_id,
            progress=progress,
            metadata=metadata
        )
        
        # Save achievement system
        await self.repository.save_achievement_system(achievement_system)
        
        # Award XP if newly unlocked
        xp_result = {}
        if update_result["newly_unlocked"]:
            xp_result = await self.add_xp(
                user_id=user_id,
                amount=achievement.xp_reward,
                source=XPSource.ACHIEVEMENT_UNLOCK,
                description=f"Unlocked achievement: {achievement.name}",
                metadata={
                    "achievement_id": achievement_id,
                    "achievement_name": achievement.name,
                    "achievement_rarity": achievement.rarity.value
                }
            )
        
        return {
            **update_result,
            "achievement_id": achievement_id,
            "name": achievement.name,
            "xp_awarded": achievement.xp_reward if update_result["newly_unlocked"] else 0,
            "level_up": xp_result.get("level_up", False) if update_result["newly_unlocked"] else False
        }
    
    async def get_achievements(
        self,
        user_id: str,
        include_hidden: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Get a user's achievements with progress.
        
        Args:
            user_id: User identifier
            include_hidden: Whether to include hidden achievements
            
        Returns:
            List of achievements with progress
        """
        # Ensure achievement definitions are loaded
        if self._achievement_initialization_task and not self._achievement_initialization_task.done():
            await self._achievement_initialization_task
        
        # Get user's achievement system
        achievement_system = await self.repository.get_achievement_system(user_id)
        
        # Get all achievements with progress
        result = []
        for achievement_id, definition in self._achievement_definitions.items():
            # Skip hidden achievements if not included
            if definition.is_hidden and not include_hidden:
                continue
                
            # Get user's progress
            user_achievement = achievement_system.get_achievement(achievement_id)
            if user_achievement:
                # Existing achievement
                result.append({
                    "id": achievement_id,
                    "name": definition.name,
                    "description": definition.description,
                    "rarity": definition.rarity.value,
                    "icon": definition.icon,
                    "category": definition.category,
                    "is_hidden": definition.is_hidden,
                    "is_unlocked": user_achievement.is_unlocked,
                    "progress": user_achievement.progress,
                    "progress_max": user_achievement.progress_max,
                    "progress_percent": user_achievement.progress_percent,
                    "unlocked_at": user_achievement.unlocked_at.isoformat() if user_achievement.unlocked_at else None,
                    "xp_reward": definition.xp_reward
                })
            else:
                # Achievement not initialized yet
                progress_max = definition.criteria.get("target_count", 1)
                result.append({
                    "id": achievement_id,
                    "name": definition.name,
                    "description": definition.description,
                    "rarity": definition.rarity.value,
                    "icon": definition.icon,
                    "category": definition.category,
                    "is_hidden": definition.is_hidden,
                    "is_unlocked": False,
                    "progress": 0,
                    "progress_max": progress_max,
                    "progress_percent": 0,
                    "unlocked_at": None,
                    "xp_reward": definition.xp_reward
                })
        
        return result
    
    async def get_achievement_categories(self) -> Dict[str, List[str]]:
        """
        Get achievement categories with IDs.
        
        Returns:
            Dict mapping categories to lists of achievement IDs
        """
        # Ensure achievement definitions are loaded
        if self._achievement_initialization_task and not self._achievement_initialization_task.done():
            await self._achievement_initialization_task
        
        # Group achievements by category
        categories = {}
        for achievement_id, definition in self._achievement_definitions.items():
            category = definition.category
            if category not in categories:
                categories[category] = []
            categories[category].append(achievement_id)
        
        return categories
    
    async def get_leaderboard(
        self,
        leaderboard_type: LeaderboardType,
        start: int = 0,
        count: int = 10,
        id_suffix: str = ""
    ) -> List[Dict[str, Any]]:
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
        entries = await self.repository.get_leaderboard(
            leaderboard_type=leaderboard_type,
            start=start,
            count=count,
            id_suffix=id_suffix
        )
        
        return [entry.to_dict() for entry in entries]
    
    async def get_user_rank(
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
        return await self.repository.get_user_leaderboard_rank(
            leaderboard_type=leaderboard_type,
            user_id=user_id,
            id_suffix=id_suffix
        )
    
    async def get_user_ranking_context(
        self,
        leaderboard_type: LeaderboardType,
        user_id: str,
        count: int = 5,
        id_suffix: str = ""
    ) -> Dict[str, Any]:
        """
        Get a user's ranking context on a leaderboard.
        
        This includes the user's entry, their rank, and entries around them.
        
        Args:
            leaderboard_type: Type of leaderboard
            user_id: User identifier
            count: Number of entries to get on each side
            id_suffix: Optional suffix for assessment-specific leaderboards
            
        Returns:
            Dict with user's ranking context
        """
        # Get user's rank
        rank = await self.repository.get_user_leaderboard_rank(
            leaderboard_type=leaderboard_type,
            user_id=user_id,
            id_suffix=id_suffix
        )
        
        if rank is None:
            return {
                "user_rank": None,
                "total_entries": await self._get_leaderboard_count(leaderboard_type, id_suffix),
                "entries": []
            }
        
        # Get entries around user
        entries = await self.repository.get_neighboring_leaderboard_entries(
            leaderboard_type=leaderboard_type,
            user_id=user_id,
            count=count,
            id_suffix=id_suffix
        )
        
        return {
            "user_rank": rank,
            "total_entries": await self._get_leaderboard_count(leaderboard_type, id_suffix),
            "entries": [entry.to_dict() for entry in entries]
        }
    
    async def _get_leaderboard_count(
        self,
        leaderboard_type: LeaderboardType,
        id_suffix: str = ""
    ) -> int:
        """
        Get the number of entries in a leaderboard.
        
        Args:
            leaderboard_type: Type of leaderboard
            id_suffix: Optional suffix for assessment-specific leaderboards
            
        Returns:
            Number of entries
        """
        try:
            leaderboard_key = self.repository._get_leaderboard_key(leaderboard_type, id_suffix)
            count = await self.repository.redis.zcard(leaderboard_key)
            return count or 0
        except Exception as e:
            logger.error(f"Error getting leaderboard count: {e}")
            return 0
    
    #
    # Assessment Integration Methods
    #
    
    async def award_question_completion_xp(
        self,
        user_id: str,
        question: BaseQuestion,
        evaluation: AnswerEvaluation,
        time_taken_ms: int,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Award XP for completing a question.
        
        Args:
            user_id: User identifier
            question: The question that was answered
            evaluation: Evaluation of the answer
            time_taken_ms: Time taken to answer in milliseconds
            session_id: Optional session identifier
            
        Returns:
            Dict with XP award results
        """
        # Calculate base XP based on difficulty
        base_xp = self._calculate_question_base_xp(question.difficulty)
        
        # Adjust for correctness
        if evaluation.is_correct:
            correctness_multiplier = 1.0
        else:
            # Still give some XP for trying, especially if close
            correctness_multiplier = max(0.1, evaluation.score * 0.2)
        
        # Adjust for speed (faster = more XP, up to 50% bonus)
        # This would be more sophisticated in production with per-question time targets
        time_bonus = 0
        if evaluation.is_correct and time_taken_ms > 0:
            avg_time_ms = 20000  # 20 seconds
            if time_taken_ms < avg_time_ms:
                # Up to 50% bonus for fast answers
                time_factor = max(0, (avg_time_ms - time_taken_ms) / avg_time_ms)
                time_bonus = int(base_xp * time_factor * 0.5)
        
        # Calculate final XP
        xp_amount = int(base_xp * correctness_multiplier) + time_bonus
        
        # Award XP
        metadata = {
            "question_id": question.id,
            "difficulty": question.difficulty.value,
            "is_correct": evaluation.is_correct,
            "score": evaluation.score,
            "time_taken_ms": time_taken_ms,
            "time_bonus": time_bonus
        }
        
        description = f"Answered {question.difficulty.value} question"
        if evaluation.is_correct:
            description += " correctly"
        
        result = await self.add_xp(
            user_id=user_id,
            amount=xp_amount,
            source=XPSource.QUESTION_COMPLETION,
            description=description,
            assessment_id=session_id,
            metadata=metadata
        )
        
        # Track for achievements
        await self._update_question_achievements(
            user_id=user_id,
            question=question,
            evaluation=evaluation,
            time_taken_ms=time_taken_ms
        )
        
        return {
            **result,
            "xp_breakdown": {
                "base_xp": base_xp,
                "correctness_multiplier": correctness_multiplier,
                "time_bonus": time_bonus,
                "total": xp_amount
            }
        }
    
    async def award_assessment_completion_xp(
        self,
        user_id: str,
        session: AssessmentSession
    ) -> Dict[str, Any]:
        """
        Award XP for completing an assessment.
        
        Args:
            user_id: User identifier
            session: The completed assessment session
            
        Returns:
            Dict with XP award results
        """
        # Base XP for completing the assessment
        base_xp = 50
        
        # Performance bonus (0-100% of base XP)
        if session.score is not None:
            performance_bonus = int(base_xp * (session.score / 100))
        else:
            performance_bonus = 0
        
        # Difficulty bonus
        difficulty_counts = {}
        total_difficulty_bonus = 0
        
        for question_id in session.questions:
            if question_id in session.answers and session.answers[question_id].evaluation:
                question_eval = session.answers[question_id].evaluation
                if question_eval.metadata and "difficulty" in question_eval.metadata:
                    difficulty = QuestionDifficulty(question_eval.metadata["difficulty"])
                    if difficulty not in difficulty_counts:
                        difficulty_counts[difficulty] = 0
                    difficulty_counts[difficulty] += 1
                    
                    # Add bonus for harder questions
                    if difficulty == QuestionDifficulty.HARD:
                        total_difficulty_bonus += 5
                    elif difficulty == QuestionDifficulty.VERY_HARD:
                        total_difficulty_bonus += 10
        
        # Calculate total XP
        total_xp = base_xp + performance_bonus + total_difficulty_bonus
        
        # Award XP
        metadata = {
            "session_id": session.id,
            "assessment_type": session.assessment_type.value,
            "score": session.score,
            "questions_answered": session.answered_count,
            "correct_count": session.correct_count,
            "performance_bonus": performance_bonus,
            "difficulty_bonus": total_difficulty_bonus,
            "difficulty_counts": {d.value: c for d, c in difficulty_counts.items()}
        }
        
        result = await self.add_xp(
            user_id=user_id,
            amount=total_xp,
            source=XPSource.ASSESSMENT_COMPLETION,
            description=f"Completed {session.assessment_type.value} assessment",
            assessment_id=session.id,
            metadata=metadata
        )
        
        # Update assessment leaderboard
        await self.repository.update_assessment_leaderboard(
            assessment_id=session.id,
            user_id=user_id,
            score=int(session.score) if session.score is not None else 0,
            metadata={
                "assessment_type": session.assessment_type.value,
                "questions_answered": session.answered_count,
                "correct_count": session.correct_count
            }
        )
        
        # Track for achievements
        await self._update_assessment_achievements(
            user_id=user_id,
            session=session
        )
        
        return {
            **result,
            "xp_breakdown": {
                "base_xp": base_xp,
                "performance_bonus": performance_bonus,
                "difficulty_bonus": total_difficulty_bonus,
                "total": total_xp
            }
        }
    
    def _calculate_question_base_xp(self, difficulty: QuestionDifficulty) -> int:
        """
        Calculate base XP for a question based on difficulty.
        
        Args:
            difficulty: Question difficulty
            
        Returns:
            Base XP amount
        """
        return {
            QuestionDifficulty.VERY_EASY: 5,
            QuestionDifficulty.EASY: 10,
            QuestionDifficulty.MEDIUM: 15,
            QuestionDifficulty.HARD: 20,
            QuestionDifficulty.VERY_HARD: 30
        }[difficulty]
    
    async def _update_question_achievements(
        self,
        user_id: str,
        question: BaseQuestion,
        evaluation: AnswerEvaluation,
        time_taken_ms: int
    ) -> None:
        """
        Update achievements related to answering questions.
        
        Args:
            user_id: User identifier
            question: The question that was answered
            evaluation: Evaluation of the answer
            time_taken_ms: Time taken to answer in milliseconds
        """
        # Get achievement system
        achievement_system = await self.repository.get_achievement_system(user_id)
        updates = []
        
        # Track total questions answered
        questions_answered_achievement = achievement_system.get_achievement("questions_answered")
        if questions_answered_achievement:
            current_progress = questions_answered_achievement.progress
            updates.append((
                "questions_answered", 
                current_progress + 1,
                {"last_question": question.id}
            ))
        
        # Track correct answers if applicable
        if evaluation.is_correct:
            correct_answers_achievement = achievement_system.get_achievement("correct_answers")
            if correct_answers_achievement:
                current_progress = correct_answers_achievement.progress
                updates.append((
                    "correct_answers", 
                    current_progress + 1,
                    {"last_question": question.id}
                ))
            
            # Track streak (requires separate streak tracking)
            # Here we assume there's another service tracking streaks
            
            # Track difficulty-specific achievements
            if question.difficulty == QuestionDifficulty.HARD:
                hard_questions_achievement = achievement_system.get_achievement("master_hard_questions")
                if hard_questions_achievement:
                    current_progress = hard_questions_achievement.progress
                    updates.append((
                        "master_hard_questions", 
                        current_progress + 1,
                        {"last_question": question.id}
                    ))
            
            elif question.difficulty == QuestionDifficulty.VERY_HARD:
                very_hard_questions_achievement = achievement_system.get_achievement("master_very_hard_questions")
                if very_hard_questions_achievement:
                    current_progress = very_hard_questions_achievement.progress
                    updates.append((
                        "master_very_hard_questions", 
                        current_progress + 1,
                        {"last_question": question.id}
                    ))
            
            # Track fast answer achievements
            if time_taken_ms < 5000:  # 5 seconds
                fast_answers_achievement = achievement_system.get_achievement("speed_demon")
                if fast_answers_achievement:
                    current_progress = fast_answers_achievement.progress
                    updates.append((
                        "speed_demon", 
                        current_progress + 1,
                        {"last_question": question.id, "time_taken_ms": time_taken_ms}
                    ))
        
        # Apply updates
        for achievement_id, progress, metadata in updates:
            try:
                await self.update_achievement_progress(
                    user_id=user_id,
                    achievement_id=achievement_id,
                    progress=progress,
                    metadata=metadata
                )
            except Exception as e:
                logger.error(f"Error updating achievement {achievement_id}: {e}")
    
    async def _update_assessment_achievements(
        self,
        user_id: str,
        session: AssessmentSession
    ) -> None:
        """
        Update achievements related to completing assessments.
        
        Args:
            user_id: User identifier
            session: The completed assessment session
        """
        # Get achievement system
        achievement_system = await self.repository.get_achievement_system(user_id)
        updates = []
        
        # Track total assessments completed
        assessments_completed_achievement = achievement_system.get_achievement("assessments_completed")
        if assessments_completed_achievement:
            current_progress = assessments_completed_achievement.progress
            updates.append((
                "assessments_completed", 
                current_progress + 1,
                {"last_assessment": session.id}
            ))
        
        # Track perfect assessments
        if session.score == 100:
            perfect_assessments_achievement = achievement_system.get_achievement("perfect_assessments")
            if perfect_assessments_achievement:
                current_progress = perfect_assessments_achievement.progress
                updates.append((
                    "perfect_assessments", 
                    current_progress + 1,
                    {"last_assessment": session.id}
                ))
        
        # Track assessment type specific achievements
        assessment_type = session.assessment_type.value
        type_achievement_id = f"{assessment_type}_specialist"
        type_achievement = achievement_system.get_achievement(type_achievement_id)
        if type_achievement:
            current_progress = type_achievement.progress
            updates.append((
                type_achievement_id, 
                current_progress + 1,
                {"last_assessment": session.id}
            ))
        
        # Apply updates
        for achievement_id, progress, metadata in updates:
            try:
                await self.update_achievement_progress(
                    user_id=user_id,
                    achievement_id=achievement_id,
                    progress=progress,
                    metadata=metadata
                )
            except Exception as e:
                logger.error(f"Error updating achievement {achievement_id}: {e}")


# Singleton instance
_gamification_service: Optional[GamificationService] = None


def get_gamification_service() -> GamificationService:
    """
    Get the singleton gamification service instance.
    
    Returns:
        Gamification service instance
    """
    global _gamification_service
    
    if _gamification_service is None:
        _gamification_service = GamificationService()
        
    return _gamification_service


async def initialize_gamification_service() -> None:
    """Initialize the gamification service."""
    service = get_gamification_service()
    await service.initialize() 