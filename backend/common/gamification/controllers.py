"""
Gamification Controllers Module

This module provides API endpoints for gamification features, including:
- Retrieving user gamification profiles
- Getting leaderboard data
- Viewing achievements
"""

import logging
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from backend.common.logger import app_logger
from backend.common.auth import get_current_user, User
from backend.common.gamification.service import get_gamification_service
from backend.common.gamification.integration import get_gamification_event_handler
from backend.common.gamification.models import LeaderboardType

# Set up module logger
logger = app_logger.getChild("gamification.controllers")

# Create router
router = APIRouter(prefix="/gamification", tags=["Gamification"])


# Response models
class XPBreakdownResponse(BaseModel):
    base_xp: int = Field(..., description="Base XP amount")
    performance_bonus: Optional[int] = Field(None, description="Bonus XP for performance")
    time_bonus: Optional[int] = Field(None, description="Bonus XP for fast answers")
    difficulty_bonus: Optional[int] = Field(None, description="Bonus XP for difficulty")
    total: int = Field(..., description="Total XP awarded")


class LevelProgressResponse(BaseModel):
    current_level: int = Field(..., description="Current level")
    next_level: int = Field(..., description="Next level")
    current_xp: int = Field(..., description="Current XP")
    next_level_xp: int = Field(..., description="XP required for next level")
    xp_required: int = Field(..., description="Total XP required to reach next level")
    remaining_xp: int = Field(..., description="Remaining XP to next level")
    progress_percent: float = Field(..., description="Progress percentage to next level")


class LeaderboardEntryResponse(BaseModel):
    user_id: str = Field(..., description="User ID")
    username: Optional[str] = Field(None, description="Username")
    rank: int = Field(..., description="Rank on leaderboard")
    score: int = Field(..., description="User score")


class LeaderboardContextResponse(BaseModel):
    user_rank: Optional[int] = Field(None, description="User's rank")
    total_entries: int = Field(..., description="Total entries on leaderboard")
    entries: List[LeaderboardEntryResponse] = Field(..., description="Neighboring entries")


class AchievementResponse(BaseModel):
    id: str = Field(..., description="Achievement ID")
    name: str = Field(..., description="Achievement name")
    description: str = Field(..., description="Achievement description")
    rarity: str = Field(..., description="Achievement rarity")
    icon: str = Field(..., description="Achievement icon")
    category: str = Field(..., description="Achievement category")
    is_hidden: bool = Field(..., description="Whether the achievement is hidden")
    is_unlocked: bool = Field(..., description="Whether the achievement is unlocked")
    progress: int = Field(..., description="Current progress")
    progress_max: int = Field(..., description="Maximum progress")
    progress_percent: float = Field(..., description="Progress percentage")
    unlocked_at: Optional[str] = Field(None, description="When the achievement was unlocked")
    xp_reward: int = Field(..., description="XP reward for unlocking")


class XPTransactionResponse(BaseModel):
    id: str = Field(..., description="Transaction ID")
    amount: int = Field(..., description="XP amount")
    source: str = Field(..., description="XP source")
    description: str = Field(..., description="Transaction description")
    timestamp: str = Field(..., description="Transaction timestamp")
    assessment_id: Optional[str] = Field(None, description="Related assessment ID")


class GamificationProfileResponse(BaseModel):
    user_id: str = Field(..., description="User ID")
    xp: int = Field(..., description="Total XP")
    level: int = Field(..., description="Current level")
    level_progress: LevelProgressResponse = Field(..., description="Level progress")
    achievements: Dict[str, Any] = Field(..., description="Achievement information")
    leaderboards: Dict[str, Any] = Field(..., description="Leaderboard information")
    recent_xp: List[XPTransactionResponse] = Field(..., description="Recent XP transactions")


@router.get("/profile", response_model=GamificationProfileResponse)
async def get_gamification_profile(
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get the current user's gamification profile.
    
    Returns:
        Dict with user's gamification profile
    """
    try:
        event_handler = get_gamification_event_handler()
        profile = await event_handler.get_user_gamification_profile(current_user.id)
        
        return profile
        
    except Exception as e:
        logger.error(f"Error getting gamification profile: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving gamification profile: {str(e)}"
        )


@router.get("/achievements", response_model=List[AchievementResponse])
async def get_achievements(
    include_hidden: bool = Query(False, description="Whether to include hidden achievements"),
    current_user: User = Depends(get_current_user)
) -> List[Dict[str, Any]]:
    """
    Get the current user's achievements.
    
    Args:
        include_hidden: Whether to include hidden achievements
        
    Returns:
        List of achievements with progress
    """
    try:
        service = get_gamification_service()
        achievements = await service.get_achievements(
            user_id=current_user.id,
            include_hidden=include_hidden
        )
        
        return achievements
        
    except Exception as e:
        logger.error(f"Error getting achievements: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving achievements: {str(e)}"
        )


@router.get("/leaderboard/{board_type}", response_model=List[LeaderboardEntryResponse])
async def get_leaderboard(
    board_type: str,
    start: int = Query(0, description="Starting rank (0-indexed)"),
    count: int = Query(10, description="Number of entries to get"),
    assessment_id: Optional[str] = Query(None, description="Specific assessment ID for assessment leaderboards")
) -> List[Dict[str, Any]]:
    """
    Get entries from a leaderboard.
    
    Args:
        board_type: Type of leaderboard (daily, weekly, all_time, assessment)
        start: Starting rank (0-indexed)
        count: Number of entries to get
        assessment_id: Specific assessment ID for assessment leaderboards
        
    Returns:
        List of leaderboard entries
    """
    try:
        service = get_gamification_service()
        
        # Map string to enum value
        leaderboard_type_map = {
            "daily": LeaderboardType.DAILY,
            "weekly": LeaderboardType.WEEKLY,
            "all_time": LeaderboardType.ALL_TIME,
            "assessment": LeaderboardType.ASSESSMENT
        }
        
        if board_type not in leaderboard_type_map:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid leaderboard type: {board_type}"
            )
        
        leaderboard_type = leaderboard_type_map[board_type]
        
        # For assessment leaderboards, assessment_id is required
        id_suffix = ""
        if leaderboard_type == LeaderboardType.ASSESSMENT:
            if not assessment_id:
                raise HTTPException(
                    status_code=400,
                    detail="Assessment ID is required for assessment leaderboards"
                )
            id_suffix = assessment_id
        
        entries = await service.get_leaderboard(
            leaderboard_type=leaderboard_type,
            start=start,
            count=count,
            id_suffix=id_suffix
        )
        
        return entries
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting leaderboard: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving leaderboard: {str(e)}"
        )


@router.get("/leaderboard/{board_type}/context", response_model=LeaderboardContextResponse)
async def get_leaderboard_context(
    board_type: str,
    count: int = Query(5, description="Number of entries to get on each side"),
    assessment_id: Optional[str] = Query(None, description="Specific assessment ID for assessment leaderboards"),
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get user's context on a leaderboard.
    
    Args:
        board_type: Type of leaderboard (daily, weekly, all_time, assessment)
        count: Number of entries to get on each side
        assessment_id: Specific assessment ID for assessment leaderboards
        
    Returns:
        Dict with user's ranking context
    """
    try:
        service = get_gamification_service()
        
        # Map string to enum value
        leaderboard_type_map = {
            "daily": LeaderboardType.DAILY,
            "weekly": LeaderboardType.WEEKLY,
            "all_time": LeaderboardType.ALL_TIME,
            "assessment": LeaderboardType.ASSESSMENT
        }
        
        if board_type not in leaderboard_type_map:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid leaderboard type: {board_type}"
            )
        
        leaderboard_type = leaderboard_type_map[board_type]
        
        # For assessment leaderboards, assessment_id is required
        id_suffix = ""
        if leaderboard_type == LeaderboardType.ASSESSMENT:
            if not assessment_id:
                raise HTTPException(
                    status_code=400,
                    detail="Assessment ID is required for assessment leaderboards"
                )
            id_suffix = assessment_id
        
        context = await service.get_user_ranking_context(
            leaderboard_type=leaderboard_type,
            user_id=current_user.id,
            count=count,
            id_suffix=id_suffix
        )
        
        return context
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting leaderboard context: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving leaderboard context: {str(e)}"
        ) 