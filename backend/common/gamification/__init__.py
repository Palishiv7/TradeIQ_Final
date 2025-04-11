"""
Gamification Package

This package provides gamification features for the assessment platform:
- XP and leveling system
- Achievements and badges
- Leaderboards

It integrates with the assessment system to provide rewards and feedback based on user performance.
"""

import logging
import asyncio
from typing import Optional

from backend.common.logger import app_logger
from backend.common.redis import get_redis_client
from backend.common.gamification.service import initialize_gamification_service
from backend.common.gamification.repository import GamificationRepository
from backend.common.gamification.achievements import get_default_achievements

# Set up module logger
logger = app_logger.getChild("gamification")


async def initialize_gamification_system() -> None:
    """
    Initialize the gamification system.
    
    This sets up all necessary components for the gamification system,
    including loading achievement definitions, setting up repositories,
    and configuring event handlers.
    """
    logger.info("Initializing gamification system...")
    
    try:
        # Initialize gamification service (singleton)
        await initialize_gamification_service()
        logger.info("Gamification service initialized")
        
        # Load default achievements into repository
        redis_client = get_redis_client()
        repository = GamificationRepository(redis_client=redis_client)
        
        # Get default achievements
        achievements = get_default_achievements()
        
        # Store achievements in repository
        for achievement_id, achievement in achievements.items():
            await repository.save_achievement_definition(achievement)
        
        logger.info(f"Loaded {len(achievements)} default achievements")
        
        logger.info("Gamification system initialized successfully")
        
    except Exception as e:
        logger.error(f"Error initializing gamification system: {e}")
        raise 