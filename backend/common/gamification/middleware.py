"""
Gamification Middleware Module

This module provides middleware for tracking gamification events
and updating user progress automatically during API requests.
"""

import logging
import time
from typing import Callable, Dict, Any, Optional

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from backend.common.logger import app_logger
from backend.common.auth import get_current_user
from backend.common.gamification.integration import get_gamification_event_handler

# Set up module logger
logger = app_logger.getChild("gamification.middleware")

class GamificationMiddleware(BaseHTTPMiddleware):
    """
    Middleware for tracking gamification events.
    
    This middleware tracks user activity and triggers
    gamification events like login streaks and active
    time tracking.
    """
    
    async def dispatch(
        self, request: Request, call_next: Callable
    ) -> Response:
        """
        Process the request and track gamification metrics.
        
        Args:
            request: The incoming request
            call_next: The next middleware in the chain
            
        Returns:
            The response from the next middleware
        """
        # Start timing
        start_time = time.time()
        
        # Process the request
        response = await call_next(request)
        
        # End timing
        processing_time = time.time() - start_time
        
        # Only track authenticated requests
        user = get_current_user()
        if user:
            try:
                # Track user activity
                event_handler = get_gamification_event_handler()
                await event_handler.record_activity(
                    user_id=user.id,
                    endpoint=str(request.url.path),
                    method=request.method,
                    response_time_ms=int(processing_time * 1000)
                )
            except Exception as e:
                # Log but don't fail the request
                logger.error(f"Error tracking gamification event: {e}")
        
        return response 