"""
WebSocket Helpers Module

This module provides helper functions for sending WebSocket messages
and real-time updates from different parts of the application.
"""

from typing import Dict, Any, List, Optional, Union
import asyncio
import json
from fastapi import FastAPI

from backend.realtime.manager import WebSocketManager
from backend.common.logger import get_logger

logger = get_logger(__name__)

async def get_websocket_manager(app: FastAPI) -> Optional[WebSocketManager]:
    """
    Get the WebSocketManager instance from the application state.
    
    Args:
        app: FastAPI application instance
        
    Returns:
        WebSocketManager instance or None if not found
    """
    if hasattr(app.state, "components") and "websocket_manager" in app.state.components:
        return app.state.components["websocket_manager"]
    return None

async def send_user_notification(
    app: FastAPI, 
    user_id: str, 
    notification_type: str, 
    data: Dict[str, Any]
) -> bool:
    """
    Send a real-time notification to a specific user.
    
    Args:
        app: FastAPI application instance
        user_id: ID of the user to notify
        notification_type: Type of notification (e.g., "assessment_feedback", "score_update")
        data: Notification data
        
    Returns:
        True if notification was sent successfully, False otherwise
    """
    manager = await get_websocket_manager(app)
    if not manager:
        logger.warning("WebSocket manager not available for sending user notification")
        return False
        
    message = {
        "type": notification_type,
        "timestamp": asyncio.get_event_loop().time(),
        "data": data
    }
    
    # Send to all user's connections
    sent_count = await manager.broadcast_to_user(message, user_id)
    return sent_count > 0

async def send_assessment_update(
    app: FastAPI,
    assessment_id: str,
    update_type: str,
    data: Dict[str, Any],
    user_id: Optional[str] = None
) -> bool:
    """
    Send a real-time update for an assessment.
    
    Args:
        app: FastAPI application instance
        assessment_id: ID of the assessment
        update_type: Type of update (e.g., "feedback", "time_update")
        data: Update data
        user_id: Optional user ID (if specified, sends only to this user)
        
    Returns:
        True if update was sent successfully, False otherwise
    """
    manager = await get_websocket_manager(app)
    if not manager:
        logger.warning("WebSocket manager not available for sending assessment update")
        return False
        
    message = {
        "type": update_type,
        "assessment_id": assessment_id,
        "timestamp": asyncio.get_event_loop().time(),
        "data": data
    }
    
    if user_id:
        # Send to specific user's assessment connection
        group = f"assessment:{assessment_id}:user:{user_id}"
        sent_count = await manager.broadcast_to_group(message, group)
    else:
        # Send to all users in this assessment
        group = f"assessment:{assessment_id}"
        sent_count = await manager.broadcast_to_group(message, group)
        
    return sent_count > 0

async def update_leaderboard(
    app: FastAPI,
    board_id: str,
    update_data: Union[Dict[str, Any], List[Dict[str, Any]]]
) -> bool:
    """
    Send a real-time update to a leaderboard.
    
    Args:
        app: FastAPI application instance
        board_id: ID of the leaderboard
        update_data: Leaderboard update data (new scores or rankings)
        
    Returns:
        True if update was sent successfully, False otherwise
    """
    manager = await get_websocket_manager(app)
    if not manager:
        logger.warning("WebSocket manager not available for updating leaderboard")
        return False
        
    message = {
        "type": "leaderboard_update",
        "board_id": board_id,
        "timestamp": asyncio.get_event_loop().time(),
        "data": update_data
    }
    
    # Send to all connections watching this leaderboard
    group = f"leaderboard:{board_id}"
    sent_count = await manager.broadcast_to_group(message, group)
    return sent_count > 0

async def broadcast_announcement(
    app: FastAPI,
    announcement: str,
    title: Optional[str] = None,
    link: Optional[str] = None,
    severity: str = "info"
) -> int:
    """
    Broadcast an announcement to all connected users.
    
    Args:
        app: FastAPI application instance
        announcement: Announcement text
        title: Optional announcement title
        link: Optional URL to include with announcement
        severity: Announcement severity ("info", "warning", "critical")
        
    Returns:
        Number of users who received the announcement
    """
    manager = await get_websocket_manager(app)
    if not manager:
        logger.warning("WebSocket manager not available for broadcasting announcement")
        return 0
        
    message = {
        "type": "announcement",
        "timestamp": asyncio.get_event_loop().time(),
        "data": {
            "text": announcement,
            "severity": severity
        }
    }
    
    if title:
        message["data"]["title"] = title
    if link:
        message["data"]["link"] = link
    
    # Broadcast to all connections
    sent_count = await manager.broadcast(message)
    return sent_count 