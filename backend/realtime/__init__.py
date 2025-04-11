"""
TradeIQ Real-time Updates Module

This module provides WebSocket-based real-time updates for the TradeIQ assessment platform,
including live feedback during assessments and real-time leaderboard updates.
"""

from backend.realtime.manager import WebSocketManager
from backend.realtime.controllers import register_websocket_routes
from backend.realtime.helpers import (
    send_user_notification,
    send_assessment_update,
    update_leaderboard,
    broadcast_announcement
)

__all__ = [
    "WebSocketManager", 
    "register_websocket_routes",
    "send_user_notification",
    "send_assessment_update",
    "update_leaderboard",
    "broadcast_announcement"
] 