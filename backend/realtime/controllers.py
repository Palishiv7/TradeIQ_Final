"""
WebSocket Controllers Module

This module provides the WebSocket route handlers and endpoint definitions
for the real-time communication features of the TradeIQ platform.
"""

from typing import Dict, Any, Optional, List, Callable
import json
import asyncio
from contextlib import asynccontextmanager

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, Query, Path, Header
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.websockets import WebSocketState

from backend.realtime.manager import WebSocketManager
from backend.common.security import get_current_user, verify_token
from backend.common.rate_limiter import RateLimiter
from backend.common.logger import get_logger

logger = get_logger(__name__)

# Initialize security
security = HTTPBearer(auto_error=False)

# WebSocket routes
ws_router = APIRouter()

async def get_websocket_manager(websocket: WebSocket) -> WebSocketManager:
    """
    Get the WebSocketManager instance from the application state.
    
    Args:
        websocket: The WebSocket connection
        
    Returns:
        WebSocketManager instance
    """
    return websocket.app.state.components.get("websocket_manager")

async def get_token_from_query(
    websocket: WebSocket,
    token: Optional[str] = Query(None)
) -> Optional[str]:
    """
    Extract token from query parameters.
    
    Args:
        websocket: The WebSocket connection
        token: Authorization token from query parameters
        
    Returns:
        The token if found, None otherwise
    """
    return token

async def get_user_from_token(
    websocket: WebSocket,
    token: Optional[str] = Depends(get_token_from_query)
) -> Optional[Dict[str, Any]]:
    """
    Get user from token for WebSocket connections.
    
    Args:
        websocket: The WebSocket connection
        token: Authorization token
        
    Returns:
        User data if token is valid, None otherwise
    """
    if not token:
        return None
        
    try:
        # Verify the token and get user data
        user_data = await verify_token(token)
        return user_data
    except Exception as e:
        logger.warning(f"Invalid token in WebSocket connection: {str(e)}")
        return None

@ws_router.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    manager: WebSocketManager = Depends(get_websocket_manager),
    user: Optional[Dict[str, Any]] = Depends(get_user_from_token)
):
    """
    Main WebSocket endpoint for general purpose connections.
    
    This endpoint handles general-purpose WebSocket connections and
    basic message routing.
    """
    # Accept the connection
    user_id = user.get("sub") if user else None
    connection_id = await manager.connect(websocket, user_id)
    
    try:
        # Handle messages
        while True:
            # Wait for messages from client
            data = await websocket.receive_text()
            
            try:
                # Parse the message
                message = json.loads(data)
                
                # Extract message type and handle it
                message_type = message.get("type")
                
                if message_type == "ping":
                    # Simple ping-pong for connection health check
                    await manager.send_personal_message({"type": "pong"}, connection_id)
                    
                elif message_type == "join_group":
                    # Handle group join requests
                    group = message.get("group")
                    if group:
                        await manager.add_to_group(connection_id, group)
                        await manager.send_personal_message(
                            {"type": "joined_group", "group": group},
                            connection_id
                        )
                        
                elif message_type == "leave_group":
                    # Handle group leave requests
                    group = message.get("group")
                    if group:
                        await manager.remove_from_group(connection_id, group)
                        await manager.send_personal_message(
                            {"type": "left_group", "group": group},
                            connection_id
                        )
                        
                elif message_type == "message":
                    # Handle direct messages to other users or groups
                    target = message.get("target")
                    target_type = message.get("target_type", "user")
                    content = message.get("content")
                    
                    if not target or not content:
                        await manager.send_personal_message(
                            {"type": "error", "message": "Invalid message format"},
                            connection_id
                        )
                        continue
                        
                    # Add sender info if user is authenticated
                    if user:
                        message["sender"] = {
                            "id": user.get("sub"),
                            "name": user.get("name", "Anonymous")
                        }
                        
                    # Route the message based on target type
                    if target_type == "user":
                        await manager.broadcast_to_user(message, target)
                    elif target_type == "group":
                        await manager.broadcast_to_group(message, target)
                    else:
                        await manager.send_personal_message(
                            {"type": "error", "message": "Invalid target type"},
                            connection_id
                        )
                        
                else:
                    # Publish event for custom handlers
                    if message_type:
                        await manager.publish_event(message_type, message)
                        
            except json.JSONDecodeError:
                # Handle invalid JSON
                await manager.send_personal_message(
                    {"type": "error", "message": "Invalid JSON format"},
                    connection_id
                )
            except Exception as e:
                # Handle other errors
                logger.error(f"Error processing WebSocket message: {str(e)}")
                await manager.send_personal_message(
                    {"type": "error", "message": "Internal server error"},
                    connection_id
                )
                
    except WebSocketDisconnect:
        # Handle client disconnection
        logger.debug(f"WebSocket client disconnected: {connection_id}")
    except Exception as e:
        # Handle other errors
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        # Clean up connection
        await manager.disconnect(connection_id)
        
@ws_router.websocket("/ws/assessment/{assessment_id}")
async def assessment_websocket_endpoint(
    websocket: WebSocket,
    assessment_id: str = Path(...),
    manager: WebSocketManager = Depends(get_websocket_manager),
    user: Optional[Dict[str, Any]] = Depends(get_user_from_token)
):
    """
    WebSocket endpoint for assessment-specific real-time updates.
    
    This endpoint provides real-time feedback and updates for users
    taking specific assessments.
    
    Args:
        websocket: WebSocket connection
        assessment_id: ID of the assessment
        manager: WebSocketManager instance
        user: Authenticated user data
    """
    # Require authentication for assessment WebSockets
    if not user:
        await websocket.close(code=1008, reason="Authentication required")
        return
        
    user_id = user.get("sub")
    connection_id = await manager.connect(websocket, user_id)
    
    try:
        # Add to assessment-specific group
        assessment_group = f"assessment:{assessment_id}"
        user_assessment_group = f"assessment:{assessment_id}:user:{user_id}"
        
        await manager.add_to_group(connection_id, assessment_group)
        await manager.add_to_group(connection_id, user_assessment_group)
        
        # Send welcome message
        await manager.send_personal_message(
            {
                "type": "assessment_connected",
                "assessment_id": assessment_id,
                "message": "Connected to assessment real-time updates"
            },
            connection_id
        )
        
        # Handle messages
        while True:
            data = await websocket.receive_text()
            
            try:
                message = json.loads(data)
                
                # Forward assessment-specific events
                if "type" in message:
                    message["assessment_id"] = assessment_id
                    message["user_id"] = user_id
                    
                    # Publish the event
                    await manager.publish_event(f"assessment:{message['type']}", message)
                    
            except json.JSONDecodeError:
                await manager.send_personal_message(
                    {"type": "error", "message": "Invalid JSON format"},
                    connection_id
                )
            except Exception as e:
                logger.error(f"Error in assessment WebSocket: {str(e)}")
                
    except WebSocketDisconnect:
        logger.debug(f"Assessment WebSocket disconnected: {connection_id}")
    except Exception as e:
        logger.error(f"Assessment WebSocket error: {str(e)}")
    finally:
        # Clean up connection
        await manager.disconnect(connection_id)

@ws_router.websocket("/ws/leaderboard/{board_id}")
async def leaderboard_websocket_endpoint(
    websocket: WebSocket,
    board_id: str = Path(...),
    manager: WebSocketManager = Depends(get_websocket_manager),
    user: Optional[Dict[str, Any]] = Depends(get_user_from_token)
):
    """
    WebSocket endpoint for leaderboard real-time updates.
    
    This endpoint provides real-time updates for leaderboards, allowing
    users to see score changes as they happen.
    
    Args:
        websocket: WebSocket connection
        board_id: ID of the leaderboard
        manager: WebSocketManager instance
        user: Authenticated user data (optional)
    """
    # Authentication is optional for leaderboards
    user_id = user.get("sub") if user else None
    connection_id = await manager.connect(websocket, user_id)
    
    try:
        # Add to leaderboard group
        leaderboard_group = f"leaderboard:{board_id}"
        await manager.add_to_group(connection_id, leaderboard_group)
        
        # Send welcome message
        await manager.send_personal_message(
            {
                "type": "leaderboard_connected",
                "board_id": board_id,
                "message": "Connected to leaderboard real-time updates"
            },
            connection_id
        )
        
        # Handle messages (mainly keep-alive for leaderboard)
        while True:
            data = await websocket.receive_text()
            
            try:
                message = json.loads(data)
                
                # Only handle ping messages for leaderboard connections
                if message.get("type") == "ping":
                    await manager.send_personal_message({"type": "pong"}, connection_id)
                    
            except json.JSONDecodeError:
                pass  # Ignore invalid JSON for leaderboard connections
                
    except WebSocketDisconnect:
        logger.debug(f"Leaderboard WebSocket disconnected: {connection_id}")
    except Exception as e:
        logger.error(f"Leaderboard WebSocket error: {str(e)}")
    finally:
        # Clean up connection
        await manager.disconnect(connection_id)

def register_websocket_routes(app) -> None:
    """
    Register WebSocket routes with the application.
    
    Args:
        app: FastAPI application instance
    """
    # Create WebSocket manager if doesn't exist
    if not hasattr(app.state, "components"):
        app.state.components = {}
        
    if "websocket_manager" not in app.state.components:
        # Get Redis client if available
        redis_client = app.state.components.get("redis_client")
        
        # Create manager
        manager = WebSocketManager(redis_client=redis_client)
        app.state.components["websocket_manager"] = manager
        
        # Start Redis listener if Redis is available
        if redis_client:
            @app.on_event("startup")
            async def start_websocket_redis_listener():
                await manager.start_redis_listener()
                
    # Register WebSocket routes
    app.include_router(ws_router) 