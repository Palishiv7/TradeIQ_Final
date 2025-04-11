"""
WebSocket Manager Module

This module provides a WebSocketManager class for handling WebSocket connections
and broadcasting messages to connected clients. It supports connection management,
group-based messaging, and authentication.
"""

import asyncio
import json
from typing import Dict, List, Set, Any, Optional, Callable, Awaitable
import logging
from datetime import datetime
import uuid

from fastapi import WebSocket, WebSocketDisconnect, status, Depends
from redis.asyncio import Redis

from backend.common.logger import get_logger

logger = get_logger(__name__)

class WebSocketManager:
    """
    WebSocket connection manager for handling real-time communication.
    
    This class manages active WebSocket connections, provides methods for
    broadcasting messages to individual connections or groups, and handles
    connection lifecycle events.
    
    Attributes:
        active_connections: Dictionary of connection IDs to WebSocket instances
        connection_groups: Dictionary mapping group names to sets of connection IDs
        user_connections: Dictionary mapping user IDs to sets of connection IDs
        redis: Optional Redis client for pub/sub in multi-server deployments
    """
    
    def __init__(self, redis_client: Optional[Redis] = None):
        """
        Initialize the WebSocketManager.
        
        Args:
            redis_client: Optional Redis client for pub/sub messaging across servers
        """
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_groups: Dict[str, Set[str]] = {}
        self.user_connections: Dict[str, Set[str]] = {}
        self.redis = redis_client
        self.listeners: Dict[str, List[Callable[[Dict[str, Any]], Awaitable[None]]]] = {}
        
    async def connect(self, websocket: WebSocket, user_id: Optional[str] = None) -> str:
        """
        Accept a WebSocket connection and register it.
        
        Args:
            websocket: The WebSocket connection to manage
            user_id: Optional user ID to associate with this connection
            
        Returns:
            Connection ID for the accepted connection
        """
        await websocket.accept()
        connection_id = str(uuid.uuid4())
        self.active_connections[connection_id] = websocket
        
        # If user ID provided, associate connection with user
        if user_id:
            if user_id not in self.user_connections:
                self.user_connections[user_id] = set()
            self.user_connections[user_id].add(connection_id)
            
        logger.debug(f"WebSocket connection {connection_id} established" + 
                    (f" for user {user_id}" if user_id else ""))
        return connection_id
        
    async def disconnect(self, connection_id: str) -> None:
        """
        Remove a WebSocket connection.
        
        Args:
            connection_id: ID of the connection to remove
        """
        if connection_id in self.active_connections:
            # Remove from active connections
            websocket = self.active_connections.pop(connection_id)
            
            # Remove from all groups
            for group in list(self.connection_groups.keys()):
                if connection_id in self.connection_groups[group]:
                    self.connection_groups[group].remove(connection_id)
                    
                    # Clean up empty groups
                    if len(self.connection_groups[group]) == 0:
                        del self.connection_groups[group]
            
            # Remove from user connections
            for user_id in list(self.user_connections.keys()):
                if connection_id in self.user_connections[user_id]:
                    self.user_connections[user_id].remove(connection_id)
                    
                    # Clean up empty user connection sets
                    if len(self.user_connections[user_id]) == 0:
                        del self.user_connections[user_id]
                        
            logger.debug(f"WebSocket connection {connection_id} disconnected")
            
    async def add_to_group(self, connection_id: str, group: str) -> None:
        """
        Add a connection to a group.
        
        Args:
            connection_id: Connection ID to add to group
            group: Group name
        """
        if connection_id not in self.active_connections:
            logger.warning(f"Cannot add connection {connection_id} to group {group}: Connection not found")
            return
            
        if group not in self.connection_groups:
            self.connection_groups[group] = set()
            
        self.connection_groups[group].add(connection_id)
        logger.debug(f"Added connection {connection_id} to group {group}")
        
    async def remove_from_group(self, connection_id: str, group: str) -> None:
        """
        Remove a connection from a group.
        
        Args:
            connection_id: Connection ID to remove from group
            group: Group name
        """
        if group in self.connection_groups and connection_id in self.connection_groups[group]:
            self.connection_groups[group].remove(connection_id)
            
            # Clean up empty groups
            if len(self.connection_groups[group]) == 0:
                del self.connection_groups[group]
                
            logger.debug(f"Removed connection {connection_id} from group {group}")
                
    async def send_personal_message(self, message: Any, connection_id: str) -> bool:
        """
        Send a message to a specific connection.
        
        Args:
            message: Message to send (will be converted to JSON if not a string)
            connection_id: Connection ID to send message to
            
        Returns:
            True if sent successfully, False otherwise
        """
        if connection_id not in self.active_connections:
            logger.warning(f"Cannot send message to connection {connection_id}: Connection not found")
            return False
            
        # Prepare message
        if not isinstance(message, str):
            message = json.dumps(message)
            
        try:
            await self.active_connections[connection_id].send_text(message)
            return True
        except Exception as e:
            logger.error(f"Error sending message to connection {connection_id}: {str(e)}")
            # Connection might be stale, remove it
            await self.disconnect(connection_id)
            return False
            
    async def broadcast(self, message: Any) -> int:
        """
        Broadcast a message to all active connections.
        
        Args:
            message: Message to broadcast
            
        Returns:
            Number of connections message was sent to
        """
        sent_count = 0
        for connection_id in list(self.active_connections.keys()):
            success = await self.send_personal_message(message, connection_id)
            if success:
                sent_count += 1
                
        return sent_count
        
    async def broadcast_to_group(self, message: Any, group: str) -> int:
        """
        Broadcast a message to a specific group of connections.
        
        Args:
            message: Message to broadcast
            group: Group name to send to
            
        Returns:
            Number of connections message was sent to
        """
        if group not in self.connection_groups:
            logger.warning(f"Cannot broadcast to group {group}: Group not found")
            return 0
            
        sent_count = 0
        for connection_id in list(self.connection_groups[group]):
            success = await self.send_personal_message(message, connection_id)
            if success:
                sent_count += 1
                
        return sent_count
        
    async def broadcast_to_user(self, message: Any, user_id: str) -> int:
        """
        Broadcast a message to all connections for a specific user.
        
        Args:
            message: Message to broadcast
            user_id: User ID to send to
            
        Returns:
            Number of connections message was sent to
        """
        if user_id not in self.user_connections:
            logger.warning(f"Cannot broadcast to user {user_id}: No active connections")
            return 0
            
        sent_count = 0
        for connection_id in list(self.user_connections[user_id]):
            success = await self.send_personal_message(message, connection_id)
            if success:
                sent_count += 1
                
        return sent_count
        
    async def register_listener(self, event_type: str, callback: Callable[[Dict[str, Any]], Awaitable[None]]) -> None:
        """
        Register a callback function for a specific event type.
        
        Args:
            event_type: Event type to listen for
            callback: Async callback function that accepts event data dict
        """
        if event_type not in self.listeners:
            self.listeners[event_type] = []
            
        self.listeners[event_type].append(callback)
        logger.debug(f"Registered listener for event type {event_type}")
        
    async def publish_event(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """
        Publish an event to all registered listeners.
        
        Args:
            event_type: Event type
            event_data: Event data
        """
        event = {
            "type": event_type,
            "timestamp": datetime.now().isoformat(),
            "data": event_data
        }
        
        # Call local listeners
        if event_type in self.listeners:
            for callback in self.listeners[event_type]:
                try:
                    await callback(event)
                except Exception as e:
                    logger.error(f"Error in event listener callback: {str(e)}")
        
        # Publish to Redis if available (for multi-server setups)
        if self.redis:
            channel = f"websocket_events:{event_type}"
            await self.redis.publish(channel, json.dumps(event))
            
    async def start_redis_listener(self) -> None:
        """
        Start Redis pubsub listener for multi-server deployments.
        This should be called during application startup.
        """
        if not self.redis:
            logger.warning("Cannot start Redis listener: No Redis client provided")
            return
            
        # Create pubsub instance
        pubsub = self.redis.pubsub()
        
        # Subscribe to all websocket event channels
        await pubsub.psubscribe("websocket_events:*")
        
        # Start listening task
        asyncio.create_task(self._redis_listener_task(pubsub))
        logger.info("Started Redis pubsub listener for WebSocket events")
        
    async def _redis_listener_task(self, pubsub) -> None:
        """
        Background task to listen for Redis pubsub messages.
        
        Args:
            pubsub: Redis pubsub instance
        """
        try:
            async for message in pubsub.listen():
                if message["type"] == "pmessage":
                    try:
                        # Extract event type from channel
                        channel = message["channel"].decode("utf-8")
                        event_type = channel.split(":", 1)[1]
                        
                        # Parse event data
                        event = json.loads(message["data"].decode("utf-8"))
                        
                        # Call local listeners
                        if event_type in self.listeners:
                            for callback in self.listeners[event_type]:
                                await callback(event)
                    except Exception as e:
                        logger.error(f"Error processing Redis pubsub message: {str(e)}")
        except asyncio.CancelledError:
            # Task was canceled, clean up
            await pubsub.punsubscribe()
            logger.info("Redis pubsub listener stopped")
        except Exception as e:
            logger.error(f"Error in Redis pubsub listener: {str(e)}")
            # Try to clean up
            try:
                await pubsub.punsubscribe()
            except:
                pass 