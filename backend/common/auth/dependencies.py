"""
Authentication dependencies for the TradeIQ assessment platform.

This module provides FastAPI dependencies for user authentication.
For development/testing, it accepts a test token.
"""

import logging
from fastapi import Header, HTTPException, status
from typing import Optional

logger = logging.getLogger(__name__)

async def get_current_user_id(authorization: Optional[str] = Header(None)) -> str:
    """
    Get the current user ID from the authorization header.
    For development/testing, accepts a test token.
    
    Args:
        authorization: Authorization header value
        
    Returns:
        User ID string
        
    Raises:
        HTTPException: If authentication fails
    """
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authorization header"
        )
    
    try:
        # Extract token from "Bearer <token>"
        scheme, token = authorization.split()
        if scheme.lower() != "bearer":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication scheme"
            )
        
        # For development/testing, accept test-token
        if token == "test-token":
            return "test-user"
            
        # TODO: Implement proper token validation
        # For now, just return the token as the user ID
        return token
        
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization header format"
        ) 