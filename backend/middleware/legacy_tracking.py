"""
Legacy API Usage Tracking Middleware

This module provides middleware for tracking usage of legacy API endpoints
to help with planning the eventual migration away from these endpoints.
"""

import time
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from backend.common.logger import get_logger

# Setup module logger
logger = get_logger(__name__)

class LegacyAPITrackingMiddleware(BaseHTTPMiddleware):
    """Middleware to track usage of legacy API endpoints."""
    
    async def dispatch(self, request: Request, call_next):
        """
        Process the request and track legacy API usage.
        
        Args:
            request: The incoming request
            call_next: The next middleware/route handler
            
        Returns:
            The response from the next handler
        """
        path = request.url.path
        
        # Only track legacy API endpoints
        if path.startswith("/v1/candlestick-patterns/"):
            start_time = time.time()
            client_ip = request.client.host if request.client else "unknown"
            user_agent = request.headers.get("user-agent", "unknown")
            endpoint = path.split("/")[-1]
            
            try:
                # Process the request
                response = await call_next(request)
                
                # Record usage metrics
                duration = time.time() - start_time
                status_code = response.status_code
                
                # Log the usage data
                logger.info(
                    f"LEGACY_API_USAGE: endpoint={endpoint}, status={status_code}, "
                    f"duration={duration:.3f}s, ip={client_ip}, agent={user_agent}"
                )
                
                return response
                
            except Exception as e:
                logger.error(f"Error in legacy API middleware: {str(e)}")
                # Ensure we still pass the request through
                return await call_next(request)
        else:
            # Not a legacy endpoint, just pass through
            return await call_next(request) 