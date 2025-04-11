"""
Central API router and utilities for the TradeIQ assessment platform.

This module provides:
- A central router that includes all assessment module routers
- Shared API utilities and middleware
- Common response models and exception handlers
"""

import logging
from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from typing import Dict, Any, List, Optional

# Configure logging
logger = logging.getLogger(__name__)

# Create main API router
main_router = APIRouter()

# Version prefix for all API routes
API_VERSION = "v1"

# Dictionary to track registered assessment modules
registered_modules: Dict[str, APIRouter] = {}


def register_assessment_module(name: str, router: APIRouter) -> None:
    """
    Register an assessment module router with the main API router.
    
    Args:
        name: Name of the assessment module
        router: FastAPI router for the assessment module
    """
    if name in registered_modules:
        logger.warning(f"Assessment module '{name}' already registered, overwriting")
    
    logger.info(f"Registering assessment module '{name}' with {len(router.routes)} routes")
    logger.info(f"Routes: {[route.path for route in router.routes]}")
    
    # Include the router with the appropriate prefix
    main_router.include_router(
        router,
        prefix=f"/{API_VERSION}/{name}",
        tags=[name]
    )
    
    registered_modules[name] = router
    logger.info(f"Registered assessment module: {name} with {len(router.routes)} routes")


# Common validation error handler
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """
    Handle validation errors and return a standardized response.
    
    Args:
        request: The incoming request
        exc: The validation exception
    
    Returns:
        A JSON response with error details
    """
    error_details = []
    for error in exc.errors():
        error_details.append({
            "location": error.get("loc", []),
            "message": error.get("msg", "Unknown validation error"),
            "type": error.get("type", "")
        })
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "status": "error",
            "message": "Validation error",
            "details": error_details
        }
    )


# Standard API response model
class APIResponse:
    """Standard API response structure"""
    
    @staticmethod
    def success(data: Any = None, message: str = "Success") -> Dict[str, Any]:
        """
        Create a success response.
        
        Args:
            data: Response data
            message: Success message
        
        Returns:
            Response dictionary
        """
        return {
            "status": "success",
            "message": message,
            "data": data
        }
    
    @staticmethod
    def error(message: str, details: Optional[Any] = None, 
              code: Optional[str] = None) -> Dict[str, Any]:
        """
        Create an error response.
        
        Args:
            message: Error message
            details: Optional error details
            code: Optional error code
        
        Returns:
            Response dictionary
        """
        response = {
            "status": "error",
            "message": message
        }
        
        if details:
            response["details"] = details
        
        if code:
            response["code"] = code
            
        return response
