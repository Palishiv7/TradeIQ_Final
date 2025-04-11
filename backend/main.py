"""
Main application entry point for the TradeIQ assessment platform.

This module serves as the central entry point for the FastAPI application,
registering all assessment modules and shared middleware.

Usage:
    - Direct: python -m backend.main
    - ASGI server: uvicorn backend.main:app
"""

import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import settings
from backend.common.logger import app_logger
from backend.database.init_db import initialize_database, close_database
from backend.assessments.candlestick_patterns.router import router as candlestick_router
from backend.middleware.legacy_tracking import LegacyAPITrackingMiddleware
from backend.assessments.candlestick_patterns.repository import init_database_schema

# Setup module logger
logger = app_logger.getChild("main")

# Create the FastAPI application
app = FastAPI(
    title="TradeIQ Assessments API",
    description="API for TradeIQ trading assessments",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Configure this based on environment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add legacy API tracking middleware
app.add_middleware(LegacyAPITrackingMiddleware)

# Register routers
app.include_router(candlestick_router, prefix="/api/assessments/candlestick", tags=["candlestick"])

@app.on_event("startup")
async def startup_event():
    """Initialize services on application startup."""
    try:
        # Initialize database
        await initialize_database(
            database_url=settings.DATABASE_URL,
            echo=settings.SQL_ECHO,
            pool_size=settings.DB_POOL_SIZE,
            max_overflow=settings.DB_MAX_OVERFLOW,
            pool_timeout=settings.DB_POOL_TIMEOUT
        )
        
        # Initialize candlestick pattern schema
        await init_database_schema()
        
        logger.info("Application startup complete")
    except Exception as e:
        logger.error(f"Failed to initialize application: {str(e)}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup services on application shutdown."""
    try:
        await close_database()
        logger.info("Application shutdown complete")
    except Exception as e:
        logger.error(f"Error during application shutdown: {str(e)}")
        raise

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Welcome to TradeIQ Assessments API"}

# Startup message
logger.info(f"Application initialized with {len(app.routes)} routes")
logger.info(f"Environment: {os.environ.get('ENV', 'development')}")

# Entry point for running the application directly
if __name__ == "__main__":
    import uvicorn
    
    # Get configuration from environment or use defaults
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 8000))
    reload_enabled = os.environ.get("RELOAD", "true").lower() == "true"
    
    logger.info(f"Starting server on {host}:{port} (reload: {reload_enabled})")
    
    # Run the application
    uvicorn.run(
        "backend.main:app",
        host=host,
        port=port,
        reload=reload_enabled,
        log_level="info"
    )
