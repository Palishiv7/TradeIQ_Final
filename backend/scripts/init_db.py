#!/usr/bin/env python3
"""
Database initialization script.

This script creates the database and runs migrations.
"""

import os
import sys
import logging
import asyncio
from pathlib import Path

# Add the backend directory to the Python path
backend_dir = Path(__file__).parent.parent
sys.path.append(str(backend_dir))

from backend.database.base import Base
from backend.database.init_db import initialize_database
from backend.assessments.candlestick_patterns.candlestick_models import CandlestickQuestion, CandlestickSession

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

async def async_main():
    """Initialize the database."""
    try:
        # Initialize database with SQLite URL
        database_url = "sqlite+aiosqlite:///./tradeiq_assessments.db"
        engine = await initialize_database(
            database_url=database_url,
            echo=True,
            pool_size=5,
            max_overflow=10,
            pool_timeout=30
        )
        
        # Create all tables
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            
        logger.info("Database initialized successfully")
            
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(async_main()) 