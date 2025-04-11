"""
Database initialization and connection management.

This module provides functions for:
1. Initializing the database schema
2. Managing database connections
3. Resetting the connection pool
"""

import os
import logging
from typing import Optional
from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.pool import QueuePool
from sqlalchemy import text
from alembic.config import Config
from alembic import command

# Import the new settings function
from backend.common.db.connection import get_database_settings
from backend.config import settings
from backend.common.logger import app_logger

# Setup module logger
logger = app_logger.getChild("database.init_db")

# Global engine instance
_engine: Optional[AsyncEngine] = None

def get_engine() -> AsyncEngine:
    """Get the global async engine instance."""
    if _engine is None:
        raise RuntimeError("Database engine not initialized. Call initialize_database() first.")
    return _engine

async def initialize_database(
    database_url: str,
    echo: bool = False,
    pool_size: int = 5,
    max_overflow: int = 10,
    pool_timeout: int = 30,
) -> AsyncEngine:
    """
    Initialize the async database engine.
    
    Args:
        database_url: Database connection URL
        echo: Whether to echo SQL statements
        pool_size: Connection pool size
        max_overflow: Maximum number of connections to allow above pool_size
        pool_timeout: Timeout for getting a connection from the pool
        
    Returns:
        AsyncEngine instance
    """
    global _engine
    
    try:
        logger.info(f"Initializing database with URL: {database_url[:10]}... and pool size: {pool_size}")
        
        # Create async engine
        _engine = create_async_engine(
            database_url,
            echo=echo,
            future=True,  # Use SQLAlchemy 2.0 style queries
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_timeout=pool_timeout
        )
        
        # Create session factory
        async_session = sessionmaker(
            _engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        
        # Test connection
        async with _engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
            await conn.commit()
            
        logger.info("Database engine initialized successfully")
        return _engine
        
    except Exception as e:
        logger.error(f"Failed to initialize async database: {str(e)}")
        raise

async def close_database() -> None:
    """Close the database engine and all connections."""
    global _engine
    
    if _engine:
        try:
            await _engine.dispose()
            _engine = None
            logger.info("Database engine closed successfully")
        except Exception as e:
            logger.error(f"Error closing database engine: {str(e)}")
            raise

# Dependency for FastAPI routes to get an async session
async def get_async_db() -> AsyncSession:
    """
    Dependency that provides an async database session.
    Ensures the session is closed afterwards.
    
    Returns:
        Async database session
    """
    if _engine is None:
        raise RuntimeError("Database not initialized or initialization failed")
        
    async with get_engine().connect() as conn:
        try:
            yield conn
            await conn.commit() # Optional: commit successful transactions automatically
        except Exception:
            await conn.rollback()
            raise
        finally:
            await conn.close() # Ensures connection is returned to the pool

async def reset_connection_pool():
    """Reset the async database connection pool."""
    global _engine
    
    if _engine:
        await _engine.dispose()
        _engine = None
        logger.info("Async engine disposed.")
    
    logger.info("Async session factory reset.")

# Remove synchronous global variables and functions if no longer needed
# engine = None
# SessionLocal = None
# db_session = None
# def get_db(): ... 