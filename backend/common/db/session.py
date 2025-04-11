"""
Database Session Management

This module provides SQLAlchemy session management functionality for both
synchronous and asynchronous database operations.
"""

import logging
from typing import AsyncGenerator, Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker, Session

from backend.common.db.connection import get_database_settings
from backend.common.logger import app_logger

# Set up logging
logger = app_logger.getChild("db.session")

# Get database settings
db_settings = get_database_settings()
database_url = db_settings["database_url"]

def get_engine_kwargs() -> Dict[str, Any]:
    """
    Get engine keyword arguments based on database type.
    Different databases support different connection options.
    """
    kwargs = {"echo": False}  # Set to True for SQL query logging
    
    if database_url.startswith("postgresql"):
        # PostgreSQL supports connection pooling
        kwargs.update({
            "pool_size": db_settings["pool_size"],
            "pool_pre_ping": True,
            "pool_recycle": 300,  # Recycle connections every 5 minutes
        })
    # SQLite and other databases use default settings
    
    return kwargs

# Create async engine with appropriate settings
engine = create_async_engine(
    database_url,
    **get_engine_kwargs()
)

# Create session factories for SQLAlchemy 1.4
# Note: In 1.4, we use sessionmaker with class_=AsyncSession
async_session_factory = sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Get an async database session.
    
    Yields:
        AsyncSession: The database session
        
    Example:
        async with get_session() as session:
            result = await session.execute(query)
            await session.commit()
    """
    session = async_session_factory()
    try:
        yield session
        await session.commit()
    except Exception as e:
        await session.rollback()
        logger.error(f"Database session error: {e}")
        raise
    finally:
        await session.close()

# For backward compatibility with sync code
def get_db_session() -> Session:
    """
    Get a synchronous database session.
    This is provided for backward compatibility and should be replaced with async sessions.
    
    Returns:
        Session: The database session
        
    Warning:
        This function is deprecated. Use get_session() for async operations instead.
    """
    logger.warning("Using deprecated sync database session. Please migrate to async sessions.")
    sync_engine = engine.sync_engine
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=sync_engine)
    return SessionLocal() 