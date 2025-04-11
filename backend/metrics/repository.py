"""
User Metrics Repository

This module provides database access for user metrics data.
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select, update, delete

from backend.metrics.models import UserMetrics
from backend.common.db.session import async_session_factory
from backend.common.logger import get_logger
from backend.assessments.candlestick_patterns.candlestick_repository import RepositoryError # Reuse existing error

logger = get_logger(__name__)

class UserMetricsRepository:
    """Repository for managing user metrics data in the database."""
    
    def __init__(self, session_factory: sessionmaker = None):
        """
        Initialize the repository with a session factory.
        
        Args:
            session_factory: SQLAlchemy session factory for creating database sessions
        """
        self._session_factory = session_factory or async_session_factory
        
    async def get_user_metrics(self, user_id: str) -> Optional[UserMetrics]:
        """
        Get metrics for a specific user.
        
        Args:
            user_id: The ID of the user
            
        Returns:
            UserMetrics if found, None otherwise
        """
        async with self._session_factory() as session:
            result = await session.execute(
                select(UserMetrics).where(UserMetrics.user_id == user_id)
            )
            return result.scalar_one_or_none()
            
    async def update_metrics(self, user_id: str, metrics_data: Dict[str, Any]) -> None:
        """
        Update metrics for a user.
        
        Args:
            user_id: The ID of the user
            metrics_data: Dictionary containing metrics to update
        """
        async with self._session_factory() as session:
            stmt = update(UserMetrics).where(
                UserMetrics.user_id == user_id
            ).values(**metrics_data)
            await session.execute(stmt)
            await session.commit()
            
    async def create_metrics(self, user_id: str, initial_data: Dict[str, Any]) -> UserMetrics:
        """
        Create new metrics entry for a user.
        
        Args:
            user_id: The ID of the user
            initial_data: Initial metrics data
            
        Returns:
            The created UserMetrics instance
        """
        metrics = UserMetrics(
            user_id=user_id,
            created_at=datetime.utcnow(),
            **initial_data
        )
        async with self._session_factory() as session:
            session.add(metrics)
            await session.commit()
            await session.refresh(metrics)
            return metrics
            
    async def delete_metrics(self, user_id: str) -> None:
        """
        Delete metrics for a user.
        
        Args:
            user_id: The ID of the user
        """
        async with self._session_factory() as session:
            stmt = delete(UserMetrics).where(UserMetrics.user_id == user_id)
            await session.execute(stmt)
            await session.commit()

    async def get_or_create_metrics(self, user_id: str) -> UserMetrics:
        """
        Retrieves user metrics by user ID, creating a new record if one doesn't exist.

        Args:
            user_id: The ID of the user.

        Returns:
            The UserMetrics ORM instance (either existing or newly created).
        
        Raises:
            RepositoryError: If there's a database error.
        """
        if not user_id:
            raise RepositoryError("User ID cannot be empty")

        try:
            async with self._session_factory() as session:
                # Try to get the existing record
                stmt = select(UserMetrics).where(UserMetrics.user_id == user_id)
                result = await session.execute(stmt)
                
                try:
                    metrics = result.scalar_one()
                    logger.debug(f"Found existing metrics for user {user_id}")
                    return metrics
                except NoResultFound:
                    # If not found, create a new one
                    logger.info(f"No existing metrics found for user {user_id}. Creating new record.")
                    async with session.begin(): # Start transaction for creation
                        new_metrics = UserMetrics(user_id=user_id)
                        session.add(new_metrics)
                        # We need to flush to get the instance back with defaults potentially set,
                        # or re-fetch after commit, but adding is sufficient if we return the object.
                        # For safety, re-fetch might be better if defaults are complex.
                    # Re-fetch after creation to ensure we have the latest state
                    result_after_create = await session.execute(select(UserMetrics).where(UserMetrics.user_id == user_id))
                    return result_after_create.scalar_one() # Should exist now
                except MultipleResultsFound:
                    logger.error(f"Database integrity error: Found multiple metrics records for user {user_id}")
                    raise RepositoryError(f"Multiple metrics records found for user {user_id}")
        except Exception as e:
            logger.error(f"Database error getting or creating metrics for user {user_id}: {e}", exc_info=True)
            raise RepositoryError(f"Database error accessing metrics for user {user_id}", original_exception=e)
            
    async def get_metrics(self, user_id: str) -> Optional[UserMetrics]:
        """
        Retrieves user metrics by user ID.

        Args:
            user_id: The ID of the user.

        Returns:
            The UserMetrics ORM instance or None if not found.
        
        Raises:
            RepositoryError: If there's a database error.
        """
        if not user_id:
            raise RepositoryError("User ID cannot be empty")
        try:
            async with self._session_factory() as session:
                stmt = select(UserMetrics).where(UserMetrics.user_id == user_id)
                result = await session.execute(stmt)
                return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Database error getting metrics for user {user_id}: {e}", exc_info=True)
            raise RepositoryError(f"Database error accessing metrics for user {user_id}", original_exception=e)

# Potential additions:
# - Methods for batch updates/fetches
# - Methods for querying based on specific metric values (e.g., leaderboard queries) 