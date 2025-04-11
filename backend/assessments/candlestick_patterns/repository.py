"""
Repository Module for Candlestick Pattern Assessments

This module provides repository implementations for storing and retrieving
candlestick pattern assessment data, including sessions, questions, and performance metrics.
"""

import json
import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Set, Union, cast, TypeVar, Generic, Type
from datetime import datetime, timedelta
from contextlib import contextmanager, asynccontextmanager

from sqlalchemy import create_engine, text, func, Table, MetaData, Column, String, JSON, insert, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.future import select

from backend.common.logger import app_logger
from backend.common.base_assessment import AssessmentSession
from backend.common.cache import get_cache, CacheManager, async_cached
from backend.common.assessment_repository import (
    QuestionRepository as BaseQuestionRepository,
    SessionRepository as BaseSessionRepository,
    Repository
)
from backend.common.db import get_db_session, DatabaseSession
from backend.common.exceptions import DatabaseError, CacheError
from backend.common.utils import serialize_datetime

from backend.assessments.base.repositories import (
    AssessmentRepository, 
    QuestionRepository, 
    SessionRepository
)
from backend.assessments.base.models import QuestionDifficulty

from backend.assessments.candlestick_patterns.candlestick_pattern_recognition import (
    CandlestickPatternQuestion, CandlestickQuestionData
)
from backend.assessments.candlestick_patterns.adaptive_difficulty import (
    UserPerformanceTracker, AdaptiveDifficultyEngine
)
from backend.assessments.candlestick_patterns.database_models import (
    Base,  # Import Base from database_models
    CandlestickSession, CandlestickQuestion, CandlestickPerformance,
    CandlestickLeaderboard, CandlestickSessionArchive, CandlestickPattern,
    CandlestickAttempt
)

# Type variables for generics
T = TypeVar('T', bound=AssessmentSession)
Q = TypeVar('Q', bound=CandlestickPatternQuestion)

# Module logger
logger = app_logger.getChild("candlestick_repository")

class BaseRepository(Generic[T]):
    """
    Base repository implementation with common functionality.
    
    This generic class provides shared methods for data access operations
    across different entity types, reducing code duplication.
    """
    
    def __init__(
        self, 
        domain_type: str, 
        model_class: Type,
        table_name: str,
        cache: Optional[Any] = None,
        db_session: Optional[Union[Session, AsyncSession]] = None,
        cache_ttl_seconds: int = 3600
    ):
        """
        Initialize the base repository.
        
        Args:
            domain_type: Type of domain object (used for cache keys)
            model_class: SQLAlchemy model class
            table_name: Database table name
            cache: Optional cache service
            db_session: Optional database session (sync or async)
            cache_ttl_seconds: Cache TTL in seconds
        """
        self.domain_type = domain_type
        self.model_class = model_class
        self.table_name = table_name
        self.cache = cache or get_cache()
        self.db_session = db_session
        self.cache_ttl_seconds = cache_ttl_seconds
        
        # Initialize async engine and session factory
        self._engine = create_async_engine(
            "sqlite+aiosqlite:///./data/tradeiq.db",
            echo=False,
            future=True
        )
        self._session_factory = sessionmaker(
            self._engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
    
    async def _get_async_session(self) -> AsyncSession:
        """
        Get an async database session.
        
        Returns:
            Async database session
        """
        if isinstance(self.db_session, AsyncSession):
            return self.db_session
        return self._session_factory()

    @asynccontextmanager
    async def _async_session_scope(self):
        """
        Provide an async transactional scope around a series of operations.
        """
        session = await self._get_async_session()
        need_close = self.db_session is None
        try:
            yield session
            if need_close:
                await session.commit()
        except SQLAlchemyError as e:
            if need_close:
                await session.rollback()
            logger.error(f"Database error in {self.domain_type} repository: {str(e)}")
            raise DatabaseError(f"Database error in {self.domain_type} repository: {str(e)}")
        except Exception as e:
            if need_close:
                await session.rollback()
            logger.error(f"Unexpected error in {self.domain_type} repository: {str(e)}")
            raise
        finally:
            if need_close:
                await session.close()
    
    def _construct_cache_key(self, id_value: str) -> str:
        """
        Construct a cache key for an entity.
        
        Args:
            id_value: Entity identifier
            
        Returns:
            Cache key
        """
        return f"{self.domain_type}:{id_value}"
    
    def _entity_to_dict(self, entity: T) -> Dict:
        """
        Convert an entity to a dictionary.
        
        Args:
            entity: Entity to convert
            
        Returns:
            Dictionary representation
        """
        if hasattr(entity, 'to_dict'):
            return entity.to_dict()
        return {c.name: getattr(entity, c.name) for c in entity.__table__.columns}
    
    def _dict_to_entity(self, data: Dict) -> T:
        """
        Convert a dictionary to an entity.
        
        Args:
            data: Dictionary to convert
            
        Returns:
            Entity
        """
        return self.model_class(**data)
    
    async def get_async(self, id_value: str) -> Optional[T]:
        """
        Get an entity by ID asynchronously.
        
        Args:
            id_value: Entity identifier
            
        Returns:
            Entity if found, None otherwise
        """
        cache_key = self._construct_cache_key(id_value)
        
        # Try to get from cache
        try:
            cached_result = await self.cache.get(cache_key)
            if cached_result is not None:
                try:
                    entity_dict = json.loads(cached_result.value, object_hook=parse_json_with_dates)
                    return self._dict_to_entity(entity_dict)
                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning(f"Error deserializing cached entity: {str(e)}")
        except Exception as e:
            logger.warning(f"Cache error: {str(e)}")
        
        # If not in cache, fetch from database using async session
        async with self._async_session_scope() as session:
            stmt = select(self.model_class).where(self.model_class.id == id_value)
            result = await session.execute(stmt)
            entity = result.scalar_one_or_none()
            
            if entity:
                # Cache the result
                try:
                    entity_dict = self._entity_to_dict(entity)
                    await self.cache.set(
                        cache_key,
                        json.dumps(entity_dict, default=str),
                        ttl=self.cache_ttl_seconds
                    )
                except Exception as e:
                    logger.warning(f"Error caching entity: {str(e)}")
                
                return entity
            
            return None
    
    async def save(self, entity: T) -> Optional[T]:
        """
        Save an entity asynchronously.
        
        Args:
            entity: Entity to save
            
        Returns:
            Saved entity if successful, None otherwise
        """
        try:
            async with self._async_session_scope() as session:
                # Merge the entity to handle both insert and update
                merged = await session.merge(entity)
                await session.commit()
                
                # Update cache if enabled
                if self.cache:
                    cache_key = self._construct_cache_key(entity.id)
                    entity_dict = self._entity_to_dict(merged)
                    await self.cache.set(
                        cache_key,
                        json.dumps(entity_dict, default=str),
                        ttl=self.cache_ttl_seconds
                    )
                
                return merged
        except Exception as e:
            logger.error(f"Error saving {self.domain_type}: {str(e)}")
            return None
    
    async def delete(self, id_value: str) -> bool:
        """
        Delete an entity by ID asynchronously.
        
        Args:
            id_value: Entity identifier
            
        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            async with self._async_session_scope() as session:
                stmt = select(self.model_class).where(self.model_class.id == id_value)
                result = await session.execute(stmt)
                entity = result.scalar_one_or_none()
                
                if entity:
                    await session.delete(entity)
                    await session.commit()
                    
                    # Delete from cache
                    if self.cache:
                        cache_key = self._construct_cache_key(id_value)
                        await self.cache.delete(cache_key)
                    
                    return True
                return False
        except Exception as e:
            logger.error(f"Error deleting {self.domain_type} with ID {id_value}: {str(e)}")
            return False


class CandlestickSessionRepository(BaseRepository[CandlestickSession]):
    """
    Repository for candlestick pattern assessment sessions.
    
    Handles CRUD operations for assessment sessions, including caching and 
    retrieving by various criteria.
    """
    
    def __init__(
        self, 
        cache: Optional[Any] = None,
        db_session: Optional[Union[Session, AsyncSession]] = None
    ):
        """
        Initialize the session repository.
        
        Args:
            cache: Optional cache service
            db_session: Optional database session (sync or async)
        """
        super().__init__(
            domain_type="candlestick_session",
            model_class=CandlestickSession,
            table_name="candlestick_session",
            cache=cache,
            db_session=db_session
        )
    
    def get_user_sessions(
        self, 
        user_id: str, 
        limit: int = 10, 
        offset: int = 0
    ) -> List[CandlestickSession]:
        """
        Get sessions for a user.
        
        Args:
            user_id: User identifier
            limit: Maximum number of sessions to return
            offset: Offset for pagination
            
        Returns:
            List of sessions
        """
        try:
            with self._session_scope() as session:
                query = session.query(CandlestickSession).filter(
                    CandlestickSession.user_id == user_id
                ).order_by(
                    CandlestickSession.created_at.desc()
                ).limit(limit).offset(offset)
                
                return query.all()
        except Exception as e:
            logger.error(f"Error getting sessions for user {user_id}: {str(e)}")
            return []
    
    def get_session_by_user_and_timestamp(
        self, 
        user_id: str, 
        timestamp: datetime
    ) -> Optional[CandlestickSession]:
        """
        Get a session by user ID and timestamp.
        
        Args:
            user_id: User identifier
            timestamp: Session timestamp
            
        Returns:
            Session if found, None otherwise
        """
        try:
            with self._session_scope() as session:
                # Define a time window (1 minute) around the timestamp
                time_window = timedelta(minutes=1)
                start_time = timestamp - time_window
                end_time = timestamp + time_window
                
                query = session.query(CandlestickSession).filter(
                    CandlestickSession.user_id == user_id,
                    CandlestickSession.created_at >= start_time,
                    CandlestickSession.created_at <= end_time
                ).order_by(
                    CandlestickSession.created_at.desc()
                )
                
                return query.first()
        except Exception as e:
            logger.error(f"Error getting session for user {user_id} at {timestamp}: {str(e)}")
            return None
    
    def get_active_sessions(self, user_id: str) -> List[CandlestickSession]:
        """
        Get active sessions for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            List of active sessions
        """
        try:
            with self._session_scope() as session:
                query = session.query(CandlestickSession).filter(
                    CandlestickSession.user_id == user_id,
                    CandlestickSession.completed_at.is_(None)
                ).order_by(
                    CandlestickSession.created_at.desc()
                )
                
                return query.all()
        except Exception as e:
            logger.error(f"Error getting active sessions for user {user_id}: {str(e)}")
            return []
    
    def archive_old_sessions(self, days_threshold: int = 30) -> int:
        """
        Archive old sessions.
        
        Args:
            days_threshold: Age threshold in days
            
        Returns:
            Number of sessions archived
        """
        try:
            with self._session_scope() as session:
                # Get cutoff date
                cutoff_date = datetime.utcnow() - timedelta(days=days_threshold)
                
                # Find sessions to archive
                old_sessions = session.query(CandlestickSession).filter(
                    CandlestickSession.created_at < cutoff_date,
                    CandlestickSession.completed_at.isnot(None)
                ).all()
                
                # Archive sessions
                archived_count = 0
                for old_session in old_sessions:
                    # Create archive entry
                    archive = CandlestickSessionArchive(
                        session_id=old_session.session_id,
                        user_id=old_session.user_id,
                        assessment_type=old_session.assessment_type,
                        created_at=old_session.created_at,
                        completed_at=old_session.completed_at,
                        data=old_session.data,
                        archived_at=datetime.utcnow()
                    )
                    
                    # Save archive entry and delete original
                    session.add(archive)
                    session.delete(old_session)
                    
                    # Delete from cache
                    cache_key = self._construct_cache_key(old_session.session_id)
                    self.cache.delete(cache_key)
                    
                    archived_count += 1
                
                return archived_count
        except Exception as e:
            logger.error(f"Error archiving old sessions: {str(e)}")
            return 0


class CandlestickQuestionRepository(BaseRepository[CandlestickQuestion]):
    """
    Repository for candlestick pattern questions.
    
    Handles CRUD operations for questions, including filtering by 
    difficulty and pattern type.
    """
    
    def __init__(
        self, 
        cache: Optional[Any] = None,
        db_session: Optional[Union[Session, AsyncSession]] = None
    ):
        """
        Initialize the question repository.
        
        Args:
            cache: Optional cache service
            db_session: Optional database session (sync or async)
        """
        super().__init__(
            domain_type="candlestick_question",
            model_class=CandlestickQuestion,
            table_name="candlestick_question",
            cache=cache,
            db_session=db_session
        )
    
    async def find_by_difficulty(
        self, 
        difficulty: Union[str, QuestionDifficulty],
        limit: int = 10,
        offset: int = 0
    ) -> List[CandlestickQuestion]:
        """
        Find questions by difficulty.
        
        Args:
            difficulty: Difficulty level
            limit: Maximum number of questions to return
            offset: Offset for pagination
            
        Returns:
            List of questions
        """
        try:
            # Convert enum to string if needed
            if isinstance(difficulty, QuestionDifficulty):
                difficulty = difficulty.value
                
            async with self._async_session_scope() as session:
                stmt = select(CandlestickQuestion).where(
                    CandlestickQuestion.difficulty == difficulty
                ).limit(limit).offset(offset)
                
                result = await session.execute(stmt)
                return list(result.scalars().all())
        except Exception as e:
            logger.error(f"Error finding questions by difficulty {difficulty}: {str(e)}")
            return []
    
    async def find_by_pattern_type(
        self, 
        pattern_type: str,
        limit: int = 10,
        offset: int = 0
    ) -> List[CandlestickQuestion]:
        """
        Find questions by pattern type.
        
        Args:
            pattern_type: Pattern type
            limit: Maximum number of questions to return
            offset: Offset for pagination
            
        Returns:
            List of questions
        """
        try:
            async with self._async_session_scope() as session:
                stmt = select(CandlestickQuestion).where(
                    CandlestickQuestion.pattern_type == pattern_type
                ).limit(limit).offset(offset)
                
                result = await session.execute(stmt)
                return list(result.scalars().all())
        except Exception as e:
            logger.error(f"Error finding questions by pattern type {pattern_type}: {str(e)}")
            return []
    
    async def find_by_difficulty_and_pattern(
        self, 
        difficulty: Union[str, QuestionDifficulty],
        pattern_type: str,
        limit: int = 10,
        offset: int = 0
    ) -> List[CandlestickQuestion]:
        """
        Find questions by difficulty and pattern type.
        
        Args:
            difficulty: Difficulty level
            pattern_type: Pattern type
            limit: Maximum number of questions to return
            offset: Offset for pagination
            
        Returns:
            List of questions
        """
        try:
            # Convert enum to string if needed
            if isinstance(difficulty, QuestionDifficulty):
                difficulty = difficulty.value
                
            async with self._async_session_scope() as session:
                stmt = select(CandlestickQuestion).where(
                    CandlestickQuestion.difficulty == difficulty,
                    CandlestickQuestion.pattern_type == pattern_type
                ).limit(limit).offset(offset)
                
                result = await session.execute(stmt)
                return list(result.scalars().all())
        except Exception as e:
            logger.error(f"Error finding questions by difficulty {difficulty} and pattern {pattern_type}: {str(e)}")
            return []

    async def save(self, question: CandlestickQuestion) -> Optional[CandlestickQuestion]:
        """
        Save a question to the database.
        
        Args:
            question: Question to save
            
        Returns:
            Saved question if successful, None otherwise
        """
        try:
            async with self._async_session_scope() as session:
                # Ensure content is properly serialized
                if isinstance(question.content, dict):
                    question.content = json.dumps(question.content)
                
                # Merge the question to handle both insert and update
                merged = await session.merge(question)
                await session.commit()
                
                # Update cache if enabled
                if self.cache:
                    cache_key = self._construct_cache_key(question.question_id)
                    await self.cache.set(
                        cache_key,
                        json.dumps(question.to_dict(), default=str),
                        ttl=self.cache_ttl_seconds
                    )
                
                return merged
        except Exception as e:
            logger.error(f"Error saving question {question.question_id}: {str(e)}")
            return None


class CandlestickPerformanceRepository(BaseRepository[CandlestickPerformance]):
    """
    Repository for user performance data in candlestick pattern assessments.
    
    Handles storing and retrieving performance metrics, including leaderboard data.
    """
    
    def __init__(
        self, 
        cache: Optional[Any] = None,
        db_session: Optional[Union[Session, AsyncSession]] = None
    ):
        """
        Initialize the performance repository.
        
        Args:
            cache: Optional cache service
            db_session: Optional database session (sync or async)
        """
        super().__init__(
            domain_type="candlestick_performance",
            model_class=CandlestickPerformance,
            table_name="candlestick_performance",
            cache=cache,
            db_session=db_session
        )
    
    def get_user_performance(self, user_id: str) -> Optional[CandlestickPerformance]:
        """
        Get performance for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Performance if found, None otherwise
        """
        return self.get(user_id)
    
    def save_user_performance(
        self, 
        user_id: str, 
        performance_data: Dict[str, Any]
    ) -> bool:
        """
        Save performance for a user.
        
        Args:
            user_id: User identifier
            performance_data: Performance data
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            # Convert to JSON
            performance_json = json.dumps(performance_data, default=serialize_datetime)
            
            # Get existing or create new
            performance = self.get_user_performance(user_id)
            if performance:
                performance.data = performance_json
                performance.updated_at = datetime.utcnow()
            else:
                performance = CandlestickPerformance(
                    user_id=user_id,
                    data=performance_json
                )
            
            # Save
            return self.save(performance)
        except Exception as e:
            logger.error(f"Error saving performance for user {user_id}: {str(e)}")
            return False
    
    def get_leaderboard(self, limit: int = 10, offset: int = 0) -> List[Dict[str, Any]]:
        """
        Get leaderboard.
        
        Args:
            limit: Maximum number of entries to return
            offset: Offset for pagination
            
        Returns:
            List of leaderboard entries
        """
        try:
            with self._session_scope() as session:
                query = session.query(CandlestickLeaderboard).order_by(
                    CandlestickLeaderboard.average_score.desc(),
                    CandlestickLeaderboard.total_score.desc()
                ).limit(limit).offset(offset)
                
                leaderboard = query.all()
                return [entry.to_dict() for entry in leaderboard]
        except Exception as e:
            logger.error(f"Error getting leaderboard: {str(e)}")
            return []
    
    def update_leaderboard_entry(
        self, 
        user_id: str, 
        session_score: int,
        patterns_identified: int,
        streak: int
    ) -> bool:
        """
        Update a leaderboard entry.
        
        Args:
            user_id: User identifier
            session_score: Score for the session
            patterns_identified: Number of patterns identified
            streak: Current streak
            
        Returns:
            True if updated successfully, False otherwise
        """
        try:
            with self._session_scope() as session:
                # Get existing entry or create new
                entry = session.query(CandlestickLeaderboard).filter(
                    CandlestickLeaderboard.user_id == user_id
                ).first()
                
                if entry:
                    # Update existing entry
                    entry.total_score += session_score
                    entry.sessions_completed += 1
                    entry.patterns_identified += patterns_identified
                    entry.highest_streak = max(entry.highest_streak, streak)
                    
                    # Recalculate average
                    if entry.sessions_completed > 0:
                        entry.average_score = entry.total_score / entry.sessions_completed
                    
                    entry.updated_at = datetime.utcnow()
                else:
                    # Create new entry
                    entry = CandlestickLeaderboard(
                        user_id=user_id,
                        total_score=session_score,
                        average_score=session_score,
                        sessions_completed=1,
                        patterns_identified=patterns_identified,
                        highest_streak=streak
                    )
                    session.add(entry)
                
                return True
        except Exception as e:
            logger.error(f"Error updating leaderboard for user {user_id}: {str(e)}")
            return False


# Replace global repository instances with a factory function
_candlestick_session_repository = None
_candlestick_question_repository = None
_candlestick_performance_repository = None

def get_candlestick_session_repository() -> CandlestickSessionRepository:
    """Get or create the candlestick session repository instance."""
    global _candlestick_session_repository
    if _candlestick_session_repository is None:
        _candlestick_session_repository = CandlestickSessionRepository()
    return _candlestick_session_repository

def get_candlestick_question_repository() -> CandlestickQuestionRepository:
    """Get or create the candlestick question repository instance."""
    global _candlestick_question_repository
    if _candlestick_question_repository is None:
        _candlestick_question_repository = CandlestickQuestionRepository()
    return _candlestick_question_repository

def get_candlestick_performance_repository() -> CandlestickPerformanceRepository:
    """Get or create the candlestick performance repository instance."""
    global _candlestick_performance_repository
    if _candlestick_performance_repository is None:
        _candlestick_performance_repository = CandlestickPerformanceRepository()
    return _candlestick_performance_repository

# Update CandlestickAssessmentRepository to use factory functions
class CandlestickAssessmentRepository:
    """
    Composite repository for candlestick pattern assessments.
    
    Provides a unified interface to access sessions, questions, and performance data.
    """
    
    def __init__(
        self,
        session_repo: Optional[CandlestickSessionRepository] = None,
        question_repo: Optional[CandlestickQuestionRepository] = None,
        performance_repo: Optional[CandlestickPerformanceRepository] = None
    ):
        """
        Initialize the assessment repository.
        
        Args:
            session_repo: Optional session repository
            question_repo: Optional question repository
            performance_repo: Optional performance repository
        """
        self.session_repo = session_repo or get_candlestick_session_repository()
        self.question_repo = question_repo or get_candlestick_question_repository()
        self.performance_repo = performance_repo or get_candlestick_performance_repository()
    
    def get_session(self, session_id: str) -> Optional[CandlestickSession]:
        """
        Get a session by ID.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session if found, None otherwise
        """
        return self.session_repo.get(session_id)
    
    def get_question(self, question_id: str) -> Optional[CandlestickQuestion]:
        """
        Get a question by ID.
        
        Args:
            question_id: Question identifier
            
        Returns:
            Question if found, None otherwise
        """
        return self.question_repo.get(question_id)
    
    def get_questions_for_session(self, session_id: str) -> List[CandlestickQuestion]:
        """
        Get questions for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            List of questions
        """
        try:
            session = self.get_session(session_id)
            if not session:
                logger.warning(f"Session not found: {session_id}")
                return []
            
            # Convert session data to get question IDs
            session_data = {}
            if session.data:
                try:
                    session_data = json.loads(session.data)
                except json.JSONDecodeError:
                    logger.warning(f"Error parsing session data for {session_id}")
                    return []
            
            # Get question IDs from session data
            question_ids = session_data.get("questions", [])
            if not question_ids:
                logger.warning(f"No questions found for session {session_id}")
                return []
            
            # Get questions
            questions = []
            for question_id in question_ids:
                question = self.get_question(question_id)
                if question:
                    questions.append(question)
            
            return questions
        except Exception as e:
            logger.error(f"Error getting questions for session {session_id}: {str(e)}")
            return []
    
    def get_user_performance(self, user_id: str) -> Dict[str, Any]:
        """
        Get performance for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Performance data
        """
        performance = self.performance_repo.get_user_performance(user_id)
        if not performance:
            return {"user_id": user_id, "data": {}}
        
        return performance.to_dict()
    
    def get_user_attempts(
        self, 
        user_id: str,
        limit: int = 50, 
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Get attempts for a user.
        
        Args:
            user_id: User identifier
            limit: Maximum number of attempts to return
            offset: Offset for pagination
            
        Returns:
            List of attempts
        """
        try:
            with self.session_repo._session_scope() as session:
                query = session.query(CandlestickAttempt).filter(
                    CandlestickAttempt.user_id == user_id
                ).order_by(
                    CandlestickAttempt.created_at.desc()
                ).limit(limit).offset(offset)
                
                attempts = query.all()
                return [attempt.to_dict() for attempt in attempts]
        except Exception as e:
            logger.error(f"Error getting attempts for user {user_id}: {str(e)}")
            return []


# Create singleton instance for global use with lazy initialization
_candlestick_assessment_repository = None

def get_candlestick_assessment_repository() -> CandlestickAssessmentRepository:
    """Get or create the candlestick assessment repository instance."""
    global _candlestick_assessment_repository
    if _candlestick_assessment_repository is None:
        _candlestick_assessment_repository = CandlestickAssessmentRepository()
    return _candlestick_assessment_repository

# Update the schema initialization to properly handle async operations
async def init_database_schema():
    """Initialize the database schema for candlestick pattern assessments."""
    try:
        from backend.database.init_db import get_engine
        engine = get_engine()
        
        # Create tables if they don't exist
        async with engine.begin() as conn:
            # Create tables in the correct order to handle dependencies
            await conn.run_sync(lambda conn: Base.metadata.create_all(
                bind=conn,
                tables=[
                    CandlestickQuestion.__table__,
                    CandlestickSession.__table__,
                    CandlestickAttempt.__table__,
                    CandlestickPerformance.__table__,
                    CandlestickLeaderboard.__table__,
                    CandlestickSessionArchive.__table__,
                    CandlestickPattern.__table__
                ]
            ))
            
        logger.info("Candlestick pattern assessment schema initialized successfully")
    except RuntimeError as e:
        if "Database engine not initialized" in str(e):
            logger.warning("Database engine not initialized yet, schema initialization will be deferred")
            return
        logger.error(f"Error initializing database schema: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error initializing database schema: {str(e)}")
        raise

# Don't try to initialize schema at module level
# try:
#     asyncio.run(init_database_schema())
# except Exception as e:
#     logger.error(f"Error initializing database schema: {e}")
#     logger.warning("Database schema initialization failed, repositories will use cache-only mode") 