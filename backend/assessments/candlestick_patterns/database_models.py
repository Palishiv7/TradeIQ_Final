"""
SQLAlchemy ORM models for candlestick pattern assessments.

This module defines the database models for candlestick pattern assessments, including:
- CandlestickSession: Represents an assessment session
- CandlestickQuestion: Represents questions in assessments
- CandlestickAttempt: Tracks user attempts at questions
- CandlestickPerformance: Records user performance data
- CandlestickLeaderboard: Maintains leaderboard entries
- CandlestickSessionArchive: Archives old sessions
- CandlestickPattern: Records detected candlestick patterns
"""

import json
import datetime
from typing import Optional, Dict, Any, List, Union
from sqlalchemy import (
    Column, String, Integer, Boolean, DateTime,
    Text, Float, ForeignKey, JSON, Index, func, select, case, MetaData
)
# Import JSONB for PostgreSQL
from sqlalchemy.dialects.postgresql import JSONB 
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, validates
from sqlalchemy.schema import UniqueConstraint

from backend.database.init_db import get_engine
from backend.common.validation import validate_json
from backend.assessments.base.models import SessionStatus
import logging

logger = logging.getLogger(__name__)

# Create metadata with naming convention
metadata = MetaData(naming_convention={
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s"
})

# Create declarative base with metadata
Base = declarative_base(metadata=metadata)


class CandlestickSession(Base):
    """
    Model for candlestick pattern assessment sessions.
    
    Tracks user sessions including question history and performance data.
    """
    __tablename__ = 'candlestick_session'
    
    session_id = Column(String(255), primary_key=True)
    user_id = Column(String(255), nullable=False, index=True)
    assessment_type = Column(String(50), nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    completed_at = Column(DateTime, nullable=True)
    status = Column(String(50), nullable=False, default=SessionStatus.CREATED.value, index=True)
    data = Column(Text, nullable=True)
    
    # Relationships
    attempts = relationship("CandlestickAttempt", back_populates="session", 
                          cascade="all, delete-orphan", lazy="dynamic")
    
    # Indexes for query optimization
    __table_args__ = (
        Index('idx_candlestick_user_created', user_id, created_at),
    )
    
    @validates('data')
    def validate_data(self, key, data):
        """Validate that data is valid JSON"""
        if data is not None:
            # Ensure data is valid JSON
            validate_json(data)
        return data
    
    def __init__(self, session_id: str, user_id: str, assessment_type: str, 
                 created_at: Optional[datetime.datetime] = None, 
                 completed_at: Optional[datetime.datetime] = None,
                 status: Optional[str] = None,
                 data: Optional[str] = None):
        """
        Initialize a candlestick session.
        
        Args:
            session_id: Unique session identifier
            user_id: User identifier
            assessment_type: Type of assessment
            created_at: When the session was created (defaults to now)
            completed_at: When the session was completed (if applicable)
            status: Session status
            data: JSON data for the session
        """
        self.session_id = session_id
        self.user_id = user_id
        self.assessment_type = assessment_type
        self.created_at = created_at or datetime.datetime.utcnow()
        self.completed_at = completed_at
        self.status = status or SessionStatus.CREATED.value
        self.data = data
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert session to dictionary representation.
        
        Returns:
            Dictionary with session data
        """
        data_dict = {}
        if self.data:
            try:
                data_dict = json.loads(self.data)
            except (json.JSONDecodeError, TypeError):
                data_dict = {}
                
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "assessment_type": self.assessment_type,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "status": self.status,
            "data": data_dict
        }


class CandlestickQuestion(Base):
    """
    Model for candlestick pattern questions.
    
    Represents individual questions in assessments, including the pattern,
    difficulty level, and content.
    """
    __tablename__ = 'candlestick_question'
    
    question_id = Column(String(255), primary_key=True)
    difficulty = Column(String(50), nullable=False, index=True)
    pattern_type = Column(String(100), nullable=False, index=True)
    content = Column(Text, nullable=False)  # Stores JSON content
    topics = Column(Text, nullable=False)  # Stores JSON array of topics
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow, nullable=False)
    
    # Relationships
    attempts = relationship("CandlestickAttempt", back_populates="question",
                          cascade="all, delete", lazy="dynamic")
    
    # Indexes for query optimization
    __table_args__ = (
        Index('idx_candlestick_difficulty_pattern', difficulty, pattern_type),
    )
    
    @validates('content')
    def validate_content(self, key, content):
        """Validate that content is valid JSON"""
        if content is not None:
            if isinstance(content, dict):
                content = json.dumps(content)
            else:
                validate_json(content)
        return content
    
    @validates('topics')
    def validate_topics(self, key, topics):
        """Validate that topics is a valid JSON array"""
        if topics is not None:
            if isinstance(topics, (list, tuple)):
                topics = json.dumps(list(topics))
            else:
                validate_json(topics)
                # Ensure it's a JSON array
                topics_list = json.loads(topics)
                if not isinstance(topics_list, list):
                    raise ValueError("Topics must be a JSON array")
        return topics
    
    def __init__(self, question_id: str, difficulty: str, pattern_type: str, 
                 content: Union[Dict[str, Any], str], topics: Union[List[str], str]):
        """
        Initialize a candlestick question.
        
        Args:
            question_id: Unique question identifier
            difficulty: Difficulty level
            pattern_type: Pattern type
            content: Question content (dict or JSON string)
            topics: List of topics (list or JSON string)
        """
        self.question_id = question_id
        self.difficulty = difficulty
        self.pattern_type = pattern_type
        self.content = content
        self.topics = topics
        self.created_at = datetime.datetime.utcnow()
        self.updated_at = self.created_at
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert question to dictionary representation.
        
        Returns:
            Dictionary with question data
        """
        # Parse content JSON string to dict
        content_dict = {}
        if self.content:
            try:
                if isinstance(self.content, str):
                    content_dict = json.loads(self.content)
                else:
                    content_dict = self.content
            except (json.JSONDecodeError, TypeError):
                logger.warning(f"Failed to decode content for question {self.question_id}")
                content_dict = {}
        
        # Parse topics JSON string to list
        topics_list = []
        if self.topics:
            try:
                if isinstance(self.topics, str):
                    topics_list = json.loads(self.topics)
                else:
                    topics_list = self.topics
            except (json.JSONDecodeError, TypeError):
                logger.warning(f"Failed to decode topics for question {self.question_id}")
                topics_list = []
                
        return {
            "question_id": self.question_id,
            "difficulty": self.difficulty,
            "pattern_type": self.pattern_type,
            "content": content_dict,
            "topics": topics_list,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }


class CandlestickAttempt(Base):
    """
    Model for user attempts at candlestick pattern questions.
    
    Tracks user answers, correctness, and time taken.
    """
    __tablename__ = 'candlestick_attempt'
    
    attempt_id = Column(String(255), primary_key=True)
    session_id = Column(String(255), ForeignKey('candlestick_session.session_id', ondelete='CASCADE'), 
                       nullable=False, index=True)
    user_id = Column(String(255), nullable=False, index=True)
    question_id = Column(String(255), ForeignKey('candlestick_question.question_id', ondelete='CASCADE'), 
                        nullable=False)
    is_correct = Column(Boolean, nullable=False, default=False)
    answer = Column(String(255), nullable=True)
    time_taken_ms = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    
    # Relationships
    session = relationship("CandlestickSession", back_populates="attempts")
    question = relationship("CandlestickQuestion", back_populates="attempts")
    
    # Composite index for performance
    __table_args__ = (
        Index('idx_candlestick_user_session_question', user_id, session_id, question_id),
    )
    
    @validates('time_taken_ms')
    def validate_time(self, key, time):
        """Validate time taken"""
        if time is not None and time < 0:
            return 0
        return time
    
    def __init__(self, attempt_id: str, session_id: str, user_id: str, 
                 question_id: str, is_correct: bool, answer: Optional[str] = None, 
                 time_taken_ms: Optional[int] = None, 
                 created_at: Optional[datetime.datetime] = None):
        """
        Initialize an attempt.
        
        Args:
            attempt_id: Unique attempt identifier
            session_id: Session identifier
            user_id: User identifier
            question_id: Question identifier
            is_correct: Whether the answer was correct
            answer: User's answer
            time_taken_ms: Time taken in milliseconds
            created_at: When the attempt was made
        """
        self.attempt_id = attempt_id
        self.session_id = session_id
        self.user_id = user_id
        self.question_id = question_id
        self.is_correct = is_correct
        self.answer = answer
        self.time_taken_ms = time_taken_ms
        self.created_at = created_at or datetime.datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert attempt to dictionary representation.
        
        Returns:
            Dictionary with attempt data
        """
        return {
            "attempt_id": self.attempt_id,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "question_id": self.question_id,
            "is_correct": self.is_correct,
            "answer": self.answer,
            "time_taken_ms": self.time_taken_ms,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


class CandlestickPerformance(Base):
    """
    Model for user performance data in candlestick pattern assessments.
    
    Tracks overall performance, pattern-specific metrics, and progress.
    """
    __tablename__ = 'candlestick_performance'
    
    user_id = Column(String(255), primary_key=True)
    data = Column(Text, nullable=True)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, 
                      onupdate=datetime.datetime.utcnow, nullable=False)
    
    @validates('data')
    def validate_data(self, key, data):
        """Validate that data is valid JSON"""
        if data is not None:
            validate_json(data)
        return data
    
    def __init__(self, user_id: str, data: Optional[str] = None):
        """
        Initialize a performance record.
        
        Args:
            user_id: User identifier
            data: JSON data for performance metrics
        """
        self.user_id = user_id
        self.data = data
        self.updated_at = datetime.datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert performance record to dictionary representation.
        
        Returns:
            Dictionary with performance data
        """
        data_dict = {}
        if self.data:
            try:
                data_dict = json.loads(self.data)
            except (json.JSONDecodeError, TypeError):
                data_dict = {}
                
        return {
            "user_id": self.user_id,
            "data": data_dict,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }


class CandlestickLeaderboard(Base):
    """
    Model for candlestick pattern assessment leaderboard.
    
    Tracks user scores and rankings.
    """
    __tablename__ = 'candlestick_leaderboard'
    
    user_id = Column(String(255), primary_key=True)
    total_score = Column(Integer, default=0, nullable=False, index=True)
    average_score = Column(Float, default=0.0, nullable=False, index=True)
    sessions_completed = Column(Integer, default=0, nullable=False)
    patterns_identified = Column(Integer, default=0, nullable=False)
    highest_streak = Column(Integer, default=0, nullable=False)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, 
                      onupdate=datetime.datetime.utcnow, nullable=False)
    
    # Index for leaderboard queries
    __table_args__ = (
        Index('idx_candlestick_leaderboard_score', average_score.desc(), total_score.desc()),
    )
    
    def __init__(self, user_id: str, total_score: int = 0, average_score: float = 0.0,
                 sessions_completed: int = 0, patterns_identified: int = 0, 
                 highest_streak: int = 0):
        """
        Initialize a leaderboard entry.
        
        Args:
            user_id: User identifier
            total_score: Total score across all sessions
            average_score: Average score per session
            sessions_completed: Number of completed sessions
            patterns_identified: Number of patterns correctly identified
            highest_streak: Highest streak of correct answers
        """
        self.user_id = user_id
        self.total_score = total_score
        self.average_score = average_score
        self.sessions_completed = sessions_completed
        self.patterns_identified = patterns_identified
        self.highest_streak = highest_streak
        self.updated_at = datetime.datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert leaderboard entry to dictionary representation.
        
        Returns:
            Dictionary with leaderboard data
        """
        return {
            "user_id": self.user_id,
            "total_score": self.total_score,
            "average_score": self.average_score,
            "sessions_completed": self.sessions_completed,
            "patterns_identified": self.patterns_identified,
            "highest_streak": self.highest_streak,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }


class CandlestickSessionArchive(Base):
    """
    Model for archived candlestick pattern assessment sessions.
    
    Stores completed sessions that have been archived for historical reference.
    """
    __tablename__ = 'candlestick_session_archive'
    
    session_id = Column(String(255), primary_key=True)
    user_id = Column(String(255), nullable=False, index=True)
    assessment_type = Column(String(50), nullable=False)
    created_at = Column(DateTime, nullable=False)
    completed_at = Column(DateTime, nullable=False)
    archived_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    data = Column(Text, nullable=True)
    
    # Index for archive queries
    __table_args__ = (
        Index('idx_candlestick_archive_user_completed', user_id, completed_at.desc()),
    )
    
    @validates('data')
    def validate_data(self, key, data):
        """Validate that data is valid JSON"""
        if data is not None:
            validate_json(data)
        return data
    
    def __init__(self, session_id: str, user_id: str, assessment_type: str,
                 created_at: datetime.datetime, completed_at: datetime.datetime,
                 data: Optional[str] = None, 
                 archived_at: Optional[datetime.datetime] = None):
        """
        Initialize an archived session.
        
        Args:
            session_id: Unique session identifier
            user_id: User identifier
            assessment_type: Type of assessment
            created_at: When the session was created
            completed_at: When the session was completed
            data: JSON data for the session
            archived_at: When the session was archived
        """
        self.session_id = session_id
        self.user_id = user_id
        self.assessment_type = assessment_type
        self.created_at = created_at
        self.completed_at = completed_at
        self.data = data
        self.archived_at = archived_at or datetime.datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert archived session to dictionary representation.
        
        Returns:
            Dictionary with archived session data
        """
        data_dict = {}
        if self.data:
            try:
                data_dict = json.loads(self.data)
            except (json.JSONDecodeError, TypeError):
                data_dict = {}
                
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "assessment_type": self.assessment_type,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "archived_at": self.archived_at.isoformat() if self.archived_at else None,
            "data": data_dict
        }


class CandlestickPattern(Base):
    """
    Model for detected candlestick patterns.
    
    Records patterns detected in market data for reference and analysis.
    """
    __tablename__ = 'candlestick_pattern'
    
    pattern_id = Column(String(255), primary_key=True)
    pattern_type = Column(String(100), nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    timeframe = Column(String(10), nullable=False)
    strength = Column(Float, nullable=False, default=0.0)
    candle_data = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow, nullable=False, index=True)
    
    # Composite indexes for common queries
    __table_args__ = (
        Index('idx_candlestick_pattern_symbol_type', symbol, pattern_type),
        Index('idx_candlestick_pattern_time_type', timestamp.desc(), pattern_type)
    )
    
    @validates('candle_data')
    def validate_candle_data(self, key, data):
        """Validate that candle data is valid JSON"""
        if data is not None:
            validate_json(data)
        return data
    
    @validates('strength')
    def validate_strength(self, key, strength):
        """Validate strength is between 0 and 1"""
        if strength < 0:
            return 0.0
        if strength > 1:
            return 1.0
        return strength
    
    def __init__(self, pattern_id: str, pattern_type: str, symbol: str,
                 timeframe: str, strength: float = 0.0, 
                 candle_data: Optional[str] = None,
                 timestamp: Optional[datetime.datetime] = None):
        """
        Initialize a pattern record.
        
        Args:
            pattern_id: Unique pattern identifier
            pattern_type: Type of candlestick pattern
            symbol: Trading symbol/instrument
            timeframe: Timeframe (e.g., '1d', '4h')
            strength: Pattern strength (0.0 to 1.0)
            candle_data: JSON data for candles forming the pattern
            timestamp: When the pattern was detected
        """
        self.pattern_id = pattern_id
        self.pattern_type = pattern_type
        self.symbol = symbol
        self.timeframe = timeframe
        self.strength = strength
        self.candle_data = candle_data
        self.timestamp = timestamp or datetime.datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert pattern record to dictionary representation.
        
        Returns:
            Dictionary with pattern data
        """
        candle_data_dict = {}
        if self.candle_data:
            try:
                candle_data_dict = json.loads(self.candle_data)
            except (json.JSONDecodeError, TypeError):
                candle_data_dict = {}
                
        return {
            "pattern_id": self.pattern_id,
            "pattern_type": self.pattern_type,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "strength": self.strength,
            "candle_data": candle_data_dict,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None
        }


def create_tables():
    """Create all tables in the database."""
    try:
        engine = get_engine()
        Base.metadata.create_all(engine)
        logger.info("Successfully created all candlestick pattern tables")
    except Exception as e:
        logger.error(f"Failed to create tables: {e}")
        raise


def drop_tables():
    """Drop all tables from the database."""
    try:
        engine = get_engine()
        Base.metadata.drop_all(engine)
        logger.info("Successfully dropped all candlestick pattern tables")
    except Exception as e:
        logger.error(f"Failed to drop tables: {e}")
        raise


# Helper methods for common queries
def get_user_performance_summary(session, user_id: str) -> Dict[str, Any]:
    """
    Get a summary of user performance across all attempts.
    
    Args:
        session: Database session
        user_id: User identifier
        
    Returns:
        Dictionary with performance summary
    """
    # Calculate performance metrics using efficient SQL queries
    query = session.query(
        func.count().label('total_attempts'),
        func.sum(case([(CandlestickAttempt.is_correct, 1)], else_=0)).label('correct_attempts'),
        func.avg(CandlestickAttempt.time_taken_ms).label('avg_time_ms'),
        func.count(func.distinct(CandlestickAttempt.session_id)).label('sessions_count')
    ).filter(CandlestickAttempt.user_id == user_id)
    
    result = query.first()
    
    if not result or result.total_attempts == 0:
        return {
            "user_id": user_id,
            "total_attempts": 0,
            "correct_attempts": 0,
            "accuracy": 0.0,
            "avg_time_ms": 0,
            "sessions_count": 0
        }
    
    # Calculate accuracy
    accuracy = result.correct_attempts / result.total_attempts if result.total_attempts > 0 else 0
    
    return {
        "user_id": user_id,
        "total_attempts": result.total_attempts,
        "correct_attempts": result.correct_attempts,
        "accuracy": accuracy,
        "avg_time_ms": result.avg_time_ms or 0,
        "sessions_count": result.sessions_count
    } 