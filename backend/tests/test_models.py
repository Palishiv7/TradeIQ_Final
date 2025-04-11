"""
Modified database models for testing purposes.

This module contains versions of database models that are compatible with
SQLite and have simplified structures for testing purposes.
"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, JSON, ForeignKey, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

# Create base class for SQLAlchemy models
Base = declarative_base()

class TimestampMixin:
    """Mixin to add created_at and updated_at timestamps to models."""
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class PatternStatistics(Base, TimestampMixin):
    """Test model for pattern statistics with a pattern_id field for testing."""
    __tablename__ = 'pattern_statistics'
    
    id = Column(Integer, primary_key=True)
    pattern_id = Column(String, index=True, nullable=False)
    pattern_name = Column(String, nullable=False)
    total_attempts = Column(Integer, default=0)
    correct_attempts = Column(Integer, default=0)
    avg_response_time = Column(Float, default=0.0)
    difficulty_rating = Column(Float, default=0.5)
    success_rate = Column(Float, default=0.0)
    avg_difficulty = Column(Float, default=0.5)
    difficulty_distribution = Column(JSON, default=lambda: {})
    
    def calculate_success_rate(self):
        """Calculate success rate based on attempts."""
        if self.total_attempts > 0:
            self.success_rate = self.correct_attempts / self.total_attempts
        return self.success_rate

class UserPerformance(Base, TimestampMixin):
    """Test model for user performance tracking."""
    __tablename__ = 'user_performance'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(String, index=True, nullable=False)
    total_assessments = Column(Integer, default=0)
    total_questions = Column(Integer, default=0)
    correct_answers = Column(Integer, default=0)
    avg_response_time_ms = Column(Integer, default=0)
    current_difficulty = Column(Float, default=0.5)
    pattern_statistics = Column(JSON, default=lambda: {})

class AssessmentAttempt(Base, TimestampMixin):
    """Test model for assessment attempts."""
    __tablename__ = 'assessment_attempts'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(String, index=True, nullable=False)
    assessment_id = Column(String, nullable=False)
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime, nullable=True)
    score = Column(Float, nullable=True)
    difficulty = Column(Float, default=0.5)
    
    # Relationships
    questions = relationship("QuestionHistory", back_populates="assessment")

class QuestionHistory(Base, TimestampMixin):
    """Test model for question history within assessments."""
    __tablename__ = 'question_history'
    
    id = Column(Integer, primary_key=True)
    assessment_id = Column(Integer, ForeignKey('assessment_attempts.id'))
    pattern_id = Column(String, nullable=False)
    question_type = Column(String, nullable=False)
    difficulty = Column(Float, default=0.5)
    is_correct = Column(Integer, nullable=True)  # 0=incorrect, 1=correct, NULL=unanswered
    response_time_ms = Column(Integer, nullable=True)
    
    # Relationships
    assessment = relationship("AssessmentAttempt", back_populates="questions")
    answers = relationship("UserAnswer", back_populates="question")

class UserAnswer(Base, TimestampMixin):
    """Test model for user answers to questions."""
    __tablename__ = 'user_answers'
    
    id = Column(Integer, primary_key=True)
    question_id = Column(Integer, ForeignKey('question_history.id'))
    answer_value = Column(String, nullable=False)
    is_correct = Column(Integer, nullable=False)  # 0=incorrect, 1=correct
    
    # Relationships
    question = relationship("QuestionHistory", back_populates="answers") 