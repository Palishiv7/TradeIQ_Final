"""
Base Assessment Module [DEPRECATED]

IMPORTANT: This module is being deprecated in favor of the models in
backend.assessments.base.models. New code should import directly from
those modules instead.

This module now serves as a compatibility layer to ensure existing
imports continue to work during the transition period.
"""

import abc
import uuid
import time
import logging
import json
from datetime import datetime, timedelta
from typing import (
    Dict, List, Any, Optional, Union, Tuple, TypeVar, Generic, Protocol, 
    Callable, Set, Iterable, AsyncIterator, cast, overload, ClassVar
)
from enum import Enum, auto
from dataclasses import dataclass, field

# Import from the canonical location
from backend.assessments.base.models import (
    AssessmentType,
    QuestionDifficulty,
    SessionStatus,
    BaseQuestion as Question,
    QuestionContent,
    AnswerEvaluation,
    UserAnswer,
    AssessmentSession
)

# Configure logging
logger = logging.getLogger(__name__)

# Define Difficulty for backward compatibility
class Difficulty(Enum):
    """Difficulty levels for questions (compatibility class)."""
    VERY_EASY = "very_easy"
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    VERY_HARD = "very_hard"
    
    @classmethod
    def from_question_difficulty(cls, difficulty: QuestionDifficulty) -> 'Difficulty':
        """Convert QuestionDifficulty to Difficulty."""
        try:
            return cls(difficulty.value)
        except ValueError:
            return cls.MEDIUM
            
    def to_question_difficulty(self) -> QuestionDifficulty:
        """Convert to QuestionDifficulty."""
        try:
            return QuestionDifficulty(self.value)
        except ValueError:
            return QuestionDifficulty.MEDIUM

# Re-export types from assessments/base for backward compatibility
__all__ = [
    'AssessmentType',
    'QuestionDifficulty',
    'SessionStatus',
    'Question',
    'QuestionContent',
    'AnswerEvaluation',
    'UserAnswer',
    'AssessmentSession',
    'DomainEvent',
    'EventDispatcher',
    'AssessmentMetrics',
    'SessionCompletedEvent',
    'QuestionGeneratedEvent',
    'Difficulty',  # Add Difficulty to exports
    'AnswerSubmittedEvent',
    'AnswerEvaluatedEvent'
]

#------------------------------------------------------------------------------
# Domain Events
#------------------------------------------------------------------------------

class DomainEvent:
    """Base class for all domain events in the assessment system"""
    
    def __init__(self, event_id: str = None, timestamp: float = None):
        """Initialize a domain event with optional ID and timestamp"""
        self.event_id = event_id or str(uuid.uuid4())
        self.timestamp = timestamp or time.time()
        self.event_type = self.__class__.__name__


class EventDispatcher:
    """Event dispatcher for domain events"""
    
    def __init__(self):
        """Initialize the event dispatcher"""
        self._subscribers = {}
    
    def subscribe(self, event_type, handler):
        """Subscribe a handler to an event type"""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(handler)
    
    def dispatch(self, event):
        """Dispatch an event to all subscribers"""
        event_type = event.__class__.__name__
        if event_type in self._subscribers:
            for handler in self._subscribers[event_type]:
                handler(event)
    
    def unsubscribe(self, event_type, handler):
        """Unsubscribe a handler from an event type"""
        if event_type in self._subscribers and handler in self._subscribers[event_type]:
            self._subscribers[event_type].remove(handler)


#------------------------------------------------------------------------------
# Assessment Events
#------------------------------------------------------------------------------

class SessionCompletedEvent(DomainEvent):
    """Event raised when an assessment session is completed"""
    
    def __init__(self, session_id, user_id, assessment_type, score, event_id=None, timestamp=None):
        """Initialize the session completed event"""
        super().__init__(event_id, timestamp)
        self.session_id = session_id
        self.user_id = user_id
        self.assessment_type = assessment_type
        self.score = score


class QuestionGeneratedEvent(DomainEvent):
    """Event raised when a question is generated"""
    
    def __init__(self, question_id, difficulty, topics, event_id=None, timestamp=None):
        """Initialize the question generated event"""
        super().__init__(event_id, timestamp)
        self.question_id = question_id
        self.difficulty = difficulty
        self.topics = topics


class AnswerSubmittedEvent(DomainEvent):
    """Event raised when an answer is submitted"""
    
    def __init__(self, session_id, question_id, user_id, answer, time_taken_ms, event_id=None, timestamp=None):
        """Initialize the answer submitted event"""
        super().__init__(event_id, timestamp)
        self.session_id = session_id
        self.question_id = question_id
        self.user_id = user_id
        self.answer = answer
        self.time_taken_ms = time_taken_ms


class AnswerEvaluatedEvent(DomainEvent):
    """Event raised when an answer is evaluated"""
    
    def __init__(self, session_id, question_id, user_id, is_correct, score, event_id=None, timestamp=None):
        """Initialize the answer evaluated event"""
        super().__init__(event_id, timestamp)
        self.session_id = session_id
        self.question_id = question_id
        self.user_id = user_id
        self.is_correct = is_correct
        self.score = score


#------------------------------------------------------------------------------
# Assessment Metrics
#------------------------------------------------------------------------------

@dataclass
class AssessmentMetrics:
    """Metrics for assessment performance"""
    
    total_questions: int = 0
    answered_questions: int = 0
    correct_answers: int = 0
    average_time_ms: float = 0
    total_score: float = 0
    difficulty_level: float = 0
    
    @property
    def accuracy(self) -> float:
        """Calculate accuracy as percentage of correct answers"""
        if self.answered_questions == 0:
            return 0
        return (self.correct_answers / self.answered_questions) * 100
    
    @property
    def completion_rate(self) -> float:
        """Calculate completion rate as percentage of questions answered"""
        if self.total_questions == 0:
            return 0
        return (self.answered_questions / self.total_questions) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary"""
        return {
            "total_questions": self.total_questions,
            "answered_questions": self.answered_questions,
            "correct_answers": self.correct_answers,
            "accuracy": self.accuracy,
            "completion_rate": self.completion_rate,
            "average_time_ms": self.average_time_ms,
            "total_score": self.total_score,
            "difficulty_level": self.difficulty_level
        } 