"""
Common Assessment Service Components [DEPRECATED]

IMPORTANT: This module is being deprecated in favor of the services in
backend.assessments.base.services. New code should import directly from
those modules instead.

This module now serves as a compatibility layer to ensure existing
imports continue to work during the transition period.
"""

import uuid
import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple, Union, Generic, TypeVar
from datetime import datetime, timedelta

from backend.common.logger import app_logger
from backend.common.base_assessment import (
    Question, AssessmentSession, AssessmentMetrics, 
    SessionCompletedEvent, EventDispatcher, QuestionGeneratedEvent
)

# Import from the canonical location
from backend.assessments.base.services import (
    AssessmentService,
    QuestionGenerator,
    AnswerEvaluator,
    ExplanationGenerator,
    PerformanceAnalyzer
)

# Module logger
logger = app_logger.getChild("common.assessment_service")

# Type variables for generic implementations
T_Question = TypeVar('T_Question', bound=Question)
T_Session = TypeVar('T_Session', bound=AssessmentSession)

# Re-export types from assessments/base for backward compatibility
__all__ = [
    'BaseSessionRepository',
    'BaseQuestionRepository',
    'AssessmentService',
    'QuestionGenerator',
    'AnswerEvaluator',
    'ExplanationGenerator',
    'PerformanceAnalyzer'
]

# Create base repository interfaces for backward compatibility
class BaseSessionRepository(ABC, Generic[T_Session]):
    """Abstract base class for session repositories."""
    
    @abstractmethod
    async def get_session(self, session_id: str) -> Optional[T_Session]:
        """Get a session by ID."""
        pass
    
    @abstractmethod
    async def save_session(self, session: T_Session) -> bool:
        """Save a session."""
        pass
    
    @abstractmethod
    async def get_user_sessions(self, user_id: str, limit: int = 10) -> List[T_Session]:
        """Get recent sessions for a user."""
        pass


class BaseQuestionRepository(ABC, Generic[T_Question]):
    """Abstract base class for question repositories."""
    
    @abstractmethod
    async def get_question(self, question_id: str) -> Optional[T_Question]:
        """Get a question by ID."""
        pass
    
    @abstractmethod
    async def save_question(self, question: T_Question) -> bool:
        """Save a question."""
        pass
    
    @abstractmethod
    async def get_questions_by_criteria(
        self,
        difficulty: Optional[float] = None,
        topics: Optional[List[str]] = None,
        exclude_ids: Optional[List[str]] = None,
        limit: int = 10
    ) -> List[T_Question]:
        """Get questions by criteria."""
        pass 