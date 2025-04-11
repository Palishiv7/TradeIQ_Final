"""
Base Assessment Controllers

This module defines the controller interfaces for the assessment architecture,
providing the API layer for interacting with assessment services.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, TypeVar, Generic

from backend.assessments.base.models import (
    BaseQuestion,
    AssessmentSession,
    AnswerEvaluation,
    QuestionDifficulty
)

from backend.assessments.base.services import AssessmentService

# Type variables for generics
T = TypeVar('T', bound=BaseQuestion)
S = TypeVar('S', bound=AssessmentSession)


class BaseAssessmentController(Generic[T, S], ABC):
    """
    Abstract base controller for assessment APIs.
    
    This controller provides the API layer for interacting with assessment
    services, handling requests and responses for assessment functionality.
    It ensures proper validation, error handling, and response formatting.
    """
    
    @property
    @abstractmethod
    def service(self) -> Any:
        """
        Get the assessment service for this controller.
        
        Returns:
            Assessment service instance
        """
        pass
    
    @abstractmethod
    async def create_session(
        self,
        user_id: str,
        question_count: int = 10,
        topics: Optional[List[str]] = None,
        difficulty: Optional[str] = None,
        settings: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a new assessment session.
        
        Args:
            user_id: The user's unique identifier
            question_count: Number of questions in the session
            topics: Optional list of topics to include
            difficulty: Optional difficulty level
            settings: Optional session settings
            
        Returns:
            Dictionary containing the created session details
            
        Raises:
            HTTPException: If session creation fails
            ValidationError: If parameters are invalid
        """
        pass
    
    @abstractmethod
    async def get_session(self, session_id: str) -> Dict[str, Any]:
        """
        Retrieve an assessment session by ID.
        
        Args:
            session_id: The session's unique identifier
            
        Returns:
            Dictionary containing the session details
            
        Raises:
            HTTPException: If session retrieval fails or session not found
        """
        pass
    
    @abstractmethod
    async def get_session_question(
        self,
        session_id: str,
        question_index: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get the current or specified question in a session.
        
        Args:
            session_id: The session's unique identifier
            question_index: Optional index of the question to retrieve
            
        Returns:
            Dictionary containing the question details
            
        Raises:
            HTTPException: If question retrieval fails or question not found
            ValidationError: If question index is invalid
        """
        pass
    
    @abstractmethod
    async def submit_answer(
        self,
        session_id: str,
        question_id: str,
        answer: Any,
        time_taken_ms: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Submit an answer to a question in a session.
        
        Args:
            session_id: The session's unique identifier
            question_id: The question's unique identifier
            answer: The user's answer
            time_taken_ms: Optional time taken to answer in milliseconds
            
        Returns:
            Dictionary containing evaluation results
            
        Raises:
            HTTPException: If answer submission fails
            ValidationError: If parameters are invalid
        """
        pass
    
    @abstractmethod
    async def next_question(self, session_id: str) -> Dict[str, Any]:
        """
        Move to the next question in a session.
        
        Args:
            session_id: The session's unique identifier
            
        Returns:
            Dictionary containing the next question details
            
        Raises:
            HTTPException: If navigation fails or no more questions
            ValidationError: If session is not in progress
        """
        pass
    
    @abstractmethod
    async def complete_session(self, session_id: str) -> Dict[str, Any]:
        """
        Complete an assessment session.
        
        Args:
            session_id: The session's unique identifier
            
        Returns:
            Dictionary containing session results
            
        Raises:
            HTTPException: If session completion fails
            ValidationError: If session is not ready to be completed
        """
        pass
    
    @abstractmethod
    async def get_session_results(self, session_id: str) -> Dict[str, Any]:
        """
        Get results for a completed session.
        
        Args:
            session_id: The session's unique identifier
            
        Returns:
            Dictionary containing session results and performance metrics
            
        Raises:
            HTTPException: If results retrieval fails
            ValidationError: If session is not completed
        """
        pass
    
    @abstractmethod
    async def get_user_sessions(
        self,
        user_id: str,
        limit: int = 10,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Get sessions for a specific user.
        
        Args:
            user_id: The user's unique identifier
            limit: Maximum number of sessions to return
            offset: Number of sessions to skip
            
        Returns:
            List of session details
            
        Raises:
            HTTPException: If sessions retrieval fails
        """
        pass
    
    @abstractmethod
    async def get_user_performance(self, user_id: str) -> Dict[str, Any]:
        """
        Get performance metrics for a user.
        
        Args:
            user_id: The user's unique identifier
            
        Returns:
            Dictionary containing performance metrics
            
        Raises:
            HTTPException: If performance retrieval fails
        """
        pass
    
    @abstractmethod
    async def get_topic_performance(
        self,
        user_id: str,
        topic: str
    ) -> Dict[str, Any]:
        """
        Get performance metrics for a user on a specific topic.
        
        Args:
            user_id: The user's unique identifier
            topic: The topic to get performance for
            
        Returns:
            Dictionary containing topic-specific performance metrics
            
        Raises:
            HTTPException: If performance retrieval fails
        """
        pass
    
    @abstractmethod
    async def get_explanation(
        self,
        question_id: str,
        user_answer: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Get an explanation for a question and optionally an answer.
        
        Args:
            question_id: The question's unique identifier
            user_answer: Optional user answer to explain
            
        Returns:
            Dictionary containing explanation details
            
        Raises:
            HTTPException: If explanation generation fails
            ValidationError: If question not found
        """
        pass
    
    @abstractmethod
    def get_recommended_topics(self, user_id: str, count: int = 3) -> List[Dict[str, Any]]:
        """
        Get recommended topics for a user to focus on based on performance.
        
        Args:
            user_id: User identifier
            count: Number of topics to recommend
            
        Returns:
            List of recommended topic data
        """
        pass 