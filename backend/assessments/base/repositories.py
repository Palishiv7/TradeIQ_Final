"""
Base Assessment Repositories

This module defines the base repository interfaces for the assessment architecture,
providing standardized data access patterns for assessment components.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, TypeVar, Generic, Set
import datetime

from backend.assessments.base.models import (
    AssessmentType,
    BaseQuestion,
    AssessmentSession
)

# Type variables for generics
T = TypeVar('T', bound=BaseQuestion)
S = TypeVar('S', bound=AssessmentSession)


class QuestionRepository(Generic[T]):
    """
    Abstract repository interface for managing assessment questions.
    
    This repository provides methods for storing, retrieving, and querying 
    questions for assessments. It serves as a base interface that should 
    be implemented by specific question repositories.
    
    Type Parameters:
        T: The specific question type managed by this repository.
    """
    
    @property
    @abstractmethod
    def domain_type(self) -> str:
        """
        Get the domain type identifier for this repository.
        
        Returns:
            String identifier for the domain type
        """
        pass
    
    @property
    @abstractmethod
    def table_name(self) -> str:
        """
        Get the database table name for this repository.
        
        Returns:
            String name of the database table
        """
        pass
    
    @abstractmethod
    async def get_by_id(self, question_id: str) -> Optional[T]:
        """
        Retrieve a question by its ID.
        
        Args:
            question_id: The unique identifier for the question
            
        Returns:
            The question if found, None otherwise
            
        Raises:
            RepositoryError: If an error occurs during retrieval
        """
        pass
    
    @abstractmethod
    async def save(self, question: T) -> bool:
        """
        Save a question to the repository.
        
        Args:
            question: The question to save
            
        Returns:
            True if successful, False otherwise
            
        Raises:
            RepositoryError: If an error occurs during save operation
        """
        pass
    
    @abstractmethod
    async def delete(self, question_id: str) -> bool:
        """
        Delete a question from the repository.
        
        Args:
            question_id: The unique identifier for the question to delete
            
        Returns:
            True if successful, False otherwise
            
        Raises:
            RepositoryError: If an error occurs during deletion
        """
        pass
    
    @abstractmethod
    async def find_by_difficulty(
        self, 
        difficulty: str, 
        limit: int = 10, 
        offset: int = 0
    ) -> List[T]:
        """
        Find questions matching the specified difficulty.
        
        Args:
            difficulty: The difficulty level to search for
            limit: Maximum number of results to return
            offset: Number of results to skip
            
        Returns:
            List of matching questions
            
        Raises:
            RepositoryError: If an error occurs during search
        """
        pass
    
    @abstractmethod
    async def find_by_topics(
        self, 
        topics: List[str], 
        limit: int = 10, 
        offset: int = 0
    ) -> List[T]:
        """
        Find questions related to the specified topics.
        
        Args:
            topics: List of topics to search for
            limit: Maximum number of results to return
            offset: Number of results to skip
            
        Returns:
            List of matching questions
            
        Raises:
            RepositoryError: If an error occurs during search
        """
        pass
    
    @abstractmethod
    async def find_by_criteria(
        self,
        criteria: Dict[str, Any],
        limit: int = 10,
        offset: int = 0
    ) -> List[T]:
        """
        Find questions matching the specified criteria.
        
        Args:
            criteria: Dictionary of search criteria
            limit: Maximum number of results to return
            offset: Number of results to skip
            
        Returns:
            List of matching questions
            
        Raises:
            RepositoryError: If an error occurs during search
        """
        pass
    
    @abstractmethod
    async def count_by_criteria(self, criteria: Dict[str, Any]) -> int:
        """
        Count questions matching the specified criteria.
        
        Args:
            criteria: Dictionary of search criteria
            
        Returns:
            Count of matching questions
            
        Raises:
            RepositoryError: If an error occurs during counting
        """
        pass


class SessionRepository(Generic[S]):
    """
    Abstract repository interface for managing assessment sessions.
    
    This repository provides methods for storing, retrieving, and querying
    assessment sessions. It serves as a base interface that should be
    implemented by specific session repositories.
    
    Type Parameters:
        S: The specific session type managed by this repository.
    """
    
    @property
    @abstractmethod
    def domain_type(self) -> str:
        """
        Get the domain type identifier for this repository.
        
        Returns:
            String identifier for the domain type
        """
        pass
    
    @property
    @abstractmethod
    def table_name(self) -> str:
        """
        Get the database table name for this repository.
        
        Returns:
            String name of the database table
        """
        pass
    
    @abstractmethod
    async def get_by_id(self, session_id: str) -> Optional[S]:
        """
        Retrieve a session by its ID.
        
        Args:
            session_id: The unique identifier for the session
            
        Returns:
            The session if found, None otherwise
            
        Raises:
            RepositoryError: If an error occurs during retrieval
        """
        pass
    
    @abstractmethod
    async def save(self, session: S) -> bool:
        """
        Save a session to the repository.
        
        Args:
            session: The session to save
            
        Returns:
            True if successful, False otherwise
            
        Raises:
            RepositoryError: If an error occurs during save operation
        """
        pass
    
    @abstractmethod
    async def delete(self, session_id: str) -> bool:
        """
        Delete a session from the repository.
        
        Args:
            session_id: The unique identifier for the session to delete
            
        Returns:
            True if successful, False otherwise
            
        Raises:
            RepositoryError: If an error occurs during deletion
        """
        pass
    
    @abstractmethod
    async def find_by_user_id(
        self, 
        user_id: str,
        limit: int = 10,
        offset: int = 0,
        status: Optional[str] = None
    ) -> List[S]:
        """
        Find sessions for a specific user.
        
        Args:
            user_id: The user's unique identifier
            limit: Maximum number of results to return
            offset: Number of results to skip
            status: Optional filter for session status
            
        Returns:
            List of matching sessions
            
        Raises:
            RepositoryError: If an error occurs during search
        """
        pass
    
    @abstractmethod
    async def find_by_date_range(
        self,
        user_id: str,
        start_date: datetime.datetime,
        end_date: datetime.datetime,
        limit: int = 10,
        offset: int = 0
    ) -> List[S]:
        """
        Find sessions within a specific date range.
        
        Args:
            user_id: The user's unique identifier
            start_date: Start of the date range
            end_date: End of the date range
            limit: Maximum number of results to return
            offset: Number of results to skip
            
        Returns:
            List of matching sessions
            
        Raises:
            RepositoryError: If an error occurs during search
        """
        pass
    
    @abstractmethod
    async def find_by_criteria(
        self,
        criteria: Dict[str, Any],
        limit: int = 10,
        offset: int = 0
    ) -> List[S]:
        """
        Find sessions matching the specified criteria.
        
        Args:
            criteria: Dictionary of search criteria
            limit: Maximum number of results to return
            offset: Number of results to skip
            
        Returns:
            List of matching sessions
            
        Raises:
            RepositoryError: If an error occurs during search
        """
        pass
    
    @abstractmethod
    async def count_by_criteria(self, criteria: Dict[str, Any]) -> int:
        """
        Count sessions matching the specified criteria.
        
        Args:
            criteria: Dictionary of search criteria
            
        Returns:
            Count of matching sessions
            
        Raises:
            RepositoryError: If an error occurs during counting
        """
        pass
    
    @abstractmethod
    async def get_latest_session(self, user_id: str) -> Optional[S]:
        """
        Get the most recent session for a user.
        
        Args:
            user_id: The user's unique identifier
            
        Returns:
            The most recent session if found, None otherwise
            
        Raises:
            RepositoryError: If an error occurs during retrieval
        """
        pass
    
    @abstractmethod
    async def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        """
        Get session statistics for a user.
        
        Args:
            user_id: The user's unique identifier
            
        Returns:
            Dictionary containing user session statistics
            
        Raises:
            RepositoryError: If an error occurs during statistics calculation
        """
        pass


class AssessmentRepository(ABC):
    """
    Abstract repository interface for assessment data access.
    
    This repository provides access to question and session repositories 
    for a specific assessment type, along with methods for retrieving 
    assessment data, user performance metrics, and recommended content.
    """
    
    @property
    @abstractmethod
    def question_repository(self) -> QuestionRepository:
        """
        Get the repository for managing assessment questions.
        
        Returns:
            Question repository instance
        """
        pass
    
    @property
    @abstractmethod
    def session_repository(self) -> SessionRepository:
        """
        Get the repository for managing assessment sessions.
        
        Returns:
            Session repository instance
        """
        pass
    
    @abstractmethod
    async def get_questions_for_session(
        self,
        difficulty: Optional[str] = None,
        topics: Optional[List[str]] = None,
        count: int = 10,
        user_id: Optional[str] = None
    ) -> List[Any]:
        """
        Get questions for a new assessment session.
        
        Args:
            difficulty: Optional difficulty level filter
            topics: Optional list of topics to include
            count: Number of questions to retrieve
            user_id: Optional user ID for personalized questions
            
        Returns:
            List of questions for the session
            
        Raises:
            RepositoryError: If an error occurs during question retrieval
        """
        pass
    
    @abstractmethod
    async def get_user_performance(self, user_id: str) -> Dict[str, Any]:
        """
        Get overall performance metrics for a user across all sessions.
        
        Args:
            user_id: The user's unique identifier
            
        Returns:
            Dictionary containing overall performance metrics
            
        Raises:
            RepositoryError: If an error occurs during metrics retrieval
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
            RepositoryError: If an error occurs during metrics retrieval
        """
        pass
    
    @abstractmethod
    async def get_recommended_topics(
        self,
        user_id: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get recommended topics for a user based on performance.
        
        Args:
            user_id: The user's unique identifier
            limit: Maximum number of recommendations to return
            
        Returns:
            List of recommended topics with reasoning
            
        Raises:
            RepositoryError: If an error occurs during recommendation generation
        """
        pass
    
    @abstractmethod
    async def get_difficulty_distribution(self, user_id: str) -> Dict[str, float]:
        """
        Get the distribution of question difficulties appropriate for a user.
        
        Args:
            user_id: The user's unique identifier
            
        Returns:
            Dictionary mapping difficulty levels to probabilities
            
        Raises:
            RepositoryError: If an error occurs during distribution calculation
        """
        pass
    
    @abstractmethod
    async def update_user_metrics(self, user_id: str, session_id: str) -> bool:
        """
        Update user performance metrics after a session is completed.
        
        Args:
            user_id: The user's unique identifier
            session_id: The session identifier
            
        Returns:
            True if successful, False otherwise
            
        Raises:
            RepositoryError: If an error occurs during metrics update
        """
        pass 