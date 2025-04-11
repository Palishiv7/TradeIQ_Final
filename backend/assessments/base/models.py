"""
Base Assessment Models

This module defines the core data models for the assessment architecture,
including sessions, questions, answers, and evaluations.
"""

import uuid
import enum
import datetime
from typing import Dict, List, Any, Optional, Union, Set
from dataclasses import dataclass, field

from backend.common.serialization import SerializableMixin


class AssessmentError(Exception):
    """Base exception class for assessment-related errors."""
    pass


@dataclass
class AssessmentMetrics:
    """Metrics for assessment performance"""
    
    total_questions: int = 0
    answered_questions: int = 0
    correct_answers: int = 0
    average_time_ms: float = 0
    total_score: float = 0
    difficulty_level: float = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
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
            "difficulty_level": self.difficulty_level,
            "metadata": self.metadata
        }


class AssessmentType(enum.Enum):
    """Types of assessment supported by the system."""
    CANDLESTICK = "candlestick"
    MARKET_FUNDAMENTAL = "market_fundamental"
    MARKET_PSYCHOLOGY = "market_psychology"
    TECHNICAL_ANALYSIS = "technical_analysis"
    RISK_MANAGEMENT = "risk_management"


class QuestionDifficulty(enum.Enum):
    """Difficulty levels for assessment questions."""
    VERY_EASY = "very_easy"
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    VERY_HARD = "very_hard"
    
    @classmethod
    def from_numeric(cls, value: int) -> 'QuestionDifficulty':
        """Convert a numeric value (1-5) to a difficulty level."""
        mapping = {
            1: cls.VERY_EASY,
            2: cls.EASY,
            3: cls.MEDIUM,
            4: cls.HARD,
            5: cls.VERY_HARD
        }
        return mapping.get(value, cls.MEDIUM)
    
    def to_numeric(self) -> int:
        """Convert difficulty level to a numeric value (1-5)."""
        return {
            QuestionDifficulty.VERY_EASY: 1,
            QuestionDifficulty.EASY: 2,
            QuestionDifficulty.MEDIUM: 3,
            QuestionDifficulty.HARD: 4,
            QuestionDifficulty.VERY_HARD: 5
        }[self]


class SessionStatus(enum.Enum):
    """Status of an assessment session."""
    CREATED = "created"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    EXPIRED = "expired"
    ABANDONED = "abandoned"


@dataclass
class BaseQuestion(SerializableMixin):
    """
    Base class for all assessment questions.
    
    This defines the common attributes for all question types,
    including identifiers, difficulty, and metadata.
    """
    
    __serializable_fields__ = [
        "id", "question_type", "question_text", "difficulty", "topics",
        "subtopics", "created_at", "metadata", "answer_options"
    ]
    
    id: str
    question_type: str
    question_text: str
    difficulty: QuestionDifficulty
    topics: List[str]
    subtopics: Optional[List[str]] = None
    created_at: datetime.datetime = field(default_factory=datetime.datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    answer_options: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate and initialize after creation."""
        # Ensure ID is provided or generate one
        if not self.id:
            self.id = str(uuid.uuid4())
        
        # Ensure question type is provided
        if not self.question_type:
            raise ValueError("Question type is required")
            
        # Ensure question text is provided
        if not self.question_text:
            raise ValueError("Question text is required")
        
        # Convert difficulty string to enum if needed
        if isinstance(self.difficulty, str):
            try:
                self.difficulty = QuestionDifficulty(self.difficulty)
            except ValueError:
                raise ValueError(f"Invalid difficulty: {self.difficulty}")
        
        # Ensure topics is not empty
        if not self.topics:
            raise ValueError("At least one topic is required")
            
        # Ensure topics and subtopics are lists
        if not isinstance(self.topics, list):
            self.topics = [self.topics]
        
        if self.subtopics is not None and not isinstance(self.subtopics, list):
            self.subtopics = [self.subtopics]
            
        # Ensure created_at is a datetime
        if isinstance(self.created_at, str):
            try:
                self.created_at = datetime.datetime.fromisoformat(self.created_at)
            except ValueError:
                self.created_at = datetime.datetime.utcnow()
    
    def get_correct_answer(self) -> Any:
        """
        Get the correct answer for this question.
        
        Returns:
            The correct answer
            
        Raises:
            NotImplementedError: Subclasses must implement this method
        """
        raise NotImplementedError("Subclasses must implement get_correct_answer")
    
    def evaluate_answer(self, user_answer: Any) -> 'AnswerEvaluation':
        """
        Evaluate a user's answer to this question.
        
        Args:
            user_answer: The user's answer
            
        Returns:
            Evaluation of the answer
            
        Raises:
            NotImplementedError: Subclasses must implement this method
        """
        raise NotImplementedError("Subclasses must implement evaluate_answer")
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert question to dictionary format.
        
        Returns:
            Dictionary representation of question
        """
        result = {
            "id": self.id,
            "question_type": self.question_type,
            "question_text": self.question_text,
            "difficulty": self.difficulty.value,
            "topics": self.topics,
            "created_at": self.created_at.isoformat()
        }
        
        if self.subtopics:
            result["subtopics"] = self.subtopics
            
        if self.metadata:
            result["metadata"] = self.metadata
            
        if self.answer_options:
            result["answer_options"] = self.answer_options
            
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BaseQuestion':
        """
        Create a question from dictionary data.
        
        Args:
            data: Dictionary containing question data
            
        Returns:
            New question instance
        """
        # Handle date conversion
        if "created_at" in data and isinstance(data["created_at"], str):
            try:
                data["created_at"] = datetime.datetime.fromisoformat(data["created_at"])
            except ValueError:
                data["created_at"] = datetime.datetime.utcnow()
                
        return cls(**data)


@dataclass
class QuestionContent(SerializableMixin):
    """
    Content of a question, including text, options, and correct option.
    
    This class encapsulates the content of a question, separate from its
    metadata and other attributes.
    """
    
    __serializable_fields__ = [
        "question_text", "options", "correct_option", "explanation", 
        "chart_data", "media"
    ]
    
    question_text: str
    options: Optional[List[Dict[str, Any]]] = None
    correct_option: Optional[str] = None
    explanation: Optional[str] = None
    chart_data: Optional[Dict[str, Any]] = None
    media: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate and initialize after creation."""
        # Ensure question text is provided
        if not self.question_text:
            raise ValueError("Question text is required")
            
        # Initialize options if not provided
        if self.options is None:
            self.options = []
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert question content to dictionary format.
        
        Returns:
            Dictionary representation of question content
        """
        result = {
            "question_text": self.question_text,
        }
        
        if self.options:
            result["options"] = self.options
            
        if self.correct_option:
            result["correct_option"] = self.correct_option
            
        if self.explanation:
            result["explanation"] = self.explanation
            
        if self.chart_data:
            result["chart_data"] = self.chart_data
            
        if self.media:
            result["media"] = self.media
            
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QuestionContent':
        """
        Create question content from dictionary data.
        
        Args:
            data: Dictionary containing question content data
            
        Returns:
            New question content instance
        """
        return cls(
            question_text=data.get("question_text", ""),
            options=data.get("options"),
            correct_option=data.get("correct_option"),
            explanation=data.get("explanation"),
            chart_data=data.get("chart_data"),
            media=data.get("media")
        )


@dataclass
class AnswerEvaluation(SerializableMixin):
    """
    Evaluation of a user's answer to a question.
    
    Includes correctness, confidence, feedback, and explanation.
    """
    
    __serializable_fields__ = [
        "is_correct", "score", "confidence", "feedback",
        "explanation", "metadata"
    ]
    
    is_correct: bool
    score: float  # 0 to 1
    confidence: float  # 0 to 1
    feedback: str
    explanation: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UserAnswer(SerializableMixin):
    """
    Records a user's answer to a question, including timing and metadata.
    """
    
    __serializable_fields__ = [
        "question_id", "answer_value", "timestamp", "time_taken_ms",
        "evaluation", "metadata"
    ]
    
    question_id: str
    answer_value: Any
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.now)
    time_taken_ms: Optional[int] = None
    evaluation: Optional[AnswerEvaluation] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AssessmentSession(SerializableMixin):
    """
    Represents an assessment session for a user.
    
    Tracks questions, answers, and session state.
    """
    
    __serializable_fields__ = [
        "id", "assessment_type", "user_id", "questions", "current_question_index",
        "user_answers", "status", "created_at", "completed_at", "settings"
    ]
    
    id: str
    assessment_type: AssessmentType
    user_id: str
    questions: List[str]  # List of question IDs
    current_question_index: int = 0
    user_answers: List[Dict[str, Any]] = field(default_factory=list)
    status: SessionStatus = SessionStatus.IN_PROGRESS
    created_at: datetime.datetime = field(default_factory=datetime.datetime.utcnow)
    completed_at: Optional[datetime.datetime] = None
    settings: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize and validate session data."""
        # Generate ID if not provided
        if not self.id:
            self.id = str(uuid.uuid4())
            
        # Ensure user_id is provided
        if not self.user_id:
            raise ValueError("User ID is required")
            
        # Ensure questions list is provided
        if not self.questions:
            raise ValueError("Questions list is required")
            
        # Convert assessment_type string to enum if needed
        if isinstance(self.assessment_type, str):
            try:
                self.assessment_type = AssessmentType(self.assessment_type)
            except ValueError:
                raise ValueError(f"Invalid assessment type: {self.assessment_type}")
                
        # Convert status string to enum if needed
        if isinstance(self.status, str):
            try:
                self.status = SessionStatus(self.status)
            except ValueError:
                raise ValueError(f"Invalid session status: {self.status}")
                
        # Ensure created_at is a datetime
        if isinstance(self.created_at, str):
            try:
                self.created_at = datetime.datetime.fromisoformat(self.created_at)
            except ValueError:
                self.created_at = datetime.datetime.utcnow()
                
        # Ensure completed_at is a datetime if present
        if isinstance(self.completed_at, str):
            try:
                self.completed_at = datetime.datetime.fromisoformat(self.completed_at)
            except ValueError:
                self.completed_at = None
                
        # Validate current_question_index
        if self.current_question_index < 0:
            self.current_question_index = 0
        elif self.current_question_index >= len(self.questions) and self.questions:
            self.current_question_index = len(self.questions) - 1
            
        # Initialize answers list to match questions length if needed
        self._ensure_answers_list_integrity()
    
    def _ensure_answers_list_integrity(self):
        """Ensure user_answers list is properly initialized."""
        if not isinstance(self.user_answers, list):
            self.user_answers = []
            
        # Pad answer list with None for unanswered questions
        if len(self.user_answers) < len(self.questions):
            self.user_answers.extend([None] * (len(self.questions) - len(self.user_answers)))
    
    def get_current_question_id(self) -> Optional[str]:
        """
        Get the ID of the current question.
        
        Returns:
            The ID of the current question or None if no questions
        """
        if not self.questions or self.current_question_index >= len(self.questions):
            return None
        return self.questions[self.current_question_index]
    
    def record_answer(self, answer_data: Dict[str, Any]) -> None:
        """
        Record a user's answer for the current question.
        
        Args:
            answer_data: Answer data including 'answer', 'is_correct', 
                         'score', and 'time_taken'
                         
        Raises:
            ValueError: If session is already completed
            IndexError: If current question index is invalid
        """
        if self.status == SessionStatus.COMPLETED:
            raise ValueError("Cannot record answer for completed session")
            
        if self.current_question_index >= len(self.questions):
            raise IndexError(f"Invalid question index: {self.current_question_index}")
        
        # Ensure answer_data has required fields
        if 'answer' not in answer_data:
            raise ValueError("Answer data must include 'answer' field")
        
        # Record timestamp if not provided
        if 'timestamp' not in answer_data:
            answer_data['timestamp'] = datetime.datetime.utcnow().isoformat()
            
        # Add question_id to the answer data
        answer_data['question_id'] = self.get_current_question_id()
        
        # Record the answer
        self._ensure_answers_list_integrity()
        self.user_answers[self.current_question_index] = answer_data
    
    def next_question(self) -> Optional[str]:
        """
        Move to the next question and return its ID.
        
        Returns:
            The ID of the next question or None if no more questions
            
        Raises:
            ValueError: If session is already completed
        """
        if self.status == SessionStatus.COMPLETED:
            raise ValueError("Cannot advance completed session")
            
        if self.current_question_index + 1 >= len(self.questions):
            return None
            
        self.current_question_index += 1
        return self.get_current_question_id()
    
    def previous_question(self) -> Optional[str]:
        """
        Move to the previous question and return its ID.
        
        Returns:
            The ID of the previous question or None if at first question
            
        Raises:
            ValueError: If session is already completed
        """
        if self.status == SessionStatus.COMPLETED:
            raise ValueError("Cannot navigate in completed session")
            
        if self.current_question_index <= 0:
            return None
            
        self.current_question_index -= 1
        return self.get_current_question_id()
    
    def complete(self) -> None:
        """
        Mark the session as completed.
        
        Raises:
            ValueError: If session is already completed
        """
        if self.status == SessionStatus.COMPLETED:
            raise ValueError("Session already completed")
            
        self.status = SessionStatus.COMPLETED
        self.completed_at = datetime.datetime.utcnow()
    
    def get_topics(self) -> List[str]:
        """
        Get the unique topics covered in this session.
        
        Returns:
            List of unique topics
        """
        # This method would typically combine topics from all questions,
        # but since we only have question IDs, this is a placeholder.
        # Actual implementation would require question data.
        return []
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get user performance metrics for this session.
        
        Returns:
            Dictionary containing performance metrics
        """
        total_questions = len(self.questions)
        answered_questions = sum(1 for a in self.user_answers if a is not None)
        correct_answers = sum(1 for a in self.user_answers if a and a.get('is_correct', False))
        
        # Calculate time metrics if available
        time_taken = [a.get('time_taken', 0) for a in self.user_answers if a is not None]
        avg_time = sum(time_taken) / len(time_taken) if time_taken else 0
        
        return {
            "total_questions": total_questions,
            "answered_questions": answered_questions,
            "correct_answers": correct_answers,
            "accuracy": correct_answers / answered_questions if answered_questions > 0 else 0,
            "completion_rate": answered_questions / total_questions if total_questions > 0 else 0,
            "average_time_per_question": avg_time
        }
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert session to dictionary format.
        
        Returns:
            Dictionary representation of session
        """
        result = {
            "id": self.id,
            "assessment_type": self.assessment_type.value,
            "user_id": self.user_id,
            "questions": self.questions,
            "current_question_index": self.current_question_index,
            "user_answers": self.user_answers,
            "status": self.status.value,
            "created_at": self.created_at.isoformat()
        }
        
        if self.completed_at:
            result["completed_at"] = self.completed_at.isoformat()
            
        if self.settings:
            result["settings"] = self.settings
            
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AssessmentSession':
        """
        Create a session from dictionary data.
        
        Args:
            data: Dictionary containing session data
            
        Returns:
            New session instance
        """
        # Handle date conversions
        if "created_at" in data and isinstance(data["created_at"], str):
            try:
                data["created_at"] = datetime.datetime.fromisoformat(data["created_at"])
            except ValueError:
                data["created_at"] = datetime.datetime.utcnow()
                
        if "completed_at" in data and isinstance(data["completed_at"], str):
            try:
                data["completed_at"] = datetime.datetime.fromisoformat(data["completed_at"])
            except ValueError:
                data["completed_at"] = None
                
        return cls(**data) 