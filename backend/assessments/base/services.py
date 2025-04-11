"""
Base Assessment Services

This module defines the service interfaces for the assessment architecture,
providing business logic for assessment functionality.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, TypeVar, Generic

from backend.common.performance.tracker import PerformanceTracker
from backend.common.performance.difficulty import AdaptiveDifficultyEngine
from backend.common.performance.forgetting import SpacedRepetitionScheduler

from backend.assessments.base.models import (
    BaseQuestion,
    AssessmentSession,
    AnswerEvaluation,
    QuestionDifficulty
)

from backend.assessments.base.repositories import (
    QuestionRepository,
    SessionRepository
)

# Type variables for generics
T = TypeVar('T', bound=BaseQuestion)
S = TypeVar('S', bound=AssessmentSession)
T_Question = TypeVar('T_Question', bound=BaseQuestion)
T_Evaluation = TypeVar('T_Evaluation', bound=AnswerEvaluation)
T_Session = TypeVar('T_Session', bound=AssessmentSession)


class QuestionGenerator(Generic[T_Question], ABC):
    """
    Abstract interface for question generation in assessments.
    
    Responsible for creating and curating appropriate questions for assessment
    sessions based on various criteria like difficulty, topic, and adaptivity.
    
    Type Parameters:
        T_Question: The type of question to generate
    """
    
    @abstractmethod
    async def generate_random_question(
        self,
        difficulty: Optional[str] = None,
        topic: Optional[str] = None
    ) -> T_Question:
        """
        Generate a random question with optional constraints.
        
        Args:
            difficulty: Optional difficulty level constraint
            topic: Optional topic constraint
            
        Returns:
            A randomly generated question
            
        Raises:
            GenerationError: If question generation fails
        """
        pass
    
    @abstractmethod
    async def generate_questions_batch(
        self,
        count: int,
        difficulty: Optional[str] = None,
        topics: Optional[List[str]] = None,
        shuffle: bool = True
    ) -> List[T_Question]:
        """
        Generate a batch of questions with optional constraints.
        
        Args:
            count: Number of questions to generate
            difficulty: Optional difficulty level constraint
            topics: Optional list of topic constraints
            shuffle: Whether to shuffle the resulting questions
            
        Returns:
            List of generated questions
            
        Raises:
            GenerationError: If question generation fails
        """
        pass
    
    @abstractmethod
    async def generate_adaptive_questions(
        self,
        user_id: str,
        count: int,
        topics: Optional[List[str]] = None
    ) -> List[T_Question]:
        """
        Generate questions adapted to a user's skill level.
        
        Args:
            user_id: User identifier to adapt for
            count: Number of questions to generate
            topics: Optional list of topic constraints
            
        Returns:
            List of adaptively generated questions
            
        Raises:
            GenerationError: If question generation fails
            UserNotFoundError: If user profile not found
        """
        pass
    
    @abstractmethod
    async def generate_quiz(
        self,
        topic: str,
        question_count: int,
        difficulty: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a complete quiz on a specific topic.
        
        Args:
            topic: The topic for the quiz
            question_count: Number of questions to include
            difficulty: Optional difficulty level constraint
            
        Returns:
            Dictionary containing quiz details and questions
            
        Raises:
            GenerationError: If quiz generation fails
        """
        pass
    
    @abstractmethod
    async def generate_spaced_repetition_questions(
        self,
        user_id: str,
        count: int
    ) -> List[T_Question]:
        """
        Generate questions following spaced repetition principles.
        
        Retrieves questions for a user based on their past performance
        and optimal spacing for memory retention.
        
        Args:
            user_id: User identifier to generate for
            count: Number of questions to generate
            
        Returns:
            List of questions for spaced repetition
            
        Raises:
            GenerationError: If question generation fails
            UserNotFoundError: If user profile not found
        """
        pass


class AnswerEvaluator(Generic[T_Question, T_Evaluation], ABC):
    """
    Abstract interface for evaluating answers in assessments.
    
    Responsible for judging the correctness of user answers, providing
    detailed evaluations, and generating feedback.
    
    Type Parameters:
        T_Question: The type of question being evaluated
        T_Evaluation: The type of evaluation result
    """
    
    @abstractmethod
    async def evaluate_answer(
        self,
        question: T_Question,
        user_answer: Any
    ) -> T_Evaluation:
        """
        Evaluate a user's answer to a question.
        
        Args:
            question: The question being answered
            user_answer: The user's answer
            
        Returns:
            Evaluation of the answer
            
        Raises:
            EvaluationError: If evaluation fails
        """
        pass
    
    @abstractmethod
    async def evaluate_session_answers(
        self,
        questions: List[T_Question],
        user_answers: List[Any]
    ) -> List[T_Evaluation]:
        """
        Evaluate all answers for a session.
        
        Args:
            questions: List of questions
            user_answers: List of corresponding user answers
            
        Returns:
            List of evaluations for each answer
            
        Raises:
            EvaluationError: If evaluation fails
            ValueError: If questions and answers lists have different lengths
        """
        pass
    
    @abstractmethod
    async def evaluate_partial_answer(
        self,
        question: T_Question,
        partial_answer: Any
    ) -> Dict[str, Any]:
        """
        Evaluate a partial or in-progress answer.
        
        Provides feedback during answer construction, before final submission.
        
        Args:
            question: The question being answered
            partial_answer: The partial answer
            
        Returns:
            Dictionary containing evaluation details and guidance
            
        Raises:
            EvaluationError: If evaluation fails
        """
        pass
    
    @abstractmethod
    async def generate_feedback(
        self,
        question: T_Question,
        user_answer: Any,
        evaluation: T_Evaluation
    ) -> str:
        """
        Generate detailed feedback for an answer.
        
        Args:
            question: The question that was answered
            user_answer: The user's answer
            evaluation: The evaluation of the answer
            
        Returns:
            Detailed feedback string
            
        Raises:
            EvaluationError: If feedback generation fails
        """
        pass
    
    @abstractmethod
    async def get_answer_statistics(
        self,
        question_id: str
    ) -> Dict[str, Any]:
        """
        Get statistics about answers for a specific question.
        
        Args:
            question_id: The question identifier
            
        Returns:
            Dictionary containing answer statistics
            
        Raises:
            EvaluationError: If statistics retrieval fails
        """
        pass


class ExplanationGenerator(Generic[T_Question], ABC):
    """
    Abstract interface for generating explanations in assessments.
    
    Responsible for creating detailed explanations for questions, answers,
    and concepts to aid in user learning and understanding.
    
    Type Parameters:
        T_Question: The type of question to generate explanations for
    """
    
    @abstractmethod
    async def explain_question(self, question: T_Question) -> str:
        """
        Generate an explanation of the question itself.
        
        Args:
            question: The question to explain
            
        Returns:
            Detailed explanation of the question
            
        Raises:
            ExplanationError: If explanation generation fails
        """
        pass
    
    @abstractmethod
    async def explain_correct_answer(self, question: T_Question) -> str:
        """
        Generate an explanation of the correct answer.
        
        Args:
            question: The question to explain the answer for
            
        Returns:
            Detailed explanation of the correct answer
            
        Raises:
            ExplanationError: If explanation generation fails
        """
        pass
    
    @abstractmethod
    async def explain_user_answer(
        self,
        question: T_Question,
        user_answer: Any,
        is_correct: bool
    ) -> str:
        """
        Generate an explanation of the user's answer.
        
        Args:
            question: The question that was answered
            user_answer: The user's answer
            is_correct: Whether the user's answer was correct
            
        Returns:
            Detailed explanation of why the user's answer was correct or incorrect
            
        Raises:
            ExplanationError: If explanation generation fails
        """
        pass
    
    @abstractmethod
    async def explain_topic(self, topic: str, difficulty: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate an explanation of a topic.
        
        Args:
            topic: The topic to explain
            difficulty: Optional difficulty level to tailor the explanation
            
        Returns:
            Dictionary containing topic explanation and related resources
            
        Raises:
            ExplanationError: If explanation generation fails
        """
        pass
    
    @abstractmethod
    async def generate_learning_resources(
        self,
        question: T_Question,
        was_correct: bool
    ) -> List[Dict[str, Any]]:
        """
        Generate learning resources related to a question.
        
        Provides additional study materials based on whether the user
        answered correctly or not.
        
        Args:
            question: The question to generate resources for
            was_correct: Whether the user answered correctly
            
        Returns:
            List of learning resources with titles and links/content
            
        Raises:
            ExplanationError: If resource generation fails
        """
        pass


class PerformanceAnalyzer(ABC):
    """
    Abstract interface for analyzing user performance in assessments.
    
    Responsible for tracking and analyzing user performance metrics,
    identifying strengths and weaknesses, and providing personalized
    recommendations for improvement.
    """
    
    @abstractmethod
    async def track_session_performance(self, session_id: str) -> bool:
        """
        Track performance for a completed assessment session.
        
        Args:
            session_id: The unique identifier of the completed session
            
        Returns:
            True if tracking was successful, False otherwise
            
        Raises:
            AnalysisError: If performance tracking fails
            SessionNotFoundError: If session not found
        """
        pass
    
    @abstractmethod
    async def track_answer_performance(
        self,
        user_id: str,
        question_id: str,
        evaluation: Dict[str, Any],
        time_taken_ms: Optional[int] = None
    ) -> bool:
        """
        Track performance for a single answer.
        
        Args:
            user_id: The user's unique identifier
            question_id: The question identifier
            evaluation: Dictionary containing evaluation results
            time_taken_ms: Time taken to answer in milliseconds
            
        Returns:
            True if tracking was successful, False otherwise
            
        Raises:
            AnalysisError: If performance tracking fails
        """
        pass
    
    @abstractmethod
    async def get_user_performance(self, user_id: str) -> Dict[str, Any]:
        """
        Get overall performance metrics for a user.
        
        Args:
            user_id: The user's unique identifier
            
        Returns:
            Dictionary containing overall performance metrics
            
        Raises:
            AnalysisError: If performance analysis fails
            UserNotFoundError: If user not found
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
            topic: The topic to analyze
            
        Returns:
            Dictionary containing topic-specific performance metrics
            
        Raises:
            AnalysisError: If performance analysis fails
            UserNotFoundError: If user not found
        """
        pass
    
    @abstractmethod
    async def get_session_performance(
        self,
        session_id: str
    ) -> Dict[str, Any]:
        """
        Get performance metrics for a specific session.
        
        Args:
            session_id: The session identifier
            
        Returns:
            Dictionary containing session performance metrics
            
        Raises:
            AnalysisError: If performance analysis fails
            SessionNotFoundError: If session not found
        """
        pass
    
    @abstractmethod
    async def get_performance_tracker(self, user_id: str) -> Dict[str, Any]:
        """
        Get the complete performance tracker for a user.
        
        Args:
            user_id: The user's unique identifier
            
        Returns:
            Dictionary containing the complete performance tracking data
            
        Raises:
            AnalysisError: If tracker retrieval fails
            UserNotFoundError: If user not found
        """
        pass
    
    @abstractmethod
    async def get_difficulty_engine(self, user_id: str) -> Dict[str, Any]:
        """
        Get the adaptive difficulty engine configuration for a user.
        
        Args:
            user_id: The user's unique identifier
            
        Returns:
            Dictionary containing difficulty engine parameters
            
        Raises:
            AnalysisError: If engine retrieval fails
            UserNotFoundError: If user not found
        """
        pass
    
    @abstractmethod
    async def get_spaced_repetition_scheduler(self, user_id: str) -> Dict[str, Any]:
        """
        Get the spaced repetition scheduler for a user.
        
        Args:
            user_id: The user's unique identifier
            
        Returns:
            Dictionary containing spaced repetition scheduling data
            
        Raises:
            AnalysisError: If scheduler retrieval fails
            UserNotFoundError: If user not found
        """
        pass
    
    @abstractmethod
    async def get_improvement_recommendations(
        self,
        user_id: str,
        limit: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Get personalized recommendations for improvement.
        
        Args:
            user_id: The user's unique identifier
            limit: Maximum number of recommendations to return
            
        Returns:
            List of improvement recommendations with explanations
            
        Raises:
            AnalysisError: If recommendation generation fails
            UserNotFoundError: If user not found
        """
        pass
    
    @abstractmethod
    async def get_strength_assessment(self, user_id: str) -> Dict[str, Any]:
        """
        Get an assessment of the user's strengths.
        
        Args:
            user_id: The user's unique identifier
            
        Returns:
            Dictionary containing strength assessment data
            
        Raises:
            AnalysisError: If assessment fails
            UserNotFoundError: If user not found
        """
        pass


class AssessmentService(Generic[T_Question, T_Session], ABC):
    """
    Abstract interface for core assessment functionality.
    
    This service coordinates all aspects of assessment operations, including
    session management, question generation, answer evaluation, explanation
    generation, and performance analysis.
    
    Type Parameters:
        T_Question: The type of question managed by this service
        T_Session: The type of session managed by this service
    """
    
    @property
    @abstractmethod
    def question_repository(self) -> QuestionRepository[T_Question]:
        """
        Get the repository for managing questions.
        
        Returns:
            Question repository instance
        """
        pass
    
    @property
    @abstractmethod
    def session_repository(self) -> SessionRepository[T_Session]:
        """
        Get the repository for managing sessions.
        
        Returns:
            Session repository instance
        """
        pass
    
    @property
    @abstractmethod
    def question_generator(self) -> QuestionGenerator[T_Question]:
        """
        Get the generator for creating questions.
        
        Returns:
            Question generator instance
        """
        pass
    
    @property
    @abstractmethod
    def answer_evaluator(self) -> AnswerEvaluator[T_Question, T_Evaluation]:
        """
        Get the evaluator for assessing answers.
        
        Returns:
            Answer evaluator instance
        """
        pass
    
    @property
    @abstractmethod
    def explanation_generator(self) -> ExplanationGenerator[T_Question]:
        """
        Get the generator for creating explanations.
        
        Returns:
            Explanation generator instance
        """
        pass
    
    @property
    @abstractmethod
    def performance_analyzer(self) -> PerformanceAnalyzer:
        """
        Get the analyzer for tracking performance.
        
        Returns:
            Performance analyzer instance
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
    ) -> T_Session:
        """
        Create a new assessment session.
        
        Args:
            user_id: The user's unique identifier
            question_count: Number of questions in the session
            topics: Optional list of topics to include
            difficulty: Optional difficulty level
            settings: Optional session settings
            
        Returns:
            Newly created assessment session
            
        Raises:
            ServiceError: If session creation fails
            ValidationError: If parameters are invalid
        """
        pass
    
    @abstractmethod
    async def get_session(self, session_id: str) -> Optional[T_Session]:
        """
        Retrieve an assessment session by ID.
        
        Args:
            session_id: The session's unique identifier
            
        Returns:
            The session if found, None otherwise
            
        Raises:
            ServiceError: If session retrieval fails
        """
        pass
    
    @abstractmethod
    async def get_question(self, question_id: str) -> Optional[T_Question]:
        """
        Retrieve a question by ID.
        
        Args:
            question_id: The question's unique identifier
            
        Returns:
            The question if found, None otherwise
            
        Raises:
            ServiceError: If question retrieval fails
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
            ServiceError: If answer submission fails
            SessionNotFoundError: If session not found
            QuestionNotFoundError: If question not found
            ValidationError: If parameters are invalid
        """
        pass
    
    @abstractmethod
    async def complete_session(self, session_id: str) -> Dict[str, Any]:
        """
        Complete an assessment session and process results.
        
        Args:
            session_id: The session's unique identifier
            
        Returns:
            Dictionary containing session results and performance metrics
            
        Raises:
            ServiceError: If session completion fails
            SessionNotFoundError: If session not found
            ValidationError: If session is not ready to be completed
        """
        pass
    
    @abstractmethod
    async def get_session_results(self, session_id: str) -> Dict[str, Any]:
        """
        Get results for a completed assessment session.
        
        Args:
            session_id: The session's unique identifier
            
        Returns:
            Dictionary containing session results and performance metrics
            
        Raises:
            ServiceError: If results retrieval fails
            SessionNotFoundError: If session not found
            ValidationError: If session is not completed
        """
        pass
    
    @abstractmethod
    async def get_user_performance(self, user_id: str) -> Dict[str, Any]:
        """
        Get performance metrics for a user across all sessions.
        
        Args:
            user_id: The user's unique identifier
            
        Returns:
            Dictionary containing performance metrics
            
        Raises:
            ServiceError: If performance retrieval fails
            UserNotFoundError: If user not found
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
            ServiceError: If explanation generation fails
            QuestionNotFoundError: If question not found
        """
        pass 