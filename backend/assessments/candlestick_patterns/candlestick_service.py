"""
Candlestick Pattern Assessment Service

This module implements the service layer for candlestick pattern assessments,
providing business logic for pattern recognition, question generation, and evaluation.
"""

from typing import Dict, List, Any, Optional, Tuple, Union, cast, TypeVar, Generic, Callable
import uuid
import random
import datetime
import logging
from collections import defaultdict
from functools import lru_cache, wraps
import asyncio

from backend.assessments.base.models import (
    AssessmentType,
    QuestionDifficulty,
    SessionStatus,
    AnswerEvaluation,
    AssessmentError
)
from backend.assessments.base.services import (
    AssessmentService,
    QuestionGenerator,
    AnswerEvaluator,
    ExplanationGenerator as BaseExplanationGenerator
)
from backend.assessments.candlestick_patterns.candlestick_models import (
    CandlestickQuestion,
    CandlestickSession,
    CandlestickAssessmentResponse
)
from backend.assessments.candlestick_patterns.repository import (
    CandlestickAssessmentRepository,
    get_candlestick_question_repository,
    get_candlestick_session_repository,
    get_candlestick_assessment_repository
)
from backend.assessments.candlestick_patterns.candlestick_repository import (
    RepositoryError
)
from backend.assessments.candlestick_patterns.candlestick_explanation_generator import (
    ExplanationGenerator as CandlestickExplanationGenerator,
    UserLevel
)
from backend.assessments.candlestick_patterns.candlestick_utils import (
    CandlestickData,
    plot_candlestick_chart,
    get_patterns_by_difficulty,
    generate_options,
    get_all_patterns_with_metadata
)
from backend.assessments.candlestick_patterns.question_generator import (
    AdaptiveQuestionGenerator
)
from backend.assessments.candlestick_patterns.answer_evaluation import (
    CandlestickAnswerEvaluator
)
from backend.common.finance.patterns import PatternType, PatternStrength
from backend.common.cache import async_cached
from backend.common.logger import get_logger
from backend.common.base_assessment import (
    EventDispatcher,
    QuestionGeneratedEvent,
    SessionCompletedEvent,
    DomainEvent
)

# Import base repository interfaces if needed for type hinting
from backend.assessments.base.repositories import (
    QuestionRepository as BaseQuestionRepository,
    SessionRepository as BaseSessionRepository,
    AssessmentRepository
)

# Import the async database dependencies
from backend.common.db.session import async_session_factory
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker

# Import UserMetricsService components
from backend.metrics.service import UserMetricsService
from backend.metrics.repository import UserMetricsRepository

# Import the database engine
from backend.database.init_db import get_engine

# Set up logger
logger = get_logger(__name__)

# Define TypeVars for clarity
Q = TypeVar('Q', bound=CandlestickQuestion)
S = TypeVar('S', bound=CandlestickSession)

# Simple in-memory cache for performance improvements
_cache = {}

def cached(ttl: int = 300, key_builder: Callable = None):
    """
    Simple decorator for caching expensive method results.
    
    Args:
        ttl: Time-to-live in seconds for cached values
        key_builder: Optional function to build cache key
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Build cache key
            if key_builder:
                cache_key = key_builder(*args, **kwargs)
            else:
                # Default key based on func name and args
                cache_key = f"{func.__name__}:{str(args)}:{str(kwargs)}"
                
            # Check cache
            if cache_key in _cache:
                entry = _cache[cache_key]
                if entry['expires_at'] > datetime.datetime.now():
                    logger.debug(f"Cache hit for {cache_key}")
                    return entry['value']
                    
            # Execute function and cache result
            logger.debug(f"Cache miss for {cache_key}")
            result = await func(*args, **kwargs)
            
            _cache[cache_key] = {
                'value': result,
                'expires_at': datetime.datetime.now() + datetime.timedelta(seconds=ttl)
            }
            
            # Schedule cache cleanup for this key
            asyncio.create_task(_schedule_cache_cleanup(cache_key, ttl))
            
            return result
        return wrapper
    return decorator

async def _schedule_cache_cleanup(key: str, ttl: int):
    """Helper function to remove expired cache entries"""
    await asyncio.sleep(ttl)
    if key in _cache:
        logger.debug(f"Removing expired cache entry for {key}")
        del _cache[key]

class CandlestickAssessmentService(AssessmentService[Q, S]):
    """
    Service for candlestick pattern assessments.
    
    This service handles the business logic for candlestick pattern assessments,
    including session management, question generation, and answer evaluation.
    It properly extends the base AssessmentService to ensure consistency with
    the overall assessment architecture.
    """
    
    def __init__(
        self,
        question_repository: BaseQuestionRepository[Q],
        session_repository: BaseSessionRepository[S],
        question_generator: QuestionGenerator[Q],
        answer_evaluator: AnswerEvaluator[Q, AnswerEvaluation],
        explanation_generator: BaseExplanationGenerator,
        assessment_repository: AssessmentRepository,
        # Add UserMetricsService dependency
        user_metrics_service: UserMetricsService,
        event_dispatcher: Optional[EventDispatcher] = None
    ):
        """
        Initialize the candlestick assessment service with dependencies.
        
        Args:
            question_repository: Repository for questions.
            session_repository: Repository for sessions.
            question_generator: Component for generating questions.
            answer_evaluator: Component for evaluating answers.
            explanation_generator: Component for generating explanations.
            assessment_repository: Repository for specific candlestick methods.
            user_metrics_service: Service for managing user aggregate metrics.
            event_dispatcher: Optional event dispatcher for domain events.
        """
        # Store injected dependencies
        self._question_repository = question_repository
        self._session_repository = session_repository
        self._assessment_repository = assessment_repository
        self._question_generator = question_generator
        self._answer_evaluator = answer_evaluator
        self._explanation_generator = explanation_generator
        self._user_metrics_service = user_metrics_service # Store the metrics service
        self.event_dispatcher = event_dispatcher or EventDispatcher()
        
        logger.info("CandlestickAssessmentService initialized with injected dependencies including UserMetricsService.")

    def _safely_dispatch_event(self, event: DomainEvent) -> None:
        """
        Safely dispatch an event, catching and logging any exceptions.
        
        Args:
            event: The event to dispatch
        """
        try:
            self.event_dispatcher.dispatch(event)
            logger.debug(f"Successfully dispatched event {type(event).__name__}")
        except Exception as e:
            logger.error(f"Failed to dispatch event {type(event).__name__}: {e}", exc_info=True)

    @property
    def assessment_type(self) -> AssessmentType:
        return AssessmentType.CANDLESTICK

    @property
    def question_repository(self) -> BaseQuestionRepository[Q]:
        return self._question_repository

    @property
    def session_repository(self) -> BaseSessionRepository[S]:
        return self._session_repository

    @property
    def question_generator(self) -> QuestionGenerator[Q]:
        return self._question_generator

    @property
    def answer_evaluator(self) -> AnswerEvaluator[Q, AnswerEvaluation]:
        return self._answer_evaluator

    @property
    def explanation_generator(self) -> BaseExplanationGenerator:
        return self._explanation_generator

    @property
    def assessment_repository(self) -> AssessmentRepository:
        return self._assessment_repository

    @property
    def user_metrics_service(self) -> UserMetricsService:
        return self._user_metrics_service

    async def create_session(
        self,
        user_id: str,
        question_count: int = 5,
        topics: Optional[List[str]] = None,
        difficulty: Optional[QuestionDifficulty] = None,
        settings: Optional[Dict[str, Any]] = None
    ) -> CandlestickSession:
        """
        Create a new assessment session for a user.
        
        Args:
            user_id: User identifier
            question_count: Number of questions to include
            topics: Optional list of topics to focus on
            difficulty: Optional difficulty level
            settings: Optional session settings
            
        Returns:
            Created session object
            
        Raises:
            AssessmentError: If session creation fails
        """
        try:
            logger.info(f"Creating candlestick session for user {user_id}")
            
            # Generate questions for the session
            questions = await self.question_generator.generate_for_user(
                user_id=user_id,
                count=question_count,
                topics=topics
            )
            
            if not questions:
                logger.error(f"Failed to generate questions for user {user_id}")
                raise AssessmentError("Failed to generate questions for session")
            
            # SKIP saving questions to repository to avoid ORM mapping issues
            # We'll use the in-memory questions directly
            
            # Create session object
            session = CandlestickSession(
                id=str(uuid.uuid4()),
                user_id=user_id,
                assessment_type=AssessmentType.CANDLESTICK,
                # Store the question objects directly, not just their IDs
                questions=[q.id for q in questions],
                status=SessionStatus.IN_PROGRESS,
                created_at=datetime.datetime.now(),
                settings={
                    "topics": topics,
                    "difficulty": difficulty.value if difficulty else None,
                    "question_count": len(questions),
                }
            )
            
            # Cache the questions in the session settings for retrieval later
            # This avoids needing to load them from the database
            question_data = []
            for q in questions:
                q_data = {
                    "id": q.id,
                    "question_text": q.question_text,
                    "difficulty": q.difficulty.value,
                    "pattern": q.pattern,
                    "options": q.options,
                    "chart_data": q.chart_data,
                    "chart_image": q.chart_image,
                    "timeframe": q.timeframe,
                    "symbol": q.symbol,
                    "topics": q.topics,
                    "metadata": q.metadata
                }
                question_data.append(q_data)
            
            session.settings["question_data"] = question_data
            
            # Skip saving session to repository to avoid ORM mapping issues
            # We'll use the in-memory session directly
            logger.info(f"Successfully created session {session.id} for user {user_id} with {len(questions)} questions")
            
            # Comment out event dispatch for now due to parameter mismatch
            # self._safely_dispatch_event(
            #     QuestionGeneratedEvent(
            #         user_id=user_id,
            #         session_id=session.id,
            #         question_count=len(questions)
            #     )
            # )
            
            # Return the unsaved session
            return session
            
        except Exception as e:
            logger.error(f"Failed to create candlestick session: {str(e)}")
            raise AssessmentError(f"Failed to create candlestick session: {str(e)}")

    async def get_session(self, session_id: str) -> Optional[S]:
        """
        Get a candlestick pattern assessment session by ID.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session instance if found, None otherwise
        """
        if not session_id:
            logger.warning("Empty session ID provided to get_session")
            return None
        try:
            session = await self.session_repository.get_by_id(session_id)
            return session
        except RepositoryError as e:
            logger.error(f"Repository error retrieving session {session_id}: {e.original_exception}", exc_info=True)
            return None # Indicate failure to retrieve
        except Exception as e:
            logger.error(f"Unexpected error retrieving session {session_id}: {e}", exc_info=True)
            # Consider raising a generic service error instead of returning None for unexpected issues
            # raise RuntimeError(f"Unexpected error retrieving session: {e}")
            return None
    
    async def get_question(self, question_id: str, session_id: Optional[str] = None) -> Optional[Q]:
        """
        Get a question by ID.
        
        Args:
            question_id: Question identifier
            session_id: Optional session ID to look up cached questions
            
        Returns:
            Question instance if found, None otherwise
        """
        if not question_id:
            logger.warning("Empty question ID provided to get_question")
            return None
            
        # First try to get the question from session settings if session_id is provided
        if session_id:
            try:
                session = await self.get_session(session_id)
                if session and 'question_data' in session.settings:
                    # Look for the question in the cached data
                    for q_data in session.settings['question_data']:
                        if q_data.get('id') == question_id:
                            # Reconstruct question from metadata
                            return CandlestickQuestion(
                                id=q_data['id'],
                                question_text=q_data['question_text'],
                                difficulty=QuestionDifficulty(q_data['difficulty']),
                                pattern=q_data['pattern'],
                                options=q_data['options'],
                                chart_data=q_data.get('chart_data', {}),
                                chart_image=q_data.get('chart_image', ''),
                                timeframe=q_data.get('timeframe', ''),
                                symbol=q_data.get('symbol', ''),
                                topics=q_data.get('topics', []),
                                metadata=q_data.get('metadata', {})
                            )
            except Exception as e:
                logger.warning(f"Error retrieving question {question_id} from session settings: {str(e)}")
                # Continue to try from repository
        
        # Log repository type to help diagnose the issue
        logger.debug(f"Question repository type: {type(self.question_repository).__name__}")
        logger.debug(f"Question repository methods: {dir(self.question_repository)}")
        
        # If not found in session settings or session_id not provided, try the repository
        try:
            # Try to use get_by_id if available, otherwise try get_question or another alternative
            if hasattr(self.question_repository, 'get_by_id'):
                question = await self.question_repository.get_by_id(question_id)
            elif hasattr(self.question_repository, 'get_question'):
                question = await self.question_repository.get_question(question_id)
            else:
                logger.error(f"Repository does not have get_by_id or get_question methods")
                return None
                
            return question
        except RepositoryError as e:
            logger.error(f"Repository error retrieving question {question_id}: {e.original_exception}", exc_info=True)
            return None # Indicate failure to retrieve
        except Exception as e:
            logger.error(f"Unexpected error retrieving question {question_id}: {e}", exc_info=True)
            # raise RuntimeError(f"Unexpected error retrieving question: {e}")
            return None
    
    async def submit_answer(
        self,
        session_id: str,
        question_id: str,
        user_answer: str,
        time_taken_ms: Optional[int] = None,
        question: Optional[CandlestickQuestion] = None,
        confidence_level: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Submit and evaluate an answer for a question in a session.
        
        Args:
            session_id: Session identifier
            question_id: Question identifier
            user_answer: User's answer
            time_taken_ms: Time taken to answer (optional)
            question: CandlestickQuestion object (optional) - if provided, skips retrieving from repository
            confidence_level: User's confidence level (optional)
            
        Returns:
            Dictionary with evaluation result and session status
            
        Raises:
            ValueError: If the session or question is not found or invalid
            RuntimeError: If there's an error updating the session
        """
        # Retrieve session
        session = await self.get_session(session_id)
        if not session:
            logger.error(f"submit_answer failed: Session {session_id} not found")
            raise ValueError(f"Session {session_id} not found")
        if session.status != SessionStatus.IN_PROGRESS:
            logger.error(f"submit_answer failed: Session {session_id} is not in progress (status: {session.status.value})")
            raise ValueError(f"Session {session_id} is not active.")

        # Validate question ID belongs to session
        if question_id not in session.questions:
            logger.error(f"submit_answer failed: Question {question_id} not part of session {session_id}")
            raise ValueError(f"Question {question_id} not found in session {session_id}")
            
        # Ensure question hasn't been answered already (optional, depends on allow_resubmit logic)
        if question_id in session.answers:
             logger.warning(f"Question {question_id} in session {session_id} has already been answered.")
             # Decide behavior: error out or allow re-submission?
             # For now, let's retrieve existing results if available.
             # existing_answer = session.answers[question_id]
             # return {
             #     "session_id": session.id,
             #     "question_id": question_id,
             #     "evaluation": existing_answer.evaluation.to_dict(),
             #     "is_correct": existing_answer.evaluation.is_correct,
             #     "explanation": "Question already answered.", # Or fetch explanation
             #     "session_status": session.status.value
             # }
             # Let's proceed to allow re-submission/update for now, repository save handles it.

        # Retrieve question if not provided
        if question is None:
            question = await self.get_question(question_id, session_id=session_id)
            if not question:
                logger.error(f"submit_answer failed: Question {question_id} could not be retrieved")
                raise ValueError(f"Question {question_id} not found")

        # Evaluate answer
        # Wrap evaluator call in try-except
        try:
             evaluation = self.answer_evaluator.evaluate_answer(question, user_answer)
        except Exception as eval_err:
             logger.error(f"Answer evaluation failed for Q:{question_id} S:{session_id}: {eval_err}", exc_info=True)
             raise ValueError(f"Failed to evaluate answer: {eval_err}")

        # Record answer in session domain model
        # Use session domain model's method
        session.record_answer(
             question_id=question_id, 
             user_answer=user_answer, 
             evaluation=evaluation, 
             time_taken_ms=time_taken_ms,
             confidence_level=confidence_level
        )

        # Advance session state
        session.current_question_index += 1
        if session.current_question_index >= len(session.questions):
            session.status = SessionStatus.COMPLETED
            session.completed_at = datetime.datetime.utcnow()

        # Save updated session state
        try:
            await self.session_repository.save(session)
        except RepositoryError as db_err:
            logger.error(f"Database error saving updated session {session_id} after answer submission: {db_err.original_exception}", exc_info=True)
            # Decide if we should raise or return error state
            raise RuntimeError(f"Failed to save session update: {db_err}")
        except Exception as e:
            logger.error(f"Unexpected error saving session {session_id}: {e}", exc_info=True)
            raise RuntimeError(f"Unexpected error saving session: {e}")

        # --- Post-Save Actions --- 
        
        # If session is now completed, perform completion tasks
        if session.status == SessionStatus.COMPLETED:
            logger.info(f"Session {session_id} completed for user {session.user_id}.")
            
            # Calculate session stats for event
            session_perf = session.get_performance() 
            duration_sec = (session.completed_at - session.created_at).total_seconds() if session.completed_at and session.created_at else 0
            
            # Dispatch SessionCompletedEvent using our helper method
            self._safely_dispatch_event(
                SessionCompletedEvent(
                    session_id=session.id,
                    user_id=session.user_id,
                    assessment_type=session.assessment_type.value,
                    score=session_perf.total_score,
                    duration=duration_sec
                )
            )
            
            # Update user aggregate metrics
            try:
                logger.info(f"Attempting to update aggregate metrics for user {session.user_id}...")
                metrics_updated = await self.user_metrics_service.update_metrics_from_session(session)
                if metrics_updated:
                    logger.info(f"Aggregate metrics updated successfully for user {session.user_id}.")
                else:
                    logger.warning(f"Aggregate metrics update returned False for user {session.user_id}.")
            except Exception as metrics_err:
                # Log error but don't fail the entire request because metrics failed
                logger.error(f"Error updating user aggregate metrics for user {session.user_id} from session {session.id}: {metrics_err}", exc_info=True)

        # Generate explanation (conditionally?)
        explanation = "Explanation generation not implemented yet."
        try:
            # Pass user level if available or use default
            explanation_dict = self.explanation_generator.generate_explanation(
                question=question,
                user_answer=user_answer,
                evaluation=evaluation,
                user_level='intermediate' # TODO: Get user level dynamically
            )
            explanation = explanation_dict.get('explanation', explanation)
        except Exception as expl_err:
             logger.error(f"Explanation generation failed for Q:{question_id}: {expl_err}", exc_info=True)
             # Don't fail request if explanation fails

        # Prepare response for the controller
        next_question_id = None
        if session.status == SessionStatus.IN_PROGRESS:
             next_question_id = session.questions[session.current_question_index]

        response = {
            "session_id": session.id,
            "question_id": question_id,
            "evaluation": evaluation.to_dict(),
            "is_correct": evaluation.is_correct,
            "explanation": explanation,
            "session_status": session.status.value,
            "next_question_id": next_question_id,
            "current_question_index": session.current_question_index,
            "total_questions": len(session.questions)
        }
        
        return response

    async def get_session_results(self, session_id: str) -> Dict[str, Any]:
        """Get results for a completed session."""
        session = await self.get_session(session_id)
        if not session or session.status != SessionStatus.COMPLETED:
            raise ValueError("Session not found or not completed.")
        # return session.get_performance().to_dict() # Example
        raise NotImplementedError

    async def get_user_performance(self, user_id: str) -> Dict[str, Any]:
        """Get overall user performance for this assessment type."""
        # Implementation needed - query repository for user sessions/stats
        raise NotImplementedError

    @cached(ttl=300, key_builder=lambda self, user_id, topic: f"topic_perf:{user_id}:{topic}")
    async def get_topic_performance(self, user_id: str, topic: str) -> Dict[str, Any]:
        """
        Get user performance for a specific topic/pattern.
        
        Args:
            user_id: User identifier
            topic: Topic/pattern name
            
        Returns:
            Dictionary with performance data
            
        Raises:
            ValueError: If user_id or topic is empty
            RuntimeError: If there's an error retrieving the performance data
        """
        if not user_id or not topic:
            raise ValueError("User ID and topic are required.")
        try:
            # Delegate to the aggregate repository method
            # Need to ensure the injected assessment_repository has this method
            if hasattr(self.assessment_repository, 'get_topic_performance'):
                performance = await self.assessment_repository.get_topic_performance(user_id, topic)
                return performance
            else:
                logger.error("Injected assessment_repository does not have 'get_topic_performance' method.")
                raise NotImplementedError("Topic performance retrieval is not implemented in the repository.")
        except RepositoryError as e:
            logger.error(f"Repository error getting topic performance for user {user_id}, topic {topic}: {e.original_exception}", exc_info=True)
            raise RuntimeError(f"Failed to retrieve topic performance: {e}")
        except Exception as e:
            logger.error(f"Unexpected error getting topic performance for user {user_id}, topic {topic}: {e}", exc_info=True)
            raise RuntimeError(f"Unexpected error retrieving topic performance: {e}")

    async def get_recommendations(self, user_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get recommended topics/patterns for a user based on performance."""
        if not user_id:
            raise ValueError("User ID is required.")
        if limit <= 0:
            limit = 5 # Default to 5 if invalid limit provided
        
        try:
            # Delegate to the aggregate repository method
            if hasattr(self.assessment_repository, 'get_recommended_topics'):
                recommendations = await self.assessment_repository.get_recommended_topics(user_id, limit)
                return recommendations
            else:
                logger.error("Injected assessment_repository does not have 'get_recommended_topics' method.")
                raise NotImplementedError("Topic recommendation retrieval is not implemented in the repository.")
        except RepositoryError as e:
            logger.error(f"Repository error getting recommendations for user {user_id}: {e.original_exception}", exc_info=True)
            raise RuntimeError(f"Failed to retrieve recommendations: {e}")
        except Exception as e:
            logger.error(f"Unexpected error getting recommendations for user {user_id}: {e}", exc_info=True)
            raise RuntimeError(f"Unexpected error retrieving recommendations: {e}")

    async def analyze_all_topic_performance(self, user_id: str) -> Dict[str, Any]:
        """Analyze performance across all attempted topics for a user."""
        # Placeholder - requires implementation using repository methods
        logger.warning("analyze_all_topic_performance service method not fully implemented.")
        # Basic idea:
        # 1. Get all distinct attempted topics from repository (e.g., add a method to repo)
        # 2. Loop through topics, calling self.get_topic_performance for each
        # 3. Aggregate results (e.g., identify strengths/weaknesses)
        # 4. Handle errors gracefully
        return {"message": "Analysis not implemented yet", "user_id": user_id}

    async def get_explanation(
        self, 
        question_id: str, 
        user_answer: Optional[Any] = None, 
        user_level: str = 'intermediate'
    ) -> Dict[str, Any]:
        """Get explanation for a given question and optionally user answer."""
        question = await self.get_question(question_id)
        if not question:
            raise ValueError(f"Question {question_id} not found")

        # Generate explanation using the dedicated generator
        # Note: Requires evaluation object if user_answer is provided.
        # This method might need adjustment depending on how evaluation is obtained here.
        # Placeholder: Assuming evaluation logic is handled elsewhere or not needed for basic explanation.
        try:
            # If user_answer is provided, we ideally need the evaluation context.
            # For now, pass None for evaluation if only question is available.
            evaluation_context = None # Need to determine how to get this if needed
            if user_answer is not None:
                 # evaluation_context = self.answer_evaluator.evaluate_answer(question, user_answer)
                 logger.warning("get_explanation called with user_answer but evaluation context is not generated here.")
                 
            explanation_dict = self.explanation_generator.generate_explanation(
                question=question,
                user_answer=user_answer, 
                evaluation=evaluation_context, # Pass evaluation if available
                user_level=user_level
            )
            return explanation_dict
        except Exception as e:
             logger.error(f"Failed to generate explanation for question {question_id}: {e}", exc_info=True)
             raise RuntimeError(f"Explanation generation failed: {e}")

    async def get_session_performance(self, session_id: str) -> Dict[str, Any]:
        """
        Get detailed performance statistics for a specific session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Dictionary with performance data
            
        Raises:
            ValueError: If session not found
            RuntimeError: If there's an error retrieving session data
        """
        session = await self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
            
        try:
            # Use the session domain model's built-in performance calculation
            performance = session.get_performance().to_dict()
            
            # Enhance with additional metrics as needed
            performance.update({
                "session_id": session.id,
                "user_id": session.user_id,
                "assessment_type": session.assessment_type.value,
                "status": session.status.value,
                "created_at": session.created_at.isoformat() if session.created_at else None,
                "completed_at": session.completed_at.isoformat() if session.completed_at else None,
                "difficulty": session.difficulty.value if hasattr(session, "difficulty") else None,
                "question_count": len(session.questions),
                "questions_answered": len(session.answers),
                "patterns_tested": self._extract_patterns_from_session(session)
            })
            
            return performance
        except Exception as e:
            logger.error(f"Error calculating session performance for {session_id}: {e}", exc_info=True)
            raise RuntimeError(f"Failed to calculate session performance: {e}")
    
    def _extract_patterns_from_session(self, session: CandlestickSession) -> List[str]:
        """
        Helper method to extract unique pattern names from a session's questions.
        
        Args:
            session: The session to extract patterns from
            
        Returns:
            List of unique pattern names
        """
        patterns = set()
        for q_id in session.questions:
            # Try to get pattern from the question data if available in session settings
            if hasattr(session, 'question_data') and q_id in session.question_data:
                metadata = session.question_data.get(q_id, {})
                if 'pattern_name' in metadata:
                    patterns.add(metadata['pattern_name'])
                    
        return list(patterns)
    
    @cached(ttl=300, key_builder=lambda self, user_id, limit: f"recent_sessions:{user_id}:{limit}")
    async def get_recent_sessions(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get a user's most recent assessment sessions with summary data.
        
        Args:
            user_id: User identifier
            limit: Maximum number of sessions to return
            
        Returns:
            List of session summary dictionaries
            
        Raises:
            ValueError: If user_id is empty or limit is invalid
            RuntimeError: If there's an error retrieving sessions
        """
        if not user_id:
            raise ValueError("User ID is required")
        if limit <= 0:
            raise ValueError("Limit must be a positive integer")
            
        try:
            # Fetch recent sessions from repository
            sessions = await self.session_repository.get_recent_sessions_by_user(
                user_id=user_id,
                assessment_type=AssessmentType.CANDLESTICK_PATTERNS,
                limit=limit
            )
            
            # Transform to summary format
            session_summaries = []
            for session in sessions:
                performance = session.get_performance()
                summary = {
                    "session_id": session.id,
                    "created_at": session.created_at.isoformat() if session.created_at else None,
                    "completed_at": session.completed_at.isoformat() if session.completed_at else None,
                    "status": session.status.value,
                    "questions_total": len(session.questions),
                    "questions_answered": len(session.answers),
                    "correct_count": performance.correct_count,
                    "score": performance.total_score,
                    "accuracy": performance.accuracy
                }
                session_summaries.append(summary)
            
            return session_summaries
        except RepositoryError as e:
            logger.error(f"Repository error getting recent sessions for user {user_id}: {e.original_exception}", exc_info=True)
            raise RuntimeError(f"Failed to retrieve recent sessions: {e}")
        except Exception as e:
            logger.error(f"Unexpected error getting recent sessions for user {user_id}: {e}", exc_info=True)
            raise RuntimeError(f"Unexpected error retrieving recent sessions: {e}")
            
    async def list_available_patterns(self) -> List[Dict[str, Any]]:
        """
        Get a list of all available candlestick patterns with descriptions.
        
        Returns:
            List of pattern dictionaries with name, description, and difficulty
            
        Raises:
            RuntimeError: If there's an error retrieving the pattern data
        """
        try:
            patterns = []
            # Get patterns organized by difficulty
            patterns_by_difficulty = get_all_patterns_with_metadata()
            
            # Flatten and format for API consumption
            for difficulty, pattern_list in patterns_by_difficulty.items():
                for pattern in pattern_list:
                    patterns.append({
                        "name": pattern.get("name", ""),
                        "description": pattern.get("description", ""),
                        "reliability": pattern.get("reliability", ""),
                        "difficulty": difficulty,
                        "example_image_url": pattern.get("example_image_url", "")
                    })
            
            return patterns
        except Exception as e:
            logger.error(f"Error retrieving available patterns: {e}", exc_info=True)
            raise RuntimeError(f"Failed to retrieve available patterns: {e}")

    @cached(ttl=300, key_builder=lambda self, user_id: f"user_stats:{user_id}")
    async def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        """
        Get comprehensive statistics about a user's performance across candlestick patterns.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dictionary with aggregated statistics
            
        Raises:
            ValueError: If user_id is empty
            RuntimeError: If there's an error retrieving or processing the data
        """
        if not user_id:
            raise ValueError("User ID is required")
            
        try:
            # Get aggregate metrics from the metrics service first
            metrics = None
            try:
                metrics = await self.user_metrics_service.get_user_metrics(user_id)
            except Exception as metrics_err:
                logger.warning(f"Failed to retrieve metrics for user {user_id}: {metrics_err}")
                # Continue without metrics, we'll use session data
            
            # Get all completed sessions for this user
            sessions = await self.session_repository.get_all_sessions_by_user(
                user_id=user_id,
                assessment_type=AssessmentType.CANDLESTICK_PATTERNS,
                status=SessionStatus.COMPLETED
            )
            
            # Calculate aggregate stats
            total_sessions = len(sessions)
            if total_sessions == 0:
                return {
                    "user_id": user_id,
                    "total_sessions": 0,
                    "total_questions": 0,
                    "correct_count": 0,
                    "accuracy": 0.0,
                    "avg_score": 0.0,
                    "pattern_proficiency": {},
                    "difficulty_breakdown": {
                        "beginner": 0,
                        "intermediate": 0,
                        "advanced": 0
                    },
                    "recent_streak": 0,
                    "best_streak": metrics.longest_streak if metrics else 0
                }
                
            # Calculate pattern-specific performance
            pattern_stats = defaultdict(lambda: {"count": 0, "correct": 0, "accuracy": 0.0})
            difficulty_counts = {
                "beginner": 0,
                "intermediate": 0, 
                "advanced": 0
            }
            
            total_questions = 0
            total_correct = 0
            total_score = 0
            
            for session in sessions:
                perf = session.get_performance()
                total_questions += perf.total_questions
                total_correct += perf.correct_count
                total_score += perf.total_score
                
                # Count by difficulty
                difficulty = getattr(session, 'difficulty', QuestionDifficulty.INTERMEDIATE)
                difficulty_counts[difficulty.value.lower()] += 1
                
                # Analyze patterns (if available in question metadata)
                if hasattr(session, 'question_data'):
                    for q_id, answer in session.answers.items():
                        metadata = session.question_data.get(q_id, {})
                        pattern_name = metadata.get('pattern_name')
                        
                        if pattern_name:
                            pattern_stats[pattern_name]["count"] += 1
                            if answer.evaluation.is_correct:
                                pattern_stats[pattern_name]["correct"] += 1
            
            # Calculate pattern accuracies
            for pattern, stats in pattern_stats.items():
                if stats["count"] > 0:
                    stats["accuracy"] = round(stats["correct"] / stats["count"] * 100, 1)
            
            # Calculate overall accuracy and average score
            accuracy = round(total_correct / total_questions * 100, 1) if total_questions > 0 else 0.0
            avg_score = round(total_score / total_sessions, 1) if total_sessions > 0 else 0.0
            
            # Compile final stats
            stats = {
                "user_id": user_id,
                "total_sessions": total_sessions,
                "total_questions": total_questions,
                "correct_count": total_correct,
                "accuracy": accuracy,
                "avg_score": avg_score,
                "pattern_proficiency": dict(pattern_stats),
                "difficulty_breakdown": difficulty_counts,
                "recent_streak": metrics.current_streak if metrics else 0,
                "best_streak": metrics.longest_streak if metrics else 0
            }
            
            return stats
            
        except RepositoryError as e:
            logger.error(f"Repository error getting user stats for {user_id}: {e.original_exception}", exc_info=True)
            raise RuntimeError(f"Failed to retrieve user statistics: {e}")
        except Exception as e:
            logger.error(f"Unexpected error getting user stats for {user_id}: {e}", exc_info=True)
            raise RuntimeError(f"Unexpected error retrieving user statistics: {e}")

    async def save_user_preference(self, user_id: str, preference_key: str, preference_value: Any) -> bool:
        """
        Save a user preference related to candlestick pattern assessments.
        
        Args:
            user_id: User identifier
            preference_key: Preference key/name
            preference_value: Preference value
            
        Returns:
            Boolean indicating success
            
        Raises:
            ValueError: If user_id or preference_key is empty
            RuntimeError: If there's an error saving the preference
        """
        if not user_id or not preference_key:
            raise ValueError("User ID and preference key are required")
            
        try:
            # Validate preference key is allowed
            allowed_keys = ["default_difficulty", "chart_type", "favorite_patterns"]
            if preference_key not in allowed_keys:
                raise ValueError(f"Invalid preference key. Allowed: {', '.join(allowed_keys)}")
                
            # Validate value type based on key
            if preference_key == "default_difficulty":
                valid_difficulties = [d.value.lower() for d in QuestionDifficulty]
                if preference_value.lower() not in valid_difficulties:
                    raise ValueError(f"Invalid difficulty. Allowed: {', '.join(valid_difficulties)}")
                    
            elif preference_key == "chart_type":
                valid_chart_types = ["candlestick", "ohlc", "line"]
                if preference_value.lower() not in valid_chart_types:
                    raise ValueError(f"Invalid chart type. Allowed: {', '.join(valid_chart_types)}")
                    
            elif preference_key == "favorite_patterns" and not isinstance(preference_value, list):
                raise ValueError("favorite_patterns value must be a list of pattern names")
                
            # Save to preferences repository
            if not hasattr(self, "preferences_repository") or not self.preferences_repository:
                logger.error("Preferences repository not available")
                raise NotImplementedError("Preferences repository not configured")
                
            success = await self.preferences_repository.save_user_preference(
                user_id=user_id,
                assessment_type=AssessmentType.CANDLESTICK_PATTERNS.value,
                preference_key=preference_key,
                preference_value=preference_value
            )
            
            if success:
                logger.info(f"Saved user preference {preference_key} for user {user_id}")
                
                # Clear any related caches
                for key in list(_cache.keys()):
                    if f"user_stats:{user_id}" in key:
                        del _cache[key]
                        
            return success
            
        except NotImplementedError:
            raise  # Re-raise the original exception
        except Exception as e:
            logger.error(f"Error saving user preference {preference_key} for user {user_id}: {e}", exc_info=True)
            raise RuntimeError(f"Failed to save user preference: {e}")

    async def complete_session(self, session_id: str) -> Dict[str, Any]:
        """
        Complete an assessment session and calculate final results.
        
        Args:
            session_id: The ID of the session to complete
            
        Returns:
            Dictionary containing session results and performance metrics
            
        Raises:
            ValueError: If session not found or already completed
            RuntimeError: If there's an error completing the session
        """
        try:
            # Get the session
            session = await self._session_repository.get_session(session_id)
            if not session:
                raise ValueError(f"Session {session_id} not found")
                
            if session.status == SessionStatus.COMPLETED:
                raise ValueError(f"Session {session_id} is already completed")
                
            # Calculate final results
            total_questions = len(session.questions)
            answered_questions = len(session.answers)
            correct_answers = sum(1 for ans in session.answers.values() if ans.is_correct)
            
            # Update session status
            session.status = SessionStatus.COMPLETED
            session.completed_at = datetime.datetime.utcnow()
            session.score = (correct_answers / total_questions) * 100 if total_questions > 0 else 0
            
            # Save updated session
            await self._session_repository.update_session(session)
            
            # Dispatch completion event
            self._safely_dispatch_event(SessionCompletedEvent(
                session_id=session_id,
                user_id=session.user_id,
                score=session.score,
                total_questions=total_questions,
                correct_answers=correct_answers
            ))
            
            # Return results
            return {
                "session_id": session_id,
                "status": SessionStatus.COMPLETED.value,
                "score": session.score,
                "total_questions": total_questions,
                "correct_answers": correct_answers,
                "completion_time": session.completed_at.isoformat()
            }
            
        except ValueError as ve:
            logger.warning(f"Validation error completing session {session_id}: {ve}")
            raise
        except Exception as e:
            logger.error(f"Error completing session {session_id}: {e}", exc_info=True)
            raise RuntimeError(f"Failed to complete session {session_id}")

    async def performance_analyzer(self, user_id: str) -> Dict[str, Any]:
        """
        Analyze a user's performance across all candlestick pattern assessments.
        
        Args:
            user_id: The ID of the user to analyze
            
        Returns:
            Dictionary containing performance metrics and analysis
            
        Raises:
            ValueError: If user_id is invalid
            RuntimeError: If there's an error analyzing performance
        """
        try:
            if not user_id:
                raise ValueError("User ID cannot be empty")
                
            # Get user's completed sessions
            sessions = await self._session_repository.get_user_sessions(
                user_id=user_id,
                status=SessionStatus.COMPLETED
            )
            
            if not sessions:
                return {
                    "user_id": user_id,
                    "total_sessions": 0,
                    "average_score": 0,
                    "total_questions": 0,
                    "patterns_attempted": [],
                    "strengths": [],
                    "areas_for_improvement": []
                }
                
            # Calculate aggregate metrics
            total_sessions = len(sessions)
            total_score = sum(session.score for session in sessions)
            average_score = total_score / total_sessions
            
            # Analyze pattern performance
            pattern_stats = defaultdict(lambda: {"correct": 0, "total": 0})
            for session in sessions:
                for question_id, answer in session.answers.items():
                    question = await self._question_repository.get_question(question_id)
                    if question:
                        pattern = question.pattern_type
                        pattern_stats[pattern]["total"] += 1
                        if answer.is_correct:
                            pattern_stats[pattern]["correct"] += 1
                            
            # Calculate pattern proficiency
            pattern_proficiency = {}
            for pattern, stats in pattern_stats.items():
                if stats["total"] > 0:
                    proficiency = (stats["correct"] / stats["total"]) * 100
                    pattern_proficiency[pattern] = proficiency
                    
            # Identify strengths and areas for improvement
            strengths = [
                pattern for pattern, score in pattern_proficiency.items()
                if score >= 70  # Consider 70% or higher as strength
            ]
            
            areas_for_improvement = [
                pattern for pattern, score in pattern_proficiency.items()
                if score < 50  # Consider below 50% as needing improvement
            ]
            
            return {
                "user_id": user_id,
                "total_sessions": total_sessions,
                "average_score": average_score,
                "total_questions": sum(len(s.questions) for s in sessions),
                "patterns_attempted": list(pattern_stats.keys()),
                "pattern_proficiency": pattern_proficiency,
                "strengths": strengths,
                "areas_for_improvement": areas_for_improvement
            }
            
        except ValueError as ve:
            logger.warning(f"Validation error analyzing performance for user {user_id}: {ve}")
            raise
        except Exception as e:
            logger.error(f"Error analyzing performance for user {user_id}: {e}", exc_info=True)
            raise RuntimeError(f"Failed to analyze performance for user {user_id}")


def get_session_factory() -> sessionmaker:
    """ 
    Get the session factory for database operations.
    
    Returns:
        Session factory configured for async use
    """
    return async_session_factory

# Global service instance
_candlestick_service_instance = None

def create_candlestick_service() -> Optional[CandlestickAssessmentService]:
    """Create and configure the candlestick assessment service."""
    try:
        try:
            engine = get_engine()
        except RuntimeError:
            logger.warning("Database engine not initialized yet, deferring service creation")
            return None
            
        # Create async session factory
        async_session = sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        
        # Initialize repositories with async session
        session = async_session()
        question_repo = get_candlestick_question_repository()
        session_repo = get_candlestick_session_repository()
        assessment_repo = get_candlestick_assessment_repository()
        
        # Initialize metrics repository
        metrics_repo = UserMetricsRepository(async_session_factory)
        
        # Initialize other dependencies
        question_generator = AdaptiveQuestionGenerator()
        answer_evaluator = CandlestickAnswerEvaluator()
        explanation_generator = CandlestickExplanationGenerator()
        user_metrics_service = UserMetricsService(metrics_repository=metrics_repo)
        event_dispatcher = EventDispatcher()
        
        # Create and store service instance
        _candlestick_service_instance = CandlestickAssessmentService(
            question_repository=question_repo,
            session_repository=session_repo,
            question_generator=question_generator,
            answer_evaluator=answer_evaluator,
            explanation_generator=explanation_generator,
            assessment_repository=assessment_repo,
            user_metrics_service=user_metrics_service,
            event_dispatcher=event_dispatcher
        )
        
        logger.info("CandlestickAssessmentService initialized successfully")
        return _candlestick_service_instance
        
    except Exception as e:
        logger.error(f"Error creating CandlestickAssessmentService: {str(e)}")
        raise

# Example of how the controller might get the service instance
# (This would typically be handled by a framework's DI system)
# candlestick_service_instance = create_candlestick_service() 