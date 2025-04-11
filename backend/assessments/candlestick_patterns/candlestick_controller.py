"""
Candlestick Pattern Assessment Controller

This module implements the API controller for candlestick pattern assessments,
providing endpoints for creating sessions, submitting answers, and getting results.

This controller properly extends the base assessment architecture by:
1. Inheriting from BaseAssessmentController
2. Using the CandlestickAssessmentService which extends AssessmentService
3. Working with CandlestickQuestion and CandlestickSession domain models
4. Following the established API patterns for assessment controllers
"""

from typing import Dict, List, Any, Optional, TypeVar, Union, cast
import time
import datetime
from fastapi import APIRouter, HTTPException, Depends, Query, Path, Body, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from backend.assessments.base.controllers import BaseAssessmentController
from backend.assessments.base.models import (
    AssessmentType,
    QuestionDifficulty,
    SessionStatus,
    BaseQuestion,
    AssessmentSession,
    AnswerEvaluation
)
from backend.assessments.candlestick_patterns.candlestick_service import (
    CandlestickAssessmentService, 
    create_candlestick_service
)
from backend.assessments.candlestick_patterns.candlestick_models import (
    CandlestickQuestion,
    CandlestickSession,
    CandlestickAssessmentResponse
)
from backend.assessments.base.models import SessionStatus as CandlestickSessionStatus
from backend.assessments.candlestick_patterns.candlestick_explanation_generator import UserLevel
from backend.common.auth.dependencies import get_current_user_id
from backend.common.logger import get_logger

# Import gamification components
from backend.common.gamification.integration import get_gamification_event_handler
from backend.common.gamification.models import XPSource, LeaderboardType

# Set up logger
logger = get_logger(__name__)

# Type variables for this controller
QuestionType = TypeVar('QuestionType', bound=CandlestickQuestion)
SessionType = TypeVar('SessionType', bound=CandlestickSession)

# Create router
router = APIRouter()
logger.info("Created candlestick router")

# Request Models
class StartAssessmentRequest(BaseModel):
    difficulty: float = Field(0.5, ge=0.0, le=1.0, description="Difficulty level between 0 and 1")
    total_questions: int = Field(5, gt=0, le=50, description="Number of questions")
    patterns: Optional[List[str]] = Field(None, description="Specific patterns to include")

class SubmitAnswerRequest(BaseModel):
    session_id: str = Field(..., description="Session identifier")
    question_id: str = Field(..., description="Question identifier")
    selected_option: str = Field(..., description="Selected answer option")
    response_time_ms: Optional[int] = Field(None, description="Time taken to answer in milliseconds")

# Define Pydantic model for submit answer body outside the class
class SubmitAnswerPayload(BaseModel):
    answer: str
    time_taken_ms: Optional[int] = None

# Define model for submitting candlestick answers
class CandlestickSubmitAnswer(BaseModel):
    answer: str
    confidence_level: Optional[float] = Field(None, ge=0.0, le=1.0, description="User's confidence level (0-1)")

class CandlestickPatternController(BaseAssessmentController[QuestionType, SessionType]):
    """
    Controller for candlestick pattern assessments.
    
    This class extends BaseAssessmentController and implements the API endpoints
    for candlestick pattern assessments, such as starting assessments, submitting
    answers, and getting results. It relies on an external router to register its endpoints.
    """
    
    def __init__(self):
        """Initialize the controller."""
        self._service = None
    
    @property
    def service(self) -> CandlestickAssessmentService:
        """Get the assessment service instance."""
        if self._service is None:
            self._service = create_candlestick_service()
            if self._service is None:
                raise RuntimeError("Service not initialized. Database connection may not be ready.")
        return self._service
    
    async def create_session(
        self,
        user_id: str,
        question_count: int,
        difficulty: QuestionDifficulty,
        topics: Optional[List[str]] = None
    ) -> SessionType:
        """Create a new assessment session."""
        return await self.service.create_session(
            user_id=user_id,
            question_count=question_count,
            difficulty=difficulty,
            topics=topics
        )
    
    async def get_session_question(
        self,
        session_id: str,
        question_id: str,
        user_id: str
    ) -> Optional[QuestionType]:
        """Get a specific question from a session."""
        return await self.service.get_question(question_id)
    
    async def next_question(
        self,
        session_id: str,
        user_id: str
    ) -> Optional[QuestionType]:
        """Get the next question in a session."""
        session = await self.service.get_session(session_id)
        if not session:
            return None
        current_index = session.current_question_index
        if current_index >= len(session.questions) - 1:
            return None
        next_question_id = session.questions[current_index + 1]
        return await self.service.get_question(next_question_id)
    
    async def get_user_sessions(
        self,
        user_id: str,
        limit: int = 10,
        offset: int = 0
    ) -> List[SessionType]:
        """Get a user's assessment sessions."""
        return await self.service.get_user_sessions(user_id, limit, offset)
    
    async def get_topic_performance(
        self,
        user_id: str,
        topic: str
    ) -> Dict[str, Any]:
        """Get a user's performance for a specific topic."""
        return await self.service.get_pattern_performance(user_id, topic)
    
    async def get_recommended_topics(
        self,
        user_id: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Get recommended topics for a user."""
        return await self.service.get_recommended_patterns(user_id, limit)
    
    async def get_explanation(
        self,
        topic: str,
        level: str = 'intermediate'
    ) -> Dict[str, Any]:
        """Get an explanation for a topic."""
        return await self.service.get_pattern_explanation(topic, UserLevel(level))

    async def start_assessment(
        self,
        user_id: Optional[str] = Body(None, description="Optional user identifier"),
        difficulty: Optional[str] = Body(None, description="Optional difficulty level"),
        topics: Optional[List[str]] = Body(None, description="Optional topic filter list"),
        question_count: Optional[int] = Body(5, description="Number of questions to include"),
        settings: Optional[Dict[str, Any]] = Body(None, description="Optional assessment settings")
    ) -> Dict[str, Any]:
        """
        Start a new candlestick pattern assessment.
        
        This endpoint creates a new assessment session for the user and returns the
        first question. The session can be customized with difficulty, topics,
        and question count.
        
        Args:
            user_id: Optional user identifier (defaults to "test-user" if not provided)
            difficulty: Optional difficulty level (e.g., "easy", "medium", "hard")
            topics: Optional list of topics to focus on
            question_count: Number of questions to include
            settings: Optional assessment settings
            
        Returns:
            Dictionary with session_id, question data, and session metadata
            
        Raises:
            HTTPException: If the assessment cannot be started
        """
        try:
            # Set default user_id for testing if not provided
            effective_user_id = user_id or "test-user"
            logger.info(f"Starting candlestick assessment for user {effective_user_id}")
            
            # Convert difficulty string to enum if provided
            difficulty_enum = None
            if difficulty:
                try:
                    difficulty_enum = QuestionDifficulty(difficulty)
                except ValueError:
                    logger.warning(f"Invalid difficulty '{difficulty}' provided, defaulting to MEDIUM.")
                    difficulty_enum = QuestionDifficulty.MEDIUM
            
            # Create a new session with the specified parameters
            session = await self.service.create_session(
                user_id=effective_user_id,
                question_count=question_count,
                topics=topics,
                difficulty=difficulty_enum,
                settings=settings
            )
            
            if not session:
                logger.error(f"Failed to create session for user {effective_user_id}")
                raise HTTPException(status_code=500, detail="Failed to create assessment session")
            
            # Get the first question
            first_question_id = session.questions[0] if session.questions else None
            
            if not first_question_id:
                logger.error(f"No questions in session {session.id}")
                raise HTTPException(status_code=500, detail="No questions available in the session")
            
            # Get the first question from the session metadata instead of calling get_question
            first_question_data = None
            if 'question_data' in session.settings:
                for q_data in session.settings['question_data']:
                    if q_data.get('id') == first_question_id:
                        first_question_data = q_data
                        break
            
            if not first_question_data:
                logger.error(f"Session {session.id} created but failed to retrieve first question {first_question_id}")
                raise HTTPException(status_code=500, detail="Failed to retrieve initial question data.")
            
            # Return session and first question information
            return {
                "session_id": session.id,
                "question": first_question_data,
                "current_index": 0,
                "total_questions": len(session.questions),
                "session_status": session.status.value,
                "user_id": session.user_id,
                "created_at": session.created_at.isoformat() if session.created_at else None
            }
            
        except AssessmentError as e:
            logger.error(f"Assessment error starting assessment for user {user_id}: {str(e)}")
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Unexpected error starting assessment for user {user_id}: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def get_session(
        self,
        session_id: str = Path(..., description="Session identifier"),
        user_id: str = Depends(get_current_user_id)
    ) -> Dict[str, Any]:
        """
        Get a candlestick pattern assessment session.
        
        Args:
            session_id: Session identifier
            user_id: User identifier
            
        Returns:
            Session data
        """
        try:
            session = await self.service.get_session(session_id)
            if not session:
                logger.warning(f"Session {session_id} not found or failed to retrieve.")
                raise HTTPException(status_code=404, detail="Session not found or could not be retrieved.")
            if session.user_id != user_id:
                logger.warning(f"User {user_id} attempted to access unauthorized session {session_id}")
                raise HTTPException(status_code=403, detail="Not authorized to access this session")
            return session.to_dict()
        except HTTPException:
            raise
        except RuntimeError as re:
            logger.error(f"Runtime error getting session {session_id}: {re}", exc_info=True)
            raise HTTPException(status_code=500, detail="An internal error occurred while retrieving the session.")
        except Exception as e:
            logger.error(f"Unexpected error getting session {session_id}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="An unexpected error occurred.")
    
    async def get_question(
        self,
        session_id: str = Path(..., description="Session identifier"),
        question_id: str = Path(..., description="Question identifier"),
        user_id: str = Depends(get_current_user_id)
    ) -> Dict[str, Any]:
        """
        Get a specific question from a session.
        
        Args:
            session_id: Session identifier
            question_id: Question identifier
            user_id: User identifier
            
        Returns:
            Question data
        """
        try:
            session = await self.service.get_session(session_id)
            if not session:
                logger.warning(f"Session {session_id} not found or failed to retrieve during get_question call.")
                raise HTTPException(status_code=404, detail="Session not found or could not be retrieved.")
            if session.user_id != user_id:
                logger.warning(f"User {user_id} attempted to access question {question_id} from unauthorized session {session_id}")
                raise HTTPException(status_code=403, detail="Not authorized to access this session")
            if question_id not in session.questions:
                logger.warning(f"User {user_id} requested question {question_id} not part of session {session_id}")
                raise HTTPException(status_code=404, detail="Question not part of this session")
            
            # Get the question from the session metadata instead of calling get_question
            question_data = None
            if 'question_data' in session.settings:
                for q_data in session.settings['question_data']:
                    if q_data.get('id') == question_id:
                        question_data = q_data
                        break
            
            if not question_data:
                logger.warning(f"Question {question_id} not found in session metadata for session {session_id}")
                raise HTTPException(status_code=404, detail="Question not found in session data")
                
            return question_data
        except HTTPException:
            raise
        except RuntimeError as re:
            logger.error(f"Runtime error getting question {question_id} for session {session_id}: {re}", exc_info=True)
            raise HTTPException(status_code=500, detail="An internal error occurred while retrieving the question.")
        except Exception as e:
            logger.error(f"Unexpected error getting question {question_id} for session {session_id}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="An unexpected error occurred.")
            
    async def submit_answer(
        self,
        session_id: str = Path(..., description="Session identifier"),
        question_id: str = Path(..., description="Question identifier"),
        answer: CandlestickSubmitAnswer = Body(..., description="User's answer data"),
        user_id: str = Depends(get_current_user_id)
    ) -> Dict[str, Any]:
        """
        Submit an answer for a question within a session.
        
        Args:
            session_id: Session identifier
            question_id: Question identifier
            answer: Answer data from the user
            user_id: User identifier
            
        Returns:
            Answer evaluation result
        """
        try:
            session = await self.service.get_session(session_id)
            if not session:
                logger.warning(f"Session {session_id} not found or failed to retrieve during submit_answer call.")
                raise HTTPException(status_code=404, detail="Session not found or could not be retrieved.")
            if session.user_id != user_id:
                logger.warning(f"User {user_id} attempted to submit answer for question {question_id} in unauthorized session {session_id}")
                raise HTTPException(status_code=403, detail="Not authorized to access this session")
            if session.status not in [CandlestickSessionStatus.IN_PROGRESS]:
                logger.warning(f"User {user_id} attempted to submit answer for question {question_id} in session {session_id} with invalid status {session.status}")
                raise HTTPException(status_code=400, detail=f"Cannot submit answers for a session with status {session.status}")
            if question_id not in session.questions:
                logger.warning(f"User {user_id} attempted to submit answer for question {question_id} which is not part of session {session_id}")
                raise HTTPException(status_code=404, detail="Question not part of this session")

            # Get the question from the session metadata
            question_data = None
            if 'question_data' in session.settings:
                for q_data in session.settings['question_data']:
                    if q_data.get('id') == question_id:
                        question_data = q_data
                        break
            
            if not question_data:
                logger.warning(f"Question {question_id} not found in session metadata for session {session_id}")
                raise HTTPException(status_code=404, detail="Question not found in session data")

            # Create a CandlestickQuestion object from the data
            question = CandlestickQuestion(
                id=question_data.get('id'),
                question_text=question_data.get('question_text', ''),
                difficulty=question_data.get('difficulty', 0.5),
                pattern=question_data.get('pattern', ''),
                correct_answer=question_data.get('correct_answer', ''),
                explanation=question_data.get('explanation', ''),
                choices=question_data.get('choices', []),
                topics=question_data.get('topics', []),
                template_id=question_data.get('template_id', ''),
                created_at=question_data.get('created_at', datetime.datetime.utcnow().isoformat())
            )
            
            # Submit the answer
            result = await self.service.submit_answer(session_id=session_id, question_id=question_id, 
                                                   user_answer=answer.answer, 
                                                   question=question,
                                                   confidence_level=answer.confidence_level)
            
            return result
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Unexpected error submitting answer for question {question_id} in session {session_id}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="An unexpected error occurred.")

    async def complete_session(
        self,
        session_id: str = Path(..., description="Session identifier"),
        user_id: str = Depends(get_current_user_id)
    ) -> Dict[str, Any]:
        logger.warning("complete_session endpoint not fully implemented with error handling")
        raise HTTPException(status_code=501, detail="Not Implemented")

    async def get_session_results(
        self,
        session_id: str = Path(..., description="Session identifier"),
        user_id: str = Depends(get_current_user_id)
    ) -> Dict[str, Any]:
        """ Get results for a completed session. """
        try:
            # First, check authorization by getting the session
            session = await self.service.get_session(session_id)
            if not session:
                logger.warning(f"Session {session_id} not found or failed to retrieve during get_session_results.")
                raise HTTPException(status_code=404, detail="Session not found or could not be retrieved.")
            if session.user_id != user_id:
                logger.warning(f"User {user_id} attempted to access results for unauthorized session {session_id}")
                raise HTTPException(status_code=403, detail="Not authorized to access this session's results.")
            
            # Now call the service method to get formatted results
            # This method internally checks if the session is completed
            results = await self.service.get_session_results(session_id)
            return results
        except ValueError as ve:
            # Catches errors like session not found (if get_session is called again inside service) or session not completed
            logger.warning(f"Validation error getting results for session {session_id}: {ve}")
            # Determine if 404 or 400 is more appropriate
            raise HTTPException(status_code=400, detail=str(ve))
        except RuntimeError as re:
            logger.error(f"Runtime error getting results for session {session_id}: {re}", exc_info=True)
            raise HTTPException(status_code=500, detail="Internal error retrieving results.")
        except Exception as e:
            logger.error(f"Unexpected error getting results for session {session_id}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Unexpected error.")

    async def get_user_performance(
        self,
        user_id: str = Depends(get_current_user_id)
    ) -> Dict[str, Any]:
        logger.warning("get_user_performance endpoint not fully implemented with error handling")
        try:
            perf = await self.service.get_user_performance(user_id)
            return perf
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=str(ve))
        except RuntimeError as re:
            logger.error(f"Runtime error getting user performance for {user_id}: {re}", exc_info=True)
            raise HTTPException(status_code=500, detail="Internal error retrieving performance.")
        except Exception as e:
            logger.error(f"Unexpected error getting user performance for {user_id}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Unexpected error.")

    async def get_pattern_performance(
        self,
        pattern: str = Path(..., description="Candlestick pattern name"),
        user_id: str = Depends(get_current_user_id)
    ) -> Dict[str, Any]:
        """ Get user performance for a specific pattern. """
        try:
            # TODO: Maybe validate pattern against PatternType enum?
            perf = await self.service.get_topic_performance(user_id, pattern)
            return perf
        except ValueError as ve:
             # Raised by service if user_id/topic empty (or potentially invalid pattern later)
             logger.warning(f"Validation error getting performance for pattern {pattern}, user {user_id}: {ve}")
             raise HTTPException(status_code=400, detail=str(ve))
        except NotImplementedError as nie:
             # Raised by service if repo method missing
             logger.error(f"Topic performance feature not implemented: {nie}")
             raise HTTPException(status_code=501, detail="This feature is not currently available.")
        except RuntimeError as re:
             # Raised by service on RepositoryError
             logger.error(f"Runtime error getting performance for pattern {pattern}, user {user_id}: {re}", exc_info=True)
             raise HTTPException(status_code=500, detail="Internal error retrieving pattern performance.")
        except Exception as e:
             logger.error(f"Unexpected error getting performance for pattern {pattern}, user {user_id}: {e}", exc_info=True)
             raise HTTPException(status_code=500, detail="Unexpected error retrieving pattern performance.")

    async def analyze_pattern_performance(
        self,
        user_id: str = Depends(get_current_user_id)
    ) -> Dict[str, Any]:
        """ Analyze overall pattern performance for the user. """
        try:
            analysis = await self.service.analyze_all_topic_performance(user_id)
            return analysis
        except ValueError as ve:
             logger.warning(f"Validation error analyzing pattern performance for {user_id}: {ve}")
             raise HTTPException(status_code=400, detail=str(ve))
        except NotImplementedError as nie:
             logger.error(f"Pattern analysis feature not implemented: {nie}")
             raise HTTPException(status_code=501, detail="This feature is not currently available.")
        except RuntimeError as re:
             logger.error(f"Runtime error analyzing pattern performance for {user_id}: {re}", exc_info=True)
             raise HTTPException(status_code=500, detail="Internal error analyzing performance.")
        except Exception as e:
             logger.error(f"Unexpected error analyzing pattern performance for {user_id}: {e}", exc_info=True)
             raise HTTPException(status_code=500, detail="Unexpected error analyzing performance.")

    async def get_recommended_patterns(
        self,
        user_id: str = Depends(get_current_user_id),
        limit: int = Query(5, gt=0, le=20, description="Max number of recommendations")
    ) -> List[Dict[str, Any]]:
        """ Get recommended patterns for the user based on performance. """
        try:
            recommendations = await self.service.get_recommendations(user_id, limit)
            return recommendations
        except ValueError as ve:
             # Raised by service if user_id empty
             logger.warning(f"Validation error getting recommendations for {user_id}: {ve}")
             raise HTTPException(status_code=400, detail=str(ve))
        except NotImplementedError as nie:
             logger.error(f"Recommendations feature not implemented: {nie}")
             raise HTTPException(status_code=501, detail="This feature is not currently available.")
        except RuntimeError as re:
             # Raised by service on RepositoryError
             logger.error(f"Runtime error getting recommendations for {user_id}: {re}", exc_info=True)
             raise HTTPException(status_code=500, detail="Internal error retrieving recommendations.")
        except Exception as e:
             logger.error(f"Unexpected error getting recommendations for {user_id}: {e}", exc_info=True)
             raise HTTPException(status_code=500, detail="Unexpected error retrieving recommendations.")
        
    async def get_pattern_explanation(
        self,
        pattern: str = Path(..., description="Candlestick pattern name"),
        level: str = Query('intermediate', description="Detail level: beginner, intermediate, advanced")
    ) -> Dict[str, Any]:
        """
        Get explanation for a specific candlestick pattern.
        
        Args:
            pattern: Pattern name
            level: User knowledge level
            
        Returns:
            Pattern explanation data
        """
        logger.warning("get_pattern_explanation endpoint not fully implemented with error handling")
        try:
            explanation = await self.service.get_explanation(
                question_id=None,
                pattern_name=pattern,
                user_level=level
            )
            return explanation
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=str(ve))
        except RuntimeError as re:
            logger.error(f"Runtime error getting explanation for {pattern}: {re}", exc_info=True)
            raise HTTPException(status_code=500, detail="Internal error retrieving explanation.")
        except Exception as e:
            logger.error(f"Unexpected error getting explanation for {pattern}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Unexpected error.")


# Create singleton instance
candlestick_controller = CandlestickPatternController()

# Register routes with the router
@router.post("/start")
async def start_assessment_endpoint(
    request: StartAssessmentRequest,
    user_id: str = Depends(get_current_user_id)
) -> Dict[str, Any]:
    """Start a new assessment session."""
    # Format request as expected by the start_assessment method
    return await candlestick_controller.start_assessment(
        user_id=user_id,
        difficulty=str(request.difficulty),
        topics=request.patterns,
        question_count=request.total_questions
    )

@router.post("/submit_answer")
async def submit_answer_endpoint(
    request: SubmitAnswerRequest,
    user_id: str = Depends(get_current_user_id)
) -> Dict[str, Any]:
    answer_data = CandlestickSubmitAnswer(
        answer=request.selected_option,
        confidence_level=None  # No confidence data in legacy model
    )
    return await candlestick_controller.submit_answer(
        session_id=request.session_id,
        question_id=request.question_id,
        answer=answer_data,
        user_id=user_id
    )

@router.get("/session/{session_id}")
async def get_session_endpoint(
    session_id: str,
    user_id: str = Depends(get_current_user_id)
) -> Dict[str, Any]:
    return await candlestick_controller.get_session(
        session_id=session_id,
        user_id=user_id
    )

@router.get("/question/{session_id}/{question_id}")
async def get_question_endpoint(
    session_id: str,
    question_id: str,
    user_id: str = Depends(get_current_user_id)
) -> Dict[str, Any]:
    return await candlestick_controller.get_question(
        session_id=session_id,
        question_id=question_id,
        user_id=user_id
    )

@router.post("/complete_session")
async def complete_session_endpoint(
    session_id: str,
    user_id: str = Depends(get_current_user_id)
) -> Dict[str, Any]:
    return await candlestick_controller.complete_session(
        session_id=session_id,
        user_id=user_id
    )

@router.get("/session_results/{session_id}")
async def get_session_results_endpoint(
    session_id: str,
    user_id: str = Depends(get_current_user_id)
) -> Dict[str, Any]:
    return await candlestick_controller.get_session_results(
        session_id=session_id,
        user_id=user_id
    )

@router.get("/user_performance")
async def get_user_performance_endpoint(
    user_id: str = Depends(get_current_user_id)
) -> Dict[str, Any]:
    return await candlestick_controller.get_user_performance(
        user_id=user_id
    )

@router.get("/pattern_performance/{pattern}")
async def get_pattern_performance_endpoint(
    pattern: str,
    user_id: str = Depends(get_current_user_id)
) -> Dict[str, Any]:
    return await candlestick_controller.get_pattern_performance(
        pattern=pattern,
        user_id=user_id
    )

@router.get("/analyze_pattern_performance")
async def analyze_pattern_performance_endpoint(
    user_id: str = Depends(get_current_user_id)
) -> Dict[str, Any]:
    return await candlestick_controller.analyze_pattern_performance(
        user_id=user_id
    )

@router.get("/recommended_patterns")
async def get_recommended_patterns_endpoint(
    user_id: str = Depends(get_current_user_id),
    limit: int = Query(5, gt=0, le=20, description="Max number of recommendations")
) -> List[Dict[str, Any]]:
    return await candlestick_controller.get_recommended_patterns(
        user_id=user_id,
        limit=limit
    )

@router.get("/pattern_explanation/{pattern}")
async def get_pattern_explanation_endpoint(
    pattern: str,
    level: str = Query('intermediate', description="Detail level: beginner, intermediate, advanced")
) -> Dict[str, Any]:
    return await candlestick_controller.get_pattern_explanation(
        pattern=pattern,
        level=level
    ) 