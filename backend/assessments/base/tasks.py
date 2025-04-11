"""
Assessment Tasks Module

This module contains task definitions for assessment-related operations,
such as session creation, question generation, and result processing.
These async tasks are designed to be executed in a background task queue
for handling long-running or resource-intensive operations.
"""

import datetime
import logging
import uuid
from typing import Any, Dict, List, Optional, Union

from backend.common.tasks.registry import task
from backend.assessments.base.models import AssessmentType, QuestionDifficulty, SessionStatus
from backend.common.errors import ValidationError, ResourceNotFoundError, ServiceError

# Set up logging
logger = logging.getLogger(__name__)


@task(
    queue="assessments",
    tags=["assessment", "sessions"],
    description="Create a new assessment session for a user",
    retry=True,
    max_retries=3,
    retry_delay=5
)
async def create_assessment_session(
    user_id: str,
    assessment_type: str,
    difficulty: str = "adaptive",
    topics: Optional[List[str]] = None,
    question_count: int = 10,
    time_limit_minutes: Optional[int] = None,
    settings: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create a new assessment session for a user.
    
    Args:
        user_id: The user's unique identifier
        assessment_type: Type of assessment (e.g., "pattern", "market_basics")
        difficulty: Difficulty level or "adaptive" for adaptive difficulty
        topics: List of topics to cover (None for all topics)
        question_count: Number of questions in the session
        time_limit_minutes: Optional time limit in minutes
        settings: Additional configuration settings for the session
        
    Returns:
        Dictionary with session details
        
    Raises:
        ValidationError: If parameters are invalid
        ServiceError: If session creation fails
    """
    logger.info(f"Creating {assessment_type} assessment session for user {user_id}")
    
    # Validate user_id
    if not user_id:
        error_msg = "User ID is required"
        logger.error(error_msg)
        raise ValidationError(error_msg)
    
    # Validate question_count
    if question_count < 1 or question_count > 50:
        error_msg = f"Invalid question count: {question_count}. Must be between 1 and 50."
        logger.error(error_msg)
        raise ValidationError(error_msg)
    
    # Validate assessment type
    try:
        assessment_enum = AssessmentType(assessment_type)
    except ValueError:
        error_msg = f"Invalid assessment type: {assessment_type}"
        logger.error(error_msg)
        raise ValidationError(error_msg)
    
    # Validate difficulty
    if difficulty != "adaptive":
        try:
            difficulty_enum = QuestionDifficulty(difficulty)
        except ValueError:
            error_msg = f"Invalid difficulty: {difficulty}"
            logger.error(error_msg)
            raise ValidationError(error_msg)
    
    # Validate time limit if provided
    if time_limit_minutes is not None and (time_limit_minutes < 1 or time_limit_minutes > 180):
        error_msg = f"Invalid time limit: {time_limit_minutes}. Must be between 1 and 180 minutes."
        logger.error(error_msg)
        raise ValidationError(error_msg)
    
    try:
        # Generate a unique session ID
        session_id = str(uuid.uuid4())
        current_time = datetime.datetime.now(datetime.timezone.utc)
        
        session = {
            "id": session_id,
            "user_id": user_id,
            "assessment_type": assessment_type,
            "difficulty": difficulty,
            "topics": topics or [],
            "question_count": question_count,
            "time_limit_minutes": time_limit_minutes,
            "created_at": current_time.isoformat(),
            "status": SessionStatus.CREATED.value,
            "questions": [],
            "current_question_index": 0,
            "settings": settings or {}
        }
        
        # Generate session questions in the background
        await generate_session_questions.delay(
            session_id=session_id,
            assessment_type=assessment_type,
            difficulty=difficulty,
            topics=topics,
            question_count=question_count
        )
        
        logger.info(f"Created assessment session {session_id} for user {user_id}")
        return session
        
    except Exception as e:
        error_msg = f"Failed to create assessment session: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise ServiceError(error_msg) from e


@task(
    queue="assessments",
    tags=["assessment", "questions"],
    description="Generate questions for an assessment session",
    retry=True,
    max_retries=3,
    retry_delay=10
)
async def generate_session_questions(
    session_id: str,
    assessment_type: str,
    difficulty: str,
    topics: Optional[List[str]] = None,
    question_count: int = 10
) -> Dict[str, Any]:
    """
    Generate questions for an assessment session.
    
    Args:
        session_id: The session's unique identifier
        assessment_type: Type of assessment
        difficulty: Difficulty level or "adaptive"
        topics: List of topics to cover
        question_count: Number of questions to generate
        
    Returns:
        Dictionary with generated questions
        
    Raises:
        ValidationError: If parameters are invalid
        ResourceNotFoundError: If session not found
        ServiceError: If question generation fails
    """
    logger.info(f"Generating {question_count} questions for session {session_id}")
    
    # Validate session_id
    if not session_id:
        error_msg = "Session ID is required"
        logger.error(error_msg)
        raise ValidationError(error_msg)
    
    try:
        # Generate questions based on the parameters
        questions = []
        for i in range(question_count):
            topic = topics[i % len(topics)] if topics else f"topic_{i+1}"
            
            question = {
                "id": f"q_{i+1}_{session_id}",
                "question_type": "multiple_choice",
                "question_text": f"Sample question #{i+1} for session {session_id}",
                "difficulty": difficulty if difficulty != "adaptive" else "intermediate",
                "topics": [topic],
                "subtopics": [],
                "options": [
                    {"id": "a", "text": "Option A"},
                    {"id": "b", "text": "Option B"},
                    {"id": "c", "text": "Option C"},
                    {"id": "d", "text": "Option D"}
                ],
                "correct_answer": "a",
                "explanation": f"Explanation for question #{i+1}"
            }
            questions.append(question)
        
        # Update the session with generated questions
        result = {
            "session_id": session_id,
            "questions": questions,
            "status": SessionStatus.IN_PROGRESS.value
        }
        
        logger.info(f"Generated {len(questions)} questions for session {session_id}")
        return result
        
    except ResourceNotFoundError as e:
        logger.error(f"Session not found: {session_id}", exc_info=True)
        raise
        
    except Exception as e:
        error_msg = f"Failed to generate questions for session {session_id}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise ServiceError(error_msg) from e


@task(
    queue="assessments",
    tags=["assessment", "evaluation"],
    description="Evaluate a user's answer to an assessment question",
    retry=True,
    max_retries=3,
    retry_delay=5
)
async def evaluate_user_answer(
    session_id: str,
    question_id: str,
    user_answer: Any,
    user_id: str,
    timing_ms: Optional[int] = None
) -> Dict[str, Any]:
    """
    Evaluate a user's answer to an assessment question.
    
    Args:
        session_id: The session's unique identifier
        question_id: The question's unique identifier
        user_answer: The user's answer
        user_id: The user's unique identifier
        timing_ms: Time taken to answer in milliseconds
        
    Returns:
        Dictionary with evaluation results
        
    Raises:
        ValidationError: If parameters are invalid
        ResourceNotFoundError: If session or question not found
        ServiceError: If evaluation fails
    """
    logger.info(f"Evaluating answer for question {question_id} in session {session_id}")
    
    # Validate required parameters
    if not session_id or not question_id or not user_id:
        error_msg = "Session ID, question ID, and user ID are required"
        logger.error(error_msg)
        raise ValidationError(error_msg)
    
    # Validate the timing if provided
    if timing_ms is not None and timing_ms < 0:
        error_msg = f"Invalid timing: {timing_ms}. Must be a non-negative value."
        logger.error(error_msg)
        raise ValidationError(error_msg)
    
    try:
        current_time = datetime.datetime.now(datetime.timezone.utc)
        
        # Simulate evaluation
        is_correct = user_answer == "a"  # Mock evaluation
        
        evaluation = {
            "session_id": session_id,
            "question_id": question_id,
            "user_id": user_id,
            "user_answer": user_answer,
            "is_correct": is_correct,
            "score": 1.0 if is_correct else 0.0,
            "confidence": 1.0,  # High confidence in the evaluation
            "feedback": "Correct answer!" if is_correct else "Incorrect. The correct answer is A.",
            "explanation": "Detailed explanation would be provided here.",
            "timing_ms": timing_ms,
            "evaluated_at": current_time.isoformat()
        }
        
        # Update performance metrics for the user asynchronously
        await update_user_performance.delay(
            user_id=user_id,
            question_id=question_id,
            is_correct=is_correct,
            difficulty=QuestionDifficulty.INTERMEDIATE.value,  # Mock difficulty
            topic="sample_topic",  # Mock topic
            timing_ms=timing_ms
        )
        
        logger.info(f"Evaluated answer for question {question_id}: {'correct' if is_correct else 'incorrect'}")
        return evaluation
        
    except ResourceNotFoundError as e:
        logger.error(f"Resource not found during evaluation: {str(e)}", exc_info=True)
        raise
        
    except Exception as e:
        error_msg = f"Failed to evaluate answer: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise ServiceError(error_msg) from e


@task(
    queue="assessments",
    tags=["assessment", "performance"],
    description="Update user performance metrics",
    retry=True,
    max_retries=3,
    retry_delay=5
)
async def update_user_performance(
    user_id: str,
    question_id: str,
    is_correct: bool,
    difficulty: str,
    topic: str,
    timing_ms: Optional[int] = None
) -> Dict[str, Any]:
    """
    Update performance metrics for a user based on their answer.
    
    Args:
        user_id: The user's unique identifier
        question_id: The question's unique identifier
        is_correct: Whether the answer was correct
        difficulty: Difficulty of the question
        topic: Topic of the question
        timing_ms: Time taken to answer in milliseconds
        
    Returns:
        Dictionary with updated performance metrics
        
    Raises:
        ValidationError: If parameters are invalid
        ResourceNotFoundError: If user not found
        ServiceError: If metrics update fails
    """
    logger.info(f"Updating performance metrics for user {user_id} on question {question_id}")
    
    # Validate required parameters
    if not user_id or not question_id:
        error_msg = "User ID and question ID are required"
        logger.error(error_msg)
        raise ValidationError(error_msg)
    
    try:
        current_time = datetime.datetime.now(datetime.timezone.utc)
        
        # Mock updated metrics
        metrics = {
            "user_id": user_id,
            "topic": topic,
            "difficulty": difficulty,
            "correct_count": 1 if is_correct else 0,
            "total_count": 1,
            "accuracy": 1.0 if is_correct else 0.0,
            "avg_time_ms": timing_ms or 0,
            "skill_level": 0.7,  # Mock skill level
            "updated_at": current_time.isoformat()
        }
        
        logger.info(f"Updated performance metrics for user {user_id} on topic {topic}")
        return metrics
        
    except ResourceNotFoundError as e:
        logger.error(f"User not found: {user_id}", exc_info=True)
        raise
        
    except Exception as e:
        error_msg = f"Failed to update performance metrics: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise ServiceError(error_msg) from e


@task(
    queue="assessments",
    tags=["assessment", "sessions"],
    description="Complete an assessment session and process results",
    retry=True,
    max_retries=3,
    retry_delay=10
)
async def complete_assessment_session(
    session_id: str,
    user_id: str
) -> Dict[str, Any]:
    """
    Complete an assessment session and process the results.
    
    Args:
        session_id: The session's unique identifier
        user_id: The user's unique identifier
        
    Returns:
        Dictionary with session results
        
    Raises:
        ValidationError: If parameters are invalid
        ResourceNotFoundError: If session not found
        ServiceError: If session completion fails
    """
    logger.info(f"Completing assessment session {session_id} for user {user_id}")
    
    # Validate required parameters
    if not session_id or not user_id:
        error_msg = "Session ID and user ID are required"
        logger.error(error_msg)
        raise ValidationError(error_msg)
    
    try:
        current_time = datetime.datetime.now(datetime.timezone.utc)
        
        # Mock session results
        results = {
            "session_id": session_id,
            "user_id": user_id,
            "status": SessionStatus.COMPLETED.value,
            "completed_at": current_time.isoformat(),
            "total_questions": 10,
            "answered_questions": 10,
            "correct_answers": 7,
            "accuracy": 0.7,
            "avg_time_ms": 5000,
            "total_score": 7.0,
            "max_score": 10.0,
            "topics": {
                "topic_1": {
                    "accuracy": 0.75,
                    "questions": 4
                },
                "topic_2": {
                    "accuracy": 0.67,
                    "questions": 6
                }
            },
            "feedback": "Good job! You've demonstrated a solid understanding of the material.",
            "recommendations": [
                "Focus more on topic_2 to improve your skills.",
                "Consider reviewing the material on pattern recognition."
            ]
        }
        
        # Generate a session summary report asynchronously
        await generate_session_summary.delay(
            session_id=session_id,
            user_id=user_id,
            results=results
        )
        
        logger.info(f"Completed assessment session {session_id} for user {user_id}")
        return results
        
    except ResourceNotFoundError as e:
        logger.error(f"Session not found: {session_id}", exc_info=True)
        raise
        
    except Exception as e:
        error_msg = f"Failed to complete session: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise ServiceError(error_msg) from e


@task(
    queue="reporting",
    tags=["assessment", "reporting"],
    description="Generate a summary report for an assessment session",
    retry=True,
    max_retries=3,
    retry_delay=15
)
async def generate_session_summary(
    session_id: str,
    user_id: str,
    results: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generate a detailed summary report for an assessment session.
    
    Args:
        session_id: The session's unique identifier
        user_id: The user's unique identifier
        results: The session results
        
    Returns:
        Dictionary with summary report details
        
    Raises:
        ValidationError: If parameters are invalid
        ResourceNotFoundError: If session not found
        ServiceError: If report generation fails
    """
    logger.info(f"Generating summary report for session {session_id}")
    
    # Validate required parameters
    if not session_id or not user_id or not results:
        error_msg = "Session ID, user ID, and results are required"
        logger.error(error_msg)
        raise ValidationError(error_msg)
    
    try:
        current_time = datetime.datetime.now(datetime.timezone.utc)
        report_id = f"report_{session_id}_{int(current_time.timestamp())}"
        
        # Mock summary report
        report = {
            "session_id": session_id,
            "user_id": user_id,
            "report_id": report_id,
            "generated_at": current_time.isoformat(),
            "title": "Assessment Session Summary",
            "performance": {
                "overall_score": f"{results['total_score']}/{results['max_score']}",
                "accuracy": f"{results['accuracy'] * 100:.1f}%",
                "skill_level": "Intermediate",
                "progress": "+5% from last assessment"
            },
            "strengths": [
                {"topic": "topic_1", "score": results['topics']['topic_1']['accuracy'], "description": "Good understanding of topic 1"},
            ],
            "areas_for_improvement": [
                {"topic": "topic_2", "score": results['topics']['topic_2']['accuracy'], "description": "Need more practice on topic 2"},
            ],
            "recommendations": results['recommendations'],
            "next_steps": [
                "Complete the recommended practice exercises",
                "Schedule a follow-up assessment in 2 weeks"
            ],
            "format": "pdf"
        }
        
        logger.info(f"Generated summary report {report_id} for session {session_id}")
        return report
        
    except ResourceNotFoundError as e:
        logger.error(f"Session not found: {session_id}", exc_info=True)
        raise
        
    except Exception as e:
        error_msg = f"Failed to generate session summary: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise ServiceError(error_msg) from e 