"""
Session Service for Candlestick Pattern Assessments

This module provides session management functionality for the candlestick pattern assessment system,
handling the creation, progression, and completion of assessment sessions.
"""

import uuid
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime

from backend.common.logger import app_logger
from backend.common.base_assessment import (
    AssessmentSession, AssessmentMetrics, EventDispatcher,
    QuestionGeneratedEvent, AnswerSubmittedEvent, AnswerEvaluatedEvent, SessionCompletedEvent
)
from backend.common.assessment_service import BaseAssessmentService
from backend.assessments.base.services import AssessmentService

from backend.assessments.candlestick_patterns.repository import (
    candlestick_session_repository, candlestick_question_repository
)
from backend.assessments.candlestick_patterns.candlestick_pattern_recognition import (
    CandlestickPatternQuestion, default_candlestick_service
)
from backend.assessments.candlestick_patterns.candlestick_config import ASSESSMENT_CONFIG

# Module logger
logger = app_logger.getChild("candlestick.session_service")

class CandlestickSessionService(BaseAssessmentService[CandlestickPatternQuestion, AssessmentSession]):
    """
    Service for managing candlestick pattern assessment sessions.
    
    This class extends the base assessment service with candlestick pattern-specific
    functionality, providing management for assessment sessions including creation,
    progression, and completion.
    """
    
    def __init__(
        self,
        event_dispatcher: Optional[EventDispatcher] = None
    ):
        """
        Initialize the candlestick session service.
        
        Args:
            event_dispatcher: Optional event dispatcher for domain events
        """
        super().__init__(
            assessment_type="candlestick",
            session_repository=candlestick_session_repository,
            question_repository=candlestick_question_repository,
            event_dispatcher=event_dispatcher or EventDispatcher()
        )
        logger.info("Initialized candlestick session service")
    
    def _create_session_object(
        self,
        session_id: str,
        user_id: str,
        target_question_count: int,
        max_duration_sec: Optional[int],
        **kwargs
    ) -> AssessmentSession:
        """
        Create a session object for candlestick assessment.
        
        Args:
            session_id: Session identifier
            user_id: User identifier
            target_question_count: Number of questions to include
            max_duration_sec: Optional maximum duration in seconds
            kwargs: Additional keyword arguments
            
        Returns:
            An assessment session
        """
        # Create session with appropriate configuration
        session = AssessmentSession(
            session_id=session_id,
            user_id=user_id,
            assessment_type="candlestick",
            target_question_count=target_question_count,
            max_duration_sec=max_duration_sec,
            created_at=kwargs.get("created_at", datetime.now()),
            metadata=kwargs.get("metadata", {
                "difficulty": kwargs.get("difficulty", "medium"),
                "patterns": kwargs.get("patterns", [])
            })
        )
        
        logger.debug(f"Created session {session_id} for user {user_id} with {target_question_count} questions")
        return session
    
    async def _generate_question(
        self,
        user_id: str,
        session_id: str
    ) -> CandlestickPatternQuestion:
        """
        Generate a candlestick pattern question.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            
        Returns:
            A candlestick pattern question
        """
        # Get the session to retrieve configuration
        session = await self.session_repository.get(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")
            
        # Extract configuration from session metadata
        difficulty = session.metadata.get("difficulty", "medium")
        patterns = session.metadata.get("patterns", [])
        
        # Delegate to the candlestick service for question generation
        question = await default_candlestick_service.generate_question(
            user_id=user_id,
            session_id=session_id,
            difficulty_level=self._convert_difficulty_to_numeric(difficulty),
            target_pattern=patterns[0] if patterns else None
        )
        
        # Log question generation
        logger.debug(f"Generated question {question.question_id} for session {session_id}")
        
        # Dispatch event 
        await self.event_dispatcher.dispatch(QuestionGeneratedEvent(
            session_id=session_id,
            question_id=question.question_id,
            user_id=user_id,
            difficulty=question.difficulty.value
        ))
        
        return question
    
    async def _evaluate_answer(
        self,
        question: CandlestickPatternQuestion,
        user_answer: str,
        user_id: str,
        session_id: str,
        time_spent_ms: int
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Evaluate a user's answer to a candlestick pattern question.
        
        Args:
            question: The question being answered
            user_answer: The user's answer
            user_id: User identifier
            session_id: Session identifier
            time_spent_ms: Time spent answering in milliseconds
            
        Returns:
            Tuple of (is_correct, confidence, details)
        """
        # Delegate to the candlestick service for answer evaluation
        result = await default_candlestick_service.evaluate_answer(
            question=question,
            user_answer=user_answer,
            user_id=user_id,
            session_id=session_id,
            time_spent_ms=time_spent_ms
        )
        
        # Log result
        is_correct, confidence, details = result
        logger.debug(f"Evaluated answer for question {question.question_id} in session {session_id}: correct={is_correct}, confidence={confidence}")
        
        # Dispatch events
        await self.event_dispatcher.dispatch(AnswerSubmittedEvent(
            session_id=session_id,
            question_id=question.question_id,
            user_id=user_id,
            selected_answer=user_answer,
            time_spent_ms=time_spent_ms
        ))
        
        await self.event_dispatcher.dispatch(AnswerEvaluatedEvent(
            session_id=session_id,
            question_id=question.question_id,
            is_correct=is_correct,
            confidence=confidence
        ))
        
        return result
    
    def _get_question_details(
        self,
        question: CandlestickPatternQuestion,
        answer_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Get details for a candlestick pattern question.
        
        Args:
            question: The question
            answer_data: Answer data from the session
            
        Returns:
            Dictionary with question details
        """
        # Extract candlestick-specific information
        pattern = question.candlestick_data.target_pattern if question.candlestick_data else ""
        
        # Get basic question information
        details = {
            "question_id": question.question_id,
            "question_text": question.content.text,
            "options": question.content.options,
            "difficulty": question.difficulty.value,
            "candlestick_data": question.candlestick_data.to_dict() if question.candlestick_data else None,
            "pattern": pattern
        }
        
        # Add answer information if available
        if answer_data:
            details.update({
                "user_answer": answer_data.get("answer"),
                "is_correct": answer_data.get("is_correct", False),
                "correct_answer": question.content.correct_option,
                "time_spent_ms": answer_data.get("time_taken_ms")
            })
            
            # Add explanation if available
            if "explanation" in answer_data:
                details["explanation"] = answer_data["explanation"]
        
        return details
    
    async def get_session_statistics(self, session_id: str) -> Dict[str, Any]:
        """
        Get detailed statistics for a completed session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Dictionary with session statistics
        """
        session = await self.session_repository.get(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")
            
        # Get all questions for this session
        questions = []
        for question_id in session.questions:
            question = await self.question_repository.get(question_id)
            if question:
                questions.append(question)
                
        # Calculate pattern-specific statistics
        pattern_stats = {}
        for question in questions:
            # Extract pattern
            if question.candlestick_data:
                pattern = question.candlestick_data.target_pattern
                if pattern:
                    if pattern not in pattern_stats:
                        pattern_stats[pattern] = {"correct": 0, "total": 0, "time_ms": 0}
                    
                    pattern_stats[pattern]["total"] += 1
                    
                    # Check answer if available
                    answer = session.answers.get(question.question_id)
                    if answer:
                        if answer.get("is_correct", False):
                            pattern_stats[pattern]["correct"] += 1
                        if "time_taken_ms" in answer:
                            pattern_stats[pattern]["time_ms"] += answer["time_taken_ms"]
        
        # Calculate accuracies
        for pattern in pattern_stats:
            if pattern_stats[pattern]["total"] > 0:
                pattern_stats[pattern]["accuracy"] = pattern_stats[pattern]["correct"] / pattern_stats[pattern]["total"]
                if pattern_stats[pattern]["total"] > 0:
                    pattern_stats[pattern]["avg_time_ms"] = pattern_stats[pattern]["time_ms"] / pattern_stats[pattern]["total"]
        
        # Build statistics object
        statistics = {
            "session_id": session_id,
            "user_id": session.user_id,
            "created_at": session.created_at.isoformat(),
            "completed_at": session.completed_at.isoformat() if session.completed_at else None,
            "total_questions": len(session.questions),
            "questions_answered": len(session.answers),
            "correct_answers": sum(1 for a in session.answers.values() if a.get("is_correct", False)),
            "score": session.metrics.score if session.metrics else 0,
            "accuracy": session.metrics.accuracy if session.metrics else 0,
            "patterns": pattern_stats
        }
        
        return statistics
    
    async def get_pattern_explanation(self, pattern: str) -> Dict[str, Any]:
        """
        Get an explanation for a specific candlestick pattern.
        
        Args:
            pattern: Pattern name
            
        Returns:
            Dictionary with pattern explanation
        """
        return await default_candlestick_service.get_pattern_explanation(pattern)
    
    def _convert_difficulty_to_numeric(self, difficulty: str) -> float:
        """
        Convert a string difficulty level to a numeric value.
        
        Args:
            difficulty: Difficulty level (easy, medium, hard)
            
        Returns:
            Numeric difficulty value (0.0-1.0)
        """
        if difficulty.lower() == "easy":
            return 0.3
        elif difficulty.lower() == "medium":
            return 0.6
        elif difficulty.lower() == "hard":
            return 0.9
        else:
            return 0.5  # Default to medium


# Create singleton instance
candlestick_session_service = CandlestickSessionService() 