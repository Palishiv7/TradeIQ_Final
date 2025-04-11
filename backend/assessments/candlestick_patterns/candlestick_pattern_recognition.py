"""
Pattern Recognition Interface for Candlestick Assessments

This module provides an implementation of the base assessment interfaces
for the candlestick pattern recognition system. It connects the domain-driven design
architecture from base_assessment.py with the specialized candlestick pattern functionality.

It serves as the main integration point between the abstract assessment framework
and the concrete candlestick implementation.
"""

import time
import logging
import uuid
import weakref
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Set, Union, cast, Type, TypeVar

from backend.common.base_assessment import (
    Question, QuestionContent, Difficulty, AssessmentSession, 
    AssessmentMetrics, QuestionGeneratedEvent, AnswerSubmittedEvent,
    AnswerEvaluatedEvent, SessionCompletedEvent, EventDispatcher
)
from backend.common.logger import app_logger
from backend.common.finance.patterns import (
    PatternType, PatternStrength, CandlestickPattern, PatternRecognitionResult
)
from backend.common.serialization import SerializableMixin

# Import from new multi-strategy pattern detection system
from backend.assessments.candlestick_patterns.pattern_detection import (
    PatternDetector, PatternMatch, DetectionStrategy,
    get_default_detector
)
from backend.assessments.candlestick_patterns.answer_evaluation import (
    MultiTierValidationPipeline, ValidationResult
)
from backend.assessments.candlestick_patterns.candlestick_difficulty import (
    DifficultyManager
)
from backend.assessments.candlestick_patterns.candlestick_utils import (
    Candle, CandlestickData, generate_options
)
from backend.assessments.candlestick_patterns.adaptive_difficulty import (
    AdaptiveDifficultyEngine
)
from backend.assessments.candlestick_patterns.candlestick_config import (
    CANDLESTICK_PATTERNS, PATTERN_DESCRIPTIONS, DIFFICULTY_LEVELS
)

# Module logger
logger = app_logger.getChild("candlestick.pattern_recognition")

class CandlestickQuestionData(SerializableMixin):
    """Custom data structure for candlestick question data."""
    
    def __init__(
        self,
        candles: List[Candle],
        target_pattern: str,
        timeframe: str,
        symbol: str,
        chart_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize candlestick question data.
        
        Args:
            candles: List of candle data
            target_pattern: The pattern to identify
            timeframe: Timeframe of the chart (e.g., "1d", "4h")
            symbol: Stock/crypto symbol
            chart_config: Additional chart configuration
        """
        if not candles:
            raise ValueError("Candles list cannot be empty")
        if not target_pattern:
            raise ValueError("Target pattern cannot be empty")
            
        self.candles = candles
        self.target_pattern = target_pattern
        self.timeframe = timeframe
        self.symbol = symbol
        self.chart_config = chart_config or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "candles": [c.to_dict() for c in self.candles],
            "target_pattern": self.target_pattern,
            "timeframe": self.timeframe,
            "symbol": self.symbol,
            "chart_config": self.chart_config
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CandlestickQuestionData':
        """Create from dictionary."""
        try:
            return cls(
                candles=[Candle.from_dict(c) for c in data.get("candles", [])],
                target_pattern=data.get("target_pattern", ""),
                timeframe=data.get("timeframe", "1d"),
                symbol=data.get("symbol", ""),
                chart_config=data.get("chart_config", {})
            )
        except Exception as e:
            logger.error(f"Error creating CandlestickQuestionData from dict: {e}")
            # Create a minimal valid object
            return cls(
                candles=[Candle(timestamp=int(datetime.now().timestamp()), 
                               open=100.0, high=101.0, low=99.0, close=100.5, volume=1000)],
                target_pattern="Unknown",
                timeframe="1d",
                symbol="ERROR",
                chart_config={"error": str(e)}
            )
        
    @property
    def serializable_fields(self) -> List[str]:
        """Get a list of fields that should be serialized."""
        return ["candles", "target_pattern", "timeframe", "symbol", "chart_config"]
    
    def get_pattern_type(self) -> Optional[PatternType]:
        """Get the pattern type as a PatternType enum value."""
        try:
            return PatternType(self.target_pattern.lower())
        except (ValueError, AttributeError) as e:
            logger.warning(f"Could not convert {self.target_pattern} to PatternType: {e}")
            return None
            
    def get_recognized_patterns(self) -> List[PatternMatch]:
        """
        Analyze this candlestick data to get recognized patterns.
        
        Returns:
            List of recognized patterns
        """
        try:
            # Use the new multi-strategy pattern detection system with async support
            import asyncio
            from functools import lru_cache
            
            # Cache detector creation to improve performance
            @lru_cache(maxsize=1)
            async def get_cached_detector():
                return await get_default_detector()
            
            # Create candlestick data
            data = CandlestickData(
                symbol=self.symbol,
                timeframe=self.timeframe,
                candles=self.candles
            )
            
            # Use event loop to run the async detector
            loop = asyncio.get_event_loop()
            detector = loop.run_until_complete(get_cached_detector())
            
            # Detect patterns with proper error handling
            patterns, error = loop.run_until_complete(detector.detect_patterns_safe(data))
            
            if error:
                logger.warning(f"Pattern detection completed with warning: {error}")
                
            return patterns
        except Exception as e:
            logger.error(f"Error recognizing patterns: {e}", exc_info=True)
            return []


class CandlestickPatternQuestion(Question):
    """Candlestick pattern-specific question implementation."""
    
    def __init__(
        self,
        question_id: str,
        content: QuestionContent,
        difficulty: Difficulty,
        created_at: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
        candlestick_data: Optional[CandlestickQuestionData] = None
    ):
        """
        Initialize a candlestick pattern question.
        
        Args:
            question_id: Unique identifier for the question
            content: Question content (text, options, correct option)
            difficulty: Difficulty level
            created_at: Creation timestamp
            metadata: Additional metadata
            candlestick_data: Candlestick-specific data
        """
        if not question_id:
            question_id = str(uuid.uuid4())
            
        super().__init__(
            question_id=question_id,
            content=content,
            difficulty=difficulty,
            created_at=created_at or datetime.now(),
            metadata=metadata or {}
        )
        
        self._candlestick_data = candlestick_data
    
    @property
    def candlestick_data(self) -> Optional[CandlestickQuestionData]:
        """Get the candlestick data."""
        return self._candlestick_data
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = super().to_dict()
        data["candlestick_data"] = self._candlestick_data.to_dict() if self._candlestick_data else None
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CandlestickPatternQuestion':
        """Create from dictionary."""
        try:
            # Extract and convert candlestick data
            candlestick_data_dict = data.pop("candlestick_data", None)
            candlestick_data = None
            if candlestick_data_dict:
                candlestick_data = CandlestickQuestionData.from_dict(candlestick_data_dict)
            
            # Create QuestionContent from content dict
            content_dict = data.pop("content", {})
            content = QuestionContent(
                text=content_dict.get("text", ""),
                options=content_dict.get("options", []),
                correct_option=content_dict.get("correct_option", "")
            )
            
            # Create difficulty from difficulty dict
            difficulty_dict = data.pop("difficulty", {})
            difficulty = Difficulty(
                level=difficulty_dict.get("level", 0.5),
                category=difficulty_dict.get("category", "medium")
            )
            
            # Extract created_at
            created_at = None
            if "created_at" in data:
                try:
                    created_at = datetime.fromisoformat(data.pop("created_at"))
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid created_at format: {e}")
                    created_at = datetime.now()
            
            return cls(
                question_id=data.pop("question_id", str(uuid.uuid4())),
                content=content,
                difficulty=difficulty,
                created_at=created_at,
                metadata=data.pop("metadata", {}),
                candlestick_data=candlestick_data
            )
        except Exception as e:
            logger.error(f"Error creating CandlestickPatternQuestion from dict: {e}")
            # Create a minimal valid question
            return cls(
                question_id=str(uuid.uuid4()),
                content=QuestionContent(
                    text="Error creating question",
                    options=["Error"],
                    correct_option="Error"
                ),
                difficulty=Difficulty(level=0.5, category="medium"),
                metadata={"error": str(e)}
            )
    
    def get_pattern_type(self) -> Optional[PatternType]:
        """Get the target pattern type."""
        if self._candlestick_data:
            return self._candlestick_data.get_pattern_type()
        return None
    
    def get_pattern_explanation(self) -> Dict[str, Any]:
        """
        Get an explanation of the target pattern.
        
        Returns:
            Dictionary with pattern explanation details
        """
        try:
            if not self._candlestick_data:
                return {"error": "No candlestick data available"}
            
            pattern_name = self._candlestick_data.target_pattern
            pattern_type = self.get_pattern_type()
            
            # Get pattern description
            description = PATTERN_DESCRIPTIONS.get(pattern_name, "No description available")
            
            # Get pattern category (bullish/bearish)
            category = "neutral"
            if pattern_type:
                if pattern_type.is_bullish:
                    category = "bullish"
                elif pattern_type.is_bearish:
                    category = "bearish"
            
            # Get pattern complexity
            complexity = "simple"
            if pattern_type:
                num_candles = pattern_type.min_candles
                if num_candles >= 3:
                    complexity = "complex"
                elif num_candles == 2:
                    complexity = "intermediate"
            
            return {
                "pattern": pattern_name,
                "description": description,
                "category": category,
                "complexity": complexity,
                "pattern_type": pattern_type.value if pattern_type else None
            }
        except Exception as e:
            logger.error(f"Error generating pattern explanation: {e}")
            return {
                "pattern": self._candlestick_data.target_pattern if self._candlestick_data else "unknown",
                "description": "Error generating explanation",
                "category": "unknown",
                "complexity": "unknown",
                "error": str(e)
            }


# Type variable for service class
T = TypeVar('T', bound='CandlestickAssessmentService')

class CandlestickAssessmentService:
    """
    Service for generating and evaluating candlestick pattern assessments.
    
    This service orchestrates the interaction between pattern detection,
    question generation, and answer evaluation components.
    """
    
    # Singleton instance
    _instance = None
    
    def __init__(
        self,
        pattern_detector: Optional[PatternDetector] = None,
        validation_pipeline: Optional[MultiTierValidationPipeline] = None,
        difficulty_manager: Optional[DifficultyManager] = None,
        event_dispatcher: Optional[EventDispatcher] = None
    ):
        """
        Initialize the assessment service.
        
        Args:
            pattern_detector: Pattern detector for identifying patterns
            validation_pipeline: Pipeline for validating answers
            difficulty_manager: Manager for controlling question difficulty
            event_dispatcher: Event dispatcher for assessment events
        """
        # Use provided detector or create default
        self.pattern_detector = pattern_detector or get_default_detector()
        
        # Create validation pipeline if not provided
        self.validation_pipeline = validation_pipeline or MultiTierValidationPipeline()
        
        # Create difficulty manager if not provided
        self.difficulty_manager = difficulty_manager or DifficultyManager()
        
        # Create event dispatcher if not provided
        self.event_dispatcher = event_dispatcher or EventDispatcher()
        
        self.logger = app_logger.getChild("candlestick.assessment_service")
        
        logger.info("CandlestickAssessmentService initialized")
    
    @classmethod
    def get_instance(cls: Type[T]) -> T:
        """
        Get the singleton instance of the service.
        
        Returns:
            Singleton instance of CandlestickAssessmentService
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    async def generate_question(
        self,
        user_id: str,
        difficulty_level: Optional[float] = None,
        target_pattern: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> CandlestickPatternQuestion:
        """
        Generate a candlestick pattern question.
        
        Args:
            user_id: User ID for whom to generate the question
            difficulty_level: Optional difficulty level override
            target_pattern: Optional target pattern override
            session_id: Optional session ID for tracking
            
        Returns:
            Generated candlestick pattern question
        """
        start_time = time.time()
        
        try:
            # Use the difficulty manager to get appropriate candles and pattern
            difficulty = difficulty_level or 0.5
            difficulty = max(0.1, min(difficulty, 1.0))  # Ensure valid range
            
            if not target_pattern:
                target_pattern, candles, symbol, timeframe = self.difficulty_manager.get_question_data(difficulty)
            else:
                # If target pattern is provided, get appropriate candles for it
                candles, symbol, timeframe = self.difficulty_manager.get_candles_for_pattern(target_pattern, difficulty)
            
            # Create candlestick data
            candlestick_data = CandlestickQuestionData(
                candles=candles,
                target_pattern=target_pattern,
                timeframe=timeframe,
                symbol=symbol
            )
            
            # Generate question text
            question_text = self._generate_question_text(target_pattern, difficulty)
            
            # Generate options
            all_patterns = list(CANDLESTICK_PATTERNS.keys())
            options = generate_options(target_pattern, all_patterns, num_options=4, difficulty=difficulty)
            
            # Create question content
            content = QuestionContent(
                text=question_text,
                options=options,
                correct_option=target_pattern
            )
            
            # Create difficulty
            difficulty_obj = Difficulty(
                level=difficulty,
                category=DIFFICULTY_LEVELS.get(round(difficulty * 5) / 5, "medium")
            )
            
            # Create question
            question = CandlestickPatternQuestion(
                question_id=str(uuid.uuid4()),
                content=content,
                difficulty=difficulty_obj,
                created_at=datetime.now(),
                metadata={
                    "category": self._get_pattern_category(target_pattern),
                    "generation_time_ms": (time.time() - start_time) * 1000
                },
                candlestick_data=candlestick_data
            )
            
            # Dispatch question generated event
            if session_id:
                try:
                    await self.event_dispatcher.dispatch(QuestionGeneratedEvent(
                        session_id=session_id,
                        question_id=question.question_id,
                        user_id=user_id,
                        difficulty=difficulty_obj.level,
                        category=question.metadata.get("category", "unknown"),
                        timestamp=datetime.now()
                    ))
                except Exception as e:
                    logger.error(f"Error dispatching question generated event: {e}")
            
            return question
        except Exception as e:
            # Generate a fallback question in case of error
            logger.error(f"Error generating question: {e}")
            
            # Create a simple fallback question
            fallback_pattern = "Doji"
            fallback_options = ["Doji", "Hammer", "Engulfing", "Harami"]
            
            content = QuestionContent(
                text="Identify the candlestick pattern.",
                options=fallback_options,
                correct_option=fallback_pattern
            )
            
            difficulty_obj = Difficulty(
                level=0.5,
                category="medium"
            )
            
            return CandlestickPatternQuestion(
                question_id=str(uuid.uuid4()),
                content=content,
                difficulty=difficulty_obj,
                created_at=datetime.now(),
                metadata={
                    "category": "unknown",
                    "generation_time_ms": (time.time() - start_time) * 1000,
                    "error": str(e)
                }
            )
    
    async def evaluate_answer(
        self,
        question: CandlestickPatternQuestion,
        user_answer: str,
        user_id: str,
        session_id: str,
        time_spent_ms: int
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Evaluate a user's answer to a question.
        
        Args:
            question: The question being answered
            user_answer: The user's answer
            user_id: User ID of the answerer
            session_id: Session ID for tracking
            time_spent_ms: Time spent answering in milliseconds
            
        Returns:
            Tuple of (is_correct, score, feedback_data)
        """
        start_time = time.time()
        
        try:
            # Get correct answer
            correct_answer = question.content.correct_option
            
            # Check if simple equality is correct
            is_correct = user_answer == correct_answer
            
            # Calculate base score (0-1)
            base_score = 1.0 if is_correct else 0.0
            
            # Run validation pipeline for more sophisticated scoring and feedback
            feedback = {}
            validation_result = None
            
            if question.candlestick_data:
                try:
                    # Get the candlestick data for validation
                    candlestick_data = CandlestickData(
                        symbol=question.candlestick_data.symbol,
                        timeframe=question.candlestick_data.timeframe,
                        candles=question.candlestick_data.candles
                    )
                    
                    # Run validation
                    validation_result = self.validation_pipeline.validate(
                        candlestick_data=candlestick_data,
                        user_answer=user_answer
                    )
                    
                    # Update score and correctness based on validation
                    if validation_result.confidence > 0.8 and not is_correct:
                        # User might be correct even if not exact match
                        is_correct = True
                        base_score = validation_result.confidence
                    elif is_correct and validation_result.confidence < 0.5:
                        # Validation suggests user might be incorrect despite exact match
                        is_correct = False
                        base_score = validation_result.confidence
                    
                    # Add validation feedback
                    feedback["validation_tier"] = validation_result.validation_tier.value
                    feedback["confidence"] = validation_result.confidence
                    feedback["alternative_patterns"] = validation_result.alternative_patterns
                except Exception as e:
                    logger.error(f"Error during answer validation: {e}")
                    feedback["validation_error"] = str(e)
            
            # Scale score by difficulty
            final_score = base_score * (1.0 + (question.difficulty.level * 0.5))
            final_score = max(0.0, min(final_score, 1.0))  # Ensure score is between 0 and 1
            
            # Add evaluation metadata
            feedback.update({
                "is_correct": is_correct,
                "correct_answer": correct_answer,
                "user_answer": user_answer,
                "base_score": base_score,
                "final_score": final_score,
                "difficulty": question.difficulty.level,
                "time_spent_ms": time_spent_ms,
                "evaluation_time_ms": (time.time() - start_time) * 1000,
                "pattern_explanation": question.get_pattern_explanation()
            })
            
            # Dispatch events
            try:
                await self.event_dispatcher.dispatch(AnswerSubmittedEvent(
                    session_id=session_id,
                    question_id=question.question_id,
                    user_id=user_id,
                    answer=user_answer,
                    timestamp=datetime.now(),
                    time_spent_ms=time_spent_ms
                ))
                
                await self.event_dispatcher.dispatch(AnswerEvaluatedEvent(
                    session_id=session_id,
                    question_id=question.question_id,
                    user_id=user_id,
                    is_correct=is_correct,
                    score=final_score,
                    timestamp=datetime.now()
                ))
            except Exception as e:
                logger.error(f"Error dispatching events: {e}")
                feedback["event_dispatch_error"] = str(e)
            
            return is_correct, final_score, feedback
        except Exception as e:
            logger.error(f"Error evaluating answer: {e}")
            return False, 0.0, {
                "error": str(e),
                "is_correct": False,
                "correct_answer": question.content.correct_option if hasattr(question, 'content') else "unknown",
                "user_answer": user_answer,
                "evaluation_time_ms": (time.time() - start_time) * 1000
            }
    
    def _get_pattern_category(self, pattern: str) -> str:
        """
        Get the category of a pattern.
        
        Args:
            pattern: Pattern name
            
        Returns:
            Category name (e.g., "reversal", "continuation")
        """
        try:
            # Try to get from pattern type
            pattern_type = PatternType(pattern.lower())
            
            if pattern_type.is_reversal:
                return "reversal"
            else:
                return "continuation"
        except (ValueError, AttributeError):
            # Fallback based on name
            pattern_lower = pattern.lower()
            
            if any(term in pattern_lower for term in ["engulfing", "hammer", "star", "harami", "piercing"]):
                return "reversal"
            elif any(term in pattern_lower for term in ["doji", "marubozu", "soldiers", "crows", "gap"]):
                return "continuation"
            else:
                return "unknown"
    
    def _generate_question_text(self, pattern: str, difficulty: float) -> str:
        """
        Generate appropriate question text based on pattern and difficulty.
        
        Args:
            pattern: Target pattern name
            difficulty: Difficulty level (0-1)
            
        Returns:
            Question text
        """
        try:
            if difficulty < 0.3:
                return "Identify the candlestick pattern shown in this chart."
            elif difficulty < 0.6:
                return "What candlestick pattern is formed in the highlighted area of this chart?"
            else:
                category = self._get_pattern_category(pattern)
                if category == "reversal":
                    return "Identify the reversal pattern in this price chart."
                else:
                    return "Identify the continuation pattern shown in this candlestick chart."
        except Exception as e:
            logger.error(f"Error generating question text: {e}")
            return "Identify the candlestick pattern in this chart."


# Singleton instance for common use
default_candlestick_service = CandlestickAssessmentService.get_instance() 