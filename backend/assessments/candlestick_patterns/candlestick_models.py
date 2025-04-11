"""
Candlestick Pattern Assessment Models

This module defines the data models for candlestick pattern assessments,
extending the base assessment models with pattern-specific fields and methods.
"""

import uuid
import datetime
import statistics
import logging
from typing import Dict, List, Any, Optional, Union, Tuple, ClassVar, cast, Set
from dataclasses import dataclass, field
from functools import lru_cache

from sqlalchemy import Column, String, JSON, DateTime, Integer, ForeignKey, Table
from sqlalchemy.orm import relationship, declared_attr

from backend.assessments.base.models import (
    BaseQuestion,
    AssessmentSession,
    QuestionDifficulty,
    AnswerEvaluation,
    AssessmentType,
    SerializableMixin,
    SessionStatus,
    AssessmentMetrics
)
from backend.common.finance.patterns import PatternType, PatternStrength
from backend.assessments.candlestick_patterns.types import UserLevel, CandlestickQuestion as BaseCandlestickQuestion
from backend.common.logger import app_logger
from backend.database.base import Base

# Setup module logger
logger = app_logger.getChild("candlestick_models")

# Performance-related constants
MAX_CACHED_PATTERN_DESCRIPTIONS: int = 128
MAX_RESPONSE_TIME_SAMPLES: int = 100
EXPLANATION_TRUNCATION_LENGTH: int = 1000
SIMILARITY_CUTOFF: float = 0.7


class CandlestickQuestion(BaseQuestion):
    """
    Data model for candlestick pattern questions.
    
    This class extends BaseQuestion with pattern-specific fields and methods.
    """
    __tablename__ = 'candlestick_questions'
    
    # Primary key
    id = Column(String(36), ForeignKey('questions.id'), primary_key=True)
    
    # Required candlestick-specific fields
    pattern = Column(String(50), nullable=False)
    pattern_strength = Column(String(20), nullable=False)
    chart_data = Column(JSON, nullable=False, default=dict)
    chart_image = Column(String(500), nullable=True)
    timeframe = Column(String(20), nullable=True)
    symbol = Column(String(20), nullable=True)
    
    # Additional optional fields
    options = Column(JSON, nullable=False, default=list)
    explanation = Column(String(2000), nullable=True)
    feedback = Column(String(1000), nullable=True)
    hints = Column(JSON, nullable=False, default=list)
    
    __mapper_args__ = {
        'polymorphic_identity': 'candlestick_question',
    }

    def __init__(self, **kwargs):
        """Initialize a candlestick question."""
        # Extract candlestick-specific fields
        candlestick_fields = {
            'pattern': kwargs.pop('pattern', None),
            'pattern_strength': kwargs.pop('pattern_strength', None),
            'chart_data': kwargs.pop('chart_data', {}),
            'chart_image': kwargs.pop('chart_image', None),
            'timeframe': kwargs.pop('timeframe', None),
            'symbol': kwargs.pop('symbol', None),
            'options': kwargs.pop('options', []),
            'explanation': kwargs.pop('explanation', None),
            'feedback': kwargs.pop('feedback', None),
            'hints': kwargs.pop('hints', [])
        }
        
        # Initialize base class with remaining kwargs
        super().__init__(**kwargs)
        
        # Set candlestick-specific fields
        for field, value in candlestick_fields.items():
            setattr(self, field, value)
        
        # Validate required fields
        if not self.pattern:
            raise ValueError("Pattern is required")
        if not self.pattern_strength:
            raise ValueError("Pattern strength is required")
            
        # Validate pattern is a known pattern
        from backend.assessments.candlestick_patterns.candlestick_config import get_patterns_all
        valid_patterns = get_patterns_all()
        if self.pattern not in valid_patterns:
            logger.warning(f"Invalid pattern: {self.pattern}")
            closest_match = self._find_closest_match(self.pattern, valid_patterns)
            if closest_match:
                logger.info(f"Using closest match: {closest_match}")
                self.pattern = closest_match
            else:
                raise ValueError(f"Invalid pattern: {self.pattern}")
    
    def _find_closest_match(self, pattern: str, valid_patterns: List[str]) -> Optional[str]:
        """Find the closest matching valid pattern."""
        pattern = pattern.lower()
        for valid_pattern in valid_patterns:
            if valid_pattern.lower() == pattern:
                return valid_pattern
        return None

    @property
    def pattern(self) -> str:
        """Get the pattern."""
        return self._pattern

    @pattern.setter
    def pattern(self, value: str) -> None:
        """Set the pattern."""
        object.__setattr__(self, '_pattern', value)

    @property
    def pattern_strength(self) -> PatternStrength:
        """Get the pattern strength."""
        return self._pattern_strength

    @pattern_strength.setter
    def pattern_strength(self, value: PatternStrength) -> None:
        """Set the pattern strength."""
        object.__setattr__(self, '_pattern_strength', value)

    @property
    def chart_data(self) -> Dict[str, Any]:
        """Get the chart data."""
        return self._chart_data

    @chart_data.setter
    def chart_data(self, value: Dict[str, Any]) -> None:
        """Set the chart data."""
        object.__setattr__(self, '_chart_data', value)

    @property
    def chart_image(self) -> str:
        """Get the chart image."""
        return self._chart_image

    @chart_image.setter
    def chart_image(self, value: str) -> None:
        """Set the chart image."""
        object.__setattr__(self, '_chart_image', value)

    @property
    def timeframe(self) -> str:
        """Get the timeframe."""
        return self._timeframe

    @timeframe.setter
    def timeframe(self, value: str) -> None:
        """Set the timeframe."""
        object.__setattr__(self, '_timeframe', value)

    @property
    def symbol(self) -> str:
        """Get the symbol."""
        return self._symbol

    @symbol.setter
    def symbol(self, value: str) -> None:
        """Set the symbol."""
        object.__setattr__(self, '_symbol', value)

    @property
    def id(self) -> str:
        """Get the id."""
        return self._id

    @id.setter
    def id(self, value: str) -> None:
        """Set the id."""
        object.__setattr__(self, '_id', value)

    @property
    def question_id(self) -> str:
        """Get the question id (alias for id)."""
        return self.id

    @question_id.setter
    def question_id(self, value: str) -> None:
        """Set the question id (alias for id)."""
    @property
    def question_type(self) -> str:
        """Get the question type."""
        return self._question_type

    @question_type.setter
    def question_type(self, value: str) -> None:
        """Set the question type."""
        object.__setattr__(self, '_question_type', value)

    @property
    def question_text(self) -> str:
        """Get the question text."""
        return self._question_text

    @question_text.setter
    def question_text(self, value: str) -> None:
        """Set the question text."""
        object.__setattr__(self, '_question_text', value)

    @property
    def difficulty(self) -> QuestionDifficulty:
        """Get the difficulty."""
        return self._difficulty

    @difficulty.setter
    def difficulty(self, value: QuestionDifficulty) -> None:
        """Set the difficulty."""
        object.__setattr__(self, '_difficulty', value)

    @property
    def topics(self) -> List[str]:
        """Get the topics list."""
        return self._topics

    @topics.setter
    def topics(self, value: List[str]) -> None:
        """Set the topics list."""
        object.__setattr__(self, '_topics', value)

    @property
    def subtopics(self) -> Optional[List[str]]:
        """Get the subtopics list."""
        return self._subtopics

    @subtopics.setter
    def subtopics(self, value: Optional[List[str]]) -> None:
        """Set the subtopics list."""
        object.__setattr__(self, '_subtopics', value)

    @property
    def created_at(self) -> datetime.datetime:
        """Get the created_at datetime."""
        return self._created_at

    @created_at.setter
    def created_at(self, value: datetime.datetime) -> None:
        """Set the created_at datetime."""
        object.__setattr__(self, '_created_at', value)

    @property
    def metadata(self) -> Dict[str, Any]:
        """Get the metadata dictionary."""
        return self._metadata

    @metadata.setter
    def metadata(self, value: Dict[str, Any]) -> None:
        """Set the metadata dictionary."""
        object.__setattr__(self, '_metadata', value)

    @property
    def answer_options(self) -> Optional[Dict[str, Any]]:
        """Get the answer options dictionary."""
        return self._answer_options

    @answer_options.setter
    def answer_options(self, value: Optional[Dict[str, Any]]) -> None:
        """Set the answer options dictionary."""
        object.__setattr__(self, '_answer_options', value)

    @property
    def options(self) -> List[str]:
        """Get the options list."""
        return self._options

    @options.setter
    def options(self, value: List[str]) -> None:
        """Set the options list."""
        object.__setattr__(self, '_options', value)

    @property
    def explanation(self) -> str:
        """Get the explanation text."""
        return self._explanation

    @explanation.setter
    def explanation(self, value: str) -> None:
        """Set the explanation text."""
        object.__setattr__(self, '_explanation', value)

    @property
    def feedback(self) -> str:
        """Get the feedback text."""
        return self._feedback

    @feedback.setter
    def feedback(self, value: str) -> None:
        """Set the feedback text."""
        object.__setattr__(self, '_feedback', value)

    @property
    def hints(self) -> List[str]:
        """Get the hints list."""
        return self._hints

    @hints.setter
    def hints(self, value: List[str]) -> None:
        """Set the hints list."""
        object.__setattr__(self, '_hints', value)
    
    def get_correct_answer(self) -> str:
        """
        Get the correct answer for this question.
        
        Returns:
            Correct pattern name
        """
        return self.pattern
    
    def evaluate_answer(self, answer: str) -> AnswerEvaluation:
        """
        Evaluate a user's answer to this question.
        
        Args:
            answer: User's answer
            
        Returns:
            Answer evaluation
        """
        try:
            # Normalize answers for comparison
            normalized_answer = str(answer).strip().upper() if answer else ""
            normalized_correct = self.get_correct_answer().strip().upper()
            
            # Determine if correct
            is_correct = normalized_answer == normalized_correct
            
            # Calculate score (100 if correct, 0 if incorrect)
            score = 100 if is_correct else 0
            
            # Get pattern type
            pattern_type = PatternType[self.pattern]
            
            # Create feedback based on correctness
            if is_correct:
                feedback = f"Correct! This is a {self.pattern.replace('_', ' ').title()} pattern."
            else:
                feedback = f"Incorrect. This is a {self.pattern.replace('_', ' ').title()} pattern."
            
            # Generate explanation
            explanation = self._generate_explanation(normalized_answer, normalized_correct)
            
            # Create evaluation
            evaluation = AnswerEvaluation(
                is_correct=is_correct,
                score=score,
                feedback=feedback,
                explanation=explanation,
                metadata={
                    "pattern": self.pattern,
                    "pattern_strength": self.pattern_strength.value,
                    "user_answer": answer,
                    "correct_answer": self.get_correct_answer(),
                    "is_bullish": pattern_type.is_bullish(),
                    "is_bearish": pattern_type.is_bearish(),
                    "timeframe": self.timeframe,
                    "symbol": self.symbol
                }
            )
            
            return evaluation
        except Exception as e:
            logger.error(f"Error evaluating answer: {str(e)}")
            # Return a default evaluation in case of error
            return AnswerEvaluation(
                is_correct=False,
                score=0,
                feedback=f"Error evaluating answer: {self.pattern}",
                explanation=f"An error occurred while evaluating your answer. The correct pattern is {self.pattern.replace('_', ' ').title()}."
            )
    
    def _generate_explanation(self, user_answer: str, correct_answer: str) -> str:
        """
        Generate an explanation for the question.
        
        Args:
            user_answer: User's answer
            correct_answer: Correct answer
            
        Returns:
            Explanation text
        """
        try:
            # Get pattern type
            pattern_type = PatternType[self.pattern]
            
            # Format pattern name
            pattern_name = self.pattern.replace("_", " ").title()
            
            # Create explanation
            explanation = f"The {pattern_name} pattern is a {pattern_type.value.lower()} candlestick pattern."
            
            # Add signal type
            if pattern_type.is_bullish():
                explanation += " It is considered a bullish signal, suggesting a potential upward price movement."
            elif pattern_type.is_bearish():
                explanation += " It is considered a bearish signal, suggesting a potential downward price movement."
            else:
                explanation += " It is considered a neutral pattern, suggesting potential market indecision."
            
            # Add pattern description based on pattern type
            if "DOJI" in self.pattern:
                explanation += " A Doji forms when the opening and closing prices are nearly equal, creating a small body with upper and lower shadows. It suggests market indecision."
            elif "HAMMER" in self.pattern:
                explanation += " The Hammer has a small body at the top with a long lower shadow. It suggests the market tested lower prices but then rejected them, closing near the high."
            elif "ENGULFING" in self.pattern:
                explanation += " This pattern consists of two candles, where the second completely engulfs the body of the first, signaling a strong reversal in price direction."
            elif "STAR" in self.pattern:
                explanation += " This is a three-candle pattern that signals a potential reversal in price direction after a strong trend."
            
            return explanation
        except Exception as e:
            logger.error(f"Error generating explanation: {str(e)}")
            return f"The {self.pattern.replace('_', ' ').title()} pattern is a candlestick pattern that can indicate potential market movement."
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary.
        
        Returns:
            Dictionary representation
        """
        try:
            base_dict = super().to_dict()
            
            # Add pattern-specific fields
            base_dict.update({
                "pattern": self.pattern,
                "pattern_strength": self.pattern_strength.value,
                "chart_data": self.chart_data,
                "chart_image": self.chart_image,
                "timeframe": self.timeframe,
                "symbol": self.symbol,
                "options": self.options
            })
            
            return base_dict
        except Exception as e:
            logger.error(f"Error converting question to dictionary: {str(e)}")
            # Return a minimal valid dictionary
            return {
                "id": self.id,
                "question_type": "candlestick_pattern",
                "pattern": self.pattern
            }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CandlestickQuestion':
        """
        Create from dictionary.
        
        Args:
            data: Dictionary representation
            
        Returns:
            CandlestickQuestion instance
        """
        try:
            # Validate required fields
            if not data or "pattern" not in data:
                raise ValueError("Missing required fields: pattern")
                
            # Convert pattern strength
            pattern_strength = PatternStrength(data["pattern_strength"]) if "pattern_strength" in data else PatternStrength.MEDIUM
            
            # Create instance
            return cls(
                id=data.get("id", str(uuid.uuid4())),
                question_type=data.get("question_type", "candlestick_pattern"),
                question_text=data.get("question_text", "Identify the candlestick pattern:"),
                difficulty=QuestionDifficulty(data["difficulty"]) if "difficulty" in data else QuestionDifficulty.MEDIUM,
                topics=data.get("topics", []),
                pattern=data["pattern"],
                pattern_strength=pattern_strength,
                chart_data=data.get("chart_data", {}),
                chart_image=data.get("chart_image", ""),
                timeframe=data.get("timeframe", ""),
                symbol=data.get("symbol", ""),
                options=data.get("options", []),
                metadata=data.get("metadata", {})
            )
        except Exception as e:
            logger.error(f"Error creating question from dictionary: {str(e)}")
            raise ValueError(f"Cannot create question from data: {str(e)}")
    
    @staticmethod
    @lru_cache(maxsize=MAX_CACHED_PATTERN_DESCRIPTIONS)
    def get_pattern_description(pattern: str, level: str = "intermediate") -> str:
        """
        Get a description of a candlestick pattern.
        
        Args:
            pattern: Pattern name
            level: User knowledge level
            
        Returns:
            Pattern description
        """
        # Validate pattern
        if not hasattr(PatternType, pattern):
            return "Unknown pattern"
        
        # Format pattern name
        pattern_name = pattern.replace("_", " ").title()
        
        # Get pattern type
        pattern_type = PatternType[pattern]
        
        # Determine signal type
        if pattern_type.is_bullish():
            signal_type = "bullish"
        elif pattern_type.is_bearish():
            signal_type = "bearish"
        else:
            signal_type = "neutral"
        
        # Create description based on level
        if level == "beginner":
            return f"{pattern_name} is a {signal_type} candlestick pattern that can indicate a possible price reversal."
        elif level == "advanced":
            if "DOJI" in pattern:
                return f"{pattern_name} is a {signal_type} candlestick pattern characterized by a small body where opening and closing prices are nearly equal, with upper and lower shadows. It represents market indecision and is often found at key support or resistance levels, potentially signaling a trend reversal or continuation based on context."
            elif "HAMMER" in pattern:
                return f"{pattern_name} is a {signal_type} candlestick pattern with a small body at the top and a long lower shadow, typically twice the length of the body. It signals that sellers drove prices down during the session but were ultimately overcome by buyers, indicating potential bullish sentiment."
            elif "ENGULFING" in pattern:
                return f"{pattern_name} is a {signal_type} two-candle reversal pattern where the second candle completely 'engulfs' the body of the first candle. It represents a shift in market sentiment with increased trading volume, often occuring at key support or resistance levels."
            elif "STAR" in pattern:
                return f"{pattern_name} is a {signal_type} three-candle pattern that signals a potential reversal. It starts with a strong trend candle, followed by a gap and a small-bodied candle (the star), and concludes with a confirming candle in the new direction. It represents a significant shift in market psychology."
            else:
                return f"{pattern_name} is a {signal_type} candlestick pattern that indicates a potential reversal in price direction. The pattern's reliability increases when it forms at key support/resistance levels or after extended trends."
        else:  # intermediate (default)
            if "DOJI" in pattern:
                return f"{pattern_name} is a {signal_type} candlestick pattern with a small body where opening and closing prices are nearly equal. It represents market indecision and potential trend reversals."
            elif "HAMMER" in pattern:
                return f"{pattern_name} is a {signal_type} candlestick pattern with a small body at the top and a long lower shadow. It signals that sellers drove prices down but were ultimately overcome by buyers."
            elif "ENGULFING" in pattern:
                return f"{pattern_name} is a {signal_type} two-candle pattern where the second candle completely 'engulfs' the body of the first. It indicates a potential reversal in the current trend."
            elif "STAR" in pattern:
                return f"{pattern_name} is a {signal_type} three-candle pattern that signals a potential reversal. It starts with a trend candle, followed by a gap and a small-bodied candle, and concludes with a confirming candle."
            else:
                return f"{pattern_name} is a {signal_type} candlestick pattern that indicates a potential reversal in price direction."

    @property
    def content(self) -> dict:
        """
        Get content dictionary for repository serialization.
        This property maps individual attributes to a content dictionary
        expected by the ORM model.
        """
        return {
            "text": self.question_text,
            "options": self.options,
            "chart_data": self.chart_data,
            "chart_image": self.chart_image,
            "timeframe": self.timeframe,
            "symbol": self.symbol,
            "metadata": self.metadata
        }
        
    @content.setter
    def content(self, value: dict) -> None:
        """
        Setter for content property - ignores the input.
        This is needed because the repository might try to set this property,
        but we generate content dynamically from individual fields.
        """
        # We don't actually set content since it's derived from other properties
        # This is just a dummy setter to avoid the "no setter" error
        pass


@dataclass
class CandlestickSession(AssessmentSession):
    """
    Session model for candlestick pattern assessments.
    
    This class extends AssessmentSession with fields specific to candlestick patterns,
    such as patterns identified and response time tracking.
    
    Attributes:
        patterns_identified: List of candlestick patterns the user has correctly identified
        average_response_time_ms: Average time taken to answer questions in milliseconds
        streak: Current streak of correct answers
        max_streak: Maximum streak of correct answers achieved in this session
        performance_by_pattern: Dictionary mapping pattern names to performance metrics
    """
    # Define slots for memory optimization - use private names
    __slots__ = (
        '_patterns_identified',
        '_average_response_time_ms',
        '_streak',
        '_max_streak',
        '_performance_by_pattern'
    )
    
    patterns_identified: List[str] = field(default_factory=list)
    average_response_time_ms: float = 0.0
    streak: int = 0
    max_streak: int = 0
    performance_by_pattern: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Initialize the session after creation with validation."""
        super().__post_init__()
        
        # Initialize private fields
        object.__setattr__(self, '_patterns_identified', self.patterns_identified)
        object.__setattr__(self, '_average_response_time_ms', self.average_response_time_ms)
        object.__setattr__(self, '_streak', self.streak)
        object.__setattr__(self, '_max_streak', self.max_streak)
        object.__setattr__(self, '_performance_by_pattern', self.performance_by_pattern)
        
        # Ensure assessment type is correct
        if self.assessment_type != AssessmentType.CANDLESTICK:
            self.assessment_type = AssessmentType.CANDLESTICK
            
        # Validate patterns identified is a list
        if not isinstance(self._patterns_identified, list):
            object.__setattr__(self, '_patterns_identified', [])
            
        # Validate streak is non-negative
        if self._streak < 0:
            object.__setattr__(self, '_streak', 0)
            
        # Validate max_streak is non-negative
        if self._max_streak < 0:
            object.__setattr__(self, '_max_streak', 0)
            
        # Validate performance_by_pattern is a dictionary
        if not isinstance(self._performance_by_pattern, dict):
            object.__setattr__(self, '_performance_by_pattern', {})
    
    @property
    def record_answer(
        self,
        question_id: str,
        answer_value: Any,
        time_taken_ms: Optional[int] = None,
        evaluation: Optional[AnswerEvaluation] = None
    ) -> None:
        """
        Record a user's answer to a question and update session metrics.
        
        Args:
            question_id: Question identifier
            answer_value: User's answer
            time_taken_ms: Time taken to answer in milliseconds
            evaluation: Answer evaluation with correctness information
        
        Raises:
            ValueError: If question_id is empty
        """
        if not question_id:
            raise ValueError("Question ID cannot be empty")
            
        try:
            # Record answer in base class
            super().record_answer(question_id, answer_value, time_taken_ms, evaluation)
            
            # Update candlestick-specific fields
            if evaluation and evaluation.is_correct:
                # Add pattern to identified patterns
                pattern = evaluation.metadata.get("pattern")
                if pattern and pattern not in self.patterns_identified:
                    self.patterns_identified.append(pattern)
                
                # Increment streak
                self.streak += 1
                
                # Update max streak if needed
                if self.streak > self.max_streak:
                    self.max_streak = self.streak
            else:
                # Reset streak
                self.streak = 0
            
            # Update average response time
            self._update_average_response_time()
            
            # Update pattern-specific performance
            if evaluation and evaluation.metadata and "pattern" in evaluation.metadata:
                self._update_pattern_performance(
                    evaluation.metadata["pattern"], 
                    evaluation.is_correct, 
                    time_taken_ms
                )
                
        except Exception as e:
            logger.error(f"Error recording answer: {str(e)}")
            # Ensure session state remains valid even after error
            if question_id in self.answers:
                self._update_average_response_time()
    
    def _update_average_response_time(self) -> None:
        """
        Update the average response time based on all recorded answers.
        Ignores answers without time data.
        """
        try:
            response_times = [
                answer.time_taken_ms 
                for answer in self.answers.values() 
                if answer and answer.time_taken_ms is not None and answer.time_taken_ms > 0
            ]
            
            if response_times:
                self.average_response_time_ms = statistics.mean(response_times)
            else:
                self.average_response_time_ms = 0.0
        except Exception as e:
            logger.error(f"Error updating average response time: {str(e)}")
            # Keep previous value in case of error
    
    def _update_pattern_performance(self, pattern: str, is_correct: bool, time_taken_ms: Optional[int]) -> None:
        """
        Update performance metrics for a specific pattern.
        
        Args:
            pattern: Pattern name
            is_correct: Whether the answer was correct
            time_taken_ms: Time taken to answer
        """
        if not pattern:
            return
            
        try:
            # Initialize pattern performance if not exists
            if pattern not in self.performance_by_pattern:
                self.performance_by_pattern[pattern] = {
                    "attempts": 0,
                    "correct": 0,
                    "accuracy": 0.0,
                    "response_times": []
                }
                
            # Update metrics
            perf = self.performance_by_pattern[pattern]
            perf["attempts"] += 1
            
            if is_correct:
                perf["correct"] += 1
                
            perf["accuracy"] = perf["correct"] / perf["attempts"]
            
            if time_taken_ms is not None and time_taken_ms > 0:
                # Limit the number of stored response times to prevent memory growth
                if len(perf["response_times"]) >= MAX_RESPONSE_TIME_SAMPLES:
                    perf["response_times"].pop(0)
                
                perf["response_times"].append(time_taken_ms)
                perf["avg_response_time"] = statistics.mean(perf["response_times"]) if perf["response_times"] else 0
        except Exception as e:
            logger.error(f"Error updating pattern performance for {pattern}: {str(e)}")
    
    def get_performance(self) -> AssessmentMetrics:
        """
        Calculate performance metrics for this session.
        
        Returns:
            An AssessmentMetrics object summarizing performance.
        """
        try:
            total_q = self.question_count
            answered_q = len(self.answers)
            correct_q = sum(1 for ans in self.answers.values() if ans and ans.evaluation and ans.evaluation.is_correct)
            
            # Calculate total score (simple accuracy for now, could be based on evaluation.score)
            total_score = (correct_q / answered_q * 100) if answered_q > 0 else 0
            
            # Difficulty could be averaged from question metadata if available, or use session setting
            session_difficulty_str = self.settings.get('difficulty', QuestionDifficulty.MEDIUM.value)
            try:
                session_difficulty_enum = QuestionDifficulty(session_difficulty_str)
                difficulty_level_numeric = session_difficulty_enum.to_numeric()
            except ValueError:
                difficulty_level_numeric = QuestionDifficulty.MEDIUM.to_numeric()
                
            # Calculate additional metrics
            metrics = AssessmentMetrics(
                total_questions=total_q,
                answered_questions=answered_q,
                correct_answers=correct_q,
                average_time_ms=self.average_response_time_ms,
                total_score=round(total_score, 2),
                difficulty_level=difficulty_level_numeric / 5.0, # Normalize to 0.0-1.0 scale
                metadata={
                    "streak": self.streak,
                    "max_streak": self.max_streak,
                    "patterns_identified": len(self.patterns_identified),
                    "patterns_by_difficulty": self._get_patterns_by_difficulty()
                }
            )
            
            return metrics
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {str(e)}")
            # Return minimal valid metrics
            return AssessmentMetrics(
                total_questions=self.question_count,
                answered_questions=len(self.answers),
                correct_answers=0,
                average_time_ms=0,
                total_score=0,
                difficulty_level=0.5
            )
    
    def _get_patterns_by_difficulty(self) -> Dict[str, List[str]]:
        """
        Group identified patterns by difficulty level based on performance.
        
        Returns:
            Dictionary mapping difficulty levels to pattern lists
        """
        result = {
            "easy": [],
            "medium": [],
            "hard": []
        }
        
        try:
            for pattern, perf in self.performance_by_pattern.items():
                if perf["attempts"] < 2:
                    continue  # Not enough data
                    
                accuracy = perf["accuracy"]
                
                if accuracy >= 0.8:
                    result["easy"].append(pattern)
                elif accuracy >= 0.5:
                    result["medium"].append(pattern)
                else:
                    result["hard"].append(pattern)
        except Exception as e:
            logger.error(f"Error grouping patterns by difficulty: {str(e)}")
            
        return result
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert session to dictionary representation.
        
        Returns:
            Dictionary representation of the session
        """
        try:
            base_dict = super().to_dict()
            
            # Add pattern-specific fields
            base_dict.update({
                "patterns_identified": self.patterns_identified,
                "average_response_time_ms": self.average_response_time_ms,
                "streak": self.streak,
                "max_streak": self.max_streak,
                "performance_by_pattern": self.performance_by_pattern
            })
            
            return base_dict
        except Exception as e:
            logger.error(f"Error converting session to dictionary: {str(e)}")
            # Return minimal valid dictionary
            return {
                "id": self.id,
                "user_id": self.user_id,
                "assessment_type": self.assessment_type.value if hasattr(self.assessment_type, "value") else str(self.assessment_type),
                "patterns_identified": self.patterns_identified,
                "streak": self.streak
            }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CandlestickSession':
        """
        Create session from dictionary representation.
        
        Args:
            data: Dictionary representation
            
        Returns:
            CandlestickSession instance
        
        Raises:
            ValueError: If essential data is missing
        """
        if not data:
            raise ValueError("Cannot create session from empty data")
            
        try:
            # Preserve base session attributes
            session = super().from_dict(data)
            
            # Extract candlestick-specific attributes with validation
            patterns_identified = data.get("patterns_identified", [])
            if not isinstance(patterns_identified, list):
                patterns_identified = []
                
            streak = data.get("streak", 0)
            if not isinstance(streak, int) or streak < 0:
                streak = 0
                
            max_streak = data.get("max_streak", 0)
            if not isinstance(max_streak, int) or max_streak < 0:
                max_streak = streak if streak > 0 else 0
                
            performance_by_pattern = data.get("performance_by_pattern", {})
            if not isinstance(performance_by_pattern, dict):
                performance_by_pattern = {}
                
            # Create instance with pattern-specific attributes
            return cls(
                id=session.id,
                user_id=session.user_id,
                assessment_type=session.assessment_type,
                status=session.status,
                questions=session.questions,
                answers=session.answers,
                created_at=session.created_at,
                updated_at=session.updated_at,
                completed_at=session.completed_at,
                score=session.score,
                max_score=session.max_score,
                metadata=session.metadata,
                settings=session.settings,
                current_question_index=session.current_question_index,
                is_completed=session.is_completed,
                patterns_identified=patterns_identified,
                average_response_time_ms=data.get("average_response_time_ms", 0.0),
                streak=streak,
                max_streak=max_streak,
                performance_by_pattern=performance_by_pattern
            )
        except Exception as e:
            logger.error(f"Error creating session from dictionary: {str(e)}")
            # Create minimal valid object
            return cls(
                id=data.get("id", str(uuid.uuid4())),
                user_id=data.get("user_id", ""),
                assessment_type=AssessmentType.CANDLESTICK
            )


@dataclass
class CandlestickAssessmentResponse(SerializableMixin):
    """
    Response model for candlestick pattern assessment API endpoints.
    
    This class encapsulates the structured response for candlestick pattern assessments,
    including correctness, feedback, explanations, and pattern information.
    
    Attributes:
        is_correct: Whether the user's answer was correct
        score: Points awarded for this answer
        feedback: Feedback message for the user
        explanation: Detailed explanation of the pattern
        pattern: Name of the candlestick pattern
        pattern_description: Description of the pattern's characteristics
        assessment_complete: Whether the assessment is complete
        next_question: Data for the next question (if available)
        performance: Performance metrics for the assessment (optional)
    """
    # Define slots for memory optimization - use private names
    __slots__ = (
        '_is_correct',
        '_score',
        '_feedback',
        '_explanation',
        '_pattern',
        '_pattern_description',
        '_assessment_complete',
        '_next_question',
        '_performance'
    )
    
    is_correct: bool = field(default=False)
    score: int = field(default=0)
    feedback: str = field(default="")
    explanation: str = field(default="")
    pattern: Optional[str] = field(default=None)
    pattern_description: Optional[str] = field(default=None)
    assessment_complete: bool = field(default=False)
    next_question: Optional[Dict[str, Any]] = field(default=None)
    performance: Optional[Dict[str, Any]] = field(default=None)
    
    def __post_init__(self) -> None:
        """Validate fields after initialization."""
        # Initialize private fields
        object.__setattr__(self, '_is_correct', self.is_correct)
        object.__setattr__(self, '_score', self.score)
        object.__setattr__(self, '_feedback', self.feedback)
        object.__setattr__(self, '_explanation', self.explanation)
        object.__setattr__(self, '_pattern', self.pattern)
        object.__setattr__(self, '_pattern_description', self.pattern_description)
        object.__setattr__(self, '_assessment_complete', self.assessment_complete)
        object.__setattr__(self, '_next_question', self.next_question)
        object.__setattr__(self, '_performance', self.performance)
        
        # Ensure score is non-negative
        if self._score < 0:
            object.__setattr__(self, '_score', 0)
            
        # Ensure string fields are strings
        if not isinstance(self._feedback, str):
            object.__setattr__(self, '_feedback', str(self._feedback) if self._feedback is not None else "")
            
        if not isinstance(self._explanation, str):
            object.__setattr__(self, '_explanation', str(self._explanation) if self._explanation is not None else "")
            
        # Ensure next_question is either None or a dictionary
        if self._next_question is not None and not isinstance(self._next_question, dict):
            object.__setattr__(self, '_next_question', None)
            
        # Ensure performance is either None or a dictionary
        if self._performance is not None and not isinstance(self._performance, dict):
            object.__setattr__(self, '_performance', None)
            
        # Truncate long explanations to conserve memory
        if len(self._explanation) > EXPLANATION_TRUNCATION_LENGTH:
            object.__setattr__(self, '_explanation', self._explanation[:EXPLANATION_TRUNCATION_LENGTH-3] + "...")

    @property
    def is_correct(self) -> bool:
        """Get whether the answer was correct."""
        return self._is_correct

    @is_correct.setter
    def is_correct(self, value: bool) -> None:
        """Set whether the answer was correct."""
        object.__setattr__(self, '_is_correct', value)

    @property
    def score(self) -> int:
        """Get the score."""
        return self._score

    @score.setter
    def score(self, value: int) -> None:
        """Set the score."""
        object.__setattr__(self, '_score', value)

    @property
    def feedback(self) -> str:
        """Get the feedback."""
        return self._feedback

    @feedback.setter
    def feedback(self, value: str) -> None:
        """Set the feedback."""
        object.__setattr__(self, '_feedback', value)

    @property
    def explanation(self) -> str:
        """Get the explanation."""
        return self._explanation

    @explanation.setter
    def explanation(self, value: str) -> None:
        """Set the explanation."""
        object.__setattr__(self, '_explanation', value)

    @property
    def pattern(self) -> Optional[str]:
        """Get the pattern."""
        return self._pattern

    @pattern.setter
    def pattern(self, value: Optional[str]) -> None:
        """Set the pattern."""
        object.__setattr__(self, '_pattern', value)

    @property
    def pattern_description(self) -> Optional[str]:
        """Get the pattern description."""
        return self._pattern_description

    @pattern_description.setter
    def pattern_description(self, value: Optional[str]) -> None:
        """Set the pattern description."""
        object.__setattr__(self, '_pattern_description', value)

    @property
    def assessment_complete(self) -> bool:
        """Get whether the assessment is complete."""
        return self._assessment_complete

    @assessment_complete.setter
    def assessment_complete(self, value: bool) -> None:
        """Set whether the assessment is complete."""
        object.__setattr__(self, '_assessment_complete', value)

    @property
    def next_question(self) -> Optional[Dict[str, Any]]:
        """Get the next question data."""
        return self._next_question

    @next_question.setter
    def next_question(self, value: Optional[Dict[str, Any]]) -> None:
        """Set the next question data."""
        object.__setattr__(self, '_next_question', value)

    @property
    def performance(self) -> Optional[Dict[str, Any]]:
        """Get the performance metrics."""
        return self._performance

    @performance.setter
    def performance(self, value: Optional[Dict[str, Any]]) -> None:
        """Set the performance metrics."""
        object.__setattr__(self, '_performance', value)

    @classmethod
    def from_evaluation(
        cls,
        evaluation: AnswerEvaluation,
        pattern: str,
        assessment_complete: bool = False,
        next_question: Optional[Dict[str, Any]] = None,
        performance: Optional[Dict[str, Any]] = None,
        user_level: str = "intermediate"
    ) -> 'CandlestickAssessmentResponse':
        """
        Create response from answer evaluation.
        
        Args:
            evaluation: Answer evaluation containing correctness and feedback
            pattern: Pattern name identified in the question
            assessment_complete: Whether the assessment is complete
            next_question: Data for the next question (if available)
            performance: Performance metrics for the session (optional)
            user_level: User knowledge level for pattern descriptions
            
        Returns:
            CandlestickAssessmentResponse instance
        
        Raises:
            ValueError: If evaluation is None
        """
        if not evaluation:
            raise ValueError("Cannot create response without evaluation")
            
        try:
            # Get pattern description with fallback for unknown patterns
            pattern_description = None
            if pattern:
                try:
                    pattern_description = CandlestickQuestion.get_pattern_description(pattern, level=user_level)
                except Exception as e:
                    logger.warning(f"Error getting pattern description for {pattern}: {str(e)}")
                    pattern_description = f"A candlestick pattern that indicates potential market movement."
            
            # Clean up next question data to remove unnecessary fields
            cleaned_next_question = None
            if next_question:
                # Keep only essential fields to reduce response size
                cleaned_next_question = {
                    k: v for k, v in next_question.items() 
                    if k in ["id", "question_text", "chart_image", "options", "difficulty"]
                }
            
            # Clean up performance data for frontend consumption
            cleaned_performance = None
            if performance:
                cleaned_performance = {
                    "total_score": performance.get("total_score", 0),
                    "accuracy": performance.get("accuracy", 0),
                    "streak": performance.get("streak", 0),
                    "max_streak": performance.get("max_streak", 0),
                    "patterns_identified": performance.get("patterns_identified", 0)
                }
            
            return cls(
                is_correct=evaluation.is_correct,
                score=max(0, evaluation.score),  # Ensure non-negative score
                feedback=evaluation.feedback or "",
                explanation=evaluation.explanation or "",
                pattern=pattern,
                pattern_description=pattern_description,
                assessment_complete=assessment_complete,
                next_question=cleaned_next_question,
                performance=cleaned_performance
            )
        except Exception as e:
            logger.error(f"Error creating assessment response: {str(e)}")
            # Return minimal valid response
            return cls(
                is_correct=getattr(evaluation, 'is_correct', False),
                score=max(0, getattr(evaluation, 'score', 0)),
                pattern=pattern,
                assessment_complete=assessment_complete
            )
            
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert response to dictionary representation.
        
        Returns:
            Dictionary representation of the response
        """
        try:
            result = super().to_dict()
            
            # Ensure all fields are serializable
            if self.next_question is None:
                result["next_question"] = None
                
            if self.performance is None:
                result["performance"] = None
                
            return result
        except Exception as e:
            logger.error(f"Error converting response to dictionary: {str(e)}")
            # Return minimal valid dictionary
            return {
                "is_correct": self.is_correct,
                "score": self.score,
                "feedback": self.feedback,
                "pattern": self.pattern,
                "assessment_complete": self.assessment_complete
            }
            
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CandlestickAssessmentResponse':
        """
        Create response from dictionary representation.
        
        Args:
            data: Dictionary representation
            
        Returns:
            CandlestickAssessmentResponse instance
        """
        if not data:
            return cls()
            
        try:
            return cls(
                is_correct=data.get("is_correct", False),
                score=max(0, int(data.get("score", 0))),
                feedback=data.get("feedback", ""),
                explanation=data.get("explanation", ""),
                pattern=data.get("pattern"),
                pattern_description=data.get("pattern_description"),
                assessment_complete=data.get("assessment_complete", False),
                next_question=data.get("next_question"),
                performance=data.get("performance")
            )
        except Exception as e:
            logger.error(f"Error creating response from dictionary: {str(e)}")
            return cls()


@dataclass
class CandlestickPatternData:
    """
    Data model for candlestick pattern information.
    
    This class encapsulates all the relevant information about a candlestick pattern,
    including its characteristics, signals, and metadata.
    """
    
    name: str
    description: str
    signal_type: str  # "bullish", "bearish", or "neutral"
    reliability: float  # 0.0 to 1.0
    timeframes: List[str]  # List of timeframes where pattern is effective
    characteristics: List[str]  # Key characteristics of the pattern
    confirmation_signals: List[str]  # Additional signals that confirm the pattern
    typical_trend: str  # "reversal" or "continuation"
    min_candles: int  # Minimum number of candles needed to form pattern
    volume_requirement: Optional[str] = None  # Volume characteristics if relevant
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate fields after initialization."""
        # Validate signal type
        if self.signal_type not in ["bullish", "bearish", "neutral"]:
            raise ValueError("Signal type must be 'bullish', 'bearish', or 'neutral'")
            
        # Validate reliability
        if not 0.0 <= self.reliability <= 1.0:
            raise ValueError("Reliability must be between 0.0 and 1.0")
            
        # Validate timeframes
        if not self.timeframes:
            raise ValueError("Must specify at least one timeframe")
            
        # Validate characteristics
        if not self.characteristics:
            raise ValueError("Must specify at least one characteristic")
            
        # Validate min_candles
        if self.min_candles < 1:
            raise ValueError("Minimum candles must be at least 1")
            
        # Validate typical trend
        if self.typical_trend not in ["reversal", "continuation"]:
            raise ValueError("Typical trend must be 'reversal' or 'continuation'")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "description": self.description,
            "signal_type": self.signal_type,
            "reliability": self.reliability,
            "timeframes": self.timeframes,
            "characteristics": self.characteristics,
            "confirmation_signals": self.confirmation_signals,
            "typical_trend": self.typical_trend,
            "min_candles": self.min_candles,
            "volume_requirement": self.volume_requirement,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CandlestickPatternData':
        """Create from dictionary representation."""
        return cls(
            name=data["name"],
            description=data["description"],
            signal_type=data["signal_type"],
            reliability=data["reliability"],
            timeframes=data["timeframes"],
            characteristics=data["characteristics"],
            confirmation_signals=data["confirmation_signals"],
            typical_trend=data["typical_trend"],
            min_candles=data["min_candles"],
            volume_requirement=data.get("volume_requirement"),
            metadata=data.get("metadata", {})
        )


@dataclass
class CandlestickPattern:
    """
    Base class for candlestick pattern definitions.
    
    This class represents a candlestick pattern with its properties and methods
    for pattern recognition and analysis.
    """
    
    name: str
    pattern_type: PatternType
    strength: PatternStrength = field(default=PatternStrength.MODERATE)
    description: str = field(default="")
    characteristics: List[str] = field(default_factory=list)
    requirements: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate and initialize pattern after creation."""
        # Validate name
        if not self.name:
            raise ValueError("Pattern name cannot be empty")
            
        # Validate pattern type
        if not isinstance(self.pattern_type, PatternType):
            try:
                self.pattern_type = PatternType[str(self.pattern_type)]
            except (KeyError, ValueError):
                raise ValueError(f"Invalid pattern type: {self.pattern_type}")
                
        # Validate strength
        if not isinstance(self.strength, PatternStrength):
            try:
                self.strength = PatternStrength[str(self.strength)]
            except (KeyError, ValueError):
                self.strength = PatternStrength.MODERATE
                
        # Ensure lists and dicts are initialized
        if not isinstance(self.characteristics, list):
            self.characteristics = []
            
        if not isinstance(self.requirements, dict):
            self.requirements = {}
            
        if not isinstance(self.metadata, dict):
            self.metadata = {}
    
    def is_bullish(self) -> bool:
        """Check if this is a bullish pattern."""
        return self.pattern_type.is_bullish()
    
    def is_bearish(self) -> bool:
        """Check if this is a bearish pattern."""
        return self.pattern_type.is_bearish()
    
    def is_reversal(self) -> bool:
        """Check if this is a reversal pattern."""
        return self.metadata.get("is_reversal", False)
    
    def is_continuation(self) -> bool:
        """Check if this is a continuation pattern."""
        return self.metadata.get("is_continuation", False)
    
    def get_reliability_score(self) -> float:
        """
        Get the reliability score of this pattern.
        
        Returns:
            Float between 0.0 and 1.0 indicating reliability
        """
        base_score = self.strength.to_numeric() / 5.0  # Convert 1-5 scale to 0.0-1.0
        
        # Adjust based on metadata if available
        if "reliability_score" in self.metadata:
            return (base_score + float(self.metadata["reliability_score"])) / 2
            
        return base_score
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert pattern to dictionary representation."""
        return {
            "name": self.name,
            "pattern_type": self.pattern_type.name,
            "strength": self.strength.value,
            "description": self.description,
            "characteristics": self.characteristics,
            "requirements": self.requirements,
            "metadata": self.metadata,
            "is_bullish": self.is_bullish(),
            "is_bearish": self.is_bearish(),
            "is_reversal": self.is_reversal(),
            "is_continuation": self.is_continuation(),
            "reliability_score": self.get_reliability_score()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CandlestickPattern':
        """Create pattern from dictionary representation."""
        if not data or "name" not in data or "pattern_type" not in data:
            raise ValueError("Missing required fields: name, pattern_type")
            
        return cls(
            name=data["name"],
            pattern_type=data["pattern_type"],
            strength=data.get("strength", PatternStrength.MODERATE),
            description=data.get("description", ""),
            characteristics=data.get("characteristics", []),
            requirements=data.get("requirements", {}),
            metadata=data.get("metadata", {})
        ) 