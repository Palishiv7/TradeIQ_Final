"""
Shared types and enums for the candlestick pattern assessment module.

This module contains type definitions that are used across multiple modules
to avoid circular dependencies.
"""

from enum import Enum
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class UserLevel(str, Enum):
    """User proficiency levels for explanation generation."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"

class ValidationResult(BaseModel):
    """Result of validating a user's answer."""
    is_correct: bool = Field(..., description="Whether the answer is correct")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in the validation")
    explanation: str = Field(..., description="Explanation of why the answer is correct/incorrect")
    feedback: Optional[str] = Field(None, description="Additional feedback for the user")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class CandlestickQuestion(BaseModel):
    """A candlestick pattern question."""
    question_id: str = Field(..., description="Unique identifier for the question")
    pattern_name: str = Field(..., description="Name of the candlestick pattern")
    chart_data: List[Dict[str, float]] = Field(..., description="OHLCV data for the chart")
    options: List[str] = Field(..., min_items=2, description="Possible answers")
    correct_option: str = Field(..., description="The correct answer")
    difficulty: float = Field(..., ge=0.0, le=1.0, description="Question difficulty from 0 to 1")
    explanation: Optional[str] = Field(None, description="Explanation of the pattern")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata") 