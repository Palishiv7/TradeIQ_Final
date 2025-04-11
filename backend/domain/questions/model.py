"""
Question Domain Model Module

This module defines the core domain entities for the question subsystem.
"""

import enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
import uuid


class Difficulty(enum.Enum):
    """
    Enum representing the difficulty level of a question.
    
    Values range from 1.0 (easiest) to 5.0 (hardest).
    """
    BEGINNER = 1.0
    EASY = 2.0
    MEDIUM = 3.0
    HARD = 4.0
    EXPERT = 5.0


@dataclass
class Question:
    """
    Represents a question in the TradeIQ assessment system.
    
    Attributes:
        question_id: Unique identifier for the question
        text: The question text
        options: List of possible answer options
        correct_answer: The index of the correct answer in the options list
        difficulty: The difficulty level of the question
        category: The category or topic of the question
        tags: List of tags associated with the question
        created_at: When the question was created
        updated_at: When the question was last updated
        metadata: Additional metadata about the question
    """
    question_id: str
    text: str
    options: List[str]
    correct_answer: int
    difficulty: Difficulty
    category: str
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def create(cls, 
              text: str,
              options: List[str],
              correct_answer: int,
              difficulty: Difficulty,
              category: str,
              tags: Optional[List[str]] = None,
              metadata: Optional[Dict[str, Any]] = None) -> 'Question':
        """
        Create a new question with a generated ID.
        
        Args:
            text: The question text
            options: List of possible answer options
            correct_answer: The index of the correct answer in the options list
            difficulty: The difficulty level of the question
            category: The category or topic of the question
            tags: List of tags associated with the question (optional)
            metadata: Additional metadata about the question (optional)
            
        Returns:
            A new Question instance
        """
        return cls(
            question_id=str(uuid.uuid4()),
            text=text,
            options=options,
            correct_answer=correct_answer,
            difficulty=difficulty,
            category=category,
            tags=tags or [],
            metadata=metadata or {}
        )
    
    def update(self, 
              text: Optional[str] = None,
              options: Optional[List[str]] = None,
              correct_answer: Optional[int] = None,
              difficulty: Optional[Difficulty] = None,
              category: Optional[str] = None,
              tags: Optional[List[str]] = None,
              metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Update the question's attributes.
        
        Args:
            text: The new question text (optional)
            options: New list of possible answer options (optional)
            correct_answer: New index of the correct answer (optional)
            difficulty: New difficulty level (optional)
            category: New category (optional)
            tags: New list of tags (optional)
            metadata: New metadata (optional)
        """
        if text is not None:
            self.text = text
        if options is not None:
            self.options = options
        if correct_answer is not None:
            self.correct_answer = correct_answer
        if difficulty is not None:
            self.difficulty = difficulty
        if category is not None:
            self.category = category
        if tags is not None:
            self.tags = tags
        if metadata is not None:
            self.metadata.update(metadata)
            
        self.updated_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the question to a dictionary.
        
        Returns:
            Dictionary representation of the question
        """
        return {
            'question_id': self.question_id,
            'text': self.text,
            'options': self.options,
            'correct_answer': self.correct_answer,
            'difficulty': self.difficulty.value,
            'category': self.category,
            'tags': self.tags,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Question':
        """
        Create a Question from a dictionary.
        
        Args:
            data: Dictionary containing question data
            
        Returns:
            A Question instance
        """
        # Convert difficulty value to enum
        difficulty_value = data.get('difficulty')
        difficulty = next((d for d in Difficulty if d.value == difficulty_value), Difficulty.MEDIUM)
        
        # Convert ISO datetime strings to datetime objects
        created_at = datetime.fromisoformat(data.get('created_at')) if data.get('created_at') else datetime.now()
        updated_at = datetime.fromisoformat(data.get('updated_at')) if data.get('updated_at') else datetime.now()
        
        return cls(
            question_id=data.get('question_id'),
            text=data.get('text'),
            options=data.get('options', []),
            correct_answer=data.get('correct_answer', 0),
            difficulty=difficulty,
            category=data.get('category', ''),
            tags=data.get('tags', []),
            created_at=created_at,
            updated_at=updated_at,
            metadata=data.get('metadata', {})
        ) 