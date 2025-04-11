"""
Memory Question Repository Module

This module provides an in-memory implementation of the QuestionRepository
interface for development and testing purposes.
"""

import asyncio
import logging
from typing import Dict, List, Optional

from .model import Question, Difficulty
from .repository import QuestionRepository

# Setup logging
logger = logging.getLogger(__name__)


class MemoryQuestionRepository(QuestionRepository):
    """
    In-memory implementation of the QuestionRepository.
    
    This implementation stores questions in memory and is intended for
    development and testing purposes only.
    """
    
    def __init__(self, initial_data: Optional[List[Question]] = None):
        """
        Initialize the repository with optional initial data.
        
        Args:
            initial_data: Optional list of Question entities to initialize with
        """
        self._questions: Dict[str, Question] = {}
        
        # Add initial data if provided
        if initial_data:
            for question in initial_data:
                self._questions[question.question_id] = question
    
    async def get_by_id(self, question_id: str) -> Optional[Question]:
        """
        Get a question by its ID.
        
        Args:
            question_id: The ID of the question to retrieve
            
        Returns:
            The Question entity if found, None otherwise
        """
        return self._questions.get(question_id)
    
    async def save(self, question: Question) -> Question:
        """
        Save a question.
        
        If the question doesn't exist, it will be created.
        If it already exists, it will be updated.
        
        Args:
            question: The Question entity to save
            
        Returns:
            The saved Question entity
        """
        self._questions[question.question_id] = question
        return question
    
    async def delete(self, question_id: str) -> bool:
        """
        Delete a question by its ID.
        
        Args:
            question_id: The ID of the question to delete
            
        Returns:
            True if the question was deleted, False otherwise
        """
        if question_id in self._questions:
            del self._questions[question_id]
            return True
        return False
    
    async def find_by_category(self, category: str, limit: int = 10) -> List[Question]:
        """
        Find questions by category.
        
        Args:
            category: The category to search for
            limit: Maximum number of questions to return
            
        Returns:
            List of matching Question entities
        """
        result = [
            question for question in self._questions.values()
            if question.category == category
        ]
        return result[:limit]
    
    async def find_by_difficulty_range(
        self, 
        min_difficulty: float, 
        max_difficulty: float,
        limit: int = 10
    ) -> List[Question]:
        """
        Find questions within a difficulty range.
        
        Args:
            min_difficulty: Minimum difficulty value (inclusive)
            max_difficulty: Maximum difficulty value (inclusive)
            limit: Maximum number of questions to return
            
        Returns:
            List of matching Question entities
        """
        result = [
            question for question in self._questions.values()
            if min_difficulty <= question.difficulty.value <= max_difficulty
        ]
        return result[:limit]
    
    def get_all(self) -> List[Question]:
        """
        Get all questions.
        
        This method is specific to the memory implementation and not part of
        the QuestionRepository interface.
        
        Returns:
            List of all Question entities
        """
        return list(self._questions.values())
    
    def clear(self) -> None:
        """
        Clear all questions.
        
        This method is specific to the memory implementation and not part of
        the QuestionRepository interface.
        """
        self._questions.clear() 