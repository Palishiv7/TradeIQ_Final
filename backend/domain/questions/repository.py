"""
Question Repository Module

This module defines the repository interfaces and implementations for
accessing and storing Question entities.
"""

import abc
import logging
from typing import Dict, List, Optional, Protocol, Tuple

from backend.common.cache import CacheService, KeyBuilder, get_cache_service
from .model import Question, Difficulty

# Setup logging
logger = logging.getLogger(__name__)


class QuestionRepository(abc.ABC):
    """
    Abstract base class for question repositories.
    
    This interface defines the contract for accessing and storing Question entities.
    """
    
    @abc.abstractmethod
    async def get_by_id(self, question_id: str) -> Optional[Question]:
        """
        Get a question by its ID.
        
        Args:
            question_id: The ID of the question to retrieve
            
        Returns:
            The Question entity if found, None otherwise
        """
        pass
    
    @abc.abstractmethod
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
        pass
    
    @abc.abstractmethod
    async def delete(self, question_id: str) -> bool:
        """
        Delete a question by its ID.
        
        Args:
            question_id: The ID of the question to delete
            
        Returns:
            True if the question was deleted, False otherwise
        """
        pass
    
    @abc.abstractmethod
    async def find_by_category(self, category: str, limit: int = 10) -> List[Question]:
        """
        Find questions by category.
        
        Args:
            category: The category to search for
            limit: Maximum number of questions to return
            
        Returns:
            List of matching Question entities
        """
        pass
    
    @abc.abstractmethod
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
        pass


class CachedQuestionRepository(QuestionRepository):
    """
    Cached implementation of the QuestionRepository.
    
    This implementation adds caching capabilities to an existing repository
    implementation (delegate), using the TradeIQ caching system.
    """
    
    def __init__(
        self, 
        delegate: QuestionRepository, 
        cache_service: Optional[CacheService] = None,
        namespace: str = "questions"
    ):
        """
        Initialize the cached repository.
        
        Args:
            delegate: The repository implementation to delegate to
            cache_service: Optional cache service to use, or None to use the default
            namespace: Cache namespace for question keys
        """
        self.delegate = delegate
        self.cache = cache_service or get_cache_service()
        self.namespace = namespace
    
    async def get_by_id(self, question_id: str) -> Optional[Question]:
        """
        Get a question by its ID, with caching.
        
        Args:
            question_id: The ID of the question to retrieve
            
        Returns:
            The Question entity if found, None otherwise
        """
        # Build cache key
        cache_key = KeyBuilder.entity_key(
            entity_type="question", 
            entity_id=question_id, 
            namespace=self.namespace
        )
        
        # Try to get from cache
        question_dict = await self.cache.get(cache_key)
        if question_dict:
            logger.debug(f"Cache hit for question {question_id}")
            return Question.from_dict(question_dict)
        
        # Cache miss, get from delegate
        logger.debug(f"Cache miss for question {question_id}")
        question = await self.delegate.get_by_id(question_id)
        
        # Cache the result, if found
        if question:
            await self.cache.set(cache_key, question.to_dict(), ttl=3600)  # 1 hour TTL
        
        return question
    
    async def save(self, question: Question) -> Question:
        """
        Save a question and update or invalidate cache.
        
        Args:
            question: The Question entity to save
            
        Returns:
            The saved Question entity
        """
        # Save to delegate repository
        saved_question = await self.delegate.save(question)
        
        # Update cache with the saved question
        cache_key = KeyBuilder.entity_key(
            entity_type="question", 
            entity_id=saved_question.question_id, 
            namespace=self.namespace
        )
        await self.cache.set(cache_key, saved_question.to_dict(), ttl=3600)  # 1 hour TTL
        
        # Invalidate related collection caches
        # This ensures collections containing this question will be refreshed
        # on next access, reflecting the updated question
        category_filter = ("category", saved_question.category)
        category_key = KeyBuilder.collection_key(
            entity_type="question",
            namespace=self.namespace,
            filters=[category_filter]
        )
        await self.cache.delete(category_key)
        
        difficulty_filter = ("difficulty", f"{saved_question.difficulty.value:.1f}")
        difficulty_key = KeyBuilder.collection_key(
            entity_type="question",
            namespace=self.namespace,
            filters=[difficulty_filter]
        )
        await self.cache.delete(difficulty_key)
        
        return saved_question
    
    async def delete(self, question_id: str) -> bool:
        """
        Delete a question and invalidate cache.
        
        Args:
            question_id: The ID of the question to delete
            
        Returns:
            True if the question was deleted, False otherwise
        """
        # Get the question first to know its category and difficulty
        # for cache invalidation
        question = await self.get_by_id(question_id)
        
        # Delete from delegate repository
        result = await self.delegate.delete(question_id)
        
        if result:
            # Delete from cache
            cache_key = KeyBuilder.entity_key(
                entity_type="question", 
                entity_id=question_id, 
                namespace=self.namespace
            )
            await self.cache.delete(cache_key)
            
            # Invalidate related collection caches if we have the question metadata
            if question:
                category_filter = ("category", question.category)
                category_key = KeyBuilder.collection_key(
                    entity_type="question",
                    namespace=self.namespace,
                    filters=[category_filter]
                )
                await self.cache.delete(category_key)
                
                difficulty_filter = ("difficulty", f"{question.difficulty.value:.1f}")
                difficulty_key = KeyBuilder.collection_key(
                    entity_type="question",
                    namespace=self.namespace,
                    filters=[difficulty_filter]
                )
                await self.cache.delete(difficulty_key)
        
        return result
    
    async def find_by_category(self, category: str, limit: int = 10) -> List[Question]:
        """
        Find questions by category, with caching.
        
        Args:
            category: The category to search for
            limit: Maximum number of questions to return
            
        Returns:
            List of matching Question entities
        """
        # Build cache key
        category_filter = ("category", category)
        cache_key = KeyBuilder.collection_key(
            entity_type="question",
            namespace=self.namespace,
            filters=[category_filter],
            limit=limit
        )
        
        # Try to get from cache
        cached_result = await self.cache.get(cache_key)
        if cached_result:
            logger.debug(f"Cache hit for questions in category '{category}'")
            return [Question.from_dict(item) for item in cached_result]
        
        # Cache miss, get from delegate
        logger.debug(f"Cache miss for questions in category '{category}'")
        questions = await self.delegate.find_by_category(category, limit)
        
        # Cache the result
        if questions:
            questions_dict = [q.to_dict() for q in questions]
            await self.cache.set(cache_key, questions_dict, ttl=600)  # 10 minutes TTL
        
        return questions
    
    async def find_by_difficulty_range(
        self, 
        min_difficulty: float, 
        max_difficulty: float,
        limit: int = 10
    ) -> List[Question]:
        """
        Find questions within a difficulty range, with caching.
        
        Args:
            min_difficulty: Minimum difficulty value (inclusive)
            max_difficulty: Maximum difficulty value (inclusive)
            limit: Maximum number of questions to return
            
        Returns:
            List of matching Question entities
        """
        # Build cache key
        difficulty_filter = ("difficulty_range", f"{min_difficulty:.1f}-{max_difficulty:.1f}")
        cache_key = KeyBuilder.collection_key(
            entity_type="question",
            namespace=self.namespace,
            filters=[difficulty_filter],
            limit=limit
        )
        
        # Try to get from cache
        cached_result = await self.cache.get(cache_key)
        if cached_result:
            logger.debug(f"Cache hit for questions in difficulty range {min_difficulty}-{max_difficulty}")
            return [Question.from_dict(item) for item in cached_result]
        
        # Cache miss, get from delegate
        logger.debug(f"Cache miss for questions in difficulty range {min_difficulty}-{max_difficulty}")
        questions = await self.delegate.find_by_difficulty_range(
            min_difficulty, max_difficulty, limit
        )
        
        # Cache the result
        if questions:
            questions_dict = [q.to_dict() for q in questions]
            await self.cache.set(cache_key, questions_dict, ttl=600)  # 10 minutes TTL
        
        return questions 