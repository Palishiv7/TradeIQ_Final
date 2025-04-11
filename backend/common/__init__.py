"""
Common Components for TradeIQ

This package contains common functionality and infrastructure that is shared
across different modules and features of the TradeIQ application.

Key components:
1. Cache Infrastructure - Flexible caching layer with memory and Redis backends
2. Assessment Framework - Base components for assessment systems
3. Logging - Centralized logging configuration
4. Error Handling - Common error handling utilities
5. Validation - Input validation and sanitization
"""

# Initialize logging
from backend.common.logger import app_logger

# Export base assessment components
from backend.common.base_assessment import (
    Question, QuestionContent, Difficulty, AssessmentSession, 
    AssessmentMetrics, EventDispatcher, QuestionGeneratedEvent,
    AnswerSubmittedEvent, AnswerEvaluatedEvent, SessionCompletedEvent
)

# Export caching components
from backend.common.cache import (
    CacheBackend, CacheKey, CacheEntry, CacheResult,
    MemoryCache, RedisCache, get_cache, get_cache_manager,
    configure_cache, configure_cache_sync, clear_all_caches
)

# Export assessment service components
from backend.common.assessment_service import (
    BaseSessionRepository, BaseQuestionRepository, AssessmentService,
    QuestionGenerator, AnswerEvaluator, ExplanationGenerator, PerformanceAnalyzer
)

# Export assessment repository components
from backend.common.assessment_repository import (
    Repository, CachingRepository, QuestionRepository, SessionRepository
)

__all__ = [
    # Logging
    'app_logger',
    
    # Base assessment
    'Question', 'QuestionContent', 'Difficulty', 'AssessmentSession',
    'AssessmentMetrics', 'EventDispatcher', 'QuestionGeneratedEvent',
    'AnswerSubmittedEvent', 'AnswerEvaluatedEvent', 'SessionCompletedEvent',
    
    # Caching
    'CacheBackend', 'CacheKey', 'CacheEntry', 'CacheResult',
    'MemoryCache', 'RedisCache', 'get_cache', 'get_cache_manager',
    'configure_cache', 'configure_cache_sync', 'clear_all_caches',
    
    # Assessment services
    'BaseSessionRepository', 'BaseQuestionRepository', 'AssessmentService',
    'QuestionGenerator', 'AnswerEvaluator', 'ExplanationGenerator', 'PerformanceAnalyzer',
    
    # Assessment repositories
    'Repository', 'CachingRepository', 'QuestionRepository', 'SessionRepository',
]

"""
Common utilities and shared functionality for the TradeIQ assessment platform.
""" 