"""
User Performance Tracking and Optimization

This package provides functionality for tracking user performance metrics,
adaptive difficulty adjustments, spaced repetition scheduling, and 
persistent storage of performance data.
"""

# Performance tracking components
from backend.common.performance.tracker import (
    PerformanceTracker,
    PerformanceMetrics,
    SessionStats,
    TopicPerformance,
    SkillLevel,
    LearningRate,
    create_tracker
)

# Adaptive difficulty components
from backend.common.performance.difficulty import (
    DifficultyLevel,
    DifficultyManager,
    DifficultyAdjustment,
    AdjustmentStrategy,
    AdaptiveDifficultyEngine
)

# Forgetting curve and spaced repetition components
from backend.common.performance.forgetting import (
    MemoryState,
    ForgettingCurveModel,
    ReviewItem,
    SpacedRepetitionScheduler
)

# Repository components for persistence
from backend.common.performance.repository import (
    PerformanceRepository,
    CachePerformanceRepository,
    TrackerRepository,
    DifficultyRepository,
    SpacedRepetitionRepository,
    tracker_repository,
    difficulty_repository,
    spaced_repetition_repository,
    create_performance_repository
)

# Define public API
__all__ = [
    # Performance tracking
    'PerformanceTracker',
    'PerformanceMetrics',
    'SessionStats',
    'TopicPerformance',
    'SkillLevel',
    'LearningRate',
    'create_tracker',
    
    # Adaptive difficulty
    'DifficultyLevel',
    'DifficultyManager',
    'DifficultyAdjustment',
    'AdjustmentStrategy',
    'AdaptiveDifficultyEngine',
    
    # Forgetting curve and spaced repetition
    'MemoryState',
    'ForgettingCurveModel',
    'ReviewItem',
    'SpacedRepetitionScheduler',
    
    # Repository components
    'PerformanceRepository',
    'CachePerformanceRepository',
    'TrackerRepository',
    'DifficultyRepository',
    'SpacedRepetitionRepository',
    'tracker_repository',
    'difficulty_repository',
    'spaced_repetition_repository',
    'create_performance_repository'
] 