"""
Candlestick Pattern Assessment Module

This module implements the candlestick pattern assessment functionality for the TradeIQ platform,
allowing users to test their ability to identify and understand candlestick patterns.
"""

from backend.assessments.candlestick_patterns.candlestick_controller import router
from backend.assessments.candlestick_patterns.candlestick_api import router as legacy_router
from backend.assessments.candlestick_patterns.candlestick_models import (
    CandlestickQuestion,
    CandlestickSession,
    CandlestickAssessmentResponse
)
from backend.assessments.candlestick_patterns.candlestick_service import CandlestickAssessmentService
from backend.assessments.candlestick_patterns.candlestick_repository import (
    CandlestickQuestionRepository,
    CandlestickSessionRepository,
    CandlestickAssessmentRepositoryImpl
)
from backend.assessments.candlestick_patterns.candlestick_explanation_generator import (
    ExplanationGenerator,
    UserLevel
)
from backend.assessments.candlestick_patterns.candlestick_utils import (
    CandlestickData,
    Candle,
    Difficulty,
    plot_candlestick_chart,
    get_patterns_by_difficulty,
    generate_options,
    get_pattern_category,
    get_pattern_description,
    format_pattern_name,
    convert_to_candlestick_pattern
)

def register_routes(app):
    """
    Register the candlestick assessment API routes with the FastAPI application.
    
    Args:
        app: FastAPI application instance
    """
    # Register the main API routes
    app.include_router(
        router, 
        prefix="/api/assessments/candlestick",
        tags=["candlestick-patterns"]
    )
    
    # Register legacy API routes
    # Note: These routes are maintained for backward compatibility
    # but will eventually be deprecated
    app.include_router(
        legacy_router,
        tags=["candlestick-patterns-legacy"]
    )

def initialize_background_tasks(app):
    """
    Register any background tasks related to candlestick patterns module.
    
    Args:
        app: FastAPI application instance
    """
    from backend.assessments.candlestick_patterns.candlestick_db_sync import register_background_tasks
    register_background_tasks(app)

def initialize_module(app):
    """
    Initialize the candlestick pattern module with routes and background tasks.
    
    This is the main entry point for setting up this module during application 
    startup. It handles both route registration and background tasks.
    
    Args:
        app: FastAPI application instance
    """
    # Register routes
    register_routes(app)
    
    # Register background tasks
    initialize_background_tasks(app)

__all__ = [
    'router',
    'legacy_router',
    'register_routes',
    'initialize_background_tasks',
    'initialize_module',
    'CandlestickQuestion',
    'CandlestickSession',
    'CandlestickAssessmentResponse',
    'CandlestickAssessmentService',
    'CandlestickQuestionRepository',
    'CandlestickSessionRepository',
    'CandlestickAssessmentRepositoryImpl',
    'ExplanationGenerator',
    'UserLevel',
    'CandlestickData',
    'Candle',
    'Difficulty',
    'plot_candlestick_chart',
    'get_patterns_by_difficulty',
    'generate_options',
    'get_pattern_category',
    'get_pattern_description',
    'format_pattern_name',
    'convert_to_candlestick_pattern'
] 