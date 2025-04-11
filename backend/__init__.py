"""
TradeIQ AI-Powered Assessment Platform

This module serves as the main entry point for the TradeIQ backend,
providing a comprehensive API for managing AI-powered trading assessments.

The platform features:
1. Dynamic assessment generation for candlestick patterns, market fundamentals, and market psychology
2. AI-based answer evaluation with detailed feedback
3. Adaptive difficulty adjustment based on user performance
4. Comprehensive analytics and performance tracking
5. Gamification features with XP, achievements, and leaderboards
"""

import os
import logging
from fastapi import FastAPI
from contextlib import asynccontextmanager
from typing import Optional, Callable, AsyncContextManager, Dict, List, Any, Type, Union
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Automatic database initialization flag
AUTO_DB_INIT = os.environ.get("AUTO_DB_INIT", "false").lower() == "true"

# Import the async initializer and cleanup
from backend.database.init_db import initialize_database, reset_connection_pool

# Assessment Factory Pattern Implementation
class AssessmentStrategy(ABC):
    """Abstract base class for all assessment strategies."""
    
    @abstractmethod
    async def generate_question(self, difficulty: float = 0.5, **kwargs) -> Dict[str, Any]:
        """Generate a question with the given difficulty."""
        pass
    
    @abstractmethod
    async def evaluate_answer(self, question_id: str, user_answer: str, **kwargs) -> Dict[str, Any]:
        """Evaluate a user's answer to a question."""
        pass
    
    @abstractmethod
    async def get_explanation(self, question_id: str, user_answer: str, **kwargs) -> Dict[str, Any]:
        """Generate an explanation for a question and answer."""
        pass
    
    @abstractmethod
    async def get_assessment_stats(self, session_id: str, **kwargs) -> Dict[str, Any]:
        """Get statistics for an assessment session."""
        pass

class AssessmentFactory:
    """
    Factory class for creating assessment objects.
    
    This factory implements a dynamic registration system for assessment strategies,
    allowing new assessment types to be added without modifying existing code.
    """
    
    _strategies: Dict[str, Type[AssessmentStrategy]] = {}
    
    @classmethod
    def register_strategy(cls, name: str, strategy_class: Type[AssessmentStrategy]) -> None:
        """
        Register an assessment strategy.
        
        Args:
            name: Name of the assessment strategy
            strategy_class: Class implementing the AssessmentStrategy interface
        """
        if name in cls._strategies:
            logger.warning(f"Assessment strategy '{name}' already registered, overwriting")
        
        cls._strategies[name] = strategy_class
        logger.info(f"Registered assessment strategy: {name}")
    
    @classmethod
    def create_assessment(cls, assessment_type: str, **kwargs) -> Optional[AssessmentStrategy]:
        """
        Create an assessment of the specified type.
        
        Args:
            assessment_type: The type of assessment to create
            **kwargs: Additional parameters to pass to the assessment constructor
            
        Returns:
            An instance of the requested assessment strategy or None if not found
        """
        if assessment_type not in cls._strategies:
            logger.error(f"Unknown assessment type: {assessment_type}")
            return None
        
        strategy_class = cls._strategies[assessment_type]
        return strategy_class(**kwargs)
    
    @classmethod
    def get_available_assessment_types(cls) -> List[str]:
        """
        Get a list of available assessment types.
        
        Returns:
            List of registered assessment type names
        """
        return list(cls._strategies.keys())

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI application lifespan context manager.
    
    Handles initialization and cleanup of database connections
    and other resources on application startup and shutdown.
    """
    # Startup actions
    logger.info("Application startup sequence initiated.")
    # Initialize database asynchronously
    logger.info("Initializing asynchronous database connection...")
    # No need for AUTO_DB_INIT check if initialization is now standard
    db_success = await initialize_database() # Await the async function
    if not db_success:
        logger.critical("ASYNC DATABASE INITIALIZATION FAILED! Application might not function correctly.")
        # Depending on policy, might want to raise an exception here to halt startup
        # raise RuntimeError("Database initialization failed")
    else:
        logger.info("Async database initialized successfully.")
    
    # Configure cache system
    try:
        logger.info("Configuring cache system...")
        from backend.common.cache import configure_cache
        # Configure with default memory cache
        cache_manager = configure_cache(
            memory_cache_size=10000,
            redis_url=os.environ.get("REDIS_URL"),
            write_through=True
        )
        logger.info(f"Cache system configured with {len(cache_manager.backends)} backends")
    except Exception as e:
        logger.error(f"Failed to configure cache system: {e}", exc_info=True)
    
    # Initialize gamification system (ensure it's async if it interacts with DB)
    try:
        logger.info("Initializing gamification system")
        from backend.common.gamification import initialize_gamification_system
        # Assuming initialize_gamification_system is async or safe to call here
        # If it needs DB access, it should use the async setup.
        await initialize_gamification_system() 
        logger.info("Gamification system initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize gamification system: {e}", exc_info=True)
    
    logger.info("Application startup sequence complete. Yielding control.")
    # Start application
    yield
    
    # Cleanup on shutdown
    logger.info("Application shutdown sequence initiated.")
    # Close async database connections
    logger.info("Cleaning up async database connections...")
    await reset_connection_pool() # Await the async function
    
    # Clean up other resources if needed
    # ...
    
    logger.info("Application shutdown sequence complete.")

def create_app(
    # Remove lifespan parameter as we define it above
    # lifespan: Optional[AsyncContextManager] = None,
    app_name: str = "TradeIQ Assessment Platform",
    app_description: str = "AI-powered platform for financial market education and assessment"
) -> FastAPI:
    """
    Create and configure a FastAPI application instance.
    
    This factory function creates a new FastAPI application with all routers
    and middleware configured.
    
    Args:
        app_name: The name of the application
        app_description: Description of the application
        
    Returns:
        Configured FastAPI application
    """
    # Create FastAPI application with the defined lifespan manager
    app = FastAPI(
        title=app_name,
        description=app_description,
        version="0.1.0",
        lifespan=lifespan # Use the defined lifespan context manager
    )
    
    # Add CORS middleware
    from fastapi.middleware.cors import CORSMiddleware
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000"],  # Frontend URL
        allow_credentials=True,
        allow_methods=["*"],  # Allow all methods
        allow_headers=["*"],  # Allow all headers
    )
    
    # Import and register routers
    from backend.api import main_router
    app.include_router(main_router, prefix="/api")
    
    # Additional routers can be registered here
    # app.include_router(other_router, prefix="/api/other", tags=["other"])
    
    # Setup exception handlers
    from backend.api import validation_exception_handler
    from fastapi.exceptions import RequestValidationError
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    
    # Register assessment modules
    _register_assessment_modules(app)
    
    logger.info(f"Application created with {len(app.routes)} routes")
    return app

def _register_assessment_modules(app: FastAPI) -> None:
    """
    Register assessment modules with the application's main router.
    
    Args:
        app: FastAPI application instance (used here primarily for logging context, 
             registration happens via the imported function)
    """
    # Import the central registration function
    from backend.api import register_assessment_module
    
    # Register candlestick patterns assessment
    try:
        logger.info("Attempting to import candlestick_patterns.router")
        # Import the router from the candlestick module
        from backend.assessments.candlestick_patterns.router import router as candlestick_router
        logger.info(f"Successfully imported candlestick_router with {len(candlestick_router.routes)} routes")
        # Use explicit name "candlestick" not a dynamic name derived from the module
        register_assessment_module(name="candlestick", router=candlestick_router)
        logger.info("Registered candlestick patterns assessment module via main_router")
    except ImportError as e:
        logger.warning(f"Candlestick assessment module not found or router not defined: {e}")
        logger.error("Import error details:", exc_info=True)
    except Exception as e:
        logger.error(f"Failed to register candlestick patterns assessment module: {e}", exc_info=True)
    
    # Register market fundamentals assessment if available
    try:
        from backend.assessments.market_fundamentals.router import router as fundamentals_router
        # Use explicit name "fundamentals" not a dynamic name derived from the module
        register_assessment_module(name="fundamentals", router=fundamentals_router)
        logger.info("Registered market fundamentals assessment module via main_router")
    except ImportError as e:
        logger.warning(f"Market fundamentals assessment module not found or router not defined: {e}")
    except Exception as e:
        logger.error(f"Failed to register market fundamentals assessment module: {e}")
    
    # Register market psychology assessment if available
    try:
        from backend.assessments.market_psychology.router import router as psychology_router
        # Use explicit name "psychology" not a dynamic name derived from the module
        register_assessment_module(name="psychology", router=psychology_router)
        logger.info("Registered market psychology assessment module via main_router")
    except ImportError as e:
        logger.warning(f"Market psychology assessment module not found or router not defined: {e}")
    except Exception as e:
        logger.error(f"Failed to register market psychology assessment module: {e}")
    
    # Register gamification API routes directly to the app (not as a standard assessment module)
    try:
        from backend.common.gamification.controllers import router as gamification_router
        # Add with a specific prefix and tags for better organization
        app.include_router(
            gamification_router, 
            prefix="/api/gamification", 
            tags=["gamification"]
        )
        logger.info("Registered gamification API routes directly under /api/gamification")
    except ImportError:
        logger.warning("Gamification controller or router not found.")
    except Exception as e:
        logger.error(f"Failed to register gamification API routes: {e}") 