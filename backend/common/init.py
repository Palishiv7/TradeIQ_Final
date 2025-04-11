"""
Initialization Module for TradeIQ

This module handles the initialization of all core components during application startup:
1. Configuration loading
2. Logging setup
3. Database initialization
4. Redis connection
5. Thread pool service
6. Cache service
7. Error handling
"""

import os
import sys
import logging
import logging.config
from typing import Dict, Any, Optional
import signal
import atexit
import asyncio
import redis
from contextlib import asynccontextmanager

from backend.common.config import get_config, reload_config
from backend.common.threading import ThreadPoolService, ThreadPoolServiceConfig
from backend.common.cache_service import (
    CacheService, CacheBackendType, set_default_cache_service, MemoryCacheBackend, RedisCacheBackend
)
from backend.common.error_handling import log_error, TradeIQError, DatabaseConnectionError, CacheConnectionError

# Configure basic logging initially
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def setup_logging() -> None:
    """
    Initialize the logging system based on configuration.
    
    Sets up appropriate handlers, formatters, and log levels.
    """
    config = get_config()
    log_config = config.logging
    
    # Get the log level 
    level = getattr(logging, log_config.level, logging.INFO)
    
    # Configure the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(log_config.format)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # Add file handler if specified
    if log_config.file_path:
        try:
            # Create directory if it doesn't exist
            log_dir = os.path.dirname(log_config.file_path)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)
                
            # Use RotatingFileHandler for log rotation
            from logging.handlers import RotatingFileHandler
            
            # Parse rotation size (e.g., "10MB" -> 10 * 1024 * 1024)
            size_str = log_config.rotation_size.upper()
            if size_str.endswith('KB'):
                max_bytes = int(size_str[:-2]) * 1024
            elif size_str.endswith('MB'):
                max_bytes = int(size_str[:-2]) * 1024 * 1024
            elif size_str.endswith('GB'):
                max_bytes = int(size_str[:-2]) * 1024 * 1024 * 1024
            else:
                max_bytes = int(size_str)
                
            file_handler = RotatingFileHandler(
                log_config.file_path,
                maxBytes=max_bytes,
                backupCount=log_config.max_backups
            )
            file_handler.setLevel(level)
            file_formatter = logging.Formatter(log_config.format)
            file_handler.setFormatter(file_formatter)
            root_logger.addHandler(file_handler)
            logger.info(f"Logging to file: {log_config.file_path}")
        except Exception as e:
            logger.error(f"Failed to set up file logging: {e}")
    
    # Set levels for third-party loggers to reduce noise
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.ERROR)
    
    logger.info(f"Logging initialized with level: {log_config.level}")

def initialize_thread_pool() -> ThreadPoolService:
    """
    Initialize the thread pool service based on configuration.
    
    Returns:
        Initialized thread pool service
    """
    config = get_config()
    thread_config = config.thread_pool
    
    # Create thread pool configuration from app config
    pool_config = ThreadPoolServiceConfig(
        min_workers=thread_config.min_workers,
        max_workers=thread_config.max_workers,
        queue_size=thread_config.queue_size,
        adaptive=thread_config.adaptive,
        cpu_target=thread_config.cpu_target,
        thread_idle_timeout=thread_config.thread_idle_timeout
    )
    
    # Initialize thread pool service
    thread_pool = ThreadPoolService(pool_config)
    logger.info(f"Thread pool initialized with {thread_pool._workers_count} workers")
    
    return thread_pool

def initialize_redis_client() -> Optional[redis.Redis]:
    """
    Initialize the Redis client based on configuration.
    
    Returns:
        Redis client or None if not configured/available
    """
    config = get_config()
    redis_config = config.redis
    
    # Don't initialize Redis if we're in testing mode and not explicitly required
    if config.is_testing and not os.environ.get("REDIS_REQUIRED"):
        logger.info("Skipping Redis initialization in testing mode")
        return None
        
    try:
        client = redis.Redis(
            host=redis_config.host,
            port=redis_config.port,
            db=redis_config.db,
            password=redis_config.password,
            ssl=redis_config.use_ssl,
            socket_timeout=redis_config.connection_timeout,
            socket_keepalive=redis_config.socket_keepalive,
            health_check_interval=30,
            retry_on_timeout=True,
            decode_responses=False  # We handle decoding ourselves
        )
        
        # Test connection
        client.ping()
        logger.info(f"Redis client initialized: {redis_config.host}:{redis_config.port}/{redis_config.db}")
        return client
    except redis.RedisError as e:
        if config.is_production:
            logger.error(f"Failed to connect to Redis: {e}")
            raise CacheConnectionError("redis", cause=e)
        else:
            logger.warning(f"Redis connection failed, continuing without Redis: {e}")
            return None

def initialize_cache_service(redis_client: Optional[redis.Redis] = None) -> CacheService:
    """
    Initialize the cache service based on configuration.
    
    Args:
        redis_client: Optional Redis client to use
        
    Returns:
        Initialized cache service
    """
    config = get_config()
    cache_config = config.cache
    
    # Determine backend type based on config and Redis availability
    backend_type = CacheBackendType.MEMORY
    backend_config = {}
    
    # Use Redis backend if configured and available
    if cache_config.use_redis and redis_client is not None:
        backend_type = CacheBackendType.REDIS
        backend_config = {
            "redis_client": redis_client,
            "key_prefix": cache_config.redis_key_prefix
        }
    else:
        # Using memory backend
        backend_config = {
            "max_size": cache_config.memory_max_size
        }
    
    # Create and initialize cache service
    cache_service = CacheService(
        backend_type=backend_type,
        default_ttl=cache_config.default_ttl,
        backend_config=backend_config
    )
    
    # Set as default cache service
    set_default_cache_service(cache_service)
    
    logger.info(f"Cache service initialized with {backend_type.value} backend")
    
    return cache_service

def initialize_database():
    """
    Initialize the database connection and run migrations if needed.
    
    This function doesn't actually establish persistent connections, but ensures
    the database is accessible and properly initialized.
    """
    config = get_config()
    
    # This will be handled by the database initialization module
    from database.init_db import initialize_database
    
    try:
        success = initialize_database()
        if success:
            logger.info("Database initialized successfully")
        else:
            error_msg = "Database initialization failed"
            if config.is_production:
                raise DatabaseConnectionError(config.database.database, details={"reason": error_msg})
            else:
                logger.warning(f"{error_msg}, application may not function correctly")
    except Exception as e:
        error_msg = f"Database initialization error: {e}"
        if config.is_production:
            log_error(e)
            raise DatabaseConnectionError(config.database.database, cause=e)
        else:
            logger.warning(f"{error_msg}, application may not function correctly")

def setup_signal_handlers():
    """Set up signal handlers for graceful shutdown"""
    thread_pool = ThreadPoolService()
    
    def handle_exit(*args):
        """Handle exit signals"""
        logger.info("Shutdown signal received, cleaning up resources...")
        thread_pool.shutdown(wait=True)
        logger.info("Thread pool shut down")
        
        # Other cleanup as needed
        
        logger.info("Cleanup complete")
        
    # Register signal handlers
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, handle_exit)
    
    # Register atexit handler
    atexit.register(handle_exit)
    
    logger.info("Signal handlers registered")

def initialize_app() -> Dict[str, Any]:
    """
    Initialize all application components.
    
    This is the main initialization function that should be called
    during application startup.
    
    Returns:
        Dictionary of initialized components
    """
    # Load and setup configuration
    config = get_config()
    
    # Initialize logging first
    setup_logging()
    logger.info(f"Initializing TradeIQ application in {config.environment.env} environment")
    
    # Initialize core components
    components = {}
    
    # Thread pool
    components["thread_pool"] = initialize_thread_pool()
    
    # Redis
    redis_client = initialize_redis_client()
    components["redis_client"] = redis_client
    
    # Cache service
    components["cache_service"] = initialize_cache_service(redis_client)
    
    # Database
    initialize_database()
    
    # Set up signal handlers for graceful shutdown
    setup_signal_handlers()
    
    logger.info("Application initialization complete")
    return components

@asynccontextmanager
async def application_lifespan(app: Any):
    """
    Application lifespan context manager for FastAPI.
    
    This handles initialization and cleanup of application components
    on startup and shutdown.
    
    Args:
        app: FastAPI application instance
    """
    # Perform application startup
    logger.info("Application starting up")
    try:
        # Initialize components
        components = initialize_app()
        
        # Store components in app state
        app.state.components = components
        
        logger.info("Application startup complete")
        yield
    except Exception as e:
        logger.error(f"Application startup failed: {e}", exc_info=True)
        # Re-raise to prevent application from starting
        raise
    finally:
        # Perform application shutdown
        logger.info("Application shutting down")
        
        if hasattr(app.state, "components"):
            # Clean up components in the reverse order of initialization
            components = app.state.components
            
            # Shutdown thread pool
            if "thread_pool" in components:
                thread_pool = components["thread_pool"]
                thread_pool.shutdown(wait=True)
                logger.info("Thread pool shut down")
            
            # Close Redis connection
            if "redis_client" in components and components["redis_client"] is not None:
                redis_client = components["redis_client"]
                redis_client.close()
                logger.info("Redis connection closed")
                
            # Clean up database connections
            from database.init_db import reset_connection_pool
            reset_connection_pool()
            logger.info("Database connections closed")
        
        logger.info("Application shutdown complete") 