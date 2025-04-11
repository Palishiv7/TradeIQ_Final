"""
Application Logger

This module provides a consistent logging interface for the application,
with configurable log levels, formatters, and handlers.
"""

import os
import sys
import json
import time
import logging
import datetime
import traceback
import functools
import asyncio
from typing import Dict, Any, Optional, Union, Callable, TypeVar

# Default log format
DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Type variable for the decorator
F = TypeVar('F', bound=Callable[..., Any])

# Export public interface
__all__ = [
    'configure_logger',
    'get_logger',
    'LoggerAdapter',
    'JsonFormatter',
    'with_context',
    'app_logger',
    'log_execution_time'
]


class JsonFormatter(logging.Formatter):
    """
    Formatter that outputs JSON strings after parsing the log record.
    
    This allows for easier integration with log aggregation systems
    that consume JSON, like ELK stack and cloud logging services.
    """
    
    def __init__(
        self,
        fmt: Optional[str] = None,
        datefmt: Optional[str] = None,
        style: str = '%',
        validate: bool = True,
        *,
        indent: Optional[int] = None
    ):
        """
        Initialize the formatter with specified format strings.
        
        Args:
            fmt: Format string
            datefmt: Date format string
            style: Style of format string
            validate: Whether to validate the format string
            indent: Indentation level for pretty printing JSON
        """
        super().__init__(fmt, datefmt, style, validate)
        self.indent = indent
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format the specified record as JSON.
        
        Args:
            record: Log record to format
            
        Returns:
            Formatted JSON string
        """
        # Create a basic log object with standard fields
        log_object = {
            "timestamp": datetime.datetime.fromtimestamp(record.created).isoformat(),
            "name": record.name,
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "filename": record.filename,
            "line": record.lineno
        }
        
        # Add exception info if available
        if record.exc_info:
            log_object["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": self.formatException(record.exc_info)
            }
        
        # Add extra fields from the record
        if hasattr(record, 'data') and isinstance(record.data, dict):
            log_object.update(record.data)
        
        # Convert to JSON
        return json.dumps(log_object, indent=self.indent)


def configure_logger(
    name: str = "tradeiq",
    level: Union[str, int] = logging.INFO,
    format_string: str = DEFAULT_LOG_FORMAT,
    date_format: str = DEFAULT_DATE_FORMAT,
    use_json: bool = False,
    log_file: Optional[str] = None,
    console_output: bool = True
) -> logging.Logger:
    """
    Configure a logger with appropriate handlers and formatters.
    
    Args:
        name: Logger name
        level: Log level
        format_string: Log format string
        date_format: Date format string
        use_json: Whether to use JSON formatting
        log_file: Path to log file (if None, no file handler is created)
        console_output: Whether to output logs to console
        
    Returns:
        Configured logger
    """
    # If level is a string, convert to logging level
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers = []
    
    # Create formatter
    if use_json:
        formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(format_string, date_format)
    
    # Add console handler if requested
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Add file handler if requested
    if log_file:
        try:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except (FileNotFoundError, PermissionError) as e:
            fallback_logger = logging.getLogger("fallback")
            fallback_logger.warning(f"Could not create log file {log_file}: {e}")
    
    return logger


def get_logger(
    name: str,
    parent: Optional[logging.Logger] = None
) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    Args:
        name: Logger name
        parent: Optional parent logger
        
    Returns:
        Logger instance
    """
    if parent:
        return parent.getChild(name)
    return logging.getLogger(name)


class LoggerAdapter(logging.LoggerAdapter):
    """
    Logger adapter that adds contextual information to log records.
    
    This allows adding consistent context to all log messages from a component,
    like user ID, request ID, or component-specific metadata.
    """
    
    def __init__(
        self,
        logger: logging.Logger,
        context: Dict[str, Any] = None
    ):
        """
        Initialize the adapter with the specified logger and context.
        
        Args:
            logger: Logger to adapt
            context: Context dictionary to add to log records
        """
        super().__init__(logger, context or {})
    
    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple:
        """
        Process the log record.
        
        Args:
            msg: Log message
            kwargs: Keyword arguments for logging call
            
        Returns:
            Tuple of (message, kwargs)
        """
        # Add our context to the extra data
        kwargs = kwargs.copy()
        extra = kwargs.get('extra', {})
        if not extra:
            extra = {}
            kwargs['extra'] = extra
        
        # Add context data to the extra dictionary
        data = extra.get('data', {})
        if not data:
            data = {}
            extra['data'] = data
        
        # Merge our context with any existing data
        if self.extra:
            data.update(self.extra)
        
        return msg, kwargs
    
    def with_context(self, **context) -> 'LoggerAdapter':
        """
        Create a new adapter with additional context.
        
        Args:
            **context: Context to add
            
        Returns:
            New logger adapter with combined context
        """
        new_context = self.extra.copy()
        new_context.update(context)
        return LoggerAdapter(self.logger, new_context)


def with_context(name: str = None, **context) -> LoggerAdapter:
    """
    Create a logger adapter with context.
    
    Args:
        name: Optional logger name
        context: Context dictionary
        
    Returns:
        Logger adapter with context
    """
    logger = get_logger(name) if name else app_logger
    return LoggerAdapter(logger, context)


def get_app_logger() -> logging.Logger:
    """
    Get or create the application logger.
    
    Returns:
        The configured application logger
    """
    logger = logging.getLogger("tradeiq")
    
    # Only configure if not already configured
    if not logger.handlers:
        return configure_logger(
            name="tradeiq",
            level=os.environ.get("LOG_LEVEL", "INFO"),
            use_json=os.environ.get("LOG_JSON", "false").lower() == "true",
            log_file=os.environ.get("LOG_FILE"),
            console_output=True
        )
    
    return logger

# Initialize the app logger
app_logger = get_app_logger()

def log_execution_time(logger: Optional[logging.Logger] = None) -> Callable[[F], F]:
    """
    Decorator to log the execution time of a function.
    
    Args:
        logger: Optional logger to use. If not provided, uses app_logger.
        
    Returns:
        Decorated function that logs its execution time
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                (logger or get_app_logger()).debug(
                    f"{func.__name__} executed in {execution_time:.3f} seconds"
                )
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                (logger or get_app_logger()).error(
                    f"{func.__name__} failed after {execution_time:.3f} seconds: {str(e)}"
                )
                raise
                
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                execution_time = time.time() - start_time
                (logger or get_app_logger()).debug(
                    f"{func.__name__} executed in {execution_time:.3f} seconds"
                )
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                (logger or get_app_logger()).error(
                    f"{func.__name__} failed after {execution_time:.3f} seconds: {str(e)}"
                )
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else wrapper
    return decorator
