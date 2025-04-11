"""
Error Handling System for TradeIQ

This module provides a comprehensive error handling framework including:
1. Custom exception hierarchy for different error types
2. Retry mechanisms with backoff for transient failures
3. Structured error logging and reporting
4. Error response generation for APIs
"""

import time
import logging
import traceback
import asyncio
import random
import functools
import json
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union, cast
from datetime import datetime
from pydantic import BaseModel, Field, validator

# Type variables
T = TypeVar('T')
F = TypeVar('F', bound=Callable)

# Configure logging
logger = logging.getLogger(__name__)

class ErrorSeverity(Enum):
    """Severity levels for errors"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class ErrorCode(Enum):
    """Standard error codes for TradeIQ"""
    # General errors
    UNKNOWN_ERROR = "unknown_error"
    VALIDATION_ERROR = "validation_error"
    AUTHENTICATION_ERROR = "authentication_error"
    AUTHORIZATION_ERROR = "authorization_error"
    NOT_FOUND_ERROR = "not_found_error"
    RATE_LIMIT_ERROR = "rate_limit_error"
    TIMEOUT_ERROR = "timeout_error"
    
    # Assessment errors
    ASSESSMENT_NOT_FOUND = "assessment_not_found"
    QUESTION_NOT_FOUND = "question_not_found"
    SESSION_NOT_FOUND = "session_not_found"
    SESSION_EXPIRED = "session_expired"
    DUPLICATE_ANSWER = "duplicate_answer"
    
    # AI model errors
    MODEL_ERROR = "model_error"
    MODEL_TIMEOUT = "model_timeout"
    MODEL_OVERLOADED = "model_overloaded"
    
    # Database errors
    DATABASE_ERROR = "database_error"
    DATABASE_CONNECTION_ERROR = "database_connection_error"
    DATABASE_QUERY_ERROR = "database_query_error"
    
    # Cache errors
    CACHE_ERROR = "cache_error"
    CACHE_CONNECTION_ERROR = "cache_connection_error"
    
    # External service errors
    EXTERNAL_SERVICE_ERROR = "external_service_error"
    EXTERNAL_SERVICE_TIMEOUT = "external_service_timeout"
    
    # Data errors
    DATA_ERROR = "data_error"
    DATA_VALIDATION_ERROR = "data_validation_error"
    DATA_INTEGRITY_ERROR = "data_integrity_error"
    
    # Infrastructure errors
    INFRASTRUCTURE_ERROR = "infrastructure_error"
    RESOURCE_EXHAUSTED = "resource_exhausted"

class ErrorInfo(BaseModel):
    """Structured information about an error"""
    code: ErrorCode
    message: str
    timestamp: datetime = Field(default_factory=datetime.now)
    severity: ErrorSeverity = ErrorSeverity.ERROR
    details: Optional[Dict[str, Any]] = None
    exception_type: Optional[str] = None
    exception_message: Optional[str] = None
    stack_trace: Optional[List[str]] = None
    context: Optional[Dict[str, Any]] = None
    
    class Config:
        use_enum_values = True
    
    @validator('stack_trace', pre=True, always=False)
    def validate_stack_trace(cls, v):
        """Format stack trace if it's a string"""
        if isinstance(v, str):
            return v.splitlines()
        return v

class TradeIQError(Exception):
    """Base exception class for all TradeIQ errors"""
    
    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.UNKNOWN_ERROR,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.message = message
        self.code = code
        self.severity = severity
        self.details = details or {}
        self.cause = cause
        self.context = context or {}
        self.timestamp = datetime.now()
    
    def to_error_info(self, include_stack_trace: bool = False) -> ErrorInfo:
        """Convert the exception to an ErrorInfo object"""
        exception_type = type(self).__name__
        exception_message = str(self)
        
        stack_trace = None
        if include_stack_trace:
            stack_trace = traceback.format_exc().splitlines()
            
        # Include cause information in details
        details = dict(self.details)
        if self.cause is not None:
            details["cause"] = {
                "type": type(self.cause).__name__,
                "message": str(self.cause)
            }
        
        return ErrorInfo(
            code=self.code,
            message=self.message,
            timestamp=self.timestamp,
            severity=self.severity,
            details=details,
            exception_type=exception_type,
            exception_message=exception_message,
            stack_trace=stack_trace,
            context=self.context
        )
    
    def to_dict(self, include_stack_trace: bool = False) -> Dict[str, Any]:
        """Convert the exception to a dictionary"""
        return self.to_error_info(include_stack_trace).dict()
    
    def to_json(self, include_stack_trace: bool = False) -> str:
        """Convert the exception to a JSON string"""
        return json.dumps(self.to_dict(include_stack_trace))
    
    def __str__(self) -> str:
        base_str = f"{self.code.value}: {self.message}"
        if self.details:
            base_str += f" (details: {self.details})"
        if self.cause:
            base_str += f" caused by {type(self.cause).__name__}: {str(self.cause)}"
        return base_str

# Specific exception classes for different error types

class ValidationError(TradeIQError):
    """Error raised when input validation fails"""
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            code=ErrorCode.VALIDATION_ERROR,
            severity=ErrorSeverity.WARNING,
            details=details,
            cause=cause,
            context=context
        )

class AuthenticationError(TradeIQError):
    """Error raised when authentication fails"""
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            code=ErrorCode.AUTHENTICATION_ERROR,
            severity=ErrorSeverity.WARNING,
            details=details,
            cause=cause,
            context=context
        )

class AuthorizationError(TradeIQError):
    """Error raised when authorization fails"""
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            code=ErrorCode.AUTHORIZATION_ERROR,
            severity=ErrorSeverity.WARNING,
            details=details,
            cause=cause,
            context=context
        )

class NotFoundError(TradeIQError):
    """Error raised when a requested resource is not found"""
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            code=ErrorCode.NOT_FOUND_ERROR,
            severity=ErrorSeverity.WARNING,
            details=details,
            cause=cause,
            context=context
        )

class RateLimitError(TradeIQError):
    """Error raised when rate limits are exceeded"""
    
    def __init__(
        self,
        message: str,
        retry_after: Optional[float] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        details = details or {}
        if retry_after is not None:
            details["retry_after_seconds"] = retry_after
            
        super().__init__(
            message=message,
            code=ErrorCode.RATE_LIMIT_ERROR,
            severity=ErrorSeverity.WARNING,
            details=details,
            cause=cause,
            context=context
        )

class TimeoutError(TradeIQError):
    """Error raised when an operation times out"""
    
    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        details = details or {}
        if operation is not None:
            details["operation"] = operation
        if timeout_seconds is not None:
            details["timeout_seconds"] = timeout_seconds
            
        super().__init__(
            message=message,
            code=ErrorCode.TIMEOUT_ERROR,
            severity=ErrorSeverity.WARNING,
            details=details,
            cause=cause,
            context=context
        )

class AssessmentError(TradeIQError):
    """Base class for assessment-related errors"""
    pass

class AssessmentNotFoundError(AssessmentError):
    """Error raised when an assessment is not found"""
    
    def __init__(
        self,
        assessment_id: str,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        details = details or {}
        details["assessment_id"] = assessment_id
        
        super().__init__(
            message=f"Assessment with ID {assessment_id} not found",
            code=ErrorCode.ASSESSMENT_NOT_FOUND,
            severity=ErrorSeverity.WARNING,
            details=details,
            cause=cause,
            context=context
        )

class QuestionNotFoundError(AssessmentError):
    """Error raised when a question is not found"""
    
    def __init__(
        self,
        question_id: str,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        details = details or {}
        details["question_id"] = question_id
        
        super().__init__(
            message=f"Question with ID {question_id} not found",
            code=ErrorCode.QUESTION_NOT_FOUND,
            severity=ErrorSeverity.WARNING,
            details=details,
            cause=cause,
            context=context
        )

class SessionNotFoundError(AssessmentError):
    """Error raised when a session is not found"""
    
    def __init__(
        self,
        session_id: str,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        details = details or {}
        details["session_id"] = session_id
        
        super().__init__(
            message=f"Session with ID {session_id} not found",
            code=ErrorCode.SESSION_NOT_FOUND,
            severity=ErrorSeverity.WARNING,
            details=details,
            cause=cause,
            context=context
        )

class SessionExpiredError(AssessmentError):
    """Error raised when a session has expired"""
    
    def __init__(
        self,
        session_id: str,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        details = details or {}
        details["session_id"] = session_id
        
        super().__init__(
            message=f"Session with ID {session_id} has expired",
            code=ErrorCode.SESSION_EXPIRED,
            severity=ErrorSeverity.WARNING,
            details=details,
            cause=cause,
            context=context
        )

class DuplicateAnswerError(AssessmentError):
    """Error raised when an answer is submitted more than once"""
    
    def __init__(
        self,
        question_id: str,
        session_id: str,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        details = details or {}
        details["question_id"] = question_id
        details["session_id"] = session_id
        
        super().__init__(
            message=f"Answer for question {question_id} in session {session_id} already submitted",
            code=ErrorCode.DUPLICATE_ANSWER,
            severity=ErrorSeverity.WARNING,
            details=details,
            cause=cause,
            context=context
        )

class ModelError(TradeIQError):
    """Base class for AI model-related errors"""
    pass

class ModelTimeoutError(ModelError):
    """Error raised when an AI model operation times out"""
    
    def __init__(
        self,
        model_id: str,
        operation: str,
        timeout_seconds: float,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        details = details or {}
        details["model_id"] = model_id
        details["operation"] = operation
        details["timeout_seconds"] = timeout_seconds
        
        super().__init__(
            message=f"Operation {operation} on model {model_id} timed out after {timeout_seconds} seconds",
            code=ErrorCode.MODEL_TIMEOUT,
            severity=ErrorSeverity.ERROR,
            details=details,
            cause=cause,
            context=context
        )

class ModelOverloadedError(ModelError):
    """Error raised when an AI model is overloaded"""
    
    def __init__(
        self,
        model_id: str,
        retry_after: Optional[float] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        details = details or {}
        details["model_id"] = model_id
        if retry_after is not None:
            details["retry_after_seconds"] = retry_after
            
        super().__init__(
            message=f"Model {model_id} is currently overloaded",
            code=ErrorCode.MODEL_OVERLOADED,
            severity=ErrorSeverity.WARNING,
            details=details,
            cause=cause,
            context=context
        )

class DatabaseError(TradeIQError):
    """Base class for database-related errors"""
    pass

class DatabaseConnectionError(DatabaseError):
    """Error raised when database connection fails"""
    
    def __init__(
        self,
        database: str,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        details = details or {}
        details["database"] = database
        
        super().__init__(
            message=f"Failed to connect to database {database}",
            code=ErrorCode.DATABASE_CONNECTION_ERROR,
            severity=ErrorSeverity.CRITICAL,
            details=details,
            cause=cause,
            context=context
        )

class DatabaseQueryError(DatabaseError):
    """Error raised when a database query fails"""
    
    def __init__(
        self,
        query_type: str,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        details = details or {}
        details["query_type"] = query_type
        
        super().__init__(
            message=f"Database query of type {query_type} failed",
            code=ErrorCode.DATABASE_QUERY_ERROR,
            severity=ErrorSeverity.ERROR,
            details=details,
            cause=cause,
            context=context
        )

class CacheError(TradeIQError):
    """Base class for cache-related errors"""
    pass

class CacheConnectionError(CacheError):
    """Error raised when cache connection fails"""
    
    def __init__(
        self,
        cache_type: str,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        details = details or {}
        details["cache_type"] = cache_type
        
        super().__init__(
            message=f"Failed to connect to cache of type {cache_type}",
            code=ErrorCode.CACHE_CONNECTION_ERROR,
            severity=ErrorSeverity.ERROR,
            details=details,
            cause=cause,
            context=context
        )

class ExternalServiceError(TradeIQError):
    """Base class for external service-related errors"""
    pass

class ExternalServiceTimeoutError(ExternalServiceError):
    """Error raised when an external service operation times out"""
    
    def __init__(
        self,
        service: str,
        operation: str,
        timeout_seconds: float,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        details = details or {}
        details["service"] = service
        details["operation"] = operation
        details["timeout_seconds"] = timeout_seconds
        
        super().__init__(
            message=f"Operation {operation} on service {service} timed out after {timeout_seconds} seconds",
            code=ErrorCode.EXTERNAL_SERVICE_TIMEOUT,
            severity=ErrorSeverity.ERROR,
            details=details,
            cause=cause,
            context=context
        )

class DataError(TradeIQError):
    """Base class for data-related errors"""
    pass

class DataValidationError(DataError):
    """Error raised when data validation fails"""
    
    def __init__(
        self,
        data_type: str,
        validation_errors: List[Dict[str, Any]],
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        details = details or {}
        details["data_type"] = data_type
        details["validation_errors"] = validation_errors
        
        super().__init__(
            message=f"Validation failed for data of type {data_type}",
            code=ErrorCode.DATA_VALIDATION_ERROR,
            severity=ErrorSeverity.WARNING,
            details=details,
            cause=cause,
            context=context
        )

class DataIntegrityError(DataError):
    """Error raised when data integrity is violated"""
    
    def __init__(
        self,
        data_type: str,
        integrity_issue: str,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        details = details or {}
        details["data_type"] = data_type
        details["integrity_issue"] = integrity_issue
        
        super().__init__(
            message=f"Data integrity violation for {data_type}: {integrity_issue}",
            code=ErrorCode.DATA_INTEGRITY_ERROR,
            severity=ErrorSeverity.ERROR,
            details=details,
            cause=cause,
            context=context
        )

class InfrastructureError(TradeIQError):
    """Base class for infrastructure-related errors"""
    pass

class ResourceExhaustedError(InfrastructureError):
    """Error raised when a resource is exhausted"""
    
    def __init__(
        self,
        resource: str,
        limit: Any,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        details = details or {}
        details["resource"] = resource
        details["limit"] = limit
        
        super().__init__(
            message=f"Resource {resource} exhausted (limit: {limit})",
            code=ErrorCode.RESOURCE_EXHAUSTED,
            severity=ErrorSeverity.CRITICAL,
            details=details,
            cause=cause,
            context=context
        )

def convert_exception(
    exception: Exception,
    default_message: str = "An unexpected error occurred",
    default_code: ErrorCode = ErrorCode.UNKNOWN_ERROR,
    default_severity: ErrorSeverity = ErrorSeverity.ERROR,
    context: Optional[Dict[str, Any]] = None
) -> TradeIQError:
    """
    Convert a standard exception to a TradeIQError.
    
    Args:
        exception: The exception to convert
        default_message: Default message if the exception has no message
        default_code: Default error code
        default_severity: Default error severity
        context: Optional additional context
        
    Returns:
        Converted TradeIQError
    """
    if isinstance(exception, TradeIQError):
        # If it's already a TradeIQError, just update the context if provided
        if context:
            exception.context.update(context)
        return exception
    
    # Use the exception message if available
    message = str(exception) or default_message
    
    # Create a new TradeIQError
    return TradeIQError(
        message=message,
        code=default_code,
        severity=default_severity,
        cause=exception,
        context=context
    )

# Retry decorator with exponential backoff
def retry(
    max_retries: int = 3,
    retry_delay: float = 1.0,
    backoff_factor: float = 2.0,
    jitter: float = 0.1,
    retry_exceptions: Tuple[Type[Exception], ...] = (Exception,),
    ignore_exceptions: Tuple[Type[Exception], ...] = (),
    on_retry: Optional[Callable[[int, Exception, float], None]] = None
):
    """
    Decorator for retrying functions when exceptions occur.
    
    Args:
        max_retries: Maximum number of retries
        retry_delay: Initial delay between retries in seconds
        backoff_factor: Factor to increase delay with each retry
        jitter: Random jitter factor to add to delay
        retry_exceptions: Tuple of exception types to retry on
        ignore_exceptions: Tuple of exception types to not retry on
        on_retry: Optional callback called before each retry
    
    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        is_async = asyncio.iscoroutinefunction(func)
        
        if is_async:
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                retries = 0
                delay = retry_delay
                
                while True:
                    try:
                        return await func(*args, **kwargs)
                    except ignore_exceptions:
                        # Don't retry these exceptions
                        raise
                    except retry_exceptions as e:
                        retries += 1
                        if retries > max_retries:
                            # Max retries reached, re-raise
                            raise
                        
                        # Calculate the next delay with jitter
                        actual_delay = delay * (1 + random.uniform(-jitter, jitter))
                        
                        # Call the retry callback if provided
                        if on_retry:
                            on_retry(retries, e, actual_delay)
                            
                        # Wait before retrying
                        logger.warning(
                            f"Retry {retries}/{max_retries} for {func.__name__} "
                            f"after {actual_delay:.2f}s due to {type(e).__name__}: {e}"
                        )
                        await asyncio.sleep(actual_delay)
                        
                        # Increase delay for next retry
                        delay *= backoff_factor
            
            return cast(F, async_wrapper)
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                retries = 0
                delay = retry_delay
                
                while True:
                    try:
                        return func(*args, **kwargs)
                    except ignore_exceptions:
                        # Don't retry these exceptions
                        raise
                    except retry_exceptions as e:
                        retries += 1
                        if retries > max_retries:
                            # Max retries reached, re-raise
                            raise
                        
                        # Calculate the next delay with jitter
                        actual_delay = delay * (1 + random.uniform(-jitter, jitter))
                        
                        # Call the retry callback if provided
                        if on_retry:
                            on_retry(retries, e, actual_delay)
                            
                        # Wait before retrying
                        logger.warning(
                            f"Retry {retries}/{max_retries} for {func.__name__} "
                            f"after {actual_delay:.2f}s due to {type(e).__name__}: {e}"
                        )
                        time.sleep(actual_delay)
                        
                        # Increase delay for next retry
                        delay *= backoff_factor
            
            return cast(F, sync_wrapper)
    
    return decorator

# Context manager for tracing errors
class ErrorTracer:
    """
    Context manager for tracing errors.
    
    Catches exceptions, logs them with a given context,
    and re-raises them as TradeIQError instances.
    """
    
    def __init__(
        self,
        operation: str,
        context: Optional[Dict[str, Any]] = None,
        log_level: int = logging.ERROR,
        include_stack_trace: bool = True,
        capture_as: Optional[Type[TradeIQError]] = None
    ):
        """
        Initialize error tracer.
        
        Args:
            operation: Name of the operation being traced
            context: Additional context to include in error reports
            log_level: Logging level for error reports
            include_stack_trace: Whether to include stack traces in logs
            capture_as: Optional TradeIQError subclass to use for capturing
        """
        self.operation = operation
        self.context = context or {}
        self.context["operation"] = operation
        self.log_level = log_level
        self.include_stack_trace = include_stack_trace
        self.capture_as = capture_as
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val is None:
            return False
            
        # Skip if it's a KeyboardInterrupt or SystemExit
        if isinstance(exc_val, (KeyboardInterrupt, SystemExit)):
            return False
            
        # Create a TradeIQError from the exception
        if self.capture_as and issubclass(self.capture_as, TradeIQError):
            # Use the specified error type
            try:
                error = self.capture_as(str(exc_val), cause=exc_val, context=self.context)
            except Exception:
                # Fallback to basic conversion if constructor fails
                error = convert_exception(exc_val, context=self.context)
        else:
            # Use default conversion
            error = convert_exception(exc_val, context=self.context)
            
        # Log the error
        log_message = f"Error in {self.operation}: {error}"
        if self.include_stack_trace:
            log_message += f"\n{traceback.format_exc()}"
            
        logger.log(self.log_level, log_message)
        
        # Re-raise the TradeIQError
        raise error from exc_val

# Async context manager for tracing errors
class AsyncErrorTracer:
    """
    Async context manager for tracing errors.
    
    Catches exceptions, logs them with a given context,
    and re-raises them as TradeIQError instances.
    """
    
    def __init__(
        self,
        operation: str,
        context: Optional[Dict[str, Any]] = None,
        log_level: int = logging.ERROR,
        include_stack_trace: bool = True,
        capture_as: Optional[Type[TradeIQError]] = None
    ):
        """
        Initialize async error tracer.
        
        Args:
            operation: Name of the operation being traced
            context: Additional context to include in error reports
            log_level: Logging level for error reports
            include_stack_trace: Whether to include stack traces in logs
            capture_as: Optional TradeIQError subclass to use for capturing
        """
        self.operation = operation
        self.context = context or {}
        self.context["operation"] = operation
        self.log_level = log_level
        self.include_stack_trace = include_stack_trace
        self.capture_as = capture_as
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_val is None:
            return False
            
        # Skip if it's a KeyboardInterrupt or SystemExit
        if isinstance(exc_val, (KeyboardInterrupt, SystemExit)):
            return False
            
        # Create a TradeIQError from the exception
        if self.capture_as and issubclass(self.capture_as, TradeIQError):
            # Use the specified error type
            try:
                error = self.capture_as(str(exc_val), cause=exc_val, context=self.context)
            except Exception:
                # Fallback to basic conversion if constructor fails
                error = convert_exception(exc_val, context=self.context)
        else:
            # Use default conversion
            error = convert_exception(exc_val, context=self.context)
            
        # Log the error
        log_message = f"Error in {self.operation}: {error}"
        if self.include_stack_trace:
            log_message += f"\n{traceback.format_exc()}"
            
        logger.log(self.log_level, log_message)
        
        # Re-raise the TradeIQError
        raise error from exc_val

def trace_errors(
    operation: str,
    context: Optional[Dict[str, Any]] = None,
    log_level: int = logging.ERROR,
    include_stack_trace: bool = True,
    capture_as: Optional[Type[TradeIQError]] = None
):
    """
    Decorator for tracing errors in functions.
    
    Args:
        operation: Name of the operation being traced
        context: Additional context to include in error reports
        log_level: Logging level for error reports
        include_stack_trace: Whether to include stack traces in logs
        capture_as: Optional TradeIQError subclass to use for capturing
        
    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        is_async = asyncio.iscoroutinefunction(func)
        
        if is_async:
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                func_context = dict(context or {})
                func_context["function"] = func.__name__
                
                tracer = AsyncErrorTracer(
                    operation=operation,
                    context=func_context,
                    log_level=log_level,
                    include_stack_trace=include_stack_trace,
                    capture_as=capture_as
                )
                
                async with tracer:
                    return await func(*args, **kwargs)
            
            return cast(F, async_wrapper)
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                func_context = dict(context or {})
                func_context["function"] = func.__name__
                
                tracer = ErrorTracer(
                    operation=operation,
                    context=func_context,
                    log_level=log_level,
                    include_stack_trace=include_stack_trace,
                    capture_as=capture_as
                )
                
                with tracer:
                    return func(*args, **kwargs)
            
            return cast(F, sync_wrapper)
    
    return decorator

# API response generator
def error_response(
    error: Union[TradeIQError, Exception],
    include_details: bool = True,
    include_stack_trace: bool = False
) -> Dict[str, Any]:
    """
    Generate a standardized API error response.
    
    Args:
        error: The error to generate a response for
        include_details: Whether to include error details
        include_stack_trace: Whether to include stack trace
        
    Returns:
        Standardized error response dictionary
    """
    # Convert to TradeIQError if needed
    if not isinstance(error, TradeIQError):
        error = convert_exception(error)
        
    # Get error info
    error_info = error.to_error_info(include_stack_trace=include_stack_trace)
    
    # Create response
    response = {
        "status": "error",
        "code": error_info.code,
        "message": error_info.message
    }
    
    # Include details if requested
    if include_details and error_info.details:
        response["details"] = error_info.details
        
    return response

# Global error logging function
def log_error(
    error: Union[TradeIQError, Exception],
    level: int = logging.ERROR,
    include_stack_trace: bool = True,
    context: Optional[Dict[str, Any]] = None
) -> None:
    """
    Log an error with standardized format.
    
    Args:
        error: The error to log
        level: Logging level
        include_stack_trace: Whether to include stack trace
        context: Additional context to include
    """
    # Convert to TradeIQError if needed
    if not isinstance(error, TradeIQError):
        error = convert_exception(error, context=context)
    elif context:
        # Update the context
        error.context.update(context)
        
    # Create log message
    message = f"ERROR [{error.code.value}]: {error.message}"
    
    # Add context if provided
    if error.context:
        context_str = ", ".join(f"{k}={v}" for k, v in error.context.items())
        message += f" (context: {context_str})"
        
    # Add cause if available
    if error.cause:
        message += f" caused by {type(error.cause).__name__}: {str(error.cause)}"
        
    # Add stack trace if requested
    if include_stack_trace:
        message += f"\n{traceback.format_exc()}"
        
    # Log the message
    logger.log(level, message) 