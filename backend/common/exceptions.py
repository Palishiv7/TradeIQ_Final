"""
Common Exception Classes

This module defines custom exceptions used throughout the application.
"""

from typing import Optional, Any


class BaseError(Exception):
    """Base class for all custom exceptions."""
    
    def __init__(self, message: str, original_exception: Optional[Exception] = None):
        """
        Initialize the exception.
        
        Args:
            message: Error message
            original_exception: Original exception that caused this error
        """
        super().__init__(message)
        self.message = message
        self.original_exception = original_exception


class DatabaseError(BaseError):
    """Exception raised for database-related errors."""
    
    def __init__(self, message: str, original_exception: Optional[Exception] = None):
        """
        Initialize the database error.
        
        Args:
            message: Error message
            original_exception: Original database exception
        """
        super().__init__(f"Database error: {message}", original_exception)


class CacheError(BaseError):
    """Exception raised for cache-related errors."""
    
    def __init__(self, message: str, original_exception: Optional[Exception] = None):
        """
        Initialize the cache error.
        
        Args:
            message: Error message
            original_exception: Original cache exception
        """
        super().__init__(f"Cache error: {message}", original_exception)


class ValidationError(BaseError):
    """Exception raised for validation errors."""
    
    def __init__(self, message: str, errors: Optional[dict] = None):
        """
        Initialize the validation error.
        
        Args:
            message: Error message
            errors: Dictionary of validation errors
        """
        super().__init__(f"Validation error: {message}")
        self.errors = errors or {}


class ConfigurationError(BaseError):
    """Exception raised for configuration-related errors."""
    
    def __init__(self, message: str, config_key: Optional[str] = None):
        """
        Initialize the configuration error.
        
        Args:
            message: Error message
            config_key: The configuration key that caused the error
        """
        super().__init__(f"Configuration error: {message}")
        self.config_key = config_key


class AuthenticationError(BaseError):
    """Exception raised for authentication-related errors."""
    
    def __init__(self, message: str, user_id: Optional[str] = None):
        """
        Initialize the authentication error.
        
        Args:
            message: Error message
            user_id: ID of the user that failed authentication
        """
        super().__init__(f"Authentication error: {message}")
        self.user_id = user_id


class AuthorizationError(BaseError):
    """Exception raised for authorization-related errors."""
    
    def __init__(self, message: str, resource: Optional[str] = None, action: Optional[str] = None):
        """
        Initialize the authorization error.
        
        Args:
            message: Error message
            resource: The resource that was being accessed
            action: The action that was being attempted
        """
        super().__init__(f"Authorization error: {message}")
        self.resource = resource
        self.action = action


class NotFoundError(BaseError):
    """Exception raised when a resource is not found."""
    
    def __init__(self, resource_type: str, resource_id: Any):
        """
        Initialize the not found error.
        
        Args:
            resource_type: Type of resource that wasn't found
            resource_id: ID of the resource that wasn't found
        """
        super().__init__(f"{resource_type} with ID {resource_id} not found")
        self.resource_type = resource_type
        self.resource_id = resource_id


class DuplicateError(BaseError):
    """Exception raised when attempting to create a duplicate resource."""
    
    def __init__(self, resource_type: str, identifier: Any):
        """
        Initialize the duplicate error.
        
        Args:
            resource_type: Type of resource that was duplicated
            identifier: The identifier that caused the duplicate
        """
        super().__init__(f"Duplicate {resource_type} with identifier {identifier}")
        self.resource_type = resource_type
        self.identifier = identifier 