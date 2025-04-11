"""
Authentication Exceptions

This module defines custom exception classes for authentication and authorization errors.
"""

class AuthError(Exception):
    """Base exception for authentication and authorization errors."""
    
    def __init__(self, message: str = "Authentication error", status_code: int = 401):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


class InvalidTokenError(AuthError):
    """Exception raised when a token is invalid."""
    
    def __init__(self, message: str = "Invalid token"):
        super().__init__(message, status_code=401)


class ExpiredTokenError(AuthError):
    """Exception raised when a token has expired."""
    
    def __init__(self, message: str = "Token has expired"):
        super().__init__(message, status_code=401)


class InvalidCredentialsError(AuthError):
    """Exception raised when credentials are invalid."""
    
    def __init__(self, message: str = "Invalid credentials"):
        super().__init__(message, status_code=401)


class InsufficientPermissionsError(AuthError):
    """Exception raised when a user does not have sufficient permissions."""
    
    def __init__(self, message: str = "Insufficient permissions"):
        super().__init__(message, status_code=403)


class UserNotFoundError(AuthError):
    """Exception raised when a user is not found."""
    
    def __init__(self, message: str = "User not found"):
        super().__init__(message, status_code=404)


class UserAlreadyExistsError(AuthError):
    """Exception raised when attempting to create a user that already exists."""
    
    def __init__(self, message: str = "User already exists"):
        super().__init__(message, status_code=409)


class AccountLockedError(AuthError):
    """Exception raised when a user account is locked."""
    
    def __init__(self, message: str = "Account is locked"):
        super().__init__(message, status_code=403)


class AccountDisabledError(AuthError):
    """Exception raised when a user account is disabled."""
    
    def __init__(self, message: str = "Account is disabled"):
        super().__init__(message, status_code=403)


class TokenRefreshError(AuthError):
    """Exception raised when a refresh token cannot be used."""
    
    def __init__(self, message: str = "Could not refresh token"):
        super().__init__(message, status_code=401)


class MissingTokenError(AuthError):
    """Exception raised when a required token is missing."""
    
    def __init__(self, message: str = "Authentication token is missing"):
        super().__init__(message, status_code=401) 