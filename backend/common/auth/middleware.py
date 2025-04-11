"""
Authentication Middleware

This module provides middleware for authenticating and authorizing API requests,
including functions for JWT token validation and role-based access control.
"""

import functools
import inspect
import threading
from typing import Any, Callable, Dict, List, Optional, Set, TypeVar, Union, cast

from backend.common.auth.exceptions import (
    AuthError,
    InvalidTokenError,
    ExpiredTokenError,
    InsufficientPermissionsError,
    InvalidCredentialsError,
    MissingTokenError
)
from backend.common.auth.jwt import (
    TokenType,
    validate_token,
    get_token_identity
)
from backend.common.auth.user import (
    User,
    UserRole,
    AuthenticatedUser
)

# Type variable for the function return type
T = TypeVar('T')

# Thread-local storage for the current user
_current_user_local = threading.local()


def get_current_user() -> Optional[AuthenticatedUser]:
    """
    Get the authenticated user for the current request context.
    
    Returns:
        The authenticated user or None if not authenticated
    """
    return getattr(_current_user_local, 'user', None)


def set_current_user(user: Optional[AuthenticatedUser]) -> None:
    """
    Set the authenticated user for the current request context.
    
    Args:
        user: The authenticated user to set
    """
    _current_user_local.user = user


def extract_token_from_header(auth_header: Optional[str]) -> Optional[str]:
    """
    Extract a JWT token from an Authorization header.
    
    Args:
        auth_header: The Authorization header value
        
    Returns:
        The JWT token or None if not found
    """
    if not auth_header:
        return None
    
    parts = auth_header.split()
    
    if len(parts) != 2 or parts[0].lower() != 'bearer':
        return None
    
    return parts[1]


def authenticate(auth_header: Optional[str]) -> AuthenticatedUser:
    """
    Authenticate a request using the Authorization header.
    
    Args:
        auth_header: The Authorization header value
        
    Returns:
        An authenticated user object
        
    Raises:
        MissingTokenError: If no token is provided
        InvalidTokenError: If the token is invalid
        ExpiredTokenError: If the token has expired
    """
    # Extract token from header
    token = extract_token_from_header(auth_header)
    
    if not token:
        raise MissingTokenError()
    
    # Validate the token
    payload = validate_token(token, TokenType.ACCESS)
    
    # Get the user ID from the token
    user_id = get_token_identity(token)
    
    # In a real application, this would fetch the user from a database
    # For now, we'll create a mock user with data from the token
    user = User(
        id=user_id,
        email=payload.get("email", f"{user_id}@example.com"),
        username=payload.get("username", f"user_{user_id}"),
        role=UserRole(payload.get("role", UserRole.STUDENT.value)),
    )
    
    # Create an authenticated user
    auth_user = AuthenticatedUser(
        user=user,
        access_token=token,
        permissions=set(payload.get("permissions", []))
    )
    
    # Store in thread-local storage
    set_current_user(auth_user)
    
    return auth_user


def require_auth(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator to require authentication for a function.
    
    Args:
        func: The function to decorate
        
    Returns:
        The decorated function
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Get the current user
        user = get_current_user()
        
        # If no user is set, we need to get the auth header and authenticate
        if user is None:
            # Try to get the auth header from kwargs
            auth_header = kwargs.get('authorization')
            
            # If not in kwargs, try to get it from the first argument
            # This works for frameworks where the request is the first argument
            if auth_header is None and args:
                first_arg = args[0]
                
                # Try to get the header from different attributes
                # that might exist on the request object
                if hasattr(first_arg, 'headers'):
                    auth_header = first_arg.headers.get('Authorization')
                elif hasattr(first_arg, 'META'):
                    auth_header = first_arg.META.get('HTTP_AUTHORIZATION')
                
            # Authenticate the request
            if auth_header:
                user = authenticate(auth_header)
                
            # If we still don't have a user, raise an error
            if user is None:
                raise MissingTokenError()
        
        # Update kwargs with the authenticated user
        kwargs['current_user'] = user
        
        # Call the original function
        return func(*args, **kwargs)
    
    return wrapper


def require_role(
    *roles: Union[UserRole, str],
    allow_admin: bool = True
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator to require specific roles for a function.
    
    Args:
        *roles: The roles to require
        allow_admin: Whether to always allow admin users
        
    Returns:
        A decorator function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        @require_auth
        def wrapper(*args, **kwargs):
            # Get the current user
            user = kwargs.get('current_user')
            
            # Convert string roles to UserRole enums
            required_roles = set()
            for role in roles:
                if isinstance(role, str):
                    required_roles.add(UserRole(role))
                else:
                    required_roles.add(role)
            
            # Check if the user has at least one of the required roles
            # or is an admin (if allow_admin is True)
            if (
                (allow_admin and user.is_admin) or
                user.role in required_roles
            ):
                return func(*args, **kwargs)
            
            raise InsufficientPermissionsError()
        
        return wrapper
    
    return decorator


def require_permission(
    *permissions: str
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator to require specific permissions for a function.
    
    Args:
        *permissions: The permissions to require
        
    Returns:
        A decorator function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        @require_auth
        def wrapper(*args, **kwargs):
            # Get the current user
            user = kwargs.get('current_user')
            
            # Check if the user has all the required permissions
            for permission in permissions:
                if not user.has_permission(permission):
                    raise InsufficientPermissionsError()
            
            return func(*args, **kwargs)
        
        return wrapper
    
    return decorator


def optional_auth(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator to optionally authenticate a request.
    
    This will attempt to authenticate the request but will not fail
    if authentication fails.
    
    Args:
        func: The function to decorate
        
    Returns:
        The decorated function
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Try to get the current user
        user = get_current_user()
        
        # If no user is set, we need to get the auth header and authenticate
        if user is None:
            # Try to get the auth header from kwargs
            auth_header = kwargs.get('authorization')
            
            # If not in kwargs, try to get it from the first argument
            # This works for frameworks where the request is the first argument
            if auth_header is None and args:
                first_arg = args[0]
                
                # Try to get the header from different attributes
                # that might exist on the request object
                if hasattr(first_arg, 'headers'):
                    auth_header = first_arg.headers.get('Authorization')
                elif hasattr(first_arg, 'META'):
                    auth_header = first_arg.META.get('HTTP_AUTHORIZATION')
                
            # Try to authenticate the request
            if auth_header:
                try:
                    user = authenticate(auth_header)
                except AuthError:
                    user = None
        
        # Update kwargs with the authenticated user (or None)
        kwargs['current_user'] = user
        
        # Call the original function
        return func(*args, **kwargs)
    
    return wrapper