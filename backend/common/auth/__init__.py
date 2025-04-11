"""
Authentication Framework

This package provides a unified authentication framework for the application,
supporting JWT-based authentication and authorization with role-based access control.
"""

from backend.common.auth.jwt import (
    create_access_token,
    create_refresh_token,
    decode_token,
    validate_token,
    get_token_identity,
    TokenType,
    JWTConfig
)

from backend.common.auth.user import (
    User,
    UserRole,
    UserStatus,
    AuthenticatedUser
)

from backend.common.auth.password import (
    hash_password,
    verify_password,
    generate_password_reset_token,
    validate_password_reset_token
)

from backend.common.auth.middleware import (
    authenticate,
    require_auth,
    require_role,
    get_current_user,
    optional_auth
)

from backend.common.auth.exceptions import (
    AuthError,
    InvalidTokenError,
    ExpiredTokenError,
    InsufficientPermissionsError,
    InvalidCredentialsError
)

from .dependencies import get_current_user_id

# Public API
__all__ = [
    # JWT tokens
    'create_access_token',
    'create_refresh_token',
    'decode_token',
    'validate_token',
    'get_token_identity',
    'TokenType',
    'JWTConfig',
    
    # User models
    'User',
    'UserRole',
    'UserStatus',
    'AuthenticatedUser',
    
    # Password utilities
    'hash_password',
    'verify_password',
    'generate_password_reset_token',
    'validate_password_reset_token',
    
    # Authentication middleware
    'authenticate',
    'require_auth',
    'require_role',
    'get_current_user',
    'optional_auth',
    
    # Exceptions
    'AuthError',
    'InvalidTokenError',
    'ExpiredTokenError',
    'InsufficientPermissionsError',
    'InvalidCredentialsError',

    'get_current_user_id',
] 