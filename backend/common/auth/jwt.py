"""
JWT Authentication Module

This module provides utilities for JWT-based authentication, including token creation,
validation, and decoding. It supports access and refresh tokens with configurable
expiration times and signing algorithms.
"""

import datetime
import enum
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union, List, Tuple

# Using PyJWT for JWT operations
import jwt

from backend.common.auth.exceptions import (
    AuthError,
    InvalidTokenError,
    ExpiredTokenError
)


class TokenType(enum.Enum):
    """Types of JWT tokens supported by the system."""
    
    ACCESS = "access"
    REFRESH = "refresh"
    RESET = "reset"
    INVITE = "invite"


@dataclass
class JWTConfig:
    """
    Configuration for JWT tokens.
    
    Attributes:
        secret_key: Secret key used for signing tokens
        algorithm: Algorithm used for signing tokens
        access_token_expires: Access token expiration time in minutes
        refresh_token_expires: Refresh token expiration time in days
        reset_token_expires: Password reset token expiration time in hours
        token_issuer: Issuer of the tokens
    """
    secret_key: str
    algorithm: str = "HS256"
    access_token_expires: int = 60  # minutes
    refresh_token_expires: int = 7  # days
    reset_token_expires: int = 24  # hours
    token_issuer: str = "tradeiq-api"


# Global JWT configuration
_jwt_config = JWTConfig(
    secret_key=os.environ.get("JWT_SECRET_KEY", "dev-secret-key"),
)


def set_jwt_config(config: JWTConfig) -> None:
    """
    Set the global JWT configuration.
    
    Args:
        config: The JWT configuration to use
    """
    global _jwt_config
    _jwt_config = config


def get_jwt_config() -> JWTConfig:
    """
    Get the current JWT configuration.
    
    Returns:
        The current JWT configuration
    """
    return _jwt_config


def create_access_token(
    subject: Union[str, int],
    additional_claims: Optional[Dict[str, Any]] = None,
    expires_in: Optional[int] = None
) -> str:
    """
    Create a new JWT access token.
    
    Args:
        subject: The subject of the token (typically a user ID)
        additional_claims: Additional claims to include in the token
        expires_in: Token expiration time in minutes (overrides config)
        
    Returns:
        The JWT access token as a string
    """
    config = get_jwt_config()
    
    # Set expiration time
    now = datetime.datetime.utcnow()
    expires_delta = datetime.timedelta(
        minutes=expires_in if expires_in is not None else config.access_token_expires
    )
    exp = now + expires_delta
    
    # Create token payload
    payload = {
        "sub": str(subject),
        "exp": exp,
        "iat": now,
        "iss": config.token_issuer,
        "type": TokenType.ACCESS.value
    }
    
    # Add additional claims
    if additional_claims:
        payload.update(additional_claims)
    
    # Encode token
    return jwt.encode(
        payload,
        config.secret_key,
        algorithm=config.algorithm
    )


def create_refresh_token(
    subject: Union[str, int],
    additional_claims: Optional[Dict[str, Any]] = None,
    expires_in: Optional[int] = None
) -> str:
    """
    Create a new JWT refresh token.
    
    Args:
        subject: The subject of the token (typically a user ID)
        additional_claims: Additional claims to include in the token
        expires_in: Token expiration time in days (overrides config)
        
    Returns:
        The JWT refresh token as a string
    """
    config = get_jwt_config()
    
    # Set expiration time
    now = datetime.datetime.utcnow()
    expires_delta = datetime.timedelta(
        days=expires_in if expires_in is not None else config.refresh_token_expires
    )
    exp = now + expires_delta
    
    # Create token payload
    payload = {
        "sub": str(subject),
        "exp": exp,
        "iat": now,
        "iss": config.token_issuer,
        "type": TokenType.REFRESH.value
    }
    
    # Add additional claims
    if additional_claims:
        payload.update(additional_claims)
    
    # Encode token
    return jwt.encode(
        payload,
        config.secret_key,
        algorithm=config.algorithm
    )


def create_token(
    subject: Union[str, int],
    token_type: TokenType,
    additional_claims: Optional[Dict[str, Any]] = None,
    expires_in: Optional[int] = None
) -> str:
    """
    Create a new JWT token of a specific type.
    
    Args:
        subject: The subject of the token (typically a user ID)
        token_type: The type of token to create
        additional_claims: Additional claims to include in the token
        expires_in: Token expiration time (units depend on token_type)
        
    Returns:
        The JWT token as a string
        
    Raises:
        ValueError: If an unsupported token type is provided
    """
    if token_type == TokenType.ACCESS:
        return create_access_token(subject, additional_claims, expires_in)
    elif token_type == TokenType.REFRESH:
        return create_refresh_token(subject, additional_claims, expires_in)
    elif token_type == TokenType.RESET:
        config = get_jwt_config()
        
        # Set expiration time
        now = datetime.datetime.utcnow()
        expires_delta = datetime.timedelta(
            hours=expires_in if expires_in is not None else config.reset_token_expires
        )
        exp = now + expires_delta
        
        # Create token payload
        payload = {
            "sub": str(subject),
            "exp": exp,
            "iat": now,
            "iss": config.token_issuer,
            "type": token_type.value
        }
        
        # Add additional claims
        if additional_claims:
            payload.update(additional_claims)
        
        # Encode token
        return jwt.encode(
            payload,
            config.secret_key,
            algorithm=config.algorithm
        )
    else:
        raise ValueError(f"Unsupported token type: {token_type}")


def decode_token(token: str) -> Dict[str, Any]:
    """
    Decode a JWT token without validation.
    
    Args:
        token: The JWT token to decode
        
    Returns:
        The decoded token payload
        
    Raises:
        InvalidTokenError: If the token is invalid or malformed
    """
    try:
        # Decode without verification
        return jwt.decode(
            token,
            options={"verify_signature": False}
        )
    except jwt.PyJWTError as e:
        raise InvalidTokenError(f"Invalid token format: {str(e)}")


def validate_token(
    token: str,
    expected_type: Optional[TokenType] = None
) -> Dict[str, Any]:
    """
    Validate a JWT token and return its payload.
    
    Args:
        token: The JWT token to validate
        expected_type: The expected token type (if not None)
        
    Returns:
        The decoded and validated token payload
        
    Raises:
        InvalidTokenError: If the token is invalid or of the wrong type
        ExpiredTokenError: If the token has expired
    """
    config = get_jwt_config()
    
    try:
        # Verify token signature and validity
        payload = jwt.decode(
            token,
            config.secret_key,
            algorithms=[config.algorithm],
            options={
                "verify_signature": True,
                "verify_exp": True,
                "verify_iat": True,
                "require": ["exp", "iat", "sub", "type"]
            }
        )
        
        # Check token type if expected_type is provided
        if expected_type is not None:
            token_type = payload.get("type")
            if token_type != expected_type.value:
                raise InvalidTokenError(
                    f"Invalid token type: expected {expected_type.value}, got {token_type}"
                )
        
        return payload
    except jwt.ExpiredSignatureError:
        raise ExpiredTokenError("Token has expired")
    except jwt.PyJWTError as e:
        raise InvalidTokenError(f"Invalid token: {str(e)}")


def get_token_identity(token: str) -> str:
    """
    Extract the subject (identity) from a JWT token.
    
    Args:
        token: The JWT token
        
    Returns:
        The subject from the token
        
    Raises:
        InvalidTokenError: If the token is invalid or doesn't contain a subject
    """
    payload = decode_token(token)
    subject = payload.get("sub")
    
    if not subject:
        raise InvalidTokenError("Token does not contain a subject claim")
    
    return subject


def is_token_valid(token: str, expected_type: Optional[TokenType] = None) -> bool:
    """
    Check if a JWT token is valid.
    
    Args:
        token: The JWT token to check
        expected_type: The expected token type (if not None)
        
    Returns:
        True if the token is valid, False otherwise
    """
    try:
        validate_token(token, expected_type)
        return True
    except (InvalidTokenError, ExpiredTokenError):
        return False 