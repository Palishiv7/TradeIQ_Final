"""
Password Utilities

This module provides utilities for secure password management, including
hashing, verification, and password reset token generation.
"""

import datetime
import hashlib
import os
import secrets
import string
import time
import uuid
from typing import Dict, Optional, Tuple

# Import JWT utilities for password reset tokens
from backend.common.auth.jwt import create_token, TokenType


def hash_password(password: str, salt: Optional[str] = None) -> Tuple[str, str]:
    """
    Hash a password using a secure algorithm.
    
    This function uses PBKDF2 with SHA-256, a salt, and 100,000 iterations
    to securely hash passwords.
    
    Args:
        password: The password to hash
        salt: Optional salt to use (if None, a new salt will be generated)
        
    Returns:
        A tuple of (hashed_password, salt)
    """
    # Generate a salt if none was provided
    if salt is None:
        salt = secrets.token_hex(16)
    
    # Use PBKDF2 with SHA-256, 100,000 iterations
    key = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode(),
        salt.encode(),
        100000,
        dklen=32
    )
    
    # Return the hex-encoded hash and salt
    return key.hex(), salt


def verify_password(password: str, hashed_password: str, salt: str) -> bool:
    """
    Verify that a password matches a stored hash.
    
    Args:
        password: The password to verify
        hashed_password: The stored password hash
        salt: The salt used for the stored hash
        
    Returns:
        True if the password matches, False otherwise
    """
    # Hash the provided password with the same salt
    calculated_hash, _ = hash_password(password, salt)
    
    # Compare the calculated hash with the stored hash
    return secrets.compare_digest(calculated_hash, hashed_password)


def generate_password_reset_token(user_id: str, expires_in: int = 24) -> str:
    """
    Generate a password reset token.
    
    Args:
        user_id: The ID of the user requesting the password reset
        expires_in: Token expiration time in hours (default: 24)
        
    Returns:
        A JWT token for password reset
    """
    # Use JWT for password reset tokens with the RESET token type
    token = create_token(
        subject=user_id,
        token_type=TokenType.RESET,
        additional_claims={
            "purpose": "password_reset",
            "created_at": datetime.datetime.utcnow().isoformat()
        },
        expires_in=expires_in
    )
    
    return token


def validate_password_reset_token(token: str) -> Optional[str]:
    """
    Validate a password reset token and return the user ID.
    
    Args:
        token: The password reset token to validate
        
    Returns:
        The user ID if the token is valid, None otherwise
    """
    from backend.common.auth.jwt import validate_token, get_token_identity, TokenType
    
    try:
        # Validate that the token is a valid RESET token
        payload = validate_token(token, TokenType.RESET)
        
        # Verify that the token was issued for password reset
        if payload.get("purpose") != "password_reset":
            return None
        
        # Return the user ID (subject)
        return get_token_identity(token)
    except Exception:
        return None


def generate_strong_password(length: int = 16) -> str:
    """
    Generate a strong random password.
    
    Args:
        length: The length of the password to generate
        
    Returns:
        A strong random password
    """
    # Define character sets for different types of characters
    lowercase = string.ascii_lowercase
    uppercase = string.ascii_uppercase
    digits = string.digits
    special = "!@#$%^&*()-_=+[]{}|;:,.<>?"
    
    # Ensure at least one of each type
    password = [
        secrets.choice(lowercase),
        secrets.choice(uppercase),
        secrets.choice(digits),
        secrets.choice(special)
    ]
    
    # Fill the rest of the password
    all_chars = lowercase + uppercase + digits + special
    password.extend(secrets.choice(all_chars) for _ in range(length - 4))
    
    # Shuffle the password characters
    secrets.SystemRandom().shuffle(password)
    
    # Convert list to string
    return "".join(password)


def is_password_compromised(password: str) -> bool:
    """
    Check if a password has been compromised (appears in data breaches).
    
    This is a placeholder implementation. In a production environment,
    this should use a service like the "Have I Been Pwned" API to check
    if the password has been compromised.
    
    Args:
        password: The password to check
        
    Returns:
        True if the password is compromised, False otherwise
    """
    # Common dictionary of compromised passwords
    # In a real implementation, this would use an API or database
    common_passwords = {
        "password", "123456", "qwerty", "admin",
        "welcome", "password123", "abc123", "letmein"
    }
    
    return password.lower() in common_passwords 