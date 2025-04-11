"""
Authentication User Models

This module defines the user models for authentication, including base user class,
roles, and statuses.
"""

import datetime
import enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from backend.common.serialization import SerializableMixin


class UserRole(enum.Enum):
    """User roles for authorization."""
    
    ADMIN = "admin"
    INSTRUCTOR = "instructor"
    STUDENT = "student"
    GUEST = "guest"


class UserStatus(enum.Enum):
    """User account statuses."""
    
    ACTIVE = "active"
    INACTIVE = "inactive"
    PENDING = "pending"
    LOCKED = "locked"
    DISABLED = "disabled"


@dataclass
class User(SerializableMixin):
    """
    Base user model with authentication and authorization information.
    
    Attributes:
        id: Unique user identifier
        email: User's email address
        username: User's username
        role: User's role
        status: User's account status
        created_at: Timestamp when the user was created
        updated_at: Timestamp when the user was last updated
        last_login_at: Timestamp of the user's last login
        first_name: User's first name
        last_name: User's last name
        profile_picture: URL to the user's profile picture
        metadata: Additional metadata about the user
    """
    id: str
    email: str
    username: str
    role: UserRole = UserRole.STUDENT
    status: UserStatus = UserStatus.ACTIVE
    created_at: datetime.datetime = field(default_factory=datetime.datetime.utcnow)
    updated_at: Optional[datetime.datetime] = None
    last_login_at: Optional[datetime.datetime] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    profile_picture: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def full_name(self) -> str:
        """Get the user's full name."""
        if self.first_name and self.last_name:
            return f"{self.first_name} {self.last_name}"
        if self.first_name:
            return self.first_name
        if self.last_name:
            return self.last_name
        return self.username
    
    @property
    def is_active(self) -> bool:
        """Check if the user account is active."""
        return self.status == UserStatus.ACTIVE
    
    @property
    def is_admin(self) -> bool:
        """Check if the user has admin role."""
        return self.role == UserRole.ADMIN
    
    @property
    def is_instructor(self) -> bool:
        """Check if the user has instructor role."""
        return self.role == UserRole.INSTRUCTOR
    
    @property
    def is_student(self) -> bool:
        """Check if the user has student role."""
        return self.role == UserRole.STUDENT
    
    def has_permission(self, permission: str) -> bool:
        """
        Check if the user has a specific permission.
        
        This is a placeholder for a more sophisticated permission system.
        Currently, it just checks basic roles.
        
        Args:
            permission: The permission to check
            
        Returns:
            True if the user has the permission, False otherwise
        """
        # Admin has all permissions
        if self.is_admin:
            return True
        
        # For now, just use a simple mapping of roles to permissions
        if permission == "view_assessments":
            return True
        elif permission == "create_assessments":
            return self.is_admin or self.is_instructor
        elif permission == "grade_assessments":
            return self.is_admin or self.is_instructor
        elif permission == "manage_users":
            return self.is_admin
        
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the user to a dictionary representation."""
        return {
            "id": self.id,
            "email": self.email,
            "username": self.username,
            "role": self.role.value,
            "status": self.status.value,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "last_login_at": self.last_login_at.isoformat() if self.last_login_at else None,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "full_name": self.full_name,
            "profile_picture": self.profile_picture,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'User':
        """Create a user from a dictionary representation."""
        role = UserRole(data.get("role", UserRole.STUDENT.value))
        status = UserStatus(data.get("status", UserStatus.ACTIVE.value))
        
        created_at = data.get("created_at")
        if created_at and isinstance(created_at, str):
            created_at = datetime.datetime.fromisoformat(created_at)
        
        updated_at = data.get("updated_at")
        if updated_at and isinstance(updated_at, str):
            updated_at = datetime.datetime.fromisoformat(updated_at)
        
        last_login_at = data.get("last_login_at")
        if last_login_at and isinstance(last_login_at, str):
            last_login_at = datetime.datetime.fromisoformat(last_login_at)
        
        return cls(
            id=data["id"],
            email=data["email"],
            username=data["username"],
            role=role,
            status=status,
            created_at=created_at,
            updated_at=updated_at,
            last_login_at=last_login_at,
            first_name=data.get("first_name"),
            last_name=data.get("last_name"),
            profile_picture=data.get("profile_picture"),
            metadata=data.get("metadata", {})
        )
    
    @property
    def serializable_fields(self) -> List[str]:
        """Get a list of fields that should be serialized."""
        return [
            "id", "email", "username", "role", "status", "created_at",
            "updated_at", "last_login_at", "first_name", "last_name",
            "profile_picture", "metadata"
        ]


@dataclass
class AuthenticatedUser:
    """
    User with authentication context.
    
    This class extends the base User model with authentication-specific
    information like tokens and session data.
    
    Attributes:
        user: The base user
        access_token: JWT access token
        refresh_token: JWT refresh token
        token_expiry: When the access token expires
        permissions: Set of permissions granted to the user
        session_id: Unique session identifier
    """
    user: User
    access_token: str
    refresh_token: Optional[str] = None
    token_expiry: Optional[datetime.datetime] = None
    permissions: Set[str] = field(default_factory=set)
    session_id: Optional[str] = None
    
    @property
    def id(self) -> str:
        """Get the user's ID."""
        return self.user.id
    
    @property
    def email(self) -> str:
        """Get the user's email."""
        return self.user.email
    
    @property
    def username(self) -> str:
        """Get the user's username."""
        return self.user.username
    
    @property
    def role(self) -> UserRole:
        """Get the user's role."""
        return self.user.role
    
    @property
    def status(self) -> UserStatus:
        """Get the user's status."""
        return self.user.status
    
    @property
    def is_active(self) -> bool:
        """Check if the user is active."""
        return self.user.is_active
    
    @property
    def is_admin(self) -> bool:
        """Check if the user is an admin."""
        return self.user.is_admin
    
    def has_permission(self, permission: str) -> bool:
        """
        Check if the user has a specific permission.
        
        Args:
            permission: The permission to check
            
        Returns:
            True if the user has the permission, False otherwise
        """
        # Check explicit permissions first
        if permission in self.permissions:
            return True
        
        # Fall back to role-based permissions
        return self.user.has_permission(permission)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the authenticated user to a dictionary representation."""
        return {
            "user": self.user.to_dict(),
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "token_expiry": self.token_expiry.isoformat() if self.token_expiry else None,
            "permissions": list(self.permissions),
            "session_id": self.session_id
        } 