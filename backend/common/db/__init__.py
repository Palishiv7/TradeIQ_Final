"""
Database Module

This package provides database connection and ORM functionality for the application.
It supports both SQL and NoSQL databases with a consistent interface.
"""

# Only export the settings function from the refactored connection module
from backend.common.db.connection import (
    get_database_settings
)

from backend.common.db.session import (
    get_session,
    get_db_session,
    AsyncSession,
    Session
)

# Remove imports from the deleted model.py
# from backend.common.db.model import (
#     BaseModel, Field, ForeignKey, relationship, query, 
#     create_tables, drop_tables
# )

# Type alias for backward compatibility
DatabaseSession = Session

__all__ = [
    # Only export the settings function
    'get_database_settings',
    
    # Remove exports related to the old connection management
    # 'DatabaseConnection', 'get_database_connection',
    # 'configure_database', 'close_database_connections',
    
    # Remove exports related to the deleted model.py
    # 'BaseModel', 'Field', 'ForeignKey', 'relationship',
    # 'query', 'create_tables', 'drop_tables',
    
    # Session management
    'get_session',
    'get_db_session',
    'AsyncSession',
    'Session',
    'DatabaseSession',
] 