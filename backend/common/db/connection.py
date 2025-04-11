"""
Database Configuration Loading

This module provides functions to load database connection settings
from environment variables or configuration files.
"""

import os
import logging
import urllib.parse
from typing import Dict, Any, Optional

from backend.common.logger import app_logger

# Module logger
logger = app_logger.getChild("db.config")

# Environment variable names
DB_TYPE_ENV = "DB_TYPE" # e.g., postgresql, sqlite
DB_HOST_ENV = "DB_HOST"
DB_PORT_ENV = "DB_PORT"
DB_NAME_ENV = "DB_NAME"
DB_USER_ENV = "DB_USER"
DB_PASSWORD_ENV = "DB_PASSWORD"
DB_PATH_ENV = "DB_PATH" # For SQLite
DB_URL_ENV = "DATABASE_URL" # Allow direct URL specification
DB_POOL_SIZE_ENV = "DB_POOL_SIZE"
DB_SSL_MODE_ENV = "DB_SSL_MODE"

# Default values
DEFAULT_DB_TYPE = "postgresql"
DEFAULT_DB_HOST = "localhost"
DEFAULT_DB_PORT = 5432
DEFAULT_DB_NAME = "tradeiq_db"
DEFAULT_DB_USER = "tradeiq_user"
DEFAULT_DB_PASSWORD = "password" # Use strong defaults or require env var
DEFAULT_DB_PATH = "./tradeiq_assessments.db" # For SQLite
DEFAULT_DB_POOL_SIZE = 10
DEFAULT_DB_SSL_MODE = "prefer" # PostgreSQL SSL mode: disable, allow, prefer, require, verify-ca, verify-full


def get_database_settings() -> Dict[str, Any]:
    """
    Loads database connection settings primarily from environment variables.
    Allows overriding with a direct DATABASE_URL.

    Returns:
        A dictionary containing database settings including the connection URL.
    """
    settings = {}
    
    # Allow direct DATABASE_URL override
    database_url = os.environ.get(DB_URL_ENV)
    if database_url:
        logger.info(f"Using direct DATABASE_URL from environment variable.")
        # Parse the URL to extract components if needed, or just store the URL
        # For simplicity here, we'll just store it and assume SQLAlchemy can handle it.
        settings["database_url"] = database_url
        # Try to infer type (basic check)
        if database_url.startswith("postgresql"): 
            settings["db_type"] = "postgresql"
        elif database_url.startswith("sqlite"):
            settings["db_type"] = "sqlite"
        else:
            settings["db_type"] = "unknown"
        settings["pool_size"] = int(os.environ.get(DB_POOL_SIZE_ENV, DEFAULT_DB_POOL_SIZE))
        return settings

    # Load individual components if DATABASE_URL is not set
    settings["db_type"] = os.environ.get(DB_TYPE_ENV, DEFAULT_DB_TYPE).lower()
    settings["host"] = os.environ.get(DB_HOST_ENV, DEFAULT_DB_HOST)
    settings["port"] = int(os.environ.get(DB_PORT_ENV, DEFAULT_DB_PORT))
    settings["database"] = os.environ.get(DB_NAME_ENV, DEFAULT_DB_NAME)
    settings["user"] = os.environ.get(DB_USER_ENV, DEFAULT_DB_USER)
    settings["password"] = os.environ.get(DB_PASSWORD_ENV, DEFAULT_DB_PASSWORD)
    settings["db_path"] = os.environ.get(DB_PATH_ENV, DEFAULT_DB_PATH)
    settings["pool_size"] = int(os.environ.get(DB_POOL_SIZE_ENV, DEFAULT_DB_POOL_SIZE))
    settings["ssl_mode"] = os.environ.get(DB_SSL_MODE_ENV, DEFAULT_DB_SSL_MODE).lower()

    # Construct DATABASE_URL based on type
    db_type = settings["db_type"]
    constructed_url = None

    if db_type == "postgresql":
        user = settings["user"]
        password = urllib.parse.quote_plus(settings["password"])
        host = settings["host"]
        port = settings["port"]
        database = settings["database"]
        # Use asyncpg driver for SQLAlchemy async
        constructed_url = f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{database}"
        # Add SSL mode if relevant and not default 'prefer'
        if settings["ssl_mode"] != "prefer" and settings["ssl_mode"] != "disable":
             # Note: asyncpg uses ssl context object or specific keywords, not simple mode string in URL usually.
             # SQLAlchemy URL might support sslmode=... for libpq sync drivers, but check asyncpg docs.
             # For simplicity, we log the mode but don't add it to URL here, assume engine config handles it.
             logger.info(f"PostgreSQL SSL mode set to '{settings['ssl_mode']}' (check engine options for asyncpg)")
             # Example if adding to URL was supported: constructed_url += f"?sslmode={settings['ssl_mode']}"
    elif db_type == "sqlite":
        # Use aiosqlite driver for SQLAlchemy async
        db_path = settings["db_path"]
        # Ensure the path starts with sqlite:/// or sqlite+aiosqlite:///
        # Use absolute path if needed, ensure it's correct for the OS
        # For async, the path format is usually `sqlite+aiosqlite:///path/to/db.sqlite`
        # Using relative path from env var/default for now.
        constructed_url = f"sqlite+aiosqlite:///{db_path}"
    # Add other DB types like MongoDB here if needed, using appropriate driver string
    # elif db_type == "mongodb":
    #     # Need Motor driver etc.
    #     pass
    else:
        logger.error(f"Unsupported DB_TYPE: {db_type}")
        raise ValueError(f"Unsupported database type: {db_type}")

    settings["database_url"] = constructed_url
    logger.info(f"Constructed database URL for {db_type}")

    return settings 