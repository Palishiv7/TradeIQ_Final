"""Application configuration module."""

import os
from typing import Optional
from pydantic import BaseSettings

class Settings(BaseSettings):
    """Application settings."""
    
    # Database settings
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/tradeiq"
    SQL_ECHO: bool = False
    DB_POOL_SIZE: int = 5
    DB_MAX_OVERFLOW: int = 10
    DB_POOL_TIMEOUT: int = 30
    
    # Logging settings
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # API settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "TradeIQ Assessments"
    
    class Config:
        """Pydantic settings config."""
        env_file = ".env"
        case_sensitive = True

# Create global settings instance
settings = Settings() 