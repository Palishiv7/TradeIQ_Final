"""
Centralized Configuration for TradeIQ

This module provides a unified configuration system for the entire TradeIQ platform.
It handles configuration from environment variables, config files, and defaults,
with proper type checking and validation.
"""

import os
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union, get_type_hints
from pydantic import BaseModel, Field, validator
import yaml

# Configure logging
logger = logging.getLogger(__name__)

class DatabaseConfig(BaseModel):
    """Database configuration"""
    host: str = Field(default="localhost", env="DB_HOST")
    port: int = Field(default=5432, env="DB_PORT")
    username: str = Field(default="postgres", env="DB_USERNAME")
    password: str = Field(default="postgres", env="DB_PASSWORD")
    database: str = Field(default="tradeiq", env="DB_DATABASE")
    min_connections: int = Field(default=5, env="DB_MIN_CONNECTIONS")
    max_connections: int = Field(default=20, env="DB_MAX_CONNECTIONS")
    connection_timeout: int = Field(default=30, env="DB_CONNECTION_TIMEOUT")
    query_timeout: int = Field(default=60, env="DB_QUERY_TIMEOUT")
    use_ssl: bool = Field(default=False, env="DB_USE_SSL")
    
    @property
    def connection_string(self) -> str:
        """Get the database connection string"""
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"

class RedisConfig(BaseModel):
    """Redis configuration"""
    host: str = Field(default="localhost", env="REDIS_HOST")
    port: int = Field(default=6379, env="REDIS_PORT")
    db: int = Field(default=0, env="REDIS_DB")
    password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    use_ssl: bool = Field(default=False, env="REDIS_USE_SSL")
    connection_timeout: int = Field(default=10, env="REDIS_CONNECTION_TIMEOUT")
    socket_keepalive: bool = Field(default=True, env="REDIS_SOCKET_KEEPALIVE")
    
    @property
    def connection_string(self) -> str:
        """Get the Redis connection string"""
        protocol = "rediss" if self.use_ssl else "redis"
        auth = f":{self.password}@" if self.password else ""
        return f"{protocol}://{auth}{self.host}:{self.port}/{self.db}"

class ThreadPoolConfig(BaseModel):
    """Thread pool configuration"""
    min_workers: int = Field(default=4, env="THREAD_POOL_MIN_WORKERS")
    max_workers: Optional[int] = Field(default=None, env="THREAD_POOL_MAX_WORKERS")
    queue_size: int = Field(default=1000, env="THREAD_POOL_QUEUE_SIZE")
    adaptive: bool = Field(default=True, env="THREAD_POOL_ADAPTIVE")
    cpu_target: float = Field(default=0.75, env="THREAD_POOL_CPU_TARGET")
    thread_idle_timeout: float = Field(default=60.0, env="THREAD_POOL_IDLE_TIMEOUT")

class CacheConfig(BaseModel):
    """Cache configuration"""
    enabled: bool = Field(default=True, env="CACHE_ENABLED")
    default_ttl: int = Field(default=300, env="CACHE_DEFAULT_TTL")  # 5 minutes
    use_redis: bool = Field(default=False, env="CACHE_USE_REDIS")
    memory_max_size: int = Field(default=10000, env="CACHE_MEMORY_MAX_SIZE")
    redis_key_prefix: str = Field(default="tradeiq:", env="CACHE_REDIS_PREFIX")

class LoggingConfig(BaseModel):
    """Logging configuration"""
    level: str = Field(default="INFO", env="LOG_LEVEL")
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        env="LOG_FORMAT"
    )
    file_path: Optional[str] = Field(default=None, env="LOG_FILE_PATH")
    rotation_size: str = Field(default="10MB", env="LOG_ROTATION_SIZE")
    max_backups: int = Field(default=5, env="LOG_MAX_BACKUPS")
    
    @validator('level')
    def validate_level(cls, v):
        """Validate log level"""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log level: {v}. Must be one of {valid_levels}")
        return v.upper()

class SecurityConfig(BaseModel):
    """Security configuration"""
    secret_key: str = Field(default="CHANGE_ME_IN_PRODUCTION", env="SECRET_KEY")
    token_expire_minutes: int = Field(default=60, env="TOKEN_EXPIRE_MINUTES")
    algorithm: str = Field(default="HS256", env="SECURITY_ALGORITHM")
    allow_origins: List[str] = Field(default=["*"], env="ALLOW_ORIGINS")
    rate_limit_enabled: bool = Field(default=True, env="RATE_LIMIT_ENABLED")
    rate_limit_requests: int = Field(default=100, env="RATE_LIMIT_REQUESTS")
    rate_limit_window: int = Field(default=60, env="RATE_LIMIT_WINDOW")  # seconds

class APIConfig(BaseModel):
    """API configuration"""
    host: str = Field(default="0.0.0.0", env="API_HOST") 
    port: int = Field(default=8000, env="API_PORT")
    workers: int = Field(default=4, env="API_WORKERS")
    debug: bool = Field(default=False, env="API_DEBUG")
    reload: bool = Field(default=False, env="API_RELOAD")
    prefix: str = Field(default="/api", env="API_PREFIX")
    docs_url: str = Field(default="/docs", env="API_DOCS_URL")
    openapi_url: str = Field(default="/openapi.json", env="API_OPENAPI_URL")
    redoc_url: str = Field(default="/redoc", env="API_REDOC_URL")

class AssessmentConfig(BaseModel):
    """Assessment configuration"""
    default_question_count: int = Field(default=10, env="ASSESSMENT_DEFAULT_QUESTIONS")
    default_time_limit_seconds: int = Field(default=60, env="ASSESSMENT_DEFAULT_TIME_LIMIT")
    session_expiry_hours: int = Field(default=24, env="ASSESSMENT_SESSION_EXPIRY_HOURS")
    default_difficulty: float = Field(default=0.5, env="ASSESSMENT_DEFAULT_DIFFICULTY")
    adaptive_difficulty: bool = Field(default=True, env="ASSESSMENT_ADAPTIVE_DIFFICULTY")
    cache_questions: bool = Field(default=True, env="ASSESSMENT_CACHE_QUESTIONS")
    question_cache_ttl: int = Field(default=3600, env="ASSESSMENT_QUESTION_CACHE_TTL")
    explanation_detail_level: int = Field(default=2, env="ASSESSMENT_EXPLANATION_DETAIL_LEVEL")
    max_concurrent_assessments: int = Field(default=5, env="ASSESSMENT_MAX_CONCURRENT")
    
    @validator('default_difficulty')
    def validate_difficulty(cls, v):
        """Validate difficulty is between 0 and 1"""
        if not 0 <= v <= 1:
            raise ValueError(f"Difficulty must be between 0 and 1, got {v}")
        return v

class AIConfig(BaseModel):
    """AI model configuration"""
    provider: str = Field(default="openai", env="AI_PROVIDER")
    model_name: str = Field(default="gpt-4", env="AI_MODEL_NAME")
    api_key: Optional[str] = Field(default=None, env="AI_API_KEY")
    api_base: Optional[str] = Field(default=None, env="AI_API_BASE")
    timeout: int = Field(default=30, env="AI_TIMEOUT")
    max_retries: int = Field(default=3, env="AI_MAX_RETRIES")
    retry_delay: float = Field(default=1.0, env="AI_RETRY_DELAY")
    backoff_factor: float = Field(default=2.0, env="AI_BACKOFF_FACTOR")
    request_batch_size: int = Field(default=5, env="AI_REQUEST_BATCH_SIZE")
    use_threading: bool = Field(default=True, env="AI_USE_THREADING")

class EnvironmentConfig(BaseModel):
    """Environment configuration"""
    env: str = Field(default="development", env="ENV")
    testing: bool = Field(default=False, env="TESTING")
    debug: bool = Field(default=False, env="DEBUG")
    
    @validator('env')
    def validate_env(cls, v):
        """Validate environment"""
        valid_envs = ['development', 'testing', 'staging', 'production']
        if v.lower() not in valid_envs:
            raise ValueError(f"Invalid environment: {v}. Must be one of {valid_envs}")
        return v.lower()

class AppConfig(BaseModel):
    """Main application configuration"""
    app_name: str = Field(default="TradeIQ", env="APP_NAME")
    version: str = Field(default="1.0.0", env="APP_VERSION")
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    thread_pool: ThreadPoolConfig = Field(default_factory=ThreadPoolConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    assessment: AssessmentConfig = Field(default_factory=AssessmentConfig)
    ai: AIConfig = Field(default_factory=AIConfig)
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    
    @property
    def is_development(self) -> bool:
        """Check if environment is development"""
        return self.environment.env == "development"
    
    @property
    def is_testing(self) -> bool:
        """Check if environment is testing"""
        return self.environment.env == "testing" or self.environment.testing
    
    @property
    def is_production(self) -> bool:
        """Check if environment is production"""
        return self.environment.env == "production"
    
    @property
    def is_staging(self) -> bool:
        """Check if environment is staging"""
        return self.environment.env == "staging"

class ConfigLoader:
    """
    Configuration loader for the application.
    
    Loads configuration from:
    1. Default values
    2. Config file
    3. Environment variables (highest priority)
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the config loader.
        
        Args:
            config_path: Path to config file (YAML or JSON)
        """
        self.config_path = config_path or os.environ.get("CONFIG_PATH")
        self._config = None
    
    def load(self) -> AppConfig:
        """
        Load configuration from all sources.
        
        Returns:
            Loaded configuration
        """
        if self._config is not None:
            return self._config
            
        # Load from file if specified
        file_config = {}
        if self.config_path:
            file_config = self._load_from_file(self.config_path)
            
        # Load from environment and merge with file config
        self._config = AppConfig(**file_config)
        return self._config
    
    def _load_from_file(self, path: str) -> Dict[str, Any]:
        """
        Load configuration from a file.
        
        Args:
            path: Path to config file
            
        Returns:
            Loaded configuration dictionary
        """
        path = Path(path)
        if not path.exists():
            logger.warning(f"Config file not found: {path}")
            return {}
            
        try:
            if path.suffix.lower() in ['.yaml', '.yml']:
                with open(path, 'r') as f:
                    return yaml.safe_load(f)
            elif path.suffix.lower() == '.json':
                with open(path, 'r') as f:
                    return json.load(f)
            else:
                logger.warning(f"Unsupported config file format: {path.suffix}")
                return {}
        except Exception as e:
            logger.error(f"Error loading config file: {e}")
            return {}

# Global configuration instance
config_loader = ConfigLoader()
config = config_loader.load()

def get_config() -> AppConfig:
    """
    Get the loaded configuration.
    
    Returns:
        Loaded configuration
    """
    return config

def reload_config(config_path: Optional[str] = None) -> AppConfig:
    """
    Reload the configuration.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Reloaded configuration
    """
    global config_loader, config
    config_loader = ConfigLoader(config_path)
    config = config_loader.load()
    return config
