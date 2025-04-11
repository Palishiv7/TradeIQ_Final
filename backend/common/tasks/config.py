"""
Task Configuration Module

This module provides configuration settings for the task scheduling system,
including broker settings, worker configurations, and scheduling options.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

# Default RabbitMQ connection settings
DEFAULT_BROKER_URL = "amqp://guest:guest@localhost:5672//"
DEFAULT_RESULT_BACKEND = "rpc://"


@dataclass
class TaskConfig:
    """
    Configuration for the task scheduling system.
    
    Attributes:
        broker_url: URL for the message broker (e.g., RabbitMQ, Redis)
        result_backend: URL for the result backend
        worker_concurrency: Number of worker processes
        task_serializer: Format for serializing task messages
        result_serializer: Format for serializing results
        accept_content: List of content types to accept
        timezone: Timezone for scheduling
        enable_utc: Whether to use UTC as the default timezone
        task_routes: Routing configuration for tasks
        task_queues: Queue configuration
        additional_options: Additional configuration options
    """
    broker_url: str = DEFAULT_BROKER_URL
    result_backend: str = DEFAULT_RESULT_BACKEND
    worker_concurrency: int = 4
    task_serializer: str = "json"
    result_serializer: str = "json"
    accept_content: List[str] = field(default_factory=lambda: ["json"])
    timezone: str = "UTC"
    enable_utc: bool = True
    task_routes: Dict[str, str] = field(default_factory=dict)
    task_queues: List[Dict[str, Any]] = field(default_factory=list)
    additional_options: Dict[str, Any] = field(default_factory=dict)
    
    def to_celery_config(self) -> Dict[str, Any]:
        """
        Convert the task configuration to a Celery configuration dictionary.
        
        Returns:
            Dictionary of Celery configuration options
        """
        config = {
            "broker_url": self.broker_url,
            "result_backend": self.result_backend,
            "worker_concurrency": self.worker_concurrency,
            "task_serializer": self.task_serializer,
            "result_serializer": self.result_serializer,
            "accept_content": self.accept_content,
            "timezone": self.timezone,
            "enable_utc": self.enable_utc,
        }
        
        if self.task_routes:
            config["task_routes"] = self.task_routes
            
        if self.task_queues:
            config["task_queues"] = self.task_queues
            
        # Add any additional options
        config.update(self.additional_options)
        
        return config


# Global task configuration instance
_task_config = None


def get_task_config() -> TaskConfig:
    """
    Get the global task configuration.
    
    Returns:
        The global task configuration
    """
    global _task_config
    
    if _task_config is None:
        # Create default configuration
        _task_config = TaskConfig()
        
        # Override with environment variables if available
        if "TASK_BROKER_URL" in os.environ:
            _task_config.broker_url = os.environ["TASK_BROKER_URL"]
        if "TASK_RESULT_BACKEND" in os.environ:
            _task_config.result_backend = os.environ["TASK_RESULT_BACKEND"]
        if "TASK_WORKER_CONCURRENCY" in os.environ:
            _task_config.worker_concurrency = int(os.environ["TASK_WORKER_CONCURRENCY"])
            
    return _task_config


def configure_tasks(config: TaskConfig) -> None:
    """
    Configure the task scheduling system.
    
    Args:
        config: Task configuration settings
    """
    global _task_config
    _task_config = config


def load_config_from_env() -> TaskConfig:
    """
    Load task configuration from environment variables.
    
    Returns:
        Task configuration populated from environment variables
    """
    config = TaskConfig()
    
    # Map environment variables to configuration options
    env_mapping = {
        "TASK_BROKER_URL": "broker_url",
        "TASK_RESULT_BACKEND": "result_backend",
        "TASK_WORKER_CONCURRENCY": "worker_concurrency",
        "TASK_SERIALIZER": "task_serializer",
        "TASK_RESULT_SERIALIZER": "result_serializer",
        "TASK_TIMEZONE": "timezone",
        "TASK_ENABLE_UTC": "enable_utc",
    }
    
    # Apply environment variables to configuration
    for env_var, config_attr in env_mapping.items():
        if env_var in os.environ:
            value = os.environ[env_var]
            
            # Convert types as needed
            if config_attr == "worker_concurrency":
                value = int(value)
            elif config_attr == "enable_utc":
                value = value.lower() in ("true", "1", "yes")
            elif config_attr == "accept_content":
                value = value.split(",")
                
            setattr(config, config_attr, value)
            
    return config 