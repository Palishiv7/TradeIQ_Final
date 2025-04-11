"""
Task Scheduling Module

This module provides a task scheduling system for managing background jobs,
periodic tasks, and asynchronous processing using Celery as the backend.
It includes components for task configuration, scheduling, registration,
and worker management.
"""

# Configuration
from backend.common.tasks.config import (
    TaskConfig,
    get_task_config,
    configure_tasks,
    load_config_from_env
)

# Scheduler
from backend.common.tasks.scheduler import (
    TaskScheduler,
    get_scheduler,
    schedule_task,
    schedule_periodic_task,
    cancel_task,
    get_task_status
)

# Registry
from backend.common.tasks.registry import (
    TaskDefinition,
    TaskRegistry,
    get_registry,
    task,
    get_task,
    get_all_tasks,
    discover_tasks
)

# Worker
from backend.common.tasks.worker import (
    WorkerConfig,
    WorkerManager,
    get_worker_manager,
    start_worker,
    start_beat,
    run_task
)

# Predefined tasks
import backend.common.tasks.predefined

# Public API
__all__ = [
    # Configuration
    'TaskConfig',
    'get_task_config',
    'configure_tasks',
    'load_config_from_env',
    
    # Scheduler
    'TaskScheduler',
    'get_scheduler',
    'schedule_task',
    'schedule_periodic_task',
    'cancel_task',
    'get_task_status',
    
    # Registry
    'TaskDefinition',
    'TaskRegistry',
    'get_registry',
    'task',
    'get_task',
    'get_all_tasks',
    'discover_tasks',
    
    # Worker
    'WorkerConfig',
    'WorkerManager',
    'get_worker_manager',
    'start_worker',
    'start_beat',
    'run_task',
] 