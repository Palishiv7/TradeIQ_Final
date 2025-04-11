"""
Task Scheduler Module

This module provides utilities for scheduling and managing background tasks
using Celery or another task queue system.
"""

import datetime
import logging
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from celery import Celery
from celery.result import AsyncResult
from celery.schedules import crontab

from backend.common.tasks.config import TaskConfig, get_task_config

# Set up logging
logger = logging.getLogger(__name__)


class TaskScheduler:
    """
    Scheduler for managing background tasks.
    
    This class provides an interface for scheduling tasks, managing periodic
    tasks, and checking task status using Celery as the backend.
    """
    
    def __init__(self, app_name: str = "tradeiq", config: Optional[TaskConfig] = None):
        """
        Initialize the task scheduler.
        
        Args:
            app_name: Name of the Celery application
            config: Task configuration (if None, uses global config)
        """
        self.app_name = app_name
        self.config = config or get_task_config()
        self._celery_app = None
        self._initialize_celery()
    
    def _initialize_celery(self) -> None:
        """Initialize the Celery application."""
        self._celery_app = Celery(
            self.app_name,
            broker=self.config.broker_url,
            backend=self.config.result_backend
        )
        
        # Apply configuration
        self._celery_app.conf.update(self.config.to_celery_config())
        
        logger.info(f"Initialized Celery app {self.app_name} with broker {self.config.broker_url}")
    
    @property
    def celery_app(self) -> Celery:
        """Get the Celery application."""
        if self._celery_app is None:
            self._initialize_celery()
        return self._celery_app
    
    def schedule_task(
        self,
        task_name: str,
        args: Optional[Tuple] = None,
        kwargs: Optional[Dict[str, Any]] = None,
        countdown: Optional[int] = None,
        eta: Optional[datetime.datetime] = None,
        queue: Optional[str] = None,
        routing_key: Optional[str] = None,
        priority: Optional[int] = None
    ) -> str:
        """
        Schedule a task for execution.
        
        Args:
            task_name: The name of the task to execute
            args: Positional arguments for the task
            kwargs: Keyword arguments for the task
            countdown: Number of seconds to wait before executing
            eta: Absolute time when the task should be executed
            queue: Queue to send the task to
            routing_key: Routing key for the task
            priority: Priority for the task
            
        Returns:
            The task ID
        """
        task = self.celery_app.send_task(
            task_name,
            args=args or (),
            kwargs=kwargs or {},
            countdown=countdown,
            eta=eta,
            queue=queue,
            routing_key=routing_key,
            priority=priority
        )
        
        logger.info(f"Scheduled task {task_name} with ID {task.id}")
        return task.id
    
    def schedule_periodic_task(
        self,
        task_name: str,
        schedule: Union[int, crontab],
        args: Optional[Tuple] = None,
        kwargs: Optional[Dict[str, Any]] = None,
        queue: Optional[str] = None,
        routing_key: Optional[str] = None,
        task_id: Optional[str] = None
    ) -> str:
        """
        Schedule a periodic task.
        
        Args:
            task_name: The name of the task to execute
            schedule: Interval in seconds or crontab
            args: Positional arguments for the task
            kwargs: Keyword arguments for the task
            queue: Queue to send the task to
            routing_key: Routing key for the task
            task_id: Custom task ID (if None, one will be generated)
            
        Returns:
            The task ID
        """
        if task_id is None:
            task_id = f"{task_name}-{uuid.uuid4()}"
        
        # Convert int schedule to interval
        if isinstance(schedule, int):
            from celery.schedules import schedule as interval_schedule
            schedule = interval_schedule(datetime.timedelta(seconds=schedule))
        
        # Add the periodic task to Celery's beat schedule
        self.celery_app.conf.beat_schedule[task_id] = {
            'task': task_name,
            'schedule': schedule,
            'args': args or (),
            'kwargs': kwargs or {},
            'options': {
                'queue': queue,
                'routing_key': routing_key
            }
        }
        
        logger.info(f"Scheduled periodic task {task_name} with ID {task_id}")
        return task_id
    
    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a scheduled task.
        
        Args:
            task_id: The ID of the task to cancel
            
        Returns:
            True if the task was cancelled, False otherwise
        """
        try:
            self.celery_app.control.revoke(task_id, terminate=True)
            logger.info(f"Cancelled task with ID {task_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel task {task_id}: {str(e)}")
            return False
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """
        Get the status of a task.
        
        Args:
            task_id: The ID of the task
            
        Returns:
            A dictionary with task status information
        """
        result = AsyncResult(task_id, app=self.celery_app)
        
        status = {
            'id': task_id,
            'status': result.status,
            'completed': result.ready(),
            'successful': result.successful() if result.ready() else None,
            'result': result.result if result.ready() and not isinstance(result.result, Exception) else None,
            'error': str(result.result) if result.ready() and isinstance(result.result, Exception) else None,
            'runtime': result.runtime if result.ready() else None,
        }
        
        return status
    
    def get_scheduled_tasks(self) -> List[Dict[str, Any]]:
        """
        Get all scheduled periodic tasks.
        
        Returns:
            A list of dictionaries with task information
        """
        beat_schedule = self.celery_app.conf.beat_schedule or {}
        
        tasks = []
        for task_id, task_info in beat_schedule.items():
            schedule_str = str(task_info['schedule'])
            if hasattr(task_info['schedule'], 'run_every'):
                schedule_str = f"every {task_info['schedule'].run_every}"
            
            tasks.append({
                'id': task_id,
                'task': task_info['task'],
                'schedule': schedule_str,
                'args': task_info.get('args', ()),
                'kwargs': task_info.get('kwargs', {}),
                'options': task_info.get('options', {})
            })
            
        return tasks
    
    def pause_periodic_task(self, task_id: str) -> bool:
        """
        Pause a periodic task.
        
        Args:
            task_id: The ID of the task
            
        Returns:
            True if the task was paused, False otherwise
        """
        if task_id in self.celery_app.conf.beat_schedule:
            # Store the task configuration
            self._paused_tasks = getattr(self, '_paused_tasks', {})
            self._paused_tasks[task_id] = self.celery_app.conf.beat_schedule[task_id]
            
            # Remove from beat schedule
            del self.celery_app.conf.beat_schedule[task_id]
            
            logger.info(f"Paused periodic task {task_id}")
            return True
        
        logger.warning(f"Task {task_id} not found in beat schedule")
        return False
    
    def resume_periodic_task(self, task_id: str) -> bool:
        """
        Resume a paused periodic task.
        
        Args:
            task_id: The ID of the task
            
        Returns:
            True if the task was resumed, False otherwise
        """
        paused_tasks = getattr(self, '_paused_tasks', {})
        
        if task_id in paused_tasks:
            # Restore the task configuration
            self.celery_app.conf.beat_schedule[task_id] = paused_tasks[task_id]
            del paused_tasks[task_id]
            
            logger.info(f"Resumed periodic task {task_id}")
            return True
        
        logger.warning(f"Task {task_id} not found in paused tasks")
        return False


# Global task scheduler instance
_scheduler = None


def get_scheduler() -> TaskScheduler:
    """
    Get the global task scheduler.
    
    Returns:
        The global task scheduler
    """
    global _scheduler
    
    if _scheduler is None:
        _scheduler = TaskScheduler()
        
    return _scheduler


def schedule_task(
    task_name: str,
    args: Optional[Tuple] = None,
    kwargs: Optional[Dict[str, Any]] = None,
    countdown: Optional[int] = None,
    eta: Optional[datetime.datetime] = None,
    queue: Optional[str] = None
) -> str:
    """
    Schedule a task for execution.
    
    Args:
        task_name: The name of the task to execute
        args: Positional arguments for the task
        kwargs: Keyword arguments for the task
        countdown: Number of seconds to wait before executing
        eta: Absolute time when the task should be executed
        queue: Queue to send the task to
        
    Returns:
        The task ID
    """
    scheduler = get_scheduler()
    return scheduler.schedule_task(
        task_name=task_name,
        args=args,
        kwargs=kwargs,
        countdown=countdown,
        eta=eta,
        queue=queue
    )


def schedule_periodic_task(
    task_name: str,
    schedule: Union[int, crontab],
    args: Optional[Tuple] = None,
    kwargs: Optional[Dict[str, Any]] = None,
    queue: Optional[str] = None,
    task_id: Optional[str] = None
) -> str:
    """
    Schedule a periodic task.
    
    Args:
        task_name: The name of the task to execute
        schedule: Interval in seconds or crontab
        args: Positional arguments for the task
        kwargs: Keyword arguments for the task
        queue: Queue to send the task to
        task_id: Custom task ID (if None, one will be generated)
        
    Returns:
        The task ID
    """
    scheduler = get_scheduler()
    return scheduler.schedule_periodic_task(
        task_name=task_name,
        schedule=schedule,
        args=args,
        kwargs=kwargs,
        queue=queue,
        task_id=task_id
    )


def cancel_task(task_id: str) -> bool:
    """
    Cancel a scheduled task.
    
    Args:
        task_id: The ID of the task to cancel
        
    Returns:
        True if the task was cancelled, False otherwise
    """
    scheduler = get_scheduler()
    return scheduler.cancel_task(task_id)


def get_task_status(task_id: str) -> Dict[str, Any]:
    """
    Get the status of a task.
    
    Args:
        task_id: The ID of the task
        
    Returns:
        A dictionary with task status information
    """
    scheduler = get_scheduler()
    return scheduler.get_task_status(task_id) 