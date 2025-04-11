"""
Worker Module

This module provides utilities for managing Celery workers
and task execution environment.
"""

import logging
import multiprocessing
import os
import socket
from typing import Any, Dict, List, Optional, Set, Union

from celery import Celery
from celery.signals import (
    beat_init, task_failure, task_postrun, task_prerun, 
    task_received, task_retry, task_success, worker_init
)

from backend.common.tasks.config import TaskConfig, get_task_config
from backend.common.tasks.registry import get_registry

# Set up logging
logger = logging.getLogger(__name__)


class WorkerConfig:
    """Configuration for Celery workers."""
    
    def __init__(
        self,
        concurrency: Optional[int] = None,
        max_tasks_per_child: Optional[int] = None,
        task_time_limit: Optional[int] = None,
        task_soft_time_limit: Optional[int] = None,
        prefetch_multiplier: Optional[int] = None,
        enable_remote_control: bool = True,
        queues: Optional[List[str]] = None,
        loglevel: str = 'INFO',
        include: Optional[List[str]] = None,
        hostname: Optional[str] = None,
        autoscale: Optional[tuple] = None
    ):
        """
        Initialize worker configuration.
        
        Args:
            concurrency: Number of child processes/threads
            max_tasks_per_child: Maximum number of tasks per child process
            task_time_limit: Hard time limit for tasks in seconds
            task_soft_time_limit: Soft time limit for tasks in seconds
            prefetch_multiplier: Worker prefetch multiplier
            enable_remote_control: Enable/disable remote control commands
            queues: List of queue names to consume from
            loglevel: Logging level
            include: List of modules to import when worker starts
            hostname: Custom hostname for the worker
            autoscale: Min/max autoscale values (min, max)
        """
        # Set default concurrency based on CPU count
        if concurrency is None:
            concurrency = multiprocessing.cpu_count()
        
        # Set default hostname based on machine hostname
        if hostname is None:
            hostname = f"{socket.gethostname()}.{os.getpid()}"
        
        self.concurrency = concurrency
        self.max_tasks_per_child = max_tasks_per_child
        self.task_time_limit = task_time_limit
        self.task_soft_time_limit = task_soft_time_limit
        self.prefetch_multiplier = prefetch_multiplier
        self.enable_remote_control = enable_remote_control
        self.queues = queues
        self.loglevel = loglevel
        self.include = include or []
        self.hostname = hostname
        self.autoscale = autoscale
    
    def to_worker_options(self) -> Dict[str, Any]:
        """
        Convert worker configuration to Celery worker options.
        
        Returns:
            A dictionary of worker options
        """
        options = {
            'concurrency': self.concurrency,
            'max_tasks_per_child': self.max_tasks_per_child,
            'task_time_limit': self.task_time_limit,
            'task_soft_time_limit': self.task_soft_time_limit,
            'prefetch_multiplier': self.prefetch_multiplier,
            'without_gossip': not self.enable_remote_control,
            'without_mingle': not self.enable_remote_control,
            'without_heartbeat': not self.enable_remote_control,
            'hostname': self.hostname,
            'loglevel': self.loglevel,
        }
        
        # Add queues if specified
        if self.queues:
            options['queues'] = self.queues
        
        # Add autoscale if specified
        if self.autoscale:
            options['autoscale'] = self.autoscale
        
        # Filter out None values
        return {k: v for k, v in options.items() if v is not None}


class WorkerManager:
    """
    Manager for Celery workers.
    
    This class provides utilities for configuring and starting
    Celery workers, and managing task execution.
    """
    
    def __init__(
        self,
        app_name: str = "tradeiq",
        config: Optional[TaskConfig] = None,
        worker_config: Optional[WorkerConfig] = None
    ):
        """
        Initialize the worker manager.
        
        Args:
            app_name: Name of the Celery application
            config: Task configuration (if None, uses global config)
            worker_config: Worker configuration (if None, uses defaults)
        """
        self.app_name = app_name
        self.config = config or get_task_config()
        self.worker_config = worker_config or WorkerConfig()
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
        
        # Register tasks from the registry
        registry = get_registry()
        registry.set_celery_app(self._celery_app)
        
        # Register signal handlers
        self._register_signals()
        
        logger.info(f"Initialized Celery app {self.app_name} with broker {self.config.broker_url}")
    
    def _register_signals(self) -> None:
        """Register Celery signal handlers."""
        
        @worker_init.connect
        def on_worker_init(sender, **kwargs):
            logger.info(f"Worker initialized: {sender}")
        
        @beat_init.connect
        def on_beat_init(sender, **kwargs):
            logger.info(f"Beat scheduler initialized: {sender}")
        
        @task_received.connect
        def on_task_received(request, **kwargs):
            logger.debug(f"Task received: {request.task}")
        
        @task_prerun.connect
        def on_task_prerun(task_id, task, **kwargs):
            logger.debug(f"Task starting: {task.name} [{task_id}]")
        
        @task_postrun.connect
        def on_task_postrun(task_id, task, retval, state, **kwargs):
            logger.debug(f"Task completed: {task.name} [{task_id}] -> {state}")
        
        @task_success.connect
        def on_task_success(sender, result, **kwargs):
            logger.info(f"Task succeeded: {sender.name}")
        
        @task_failure.connect
        def on_task_failure(sender, task_id, exception, traceback, **kwargs):
            logger.error(f"Task failed: {sender.name} [{task_id}] -> {exception}")
        
        @task_retry.connect
        def on_task_retry(sender, request, reason, einfo, **kwargs):
            logger.warning(f"Task retrying: {sender.name} -> {reason}")
    
    @property
    def celery_app(self) -> Celery:
        """Get the Celery application."""
        if self._celery_app is None:
            self._initialize_celery()
        return self._celery_app
    
    def start_worker(
        self,
        worker_config: Optional[WorkerConfig] = None,
        beat: bool = False,
        detach: bool = False,
        discover_modules: Optional[List[str]] = None
    ) -> None:
        """
        Start a Celery worker.
        
        Args:
            worker_config: Worker configuration (if None, uses default)
            beat: Whether to run the beat scheduler with the worker
            detach: Run the worker in the background
            discover_modules: List of modules to discover tasks from
        """
        config = worker_config or self.worker_config
        worker_options = config.to_worker_options()
        
        # Discover tasks in modules
        if discover_modules:
            registry = get_registry()
            for module in discover_modules:
                registry.discover_tasks(module)
        
        # Add modules to include list
        if discover_modules:
            include_modules = list(config.include)
            include_modules.extend(discover_modules)
            worker_options['include'] = include_modules
        
        # Add beat option if specified
        if beat:
            worker_options['beat'] = True
        
        # Add detach option if specified
        if detach:
            worker_options['detach'] = True
        
        logger.info(f"Starting Celery worker with options: {worker_options}")
        
        # Start the worker - this is a blocking call
        worker = self.celery_app.Worker(**worker_options)
        worker.start()
    
    def start_beat(self, detach: bool = False) -> None:
        """
        Start the Celery beat scheduler.
        
        Args:
            detach: Run the scheduler in the background
        """
        from celery.apps.beat import Beat
        
        beat_options = {
            'loglevel': self.worker_config.loglevel,
        }
        
        if detach:
            beat_options['detach'] = True
        
        logger.info("Starting Celery beat scheduler")
        
        # Start the beat scheduler - this is a blocking call
        beat = Beat(app=self.celery_app, **beat_options)
        beat.run()
    
    def run_task(
        self,
        task_name: str,
        args: Optional[List[Any]] = None,
        kwargs: Optional[Dict[str, Any]] = None,
        synchronous: bool = False,
        timeout: Optional[int] = None,
        retry: bool = True,
        max_retries: Optional[int] = None,
        retry_backoff: bool = False,
        queue: Optional[str] = None
    ) -> Any:
        """
        Run a task.
        
        Args:
            task_name: The name of the task to run
            args: Positional arguments for the task
            kwargs: Keyword arguments for the task
            synchronous: Whether to run the task synchronously
            timeout: Timeout for the task result in seconds
            retry: Whether to retry the task on failure
            max_retries: Maximum number of retries
            retry_backoff: Whether to use exponential backoff for retries
            queue: Queue to send the task to
            
        Returns:
            The task result (if synchronous) or AsyncResult
        """
        task = self.celery_app.tasks.get(task_name)
        
        if task is None:
            raise ValueError(f"Task not found: {task_name}")
        
        # Prepare arguments
        task_args = args or []
        task_kwargs = kwargs or {}
        
        # Prepare task options
        options = {}
        
        if retry is not None:
            options['retry'] = retry
            
        if max_retries is not None:
            options['max_retries'] = max_retries
            
        if retry_backoff is not None:
            options['retry_backoff'] = retry_backoff
            
        if queue is not None:
            options['queue'] = queue
        
        # Run the task
        if synchronous:
            return task.apply(args=task_args, kwargs=task_kwargs, **options).get(
                timeout=timeout
            )
        else:
            return task.apply_async(args=task_args, kwargs=task_kwargs, **options)


# Global worker manager instance
_worker_manager = None


def get_worker_manager() -> WorkerManager:
    """
    Get the global worker manager.
    
    Returns:
        The global worker manager
    """
    global _worker_manager
    
    if _worker_manager is None:
        _worker_manager = WorkerManager()
        
    return _worker_manager


def start_worker(
    worker_config: Optional[WorkerConfig] = None,
    beat: bool = False,
    detach: bool = False,
    discover_modules: Optional[List[str]] = None
) -> None:
    """
    Start a Celery worker.
    
    Args:
        worker_config: Worker configuration (if None, uses default)
        beat: Whether to run the beat scheduler with the worker
        detach: Run the worker in the background
        discover_modules: List of modules to discover tasks from
    """
    worker_manager = get_worker_manager()
    worker_manager.start_worker(
        worker_config=worker_config,
        beat=beat,
        detach=detach,
        discover_modules=discover_modules
    )


def start_beat(detach: bool = False) -> None:
    """
    Start the Celery beat scheduler.
    
    Args:
        detach: Run the scheduler in the background
    """
    worker_manager = get_worker_manager()
    worker_manager.start_beat(detach=detach)


def run_task(
    task_name: str,
    args: Optional[List[Any]] = None,
    kwargs: Optional[Dict[str, Any]] = None,
    synchronous: bool = False,
    timeout: Optional[int] = None,
    retry: bool = True,
    max_retries: Optional[int] = None,
    retry_backoff: bool = False,
    queue: Optional[str] = None
) -> Any:
    """
    Run a task.
    
    Args:
        task_name: The name of the task to run
        args: Positional arguments for the task
        kwargs: Keyword arguments for the task
        synchronous: Whether to run the task synchronously
        timeout: Timeout for the task result in seconds
        retry: Whether to retry the task on failure
        max_retries: Maximum number of retries
        retry_backoff: Whether to use exponential backoff for retries
        queue: Queue to send the task to
        
    Returns:
        The task result (if synchronous) or AsyncResult
    """
    worker_manager = get_worker_manager()
    return worker_manager.run_task(
        task_name=task_name,
        args=args,
        kwargs=kwargs,
        synchronous=synchronous,
        timeout=timeout,
        retry=retry,
        max_retries=max_retries,
        retry_backoff=retry_backoff,
        queue=queue
    ) 