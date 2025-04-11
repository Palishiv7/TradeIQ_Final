"""
Task Registry Module

This module provides a registry for task registration and discovery,
enabling a centralized system for managing available tasks.
"""

import importlib
import inspect
import logging
import pkgutil
from typing import Any, Callable, Dict, List, Optional, Set, Type, Union

from celery import Celery, Task
from celery.app.task import Task as CeleryTask

# Set up logging
logger = logging.getLogger(__name__)


class TaskDefinition:
    """
    Definition of a task that can be registered and executed.
    
    This class wraps a task function or class with metadata about
    the task, including its name, description, and parameters.
    """
    
    def __init__(
        self,
        name: str,
        func: Callable,
        description: Optional[str] = None,
        queue: Optional[str] = None,
        rate_limit: Optional[str] = None,
        retry: bool = True,
        max_retries: Optional[int] = None,
        retry_backoff: bool = False,
        retry_backoff_max: Optional[int] = None,
        retry_jitter: bool = False,
        soft_time_limit: Optional[int] = None,
        hard_time_limit: Optional[int] = None,
        tags: Optional[List[str]] = None,
        options: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a task definition.
        
        Args:
            name: The fully qualified name of the task
            func: The function or class implementing the task
            description: Human-readable description of the task
            queue: Queue to send the task to
            rate_limit: Maximum rate limit for the task
            retry: Whether to retry the task on failure
            max_retries: Maximum number of retries
            retry_backoff: Whether to use exponential backoff for retries
            retry_backoff_max: Maximum backoff in seconds
            retry_jitter: Whether to add random jitter to retry delays
            soft_time_limit: Soft time limit in seconds
            hard_time_limit: Hard time limit in seconds
            tags: List of tags for categorizing the task
            options: Additional options for the task
        """
        self.name = name
        self.func = func
        self.description = description or getattr(func, '__doc__', '') or ''
        self.queue = queue
        self.rate_limit = rate_limit
        self.retry = retry
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.retry_backoff_max = retry_backoff_max
        self.retry_jitter = retry_jitter
        self.soft_time_limit = soft_time_limit
        self.hard_time_limit = hard_time_limit
        self.tags = tags or []
        self.options = options or {}
        
        # Extract function signature
        self.signature = inspect.signature(func)
        
        # Parse parameters
        self.parameters = {}
        for name, param in self.signature.parameters.items():
            # Skip self parameter for methods
            if name == 'self':
                continue
                
            param_info = {
                'name': name,
                'kind': str(param.kind),
                'required': param.default is inspect.Parameter.empty,
            }
            
            if param.default is not inspect.Parameter.empty:
                param_info['default'] = param.default
                
            if param.annotation is not inspect.Parameter.empty:
                param_info['annotation'] = str(param.annotation)
                
            self.parameters[name] = param_info
    
    def register_with_celery(self, app: Celery) -> CeleryTask:
        """
        Register the task with a Celery application.
        
        Args:
            app: The Celery application to register with
            
        Returns:
            The registered Celery task
        """
        options = {
            'name': self.name,
            'queue': self.queue,
            'rate_limit': self.rate_limit,
            'retry': self.retry,
            'max_retries': self.max_retries,
            'retry_backoff': self.retry_backoff,
            'retry_backoff_max': self.retry_backoff_max,
            'retry_jitter': self.retry_jitter,
            'soft_time_limit': self.soft_time_limit,
            'time_limit': self.hard_time_limit,
            **self.options
        }
        
        # Filter out None values
        options = {k: v for k, v in options.items() if v is not None}
        
        # Register the task with Celery
        celery_task = app.task(**options)(self.func)
        
        logger.info(f"Registered task {self.name} with Celery")
        return celery_task
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the task definition to a dictionary.
        
        Returns:
            A dictionary representation of the task definition
        """
        return {
            'name': self.name,
            'description': self.description,
            'queue': self.queue,
            'parameters': self.parameters,
            'rate_limit': self.rate_limit,
            'retry': self.retry,
            'max_retries': self.max_retries,
            'retry_backoff': self.retry_backoff,
            'retry_backoff_max': self.retry_backoff_max,
            'retry_jitter': self.retry_jitter,
            'soft_time_limit': self.soft_time_limit,
            'hard_time_limit': self.hard_time_limit,
            'tags': self.tags,
            'options': self.options
        }


class TaskRegistry:
    """
    Registry for managing tasks.
    
    This class provides a central registry for tasks, allowing
    registration, discovery, and management of tasks.
    """
    
    def __init__(self):
        """Initialize an empty task registry."""
        self._tasks: Dict[str, TaskDefinition] = {}
        self._celery_app: Optional[Celery] = None
    
    def register_task(
        self,
        func: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None,
        queue: Optional[str] = None,
        rate_limit: Optional[str] = None,
        retry: bool = True,
        max_retries: Optional[int] = None,
        retry_backoff: bool = False,
        retry_backoff_max: Optional[int] = None,
        retry_jitter: bool = False,
        soft_time_limit: Optional[int] = None,
        hard_time_limit: Optional[int] = None,
        tags: Optional[List[str]] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> TaskDefinition:
        """
        Register a task with the registry.
        
        Args:
            func: The function or class implementing the task
            name: The name of the task (if None, uses func.__name__)
            description: Human-readable description of the task
            queue: Queue to send the task to
            rate_limit: Maximum rate limit for the task
            retry: Whether to retry the task on failure
            max_retries: Maximum number of retries
            retry_backoff: Whether to use exponential backoff for retries
            retry_backoff_max: Maximum backoff in seconds
            retry_jitter: Whether to add random jitter to retry delays
            soft_time_limit: Soft time limit in seconds
            hard_time_limit: Hard time limit in seconds
            tags: List of tags for categorizing the task
            options: Additional options for the task
            
        Returns:
            The registered task definition
        """
        # If name is not provided, use the function's module and name
        if name is None:
            module = func.__module__
            name = f"{module}.{func.__name__}"
        
        # Create task definition
        task_def = TaskDefinition(
            name=name,
            func=func,
            description=description,
            queue=queue,
            rate_limit=rate_limit,
            retry=retry,
            max_retries=max_retries,
            retry_backoff=retry_backoff,
            retry_backoff_max=retry_backoff_max,
            retry_jitter=retry_jitter,
            soft_time_limit=soft_time_limit,
            hard_time_limit=hard_time_limit,
            tags=tags,
            options=options
        )
        
        # Register in the registry
        self._tasks[name] = task_def
        
        # Register with Celery if available
        if self._celery_app:
            task_def.register_with_celery(self._celery_app)
        
        logger.info(f"Registered task {name} in registry")
        return task_def
    
    def task(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
        queue: Optional[str] = None,
        rate_limit: Optional[str] = None,
        retry: bool = True,
        max_retries: Optional[int] = None,
        retry_backoff: bool = False,
        retry_backoff_max: Optional[int] = None,
        retry_jitter: bool = False,
        soft_time_limit: Optional[int] = None,
        hard_time_limit: Optional[int] = None,
        tags: Optional[List[str]] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Callable[[Callable], Callable]:
        """
        Decorator for registering tasks.
        
        This decorator registers a function as a task with the registry.
        
        Args:
            name: The name of the task (if None, uses func.__name__)
            description: Human-readable description of the task
            queue: Queue to send the task to
            rate_limit: Maximum rate limit for the task
            retry: Whether to retry the task on failure
            max_retries: Maximum number of retries
            retry_backoff: Whether to use exponential backoff for retries
            retry_backoff_max: Maximum backoff in seconds
            retry_jitter: Whether to add random jitter to retry delays
            soft_time_limit: Soft time limit in seconds
            hard_time_limit: Hard time limit in seconds
            tags: List of tags for categorizing the task
            options: Additional options for the task
            
        Returns:
            A decorator function
        """
        def decorator(func: Callable) -> Callable:
            self.register_task(
                func=func,
                name=name,
                description=description,
                queue=queue,
                rate_limit=rate_limit,
                retry=retry,
                max_retries=max_retries,
                retry_backoff=retry_backoff,
                retry_backoff_max=retry_backoff_max,
                retry_jitter=retry_jitter,
                soft_time_limit=soft_time_limit,
                hard_time_limit=hard_time_limit,
                tags=tags,
                options=options
            )
            return func
        
        return decorator
    
    def get_task(self, name: str) -> Optional[TaskDefinition]:
        """
        Get a task definition by name.
        
        Args:
            name: The name of the task
            
        Returns:
            The task definition, or None if not found
        """
        return self._tasks.get(name)
    
    def get_all_tasks(self) -> Dict[str, TaskDefinition]:
        """
        Get all registered tasks.
        
        Returns:
            A dictionary of task names to task definitions
        """
        return self._tasks.copy()
    
    def filter_tasks(
        self,
        queue: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> Dict[str, TaskDefinition]:
        """
        Filter tasks by criteria.
        
        Args:
            queue: Filter tasks by queue
            tags: Filter tasks by tags (any tag in the list)
            
        Returns:
            A dictionary of filtered task names to task definitions
        """
        result = {}
        
        for name, task_def in self._tasks.items():
            # Filter by queue
            if queue is not None and task_def.queue != queue:
                continue
            
            # Filter by tags
            if tags is not None:
                if not any(tag in task_def.tags for tag in tags):
                    continue
            
            result[name] = task_def
        
        return result
    
    def set_celery_app(self, app: Celery) -> None:
        """
        Set the Celery application for the registry.
        
        This registers all existing tasks with the Celery application.
        
        Args:
            app: The Celery application
        """
        self._celery_app = app
        
        # Register all tasks with Celery
        for task_def in self._tasks.values():
            task_def.register_with_celery(app)
        
        logger.info(f"Set Celery app for registry with {len(self._tasks)} tasks")
    
    def discover_tasks(self, package_name: str) -> Set[str]:
        """
        Discover and register tasks in a package.
        
        This function recursively searches a package for functions
        decorated with the @task decorator and registers them.
        
        Args:
            package_name: The name of the package to search
            
        Returns:
            A set of task names discovered
        """
        discovered = set()
        
        try:
            package = importlib.import_module(package_name)
            package_path = getattr(package, '__path__', [])
            
            for _, name, is_pkg in pkgutil.iter_modules(package_path):
                full_name = f"{package_name}.{name}"
                
                try:
                    module = importlib.import_module(full_name)
                    
                    # If it's a package, recurse
                    if is_pkg:
                        discovered.update(self.discover_tasks(full_name))
                    
                    # Look for task functions in the module
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        
                        # Check if the attribute is a function and has task attributes
                        if callable(attr) and hasattr(attr, '__task_options__'):
                            options = getattr(attr, '__task_options__', {})
                            task_name = options.get('name', f"{full_name}.{attr_name}")
                            
                            # Register the task
                            self.register_task(attr, **options)
                            discovered.add(task_name)
                    
                except ImportError as e:
                    logger.warning(f"Failed to import module {full_name}: {e}")
        
        except ImportError as e:
            logger.warning(f"Failed to import package {package_name}: {e}")
        
        return discovered


# Global task registry instance
_registry = None


def get_registry() -> TaskRegistry:
    """
    Get the global task registry.
    
    Returns:
        The global task registry
    """
    global _registry
    
    if _registry is None:
        _registry = TaskRegistry()
        
    return _registry


def task(
    name: Optional[str] = None,
    description: Optional[str] = None,
    queue: Optional[str] = None,
    rate_limit: Optional[str] = None,
    retry: bool = True,
    max_retries: Optional[int] = None,
    retry_backoff: bool = False,
    retry_backoff_max: Optional[int] = None,
    retry_jitter: bool = False,
    soft_time_limit: Optional[int] = None,
    hard_time_limit: Optional[int] = None,
    tags: Optional[List[str]] = None,
    options: Optional[Dict[str, Any]] = None
) -> Callable[[Callable], Callable]:
    """
    Decorator for registering tasks with the global registry.
    
    This decorator registers a function as a task with the global registry.
    
    Args:
        name: The name of the task (if None, uses func.__name__)
        description: Human-readable description of the task
        queue: Queue to send the task to
        rate_limit: Maximum rate limit for the task
        retry: Whether to retry the task on failure
        max_retries: Maximum number of retries
        retry_backoff: Whether to use exponential backoff for retries
        retry_backoff_max: Maximum backoff in seconds
        retry_jitter: Whether to add random jitter to retry delays
        soft_time_limit: Soft time limit in seconds
        hard_time_limit: Hard time limit in seconds
        tags: List of tags for categorizing the task
        options: Additional options for the task
        
    Returns:
        A decorator function
    """
    registry = get_registry()
    
    def decorator(func: Callable) -> Callable:
        # Store task options on the function
        func.__task_options__ = {
            'name': name,
            'description': description,
            'queue': queue,
            'rate_limit': rate_limit,
            'retry': retry,
            'max_retries': max_retries,
            'retry_backoff': retry_backoff,
            'retry_backoff_max': retry_backoff_max,
            'retry_jitter': retry_jitter,
            'soft_time_limit': soft_time_limit,
            'hard_time_limit': hard_time_limit,
            'tags': tags,
            'options': options
        }
        
        # Register the task
        registry.register_task(
            func=func,
            name=name,
            description=description,
            queue=queue,
            rate_limit=rate_limit,
            retry=retry,
            max_retries=max_retries,
            retry_backoff=retry_backoff,
            retry_backoff_max=retry_backoff_max,
            retry_jitter=retry_jitter,
            soft_time_limit=soft_time_limit,
            hard_time_limit=hard_time_limit,
            tags=tags,
            options=options
        )
        
        return func
    
    return decorator


def get_task(name: str) -> Optional[TaskDefinition]:
    """
    Get a task definition by name from the global registry.
    
    Args:
        name: The name of the task
        
    Returns:
        The task definition, or None if not found
    """
    registry = get_registry()
    return registry.get_task(name)


def get_all_tasks() -> Dict[str, TaskDefinition]:
    """
    Get all registered tasks from the global registry.
    
    Returns:
        A dictionary of task names to task definitions
    """
    registry = get_registry()
    return registry.get_all_tasks()


def discover_tasks(package_name: str) -> Set[str]:
    """
    Discover and register tasks in a package.
    
    This function recursively searches a package for functions
    decorated with the @task decorator and registers them.
    
    Args:
        package_name: The name of the package to search
        
    Returns:
        A set of task names discovered
    """
    registry = get_registry()
    return registry.discover_tasks(package_name) 