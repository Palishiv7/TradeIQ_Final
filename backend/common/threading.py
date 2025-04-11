"""
Thread Pool Service for TradeIQ

This module provides a centralized thread pool implementation for optimal
handling of concurrent tasks across TradeIQ modules.

Key features:
1. Adaptive thread management based on system resources
2. Priority-based task scheduling
3. Performance monitoring and metrics collection
4. Task cancellation and timeout management
5. Backpressure handling for stability
"""

import concurrent.futures
import threading
import queue
import time
import logging
import os
import psutil
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Generic, Union
from enum import Enum
from dataclasses import dataclass
from functools import wraps
import asyncio
from contextlib import contextmanager

# Type definitions
T = TypeVar('T')
R = TypeVar('R')

# Configure logging
logger = logging.getLogger(__name__)

class TaskPriority(Enum):
    """Task priority levels for the thread pool scheduler"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3

@dataclass
class TaskMetrics:
    """Metrics collected for each task execution"""
    task_id: str
    start_time: float
    end_time: float = 0
    success: bool = False
    error: Optional[Exception] = None
    
    @property
    def duration(self) -> float:
        """Calculate task execution duration in seconds"""
        if self.end_time > 0:
            return self.end_time - self.start_time
        return 0

class Task(Generic[T, R]):
    """Encapsulates a task to be executed by the thread pool"""
    def __init__(
        self, 
        func: Callable[..., R], 
        args: Tuple = (), 
        kwargs: Dict[str, Any] = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        timeout: Optional[float] = None,
        task_id: Optional[str] = None
    ):
        """
        Initialize a new task.
        
        Args:
            func: The function to execute
            args: Positional arguments for the function
            kwargs: Keyword arguments for the function
            priority: Task priority level
            timeout: Maximum execution time in seconds
            task_id: Optional identifier for the task
        """
        self.func = func
        self.args = args
        self.kwargs = kwargs or {}
        self.priority = priority
        self.timeout = timeout
        self.task_id = task_id or f"task_{id(self)}"
        self.created_at = time.time()
        self.future: Optional[concurrent.futures.Future] = None
        self.metrics = TaskMetrics(task_id=self.task_id, start_time=0)
        
    def __lt__(self, other: 'Task') -> bool:
        """Compare tasks for priority queue ordering"""
        if not isinstance(other, Task):
            return NotImplemented
        return (
            (-self.priority.value, self.created_at) < 
            (-other.priority.value, other.created_at)
        )

class ThreadPoolServiceConfig:
    """Configuration for the thread pool service"""
    def __init__(
        self,
        min_workers: int = 4,
        max_workers: int = None,
        thread_name_prefix: str = "tradeiq-worker",
        queue_size: int = 1000,
        monitor_interval: int = 10,
        adaptive: bool = True,
        cpu_target: float = 0.75,
        thread_idle_timeout: float = 60.0
    ):
        """
        Initialize thread pool configuration.
        
        Args:
            min_workers: Minimum number of worker threads
            max_workers: Maximum number of worker threads (None = CPU count * 5)
            thread_name_prefix: Prefix for worker thread names
            queue_size: Maximum size of the task queue
            monitor_interval: Seconds between monitoring checks
            adaptive: Whether to adaptively adjust thread count
            cpu_target: Target CPU utilization (0.0-1.0)
            thread_idle_timeout: Seconds before idle threads are removed
        """
        cpu_count = os.cpu_count() or 4
        self.min_workers = max(1, min(min_workers, cpu_count * 2))
        self.max_workers = max_workers or (cpu_count * 5)
        self.thread_name_prefix = thread_name_prefix
        self.queue_size = max(100, queue_size)
        self.monitor_interval = max(5, monitor_interval)
        self.adaptive = adaptive
        self.cpu_target = min(0.95, max(0.1, cpu_target))
        self.thread_idle_timeout = max(10.0, thread_idle_timeout)

class ThreadPoolService:
    """
    A central thread pool service for handling concurrent tasks efficiently.
    
    Features:
    - Priority-based task scheduling
    - Adaptive thread count based on system load
    - Performance monitoring and metrics
    - Support for task timeouts and cancellation
    """
    _instance = None
    _lock = threading.RLock()
    
    def __new__(cls, *args, **kwargs):
        """Implement singleton pattern for the thread pool service"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ThreadPoolService, cls).__new__(cls)
            return cls._instance
    
    def __init__(self, config: Optional[ThreadPoolServiceConfig] = None):
        """
        Initialize the thread pool service.
        
        Args:
            config: Thread pool configuration
        """
        with self._lock:
            # Skip initialization if already initialized
            if hasattr(self, '_initialized') and self._initialized:
                return
                
            self.config = config or ThreadPoolServiceConfig()
            
            # Core components
            self._task_queue = queue.PriorityQueue(self.config.queue_size)
            self._executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=self.config.min_workers,
                thread_name_prefix=self.config.thread_name_prefix
            )
            
            # State tracking
            self._workers_count = self.config.min_workers
            self._active_tasks = {}
            self._metrics_history = []
            self._shutdown = False
            self._metrics_lock = threading.RLock()
            
            # Monitoring thread
            self._monitor_thread = threading.Thread(
                target=self._monitoring_loop,
                name=f"{self.config.thread_name_prefix}-monitor",
                daemon=True
            )
            self._monitor_thread.start()
            
            # Worker thread
            self._worker_thread = threading.Thread(
                target=self._worker_loop,
                name=f"{self.config.thread_name_prefix}-dispatcher",
                daemon=True
            )
            self._worker_thread.start()
            
            self._initialized = True
            logger.info(f"ThreadPoolService initialized with {self._workers_count} workers")
    
    def submit(
        self, 
        func: Callable[..., R], 
        *args, 
        priority: TaskPriority = TaskPriority.NORMAL,
        timeout: Optional[float] = None,
        task_id: Optional[str] = None,
        **kwargs
    ) -> concurrent.futures.Future:
        """
        Submit a task to the thread pool.
        
        Args:
            func: The function to execute
            *args: Positional arguments for the function
            priority: Task priority level
            timeout: Maximum execution time in seconds
            task_id: Optional identifier for the task
            **kwargs: Keyword arguments for the function
            
        Returns:
            Future object for the submitted task
        
        Raises:
            RuntimeError: If the thread pool service is shutting down
            queue.Full: If the task queue is full and the call would block
        """
        if self._shutdown:
            raise RuntimeError("ThreadPoolService is shutting down")
        
        task = Task(func, args, kwargs, priority, timeout, task_id)
        future = concurrent.futures.Future()
        task.future = future
        
        try:
            self._task_queue.put_nowait(task)
            logger.debug(f"Task {task.task_id} submitted with priority {priority.name}")
            return future
        except queue.Full:
            logger.warning(f"Task queue full, task {task.task_id} rejected")
            future.set_exception(RuntimeError("Task queue is full"))
            return future
    
    def shutdown(self, wait: bool = True):
        """
        Shut down the thread pool service.
        
        Args:
            wait: Whether to wait for all pending tasks to complete
        """
        with self._lock:
            if self._shutdown:
                return
                
            self._shutdown = True
            logger.info("Shutting down ThreadPoolService")
            
            # Signal worker threads to exit
            try:
                # Put a sentinel to signal worker thread to exit
                self._task_queue.put(None, block=False)
            except queue.Full:
                pass
                
            # Shutdown the executor
            self._executor.shutdown(wait=wait)
    
    def _worker_loop(self):
        """Main worker loop that dispatches tasks to the executor"""
        while not self._shutdown:
            try:
                # Get next task from the queue
                task = self._task_queue.get(block=True, timeout=1.0)
                
                # Handle sentinel value for shutdown
                if task is None:
                    break
                    
                # Submit the task to the executor
                with self._lock:
                    if self._shutdown:
                        break
                        
                    # Track the active task
                    self._active_tasks[task.task_id] = task
                    
                    # Submit to executor with wrapper
                    submit_future = self._executor.submit(
                        self._task_wrapper, task
                    )
                    
                    # Link the futures
                    submit_future.add_done_callback(
                        lambda f, task_id=task.task_id: self._task_done(task_id, f)
                    )
                    
            except queue.Empty:
                # Timeout on queue.get, continue the loop
                continue
            except Exception as e:
                logger.error(f"Error in worker loop: {e}", exc_info=True)
                time.sleep(1.0)  # Avoid tight loop on errors
    
    def _task_wrapper(self, task: Task) -> Any:
        """
        Wrapper function that executes a task with timeout handling.
        
        Args:
            task: The task to execute
            
        Returns:
            The result of the task function
        """
        # Update metrics
        task.metrics.start_time = time.time()
        
        try:
            # Execute the task with timeout if specified
            if task.timeout is not None:
                return self._execute_with_timeout(task)
            else:
                return task.func(*task.args, **task.kwargs)
        except Exception as e:
            # Record the error in metrics
            task.metrics.error = e
            logger.error(f"Error executing task {task.task_id}: {e}", exc_info=True)
            raise
        finally:
            # Update completion metrics
            task.metrics.end_time = time.time()
    
    def _execute_with_timeout(self, task: Task) -> Any:
        """
        Execute a task with timeout handling.
        
        Args:
            task: The task to execute
            
        Returns:
            The result of the task function
            
        Raises:
            TimeoutError: If the task exceeds its timeout
        """
        result_queue = queue.Queue(1)
        exception_queue = queue.Queue(1)
        
        def target():
            try:
                result = task.func(*task.args, **task.kwargs)
                result_queue.put(result)
            except Exception as e:
                exception_queue.put(e)
        
        thread = threading.Thread(target=target)
        thread.daemon = True
        thread.start()
        thread.join(task.timeout)
        
        if thread.is_alive():
            # Task timed out
            task.metrics.error = TimeoutError(f"Task {task.task_id} timed out after {task.timeout} seconds")
            raise task.metrics.error
        
        if not exception_queue.empty():
            # Task raised an exception
            exception = exception_queue.get()
            task.metrics.error = exception
            raise exception
        
        # Task completed successfully
        return result_queue.get()
    
    def _task_done(self, task_id: str, future: concurrent.futures.Future):
        """
        Handle completion of a task.
        
        Args:
            task_id: ID of the completed task
            future: Future object from the executor
        """
        with self._lock:
            # Get the original task
            task = self._active_tasks.pop(task_id, None)
            if task is None:
                return
                
            # Update task metrics
            task.metrics.success = not future.exception()
                
            # Store metrics history
            with self._metrics_lock:
                self._metrics_history.append(task.metrics)
                # Limit history size
                if len(self._metrics_history) > 1000:
                    self._metrics_history = self._metrics_history[-1000:]
                
            # Transfer result or exception to the original future
            if future.exception():
                task.future.set_exception(future.exception())
            else:
                task.future.set_result(future.result())
    
    def _monitoring_loop(self):
        """Background thread for monitoring and adjusting thread pool size"""
        while not self._shutdown:
            try:
                if self.config.adaptive:
                    self._adjust_thread_count()
                    
                # Clean up old metrics
                self._clean_metrics_history()
                    
                # Sleep until next monitoring cycle
                time.sleep(self.config.monitor_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}", exc_info=True)
                time.sleep(self.config.monitor_interval)
    
    def _adjust_thread_count(self):
        """Adaptively adjust the thread count based on system load"""
        # Get current CPU utilization
        cpu_percent = psutil.cpu_percent(interval=0.1) / 100.0
        
        # Get queue and thread metrics
        queue_size = self._task_queue.qsize()
        queue_utilization = queue_size / self.config.queue_size
        
        with self._lock:
            # Calculate target number of workers
            if cpu_percent > self.config.cpu_target and queue_utilization > 0.1:
                # System is overloaded, reduce threads if we have more than minimum
                if self._workers_count > self.config.min_workers:
                    new_count = max(
                        self.config.min_workers,
                        self._workers_count - 1
                    )
                    self._adjust_executor_size(new_count)
                    logger.info(f"Reducing thread count to {new_count} due to high CPU load ({cpu_percent:.2f})")
            elif cpu_percent < self.config.cpu_target * 0.7 and queue_utilization > 0.5:
                # System has capacity and queue is filling up, increase threads
                if self._workers_count < self.config.max_workers:
                    new_count = min(
                        self.config.max_workers,
                        self._workers_count + 1
                    )
                    self._adjust_executor_size(new_count)
                    logger.info(f"Increasing thread count to {new_count} due to queue buildup (util: {queue_utilization:.2f})")
    
    def _adjust_executor_size(self, new_size: int):
        """
        Adjust the size of the thread pool executor.
        
        Args:
            new_size: New thread pool size
        """
        # This is a simplified approach since ThreadPoolExecutor doesn't support dynamic resizing
        # For production, consider using a custom executor implementation
        with self._lock:
            if new_size == self._workers_count:
                return
                
            # Create a new executor with the desired size
            old_executor = self._executor
            self._executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=new_size,
                thread_name_prefix=self.config.thread_name_prefix
            )
            self._workers_count = new_size
            
            # Shutdown old executor without waiting
            old_executor.shutdown(wait=False)
    
    def _clean_metrics_history(self):
        """Clean up old metrics data to prevent memory growth"""
        with self._metrics_lock:
            # Keep only the last 1000 metrics or metrics from the last hour
            cutoff_time = time.time() - 3600
            self._metrics_history = [
                m for m in self._metrics_history 
                if m.start_time > cutoff_time
            ][-1000:]
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current metrics for the thread pool service.
        
        Returns:
            Dictionary of metrics
        """
        with self._metrics_lock:
            # Calculate aggregate metrics
            completed_tasks = len(self._metrics_history)
            successful_tasks = sum(1 for m in self._metrics_history if m.success)
            
            if completed_tasks > 0:
                success_rate = successful_tasks / completed_tasks
                avg_duration = sum(m.duration for m in self._metrics_history) / completed_tasks
            else:
                success_rate = 1.0
                avg_duration = 0.0
            
            # Current state
            active_tasks = len(self._active_tasks)
            queue_size = self._task_queue.qsize()
            
            return {
                "workers": self._workers_count,
                "active_tasks": active_tasks,
                "queue_size": queue_size,
                "queue_capacity": self.config.queue_size,
                "completed_tasks": completed_tasks,
                "success_rate": success_rate,
                "avg_duration": avg_duration,
                "cpu_percent": psutil.cpu_percent(interval=None)
            }

# Decorator for easily submitting functions to the thread pool
def async_task(
    priority: TaskPriority = TaskPriority.NORMAL,
    timeout: Optional[float] = None
):
    """
    Decorator to run a function asynchronously in the thread pool.
    
    Args:
        priority: Task priority level
        timeout: Maximum execution time in seconds
        
    Returns:
        Decorated function that returns a Future
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            thread_pool = ThreadPoolService()
            return thread_pool.submit(
                func, *args, 
                priority=priority, 
                timeout=timeout,
                **kwargs
            )
        return wrapper
    return decorator

# Context manager for thread pool service
@contextmanager
def thread_pool_context(config: Optional[ThreadPoolServiceConfig] = None):
    """
    Context manager for the thread pool service.
    
    Args:
        config: Thread pool configuration
        
    Yields:
        ThreadPoolService instance
    """
    service = ThreadPoolService(config)
    try:
        yield service
    finally:
        service.shutdown(wait=True)

# Async utility functions
async def run_in_thread_pool(
    func: Callable[..., R],
    *args,
    priority: TaskPriority = TaskPriority.NORMAL,
    timeout: Optional[float] = None,
    **kwargs
) -> R:
    """
    Run a function in the thread pool and await its result.
    
    Args:
        func: The function to execute
        *args: Positional arguments for the function
        priority: Task priority level
        timeout: Maximum execution time in seconds
        **kwargs: Keyword arguments for the function
        
    Returns:
        The result of the function
        
    Raises:
        Exception: Any exception raised by the function
    """
    thread_pool = ThreadPoolService()
    loop = asyncio.get_event_loop()
    future = thread_pool.submit(
        func, *args, 
        priority=priority, 
        timeout=timeout,
        **kwargs
    )
    return await asyncio.wrap_future(future) 