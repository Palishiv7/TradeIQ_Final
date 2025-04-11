"""
Predefined Tasks Module

This module contains predefined tasks that can be used across
the application for common operations like data processing,
notifications, and system maintenance.
"""

import datetime
import logging
import os
import time
from typing import Any, Dict, List, Optional, Union

from backend.common.tasks.registry import task

# Set up logging
logger = logging.getLogger(__name__)


# ---- System Maintenance Tasks ----

@task(
    queue="maintenance",
    tags=["system", "maintenance"],
    description="Clean up old temporary files",
    retry=True,
    max_retries=3,
    retry_backoff=True
)
def cleanup_temp_files(
    directory: str,
    max_age_days: int = 7,
    file_pattern: Optional[str] = None,
    dry_run: bool = False
) -> Dict[str, Any]:
    """
    Clean up temporary files older than the specified age.
    
    Args:
        directory: Directory to clean up
        max_age_days: Maximum age of files in days
        file_pattern: Optional file pattern to match (e.g., "*.tmp")
        dry_run: If True, only report what would be deleted without actually deleting
        
    Returns:
        Dictionary with deletion stats (deleted_count, total_size, etc.)
    """
    logger.info(f"Starting cleanup of temporary files in {directory}")
    
    # Calculate cutoff time
    cutoff_time = time.time() - (max_age_days * 86400)
    
    stats = {
        "deleted_count": 0,
        "skipped_count": 0,
        "error_count": 0,
        "total_size": 0,
        "dry_run": dry_run,
        "deleted_files": []
    }
    
    # Walk through the directory
    for root, _, files in os.walk(directory):
        for filename in files:
            # Check if file matches pattern
            if file_pattern and not _matches_pattern(filename, file_pattern):
                continue
                
            file_path = os.path.join(root, filename)
            
            try:
                # Get file stats
                file_stat = os.stat(file_path)
                file_mtime = file_stat.st_mtime
                file_size = file_stat.st_size
                
                # Check if file is older than cutoff
                if file_mtime < cutoff_time:
                    if not dry_run:
                        os.remove(file_path)
                        
                    stats["deleted_count"] += 1
                    stats["total_size"] += file_size
                    stats["deleted_files"].append(file_path)
                    logger.debug(f"Deleted file: {file_path} (size: {file_size} bytes)")
                else:
                    stats["skipped_count"] += 1
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {str(e)}")
                stats["error_count"] += 1
    
    log_msg = (
        f"Cleanup complete: {stats['deleted_count']} files deleted, "
        f"{stats['skipped_count']} files skipped, "
        f"{stats['error_count']} errors"
    )
    if dry_run:
        log_msg = f"DRY RUN: {log_msg}"
    
    logger.info(log_msg)
    return stats


@task(
    queue="maintenance",
    tags=["system", "maintenance"],
    description="Monitor system health",
    soft_time_limit=60
)
def monitor_system_health() -> Dict[str, Any]:
    """
    Monitor system health and return metrics.
    
    Returns:
        Dictionary with system health metrics
    """
    import psutil
    
    metrics = {
        "timestamp": datetime.datetime.now().isoformat(),
        "cpu": {
            "percent": psutil.cpu_percent(interval=1),
            "count": psutil.cpu_count(),
            "load_avg": os.getloadavg() if hasattr(os, "getloadavg") else None
        },
        "memory": {
            "total": psutil.virtual_memory().total,
            "available": psutil.virtual_memory().available,
            "percent": psutil.virtual_memory().percent
        },
        "disk": {},
        "network": {
            "connections": len(psutil.net_connections())
        }
    }
    
    # Collect disk usage
    for mount in psutil.disk_partitions():
        try:
            usage = psutil.disk_usage(mount.mountpoint)
            metrics["disk"][mount.mountpoint] = {
                "total": usage.total,
                "used": usage.used,
                "free": usage.free,
                "percent": usage.percent
            }
        except Exception:
            # Skip this mount point
            pass
    
    logger.info(f"System health check: CPU: {metrics['cpu']['percent']}%, "
               f"Memory: {metrics['memory']['percent']}%")
    
    return metrics


# ---- Data Processing Tasks ----

@task(
    queue="data_processing",
    tags=["data", "processing"],
    description="Process market data updates",
    retry=True,
    max_retries=5,
    retry_backoff=True
)
def process_market_data_updates(
    symbols: List[str],
    start_date: Union[str, datetime.date],
    end_date: Optional[Union[str, datetime.date]] = None,
    data_source: str = "default",
    interval: str = "1d"
) -> Dict[str, Any]:
    """
    Process market data updates for the specified symbols.
    
    Args:
        symbols: List of market symbols to process
        start_date: Start date for data processing
        end_date: End date for data processing (defaults to today)
        data_source: Data source identifier
        interval: Data interval (e.g., "1d", "1h", "5m")
        
    Returns:
        Dictionary with processing stats
    """
    logger.info(f"Processing market data for {len(symbols)} symbols from {start_date} to {end_date}")
    
    # Convert dates if needed
    if isinstance(start_date, str):
        start_date = datetime.datetime.fromisoformat(start_date).date()
    
    if end_date is None:
        end_date = datetime.date.today()
    elif isinstance(end_date, str):
        end_date = datetime.datetime.fromisoformat(end_date).date()
    
    stats = {
        "symbols_processed": 0,
        "data_points_processed": 0,
        "errors": [],
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "interval": interval,
        "data_source": data_source
    }
    
    # Process each symbol
    for symbol in symbols:
        try:
            # This would call your market data processing logic
            # For demonstration, we'll just simulate processing
            processed_points = _simulate_data_processing(
                symbol, start_date, end_date, interval
            )
            stats["data_points_processed"] += processed_points
            stats["symbols_processed"] += 1
        except Exception as e:
            logger.error(f"Error processing data for symbol {symbol}: {str(e)}")
            stats["errors"].append({"symbol": symbol, "error": str(e)})
    
    logger.info(f"Market data processing complete: {stats['symbols_processed']} symbols, "
               f"{stats['data_points_processed']} data points")
    
    return stats


@task(
    queue="data_processing",
    tags=["data", "generation"],
    description="Generate assessment questions",
    retry=True,
    max_retries=3
)
def generate_assessment_questions(
    assessment_type: str,
    difficulty: str,
    topics: List[str],
    count: int = 10
) -> Dict[str, Any]:
    """
    Generate assessment questions for specified topics and difficulty.
    
    Args:
        assessment_type: Type of assessment (e.g., "pattern", "market_basics")
        difficulty: Difficulty level ("beginner", "intermediate", "advanced")
        topics: List of topics to cover
        count: Number of questions to generate
        
    Returns:
        Dictionary with generated questions and metadata
    """
    logger.info(f"Generating {count} {difficulty} questions for {assessment_type} on topics: {topics}")
    
    # In a real implementation, this would call your question generation service
    # For demonstration, we'll just return placeholder data
    
    result = {
        "assessment_type": assessment_type,
        "difficulty": difficulty,
        "topics": topics,
        "count": count,
        "questions": [],
        "metadata": {
            "generation_time": datetime.datetime.now().isoformat(),
            "version": "1.0"
        }
    }
    
    # Simulate question generation
    for i in range(count):
        # For demonstration only - this would be your actual generation logic
        question = {
            "id": f"q_{assessment_type}_{i+1}",
            "text": f"Sample question #{i+1} for {assessment_type} on {topics[i % len(topics)]}",
            "difficulty": difficulty,
            "topic": topics[i % len(topics)],
            "options": [
                {"id": "a", "text": "Option A"},
                {"id": "b", "text": "Option B"},
                {"id": "c", "text": "Option C"},
                {"id": "d", "text": "Option D"}
            ],
            "correct_answer": "a"
        }
        result["questions"].append(question)
    
    logger.info(f"Generated {len(result['questions'])} questions for {assessment_type}")
    return result


# ---- Notification Tasks ----

@task(
    queue="notifications",
    tags=["notifications", "email"],
    description="Send email notification",
    retry=True,
    max_retries=3,
    retry_backoff=True
)
def send_email_notification(
    recipient: str,
    subject: str,
    message: str,
    template_id: Optional[str] = None,
    template_data: Optional[Dict[str, Any]] = None,
    attachments: Optional[List[Dict[str, str]]] = None,
    priority: str = "normal"
) -> Dict[str, Any]:
    """
    Send an email notification.
    
    Args:
        recipient: Email recipient
        subject: Email subject
        message: Email body (plain text)
        template_id: Optional ID of email template to use
        template_data: Optional data for email template
        attachments: Optional list of attachments
        priority: Email priority ("high", "normal", "low")
        
    Returns:
        Dictionary with sending status
    """
    logger.info(f"Sending email to {recipient}: {subject}")
    
    # In a real implementation, this would call your email service
    # For demonstration, we'll just simulate sending
    
    # Simulate email sending
    time.sleep(0.5)  # Simulate sending delay
    
    result = {
        "recipient": recipient,
        "subject": subject,
        "sent_at": datetime.datetime.now().isoformat(),
        "status": "sent",
        "message_id": f"msg_{int(time.time())}_{hash(recipient) % 10000}",
        "priority": priority
    }
    
    logger.info(f"Email sent to {recipient}: {result['message_id']}")
    return result


# ---- Helper Functions ----

def _matches_pattern(filename: str, pattern: str) -> bool:
    """Check if a filename matches a simple pattern (glob-style)."""
    import fnmatch
    return fnmatch.fnmatch(filename, pattern)


def _simulate_data_processing(
    symbol: str,
    start_date: datetime.date,
    end_date: datetime.date,
    interval: str
) -> int:
    """Simulate processing data for a symbol and return the number of data points."""
    # Calculate the number of intervals between start and end date
    delta = end_date - start_date
    
    if interval == "1d":
        return delta.days
    elif interval == "1h":
        return delta.days * 24
    elif interval == "5m":
        return delta.days * 24 * 12
    else:
        return delta.days 