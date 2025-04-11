"""
Common utility functions for the TradeIQ backend.

This module provides utility functions used across different parts of the application,
particularly for data serialization, type conversion, and common operations.
"""

import datetime
from typing import Any, Dict, List, Optional, Union

def serialize_datetime(obj: Any) -> str:
    """
    Serialize datetime objects to ISO format strings.
    
    This function is used as the default serializer for json.dumps() when
    dealing with datetime objects.
    
    Args:
        obj: Object to serialize
        
    Returns:
        ISO format string if obj is a datetime, otherwise raises TypeError
    """
    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")

def parse_json_with_dates(obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse JSON dictionary and convert ISO date strings back to datetime objects.
    
    Args:
        obj: Dictionary to parse
        
    Returns:
        Dictionary with datetime strings converted to datetime objects
    """
    for key, value in obj.items():
        if isinstance(value, str):
            try:
                # Try to parse as datetime
                obj[key] = datetime.datetime.fromisoformat(value)
            except ValueError:
                # If not a valid datetime string, leave as is
                pass
        elif isinstance(value, dict):
            obj[key] = parse_json_with_dates(value)
    return obj

def format_duration(seconds: float) -> str:
    """
    Format a duration in seconds to a human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string (e.g., "2h 30m", "45s")
    """
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        minutes = int(seconds / 60)
        seconds = int(seconds % 60)
        return f"{minutes}m {seconds}s"
    else:
        hours = int(seconds / 3600)
        minutes = int((seconds % 3600) / 60)
        return f"{hours}h {minutes}m"

def safe_divide(numerator: Union[int, float], denominator: Union[int, float], default: Union[int, float] = 0) -> float:
    """
    Safely divide two numbers, returning a default value if denominator is zero.
    
    Args:
        numerator: Number to divide
        denominator: Number to divide by
        default: Value to return if denominator is zero
        
    Returns:
        Result of division or default value
    """
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except (TypeError, ValueError):
        return default

def truncate_string(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate a string to a maximum length, adding a suffix if truncated.
    
    Args:
        text: String to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated string
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix

def parse_bool(value: Union[str, bool, int, None]) -> bool:
    """
    Parse a value as a boolean.
    
    Args:
        value: Value to parse
        
    Returns:
        Boolean value
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    if isinstance(value, str):
        return value.lower() in ("yes", "true", "t", "1", "on")
    return False

def get_nested_value(obj: Dict[str, Any], path: str, default: Any = None) -> Any:
    """
    Get a nested value from a dictionary using a dot-separated path.
    
    Args:
        obj: Dictionary to search
        path: Dot-separated path (e.g., "user.profile.name")
        default: Default value if path not found
        
    Returns:
        Value at path or default
    """
    try:
        parts = path.split(".")
        current = obj
        for part in parts:
            current = current[part]
        return current
    except (KeyError, TypeError, AttributeError):
        return default

def set_nested_value(obj: Dict[str, Any], path: str, value: Any) -> None:
    """
    Set a nested value in a dictionary using a dot-separated path.
    
    Args:
        obj: Dictionary to modify
        path: Dot-separated path (e.g., "user.profile.name")
        value: Value to set
    """
    parts = path.split(".")
    current = obj
    for part in parts[:-1]:
        if part not in current:
            current[part] = {}
        current = current[part]
    current[parts[-1]] = value

def merge_dicts(dict1: Dict[str, Any], dict2: Dict[str, Any], overwrite: bool = True) -> Dict[str, Any]:
    """
    Merge two dictionaries recursively.
    
    Args:
        dict1: First dictionary
        dict2: Second dictionary
        overwrite: Whether to overwrite values in dict1 with values from dict2
        
    Returns:
        Merged dictionary
    """
    result = dict1.copy()
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value, overwrite)
        elif key not in result or overwrite:
            result[key] = value
    return result 