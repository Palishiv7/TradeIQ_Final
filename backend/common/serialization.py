"""
Serialization Utilities

This module provides utilities for serializing and deserializing objects across the application,
with support for various formats and handling of complex types like datetime.
"""

import json
import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Type, TypeVar, get_type_hints
from dataclasses import is_dataclass, asdict

# Type variable for generic typing
T = TypeVar('T')

class SerializationFormat(Enum):
    """Supported serialization formats."""
    JSON = "json"
    DICT = "dict"


def serialize(
    obj: Any,
    format: SerializationFormat = SerializationFormat.DICT,
    exclude_none: bool = False,
    exclude_fields: Optional[List[str]] = None
) -> Union[Dict[str, Any], str]:
    """
    Serialize an object to the specified format.
    
    Args:
        obj: The object to serialize
        format: Output format (JSON string or Python dict)
        exclude_none: Whether to exclude None values
        exclude_fields: Optional list of field names to exclude
        
    Returns:
        Serialized object as a dict or JSON string
    """
    exclude_fields = exclude_fields or []
    
    # Handle None
    if obj is None:
        return {} if format == SerializationFormat.DICT else "{}"
    
    # Handle primitive types
    if isinstance(obj, (str, int, float, bool)):
        return obj if format == SerializationFormat.DICT else json.dumps(obj)
    
    # Handle datetime
    if isinstance(obj, datetime.datetime):
        dt_str = obj.isoformat()
        return dt_str if format == SerializationFormat.DICT else f'"{dt_str}"'
    
    if isinstance(obj, datetime.date):
        date_str = obj.isoformat()
        return date_str if format == SerializationFormat.DICT else f'"{date_str}"'
    
    # Handle Enum
    if isinstance(obj, Enum):
        return obj.value if format == SerializationFormat.DICT else json.dumps(obj.value)
    
    # Handle lists
    if isinstance(obj, list):
        serialized_list = [
            serialize(item, SerializationFormat.DICT, exclude_none, exclude_fields)
            for item in obj
        ]
        return serialized_list if format == SerializationFormat.DICT else json.dumps(serialized_list)
    
    # Handle dictionaries
    if isinstance(obj, dict):
        result = {}
        for key, value in obj.items():
            if key in exclude_fields:
                continue
            if exclude_none and value is None:
                continue
            result[key] = serialize(value, SerializationFormat.DICT, exclude_none, exclude_fields)
        return result if format == SerializationFormat.DICT else json.dumps(result)
    
    # Handle dataclasses
    if is_dataclass(obj):
        return serialize(
            asdict(obj), 
            format, 
            exclude_none, 
            exclude_fields
        )
    
    # Handle objects with to_dict or __dict__
    if hasattr(obj, 'to_dict') and callable(getattr(obj, 'to_dict')):
        obj_dict = obj.to_dict()
        return serialize(obj_dict, format, exclude_none, exclude_fields)
    
    if hasattr(obj, '__dict__'):
        obj_dict = {
            k: v for k, v in obj.__dict__.items() 
            if not k.startswith('_') and k not in exclude_fields
        }
        if exclude_none:
            obj_dict = {k: v for k, v in obj_dict.items() if v is not None}
        return serialize(obj_dict, format, exclude_none, exclude_fields)
    
    # Fallback: Try direct JSON serialization
    try:
        if format == SerializationFormat.DICT:
            return obj
        else:
            return json.dumps(obj)
    except (TypeError, ValueError):
        return str(obj) if format == SerializationFormat.DICT else json.dumps(str(obj))


def deserialize(
    data: Union[Dict[str, Any], str],
    target_class: Type[T],
    format: SerializationFormat = SerializationFormat.DICT
) -> T:
    """
    Deserialize data to an instance of the target class.
    
    Args:
        data: The data to deserialize (dict or JSON string)
        target_class: The class to instantiate
        format: Input format (JSON string or Python dict)
        
    Returns:
        An instance of the target class
    """
    if format == SerializationFormat.JSON and isinstance(data, str):
        data = json.loads(data)
    
    # Simple case: the target is a primitive type
    if target_class in (str, int, float, bool, dict, list):
        return data
    
    # Handle dataclasses
    if is_dataclass(target_class):
        field_types = get_type_hints(target_class)
        init_kwargs = {}
        
        for field_name, field_type in field_types.items():
            if field_name in data:
                if field_type == datetime.datetime and isinstance(data[field_name], str):
                    init_kwargs[field_name] = datetime.datetime.fromisoformat(data[field_name])
                elif field_type == datetime.date and isinstance(data[field_name], str):
                    init_kwargs[field_name] = datetime.date.fromisoformat(data[field_name])
                else:
                    init_kwargs[field_name] = data[field_name]
        
        return target_class(**init_kwargs)
    
    # Handle classes with from_dict method
    if hasattr(target_class, 'from_dict') and callable(getattr(target_class, 'from_dict')):
        return target_class.from_dict(data)
    
    # Default case: try to initialize with the data as kwargs
    try:
        return target_class(**data)
    except (TypeError, ValueError):
        # Fallback: create an instance and set attributes
        instance = target_class()
        for key, value in data.items():
            setattr(instance, key, value)
        return instance


def to_json(obj: Any, pretty: bool = False, exclude_none: bool = False) -> str:
    """
    Serialize an object to a JSON string.
    
    Args:
        obj: The object to serialize
        pretty: Whether to format the JSON with indentation
        exclude_none: Whether to exclude None values
        
    Returns:
        JSON string representation
    """
    indent = 2 if pretty else None
    dict_data = serialize(obj, SerializationFormat.DICT, exclude_none)
    return json.dumps(dict_data, indent=indent, ensure_ascii=False, default=str)


def from_json(json_str: str, target_class: Type[T]) -> T:
    """
    Deserialize a JSON string to an instance of the target class.
    
    Args:
        json_str: The JSON string to deserialize
        target_class: The class to instantiate
        
    Returns:
        An instance of the target class
    """
    data = json.loads(json_str)
    return deserialize(data, target_class)


# Common serialization mixins to be used by domain models

class SerializableMixin:
    """
    Mixin that provides serialization capabilities to a class.
    
    Classes using this mixin must define:
    1. __serializable_fields__ - list of field names to include in serialization
    2. __optional_fields__ - list of field names that are optional during deserialization
    """
    
    __serializable_fields__: List[str] = []
    __optional_fields__: List[str] = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the object to a dictionary."""
        result = {}
        for field in self.__serializable_fields__:
            if hasattr(self, field):
                value = getattr(self, field)
                result[field] = serialize(value, SerializationFormat.DICT)
        return result
    
    def to_json(self, pretty: bool = False) -> str:
        """Convert the object to a JSON string."""
        return to_json(self.to_dict(), pretty)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SerializableMixin':
        """Create an instance from a dictionary."""
        init_kwargs = {}
        for field in cls.__serializable_fields__:
            if field in data:
                init_kwargs[field] = data[field]
            elif field not in cls.__optional_fields__:
                raise ValueError(f"Missing required field: {field}")
        
        return cls(**init_kwargs)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'SerializableMixin':
        """Create an instance from a JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data) 