"""
Key Builder Module

This module provides utilities for creating standardized cache keys,
ensuring consistent key structure and handling complex parameters appropriately.
"""

import hashlib
import inspect
import json
from functools import wraps
from typing import Any, Callable, Dict, Optional, Tuple, Union

class KeyBuilder:
    """
    Utility for building standardized cache keys.
    
    This class provides static methods for generating cache keys with
    consistent format, appropriate handling of complex parameters,
    and support for namespacing and versioning.
    """
    
    @staticmethod
    def build(*parts: Any, namespace: Optional[str] = None,
              version: Optional[str] = None) -> str:
        """
        Build a cache key from parts.
        
        Args:
            *parts: Parts of the key, will be converted to strings and joined
            namespace: Optional namespace for the key
            version: Optional version string for the key
            
        Returns:
            A colon-separated key string
        """
        # Process each part to ensure consistent representation
        processed_parts = []
        
        # Add namespace if provided
        if namespace:
            processed_parts.append(str(namespace))
            
        # Process each part
        for part in parts:
            if part is None:
                processed_parts.append("null")
            elif isinstance(part, (int, float, bool, str)):
                # Basic types can be converted directly
                processed_parts.append(str(part))
            elif isinstance(part, (dict, list, tuple, set)):
                # Complex types are hashed for consistency
                part_json = json.dumps(part, sort_keys=True)
                part_hash = hashlib.md5(part_json.encode()).hexdigest()[:10]
                processed_parts.append(part_hash)
            else:
                # For other types, use string representation with class name prefix
                class_name = part.__class__.__name__
                str_value = str(part)
                if len(str_value) > 40:  # Truncate long strings
                    str_value = hashlib.md5(str_value.encode()).hexdigest()[:10]
                processed_parts.append(f"{class_name}:{str_value}")
        
        # Add version if provided
        if version:
            processed_parts.append(f"v{version}")
            
        # Join parts with colon
        return ":".join(processed_parts)
    
    @staticmethod
    def function_key(func: Callable, *args, namespace: Optional[str] = None,
                    version: Optional[str] = None, **kwargs) -> str:
        """
        Build a cache key for a function call.
        
        Args:
            func: The function being called
            *args: Positional arguments to the function
            namespace: Optional namespace for the key
            version: Optional version string for the key
            **kwargs: Keyword arguments to the function
            
        Returns:
            A cache key for the function call
        """
        # Get function module and name
        module_name = func.__module__
        func_name = func.__qualname__
        
        # Create parts for the key
        parts = [module_name, func_name]
        
        # Add args if present
        if args:
            parts.append("args")
            parts.extend(args)
        
        # Add kwargs if present (sorted for consistency)
        if kwargs:
            parts.append("kwargs")
            for k, v in sorted(kwargs.items()):
                parts.append(k)
                parts.append(v)
                
        # Build key with namespace and version
        return KeyBuilder.build(*parts, namespace=namespace, version=version)
    
    @staticmethod
    def query_key(query_type: str, params: Dict[str, Any], namespace: Optional[str] = None,
                 version: Optional[str] = None) -> str:
        """
        Build a cache key for a database query.
        
        Args:
            query_type: Type of query (e.g., 'select', 'count')
            params: Query parameters
            namespace: Optional namespace for the key
            version: Optional version string for the key
            
        Returns:
            A cache key for the query
        """
        # Create parts for the key
        parts = ["query", query_type]
        
        # Sort params for consistent key generation
        param_json = json.dumps(params, sort_keys=True)
        param_hash = hashlib.md5(param_json.encode()).hexdigest()[:16]
        parts.append(param_hash)
        
        # Build key with namespace and version
        return KeyBuilder.build(*parts, namespace=namespace, version=version)
    
    @staticmethod
    def entity_key(entity_type: str, entity_id: Union[str, int], 
                  subresource: Optional[str] = None, namespace: Optional[str] = None,
                  version: Optional[str] = None) -> str:
        """
        Build a cache key for an entity.
        
        Args:
            entity_type: Type of entity (e.g., 'user', 'question')
            entity_id: ID of the entity
            subresource: Optional subresource of the entity
            namespace: Optional namespace for the key
            version: Optional version string for the key
            
        Returns:
            A cache key for the entity
        """
        # Create parts for the key
        parts = [entity_type, str(entity_id)]
        
        # Add subresource if provided
        if subresource:
            parts.append(subresource)
            
        # Build key with namespace and version
        return KeyBuilder.build(*parts, namespace=namespace, version=version)
    
    @staticmethod
    def collection_key(entity_type: str, *filters, sort: Optional[str] = None,
                      limit: Optional[int] = None, namespace: Optional[str] = None,
                      version: Optional[str] = None) -> str:
        """
        Build a cache key for a collection of entities.
        
        Args:
            entity_type: Type of entities in the collection
            *filters: Filters applied to the collection
            sort: Optional sort order
            limit: Optional limit on collection size
            namespace: Optional namespace for the key
            version: Optional version string for the key
            
        Returns:
            A cache key for the collection
        """
        # Create parts for the key
        parts = [entity_type, "collection"]
        
        # Add filters if present
        if filters:
            filter_parts = []
            for filter_item in filters:
                if isinstance(filter_item, tuple) and len(filter_item) == 2:
                    filter_parts.append(f"{filter_item[0]}:{filter_item[1]}")
                else:
                    filter_parts.append(str(filter_item))
            parts.extend(filter_parts)
        
        # Add sort if provided
        if sort:
            parts.append(f"sort:{sort}")
            
        # Add limit if provided
        if limit:
            parts.append(f"limit:{limit}")
            
        # Build key with namespace and version
        return KeyBuilder.build(*parts, namespace=namespace, version=version) 