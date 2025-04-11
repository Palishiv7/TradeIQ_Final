"""
Data Validation Utilities for TradeIQ

This module provides standardized data validation capabilities including:
1. Custom validators for domain-specific data types
2. Validation pipeline with detailed error reporting
3. Schema registry for centralized validation rule management
4. Decorators for easy validation in API endpoints
"""

import re
import json
import logging
import functools
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, TypeVar, Union, cast, get_type_hints
from datetime import datetime, date, time
from uuid import UUID
import inspect
from pydantic import BaseModel, Field, validator, root_validator, ValidationError, create_model, Extra

from backend.common.error_handling import DataValidationError, TradeIQError

# Type variables
T = TypeVar('T', bound=BaseModel)
F = TypeVar('F', bound=Callable)

# Configure logging
logger = logging.getLogger(__name__)

# Regex patterns for common validation
PATTERNS = {
    "email": r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$",
    "username": r"^[a-zA-Z0-9_]{3,32}$",
    "password": r"^(?=.*[A-Za-z])(?=.*\d)(?=.*[@$!%*#?&])[A-Za-z\d@$!%*#?&]{8,}$",
    "phone": r"^\+?[0-9]{10,15}$",
    "uuid": r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    "session_id": r"^s_[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    "question_id": r"^q_[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
}

class ValidationResult:
    """Result of a validation operation"""
    
    def __init__(
        self,
        is_valid: bool,
        errors: Optional[List[Dict[str, Any]]] = None,
        validated_data: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize validation result.
        
        Args:
            is_valid: Whether the validation passed
            errors: List of validation errors
            validated_data: Validated data if validation passed
        """
        self.is_valid = is_valid
        self.errors = errors or []
        self.validated_data = validated_data or {}
    
    def __bool__(self) -> bool:
        """Allow using the result in boolean context"""
        return self.is_valid
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert validation result to a dictionary"""
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "validated_data": self.validated_data
        }
    
    def raise_if_invalid(self, data_type: str = "data") -> Dict[str, Any]:
        """
        Raise an exception if validation failed.
        
        Args:
            data_type: Type of data being validated
            
        Returns:
            Validated data if validation passed
            
        Raises:
            DataValidationError: If validation failed
        """
        if not self.is_valid:
            raise DataValidationError(
                data_type=data_type,
                validation_errors=self.errors
            )
        return self.validated_data

# Schema registry for centralized schema management
class SchemaRegistry:
    """Registry for validation schemas"""
    
    _instance = None
    
    def __new__(cls):
        """Implement singleton pattern"""
        if cls._instance is None:
            cls._instance = super(SchemaRegistry, cls).__new__(cls)
            cls._instance._schemas = {}
        return cls._instance
    
    def register(self, name: str, schema: Type[BaseModel]) -> None:
        """
        Register a schema.
        
        Args:
            name: Name to register the schema under
            schema: Pydantic model class to register
        """
        self._schemas[name] = schema
        logger.debug(f"Registered schema: {name}")
    
    def get(self, name: str) -> Optional[Type[BaseModel]]:
        """
        Get a registered schema.
        
        Args:
            name: Name of the schema to get
            
        Returns:
            Registered schema or None if not found
        """
        return self._schemas.get(name)
    
    def validate(
        self,
        name: str,
        data: Dict[str, Any],
        partial: bool = False
    ) -> ValidationResult:
        """
        Validate data against a registered schema.
        
        Args:
            name: Name of the schema to validate against
            data: Data to validate
            partial: Whether to allow partial data
            
        Returns:
            Validation result
            
        Raises:
            ValueError: If the schema is not registered
        """
        schema = self.get(name)
        if schema is None:
            raise ValueError(f"Schema not registered: {name}")
            
        return validate_against_model(schema, data, partial=partial)

# Function to validate data against a Pydantic model
def validate_against_model(
    model: Type[BaseModel],
    data: Dict[str, Any],
    partial: bool = False
) -> ValidationResult:
    """
    Validate data against a Pydantic model.
    
    Args:
        model: Pydantic model class to validate against
        data: Data to validate
        partial: Whether to allow partial data
        
    Returns:
        Validation result
    """
    try:
        # If partial validation is allowed, create a copy of the model with all fields optional
        if partial:
            # Create dynamic model with all fields optional
            fields = {}
            for name, field in model.__fields__.items():
                fields[name] = (
                    Optional[field.type_],
                    Field(default=None, **{k: v for k, v in field.field_info.extra.items() if k != 'default'})
                )
            
            # Create model with optional fields
            optional_model = create_model(
                f"Optional{model.__name__}",
                __module__=model.__module__,
                __base__=model,
                **fields
            )
            
            # Validate with optional model
            instance = optional_model(**data)
            
            # Filter out None values for return
            validated_data = {k: v for k, v in instance.dict().items() if v is not None}
        else:
            # Validate with original model
            instance = model(**data)
            validated_data = instance.dict()
            
        return ValidationResult(
            is_valid=True,
            validated_data=validated_data
        )
    except ValidationError as e:
        # Convert Pydantic validation errors to our format
        errors = []
        for error in e.errors():
            errors.append({
                "field": ".".join(str(loc) for loc in error["loc"]),
                "message": error["msg"],
                "type": error["type"]
            })
            
        return ValidationResult(
            is_valid=False,
            errors=errors
        )

# Decorator for validating function arguments
def validate_args(
    *,
    model: Type[BaseModel] = None,
    schema_name: str = None,
    arg_name: str = "data",
    partial: bool = False,
    raise_exception: bool = True
):
    """
    Decorator for validating function arguments.
    
    This decorator validates a dictionary argument against a Pydantic model
    or a registered schema.
    
    Args:
        model: Pydantic model class to validate against
        schema_name: Name of the registered schema to validate against
        arg_name: Name of the argument to validate
        partial: Whether to allow partial data
        raise_exception: Whether to raise an exception for validation errors
        
    Returns:
        Decorated function
    """
    if model is None and schema_name is None:
        raise ValueError("Either model or schema_name must be provided")
        
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get the argument to validate
            data = None
            if arg_name in kwargs:
                data = kwargs[arg_name]
            else:
                # Try to find by position
                sig = inspect.signature(func)
                param_names = list(sig.parameters.keys())
                if arg_name in param_names:
                    arg_index = param_names.index(arg_name)
                    if arg_index < len(args):
                        data = args[arg_index]
            
            if data is None:
                # Argument not found, just call the function
                return func(*args, **kwargs)
                
            # Get the validation model
            validation_model = None
            if model is not None:
                validation_model = model
            else:
                registry = SchemaRegistry()
                validation_model = registry.get(schema_name)
                if validation_model is None:
                    raise ValueError(f"Schema not registered: {schema_name}")
                    
            # Validate the data
            result = validate_against_model(validation_model, data, partial=partial)
            
            if not result.is_valid and raise_exception:
                # Raise validation error
                raise DataValidationError(
                    data_type=validation_model.__name__,
                    validation_errors=result.errors
                )
                
            # Replace the argument with validated data if validation passed
            if result.is_valid:
                if arg_name in kwargs:
                    kwargs[arg_name] = result.validated_data
                else:
                    # Replace positional argument
                    args = list(args)
                    arg_index = param_names.index(arg_name)
                    args[arg_index] = result.validated_data
                    args = tuple(args)
                    
            # Call the function
            return func(*args, **kwargs)
        
        return cast(F, wrapper)
    
    return decorator

# Common validators for reuse

def validate_pattern(pattern: str, description: str = "value"):
    """
    Create a validator function for pattern validation.
    
    Args:
        pattern: Regex pattern to validate against
        description: Description of the value for error messages
        
    Returns:
        Validator function
    """
    compiled_pattern = re.compile(pattern)
    
    def validator_func(value: str) -> str:
        if not isinstance(value, str):
            raise ValueError(f"{description} must be a string")
            
        if not compiled_pattern.match(value):
            raise ValueError(f"{description} has invalid format")
            
        return value
    
    return validator_func

def validate_length(min_length: Optional[int] = None, max_length: Optional[int] = None, description: str = "value"):
    """
    Create a validator function for length validation.
    
    Args:
        min_length: Minimum length
        max_length: Maximum length
        description: Description of the value for error messages
        
    Returns:
        Validator function
    """
    def validator_func(value: Union[str, List, Dict, Tuple, Set]) -> Union[str, List, Dict, Tuple, Set]:
        if not hasattr(value, "__len__"):
            raise ValueError(f"{description} must have a length")
            
        length = len(value)
        
        if min_length is not None and length < min_length:
            raise ValueError(f"{description} must be at least {min_length} characters")
            
        if max_length is not None and length > max_length:
            raise ValueError(f"{description} must be at most {max_length} characters")
            
        return value
    
    return validator_func

def validate_range(min_value: Optional[Union[int, float]] = None, max_value: Optional[Union[int, float]] = None, description: str = "value"):
    """
    Create a validator function for range validation.
    
    Args:
        min_value: Minimum value
        max_value: Maximum value
        description: Description of the value for error messages
        
    Returns:
        Validator function
    """
    def validator_func(value: Union[int, float]) -> Union[int, float]:
        if not isinstance(value, (int, float)):
            raise ValueError(f"{description} must be a number")
            
        if min_value is not None and value < min_value:
            raise ValueError(f"{description} must be at least {min_value}")
            
        if max_value is not None and value > max_value:
            raise ValueError(f"{description} must be at most {max_value}")
            
        return value
    
    return validator_func

def validate_enum(enum_class: Type[Enum], description: str = "value"):
    """
    Create a validator function for enum validation.
    
    Args:
        enum_class: Enum class to validate against
        description: Description of the value for error messages
        
    Returns:
        Validator function
    """
    def validator_func(value: Any) -> Any:
        try:
            # Handle both value and name matches
            if isinstance(value, str) and hasattr(enum_class, value):
                return getattr(enum_class, value)
            return enum_class(value)
        except (ValueError, KeyError):
            valid_values = [e.value for e in enum_class]
            raise ValueError(f"{description} must be one of: {', '.join(str(v) for v in valid_values)}")
    
    return validator_func

def validate_one_of(valid_values: List[Any], description: str = "value"):
    """
    Create a validator function for one-of validation.
    
    Args:
        valid_values: List of valid values
        description: Description of the value for error messages
        
    Returns:
        Validator function
    """
    def validator_func(value: Any) -> Any:
        if value not in valid_values:
            raise ValueError(f"{description} must be one of: {', '.join(str(v) for v in valid_values)}")
        return value
    
    return validator_func

def validate_json(description: str = "value"):
    """
    Create a validator function for JSON validation.
    
    Args:
        description: Description of the value for error messages
        
    Returns:
        Validator function
    """
    def validator_func(value: Union[str, Dict, List]) -> Dict:
        if isinstance(value, (dict, list)):
            # Already parsed
            return value
            
        if not isinstance(value, str):
            raise ValueError(f"{description} must be a JSON string or object")
            
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            raise ValueError(f"{description} must be valid JSON")
    
    return validator_func

# Predefined validators for common patterns
validate_email = validate_pattern(PATTERNS["email"], "Email")
validate_username = validate_pattern(PATTERNS["username"], "Username")
validate_password = validate_pattern(PATTERNS["password"], "Password")
validate_phone = validate_pattern(PATTERNS["phone"], "Phone number")
validate_uuid = validate_pattern(PATTERNS["uuid"], "UUID")
validate_session_id = validate_pattern(PATTERNS["session_id"], "Session ID")
validate_question_id = validate_pattern(PATTERNS["question_id"], "Question ID")

# Base class for all validation models
class BaseValidationModel(BaseModel):
    """Base model for all validation models with common configuration"""
    
    class Config:
        extra = Extra.forbid  # Forbid extra fields by default
        validate_assignment = True  # Validate when assigning attributes
        arbitrary_types_allowed = True  # Allow arbitrary types

# Create the global schema registry
schema_registry = SchemaRegistry() 