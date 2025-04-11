import os
import time
import json
import pickle
import uuid
import logging
from typing import Dict, List, Any, Optional, Union, Tuple, Type, Callable
from abc import ABC, abstractmethod
from enum import Enum
import numpy as np
import datetime
from pathlib import Path

# Setup logging
logger = logging.getLogger("ai_engine")

class ModelVersion:
    """
    Class for managing model versions with semantic versioning.
    """
    def __init__(self, major: int = 0, minor: int = 1, patch: int = 0, metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize a model version.
        
        Args:
            major: Major version number (breaking changes)
            minor: Minor version number (non-breaking features)
            patch: Patch version number (bug fixes)
            metadata: Optional metadata about the model
        """
        self.major = major
        self.minor = minor
        self.patch = patch
        self.metadata = metadata or {}
        self.created_at = datetime.datetime.now().isoformat()
    
    def __str__(self) -> str:
        """
        Get string representation of the version.
        
        Returns:
            Version string in semantic versioning format
        """
        return f"{self.major}.{self.minor}.{self.patch}"
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary.
        
        Returns:
            Dictionary representation of the version
        """
        return {
            "major": self.major,
            "minor": self.minor,
            "patch": self.patch,
            "created_at": self.created_at,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelVersion':
        """
        Create from dictionary.
        
        Args:
            data: Dictionary representation of the version
            
        Returns:
            ModelVersion instance
        """
        instance = cls(
            major=data.get("major", 0),
            minor=data.get("minor", 1),
            patch=data.get("patch", 0),
            metadata=data.get("metadata", {})
        )
        instance.created_at = data.get("created_at", datetime.datetime.now().isoformat())
        return instance
    
    def __eq__(self, other: 'ModelVersion') -> bool:
        """
        Check if two versions are equal.
        
        Args:
            other: Other ModelVersion to compare with
            
        Returns:
            True if versions are equal, False otherwise
        """
        return (self.major == other.major and 
                self.minor == other.minor and 
                self.patch == other.patch)
    
    def __lt__(self, other: 'ModelVersion') -> bool:
        """
        Check if this version is less than another version.
        
        Args:
            other: Other ModelVersion to compare with
            
        Returns:
            True if this version is less than other, False otherwise
        """
        if self.major != other.major:
            return self.major < other.major
        if self.minor != other.minor:
            return self.minor < other.minor
        return self.patch < other.patch

class ModelStatus(Enum):
    """Enum for model status."""
    LOADING = "loading"
    READY = "ready"
    ERROR = "error"
    DEPRECATED = "deprecated"

class BaseModel(ABC):
    """
    Abstract base class for all ML models.
    """
    def __init__(self, model_id: str, version: ModelVersion):
        """
        Initialize a model.
        
        Args:
            model_id: Unique identifier for the model
            version: Model version
        """
        self.model_id = model_id
        self.version = version
        self.status = ModelStatus.LOADING
        self.load_time = None
        self.last_inference_time = None
        self.inference_count = 0
        self.error = None
    
    @abstractmethod
    def preprocess(self, inputs: Any) -> Any:
        """
        Preprocess inputs for model inference.
        
        Args:
            inputs: Raw inputs
            
        Returns:
            Preprocessed inputs
        """
        pass
    
    @abstractmethod
    def predict(self, preprocessed_inputs: Any) -> Any:
        """
        Run inference on preprocessed inputs.
        
        Args:
            preprocessed_inputs: Preprocessed inputs
            
        Returns:
            Raw model outputs
        """
        pass
    
    @abstractmethod
    def postprocess(self, outputs: Any) -> Any:
        """
        Postprocess model outputs.
        
        Args:
            outputs: Raw model outputs
            
        Returns:
            Processed outputs
        """
        pass
    
    def infer(self, inputs: Any) -> Tuple[Any, Dict[str, Any]]:
        """
        End-to-end inference pipeline.
        
        Args:
            inputs: Raw inputs
            
        Returns:
            Tuple of (results, metrics)
        """
        start_time = time.time()
        
        try:
            if self.status != ModelStatus.READY:
                raise RuntimeError(f"Model is not ready: {self.status.value}")
            
            # Preprocessing
            preprocess_start = time.time()
            preprocessed = self.preprocess(inputs)
            preprocess_time = time.time() - preprocess_start
            
            # Inference
            inference_start = time.time()
            outputs = self.predict(preprocessed)
            inference_time = time.time() - inference_start
            
            # Postprocessing
            postprocess_start = time.time()
            results = self.postprocess(outputs)
            postprocess_time = time.time() - postprocess_start
            
            # Update metrics
            self.last_inference_time = time.time()
            self.inference_count += 1
            
            # Calculate metrics
            total_time = time.time() - start_time
            metrics = {
                "preprocess_time": preprocess_time,
                "inference_time": inference_time,
                "postprocess_time": postprocess_time,
                "total_time": total_time,
                "status": "success"
            }
            
            return results, metrics
            
        except Exception as e:
            self.error = str(e)
            logger.error(f"Inference error in model {self.model_id}: {str(e)}")
            
            # Calculate metrics even if there's an error
            total_time = time.time() - start_time
            metrics = {
                "error": str(e),
                "total_time": total_time,
                "status": "error"
            }
            
            return None, metrics
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_id": self.model_id,
            "version": str(self.version),
            "version_full": self.version.to_dict(),
            "status": self.status.value,
            "load_time": self.load_time,
            "last_inference_time": self.last_inference_time,
            "inference_count": self.inference_count,
            "error": self.error,
            "model_type": self.__class__.__name__
        }

class ModelRegistry:
    """
    Registry for managing models.
    """
    def __init__(self, models_dir: str = "models"):
        """
        Initialize the model registry.
        
        Args:
            models_dir: Directory to store model files
        """
        self.models_dir = models_dir
        self.models: Dict[str, BaseModel] = {}
        self.active_versions: Dict[str, str] = {}  # model_type -> model_id
        
        # Ensure models directory exists
        os.makedirs(models_dir, exist_ok=True)
    
    def register_model(self, model: BaseModel) -> None:
        """
        Register a model.
        
        Args:
            model: Model to register
        """
        self.models[model.model_id] = model
        logger.info(f"Registered model {model.model_id} with version {model.version}")
    
    def get_model(self, model_id: str) -> Optional[BaseModel]:
        """
        Get a model by ID.
        
        Args:
            model_id: Model ID
            
        Returns:
            Model if found, None otherwise
        """
        return self.models.get(model_id)
    
    def set_active_version(self, model_type: str, model_id: str) -> None:
        """
        Set the active version for a model type.
        
        Args:
            model_type: Model type
            model_id: Model ID to set as active
        
        Raises:
            ValueError: If model ID is not registered
        """
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} is not registered")
        
        self.active_versions[model_type] = model_id
        logger.info(f"Set active version for {model_type} to {model_id}")
    
    def get_active_model(self, model_type: str) -> Optional[BaseModel]:
        """
        Get the active model for a type.
        
        Args:
            model_type: Model type
            
        Returns:
            Active model if set, None otherwise
        """
        model_id = self.active_versions.get(model_type)
        if model_id:
            return self.models.get(model_id)
        return None
    
    def list_models(self, model_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all registered models.
        
        Args:
            model_type: Optional filter by model type
            
        Returns:
            List of model information dictionaries
        """
        models_info = []
        for model_id, model in self.models.items():
            if model_type and not model_id.startswith(f"{model_type}_"):
                continue
                
            info = model.get_info()
            info["is_active"] = (model_id == self.active_versions.get(model_type, ""))
            models_info.append(info)
        
        return models_info
    
    def save_model_metadata(self, model_id: str) -> None:
        """
        Save model metadata to disk.
        
        Args:
            model_id: Model ID
        """
        model = self.get_model(model_id)
        if not model:
            logger.warning(f"Cannot save metadata for unknown model: {model_id}")
            return
            
        metadata_path = os.path.join(self.models_dir, f"{model_id}_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump({
                "model_id": model.model_id,
                "version": model.version.to_dict(),
                "status": model.status.value,
                "load_time": model.load_time,
                "last_inference_time": model.last_inference_time,
                "inference_count": model.inference_count,
                "error": model.error
            }, f, indent=2)
    
    def load_model_metadata(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Load model metadata from disk.
        
        Args:
            model_id: Model ID
            
        Returns:
            Model metadata if found, None otherwise
        """
        metadata_path = os.path.join(self.models_dir, f"{model_id}_metadata.json")
        if not os.path.exists(metadata_path):
            return None
            
        with open(metadata_path, "r") as f:
            return json.load(f)
    
    def save_registry_state(self) -> None:
        """Save the registry state to disk."""
        state_path = os.path.join(self.models_dir, "registry_state.json")
        with open(state_path, "w") as f:
            json.dump({
                "active_versions": self.active_versions
            }, f, indent=2)
    
    def load_registry_state(self) -> None:
        """Load the registry state from disk."""
        state_path = os.path.join(self.models_dir, "registry_state.json")
        if not os.path.exists(state_path):
            return
            
        with open(state_path, "r") as f:
            state = json.load(f)
            self.active_versions = state.get("active_versions", {})

class InferenceResult:
    """Class to store and manage inference results."""
    
    def __init__(self, 
                 predictions: List[Any],
                 confidence_scores: List[float],
                 model_id: str,
                 model_version: str,
                 inference_time: float,
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize an inference result.
        
        Args:
            predictions: List of predictions
            confidence_scores: List of confidence scores
            model_id: ID of the model used
            model_version: Version of the model used
            inference_time: Time taken for inference
            metadata: Optional metadata
        """
        self.id = str(uuid.uuid4())
        self.predictions = predictions
        self.confidence_scores = confidence_scores
        self.model_id = model_id
        self.model_version = model_version
        self.inference_time = inference_time
        self.timestamp = time.time()
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "id": self.id,
            "predictions": self.predictions,
            "confidence_scores": self.confidence_scores,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "inference_time": self.inference_time,
            "timestamp": self.timestamp,
            "datetime": datetime.datetime.fromtimestamp(self.timestamp).isoformat(),
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'InferenceResult':
        """
        Create from dictionary.
        
        Args:
            data: Dictionary representation
            
        Returns:
            InferenceResult instance
        """
        instance = cls(
            predictions=data.get("predictions", []),
            confidence_scores=data.get("confidence_scores", []),
            model_id=data.get("model_id", "unknown"),
            model_version=data.get("model_version", "0.0.0"),
            inference_time=data.get("inference_time", 0.0),
            metadata=data.get("metadata", {})
        )
        instance.id = data.get("id", instance.id)
        instance.timestamp = data.get("timestamp", instance.timestamp)
        return instance
    
    def get_top_prediction(self) -> Tuple[Any, float]:
        """
        Get the top prediction and its confidence score.
        
        Returns:
            Tuple of (prediction, confidence_score)
        """
        if not self.predictions or not self.confidence_scores:
            return None, 0.0
            
        top_idx = np.argmax(self.confidence_scores)
        return self.predictions[top_idx], self.confidence_scores[top_idx]
    
    def get_predictions_above_threshold(self, threshold: float = 0.5) -> List[Tuple[Any, float]]:
        """
        Get predictions with confidence scores above a threshold.
        
        Args:
            threshold: Confidence threshold
            
        Returns:
            List of (prediction, confidence_score) tuples
        """
        results = []
        for pred, conf in zip(self.predictions, self.confidence_scores):
            if conf >= threshold:
                results.append((pred, conf))
        
        # Sort by confidence (highest first)
        results.sort(key=lambda x: x[1], reverse=True)
        return results

# Create a singleton registry instance
registry = ModelRegistry()

# Export classes and functions
__all__ = [
    "ModelVersion",
    "ModelStatus",
    "BaseModel",
    "ModelRegistry",
    "InferenceResult",
    "registry"
]
