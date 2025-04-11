"""
Candlestick Pattern AI Module

This module provides AI-based detection and analysis of candlestick patterns.
It includes both rule-based and CNN-based pattern recognition models.
"""

import os
import time
from typing import Dict, List, Any, Optional, Union, Tuple, ClassVar, Set
from enum import Enum
import logging
from datetime import datetime
from abc import ABC, abstractmethod
import weakref
import functools
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import threading

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logging.warning("OpenCV not available. Image processing will be limited.")

# Import from base assessment architecture
from backend.assessments.base.models import QuestionDifficulty
from backend.assessments.base.services import ExplanationGenerator

# Import from common modules
from backend.common.ai_engine import (
    BaseModel, ModelVersion, ModelStatus, InferenceResult, registry
)
from backend.common.logger import app_logger
from backend.common.cache import cached, async_cached

# Import candlestick-specific modules
from backend.assessments.candlestick_patterns.candlestick_config import (
    CANDLESTICK_PATTERNS, DIFFICULTY_LEVELS, ASSESSMENT_CONFIG
)
from backend.assessments.candlestick_patterns.candlestick_utils import (
    Candle, CandlestickData, normalize_market_data
)

# Setup logging
logger = app_logger.getChild("candlestick_ai")

# Constants for pattern detection
PATTERN_CONFIDENCE_THRESHOLD = 0.6
MAX_BATCH_SIZE = 16
MODEL_CACHE_TTL = 3600  # 1 hour


# Define pattern taxonomy - moved to a class to avoid global variables
class PatternTaxonomy:
    """Classification and details of candlestick patterns."""
    
    class Category(Enum):
        """Enum for candlestick pattern categories."""
        SINGLE = "single"
        DOUBLE = "double"
        TRIPLE = "triple"
        COMPLEX = "complex"

    # Pattern definitions
    PATTERNS: ClassVar[Dict[Category, List[Dict[str, str]]]] = {
        Category.SINGLE: [
            {"name": "Doji", "description": "A candlestick with very small or no body (open and close are almost equal), indicating market indecision."},
            {"name": "Hammer", "description": "A bullish reversal pattern that forms during a downtrend, with a small body at the upper end and a long lower shadow."},
            # Other single patterns...
        ],
        Category.DOUBLE: [
            {"name": "Bullish Engulfing", "description": "A two-candle pattern where a bullish candle completely engulfs the previous bearish candle, indicating a potential upward reversal."},
            # Other double patterns...
        ],
        # Triple and complex patterns...
    }
    
    # Cache for pattern lookups
    _pattern_cache: ClassVar[Dict[str, Dict[str, Any]]] = {}
    
    @classmethod
    def get_all_patterns(cls) -> List[Dict[str, str]]:
        """Get a flat list of all patterns."""
        if "all_patterns" not in cls._pattern_cache:
            patterns = []
            for category_patterns in cls.PATTERNS.values():
                patterns.extend(category_patterns)
            cls._pattern_cache["all_patterns"] = patterns
        return cls._pattern_cache["all_patterns"]
    
    @classmethod
    def get_pattern_description(cls, pattern_name: str) -> Optional[str]:
        """Get description for a pattern by name."""
        cache_key = f"desc_{pattern_name}"
        if cache_key in cls._pattern_cache:
            return cls._pattern_cache[cache_key]
            
        for category_patterns in cls.PATTERNS.values():
            for pattern in category_patterns:
                if pattern["name"] == pattern_name:
                    cls._pattern_cache[cache_key] = pattern["description"]
                    return pattern["description"]
        
        cls._pattern_cache[cache_key] = None
        return None
    
    @classmethod
    def get_pattern_category(cls, pattern_name: str) -> Optional[Category]:
        """Get category for a pattern by name."""
        cache_key = f"cat_{pattern_name}"
        if cache_key in cls._pattern_cache:
            return cls._pattern_cache[cache_key]
            
        for category, category_patterns in cls.PATTERNS.items():
            for pattern in category_patterns:
                if pattern["name"] == pattern_name:
                    cls._pattern_cache[cache_key] = category
                    return category
        
        cls._pattern_cache[cache_key] = None
        return None
        
    @classmethod
    def get_patterns_by_category(cls, category: Category) -> List[Dict[str, str]]:
        """Get all patterns in a specific category."""
        return cls.PATTERNS.get(category, [])
        
    @classmethod
    def get_pattern_names(cls) -> Set[str]:
        """Get a set of all pattern names."""
        if "pattern_names" not in cls._pattern_cache:
            pattern_names = {pattern["name"] for patterns in cls.PATTERNS.values() 
                            for pattern in patterns}
            cls._pattern_cache["pattern_names"] = pattern_names
        return cls._pattern_cache["pattern_names"]

# Feature extraction for candlestick patterns
class CandlestickAnalyzer:
    """Utility class for analyzing candlestick data."""
    
    # Cache for computed features
    _feature_cache = weakref.WeakKeyDictionary()
    
    @staticmethod
    def extract_features(candles: List[Candle]) -> Dict[str, float]:
        """
        Extract numerical features from candlestick data.
        
        Args:
            candles: List of candle objects to analyze
            
        Returns:
            Dictionary of numerical features extracted from the candles
        """
        if not candles:
            return {}
        
        # Use cache for repeated analysis of same candles
        cache_key = tuple(id(c) for c in candles)
        if cache_key in CandlestickAnalyzer._feature_cache:
            return CandlestickAnalyzer._feature_cache[cache_key]
            
        features = {}
        
        # Basic stats
        features["count"] = len(candles)
        
        # Calculate body sizes
        body_sizes = [c.body_size for c in candles]
        features["avg_body_size"] = sum(body_sizes) / len(body_sizes) if body_sizes else 0
        features["max_body_size"] = max(body_sizes) if body_sizes else 0
        features["min_body_size"] = min(body_sizes) if body_sizes else 0
        
        # Calculate shadow sizes
        upper_shadows = [c.upper_shadow for c in candles]
        lower_shadows = [c.lower_shadow for c in candles]
        features["avg_upper_shadow"] = sum(upper_shadows) / len(upper_shadows) if upper_shadows else 0
        features["avg_lower_shadow"] = sum(lower_shadows) / len(lower_shadows) if lower_shadows else 0
        
        # Calculate price movements
        if len(candles) > 1:
            price_changes = [(candles[i].close - candles[i-1].close) for i in range(1, len(candles))]
            features["price_trend"] = sum(1 for pc in price_changes if pc > 0) / len(price_changes)
            features["avg_price_change"] = sum(price_changes) / len(price_changes)
            features["price_volatility"] = np.std(price_changes) if len(price_changes) > 1 else 0
        
        # Volume analysis if available
        volumes = [getattr(c, 'volume', 0) for c in candles]
        if any(volumes):
            features["avg_volume"] = sum(volumes) / len(volumes)
            features["volume_trend"] = sum(1 for i in range(1, len(volumes)) 
                                         if volumes[i] > volumes[i-1]) / (len(volumes) - 1) if len(volumes) > 1 else 0
        
        # Store in cache for future use
        CandlestickAnalyzer._feature_cache[cache_key] = features
        return features
    
    @staticmethod
    def preprocess_image(image: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        """
        Preprocess image for model input.
        
        Args:
            image: Input image as numpy array
            target_size: Target size for resized image
            
        Returns:
            Preprocessed image ready for model input
            
        Raises:
            ImportError: If OpenCV is not available
        """
        if not CV2_AVAILABLE:
            raise ImportError("OpenCV is required for image preprocessing")
            
        try:
            # Handle empty or invalid images
            if image is None or image.size == 0 or np.max(image) == np.min(image):
                # Return blank image of target size with 3 channels
                return np.zeros((*target_size, 3), dtype=np.float32)
            
            # Convert to RGB if grayscale
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:  # Convert RGBA to RGB
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            
            # Resize to target size
            image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
            
            # Normalize values to [0, 1]
            if image.max() > 1.0:
                image = image.astype(np.float32) / 255.0
                
            return image.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            # Return zeros as fallback
            return np.zeros((*target_size, 3), dtype=np.float32)
    
    @staticmethod
    def candlestick_to_image(data: CandlestickData) -> np.ndarray:
        """
        Convert candlestick data to image representation.
        
        Args:
            data: CandlestickData object to convert
            
        Returns:
            Image representation of candlestick data as numpy array
        """
        try:
            # Use data's built-in chart plotting
            image_str = data.plot_candlestick_chart()
            
            # Decode base64 image if OpenCV is available
            if CV2_AVAILABLE:
                import base64
                from io import BytesIO
                
                # Handle empty image data
                if not image_str:
                    return np.zeros((224, 224, 3), dtype=np.uint8)
                
                try:
                    image_bytes = base64.b64decode(image_str)
                    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
                    return image
                except Exception as e:
                    logger.error(f"Error decoding image: {str(e)}")
                    return np.zeros((224, 224, 3), dtype=np.uint8)
            else:
                # Return placeholder data if OpenCV not available
                return np.zeros((224, 224, 3), dtype=np.uint8)
        except Exception as e:
            logger.error(f"Error converting candlestick to image: {str(e)}")
            return np.zeros((224, 224, 3), dtype=np.uint8)
    
    @staticmethod
    def analyze_trend(candles: List[Candle]) -> Dict[str, Any]:
        """
        Analyze market trend from candlestick data.
        
        Args:
            candles: List of candles to analyze
            
        Returns:
            Dictionary with trend information
        """
        if not candles or len(candles) < 3:
            return {"trend": "unknown", "strength": 0.0}
            
        # Calculate price movements
        closes = [c.close for c in candles]
        price_changes = [closes[i] - closes[i-1] for i in range(1, len(closes))]
        
        # Simple moving average
        if len(closes) >= 5:
            sma5 = sum(closes[-5:]) / 5
            sma_trend = "bullish" if closes[-1] > sma5 else "bearish"
        else:
            sma_trend = "unknown"
            
        # Calculate trend strength
        up_moves = sum(1 for pc in price_changes if pc > 0)
        down_moves = sum(1 for pc in price_changes if pc < 0)
        
        if up_moves > down_moves:
            trend = "bullish"
            strength = up_moves / len(price_changes)
        elif down_moves > up_moves:
            trend = "bearish"
            strength = down_moves / len(price_changes)
        else:
            trend = "sideways"
            strength = 0.5
            
        return {
            "trend": trend,
            "strength": strength,
            "sma_trend": sma_trend,
            "volatility": np.std(price_changes) if len(price_changes) > 1 else 0
        }


# Pattern recognition rules
class PatternDetector:
    """Rules for detecting candlestick patterns."""
    
    def __init__(self):
        """Initialize pattern detector with optimized config."""
        self.detection_thresholds = {
            "doji": 0.1,  # Body to range ratio threshold
            "hammer": 0.2,  # Body to range ratio threshold for hammer
            "engulfing": 0.8,  # Body overlap percentage for engulfing patterns
            "star": 0.3,  # Star pattern body size threshold
            "harami": 0.7,  # Harami containment factor
        }
        # Cache for detection results
        self._detection_cache = {}
    
    def is_doji(self, candle: Candle, threshold: Optional[float] = None) -> bool:
        """
        Check if a candle is a doji pattern.
        
        Args:
            candle: The candle to check
            threshold: Body to range ratio threshold (defaults to class threshold)
            
        Returns:
            Boolean indicating if the candle is a doji
        """
        # Use default threshold if not specified
        threshold = threshold or self.detection_thresholds["doji"]
        
        # Validate inputs to prevent division by zero
        if candle.range_size == 0:
            return False
            
        # Calculate the ratio of body size to total range
        body_to_range_ratio = candle.body_size / candle.range_size
        
        # A doji has a very small body relative to its range
        return body_to_range_ratio <= threshold
    
    def is_hammer(self, candle: Candle, context_candles: Optional[List[Candle]] = None) -> Tuple[bool, float]:
        """
        Check if a candle is a hammer pattern.
        
        Args:
            candle: The candle to check
            context_candles: Previous candles to establish trend context
            
        Returns:
            Tuple of (is_hammer, confidence)
        """
        # No hammer if no range
        if candle.range_size == 0:
            return False, 0.0
        
        # Check for small body
        body_to_range_ratio = candle.body_size / candle.range_size
        if body_to_range_ratio > self.detection_thresholds["hammer"]:
            return False, 0.0
            
        # Check for long lower shadow (2-3x the body size)
        if candle.lower_shadow < candle.body_size * 2:
            return False, 0.0
            
        # Check for minimal upper shadow
        if candle.upper_shadow > candle.body_size:
            return False, 0.0
            
        # Calculate confidence based on pattern clarity
        confidence = 0.6  # Base confidence
        
        # Increase confidence if in a downtrend (context matters)
        if context_candles and len(context_candles) >= 3:
            # Check if we're in a downtrend
            trend = CandlestickAnalyzer.analyze_trend(context_candles)
            if trend["trend"] == "bearish":
                confidence += 0.2 * trend["strength"]
        
        return True, min(confidence, 1.0)  # Cap at 1.0
    
    def is_engulfing(self, candles: List[Candle]) -> Tuple[bool, float, str]:
        """
        Check for engulfing patterns (bullish or bearish).
        
        Args:
            candles: List of candles to check (need at least 2)
            
        Returns:
            Tuple of (is_engulfing, confidence, pattern_type)
        """
        if len(candles) < 2:
            return False, 0.0, ""
            
        # Get the last two candles
        prev_candle, current_candle = candles[-2], candles[-1]
        
        # Check if current candle's body engulfs previous candle's body
        is_body_engulfing = (
            current_candle.open > prev_candle.open and 
            current_candle.close < prev_candle.close
        ) or (
            current_candle.open < prev_candle.open and 
            current_candle.close > prev_candle.close
        )
        
        if not is_body_engulfing:
            return False, 0.0, ""
            
        # Determine pattern type (bullish or bearish)
        if current_candle.close > current_candle.open:  # Current is bullish
            pattern_type = "Bullish Engulfing"
        else:  # Current is bearish
            pattern_type = "Bearish Engulfing"
            
        # Calculate confidence based on size ratio
        current_body_size = abs(current_candle.close - current_candle.open)
        prev_body_size = abs(prev_candle.close - prev_candle.open)
        
        # More confidence if current body is significantly larger
        size_ratio = current_body_size / prev_body_size if prev_body_size > 0 else 2.0
        confidence = min(0.7 + (size_ratio - 1) * 0.15, 0.95)
        
        return True, confidence, pattern_type
    
    def detect_patterns(self, candles: List[Candle]) -> List[Dict[str, Any]]:
        """
        Detect all candlestick patterns in the given data.
        
        Args:
            candles: List of candles to analyze
            
        Returns:
            List of detected patterns with confidence scores
        """
        if not candles:
            return []
            
        # Create a cache key from candle IDs
        cache_key = tuple(id(c) for c in candles)
        if cache_key in self._detection_cache:
            return self._detection_cache[cache_key]
            
        detected_patterns = []
        
        try:
            # Single candle patterns
            if len(candles) >= 1:
                current = candles[-1]
                
                # Check for Doji
                if self.is_doji(current):
                    detected_patterns.append({
                        "name": "Doji",
                        "confidence": 0.9,
                        "index": len(candles) - 1
                    })
                
                # Check for Hammer (with context)
                is_hammer, hammer_confidence = self.is_hammer(
                    current, 
                    context_candles=candles[:-1] if len(candles) > 1 else None
                )
                if is_hammer:
                    detected_patterns.append({
                        "name": "Hammer",
                        "confidence": hammer_confidence,
                        "index": len(candles) - 1
                    })
                
                # Add more single candle patterns here...
            
            # Multi-candle patterns
            if len(candles) >= 2:
                # Check for engulfing patterns
                is_engulfing, engulfing_confidence, engulfing_type = self.is_engulfing(candles)
                if is_engulfing:
                    detected_patterns.append({
                        "name": engulfing_type,
                        "confidence": engulfing_confidence,
                        "indices": [len(candles) - 2, len(candles) - 1]
                    })
                
                # Add more multi-candle patterns here...
                
            # Sort patterns by confidence (highest first)
            detected_patterns.sort(key=lambda x: x["confidence"], reverse=True)
            
            # Cache detection results
            self._detection_cache[cache_key] = detected_patterns
            
            return detected_patterns
            
        except Exception as e:
            logger.error(f"Error during pattern detection: {str(e)}")
            return []
    
    def clear_cache(self):
        """Clear the pattern detection cache."""
        self._detection_cache.clear()


# Base model for candlestick pattern recognition
class CandlestickModel(BaseModel):
    """Base class for candlestick pattern recognition models."""
    
    def __init__(
        self, 
        model_id: str, 
        version: ModelVersion,
        pattern_categories: Optional[List[PatternTaxonomy.Category]] = None
    ):
        """
        Initialize the model.
        
        Args:
            model_id: Unique identifier for the model
            version: Version information
            pattern_categories: List of pattern categories to detect
        """
        super().__init__(model_id, version)
        self.pattern_categories = pattern_categories or list(PatternTaxonomy.Category)
        
        # Initialize pattern list
        self.patterns = []
        for category in self.pattern_categories:
            for pattern in PatternTaxonomy.PATTERNS.get(category, []):
                self.patterns.append(pattern["name"])
                
        # Set up caching for expensive operations
        self._cache = {}
        self._cache_ttl = 300  # 5 minutes
        self._cache_timestamps = {}
        
        # Initialize stats
        self.total_inferences = 0
        self.successful_inferences = 0
        self.avg_inference_time = 0.0
    
    def preprocess(self, inputs: Union[CandlestickData, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Preprocess input data for model inference.
        
        Args:
            inputs: CandlestickData object or dictionary of market data
            
        Returns:
            Preprocessed inputs ready for prediction
            
        Raises:
            ValueError: If input format is invalid
        """
        # Convert dictionary to CandlestickData if needed
        if isinstance(inputs, dict):
            try:
                if "candles" in inputs and "symbol" in inputs:
                    candlestick_data = CandlestickData.from_dict(inputs)
                else:
                    # Try to normalize the data first
                    normalized_data = normalize_market_data(inputs)
                    candlestick_data = CandlestickData.from_dict(normalized_data)
            except Exception as e:
                logger.error(f"Error converting input data: {str(e)}")
                raise ValueError(f"Invalid input format: {str(e)}")
        else:
            candlestick_data = inputs
        
        # Extract features and convert to image if needed
        try:
            # Generate image representation if CV2 available
            if CV2_AVAILABLE:
                image = CandlestickAnalyzer.candlestick_to_image(candlestick_data)
            else:
                image = np.zeros((224, 224, 3))
                
            # Extract numerical features
            features = CandlestickAnalyzer.extract_features(candlestick_data.candles)
            
            # Add trend analysis
            trend_info = CandlestickAnalyzer.analyze_trend(candlestick_data.candles)
            features.update(trend_info)
            
        except Exception as e:
            logger.error(f"Error preprocessing data: {str(e)}")
            image = np.zeros((224, 224, 3))
            features = {}
        
        return {
            "image": image,
            "features": features,
            "original_data": candlestick_data.to_dict(),
            "candle_count": len(candlestick_data.candles),
            "candles": candlestick_data.candles,
            "symbol": candlestick_data.symbol
        }
    
    @abstractmethod
    def predict(self, preprocessed_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run model inference on preprocessed inputs.
        
        Args:
            preprocessed_inputs: Data prepared by preprocess method
            
        Returns:
            Model predictions with metadata
        """
        pass
    
    def postprocess(self, outputs: Dict[str, Any]) -> InferenceResult:
        """
        Postprocess model outputs to generate pattern predictions.
        
        Args:
            outputs: Raw model outputs from predict method
            
        Returns:
            Structured inference result
        """
        # Get predictions and confidence scores
        predictions = outputs.get("predictions", [])
        confidence_scores = outputs.get("confidence_scores", [])
        
        # Validate output lengths match
        if len(predictions) != len(confidence_scores):
            logger.warning(f"Prediction count ({len(predictions)}) doesn't match confidence count ({len(confidence_scores)})")
            # Truncate to shorter length
            min_len = min(len(predictions), len(confidence_scores))
            predictions = predictions[:min_len]
            confidence_scores = confidence_scores[:min_len]
        
        # Update stats
        self.total_inferences += 1
        self.successful_inferences += 1
        
        # Update average inference time
        inference_time = outputs.get("inference_time", 0.0)
        self.avg_inference_time = (self.avg_inference_time * (self.total_inferences - 1) + inference_time) / self.total_inferences
        
        # Create inference result
        return InferenceResult(
            predictions=predictions,
            confidence_scores=confidence_scores,
            model_id=self.model_id,
            model_version=str(self.version),
            inference_time=inference_time,
            metadata={
                "pattern_categories": [c.value for c in self.pattern_categories],
                "threshold_used": outputs.get("threshold", 0.5),
                "original_data_shape": outputs.get("original_data_shape", None),
                "candle_count": outputs.get("candle_count", 0),
                "model_stats": {
                    "total_inferences": self.total_inferences,
                    "successful_inferences": self.successful_inferences,
                    "avg_inference_time": self.avg_inference_time
                }
            }
        )
    
    def get_pattern_description(self, pattern_name: str) -> Optional[str]:
        """
        Get description for a pattern.
        
        Args:
            pattern_name: Name of the pattern
            
        Returns:
            Pattern description or None if not found
        """
        return PatternTaxonomy.get_pattern_description(pattern_name)
    
    def get_pattern_category(self, pattern_name: str) -> Optional[PatternTaxonomy.Category]:
        """
        Get category for a pattern.
        
        Args:
            pattern_name: Name of the pattern
            
        Returns:
            Pattern category or None if not found
        """
        return PatternTaxonomy.get_pattern_category(pattern_name)
    
    def _add_to_cache(self, key: str, value: Any) -> None:
        """Add value to cache with timestamp."""
        self._cache[key] = value
        self._cache_timestamps[key] = time.time()
    
    def _get_from_cache(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired."""
        if key in self._cache:
            timestamp = self._cache_timestamps.get(key, 0)
            if time.time() - timestamp < self._cache_ttl:
                return self._cache[key]
            else:
                # Remove expired entry
                del self._cache[key]
                del self._cache_timestamps[key]
        return None
    
    def clear_cache(self) -> None:
        """Clear all cached values."""
        self._cache.clear()
        self._cache_timestamps.clear()


# Rule-based model implementation
class RuleBasedModel(CandlestickModel):
    """Rule-based candlestick pattern recognition."""
    
    def __init__(self, model_id: str = "candlestick_rule_based", version: Optional[ModelVersion] = None):
        """
        Initialize rule-based model.
        
        Args:
            model_id: Unique identifier for the model
            version: Version information
        """
        actual_version = version or ModelVersion(
            major=1, minor=0, patch=0, 
            status=ModelStatus.PRODUCTION,
            description="Rule-based candlestick pattern detector"
        )
        super().__init__(model_id, actual_version)
        self.detector = PatternDetector()
        
        # Performance tracking
        self.execution_times = []
        self.max_execution_times = 100  # Limit the number of tracked times
    
    def predict(self, preprocessed_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run pattern detection using rules.
        
        Args:
            preprocessed_inputs: Preprocessed input data
            
        Returns:
            Dictionary with prediction results
        """
        start_time = time.time()
        
        try:
            # Get candles from original data
            original_data = preprocessed_inputs.get("original_data", {})
            candle_data = CandlestickData.from_dict(original_data)
            candles = candle_data.candles
            
            # Check for cached results
            cache_key = f"pred_{id(candle_data)}"
            cached_result = self._get_from_cache(cache_key)
            if cached_result is not None:
                cached_result["from_cache"] = True
                return cached_result
            
            # Apply rules to detect patterns
            detected_patterns = self.detector.detect_patterns(candles)
            
            # Extract pattern names and confidence scores
            predictions = []
            confidences = []
            for pattern in detected_patterns:
                predictions.append(pattern["name"])
                confidences.append(pattern["confidence"])
            
            # If no patterns detected, add a "No Pattern" prediction
            if not predictions:
                predictions.append("No Pattern")
                confidences.append(1.0)
                
            # Ensure we return at most 5 patterns
            predictions = predictions[:5]
            confidences = confidences[:5]
            
            inference_time = time.time() - start_time
            
            # Track execution time
            self.execution_times.append(inference_time)
            if len(self.execution_times) > self.max_execution_times:
                self.execution_times.pop(0)
            
            # Prepare result
            result = {
                "predictions": predictions,
                "confidence_scores": confidences,
                "inference_time": inference_time,
                "threshold": 0.6,
                "candle_count": len(candles),
                "avg_execution_time": sum(self.execution_times) / len(self.execution_times) if self.execution_times else 0,
                "from_cache": False
            }
            
            # Cache the result
            self._add_to_cache(cache_key, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in rule-based prediction: {str(e)}")
            inference_time = time.time() - start_time
            
            # Return fallback result
            return {
                "predictions": ["Error", "No Pattern"],
                "confidence_scores": [0.0, 1.0],
                "inference_time": inference_time,
                "threshold": 0.6,
                "candle_count": 0,
                "error": str(e)
            }
    
    def bulk_predict(self, multiple_inputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process multiple inputs in parallel.
        
        Args:
            multiple_inputs: List of preprocessed inputs
            
        Returns:
            List of prediction results
        """
        if not multiple_inputs:
            return []
            
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=min(8, len(multiple_inputs))) as executor:
            results = list(executor.map(self.predict, multiple_inputs))
            
        return results


# CNN-based model implementation 
class CNNModel(CandlestickModel):
    """CNN-based candlestick pattern recognition."""
    
    def __init__(
        self, 
        model_id: str = "candlestick_cnn", 
        version: Optional[ModelVersion] = None,
        pattern_categories: Optional[List[PatternTaxonomy.Category]] = None,
        batch_size: int = 8,
        use_quantization: bool = True,
        confidence_threshold: float = 0.5,
        model_path: Optional[str] = None
    ):
        """
        Initialize CNN model.
        
        Args:
            model_id: Unique identifier for the model
            version: Version information
            pattern_categories: List of pattern categories to detect
            batch_size: Batch size for inference
            use_quantization: Whether to use TFLite quantization
            confidence_threshold: Threshold for pattern detection
            model_path: Path to model file (optional)
        """
        actual_version = version or ModelVersion(
            major=1, minor=0, patch=0, 
            status=ModelStatus.PRODUCTION,
            description="CNN-based candlestick pattern detector"
        )
        super().__init__(model_id, actual_version, pattern_categories)
        self.batch_size = min(batch_size, MAX_BATCH_SIZE)  # Limit batch size
        self.use_quantization = use_quantization
        self.confidence_threshold = confidence_threshold
        self.model_path = model_path
        
        # TensorFlow status
        self.tensorflow_available = False
        self.tf_model = None
        self.tflite_model = None
        self.tflite_interpreter = None
        self.rule_based_fallback = None
        self.model_loaded = False
        self.model_load_attempted = False
        
        # Resource management
        self._tf_session = None
        self._gpu_memory_limit = 0.3  # Use at most 30% of GPU memory
        
        # Performance tracking
        self.tf_execution_times = []
        self.inference_count = 0
        
        # Attempt to initialize TensorFlow
        self._initialize_model()
    
    def _initialize_model(self):
        """
        Initialize the underlying TensorFlow model.
        
        This method attempts to load TensorFlow and the model.
        If TensorFlow is not available, it will set up a rule-based fallback.
        """
        if self.model_load_attempted:
            return
        
        self.model_load_attempted = True
        
        try:
            # Import TensorFlow conditionally
            import tensorflow as tf
            self.tensorflow_available = True
            
            # Configure TensorFlow to use limited GPU memory
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                try:
                    # Limit memory growth to avoid OOM errors
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    
                    # Set memory limit if supported
                    if hasattr(tf.config.experimental, 'set_virtual_device_configuration'):
                        tf.config.experimental.set_virtual_device_configuration(
                            gpus[0],
                            [tf.config.experimental.VirtualDeviceConfiguration(
                                memory_limit=int(self._gpu_memory_limit * 1024))]
                        )
                except RuntimeError as e:
                    logger.warning(f"GPU memory configuration failed: {str(e)}")
            
            # Load model based on path or use default location
            model_path = self.model_path or os.path.join(
                os.path.dirname(__file__), 
                "models", 
                "candlestick_cnn.h5"
            )
            
            # Check if model file exists
            if not os.path.exists(model_path):
                logger.warning(f"Model file not found at {model_path}")
                self._fallback_to_rules()
                return
                
            # Load TensorFlow model
            try:
                self.tf_model = tf.keras.models.load_model(model_path)
                logger.info(f"Loaded TensorFlow model from {model_path}")
                
                # Create TFLite model if quantization is enabled
                if self.use_quantization:
                    self._create_tflite_model()
                    
                self.model_loaded = True
                
            except Exception as e:
                logger.error(f"Error loading TensorFlow model: {str(e)}")
                self._fallback_to_rules()
                
        except ImportError:
            logger.warning("TensorFlow not available. Using rule-based fallback.")
            self._fallback_to_rules()
    
    def _create_tflite_model(self):
        """Create and load a TFLite model from the TensorFlow model."""
        try:
            import tensorflow as tf
            
            # Convert model to TFLite
            converter = tf.lite.TFLiteConverter.from_keras_model(self.tf_model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            tflite_model = converter.convert()
            
            # Create interpreter
            self.tflite_interpreter = tf.lite.Interpreter(model_content=tflite_model)
            self.tflite_interpreter.allocate_tensors()
            
            # Get input and output details
            self.input_details = self.tflite_interpreter.get_input_details()
            self.output_details = self.tflite_interpreter.get_output_details()
            
            logger.info("TFLite model created and loaded successfully")
            
        except Exception as e:
            logger.error(f"Error creating TFLite model: {str(e)}")
            self.use_quantization = False
    
    def _fallback_to_rules(self):
        """Initialize rule-based fallback model."""
        if not self.rule_based_fallback:
            self.rule_based_fallback = RuleBasedModel()
            logger.info("Initialized rule-based fallback model")
    
    def _predict_with_tf(self, image):
        """
        Run prediction using TensorFlow model.
        
        Args:
            image: Preprocessed image
            
        Returns:
            Model predictions
        """
        import tensorflow as tf
        
        # Ensure image has batch dimension
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
            
        # Run prediction
        predictions = self.tf_model.predict(image, batch_size=1, verbose=0)
        
        return predictions
    
    def _predict_with_tflite(self, image):
        """
        Run prediction using TFLite model.
        
        Args:
            image: Preprocessed image
            
        Returns:
            Model predictions
        """
        # Ensure image has batch dimension
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
            
        # Set input tensor
        input_shape = self.input_details[0]['shape']
        if image.shape != tuple(input_shape):
            # Resize image to match input shape
            image = np.resize(image, input_shape)
            
        self.tflite_interpreter.set_tensor(self.input_details[0]['index'], image.astype(np.float32))
        
        # Run inference
        self.tflite_interpreter.invoke()
        
        # Get output tensor
        predictions = self.tflite_interpreter.get_tensor(self.output_details[0]['index'])
        
        return predictions
    
    def predict(self, preprocessed_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run pattern detection using CNN or fallback to rules.
        
        Args:
            preprocessed_inputs: Preprocessed input data
            
        Returns:
            Dictionary with prediction results
        """
        start_time = time.time()
        self.inference_count += 1
        
        # Try to initialize model if not already done
        if not self.model_loaded and not self.model_load_attempted:
            self._initialize_model()
        
        if not self.tensorflow_available or not self.model_loaded:
            # Use rule-based fallback
            self._fallback_to_rules()
            result = self.rule_based_fallback.predict(preprocessed_inputs)
            result["fallback_used"] = True
            return result
        
        # Get preprocessed image
        image = preprocessed_inputs.get("image")
        
        # Ensure image has correct shape
        if image is None or image.size == 0:
            self._fallback_to_rules()
            result = self.rule_based_fallback.predict(preprocessed_inputs)
            result["fallback_used"] = True
            result["fallback_reason"] = "Invalid image"
            return result
        
        # Run prediction
        try:
            # Check cache first
            cache_key = f"cnn_pred_{hash(image.tobytes())}"
            cached_result = self._get_from_cache(cache_key)
            if cached_result is not None:
                cached_result["from_cache"] = True
                return cached_result
            
            # Use TF or TFLite based on settings
            if self.use_quantization and self.tflite_interpreter:
                # TFLite prediction
                raw_predictions = self._predict_with_tflite(image)
            else:
                # TensorFlow prediction
                raw_predictions = self._predict_with_tf(image)
            
            # Process predictions
            # Assuming output is a probability distribution over patterns
            pattern_indices = np.argsort(raw_predictions[0])[::-1]
            top_indices = pattern_indices[:5]  # Get top 5 predictions
            
            pattern_names = []
            confidences = []
            
            for idx in top_indices:
                if idx < len(self.patterns):
                    pattern_name = self.patterns[idx]
                    confidence = float(raw_predictions[0][idx])
                    
                    # Only include predictions above threshold
                    if confidence >= self.confidence_threshold:
                        pattern_names.append(pattern_name)
                        confidences.append(confidence)
            
            # If no patterns meet threshold, add "No Pattern"
            if not pattern_names:
                pattern_names.append("No Pattern")
                confidences.append(1.0)
                
            inference_time = time.time() - start_time
            self.tf_execution_times.append(inference_time)
            
            # Limit the size of execution times list
            if len(self.tf_execution_times) > 100:
                self.tf_execution_times.pop(0)
                
            result = {
                "predictions": pattern_names,
                "confidence_scores": confidences,
                "inference_time": inference_time,
                "threshold": self.confidence_threshold,
                "original_data_shape": image.shape if image is not None else None,
                "candle_count": preprocessed_inputs.get("candle_count", 0),
                "avg_execution_time": sum(self.tf_execution_times) / len(self.tf_execution_times) if self.tf_execution_times else 0,
                "model_type": "tflite" if self.use_quantization else "tensorflow",
                "fallback_used": False,
                "from_cache": False
            }
            
            # Cache the result
            self._add_to_cache(cache_key, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error during CNN prediction: {str(e)}")
            self._fallback_to_rules()
            result = self.rule_based_fallback.predict(preprocessed_inputs)
            result["fallback_used"] = True
            result["fallback_reason"] = str(e)
            return result
    
    def __del__(self):
        """Clean up resources when object is destroyed."""
        # Clean up TensorFlow resources
        if hasattr(self, 'tf_model') and self.tf_model is not None:
            try:
                # Explicitly delete the model to free memory
                del self.tf_model
            except:
                pass
            
        # Clean up TFLite resources
        if hasattr(self, 'tflite_interpreter') and self.tflite_interpreter is not None:
            try:
                del self.tflite_interpreter
            except:
                pass


# Factory for creating pattern models
class ModelFactory:
    """Factory for creating candlestick pattern models."""
    
    _instance_cache = weakref.WeakValueDictionary()
    
    @classmethod
    def create_model(cls, model_type: str, **kwargs) -> CandlestickModel:
        """
        Create a model of the specified type.
        
        Args:
            model_type: Type of model to create ('rule_based' or 'cnn')
            **kwargs: Additional parameters for model initialization
            
        Returns:
            Initialized model instance
            
        Raises:
            ValueError: If model type is not supported
        """
        # Generate a cache key from model type and kwargs
        cache_key = f"{model_type}_{hash(frozenset(kwargs.items()))}"
        
        # Check for cached instance
        if cache_key in cls._instance_cache:
            cached_instance = cls._instance_cache[cache_key]
            if cached_instance is not None:
                logger.debug(f"Using cached model instance for {model_type}")
                return cached_instance
        
        # Create new instance
        try:
            if model_type == "rule_based":
                instance = RuleBasedModel(**kwargs)
            elif model_type == "cnn":
                instance = CNNModel(**kwargs)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
                
            # Cache the instance
            cls._instance_cache[cache_key] = instance
            
            return instance
            
        except Exception as e:
            logger.error(f"Error creating model {model_type}: {str(e)}")
            # Fall back to rule-based model if CNN creation fails
            if model_type == "cnn":
                logger.warning("Falling back to rule_based model")
                return cls.create_model("rule_based", **kwargs)
            raise

    @classmethod
    def clear_cache(cls):
        """Clear the model instance cache."""
        cls._instance_cache.clear()


# Service for pattern detection
class PatternService:
    """Service for detecting candlestick patterns."""
    
    def __init__(
        self, 
        model_type: str = "candlestick", 
        confidence_threshold: float = 0.6,
        preferred_model: str = "rule_based",
        model_ttl: int = 3600  # Time-to-live for models in seconds
    ):
        """
        Initialize the service.
        
        Args:
            model_type: Type identifier for the model
            confidence_threshold: Threshold for pattern detection
            preferred_model: Preferred implementation ('rule_based' or 'cnn')
            model_ttl: Time-to-live for models in seconds
        """
        self.model_type = model_type
        self.confidence_threshold = confidence_threshold
        self.preferred_model = preferred_model
        self.model_ttl = model_ttl
        self.logger = logger.getChild("detection_service")
        
        # Locks for thread safety
        self._model_lock = threading.RLock()
        
        # Model creation timestamp
        self._model_creation_time = 0
        
        # Initialize with the preferred model type
        self._ensure_model()
    
    def _ensure_model(self):
        """
        Initialize models if no models are available.
        
        This method is thread-safe and handles model initialization,
        including fallback to rule-based model if CNN initialization fails.
        """
        with self._model_lock:
            # Check if there's an active model
            active_model = registry.get_active_model(self.model_type)
            
            # Check if model is expired
            current_time = time.time()
            if active_model is not None and (current_time - self._model_creation_time) > self.model_ttl:
                logger.info(f"Model TTL expired after {self.model_ttl} seconds, reinitializing")
                active_model = None
            
            if active_model is None:
                # No active model, create and register preferred model
                self._model_creation_time = current_time
                
                if self.preferred_model == "cnn":
                    try:
                        model = ModelFactory.create_model("cnn", confidence_threshold=self.confidence_threshold)
                        registry.register_model(model)
                        registry.set_active_version(self.model_type, model.model_id)
                        self.logger.info(f"Registered and activated CNN model: {model.model_id}")
                    except Exception as e:
                        self.logger.warning(f"Failed to initialize CNN model: {str(e)}. Falling back to rule-based.")
                        self._register_rule_based()
                else:
                    self._register_rule_based()
    
    def _register_rule_based(self):
        """Register and activate a rule-based model."""
        with self._model_lock:
            try:
                rule_based_model = ModelFactory.create_model("rule_based")
                registry.register_model(rule_based_model)
                registry.set_active_version(self.model_type, rule_based_model.model_id)
                self.logger.info(f"Registered and activated rule-based model: {rule_based_model.model_id}")
            except Exception as e:
                self.logger.error(f"Failed to register rule-based model: {str(e)}")
                raise
    
    def detect_patterns(self, market_data: Union[Dict[str, Any], CandlestickData]) -> Dict[str, Any]:
        """
        Detect patterns in market data.
        
        Args:
            market_data: Market data as dictionary or CandlestickData object
            
        Returns:
            Dictionary with detection results
        """
        start_time = time.time()
        
        try:
            # Get active model
            model = registry.get_active_model(self.model_type)
            if model is None:
                self._ensure_model()
                model = registry.get_active_model(self.model_type)
                
                if model is None:
                    raise ValueError("Failed to initialize pattern detection model")
            
            # Run inference
            results, metrics = model.infer(market_data)
            
            if results is None:
                error_msg = metrics.get("error", "Unknown error")
                self.logger.error(f"Error during pattern detection: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg,
                    "patterns": [],
                    "detected_patterns": [],
                    "confidence_scores": [],
                    "model_id": model.model_id,
                    "model_version": str(model.version),
                    "inference_time": metrics.get("total_time", 0),
                    "total_time": time.time() - start_time
                }
            
            # Filter by confidence threshold
            patterns_above_threshold = results.get_predictions_above_threshold(self.confidence_threshold)
            
            # Add pattern descriptions
            patterns_with_info = []
            for pattern_name, confidence in patterns_above_threshold:
                if pattern_name == "No Pattern":
                    continue
                    
                pattern_info = {
                    "name": pattern_name,
                    "confidence": confidence,
                    "description": model.get_pattern_description(pattern_name),
                    "category": model.get_pattern_category(pattern_name).value if model.get_pattern_category(pattern_name) else None
                }
                patterns_with_info.append(pattern_info)
            
            # Get top prediction
            top_prediction = results.get_top_prediction() if results.predictions else ("No Pattern", 0.0)
            top_pattern_name, top_confidence = top_prediction
            
            # Handle "No Pattern" case
            if not patterns_with_info and top_pattern_name != "No Pattern":
                patterns_with_info.append({
                    "name": top_pattern_name,
                    "confidence": top_confidence,
                    "description": model.get_pattern_description(top_pattern_name),
                    "category": model.get_pattern_category(top_pattern_name).value if model.get_pattern_category(top_pattern_name) else None
                })
            
            # Calculate total processing time
            total_time = time.time() - start_time
            
            return {
                "success": True,
                "patterns": patterns_with_info,
                "top_pattern": patterns_with_info[0]["name"] if patterns_with_info else None,
                "top_confidence": patterns_with_info[0]["confidence"] if patterns_with_info else 0,
                "model_id": results.model_id,
                "model_version": results.model_version,
                "inference_time": results.inference_time,
                "total_time": total_time,
                "threshold_used": self.confidence_threshold,
                "raw_predictions": list(zip(results.predictions, results.confidence_scores)),
                "detected_patterns": patterns_with_info,
                "confidence_scores": results.confidence_scores,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Unhandled exception in detect_patterns: {str(e)}", exc_info=True)
            total_time = time.time() - start_time
            
            return {
                "success": False,
                "error": str(e),
                "patterns": [],
                "detected_patterns": [],
                "confidence_scores": [],
                "model_id": getattr(model, 'model_id', 'unknown') if 'model' in locals() else 'unknown',
                "model_version": str(getattr(model, 'version', 'unknown')) if 'model' in locals() else 'unknown',
                "inference_time": 0,
                "total_time": total_time
            }
    
    @async_cached(key_prefix="candlestick_pattern", ttl=MODEL_CACHE_TTL)
    async def async_detect_patterns(self, market_data: Union[Dict[str, Any], CandlestickData]) -> Dict[str, Any]:
        """
        Async version of detect_patterns.
        
        Args:
            market_data: Market data as dictionary or CandlestickData object
            
        Returns:
            Dictionary with detection results
        """
        import asyncio
        return await asyncio.to_thread(self.detect_patterns, market_data)
    
    def swap_model(self, model_type: str = "rule_based") -> Dict[str, Any]:
        """
        Hot-swap the active model.
        
        Args:
            model_type: Type of model to swap to ('rule_based' or 'cnn')
            
        Returns:
            Dictionary with swap result
        """
        start_time = time.time()
        
        with self._model_lock:
            try:
                # Check for existing model of the requested type
                model_ids = []
                for model_info in registry.list_models(self.model_type):
                    if model_type in model_info["model_id"]:
                        model_ids.append(model_info["model_id"])
                
                # If model exists, activate it
                if model_ids:
                    registry.set_active_version(self.model_type, model_ids[0])
                    self.logger.info(f"Activated existing model: {model_ids[0]}")
                    
                    return {
                        "success": True,
                        "model_id": model_ids[0],
                        "message": f"Activated existing {model_type} model",
                        "execution_time": time.time() - start_time
                    }
                
                # Create new model
                model = ModelFactory.create_model(model_type, confidence_threshold=self.confidence_threshold)
                registry.register_model(model)
                registry.set_active_version(self.model_type, model.model_id)
                
                # Update creation time
                self._model_creation_time = time.time()
                
                self.logger.info(f"Created and activated new {model_type} model: {model.model_id}")
                return {
                    "success": True,
                    "model_id": model.model_id,
                    "message": f"Created and activated new {model_type} model",
                    "execution_time": time.time() - start_time
                }
                
            except Exception as e:
                self.logger.error(f"Error during model swap: {str(e)}", exc_info=True)
                return {
                    "success": False,
                    "error": str(e),
                    "message": f"Failed to swap to {model_type} model",
                    "execution_time": time.time() - start_time
                }
    
    def clear_caches(self):
        """Clear all caches in the service and models."""
        # Clear model factory cache
        ModelFactory.clear_cache()
        
        # Clear caches in active model
        model = registry.get_active_model(self.model_type)
        if model:
            model.clear_cache()
            
            # Clear detector cache if it's a rule-based model
            if isinstance(model, RuleBasedModel) and hasattr(model, 'detector'):
                model.detector.clear_cache()
                
        self.logger.info("All caches cleared")


# Create singleton service instance
pattern_service = PatternService()

# Export classes and functions
__all__ = [
    "PatternTaxonomy",
    "CandlestickAnalyzer",
    "PatternDetector",
    "CandlestickModel",
    "RuleBasedModel",
    "CNNModel",
    "ModelFactory",
    "PatternService",
    "pattern_service"
]