"""
Model-Based Pattern Detectors

This module provides pattern detectors that use machine learning models
for identifying candlestick patterns.
"""

import os
import time
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass

from backend.common.logger import get_logger
from backend.assessments.candlestick_patterns.candlestick_utils import Candle, CandlestickData
from backend.assessments.candlestick_patterns.pattern_detection.interface import (
    PatternDetector, PatternMatch, DetectionStrategy
)

# Configure logger
logger = get_logger(__name__)

# Try importing ML/DL dependencies
try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False
    logger.warning("ONNX Runtime not available. Model-based detection will be limited.")

try:
    from PIL import Image
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    HAS_IMAGING = True
except ImportError:
    HAS_IMAGING = False
    logger.warning("Imaging libraries not available. Chart rendering will be disabled.")


class ModelBasedDetector(PatternDetector):
    """
    Base class for pattern detectors using machine learning models.
    
    This class provides common functionality for all model-based detectors,
    including model loading, inference, and image preprocessing.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        model_name: str = "DefaultModel",
        confidence_threshold: float = 0.65,
        use_onnx: bool = True,
        name: Optional[str] = None
    ):
        """
        Initialize the model-based detector.
        
        Args:
            model_path: Path to the model file
            model_name: Name/identifier for the model architecture
            confidence_threshold: Minimum confidence for pattern detection
            use_onnx: Whether to use ONNX Runtime for inference
            name: Optional custom name for this detector
        """
        super().__init__(name or f"{model_name}Detector")
        self.model_path = model_path
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.use_onnx = use_onnx and HAS_ONNX
        
        # Model and session will be initialized lazily
        self.model = None
        self.session = None
        self.metadata = {}
        
        # Load model if path is provided and exists
        if model_path and os.path.exists(model_path):
            try:
                self._load_model()
            except Exception as e:
                logger.error(f"Failed to load model {model_name} from {model_path}: {e}")
    
    def _load_model(self) -> None:
        """Load the machine learning model for inference."""
        # Check if model path exists
        if not self.model_path or not os.path.exists(self.model_path):
            logger.error(f"Model path not found: {self.model_path}")
            return
        
        try:
            if self.use_onnx:
                # Load ONNX model
                logger.info(f"Loading ONNX model from {self.model_path}")
                
                # Create ONNX Runtime session with optimizations
                sess_options = ort.SessionOptions()
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                sess_options.intra_op_num_threads = 4  # Use 4 threads for operations
                
                # Create inference session
                self.session = ort.InferenceSession(
                    self.model_path, 
                    sess_options=sess_options,
                    providers=['CPUExecutionProvider']  # Use 'CUDAExecutionProvider' for GPU
                )
                
                # Get model metadata
                self.metadata = {
                    "input_name": self.session.get_inputs()[0].name,
                    "input_shape": self.session.get_inputs()[0].shape,
                    "output_names": [output.name for output in self.session.get_outputs()],
                }
                
                logger.info(f"Successfully loaded ONNX model: {self.model_name}")
                logger.debug(f"Model metadata: {self.metadata}")
                
                self.model = True  # Flag that model is loaded
            else:
                # Handle other model types if needed
                logger.warning("Non-ONNX models are not currently supported")
                self.model = None
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.model = None
            self.session = None
    
    def _preprocess_data(self, data: CandlestickData) -> Optional[np.ndarray]:
        """
        Preprocess candlestick data for model input.
        
        Args:
            data: Candlestick data to process
            
        Returns:
            Preprocessed numpy array for model input or None if preprocessing fails
        """
        if not HAS_IMAGING:
            logger.error("Imaging libraries not available for preprocessing")
            return None
            
        try:
            # Generate candlestick chart image
            from backend.assessments.candlestick_patterns.candlestick_utils import plot_candlestick_chart
            
            # Use the utility function to plot the chart
            chart_img = plot_candlestick_chart(data.candles, width=224, height=224, as_array=True)
            
            if chart_img is None:
                logger.error("Failed to generate chart image")
                return None
            
            # Normalize the image to [0, 1] range
            chart_img = chart_img.astype(np.float32) / 255.0
            
            # Ensure correct shape with batch dimension
            if len(chart_img.shape) == 3:  # HWC format
                # For models expecting NCHW format
                chart_img = np.transpose(chart_img, (2, 0, 1))  # Convert to CHW
                chart_img = np.expand_dims(chart_img, axis=0)  # Add batch dimension
            else:
                logger.error(f"Unexpected image shape: {chart_img.shape}")
                return None
            
            return chart_img
        except Exception as e:
            logger.error(f"Error preprocessing data: {e}")
            return None
    
    def _run_inference(self, preprocessed_data: np.ndarray) -> Optional[Dict[str, np.ndarray]]:
        """
        Run model inference.
        
        Args:
            preprocessed_data: Preprocessed input data
            
        Returns:
            Dictionary of model outputs or None if inference fails
        """
        if self.session is None:
            logger.error("No model session available for inference")
            return None
            
        try:
            # Prepare input
            input_name = self.metadata.get("input_name", "input")
            inputs = {input_name: preprocessed_data}
            
            # Run inference
            outputs = self.session.run(None, inputs)
            
            # Create output dictionary
            output_names = self.metadata.get("output_names", ["output"])
            return {name: output for name, output in zip(output_names, outputs)}
        except Exception as e:
            logger.error(f"Error running inference: {e}")
            return None
    
    def _postprocess_results(
        self,
        inference_outputs: Dict[str, np.ndarray],
        data: CandlestickData
    ) -> List[PatternMatch]:
        """
        Process model predictions to create pattern matches.
        
        Args:
            inference_outputs: Model inference outputs
            data: Original candlestick data
            
        Returns:
            List of detected patterns
        """
        # This is a placeholder - subclasses should implement this
        return []
    
    def detect_patterns(self, data: CandlestickData) -> List[PatternMatch]:
        """
        Detect patterns using the machine learning model.
        
        Args:
            data: Candlestick data to analyze
            
        Returns:
            List of detected patterns
        """
        start_time = time.time()
        
        # Check if model is loaded
        if not self.model:
            if self.model_path:
                logger.warning(f"Model {self.model_name} not loaded. Attempting to load...")
                self._load_model()
                
            if not self.model:
                logger.error(f"No model available for {self.name}")
                return []
        
        try:
            # Preprocess data
            preprocessed_data = self._preprocess_data(data)
            if preprocessed_data is None:
                return []
            
            # Run inference
            inference_outputs = self._run_inference(preprocessed_data)
            if inference_outputs is None:
                return []
            
            # Postprocess results
            patterns = self._postprocess_results(inference_outputs, data)
            
            # Add detection strategy and time information
            detection_time_ms = (time.time() - start_time) * 1000
            for pattern in patterns:
                pattern.detection_strategy = self.get_strategy_type()
                pattern.detection_time_ms = detection_time_ms
            
            return patterns
        except Exception as e:
            logger.error(f"Error in model-based pattern detection: {e}")
            return []
    
    def get_strategy_type(self) -> DetectionStrategy:
        """Get the detection strategy type."""
        return DetectionStrategy.ML_BASED


class CNNPatternDetector(ModelBasedDetector):
    """
    Pattern detector using a CNN model.
    
    This detector uses a convolutional neural network to identify
    patterns in candlestick chart images.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.65,
        class_labels: Optional[List[str]] = None,
        name: Optional[str] = None
    ):
        """
        Initialize the CNN pattern detector.
        
        Args:
            model_path: Path to the model file
            confidence_threshold: Minimum confidence for pattern detection
            class_labels: List of class labels for the model outputs
            name: Optional custom name for this detector
        """
        super().__init__(
            model_path=model_path,
            model_name="CNN",
            confidence_threshold=confidence_threshold,
            name=name or "CNNPatternDetector"
        )
        
        # Initialize with default class labels if not provided
        self.class_labels = class_labels or [
            "No Pattern", "Doji", "Hammer", "Shooting Star", 
            "Bullish Engulfing", "Bearish Engulfing", 
            "Morning Star", "Evening Star"
        ]
        
        # Mapping of which patterns are bullish
        self.bullish_patterns = {
            "Doji": None,  # Can be either
            "Hammer": True,
            "Shooting Star": False,
            "Bullish Engulfing": True,
            "Bearish Engulfing": False,
            "Morning Star": True,
            "Evening Star": False,
            "Bullish Harami": True,
            "Bearish Harami": False
        }
    
    def _postprocess_results(
        self,
        inference_outputs: Dict[str, np.ndarray],
        data: CandlestickData
    ) -> List[PatternMatch]:
        """
        Process CNN model predictions to create pattern matches.
        
        Args:
            inference_outputs: Model inference outputs
            data: Original candlestick data
            
        Returns:
            List of detected patterns
        """
        patterns = []
        
        # Get the main output (usually class probabilities)
        output_name = list(inference_outputs.keys())[0]
        probs = inference_outputs[output_name]
        
        # Reshape if needed
        if len(probs.shape) > 2:
            probs = probs.reshape(probs.shape[0], -1)
        
        # For each prediction in the batch
        for i, batch_probs in enumerate(probs):
            # Get top predictions
            for class_idx, confidence in enumerate(batch_probs):
                # Skip if below threshold or "No Pattern" class
                if confidence < self.confidence_threshold or class_idx == 0:
                    continue
                
                # Get pattern name
                if class_idx < len(self.class_labels):
                    pattern_name = self.class_labels[class_idx]
                else:
                    pattern_name = f"Pattern_{class_idx}"
                
                # Get bullish/bearish direction
                bullish = self.bullish_patterns.get(pattern_name)
                
                # Estimate which candles form the pattern
                candle_indices = self._estimate_pattern_indices(pattern_name, data.candles)
                
                # Create pattern match
                pattern = PatternMatch(
                    pattern_name=pattern_name,
                    confidence=float(confidence),
                    candle_indices=candle_indices,
                    bullish=bullish,
                    description=f"ML-detected {pattern_name} pattern",
                    detection_strategy=self.get_strategy_type(),
                    metadata={
                        "raw_score": float(confidence),
                        "detected_by_model": self.model_name
                    }
                )
                
                patterns.append(pattern)
        
        return patterns
    
    def _estimate_pattern_indices(self, pattern_name: str, candles: List[Candle]) -> List[int]:
        """
        Estimate which candles are part of the detected pattern.
        
        Args:
            pattern_name: Name of the detected pattern
            candles: List of candlesticks
            
        Returns:
            List of candle indices that likely form the pattern
        """
        # Focus on the most recent candles
        if not candles:
            return []
            
        # Number of candles to include based on pattern type
        if pattern_name in ["Morning Star", "Evening Star", "Three White Soldiers", "Three Black Crows"]:
            # Triple candlestick patterns
            num_candles = 3
        elif pattern_name in ["Bullish Engulfing", "Bearish Engulfing", "Bullish Harami", "Bearish Harami"]:
            # Double candlestick patterns
            num_candles = 2
        else:
            # Single candlestick patterns
            num_candles = 1
        
        # Get the most recent `num_candles` indices
        last_idx = len(candles) - 1
        return list(range(last_idx - num_candles + 1, last_idx + 1))
    
    def get_strategy_type(self) -> DetectionStrategy:
        """Get the detection strategy type."""
        return DetectionStrategy.CNN


class ResNetDetector(CNNPatternDetector):
    """
    Pattern detector using a ResNet model.
    
    This detector uses a ResNet architecture for identifying
    more complex patterns in candlestick chart images.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.7,
        class_labels: Optional[List[str]] = None,
        name: Optional[str] = None
    ):
        """
        Initialize the ResNet pattern detector.
        
        Args:
            model_path: Path to the model file
            confidence_threshold: Minimum confidence for pattern detection
            class_labels: List of class labels for the model outputs
            name: Optional custom name for this detector
        """
        super().__init__(
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            class_labels=class_labels,
            name=name or "ResNetDetector"
        )
        self.model_name = "ResNet"
    
    def get_strategy_type(self) -> DetectionStrategy:
        """Get the detection strategy type."""
        return DetectionStrategy.RESNET


class EfficientNetDetector(CNNPatternDetector):
    """
    Pattern detector using an EfficientNet model.
    
    This detector uses the EfficientNet architecture for efficient and
    accurate pattern detection with reduced computational requirements.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.7,
        class_labels: Optional[List[str]] = None,
        name: Optional[str] = None
    ):
        """
        Initialize the EfficientNet pattern detector.
        
        Args:
            model_path: Path to the model file
            confidence_threshold: Minimum confidence for pattern detection
            class_labels: List of class labels for the model outputs
            name: Optional custom name for this detector
        """
        super().__init__(
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            class_labels=class_labels,
            name=name or "EfficientNetDetector"
        )
        self.model_name = "EfficientNet"
    
    def get_strategy_type(self) -> DetectionStrategy:
        """Get the detection strategy type."""
        return DetectionStrategy.EFFICIENTNET


class ConvNeXtDetector(CNNPatternDetector):
    """
    Pattern detector using a ConvNeXt model.
    
    This detector uses the ConvNeXt architecture, which combines the strengths of
    CNNs and Vision Transformers for state-of-the-art pattern recognition with
    lower computational requirements than traditional Transformer models.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.7,
        class_labels: Optional[List[str]] = None,
        use_advanced_postprocessing: bool = True,
        use_context_window: bool = True,
        confidence_calibration: bool = True,
        name: Optional[str] = None
    ):
        """
        Initialize the ConvNeXt pattern detector.
        
        Args:
            model_path: Path to the model file
            confidence_threshold: Minimum confidence for pattern detection
            class_labels: List of class labels for the model outputs
            use_advanced_postprocessing: Whether to use advanced confidence scoring
            use_context_window: Whether to use surrounding candles for context
            confidence_calibration: Whether to apply calibration to confidence scores
            name: Optional custom name for this detector
        """
        # Expanded class labels for ConvNeXt model
        default_labels = [
            "No Pattern", "Doji", "Hammer", "Shooting Star", 
            "Bullish Engulfing", "Bearish Engulfing", 
            "Morning Star", "Evening Star", "Bullish Harami", "Bearish Harami",
            "Three White Soldiers", "Three Black Crows", "Tweezer Top", "Tweezer Bottom",
            "Abandoned Baby Top", "Abandoned Baby Bottom", "Three Inside Up", "Three Inside Down"
        ]
        
        super().__init__(
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            class_labels=class_labels or default_labels,
            name=name or "ConvNeXtDetector"
        )
        self.model_name = "ConvNeXt"
        self.use_advanced_postprocessing = use_advanced_postprocessing
        self.use_context_window = use_context_window
        self.confidence_calibration = confidence_calibration
        
        # Calibration parameters for better confidence scoring
        self.calibration_params = {
            "scale": 1.2,  # Scaling factor for calibration
            "shift": 0.1,  # Shift factor for calibration
            "pattern_priors": {  # Prior probabilities based on pattern frequency
                "Doji": 0.15,
                "Hammer": 0.08,
                "Shooting Star": 0.07,
                "Bullish Engulfing": 0.12,
                "Bearish Engulfing": 0.12,
                "Morning Star": 0.04,
                "Evening Star": 0.04,
                "Bullish Harami": 0.06,
                "Bearish Harami": 0.06,
                "Three White Soldiers": 0.03,
                "Three Black Crows": 0.03,
                "Tweezer Top": 0.05,
                "Tweezer Bottom": 0.05,
                "Abandoned Baby Top": 0.02,
                "Abandoned Baby Bottom": 0.02,
                "Three Inside Up": 0.03,
                "Three Inside Down": 0.03
            }
        }
        
        # Update bullish patterns dictionary with additional patterns
        self.bullish_patterns.update({
            "Three White Soldiers": True,
            "Three Black Crows": False,
            "Tweezer Top": False,
            "Tweezer Bottom": True,
            "Abandoned Baby Top": False,
            "Abandoned Baby Bottom": True,
            "Three Inside Up": True,
            "Three Inside Down": False,
        })
    
    def _preprocess_data(self, data: CandlestickData) -> Optional[np.ndarray]:
        """
        Enhanced preprocessing for ConvNeXt model.
        
        This method applies specialized preprocessing for ConvNeXt, including:
        1. Dynamic channel expansion for volume and volatility
        2. Context window for surrounding price action
        3. Scale normalization optimized for the model
        
        Args:
            data: Candlestick data to process
            
        Returns:
            Preprocessed tensor for model input or None if preprocessing fails
        """
        if not HAS_IMAGING:
            logger.error("Imaging libraries not available for preprocessing")
            return None
            
        try:
            # Get candles for processing
            candles = data.candles
            if len(candles) < 5:
                logger.warning(f"Not enough candles for preprocessing: {len(candles)}")
                return None
            
            # Generate chart image with enhanced rendering
            from backend.assessments.candlestick_patterns.candlestick_utils import plot_candlestick_chart
            
            # Use context window if enabled
            window_size = 30 if self.use_context_window else 15
            start_idx = max(0, len(candles) - window_size)
            window_candles = candles[start_idx:]
            
            # Create enhanced chart with volume and trend indicators
            chart_img = plot_candlestick_chart(
                window_candles, 
                width=224, 
                height=224, 
                show_volume=True,
                highlight_patterns=True,
                as_array=True
            )
            
            if chart_img is None:
                logger.error("Failed to generate chart image")
                return None
            
            # Normalize the image for ConvNeXt
            # ConvNeXt expects images normalized with mean [0.485, 0.456, 0.406] and std [0.229, 0.224, 0.225]
            img_array = np.array(chart_img).astype(np.float32) / 255.0
            
            # Apply ImageNet normalization
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_array = (img_array - mean) / std
            
            # Add batch dimension and convert to proper format
            img_tensor = np.expand_dims(img_array.transpose(2, 0, 1), axis=0)
            
            return img_tensor
            
        except Exception as e:
            logger.error(f"Error in preprocessing: {e}")
            return None
    
    def _postprocess_results(self, results: np.ndarray, data: CandlestickData) -> List[PatternMatch]:
        """
        Enhanced postprocessing for ConvNeXt model outputs.
        
        Args:
            results: Raw model output
            data: Original candlestick data
            
        Returns:
            List of detected patterns with refined confidence scores
        """
        if not self.use_advanced_postprocessing:
            # Fall back to base implementation if advanced processing is disabled
            return super()._postprocess_results(results, data)
        
        try:
            patterns = []
            candles = data.candles
            
            # Apply softmax to convert logits to probabilities if needed
            if results.max() > 1.0 or results.min() < 0.0:
                # These are logits, apply softmax
                exp_results = np.exp(results - np.max(results, axis=1, keepdims=True))
                probabilities = exp_results / np.sum(exp_results, axis=1, keepdims=True)
            else:
                # Already normalized
                probabilities = results
            
            # Get the top predicted patterns
            for batch_idx in range(probabilities.shape[0]):
                batch_probs = probabilities[batch_idx]
                
                # Get top 3 predictions
                top_indices = np.argsort(batch_probs)[::-1][:3]
                
                for idx in top_indices:
                    confidence = float(batch_probs[idx])
                    pattern_name = self.class_labels[idx] if idx < len(self.class_labels) else "Unknown"
                    
                    # Skip "No Pattern" class and low confidence predictions
                    if pattern_name == "No Pattern" or confidence < self.confidence_threshold:
                        continue
                    
                    # Apply confidence calibration if enabled
                    if self.confidence_calibration:
                        # Get prior probability for this pattern
                        prior = self.calibration_params["pattern_priors"].get(pattern_name, 0.05)
                        
                        # Apply calibration formula: confidence = (confidence * scale + shift) * prior
                        confidence = min(1.0, max(0.0, 
                            (confidence * self.calibration_params["scale"] + self.calibration_params["shift"]) * 
                            (1.0 + prior)
                        ))
                    
                    # Determine candle indices for this pattern
                    candle_indices = self._get_pattern_indices(pattern_name, candles)
                    
                    # Determine if pattern is bullish or bearish
                    bullish = self.bullish_patterns.get(pattern_name)
                    
                    # Create pattern match
                    pattern = PatternMatch(
                        pattern_name=pattern_name,
                        confidence=confidence,
                        candle_indices=candle_indices,
                        bullish=bullish,
                        description=f"ConvNeXt detected {pattern_name} pattern",
                        detection_strategy=DetectionStrategy.CNN,
                        metadata={
                            "raw_confidence": float(batch_probs[idx]),
                            "model_name": "ConvNeXt",
                            "top_alternatives": [
                                {"pattern": self.class_labels[alt_idx], 
                                 "confidence": float(batch_probs[alt_idx])}
                                for alt_idx in top_indices[1:3]  # Get next 2 alternatives
                            ]
                        }
                    )
                    
                    patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error in postprocessing: {e}")
            return []
    
    def _get_pattern_indices(self, pattern_name: str, candles: List[Candle]) -> List[int]:
        """
        Determine the indices of candles involved in the pattern.
        
        Args:
            pattern_name: Name of the detected pattern
            candles: List of candles
            
        Returns:
            Indices of candles involved in the pattern
        """
        # Get the number of candles in the pattern
        num_candles = 1  # Default for single-candle patterns
        
        if pattern_name in ["Bullish Engulfing", "Bearish Engulfing", 
                           "Bullish Harami", "Bearish Harami",
                           "Tweezer Top", "Tweezer Bottom"]:
            num_candles = 2
        elif pattern_name in ["Morning Star", "Evening Star", 
                             "Three White Soldiers", "Three Black Crows",
                             "Abandoned Baby Top", "Abandoned Baby Bottom",
                             "Three Inside Up", "Three Inside Down"]:
            num_candles = 3
        
        # Return the last N candles where N is the number of candles in the pattern
        end_idx = len(candles) - 1
        start_idx = max(0, end_idx - num_candles + 1)
        return list(range(start_idx, end_idx + 1))
        
    def get_strategy_type(self) -> DetectionStrategy:
        """Get the detection strategy type."""
        return DetectionStrategy.CNN


def are_ml_models_available() -> bool:
    """
    Check if machine learning dependencies are available.
    
    Returns:
        True if ONNX and imaging libraries are available
    """
    return HAS_ONNX and HAS_IMAGING


def get_available_model_paths() -> Dict[str, str]:
    """
    Get paths to available model files.
    
    This function searches for ONNX model files in the standard model directories.
    
    Returns:
        Dictionary of available model paths by type
    """
    # Default paths to check
    model_dirs = [
        os.getenv("MODEL_DIR", "./models"),
        os.path.join(os.path.dirname(__file__), "../../../models"),
        os.path.join(os.path.dirname(__file__), "../../../../models"),
        "/app/models"
    ]
    
    available_models = {}
    
    # Search for model files
    for model_dir in model_dirs:
        if not os.path.exists(model_dir):
            continue
            
        # Check for ResNet models
        resnet_patterns = ["resnet", "resnet50", "resnet_pattern"]
        for pattern in resnet_patterns:
            for filename in os.listdir(model_dir):
                if pattern in filename.lower() and filename.endswith(".onnx"):
                    available_models["resnet"] = os.path.join(model_dir, filename)
                    break
            if "resnet" in available_models:
                break
                
        # Check for EfficientNet models
        efficientnet_patterns = ["efficientnet", "efficient_net", "effnet"]
        for pattern in efficientnet_patterns:
            for filename in os.listdir(model_dir):
                if pattern in filename.lower() and filename.endswith(".onnx"):
                    available_models["efficientnet"] = os.path.join(model_dir, filename)
                    break
            if "efficientnet" in available_models:
                break
                
        # Check for ConvNeXt models
        convnext_patterns = ["convnext", "conv_next", "convnxt"]
        for pattern in convnext_patterns:
            for filename in os.listdir(model_dir):
                if pattern in filename.lower() and filename.endswith(".onnx"):
                    available_models["convnext"] = os.path.join(model_dir, filename)
                    break
            if "convnext" in available_models:
                break
    
    return available_models 