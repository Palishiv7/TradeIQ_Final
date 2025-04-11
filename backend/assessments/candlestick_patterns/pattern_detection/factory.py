"""
Pattern Detector Factory

This module provides factory functions for creating and configuring pattern detectors.
"""

import os
import logging
from functools import lru_cache
from typing import Optional, Dict, Any, List, Union, Tuple

from backend.common.logger import get_logger
from backend.assessments.candlestick_patterns.pattern_detection.interface import (
    PatternDetector, DetectionStrategy
)

# Configure logger
logger = get_logger(__name__)

# Default model paths relative to project root
DEFAULT_CNN_MODEL_PATH = "models/candlestick/cnn_pattern_detector.h5"
DEFAULT_RESNET_MODEL_PATH = "models/candlestick/resnet_pattern_detector.h5"
DEFAULT_EFFICIENTNET_MODEL_PATH = "models/candlestick/efficientnet_pattern_detector.h5"

# List of default pattern classes to detect
DEFAULT_PATTERN_CLASSES = [
    "doji", "hammer", "inverted_hammer", "bullish_engulfing", "bearish_engulfing",
    "morning_star", "evening_star", "bullish_harami", "bearish_harami", 
    "shooting_star", "hanging_man", "three_white_soldiers", "three_black_crows",
    "piercing_line", "dark_cloud_cover", "tweezer_top", "tweezer_bottom"
]

@lru_cache(maxsize=1)
async def get_default_detector(config: Optional[Dict[str, Any]] = None) -> PatternDetector:
    """
    Create the default recommended detector - a composite detector that combines
    rule-based and model-based approaches for maximum accuracy and reliability.
    
    This implementation prefers rule-based detection but augments with ML models
    when available for improved accuracy. The function is cached to avoid
    recreating detectors unnecessarily.
    
    Args:
        config: Optional configuration dictionary
    
    Returns:
        Configured pattern detector instance
    """
    config = config or {}
    logger.info("Creating default composite pattern detector")
    
    # Initialize detectors list
    detectors = []
    
    # Always add rule-based detector as primary
    from backend.assessments.candlestick_patterns.pattern_detection.rule_based import RuleBasedDetector
    rule_detector = RuleBasedDetector(
        name="DefaultRuleDetector",
        min_confidence=config.get("min_confidence", 0.6),
        max_patterns=config.get("max_patterns", None)
    )
    await rule_detector.initialize()
    detectors.append(rule_detector)
    
    # Add model-based detectors if enabled and models available
    try:
        from backend.assessments.candlestick_patterns.pattern_detection.model_based import (
            CNNPatternDetector, EfficientNetDetector
        )
        
        # Check if ML models are enabled in config
        if config.get("use_ml_models", True):
            # Try to find model files
            cnn_path = config.get("cnn_model_path", DEFAULT_CNN_MODEL_PATH)
            efficientnet_path = config.get("efficientnet_model_path", DEFAULT_EFFICIENTNET_MODEL_PATH)
            
            # Add CNN detector if model exists
            if os.path.exists(cnn_path):
                logger.info(f"Adding CNN detector with model: {cnn_path}")
                cnn_detector = CNNPatternDetector(
                    model_path=cnn_path,
                    name="DefaultCNNDetector",
                    min_confidence=config.get("min_confidence", 0.5)
                )
                await cnn_detector.initialize()
                detectors.append(cnn_detector)
            else:
                logger.warning(f"CNN model not found at {cnn_path}, skipping")
            
            # Add EfficientNet detector if model exists
            if os.path.exists(efficientnet_path):
                logger.info(f"Adding EfficientNet detector with model: {efficientnet_path}")
                effnet_detector = EfficientNetDetector(
                    model_path=efficientnet_path,
                    name="DefaultEfficientNetDetector",
                    min_confidence=config.get("min_confidence", 0.5)
                )
                await effnet_detector.initialize()
                detectors.append(effnet_detector)
            else:
                logger.warning(f"EfficientNet model not found at {efficientnet_path}, skipping")
    except ImportError:
        logger.warning("ML dependencies not available, using only rule-based detection")
    except Exception as e:
        logger.exception(f"Error setting up ML detectors: {e}")
    
    # If we have multiple detectors, use an ensemble
    if len(detectors) > 1:
        try:
            from backend.assessments.candlestick_patterns.pattern_detection.ensemble import WeightedConsensusDetector
            
            # Define weights based on detector type
            weights = config.get("detector_weights", {
                "DefaultRuleDetector": 0.6,
                "DefaultCNNDetector": 0.2,
                "DefaultEfficientNetDetector": 0.2
            })
            
            logger.info(f"Creating weighted consensus detector with {len(detectors)} sub-detectors")
            consensus_detector = WeightedConsensusDetector(
        detectors=detectors,
                detector_weights=weights,
        name="DefaultConsensusDetector",
                min_confidence=config.get("min_confidence", 0.6),
                max_patterns=config.get("max_patterns", 5)
            )
            await consensus_detector.initialize()
            return consensus_detector
        except Exception as e:
            logger.exception(f"Error creating consensus detector: {e}")
            # Fall back to the rule-based detector if consensus fails
            return detectors[0]
    else:
        # If only one detector was created, return it
        return detectors[0]

async def create_detector(
    strategy: Union[str, DetectionStrategy],
    config: Optional[Dict[str, Any]] = None
) -> PatternDetector:
    """
    Create a pattern detector based on the specified detection strategy.
    
    Provides a unified factory method to instantiate different types of
    pattern detectors with appropriate configuration.
    
    Args:
        strategy: Detection strategy to use
        config: Optional configuration for the detector
        
    Returns:
        Configured pattern detector instance
        
    Raises:
        ValueError: If an invalid strategy is specified
        ImportError: If dependencies for the selected strategy are not available
    """
    config = config or {}
    
    # Convert string strategy to enum if needed
    if isinstance(strategy, str):
        try:
            strategy = DetectionStrategy(strategy.upper())
        except ValueError:
            available_strategies = [s.value for s in DetectionStrategy]
            raise ValueError(f"Invalid detection strategy: {strategy}. Available strategies: {available_strategies}")
    
    # Determine model path based on strategy or config
    model_path = None
    if strategy in [DetectionStrategy.CNN, DetectionStrategy.RESNET, 
                   DetectionStrategy.EFFICIENTNET, DetectionStrategy.ML_BASED]:
        # Use strategy-specific default or from config
        if strategy == DetectionStrategy.CNN:
            model_path = config.get("model_path", DEFAULT_CNN_MODEL_PATH)
        elif strategy == DetectionStrategy.RESNET:
            model_path = config.get("model_path", DEFAULT_RESNET_MODEL_PATH)
        elif strategy == DetectionStrategy.EFFICIENTNET:
            model_path = config.get("model_path", DEFAULT_EFFICIENTNET_MODEL_PATH)
        else:  # ML_BASED - try in order: EfficientNet, CNN, ResNet
            for path in [
                config.get("model_path"),
                DEFAULT_EFFICIENTNET_MODEL_PATH,
                DEFAULT_CNN_MODEL_PATH,
                DEFAULT_RESNET_MODEL_PATH
            ]:
                if path and os.path.exists(path):
                    model_path = path
                    break
    
    # Create appropriate detector based on strategy
    if strategy == DetectionStrategy.RULE_BASED:
        from backend.assessments.candlestick_patterns.pattern_detection.rule_based import RuleBasedDetector
        detector = RuleBasedDetector(
            name=config.get("name", "RuleBasedDetector"),
            min_confidence=config.get("min_confidence", 0.5),
            max_patterns=config.get("max_patterns", None),
            config=config
        )
    
    elif strategy == DetectionStrategy.GEOMETRIC:
        from backend.assessments.candlestick_patterns.pattern_detection.rule_based import GeometricPatternDetector
        detector = GeometricPatternDetector(
            name=config.get("name", "GeometricPatternDetector"),
            min_confidence=config.get("min_confidence", 0.5),
            max_patterns=config.get("max_patterns", None),
            config=config
        )
    
    elif strategy == DetectionStrategy.STATISTICAL:
        from backend.assessments.candlestick_patterns.pattern_detection.rule_based import StatisticalPatternDetector
        detector = StatisticalPatternDetector(
            name=config.get("name", "StatisticalPatternDetector"),
            min_confidence=config.get("min_confidence", 0.5),
            max_patterns=config.get("max_patterns", None),
            config=config
        )
    
    elif strategy == DetectionStrategy.CNN:
        try:
            from backend.assessments.candlestick_patterns.pattern_detection.model_based import CNNPatternDetector
            detector = CNNPatternDetector(
                model_path=model_path,
                name=config.get("name", "CNNPatternDetector"),
                min_confidence=config.get("min_confidence", 0.5),
                max_patterns=config.get("max_patterns", None),
                config=config
            )
        except ImportError:
            logger.error("CNN dependencies not available, falling back to rule-based detection")
            from backend.assessments.candlestick_patterns.pattern_detection.rule_based import RuleBasedDetector
            detector = RuleBasedDetector(
                name="FallbackRuleDetector",
                min_confidence=config.get("min_confidence", 0.5)
            )
    
    elif strategy == DetectionStrategy.RESNET:
        try:
            from backend.assessments.candlestick_patterns.pattern_detection.model_based import ResNetDetector
            detector = ResNetDetector(
                model_path=config.get("model_path"),
                name=config.get("name", "ResNetDetector"),
                confidence_threshold=config.get("confidence_threshold", 0.7)
            )
        except ImportError:
            logger.error("ResNet dependencies not available, falling back to rule-based detection")
            from backend.assessments.candlestick_patterns.pattern_detection.rule_based import RuleBasedDetector
            detector = RuleBasedDetector(
                name="FallbackRuleDetector",
                min_confidence=config.get("min_confidence", 0.5)
            )
        
    elif strategy == DetectionStrategy.EFFICIENTNET:
        try:
            from backend.assessments.candlestick_patterns.pattern_detection.model_based import EfficientNetDetector
            detector = EfficientNetDetector(
                model_path=model_path,
                name=config.get("name", "EfficientNetDetector"),
                min_confidence=config.get("min_confidence", 0.5),
                max_patterns=config.get("max_patterns", None),
                config=config
            )
        except ImportError:
            logger.error("EfficientNet dependencies not available, falling back to rule-based detection")
            from backend.assessments.candlestick_patterns.pattern_detection.rule_based import RuleBasedDetector
            detector = RuleBasedDetector(
                name="FallbackRuleDetector",
                min_confidence=config.get("min_confidence", 0.5)
            )
    
    elif strategy in [DetectionStrategy.ENSEMBLE, DetectionStrategy.WEIGHTED_CONSENSUS]:
        # Create an ensemble of multiple detector types
        from backend.assessments.candlestick_patterns.pattern_detection.ensemble import WeightedConsensusDetector
        
        # Configure the detectors to include in the ensemble
        detector_configs = config.get("detectors", [
            {"strategy": DetectionStrategy.RULE_BASED, "weight": 0.6},
            {"strategy": DetectionStrategy.CNN, "weight": 0.2},
            {"strategy": DetectionStrategy.EFFICIENTNET, "weight": 0.2}
        ])
        
        # Create each sub-detector and build the ensemble
        sub_detectors = []
        detector_weights = {}
        
        for i, det_config in enumerate(detector_configs):
            # Extract detector-specific configuration
            det_strategy = det_config.get("strategy", DetectionStrategy.RULE_BASED)
            det_name = det_config.get("name", f"Detector_{i}")
            det_weight = det_config.get("weight", 1.0 / len(detector_configs))
            
            try:
                # Recursively create the sub-detector
                sub_detector = await create_detector(det_strategy, {**config, "name": det_name})
                sub_detectors.append(sub_detector)
                detector_weights[sub_detector.name] = det_weight
            except Exception as e:
                logger.warning(f"Failed to create sub-detector {det_name}: {e}")
        
        # If no sub-detectors could be created, fall back to rule-based
        if not sub_detectors:
            logger.error("No sub-detectors could be created for ensemble, falling back to rule-based")
            from backend.assessments.candlestick_patterns.pattern_detection.rule_based import RuleBasedDetector
            detector = RuleBasedDetector(
                name="FallbackRuleDetector",
                min_confidence=config.get("min_confidence", 0.5)
            )
        else:
            # Create the ensemble detector
            detector = WeightedConsensusDetector(
                detectors=sub_detectors,
                detector_weights=detector_weights,
                name=config.get("name", "WeightedConsensusDetector"),
                min_confidence=config.get("min_confidence", 0.6),
                max_patterns=config.get("max_patterns", None)
            )
    
    elif strategy == DetectionStrategy.ML_BASED:
        # Try to use the best available ML model
        try:
            # Try EfficientNet first
            from backend.assessments.candlestick_patterns.pattern_detection.model_based import EfficientNetDetector
            if model_path and os.path.exists(model_path):
                detector = EfficientNetDetector(
                    model_path=model_path,
                    name=config.get("name", "MLPatternDetector"),
                    min_confidence=config.get("min_confidence", 0.5),
                    max_patterns=config.get("max_patterns", None),
                    config=config
                )
            else:
                # Fall back to rule-based if no model available
                logger.warning(f"ML model not found at {model_path}, falling back to rule-based detection")
                from backend.assessments.candlestick_patterns.pattern_detection.rule_based import RuleBasedDetector
                detector = RuleBasedDetector(
                    name="FallbackRuleDetector",
                    min_confidence=config.get("min_confidence", 0.5)
                )
        except ImportError:
            # If ML dependencies not available, use rule-based
            logger.warning("ML dependencies not available, falling back to rule-based detection")
            from backend.assessments.candlestick_patterns.pattern_detection.rule_based import RuleBasedDetector
            detector = RuleBasedDetector(
                name="FallbackRuleDetector",
                min_confidence=config.get("min_confidence", 0.5)
            )
    
    else:
        raise ValueError(f"Unsupported detection strategy: {strategy}")
    
    # Initialize the detector
    await detector.initialize()
    return detector