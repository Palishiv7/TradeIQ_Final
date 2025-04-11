"""
Candlestick Pattern Detection Module

This module provides a flexible and extensible system for detecting candlestick patterns
in financial market data. It supports multiple detection strategies including:

1. Rule-based detection: Using geometric and statistical rules
2. ML-based detection: Using machine learning models (CNN, ResNet, EfficientNet)
3. Ensemble detection: Combining multiple strategies for improved accuracy

The module is designed to be easy to use while providing advanced capabilities for
professional applications. It can detect a wide range of common candlestick patterns
and provides detailed metadata about each detection.
"""

from backend.assessments.candlestick_patterns.pattern_detection.interface import (
    PatternDetector,
    PatternMatch,
    DetectionStrategy
)

from backend.assessments.candlestick_patterns.pattern_detection.rule_based import (
    RuleBasedDetector,
    GeometricPatternDetector,
    StatisticalPatternDetector
)

from backend.assessments.candlestick_patterns.pattern_detection.model_based import (
    ModelBasedDetector,
    CNNPatternDetector,
    ResNetDetector,
    EfficientNetDetector,
    are_ml_models_available
)

from backend.assessments.candlestick_patterns.pattern_detection.ensemble import (
    EnsembleDetector,
    WeightedConsensusDetector
)

from backend.assessments.candlestick_patterns.pattern_detection.factory import (
    get_default_detector,
    create_detector
)

__all__ = [
    # Core interfaces
    'PatternDetector',
    'PatternMatch',
    'DetectionStrategy',
    
    # Rule-based detectors
    'RuleBasedDetector',
    'GeometricPatternDetector',
    'StatisticalPatternDetector',
    
    # ML-based detectors
    'ModelBasedDetector',
    'CNNPatternDetector',
    'ResNetDetector',
    'EfficientNetDetector',
    'are_ml_models_available',
    
    # Ensemble detectors
    'EnsembleDetector',
    'WeightedConsensusDetector',
    
    # Factory functions
    'get_default_detector',
    'create_detector'
] 