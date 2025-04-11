"""
AI-Based Answer Evaluation for Candlestick Pattern Assessment

This module implements:
1. Multi-tier validation pipeline for candlestick pattern detection
2. Smart validation that goes beyond simple string matching
3. Confidence-based scoring system
4. Detailed feedback generation based on validation results

The validation pipeline includes primary pattern detection using rule-based algorithms,
secondary validation with confidence thresholds, and edge case detection for ambiguous patterns.
"""

import time
import asyncio
import json
import threading
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
import difflib
import os

# Import from base assessment architecture
from backend.assessments.base.models import AnswerEvaluation as BaseAnswerEvaluation
from backend.assessments.base.services import AnswerEvaluator, ExplanationGenerator as BaseExplanationGenerator

from backend.assessments.candlestick_patterns.candlestick_utils import CandlestickData
from backend.assessments.candlestick_patterns.pattern_detection import (
    PatternMatch, DetectionStrategy, get_default_detector
)
from backend.common.logger import get_logger

# Import from types module
from backend.assessments.candlestick_patterns.types import (
    CandlestickQuestion,
    ValidationResult as TypesValidationResult,
    UserLevel
)

logger = get_logger(__name__)

class ValidationTier(Enum):
    """Enum representing different validation tiers in the pipeline."""
    PRIMARY = "primary"
    SECONDARY = "secondary"
    EDGE_CASE = "edge_case"
    AI_BASED = "ai_based"  # New validation tier for AI-based evaluation

@dataclass
class ValidationResult:
    """Class to represent the result of pattern validation."""
    pattern_name: str
    user_answer: str
    confidence: float
    is_valid: bool
    validation_tier: ValidationTier
    alternative_patterns: List[str] = field(default_factory=list)
    execution_time_ms: float = 0
    error: Optional[str] = None
    explanation: Optional[str] = None  # Added field for generated explanation
    context_features: Dict[str, Any] = field(default_factory=dict)  # Added field for context features
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "pattern_name": self.pattern_name,
            "user_answer": self.user_answer,
            "confidence": self.confidence,
            "is_valid": self.is_valid,
            "validation_tier": self.validation_tier.value,
            "alternative_patterns": self.alternative_patterns,
            "execution_time_ms": self.execution_time_ms,
            "error": self.error,
            "explanation": self.explanation,
            "context_features": self.context_features
        }

class MultiTierValidationPipeline:
    """
    Multi-tier validation pipeline for candlestick pattern assessment.
    
    This class implements a validation pipeline with four tiers:
    1. Primary validation: Pattern detection using rule-based algorithms
    2. Secondary validation: Additional validation with confidence thresholds
    3. Edge case detection: Handle ambiguous patterns
    4. AI-based validation: Advanced evaluation using context and similarity
    
    The pipeline provides fallback mechanisms to ensure reliable validation
    even if some components fail.
    """
    
    def __init__(self, confidence_threshold: float = 0.7):
        """
        Initialize the validation pipeline.
        
        Args:
            confidence_threshold: Minimum confidence for pattern validation
        """
        self.confidence_threshold = max(0.1, min(confidence_threshold, 0.9))  # Ensure reasonable range
        self.edge_case_detector = EdgeCaseDetector()
        
        # Initialize pattern detector as None - will be initialized on demand
        self._pattern_detector = None
        self._detector_initialized = False
        
        # Initialize pattern similarity matrix
        self._initialize_pattern_similarity()
        
        # Setup asyncio event loop for async operations
        import asyncio
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        
        logger.info(f"Initialized MultiTierValidationPipeline with confidence threshold {self.confidence_threshold}")
    
    def __del__(self):
        """Cleanup resources when object is garbage collected."""
        if hasattr(self, '_loop') and self._loop:
            try:
                self._loop.close()
            except Exception as e:
                logger.error(f"Error closing event loop: {e}")
    
    async def _get_pattern_detector(self):
        """Get the pattern detector, initializing it if necessary."""
        if not self._detector_initialized:
            try:
                self._pattern_detector = await get_default_detector()
                self._detector_initialized = True
            except Exception as e:
                logger.error(f"Error initializing pattern detector: {e}")
                return None
        return self._pattern_detector
    
    def _ensure_pattern_detector(self):
        """Ensure the pattern detector is initialized."""
        if not self._detector_initialized:
            try:
                # Run the async initialization in the event loop
                self._pattern_detector = self._loop.run_until_complete(self._get_pattern_detector())
                self._detector_initialized = True
            except Exception as e:
                logger.error(f"Error ensuring pattern detector: {e}")
        return self._pattern_detector
    
    def _initialize_pattern_similarity(self):
        """Initialize pattern similarity matrix for fuzzy matching."""
        # Define pattern groups (patterns that are often confused)
        self.pattern_groups = {
            "doji_group": ["doji", "dragonfly doji", "gravestone doji", "long legged doji"],
            "reversal_top": ["evening star", "bearish engulfing", "shooting star", "dark cloud cover"],
            "reversal_bottom": ["morning star", "bullish engulfing", "hammer", "piercing line"],
            "trend_confirmation": ["three white soldiers", "three black crows"],
            "indecision": ["spinning top", "high wave", "doji"]
        }
        
        # Create a mapping of pattern to related patterns
        self.related_patterns = {}
        for group_name, patterns in self.pattern_groups.items():
            for pattern in patterns:
                self.related_patterns[pattern] = [p for p in patterns if p != pattern]
    
    def validate(self, candlestick_data: CandlestickData, user_answer: str) -> ValidationResult:
        """
        Validate a user's answer against detected patterns in candlestick data.
        
        Args:
            candlestick_data: The candlestick data to analyze
            user_answer: The user's answer (pattern name)
            
        Returns:
            ValidationResult with validation details
        """
        # Normalize user answer (remove case sensitivity, extra spaces)
        normalized_user_answer = user_answer.strip().lower()
        start_time = time.time()
        
        # Validate inputs
        if not candlestick_data or not candlestick_data.candles or len(candlestick_data.candles) == 0:
            return ValidationResult(
                pattern_name="Invalid Data",
                user_answer=user_answer,
                confidence=0.0,
                is_valid=False,
                validation_tier=ValidationTier.PRIMARY,
                error="Invalid or empty candlestick data",
                execution_time_ms=(time.time() - start_time) * 1000
            )
        
        if not normalized_user_answer:
            return ValidationResult(
                pattern_name="Empty Answer",
                user_answer=user_answer,
                confidence=0.0,
                is_valid=False,
                validation_tier=ValidationTier.PRIMARY,
                error="Empty user answer",
                execution_time_ms=(time.time() - start_time) * 1000
            )
        
        try:
            # Try primary validation first
            result = self._run_primary_validation(candlestick_data, normalized_user_answer)
            
            # If primary validation fails or has low confidence, try secondary validation
            if not result.is_valid or result.confidence < self.confidence_threshold:
                try:
                    secondary_result = self._run_secondary_validation(
                        candlestick_data, normalized_user_answer, result
                    )
                    
                    # Use secondary result if it has higher confidence
                    if secondary_result and secondary_result.confidence > result.confidence:
                        result = secondary_result
                except Exception as e:
                    logger.error(f"Error in secondary validation: {e}")
                    # Continue with existing result if secondary validation fails
            
            # Check for edge cases if still not valid or low confidence
            if not result.is_valid or result.confidence < self.confidence_threshold:
                try:
                    edge_result = self._run_edge_case_detection(
                        candlestick_data, normalized_user_answer, result
                    )
                    
                    # Use edge case result if it has higher confidence
                    if edge_result and edge_result.confidence > result.confidence:
                        result = edge_result
                except Exception as e:
                    logger.error(f"Error in edge case detection: {e}")
                    # Continue with existing result if edge case detection fails
            
            # Finally, run AI-based validation as the most sophisticated approach
            if not result.is_valid or result.confidence < 0.9:  # Higher threshold for triggering AI
                try:
                    ai_result = self._run_ai_validation(
                        candlestick_data, normalized_user_answer, result
                    )
                    
                    # Use AI result if it has higher confidence
                    if ai_result and ai_result.confidence > result.confidence:
                        result = ai_result
                except Exception as e:
                    logger.error(f"Error in AI validation: {e}")
                    # Continue with existing result if AI validation fails
            
            # Add context features for explanation generation
            try:
                result.context_features = self._extract_context_features(candlestick_data, result)
            except Exception as e:
                logger.error(f"Error extracting context features: {e}")
                # Ensure context_features is never None
                if result.context_features is None:
                    result.context_features = {}
            
            # Set final execution time
            result.execution_time_ms = (time.time() - start_time) * 1000
            return result
            
        except Exception as e:
            # Fallback to simple string matching in case of errors
            logger.error(f"Error in validation pipeline: {e}")
            
            # Simple string matching as last resort
            detected_patterns = [p.lower() for p in self._get_all_pattern_names()]
            is_valid = normalized_user_answer in detected_patterns
            
            return ValidationResult(
                pattern_name="Unknown" if not is_valid else normalized_user_answer.capitalize(),
                user_answer=user_answer,
                confidence=0.5 if is_valid else 0.0,
                is_valid=is_valid,
                validation_tier=ValidationTier.PRIMARY,
                error=str(e),
                execution_time_ms=(time.time() - start_time) * 1000
            )
    
    def _extract_context_features(
        self, candlestick_data: CandlestickData, result: ValidationResult
    ) -> Dict[str, Any]:
        """
        Extract context features for explanation generation.
        
        Args:
            candlestick_data: The candlestick data
            result: The validation result
            
        Returns:
            Dictionary of context features
        """
        candles = candlestick_data.candles
        features = {}
        
        # Skip if no candles or pattern
        if not candles or not result.pattern_name:
            return features
        
        try:
            # Calculate basic features
            features["candle_count"] = len(candles)
            
            # Calculate trend features
            closes = [c.close for c in candles[-10:] if candles]
            if len(closes) >= 5:
                features["prior_trend"] = "uptrend" if closes[0] < closes[-1] else "downtrend"
                features["trend_strength"] = abs(closes[-1] - closes[0]) / closes[0] if closes[0] > 0 else 0
            
            # Volume features
            volumes = [c.volume for c in candles[-10:] if candles]
            if volumes:
                avg_volume = sum(volumes) / len(volumes)
                features["volume_trend"] = "increasing" if volumes[-1] > avg_volume else "decreasing"
                
            # Pattern-specific features
            if "doji" in result.pattern_name.lower():
                # Doji specifics (body to range ratio)
                if candles and len(candles) > 0:
                    last_candle = candles[-1]
                    body_size = abs(last_candle.close - last_candle.open)
                    range_size = last_candle.high - last_candle.low
                    if range_size > 0:
                        features["body_to_range_ratio"] = body_size / range_size
                    
            # Add verification status based on tier
            features["verification_level"] = result.validation_tier.value
            features["confidence_score"] = result.confidence
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting context features: {e}")
            return {}
    
    def _run_primary_validation(
        self, candlestick_data: CandlestickData, user_answer: str
    ) -> ValidationResult:
        """
        Run primary validation using pattern detection.
        
        Args:
            candlestick_data: The candlestick data to analyze
            user_answer: The user's answer (pattern name)
            
        Returns:
            ValidationResult with primary validation details
        """
        start_time = time.time()
        
        try:
            # Ensure detector is initialized
            detector = self._ensure_pattern_detector()
            if not detector:
                return ValidationResult(
                    pattern_name="Unknown",
                    user_answer=user_answer,
                    confidence=0.0,
                    is_valid=False,
                    validation_tier=ValidationTier.PRIMARY,
                    error="Pattern detector initialization failed",
                    execution_time_ms=(time.time() - start_time) * 1000
                )
            
            # Run async pattern detection in the event loop
            try:
                patterns, error = self._loop.run_until_complete(
                    detector.detect_patterns_safe(candlestick_data)
                )
                
                if error:
                    logger.warning(f"Pattern detection completed with warning: {error}")
            except Exception as e:
                logger.error(f"Error during pattern detection: {e}")
                return ValidationResult(
                    pattern_name="Unknown",
                    user_answer=user_answer,
                    confidence=0.0,
                    is_valid=False,
                    validation_tier=ValidationTier.PRIMARY,
                    error=f"Pattern detection failed: {str(e)}",
                    execution_time_ms=(time.time() - start_time) * 1000
                )
            
            if not patterns:
                return ValidationResult(
                    pattern_name="No Patterns",
                    user_answer=user_answer,
                    confidence=0.0,
                    is_valid=False,
                    validation_tier=ValidationTier.PRIMARY,
                    error="No patterns detected in the data",
                    execution_time_ms=(time.time() - start_time) * 1000
                )
                
            # Sort by confidence (highest first)
            patterns = sorted(patterns, key=lambda x: x.confidence, reverse=True)
            
            # Check if the user's answer matches any of the detected patterns
            for pattern in patterns:
                pattern_name = pattern.pattern_name.lower()
                
                # Direct match
                if pattern_name == user_answer:
                    return ValidationResult(
                        pattern_name=pattern.pattern_name,
                        user_answer=user_answer,
                        confidence=pattern.confidence,
                        is_valid=True,
                        validation_tier=ValidationTier.PRIMARY,
                        alternative_patterns=[p.pattern_name for p in patterns[:3] if p.pattern_name.lower() != user_answer],
                        execution_time_ms=(time.time() - start_time) * 1000,
                        context_features=self._extract_context_features(candlestick_data, None)
                    )
            
            # No direct match found - return top pattern but mark as invalid
            top_pattern = patterns[0]
            return ValidationResult(
                pattern_name=top_pattern.pattern_name,
                user_answer=user_answer,
                confidence=top_pattern.confidence * 0.8,  # Slightly reduce confidence
                is_valid=False,
                validation_tier=ValidationTier.PRIMARY,
                alternative_patterns=[p.pattern_name for p in patterns[:3]],
                execution_time_ms=(time.time() - start_time) * 1000,
                context_features=self._extract_context_features(candlestick_data, None)
            )
                
        except Exception as e:
            logger.error(f"Error in primary validation: {e}", exc_info=True)
            return ValidationResult(
                pattern_name="Error",
                user_answer=user_answer,
                confidence=0.0,
                is_valid=False,
                validation_tier=ValidationTier.PRIMARY,
                error=f"Error in primary validation: {str(e)}",
                execution_time_ms=(time.time() - start_time) * 1000
            )
    
    def _run_secondary_validation(
        self, candlestick_data: CandlestickData, user_answer: str, 
        primary_result: Optional[ValidationResult] = None
    ) -> ValidationResult:
        """
        Run secondary validation with additional pattern similarity checks.
        
        Args:
            candlestick_data: The candlestick data to analyze
            user_answer: The user's answer (pattern name)
            primary_result: Optional result from primary validation
            
        Returns:
            ValidationResult with secondary validation details
        """
        start_time = time.time()
        
        try:
            # Use our pattern detector to get detected patterns
            detector = self._ensure_pattern_detector()
            if not detector:
                return ValidationResult(
                    pattern_name="Unknown",
                    user_answer=user_answer,
                    confidence=0.0,
                    is_valid=False,
                    validation_tier=ValidationTier.SECONDARY,
                    error="Pattern detector initialization failed",
                    execution_time_ms=(time.time() - start_time) * 1000
                )
            
            # Run async pattern detection in the event loop
            try:
                patterns, error = self._loop.run_until_complete(
                    detector.detect_patterns_safe(candlestick_data)
                )
                
                if error:
                    logger.warning(f"Pattern detection completed with warning: {error}")
            except Exception as e:
                logger.error(f"Error during pattern detection in secondary validation: {e}")
                return ValidationResult(
                    pattern_name="Unknown",
                    user_answer=user_answer,
                    confidence=0.0,
                    is_valid=False,
                    validation_tier=ValidationTier.SECONDARY,
                    error=f"Pattern detection failed: {str(e)}",
                    execution_time_ms=(time.time() - start_time) * 1000
                )
            
            # If no patterns detected, return the previous result or invalid result
            if not patterns:
                if primary_result:
                    return primary_result
                    
                return ValidationResult(
                    pattern_name="No Patterns",
                    user_answer=user_answer,
                    confidence=0.0,
                    is_valid=False,
                    validation_tier=ValidationTier.SECONDARY,
                    error="No patterns detected in the data",
                    execution_time_ms=(time.time() - start_time) * 1000
                )
            
            # Sort patterns by confidence
            patterns = sorted(patterns, key=lambda x: x.confidence, reverse=True)
            
            # Check for fuzzy matches between user answer and pattern names
            best_match = None
            best_match_score = 0
            
            # Try to find the best match using fuzzy matching
            for pattern in patterns:
                pattern_name = pattern.pattern_name.lower()
                
                # First check for partial string containment
                if user_answer in pattern_name or pattern_name in user_answer:
                    score = 0.8 * pattern.confidence  # 80% of the pattern confidence
                    if score > best_match_score:
                        best_match = pattern
                        best_match_score = score
                
                # Use difflib for more sophisticated string similarity
                similarity = difflib.SequenceMatcher(None, user_answer, pattern_name).ratio()
                if similarity > 0.7:  # Threshold for considering it a potential match
                    score = similarity * pattern.confidence
                    if score > best_match_score:
                        best_match = pattern
                        best_match_score = score
                
                # Check for related patterns
                if self._are_patterns_similar(user_answer, pattern_name):
                    score = 0.75 * pattern.confidence  # 75% of the pattern confidence
                    if score > best_match_score:
                        best_match = pattern
                        best_match_score = score
            
            # If we found a good match, return it
            if best_match and best_match_score >= self.confidence_threshold * 0.7:
                return ValidationResult(
                    pattern_name=best_match.pattern_name,
                    user_answer=user_answer,
                    confidence=best_match_score,
                    is_valid=best_match_score >= self.confidence_threshold,
                    validation_tier=ValidationTier.SECONDARY,
                    alternative_patterns=[p.pattern_name for p in patterns[:3] if p.pattern_name.lower() != best_match.pattern_name.lower()],
                    execution_time_ms=(time.time() - start_time) * 1000,
                    context_features=self._extract_context_features(candlestick_data, primary_result)
                )
            
            # If no good match, return top pattern with low confidence
            top_pattern = patterns[0]
            return ValidationResult(
                pattern_name=top_pattern.pattern_name,
                user_answer=user_answer,
                confidence=0.5 * top_pattern.confidence,  # 50% confidence due to low match quality
                is_valid=False,
                validation_tier=ValidationTier.SECONDARY,
                alternative_patterns=[p.pattern_name for p in patterns[:3]],
                execution_time_ms=(time.time() - start_time) * 1000,
                context_features=self._extract_context_features(candlestick_data, primary_result)
            )
                
        except Exception as e:
            logger.error(f"Error in secondary validation: {e}", exc_info=True)
            if primary_result:
                return primary_result
                
            return ValidationResult(
                pattern_name="Error",
                user_answer=user_answer,
                confidence=0.0,
                is_valid=False,
                validation_tier=ValidationTier.SECONDARY,
                error=f"Error in secondary validation: {str(e)}",
                execution_time_ms=(time.time() - start_time) * 1000
            )
    
    def _run_edge_case_detection(
        self, candlestick_data: CandlestickData, user_answer: str, 
        previous_result: Optional[ValidationResult] = None
    ) -> ValidationResult:
        """
        Run edge case detection for ambiguous patterns.
        
        Args:
            candlestick_data: The candlestick data to analyze
            user_answer: The user's answer (pattern name)
            previous_result: Optional result from previous validation tiers
            
        Returns:
            ValidationResult with edge case detection details
        """
        start_time = time.time()
        
        try:
            # Check if pattern is ambiguous
            is_ambiguous, ambiguity_score = self.edge_case_detector.is_pattern_ambiguous(candlestick_data)
            
            if not is_ambiguous:
                # If not ambiguous and we have a previous result, return it
                if previous_result:
                    return previous_result
                
                # Otherwise, perform a basic validation
                detector = self._ensure_pattern_detector()
                if not detector:
                    return ValidationResult(
                        pattern_name="Unknown",
                        user_answer=user_answer,
                        confidence=0.0,
                        is_valid=False,
                        validation_tier=ValidationTier.EDGE_CASE,
                        error="Pattern detector initialization failed",
                        execution_time_ms=(time.time() - start_time) * 1000
                    )
                
                # Run async pattern detection in the event loop
                try:
                    patterns, error = self._loop.run_until_complete(
                        detector.detect_patterns_safe(candlestick_data)
                    )
                    
                    if error:
                        logger.warning(f"Pattern detection completed with warning: {error}")
                except Exception as e:
                    logger.error(f"Error during pattern detection in edge case detection: {e}")
                    if previous_result:
                        return previous_result
                    
                    return ValidationResult(
                        pattern_name="Unknown",
                        user_answer=user_answer,
                        confidence=0.0,
                        is_valid=False,
                        validation_tier=ValidationTier.EDGE_CASE,
                        error=f"Pattern detection failed: {str(e)}",
                        execution_time_ms=(time.time() - start_time) * 1000
                    )
                
                if not patterns:
                    if previous_result:
                        return previous_result
                    
                    return ValidationResult(
                        pattern_name="No Patterns",
                        user_answer=user_answer,
                        confidence=0.0,
                        is_valid=False,
                        validation_tier=ValidationTier.EDGE_CASE,
                        error="No patterns detected",
                        execution_time_ms=(time.time() - start_time) * 1000
                    )
                
                # Basic string matching with the detected patterns
                for pattern in patterns:
                    if pattern.pattern_name.lower() == user_answer:
                        return ValidationResult(
                            pattern_name=pattern.pattern_name,
                            user_answer=user_answer,
                            confidence=pattern.confidence,
                            is_valid=True,
                            validation_tier=ValidationTier.EDGE_CASE,
                            alternative_patterns=[p.pattern_name for p in patterns[:3] if p.pattern_name.lower() != user_answer],
                            execution_time_ms=(time.time() - start_time) * 1000,
                            context_features=self._extract_context_features(candlestick_data, previous_result)
                        )
                
                # If no direct match, return previous result or top pattern
                if previous_result:
                    return previous_result
                
                top_pattern = patterns[0]
                return ValidationResult(
                    pattern_name=top_pattern.pattern_name,
                    user_answer=user_answer,
                    confidence=0.6 * top_pattern.confidence,  # 60% confidence due to no match
                    is_valid=False,
                    validation_tier=ValidationTier.EDGE_CASE,
                    alternative_patterns=[p.pattern_name for p in patterns[:3]],
                    execution_time_ms=(time.time() - start_time) * 1000,
                    context_features=self._extract_context_features(candlestick_data, previous_result)
                )
            
            # Pattern is ambiguous, get characteristics and find the best match
            characteristics = self.edge_case_detector.analyze_pattern_characteristics(candlestick_data)
            
            # Get all possible patterns for comparison
            all_patterns = self._get_all_pattern_names()
            
            # For ambiguous patterns, be more lenient with matching
            for pattern_name in all_patterns:
                # Check if user answer contains or is contained in the pattern name
                if (user_answer in pattern_name.lower() or 
                    pattern_name.lower() in user_answer or
                    self._are_patterns_similar(user_answer, pattern_name.lower())):
                    
                    # For ambiguous patterns, accept similar patterns with higher confidence
                    confidence = 0.8 - (0.2 * ambiguity_score)  # Adjust confidence based on ambiguity
                    
                    return ValidationResult(
                        pattern_name=pattern_name,
                        user_answer=user_answer,
                        confidence=confidence,
                        is_valid=True,  # Consider it valid due to ambiguity
                        validation_tier=ValidationTier.EDGE_CASE,
                        alternative_patterns=[p for p in all_patterns[:5] if p.lower() != pattern_name.lower() and p.lower() != user_answer],
                        execution_time_ms=(time.time() - start_time) * 1000,
                        context_features=characteristics
                    )
            
            # If no match found even with more lenient matching, return previous result if available
            if previous_result:
                return previous_result
                
            # Return a fallback result for ambiguous patterns
            return ValidationResult(
                pattern_name="Ambiguous Pattern",
                user_answer=user_answer,
                confidence=0.5,  # Moderate confidence for ambiguous patterns
                is_valid=False,
                validation_tier=ValidationTier.EDGE_CASE,
                alternative_patterns=all_patterns[:5],
                execution_time_ms=(time.time() - start_time) * 1000,
                context_features=characteristics
            )
                
        except Exception as e:
            logger.error(f"Error in edge case detection: {e}", exc_info=True)
            if previous_result:
                return previous_result
                
            return ValidationResult(
                pattern_name="Error",
                user_answer=user_answer,
                confidence=0.0,
                is_valid=False,
                validation_tier=ValidationTier.EDGE_CASE,
                error=f"Error in edge case detection: {str(e)}",
                execution_time_ms=(time.time() - start_time) * 1000
            )
    
    def _are_patterns_similar(self, pattern1: str, pattern2: str) -> bool:
        """
        Check if two pattern names are similar.
        
        Args:
            pattern1: First pattern name
            pattern2: Second pattern name
            
        Returns:
            True if patterns are similar, False otherwise
        """
        # Convert to lowercase for case-insensitive comparison
        p1 = pattern1.lower()
        p2 = pattern2.lower()
        
        # Check for exact match
        if p1 == p2:
            return True
        
        # Check if one is a substring of the other
        if p1 in p2 or p2 in p1:
            return True
        
        # Check for common variations
        variations = {
            "engulfing": ["engulfing pattern", "engulfing candle"],
            "doji": ["doji star", "doji candlestick"],
            "hammer": ["hammer pattern", "hammer candlestick"],
            "harami": ["harami pattern", "harami cross"],
            "star": ["star pattern", "evening star", "morning star"]
        }
        
        # Check each set of variations
        for base, variants in variations.items():
            if base in p1 or any(v in p1 for v in variants):
                if base in p2 or any(v in p2 for v in variants):
                    return True
        
        return False
    
    def _get_all_pattern_names(self) -> List[str]:
        """
        Get a list of all candlestick pattern names.
        
        Returns:
            List of pattern names
        """
        # This could be loaded from configuration
        # For now, return a hardcoded list of common patterns
        return [
            "Doji",
            "Hammer",
            "Shooting Star",
            "Bullish Engulfing",
            "Bearish Engulfing",
            "Bullish Harami",
            "Bearish Harami",
            "Morning Star",
            "Evening Star",
            "Three White Soldiers",
            "Three Black Crows"
        ]

    def _run_ai_validation(
        self, 
        candlestick_data: CandlestickData, 
        user_answer: str,
        previous_result: ValidationResult
    ) -> ValidationResult:
        """
        Run AI-based validation to handle complex cases.
        
        This method uses a combination of:
        1. Fuzzy pattern name matching
        2. Context-aware pattern recognition
        3. Confidence calibration based on pattern characteristics
        
        Args:
            candlestick_data: The candlestick data to analyze
            user_answer: The user's answer (pattern name)
            previous_result: Result from previous validation tiers
            
        Returns:
            ValidationResult from AI-based validation
        """
        start_time = time.time()
        
        # Start with the previous result's confidence
        base_confidence = previous_result.confidence
        pattern_name = previous_result.pattern_name
        is_valid = previous_result.is_valid
        alternatives = previous_result.alternative_patterns.copy() if previous_result.alternative_patterns else []
        
        try:
            # Use our pattern detector to get detected patterns
            detector_patterns = self._ensure_pattern_detector().detect_patterns(candlestick_data)
            
            # If no patterns detected, return the previous result
            if not detector_patterns:
                return previous_result
            
            # Sort patterns by confidence
            sorted_patterns = sorted(detector_patterns, key=lambda p: p.confidence, reverse=True)
            
            # Get the top detected pattern
            top_pattern = sorted_patterns[0]
            top_pattern_name = top_pattern.pattern_name.lower()
            
            # Fuzzy string matching for the user answer
            all_pattern_names = [p.pattern_name.lower() for p in sorted_patterns]
            all_pattern_names.extend(self._get_all_pattern_names())
            
            # Get closest matches using difflib
            close_matches = difflib.get_close_matches(user_answer, all_pattern_names, n=3, cutoff=0.6)
            
            # Check if the user answer is a close match to the top pattern
            if close_matches and top_pattern_name in close_matches:
                is_valid = True
                similarity_ratio = difflib.SequenceMatcher(None, user_answer, top_pattern_name).ratio()
                confidence_boost = similarity_ratio * 0.2  # Up to 0.2 boost based on string similarity
                new_confidence = min(1.0, top_pattern.confidence + confidence_boost)
                
                # If it's a close match but not exact, provide explanation
                if user_answer != top_pattern_name:
                    explanation = f"Your answer '{user_answer}' closely matches '{top_pattern_name}' which is the correct pattern."
                else:
                    explanation = f"Your answer '{user_answer}' is correct."
                    
                # Add alternatives based on other close matches
                alternatives = [m for m in close_matches if m != top_pattern_name]
                
                # Create and return the validation result
                return ValidationResult(
                    pattern_name=top_pattern_name,
                    user_answer=user_answer,
                    confidence=new_confidence,
                    is_valid=is_valid,
                    validation_tier=ValidationTier.AI_BASED,
                    alternative_patterns=alternatives,
                    execution_time_ms=(time.time() - start_time) * 1000,
                    explanation=explanation
                )
            
            # If user answer isn't close to top pattern, check if it's close to any detected pattern
            for pattern in sorted_patterns:
                pattern_name_lower = pattern.pattern_name.lower()
                if user_answer in close_matches and pattern_name_lower in close_matches:
                    # User answer is close to one of the detected patterns
                    is_valid = True
                    similarity_ratio = difflib.SequenceMatcher(None, user_answer, pattern_name_lower).ratio()
                    confidence_boost = similarity_ratio * 0.15  # Slightly lower boost for non-top patterns
                    new_confidence = min(1.0, pattern.confidence + confidence_boost)
                    
                    explanation = f"Your answer '{user_answer}' is valid as it identifies the {pattern_name_lower} pattern."
                    
                    # Create and return the validation result
                    return ValidationResult(
                        pattern_name=pattern_name_lower,
                        user_answer=user_answer,
                        confidence=new_confidence,
                        is_valid=is_valid,
                        validation_tier=ValidationTier.AI_BASED,
                        alternative_patterns=[top_pattern_name] + [p.pattern_name.lower() for p in sorted_patterns[1:3]],
                        execution_time_ms=(time.time() - start_time) * 1000,
                        explanation=explanation
                    )
            
            # Check if user answer is a conceptually related pattern
            for pattern in sorted_patterns:
                pattern_name_lower = pattern.pattern_name.lower()
                related = self.related_patterns.get(pattern_name_lower, [])
                
                if user_answer in related:
                    # User identified a related pattern, partial credit
                    is_valid = False  # Not technically correct, but related
                    relation_confidence = 0.5  # Partial confidence for related pattern
                    
                    explanation = (
                        f"Your answer '{user_answer}' is related to the correct pattern '{pattern_name_lower}', "
                        f"but they have important differences."
                    )
                    
                    # Create and return the validation result for related pattern
                    return ValidationResult(
                        pattern_name=pattern_name_lower,
                        user_answer=user_answer,
                        confidence=relation_confidence,
                        is_valid=is_valid,
                        validation_tier=ValidationTier.AI_BASED,
                        alternative_patterns=[pattern_name_lower],
                        execution_time_ms=(time.time() - start_time) * 1000,
                        explanation=explanation
                    )
            
            # If we got here, revert to previous result but add insights
            if top_pattern and not is_valid:
                # Previous result wasn't valid, but we have a detected pattern
                explanation = (
                    f"The pattern in the chart is actually a {top_pattern_name} pattern. "
                    f"Your answer '{user_answer}' doesn't match this pattern."
                )
                
                # Create enhanced result with explanation
                return ValidationResult(
                    pattern_name=top_pattern_name,
                    user_answer=user_answer,
                    confidence=top_pattern.confidence,
                    is_valid=False,
                    validation_tier=ValidationTier.AI_BASED,
                    alternative_patterns=[p.pattern_name.lower() for p in sorted_patterns[:3]],
                    execution_time_ms=(time.time() - start_time) * 1000,
                    explanation=explanation
                )
            
            # Final fallback: return enhanced previous result
            result = ValidationResult(
                pattern_name=previous_result.pattern_name,
                user_answer=user_answer,
                confidence=previous_result.confidence,
                is_valid=previous_result.is_valid,
                validation_tier=ValidationTier.AI_BASED,
                alternative_patterns=previous_result.alternative_patterns,
                execution_time_ms=(time.time() - start_time) * 1000,
                explanation=f"Based on our analysis, the pattern is most likely a {pattern_name if pattern_name else 'difficult to identify'} pattern."
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in AI validation: {e}")
            
            # Return the previous result if AI validation fails
            return previous_result

class EdgeCaseDetector:
    """
    Specialized detector for ambiguous or edge case candlestick patterns.
    
    This class implements heuristics and specialized algorithms to detect
    and handle ambiguous patterns that may be difficult to classify with
    standard pattern detection algorithms.
    """
    
    def __init__(self, ambiguity_threshold: float = 0.7):
        """
        Initialize the edge case detector.
        
        Args:
            ambiguity_threshold: Threshold for considering a pattern ambiguous
        """
        self.ambiguity_threshold = ambiguity_threshold
        self._loop = None
        self._detector = None
    
    def _get_event_loop(self):
        """Get or create an event loop for async operations."""
        if self._loop is None:
            import asyncio
            try:
                self._loop = asyncio.get_event_loop()
            except RuntimeError:
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
        return self._loop
    
    async def _get_detector(self):
        """Get the pattern detector, initializing if necessary."""
        if self._detector is None:
            from backend.assessments.candlestick_patterns.pattern_detection import get_default_detector
            self._detector = await get_default_detector()
        return self._detector
    
    def is_pattern_ambiguous(self, data: CandlestickData) -> Tuple[bool, float]:
        """
        Determine if a candlestick pattern is ambiguous.
        
        Args:
            data: Candlestick data to analyze
            
        Returns:
            Tuple of (is_ambiguous, ambiguity_score)
        """
        try:
            # Get the pattern detector
            loop = self._get_event_loop()
            detector = loop.run_until_complete(self._get_detector())
            
            # Detect patterns with error handling
            patterns, error = loop.run_until_complete(detector.detect_patterns_safe(data))
            
            if error:
                logger.warning(f"Pattern detection completed with warning: {error}")
            
            if not patterns:
                return False, 0.0
                
            # Sort by confidence
            patterns = sorted(patterns, key=lambda x: x.confidence, reverse=True)
            
            # If only one pattern detected, it's not ambiguous
            if len(patterns) == 1:
                return False, 0.0
                
            # If top pattern has high confidence and second pattern has low confidence,
            # the pattern is not ambiguous
            if len(patterns) >= 2:
                top_confidence = patterns[0].confidence
                second_confidence = patterns[1].confidence
                
                # Check the confidence gap
                confidence_gap = top_confidence - second_confidence
                
                # Small gap indicates ambiguity
                if confidence_gap < 0.3 and second_confidence > 0.4:
                    ambiguity_score = 1.0 - confidence_gap
                    return True, ambiguity_score
            
            # Check for overlapping pattern indices - can indicate ambiguity
            if len(patterns) >= 2:
                pattern1_indices = set(patterns[0].candle_indices)
                pattern2_indices = set(patterns[1].candle_indices)
                
                overlap_ratio = len(pattern1_indices & pattern2_indices) / len(pattern1_indices | pattern2_indices)
                
                if overlap_ratio > 0.5:
                    ambiguity_score = overlap_ratio
                    return True, ambiguity_score
            
            return False, 0.0
                
        except Exception as e:
            logger.error(f"Error checking pattern ambiguity: {e}", exc_info=True)
            return False, 0.0
    
    def analyze_pattern_characteristics(self, data: CandlestickData) -> Dict[str, Any]:
        """
        Analyze the characteristics of potentially ambiguous patterns.
        
        Args:
            data: Candlestick data to analyze
            
        Returns:
            Dictionary of pattern characteristics
        """
        characteristics = {}
        
        try:
            # Get candles data
            candles = data.candles
            if not candles or len(candles) < 3:
                return {"error": "Insufficient candles for analysis"}
            
            # Get the pattern detector
            loop = self._get_event_loop()
            detector = loop.run_until_complete(self._get_detector())
            
            # Detect patterns with error handling
            patterns, error = loop.run_until_complete(detector.detect_patterns_safe(data))
            
            if error:
                logger.warning(f"Pattern detection completed with warning in analyze_pattern_characteristics: {error}")
            
            # Calculate overall trend
            closes = [c.close for c in candles]
            opens = [c.open for c in candles]
            highs = [c.high for c in candles]
            lows = [c.low for c in candles]
            
            # Determine trend
            if len(closes) >= 5:
                start_avg = sum(closes[:3]) / 3
                end_avg = sum(closes[-3:]) / 3
                
                if end_avg > start_avg * 1.02:
                    characteristics["trend"] = "uptrend"
                elif end_avg < start_avg * 0.98:
                    characteristics["trend"] = "downtrend"
                else:
                    characteristics["trend"] = "sideways"
            
            # Calculate volatility
            if len(highs) > 0 and len(lows) > 0:
                avg_range = sum([(h - l) / l for h, l in zip(highs, lows)]) / len(highs)
                characteristics["volatility"] = avg_range
                
                if avg_range > 0.03:
                    characteristics["volatility_description"] = "high"
                elif avg_range > 0.01:
                    characteristics["volatility_description"] = "medium"
                else:
                    characteristics["volatility_description"] = "low"
            
            # Check for key reversal patterns
            if len(patterns) > 0:
                pattern_names = [p.pattern_name.lower() for p in patterns]
                
                # Check for common reversal patterns
                reversal_patterns = ["engulfing", "hammer", "doji", "star", "harami"]
                found_reversals = [p for p in pattern_names if any(r in p for r in reversal_patterns)]
                
                if found_reversals:
                    characteristics["potential_reversal"] = True
                    characteristics["reversal_patterns"] = found_reversals
                    
                    # Check pattern direction
                    bullish_patterns = [p for p in patterns if p.bullish]
                    bearish_patterns = [p for p in patterns if p.bullish is False]
                    
                    if len(bullish_patterns) > len(bearish_patterns):
                        characteristics["reversal_direction"] = "bullish"
                    elif len(bearish_patterns) > len(bullish_patterns):
                        characteristics["reversal_direction"] = "bearish"
                    else:
                        characteristics["reversal_direction"] = "mixed"
            
            # Get confidence range of detected patterns
            if patterns:
                characteristics["max_confidence"] = max(p.confidence for p in patterns)
                characteristics["min_confidence"] = min(p.confidence for p in patterns)
                characteristics["avg_confidence"] = sum(p.confidence for p in patterns) / len(patterns)
            
            return characteristics
                
        except Exception as e:
            logger.error(f"Error analyzing pattern characteristics: {e}", exc_info=True)
            return {"error": str(e)}

class ThreadedValidationPipeline(MultiTierValidationPipeline):
    """
    Threaded implementation of the multi-tier validation pipeline.
    
    This class uses a thread pool to execute validation tiers concurrently,
    improving performance for real-time assessments. It includes proper resource
    management and timeout handling.
    """
    
    def __init__(self, 
             confidence_threshold: float = 0.7, 
             max_workers: int = 4,
             timeout_seconds: float = 2.0):
        """
        Initialize the threaded validation pipeline.
        
        Args:
            confidence_threshold: Minimum confidence for pattern validation
            max_workers: Maximum number of worker threads
            timeout_seconds: Timeout for validation operations
        """
        super().__init__(confidence_threshold)
        self.max_workers = max(1, min(max_workers, os.cpu_count() or 4))
        self.timeout_seconds = timeout_seconds
        
        # Initialize thread pool and resource management
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)
        self._thread_loops = {}  # Per-thread event loops
        self._executor_lock = threading.RLock()  # Lock for thread-safe operations
        self._shutdown = False
        
        # Cache for common resources
        self._pattern_names_cache = None
        
        logger.info(f"Initialized ThreadedValidationPipeline with {self.max_workers} workers and {timeout_seconds}s timeout")
    
    def __del__(self):
        """Cleanup resources when the object is garbage collected."""
        self.shutdown()
    
    def shutdown(self):
        """Shutdown the thread pool and cleanup resources."""
        with self._executor_lock:
            if not self._shutdown and hasattr(self, '_executor'):
                self._shutdown = True
                try:
                    # Cancel any pending futures
                    self._executor.shutdown(wait=True, cancel_futures=True)
                    
                    # Clean up per-thread resources
                    for loop in self._thread_loops.values():
                        if loop.is_running():
                            try:
                                loop.stop()
                            except Exception:
                                pass
                    self._thread_loops.clear()
                    
                    logger.info("Successfully shutdown ThreadedValidationPipeline executor")
                except Exception as e:
                    logger.error(f"Error shutting down executor: {e}", exc_info=True)
    
    def _get_thread_loop(self):
        """Get or create an event loop for the current thread."""
        thread_id = threading.get_ident()
        
        with self._executor_lock:
            if thread_id not in self._thread_loops:
                try:
                    loop = asyncio.new_event_loop()
                    self._thread_loops[thread_id] = loop
                    asyncio.set_event_loop(loop)
                    logger.debug(f"Created new event loop for thread {thread_id}")
                except Exception as e:
                    logger.error(f"Error creating event loop for thread {thread_id}: {e}")
                    # Fallback to main loop
                    try:
                        loop = asyncio.get_event_loop()
                        self._thread_loops[thread_id] = loop
                    except RuntimeError:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        self._thread_loops[thread_id] = loop
            
            return self._thread_loops[thread_id]
    
    def _get_all_pattern_names(self) -> List[str]:
        """Get all pattern names with caching for performance."""
        if self._pattern_names_cache is None:
            self._pattern_names_cache = super()._get_all_pattern_names()
        return self._pattern_names_cache
    
    def validate(self, candlestick_data: CandlestickData, user_answer: str) -> ValidationResult:
        """
        Validate a user's answer using concurrent validation tiers.
        
        This method runs multiple validation strategies concurrently and
        selects the most confident result.
        
        Args:
            candlestick_data: The candlestick data to analyze
            user_answer: The user's answer (pattern name)
            
        Returns:
            ValidationResult with validation details
        """
        # Normalize user answer
        normalized_user_answer = user_answer.strip().lower()
        original_answer = user_answer
        start_time = time.time()
        
        # Validate inputs
        if not candlestick_data or not candlestick_data.candles or len(candlestick_data.candles) == 0:
            return self._create_fallback_validation(
                normalized_user_answer,
                original_answer,
                start_time,
                error="Invalid or empty candlestick data"
            )
        
        if not normalized_user_answer:
            return self._create_fallback_validation(
                normalized_user_answer,
                original_answer,
                start_time,
                error="Empty user answer"
            )
        
        try:
            # Submit validation tasks to thread pool
            with self._executor_lock:
                if self._shutdown:
                    return self._create_fallback_validation(
                        normalized_user_answer,
                        original_answer,
                        start_time,
                        error="Validation pipeline is shutting down"
                    )
                
                # Define validation task map with prioritization
                validation_tasks = [
                    (ValidationTier.PRIMARY, lambda: self._run_primary_validation(
                        candlestick_data, normalized_user_answer
                    )),
                    (ValidationTier.SECONDARY, lambda: self._run_secondary_validation(
                        candlestick_data, normalized_user_answer, None
                    )),
                    (ValidationTier.EDGE_CASE, lambda: self._run_edge_case_detection(
                        candlestick_data, normalized_user_answer, None
                    )),
                    (ValidationTier.AI_BASED, lambda: self._run_ai_validation(
                        candlestick_data, normalized_user_answer, None
                    ))
                ]
                
                # Submit each validation task to the thread pool
                future_to_tier = {}
                for tier, task in validation_tasks:
                    future = self._executor.submit(task)
                    future_to_tier[future] = tier
                
                # Track results and handle timeouts
                results = []
                
                try:
                    # Wait for first task to complete to get early results
                    for future in concurrent.futures.as_completed(future_to_tier.keys(), timeout=self.timeout_seconds):
                        try:
                            tier = future_to_tier[future]
                            result = future.result()
                            results.append((tier, result))
                            logger.debug(f"Completed validation for tier {tier.value}")
                            
                            # If we have a high-confidence result, we can stop waiting
                            if result.confidence > 0.9 and result.is_valid:
                                break
                        except Exception as e:
                            logger.error(f"Error in validation tier {future_to_tier[future].value}: {e}")
                        
                        # Stop if we've hit the timeout
                        if time.time() - start_time > self.timeout_seconds:
                            break
                    
                except concurrent.futures.TimeoutError:
                    logger.warning(f"Timeout waiting for validation results after {self.timeout_seconds}s")
                
                # Cancel any remaining tasks
                for future in [f for f in future_to_tier.keys() if not f.done()]:
                    future.cancel()
                    logger.debug(f"Cancelled validation for tier {future_to_tier[future].value}")
                
                # Select best result based on confidence and validity
                if not results:
                    return self._create_fallback_validation(
                        normalized_user_answer, 
                        original_answer,
                        start_time,
                        error="All validation tiers failed or timed out"
                    )
                
                # Sort results by confidence (highest first) and prioritize valid results
                results.sort(key=lambda x: (x[1].is_valid, x[1].confidence), reverse=True)
                
                # Get highest confidence result
                best_tier, best_result = results[0]
                logger.info(f"Selected {best_tier.value} validation with confidence {best_result.confidence:.2f}")
                
                # Enhance with context features if needed
                if not best_result.context_features:
                    try:
                        best_result.context_features = self._extract_context_features(
                            candlestick_data, best_result
                        )
                    except Exception as e:
                        logger.error(f"Error extracting context features: {e}")
                        # Ensure context_features is never None
                        if best_result.context_features is None:
                            best_result.context_features = {}
                
                # Update execution time
                best_result.execution_time_ms = (time.time() - start_time) * 1000
                
                return best_result
                
        except Exception as e:
            logger.error(f"Error in ThreadedValidationPipeline: {e}", exc_info=True)
            return self._create_fallback_validation(
                normalized_user_answer,
                original_answer,
                start_time,
                error=f"Validation error: {str(e)}"
            )

    def _create_fallback_validation(self, 
                                  normalized_user_answer: str, 
                                  original_answer: str,
                                  start_time: float,
                                  error: Optional[str] = None) -> ValidationResult:
        """Create a fallback validation result when normal validation fails."""
        # Try to make a simple string-based validation as a fallback
        fallback_confidence = 0.1
        fallback_valid = False
        fallback_pattern = "unknown"
        
        try:
            # Get all patterns and find closest match
            all_patterns = [p.lower() for p in self._get_all_pattern_names()]
            
            if all_patterns and normalized_user_answer:
                # First check for exact matches
                if normalized_user_answer in all_patterns:
                    fallback_pattern = normalized_user_answer
                    fallback_valid = True
                    fallback_confidence = 0.8
                else:
                    # Try fuzzy matching
                    closest_matches = difflib.get_close_matches(
                        normalized_user_answer, all_patterns, n=3, cutoff=0.6
                    )
                    
                    if closest_matches:
                        fallback_pattern = closest_matches[0]
                        fallback_valid = normalized_user_answer == fallback_pattern
                        
                        # Calculate string similarity for confidence
                        similarity = difflib.SequenceMatcher(
                            None, normalized_user_answer, fallback_pattern
                        ).ratio()
                        fallback_confidence = max(0.1, min(0.5, similarity))
        except Exception as fallback_error:
            logger.error(f"Error in fallback validation: {fallback_error}")
        
        # Create the fallback validation result
        return ValidationResult(
            pattern_name=fallback_pattern.capitalize() if fallback_pattern != "unknown" else fallback_pattern,
            user_answer=original_answer,
            confidence=fallback_confidence,
            is_valid=fallback_valid,
            validation_tier=ValidationTier.PRIMARY,
            alternative_patterns=[],
            execution_time_ms=(time.time() - start_time) * 1000,
            error=error
        )

class CandlestickAnswerEvaluator(AnswerEvaluator[CandlestickQuestion, BaseAnswerEvaluation]):
    """
    Evaluator for candlestick pattern assessment answers.
    
    This class implements the answer evaluation logic for candlestick pattern
    questions, using a multi-tier validation pipeline and explanation generation.
    """
    
    def __init__(
        self,
        validation_pipeline: Optional[MultiTierValidationPipeline] = None,
        explanation_generator: Optional[BaseExplanationGenerator] = None,
        default_score_correct: float = 1.0,
        default_score_incorrect: float = 0.0,
        partial_credit: bool = True
    ):
        """
        Initialize the evaluator with optional components.
        
        Args:
            validation_pipeline: Pipeline for validating user answers
            explanation_generator: Generator for explanations
            default_score_correct: Score for correct answers
            default_score_incorrect: Score for incorrect answers
            partial_credit: Whether to give partial credit for close answers
        """
        self.validation_pipeline = validation_pipeline or MultiTierValidationPipeline()
        self.explanation_generator = explanation_generator
        self.default_score_correct = default_score_correct
        self.default_score_incorrect = default_score_incorrect
        self.partial_credit = partial_credit
        
        # Initialize statistics tracking
        self._answer_stats: Dict[str, Dict[str, Any]] = {}
        
        logger.info("Initialized CandlestickAnswerEvaluator")

    async def evaluate_answer(self, question: CandlestickQuestion, user_answer: Any) -> BaseAnswerEvaluation:
        """
        Evaluate a user's answer to a candlestick pattern question.
        
        Args:
            question: The question being answered
            user_answer: The user's answer
            
        Returns:
            AnswerEvaluation with evaluation results
        """
        start_time = time.time()
        
        try:
            # Validate inputs
            if not question:
                return BaseAnswerEvaluation(
                    is_correct=False,
                    score=0.0,
                    confidence=0.0,
                    feedback="Invalid question"
                )
            
            if user_answer is None:
                return BaseAnswerEvaluation(
                    is_correct=False,
                    score=0.0,
                    confidence=0.0,
                    feedback="No answer provided"
                )
            
            # Extract candlestick data from question
            candlestick_data = self._extract_candlestick_data(question)
            if not candlestick_data:
                return BaseAnswerEvaluation(
                    is_correct=False,
                    score=0.0,
                    confidence=0.0,
                    feedback="Could not retrieve candlestick data from question"
                )
            
            # Convert user_answer to string if it's not already
            str_answer = str(user_answer).strip() if user_answer is not None else ""
            
            # Get correct answer from question
            correct_pattern = question.correct_answer
            
            # Validate the user's answer
            validation_result = self.validation_pipeline.validate(candlestick_data, str_answer)
            
            # Determine if the answer is correct
            is_correct = validation_result.is_valid
            
            # Calculate score based on correctness and confidence
            score = self.default_score_correct if is_correct else self.default_score_incorrect
            
            # Adjust score based on confidence if partially correct
            if is_correct and validation_result.confidence < 0.9:
                score = score * validation_result.confidence
            elif not is_correct and self.partial_credit and validation_result.confidence > 0.5:
                # Award partial credit for close answers
                score = self.default_score_correct * (validation_result.confidence * 0.5)
            
            # Generate explanation if we have an explanation generator
            explanation = ""
            if self.explanation_generator:
                try:
                    user_level = UserLevel.BEGINNER  # Default to beginner if not specified
                    if hasattr(question, 'user_level'):
                        user_level = question.user_level
                    elif hasattr(question, 'difficulty'):
                        if question.difficulty <= 0.3:
                            user_level = UserLevel.BEGINNER
                        elif question.difficulty <= 0.7:
                            user_level = UserLevel.INTERMEDIATE
                        else:
                            user_level = UserLevel.ADVANCED
                            
                    explanation = await self.explanation_generator.explain_user_answer(
                        question, user_answer, is_correct, user_level
                    )
                except Exception as explanation_error:
                    logger.error(f"Error generating explanation: {explanation_error}")
            
            # If no explanation was generated, use a default one
            if not explanation:
                explanation = f"This is a {correct_pattern} pattern." if is_correct else \
                           f"The correct answer is {correct_pattern}."
            
            # Create detailed feedback
            feedback = self._generate_feedback(
                validation_result, correct_pattern, str_answer, is_correct
            )
            
            # Calculate total time
            total_time_ms = (time.time() - start_time) * 1000
            
            # Update statistics if question has an ID
            if hasattr(question, 'id'):
                self._update_answer_statistics(question.id, validation_result)
            
            # Create and return evaluation with performance metrics
            return BaseAnswerEvaluation(
                is_correct=is_correct,
                score=score,
                confidence=validation_result.confidence,
                feedback=feedback,
                explanation=explanation,
                metadata={
                    'validation_tier': validation_result.validation_tier.value,
                    'alternative_patterns': validation_result.alternative_patterns or [],
                    'execution_time_ms': validation_result.execution_time_ms,
                    'total_evaluation_time_ms': total_time_ms,
                    'context_features': validation_result.context_features or {},
                    'validation_result': validation_result
                }
            )
            
        except Exception as e:
            logger.error(f"Error evaluating answer: {e}", exc_info=True)
            return BaseAnswerEvaluation(
                is_correct=False,
                score=0.0,
                confidence=0.0,
                feedback=f"Error evaluating answer: {str(e)}",
                explanation="There was an error processing your answer."
            )

    def _update_answer_statistics(
        self,
        question_id: str,
        evaluation: BaseAnswerEvaluation,
        validation_result: Optional[ValidationResult] = None
    ) -> None:
        """
        Update answer statistics for a question.
        
        Args:
            question_id: The question identifier
            evaluation: The evaluation result
            validation_result: Optional validation result with additional details
        """
        try:
            # Get or create stats for this question
            if question_id not in self._answer_stats:
                self._answer_stats[question_id] = {
                    "total_attempts": 0,
                    "correct_attempts": 0,
                    "incorrect_attempts": 0,
                    "average_score": 0.0,
                    "common_mistakes": {},
                    "average_confidence": 0.0,
                    "validation_tier_counts": {
                        "primary": 0,
                        "secondary": 0,
                        "edge_case": 0,
                        "ai_based": 0
                    }
                }
            
            stats = self._answer_stats[question_id]
            
            # Update basic statistics
            stats["total_attempts"] += 1
            if evaluation.is_correct:
                stats["correct_attempts"] += 1
            else:
                stats["incorrect_attempts"] += 1
            
            # Update average score
            total_score = (stats["average_score"] * (stats["total_attempts"] - 1) + evaluation.score)
            stats["average_score"] = total_score / stats["total_attempts"]
            
            # Update validation tier counts if available
            if validation_result and validation_result.validation_tier:
                tier_name = validation_result.validation_tier.value
                stats["validation_tier_counts"][tier_name] = (
                    stats["validation_tier_counts"].get(tier_name, 0) + 1
                )
            
            # Update common mistakes for incorrect answers
            if not evaluation.is_correct and hasattr(evaluation, 'user_answer'):
                user_answer = str(evaluation.user_answer)
                stats["common_mistakes"][user_answer] = stats["common_mistakes"].get(user_answer, 0) + 1
            
            # Update average confidence if available
            if validation_result:
                total_confidence = (stats["average_confidence"] * (stats["total_attempts"] - 1) + 
                                  validation_result.confidence)
                stats["average_confidence"] = total_confidence / stats["total_attempts"]
                
        except Exception as e:
            logger.error(f"Error updating answer statistics: {str(e)}")

    def _generate_feedback(
        self, 
        validation_result: ValidationResult, 
        correct_pattern: str, 
        user_answer: str,
        is_correct: bool
    ) -> str:
        """Generate detailed feedback based on validation results."""
        if is_correct:
            if validation_result.confidence >= 0.9:
                return f"Correct! You identified the {correct_pattern} pattern."
            else:
                return f"Correct! You identified the {correct_pattern} pattern, although the pattern is somewhat ambiguous."
        
        # Incorrect answer feedback
        feedback = f"Not quite. The correct pattern is {correct_pattern}."
        
        # Add similarity feedback if available
        if validation_result.alternative_patterns and user_answer.lower() in [p.lower() for p in validation_result.alternative_patterns]:
            feedback += f" Your answer '{user_answer}' was identified as an alternative possibility."
        elif validation_result.pattern_name.lower() != "unknown" and validation_result.pattern_name.lower() != "no patterns":
            feedback += f" Your answer was interpreted as '{validation_result.pattern_name}'."
        
        # Add confidence-based guidance
        if validation_result.confidence > 0.7:
            feedback += " Your answer was close to the correct pattern."
        elif validation_result.confidence > 0.4:
            feedback += " Consider reviewing the distinguishing features of this pattern."
        else:
            feedback += " Review the key characteristics of common candlestick patterns."
        
        return feedback

    def _extract_candlestick_data(self, question: CandlestickQuestion) -> Optional[CandlestickData]:
        """Extract candlestick data from a question."""
        try:
            # First try direct access
            if hasattr(question, 'chart_data') and question.chart_data:
                return CandlestickData.from_chart_data(question.chart_data)
            
            # Next, try to get data from the candlestick attribute
            if hasattr(question, 'candlestick_data') and question.candlestick_data:
                return question.candlestick_data
                
            # Finally, try to get data from the data attribute
            if hasattr(question, 'data') and question.data:
                if isinstance(question.data, CandlestickData):
                    return question.data
                return CandlestickData.from_chart_data(question.data)
            
            return None
        except Exception as e:
            logger.error(f"Error extracting candlestick data: {e}")
            return None

    async def evaluate_session_answers(
        self,
        questions: List[CandlestickQuestion],
        user_answers: List[Any]
    ) -> List[BaseAnswerEvaluation]:
        """
        Evaluate all answers for a session.
        
        Args:
            questions: List of questions
            user_answers: List of corresponding user answers
            
        Returns:
            List of evaluations for each answer
            
        Raises:
            ValueError: If questions and answers lists have different lengths
        """
        if len(questions) != len(user_answers):
            raise ValueError("Number of questions and answers must match")
            
        # Evaluate answers concurrently for better performance
        evaluations = await asyncio.gather(*[
            self.evaluate_answer(question, answer)
            for question, answer in zip(questions, user_answers)
        ])
        
        return evaluations

    async def evaluate_partial_answer(
        self,
        question: CandlestickQuestion,
        partial_answer: Any
    ) -> Dict[str, Any]:
        """
        Evaluate a partial or in-progress answer.
        
        Provides feedback during answer construction, before final submission.
        
        Args:
            question: The question being answered
            partial_answer: The partial answer
            
        Returns:
            Dictionary containing evaluation details and guidance
        """
        try:
            # Extract candlestick data from question
            candlestick_data = self._extract_candlestick_data(question)
            if not candlestick_data:
                return {
                    "is_valid": False,
                    "confidence": 0.0,
                    "feedback": "Unable to evaluate partial answer: missing candlestick data",
                    "suggestions": []
                }
            
            # Run validation with lower confidence threshold for partial answers
            validation_result = self.validation_pipeline.validate(
                candlestick_data=candlestick_data,
                user_answer=str(partial_answer)
            )
            
            # Generate suggestions based on partial match
            suggestions = []
            if validation_result.alternative_patterns:
                suggestions = [
                    pattern for pattern in validation_result.alternative_patterns
                    if str(partial_answer).lower() in pattern.lower()
                ]
            
            return {
                "is_valid": validation_result.is_valid,
                "confidence": validation_result.confidence,
                "feedback": validation_result.explanation or "Continue entering your answer...",
                "suggestions": suggestions[:5]  # Limit to top 5 suggestions
            }
            
        except Exception as e:
            logger.error(f"Error evaluating partial answer: {str(e)}")
            return {
                "is_valid": False,
                "confidence": 0.0,
                "feedback": "Error evaluating partial answer",
                "suggestions": []
            }

    async def generate_feedback(
        self,
        question: CandlestickQuestion,
        user_answer: Any,
        evaluation: BaseAnswerEvaluation
    ) -> str:
        """
        Generate detailed feedback for an answer.
        
        Args:
            question: The question that was answered
            user_answer: The user's answer
            evaluation: The evaluation of the answer
            
        Returns:
            Detailed feedback string
        """
        try:
            # Get correct pattern from question
            correct_pattern = question.correct_answer
            if not correct_pattern:
                return "Unable to generate feedback: missing correct answer"
            
            # Get validation result from evaluation metadata
            validation_result = None
            if hasattr(evaluation, 'metadata') and evaluation.metadata:
                validation_result = evaluation.metadata.get('validation_result')
            
            if not validation_result:
                # Create a basic validation result if none exists
                validation_result = ValidationResult(
                    pattern_name=correct_pattern,
                    user_answer=str(user_answer),
                    confidence=evaluation.score,
                    is_valid=evaluation.is_correct,
                    validation_tier=ValidationTier.PRIMARY
                )
            
            # Generate feedback using validation result
            return self._generate_feedback(
                validation_result=validation_result,
                correct_pattern=correct_pattern,
                user_answer=str(user_answer),
                is_correct=evaluation.is_correct
            )
            
        except Exception as e:
            logger.error(f"Error generating feedback: {str(e)}")
            return "Unable to generate detailed feedback at this time."

    async def get_answer_statistics(
        self,
        question_id: str
    ) -> Dict[str, Any]:
        """
        Get statistics about answers for a specific question.
        
        Args:
            question_id: The question identifier
            
        Returns:
            Dictionary containing answer statistics
        """
        try:
            # Get stats for this question, or create new stats
            stats = self._answer_stats.get(question_id, {
                "total_attempts": 0,
                "correct_attempts": 0,
                "incorrect_attempts": 0,
                "average_score": 0.0,
                "common_mistakes": {},
                "average_confidence": 0.0,
                "validation_tier_counts": {
                    "primary": 0,
                    "secondary": 0,
                    "edge_case": 0,
                    "ai_based": 0
                }
            })
            
            return {
                "question_id": question_id,
                "statistics": stats,
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Error getting answer statistics: {str(e)}")
            return {
                "question_id": question_id,
                "error": "Unable to retrieve statistics",
                "timestamp": time.time()
            }