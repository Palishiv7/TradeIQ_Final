"""
Ensemble Pattern Detectors

This module provides ensemble pattern detectors that combine multiple 
detection strategies for improved accuracy and reliability.
"""

import time
import threading
import logging
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from collections import Counter, defaultdict
import asyncio

from backend.common.logger import get_logger
from backend.assessments.candlestick_patterns.candlestick_utils import Candle, CandlestickData
from backend.assessments.candlestick_patterns.pattern_detection.interface import (
    PatternDetector, PatternMatch, DetectionStrategy
)

# Configure logger
logger = get_logger(__name__)


class EnsembleDetector(PatternDetector):
    """
    Base class for ensemble pattern detectors.
    
    This class provides functionality for combining the results of multiple detectors
    through various ensemble methods.
    """
    
    def __init__(
        self,
        detectors: List[PatternDetector],
        name: str = "EnsembleDetector",
        min_confidence: float = 0.5,
        max_patterns: Optional[int] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the ensemble detector.
        
        Args:
            detectors: List of pattern detectors to use in the ensemble
            name: Custom name for this detector
            min_confidence: Minimum confidence threshold for patterns (0.0-1.0)
            max_patterns: Maximum number of patterns to return (None for unlimited)
            config: Additional configuration parameters
        
        Raises:
            ValueError: If detectors list is empty
        """
        if not detectors:
            raise ValueError("Ensemble detector requires at least one sub-detector")
            
        strategy = DetectionStrategy.ENSEMBLE
        super().__init__(
            name=name, 
            strategy=strategy,
            min_confidence=min_confidence,
            max_patterns=max_patterns,
            config=config or {}
        )
        
        self.detectors = detectors
        self.timeout_seconds = self.config.get("timeout_seconds", 5.0)
    
    async def initialize(self) -> bool:
        """
        Initialize all component detectors.
        
        Returns:
            True if all detectors were initialized successfully, False otherwise
        """
        initialization_results = await asyncio.gather(
            *[detector.initialize() for detector in self.detectors],
            return_exceptions=True
        )
        
        # Check if any detectors failed to initialize
        success = all(
            result is True 
            for result in initialization_results 
            if not isinstance(result, Exception)
        )
        
        self._initialized = success
        return success
    
    async def detect_patterns(self, candlestick_data: CandlestickData) -> List[PatternMatch]:
        """
        Detect patterns using all component detectors.
        
        This implementation runs all detectors concurrently and combines their results.
        
        Args:
            candlestick_data: Candlestick data to analyze
            
        Returns:
            Combined list of detected patterns
            
        Raises:
            ValueError: If candlestick_data is invalid
            RuntimeError: If all detectors fail
        """
        if not self.is_initialized():
            await self.initialize()
            
        if not candlestick_data or not candlestick_data.candles:
            raise ValueError("Cannot detect patterns: empty candlestick data")
            
        start_time = time.time()
        all_patterns: List[PatternMatch] = []
        
        # Create tasks for all detectors
        detector_tasks = {
            detector.name: asyncio.create_task(detector.detect_patterns_safe(candlestick_data))
            for detector in self.detectors
        }
        
        # Wait for all tasks to complete or timeout
        try:
            completed, pending = await asyncio.wait(
                detector_tasks.values(),
                timeout=self.timeout_seconds,
                return_when=asyncio.ALL_COMPLETED
            )
            
            # Cancel any pending tasks
            for task in pending:
                task.cancel()
            
            # Collect results from completed tasks
            success_count = 0
            for task in completed:
                try:
                    patterns, error = await task
                    if error is None:
                        all_patterns.extend(patterns)
                        success_count += 1
                    else:
                        logger.warning(f"Detector error: {error}")
                except Exception as e:
                    logger.exception(f"Error processing detector result: {e}")
            
            if success_count == 0 and self.detectors:
                logger.error("All detectors failed to detect patterns")
                if len(self.detectors) == 1:
                    # If only one detector, propagate the error
                    raise RuntimeError("Pattern detection failed: sole detector failed")
        except asyncio.TimeoutError:
            logger.warning(f"Detector timeout after {self.timeout_seconds} seconds")
            # Cancel all tasks
            for task in detector_tasks.values():
                task.cancel()
        
        # Process and combine results
        combined_patterns = self._combine_patterns(all_patterns)
        
        # Add detection metadata
        detection_time_ms = (time.time() - start_time) * 1000
        for pattern in combined_patterns:
            if not pattern.detection_strategy:
                pattern.detection_strategy = self.strategy
            pattern.detection_time_ms = detection_time_ms
            pattern.metadata["ensemble_detector"] = self.name
        
        # Apply filtering based on confidence threshold and max patterns
        return self.filter_patterns(combined_patterns)
    
    def _combine_patterns(self, patterns: List[PatternMatch]) -> List[PatternMatch]:
        """
        Combine patterns from multiple detectors.
        
        Default implementation just returns all patterns sorted by confidence.
        
        Args:
            patterns: All patterns from all detectors
            
        Returns:
            Combined list of patterns
        """
        # Sort by confidence (descending)
        return sorted(patterns, key=lambda p: p.confidence, reverse=True)


class WeightedConsensusDetector(EnsembleDetector):
    """
    Ensemble detector using weighted consensus voting.
    
    This detector combines the results from multiple detectors using a weighted consensus
    approach, where patterns found by multiple detectors receive higher confidence.
    """
    
    def __init__(
        self,
        detectors: List[PatternDetector],
        detector_weights: Optional[Dict[str, float]] = None,
        name: str = "WeightedConsensusDetector",
        min_confidence: float = 0.6,
        max_patterns: Optional[int] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the weighted consensus detector.
        
        Args:
            detectors: List of pattern detectors to use in the ensemble
            detector_weights: Optional weights for each detector (by name)
            name: Custom name for this detector
            min_confidence: Minimum confidence threshold for patterns (0.0-1.0)
            max_patterns: Maximum number of patterns to return (None for unlimited)
            config: Additional configuration parameters
        """
        # Prepare configuration
        config = config or {}
        
        # Call parent constructor with WEIGHTED_CONSENSUS strategy
        super().__init__(
            detectors=detectors,
            name=name,
            min_confidence=min_confidence,
            max_patterns=max_patterns,
            config=config
        )
        
        # Override strategy type
        self.strategy = DetectionStrategy.WEIGHTED_CONSENSUS
        
        # Get configuration parameters
        self.consensus_threshold = config.get("consensus_threshold", 0.6)
        self.similarity_threshold = config.get("similarity_threshold", 0.7)
        self.dynamic_weighting = config.get("dynamic_weighting", True)
        
        # Initialize detector weights (default to equal weights)
        if detector_weights is None:
            # Equal weights for all detectors
            detector_weights = {detector.name: 1.0 for detector in detectors}
        
        # Normalize weights to sum to 1.0
        total_weight = sum(detector_weights.values())
        self.detector_weights = {
            name: weight / total_weight 
            for name, weight in detector_weights.items()
        } if total_weight > 0 else {
            name: 1.0 / len(detector_weights) 
            for name in detector_weights
        }
        
        # Performance tracking for dynamic weighting
        self.detector_performance = {
            detector.name: {
                "patterns_detected": 0,
                "consensus_matches": 0,
                "average_confidence": 0.0,
                "total_confidence": 0.0
            } for detector in detectors
        }
    
    def _combine_patterns(self, patterns: List[PatternMatch]) -> List[PatternMatch]:
        """
        Combine patterns using weighted consensus approach.
        
        This implementation:
        1. Groups similar patterns (same type on same candles)
        2. Applies detector weights to confidence scores
        3. Merges similar patterns with weighted confidence
        4. Boosts confidence based on consensus level
        
        Args:
            patterns: All patterns from all detectors
            
        Returns:
            Combined list of patterns with consensus confidence
        """
        if not patterns:
            return []
        
        # Ensure all patterns have detector information
        for pattern in patterns:
            if "detector_name" not in pattern.metadata:
                detector_name = (
                    getattr(pattern.detection_strategy, "name", None) or 
                    (pattern.detection_strategy.value if pattern.detection_strategy else None) or 
                    "unknown"
                )
                pattern.metadata["detector_name"] = detector_name
        
        # Group similar patterns
        pattern_groups = self._group_similar_patterns(patterns)
        
        # Update detector performance metrics if enabled
        if self.dynamic_weighting:
            self._update_detector_performance(patterns, pattern_groups)
            self._adjust_detector_weights()
        
        # Create consensus patterns
        consensus_patterns = []
        
        # Process each group of similar patterns
        for group_key, group in pattern_groups.items():
            # Skip groups with only one pattern below threshold
            if len(group) == 1 and group[0].confidence < self.consensus_threshold:
                continue
            
            # Calculate weighted confidence
            weighted_confidence = 0.0
            total_weight = 0.0
            bullish_votes = 0
            bearish_votes = 0
            neutral_votes = 0
            detector_contributions = {}
            consolidated_metadata = {}
            best_description = None
            
            # Get best pattern from the group for base attributes
            base_pattern = max(group, key=lambda p: p.confidence)
            
            # Process each pattern in the group
            for pattern in group:
                # Get detector weight
                detector_name = pattern.metadata.get("detector_name", "unknown")
                weight = self.detector_weights.get(detector_name, 1.0)
                
                # Track weighted confidence
                weighted_confidence += pattern.confidence * weight
                total_weight += weight
                
                # Track direction votes
                if pattern.bullish is True:
                    bullish_votes += weight
                elif pattern.bullish is False:
                    bearish_votes += weight
                else:
                    neutral_votes += weight
                
                # Track detector contribution
                detector_contributions[detector_name] = pattern.confidence
                
                # Track best description
                if (not best_description or (pattern.description and 
                        (not best_description or len(pattern.description) > len(best_description)))):
                    best_description = pattern.description
                
                # Merge metadata (collect all non-None values)
                for key, value in pattern.metadata.items():
                    if key not in ('detector_name', 'detection_time_ms') and value is not None:
                        if key not in consolidated_metadata:
                            consolidated_metadata[key] = []
                        if isinstance(value, (int, float, bool, str)):
                            consolidated_metadata[key].append(value)
            
            # Skip if no valid weights
            if total_weight <= 0:
                continue
            
            # Calculate final confidence
            confidence = weighted_confidence / total_weight
            
            # Apply consensus boost
            # The more detectors agree, the higher the confidence
            detector_count = len(set(p.metadata.get("detector_name", "unknown") for p in group))
            consensus_boost = min(0.2, (detector_count - 1) * 0.05)
            confidence = min(1.0, confidence + consensus_boost)
            
            # Determine consensus direction
            direction = None
            if bullish_votes > bearish_votes and bullish_votes > neutral_votes:
                direction = True  # Bullish
            elif bearish_votes > bullish_votes and bearish_votes > neutral_votes:
                direction = False  # Bearish
            # Otherwise leave as None (neutral)
            
            # Finalize metadata by averaging numerical values
            for key, values in consolidated_metadata.items():
                if values and all(isinstance(v, (int, float)) for v in values):
                    consolidated_metadata[key] = sum(values) / len(values)
                else:
                    # For non-numeric, keep most frequent value
                    counter = Counter(values)
                    consolidated_metadata[key] = counter.most_common(1)[0][0] if counter else None
            
            # Add consensus information
            consolidated_metadata.update({
                "consensus_detectors": detector_count,
                "consensus_boost": consensus_boost,
                "detector_contributions": detector_contributions,
                "group_size": len(group),
                "consensus_level": detector_count / len(self.detectors) if self.detectors else 0
            })
            
            # Create consensus pattern
            consensus_pattern = PatternMatch(
                pattern_name=base_pattern.pattern_name,
                confidence=confidence,
                candle_indices=base_pattern.candle_indices,
                bullish=direction,
                description=best_description or base_pattern.description,
                detection_strategy=DetectionStrategy.WEIGHTED_CONSENSUS,
                detection_time_ms=max(p.detection_time_ms for p in group),
                metadata=consolidated_metadata
            )
            
            consensus_patterns.append(consensus_pattern)
        
        # Sort by confidence
        consensus_patterns.sort(key=lambda p: p.confidence, reverse=True)
        
        return consensus_patterns
    
    def _group_similar_patterns(self, patterns: List[PatternMatch]) -> Dict[str, List[PatternMatch]]:
        """
        Group similar patterns together for consensus determination.
        
        This method identifies patterns that refer to the same underlying market pattern,
        even if they have slightly different names or indices.
        
        Args:
            patterns: List of all detected patterns
            
        Returns:
            Dictionary mapping group keys to lists of similar patterns
        """
        if not patterns:
            return {}
            
        # Create groups based on pattern similarity
        pattern_groups: Dict[str, List[PatternMatch]] = {}
        
        for pattern in patterns:
            # Create a group key based on pattern attributes
            # (we consider patterns with overlapping candles and related names as the same)
            group_key = None
            
            # Try to find an existing group for this pattern
            for key, group in pattern_groups.items():
                # Check if this pattern belongs to an existing group
                if self._is_similar_to_group(pattern, group):
                    group_key = key
                    break
            
            # Create a new group if no matching group was found
            if group_key is None:
                # Use pattern name and candle range as the group key
                candle_range = f"{min(pattern.candle_indices)}-{max(pattern.candle_indices)}" if pattern.candle_indices else "none"
                group_key = f"{pattern.pattern_name}_{candle_range}"
                pattern_groups[group_key] = []
            
            # Add pattern to its group
            pattern_groups[group_key].append(pattern)
        
        return pattern_groups
    
    def _is_similar_to_group(self, pattern: PatternMatch, group: List[PatternMatch]) -> bool:
        """
        Check if a pattern is similar to patterns in a group.
        
        Args:
            pattern: Pattern to check
            group: Group of patterns to compare against
            
        Returns:
            True if the pattern is similar to the group, False otherwise
        """
        if not group:
            return False
            
        # Compare with first pattern in group
        reference = group[0]
        
        # Check candle overlap
        candle_overlap = self._get_candle_overlap(pattern, reference)
        if candle_overlap < self.similarity_threshold:
            return False
            
        # Check pattern name similarity
        if not self._are_related_patterns(pattern.pattern_name, reference.pattern_name):
            return False
            
        # Check if bullish/bearish directions match (if both specified)
        if (pattern.bullish is not None and 
            reference.bullish is not None and 
            pattern.bullish != reference.bullish):
            return False
            
        return True
    
    def _get_candle_overlap(self, pattern1: PatternMatch, pattern2: PatternMatch) -> float:
        """
        Calculate the overlap ratio between candle indices of two patterns.
        
        Args:
            pattern1: First pattern
            pattern2: Second pattern
            
        Returns:
            Overlap ratio (0.0-1.0)
        """
        # Handle empty indices
        if not pattern1.candle_indices or not pattern2.candle_indices:
            return 0.0
            
        # Convert to sets for efficient intersection/union
        indices1 = set(pattern1.candle_indices)
        indices2 = set(pattern2.candle_indices)
        
        # Calculate intersection and union
        intersection = indices1.intersection(indices2)
        union = indices1.union(indices2)
        
        # Calculate Jaccard similarity (intersection over union)
        if not union:
            return 0.0
            
        return len(intersection) / len(union)
    
    def _are_related_patterns(self, pattern1: str, pattern2: str) -> bool:
        """
        Check if two pattern names are related.
        
        This handles cases where different detectors use slightly different names
        for the same underlying pattern.
        
        Args:
            pattern1: First pattern name
            pattern2: Second pattern name
            
        Returns:
            True if patterns are related, False otherwise
        """
        # If names are identical, they're related
        if pattern1.lower() == pattern2.lower():
            return True
            
        # Common pattern variations
        pattern_variations = {
            "doji": ["doji", "neutral doji", "doji star"],
            "hammer": ["hammer", "bullish hammer", "hanging man"],
            "shooting star": ["shooting star", "inverted hammer", "bearish inverted hammer"],
            "engulfing": ["bullish engulfing", "bearish engulfing", "engulfing pattern"],
            "harami": ["bullish harami", "bearish harami", "harami pattern"],
            "morning star": ["morning star", "morning star doji", "bullish reversal"],
            "evening star": ["evening star", "evening star doji", "bearish reversal"],
            "marubozu": ["marubozu", "bullish marubozu", "bearish marubozu"],
            "three white soldiers": ["three white soldiers", "three advancing soldiers", "three bullish candles"],
            "three black crows": ["three black crows", "three declining crows", "three bearish candles"],
        }
        
        # Check if patterns belong to same variation group
        p1_lower = pattern1.lower()
        p2_lower = pattern2.lower()
        
        for variations in pattern_variations.values():
            if any(v in p1_lower for v in variations) and any(v in p2_lower for v in variations):
                return True
                
        return False
    
    def _update_detector_performance(
        self, 
        all_patterns: List[PatternMatch], 
        pattern_groups: Dict[str, List[PatternMatch]]
    ) -> None:
        """
        Update detector performance metrics based on consensus results.
        
        This tracks how often each detector agrees with the consensus,
        which is used for dynamic weight adjustment.
        
        Args:
            all_patterns: All patterns from all detectors
            pattern_groups: Grouped patterns for consensus calculation
        """
        # Skip if no patterns or groups
        if not all_patterns or not pattern_groups:
            return
            
        # Initialize detector pattern counts
        detector_pattern_counts = {detector.name: 0 for detector in self.detectors}
        for pattern in all_patterns:
            detector_name = pattern.metadata.get("detector_name", "unknown")
            detector_pattern_counts[detector_name] = detector_pattern_counts.get(detector_name, 0) + 1
        
        # Count consensus matches by detector
        detector_consensus_matches = {detector.name: 0 for detector in self.detectors}
        for group in pattern_groups.values():
            # Only consider groups with multiple patterns
            if len(group) <= 1:
                continue
                
            # Get detectors that contributed to this group
            detectors_in_group = set(
                p.metadata.get("detector_name", "unknown") 
                for p in group
            )
            
            # Update match counts for each detector
            for detector_name in detectors_in_group:
                detector_consensus_matches[detector_name] = detector_consensus_matches.get(detector_name, 0) + 1
        
        # Update detector performance metrics
        for detector_name, pattern_count in detector_pattern_counts.items():
            if detector_name not in self.detector_performance:
                self.detector_performance[detector_name] = {
                    "patterns_detected": 0,
                    "consensus_matches": 0,
                    "average_confidence": 0.0,
                    "total_confidence": 0.0
                }
            
            # Update pattern count
            self.detector_performance[detector_name]["patterns_detected"] += pattern_count
            
            # Update consensus matches
            consensus_matches = detector_consensus_matches.get(detector_name, 0)
            self.detector_performance[detector_name]["consensus_matches"] += consensus_matches
            
            # Update confidence metrics
            detector_patterns = [p for p in all_patterns if p.metadata.get("detector_name", "unknown") == detector_name]
            if detector_patterns:
                avg_confidence = sum(p.confidence for p in detector_patterns) / len(detector_patterns)
                total_confidence = sum(p.confidence for p in detector_patterns)
                
                # Update using exponential moving average (EMA)
                alpha = 0.2  # Weight for new observation
                current_avg = self.detector_performance[detector_name]["average_confidence"]
                if current_avg == 0:
                    # First update
                    self.detector_performance[detector_name]["average_confidence"] = avg_confidence
                else:
                    # EMA update
                    self.detector_performance[detector_name]["average_confidence"] = (
                        alpha * avg_confidence + (1 - alpha) * current_avg
                    )
                
                # Update total confidence
                self.detector_performance[detector_name]["total_confidence"] += total_confidence
    
    def _adjust_detector_weights(self) -> None:
        """
        Adjust detector weights based on performance metrics.
        
        This gives higher weights to detectors that:
        1. More often agree with consensus (reliable)
        2. Detect patterns with higher confidence (decisive)
        3. Have consistently good performance over time
        """
        # Skip if no performance data
        if not self.detector_performance:
            return
            
        # Calculate new weights based on performance metrics
        new_weights = {}
        
        for detector_name, metrics in self.detector_performance.items():
            # Skip detectors with no detections
            if metrics["patterns_detected"] == 0:
                new_weights[detector_name] = self.detector_weights.get(detector_name, 1.0)
                continue
            
            # Calculate consensus rate (agreement with other detectors)
            consensus_rate = (
                metrics["consensus_matches"] / metrics["patterns_detected"] 
                if metrics["patterns_detected"] > 0 else 0
            )
            
            # Get current confidence metrics
            avg_confidence = metrics["average_confidence"]
            
            # Calculate performance score
            # Higher score for higher consensus rate and higher confidence
            performance_score = (0.7 * consensus_rate) + (0.3 * avg_confidence)
            
            # Apply a sigmoid function to map scores to weights
            # This ensures weights stay in a reasonable range
            import math
            sigmoid = lambda x: 1 / (1 + math.exp(-5 * (x - 0.5)))
            
            # Calculate new weight, keeping it between 0.2 and 2.0
            raw_weight = 0.2 + (1.8 * sigmoid(performance_score))
            
            # Smooth weight changes with the existing weight
            current_weight = self.detector_weights.get(detector_name, 1.0)
            smoothed_weight = (0.8 * current_weight) + (0.2 * raw_weight)
            
            new_weights[detector_name] = smoothed_weight
        
        # Normalize weights to sum to 1.0
        total_weight = sum(new_weights.values())
        if total_weight > 0:
            self.detector_weights = {
                name: weight / total_weight 
                for name, weight in new_weights.items()
            }
        else:
            # Fallback to equal weights if normalization fails
            self.detector_weights = {
                name: 1.0 / len(new_weights) 
                for name in new_weights
            }
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the weighted consensus detector.
        
        Returns:
            Dictionary with detector status and weights
        """
        status = super().get_status()
        status.update({
            "consensus_threshold": self.consensus_threshold,
            "similarity_threshold": self.similarity_threshold,
            "dynamic_weighting": self.dynamic_weighting,
            "detector_weights": self.detector_weights.copy(),
            "sub_detectors": [detector.name for detector in self.detectors]
        })
        return status 