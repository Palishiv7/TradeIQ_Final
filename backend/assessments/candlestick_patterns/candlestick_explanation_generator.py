"""
Candlestick Pattern Explanation Generator

This module provides functionality for generating explanations of candlestick patterns
with different levels of detail based on user knowledge level.

This implementation extends the base ExplanationGenerator from the assessment architecture
to provide specialized explanations for candlestick patterns.
"""

from typing import Dict, Any, List, Optional, Tuple, cast
import json
import os
from enum import Enum, auto
from dataclasses import dataclass, field
import random
from functools import lru_cache

# Import from base assessment architecture
from backend.assessments.base.services import ExplanationGenerator as BaseExplanationGenerator
from backend.assessments.base.models import BaseQuestion

# Import other modules
from backend.common.finance.patterns import PatternType
from backend.common.logger import get_logger
from backend.assessments.candlestick_patterns.answer_evaluation import ValidationResult
from backend.assessments.candlestick_patterns.types import UserLevel
from backend.assessments.candlestick_patterns.candlestick_pattern_recognition import CandlestickPatternQuestion

# Set up logger
logger = get_logger(__name__)


class ExplanationFormat(Enum):
    """
    Format types for explanations.
    """
    TEXT = "text"
    HTML = "html"
    MARKDOWN = "markdown"


@dataclass
class PatternExplanation:
    """
    Data model for pattern explanations with different detail levels.
    """
    pattern: str
    name: str
    type: str
    description: Dict[str, str] = field(default_factory=dict)
    formation: Dict[str, str] = field(default_factory=dict)
    psychology: Dict[str, str] = field(default_factory=dict)
    signal_type: str = ""
    reliability: str = ""
    time_frame: str = ""
    examples: List[Dict[str, Any]] = field(default_factory=list)
    related_patterns: List[str] = field(default_factory=list)
    market_statistics: Dict[str, Any] = field(default_factory=dict)
    visual_characteristics: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "pattern": self.pattern,
            "name": self.name,
            "type": self.type,
            "description": self.description,
            "formation": self.formation,
            "psychology": self.psychology,
            "signal_type": self.signal_type,
            "reliability": self.reliability,
            "time_frame": self.time_frame,
            "examples": self.examples,
            "related_patterns": self.related_patterns,
            "market_statistics": self.market_statistics,
            "visual_characteristics": self.visual_characteristics
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PatternExplanation':
        """
        Create from dictionary.
        
        Args:
            data: Dictionary representation
            
        Returns:
            PatternExplanation instance
        """
        return cls(
            pattern=data["pattern"],
            name=data["name"],
            type=data["type"],
            description=data.get("description", {}),
            formation=data.get("formation", {}),
            psychology=data.get("psychology", {}),
            signal_type=data.get("signal_type", ""),
            reliability=data.get("reliability", ""),
            time_frame=data.get("time_frame", ""),
            examples=data.get("examples", []),
            related_patterns=data.get("related_patterns", []),
            market_statistics=data.get("market_statistics", {}),
            visual_characteristics=data.get("visual_characteristics", {})
        )


class ExplanationGenerator(BaseExplanationGenerator):
    """
    Generator for candlestick pattern explanations.
    
    This class provides methods for generating explanations with different levels
    of detail based on user knowledge level and context.
    """
    
    def __init__(self, explanations_path: Optional[str] = None):
        """
        Initialize the explanation generator.
        
        Args:
            explanations_path: Path to explanations JSON file
        """
        # Default explanations path
        if explanations_path is None:
            package_dir = os.path.dirname(os.path.abspath(__file__))
            explanations_path = os.path.join(package_dir, "data", "pattern_explanations.json")
        
        self._explanations_path = explanations_path
        self._explanations = self._load_explanations()
        
        # Load market examples database
        self._market_examples = self._load_market_examples()
        
        # Initialize feedback templates
        self._initialize_feedback_templates()
        
        logger.info(f"ExplanationGenerator initialized with {len(self._explanations)} patterns")
    
    @property
    def explanations(self) -> Dict[str, PatternExplanation]:
        """Get the explanations dictionary."""
        return self._explanations
    
    @property
    def market_examples(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get the market examples dictionary."""
        return self._market_examples
    
    def _initialize_feedback_templates(self) -> None:
        """Initialize templates for feedback generation."""
        self.correct_templates = [
            "Excellent! You correctly identified the {pattern} pattern. {explanation}",
            "That's right! This is a {pattern} pattern. {explanation}",
            "Perfect! You've correctly recognized the {pattern} pattern. {explanation}",
            "Correct! This chart shows a {pattern} pattern. {explanation}"
        ]
        
        self.incorrect_templates = [
            "Not quite. This is actually a {correct_pattern} pattern, not a {user_answer}. {explanation}",
            "The pattern shown is a {correct_pattern}, not a {user_answer}. {explanation}",
            "That's not correct. This chart displays a {correct_pattern} pattern. {explanation}",
            "This is a {correct_pattern} pattern. Your answer ({user_answer}) is incorrect. {explanation}"
        ]
        
        self.partial_templates = [
            "Close! You identified a {user_answer}, which is related to the {correct_pattern} pattern. {explanation}",
            "You're on the right track with {user_answer}, but this is actually a {correct_pattern} pattern. {explanation}",
            "Your answer ({user_answer}) is related to the correct pattern ({correct_pattern}). {explanation}"
        ]
        
        self.hint_templates = [
            "Hint: Look at the relationship between the open and close prices in the key candles.",
            "Hint: Pay attention to the length of the shadows relative to the body.",
            "Hint: Notice the overall trend before this pattern appears.",
            "Hint: Consider the volume changes along with the price action."
        ]
    
    def generate_question_explanation(self, question: BaseQuestion) -> str:
        """
        Generate explanation for a question.
        
        Args:
            question: Question to explain
            
        Returns:
            Question explanation
        """
        try:
            from backend.assessments.candlestick_patterns.candlestick_pattern_recognition import CandlestickPatternQuestion
            
            # Convert to specific question type
            if not isinstance(question, CandlestickPatternQuestion):
                logger.error(f"Expected CandlestickPatternQuestion, got {type(question)}")
                return "Cannot generate explanation for this question type."
            
            # Get pattern and generate explanation
            pattern = question.get_pattern_type()
            if pattern:
                explanation = self.generate_explanation(pattern.value, UserLevel.INTERMEDIATE)
                return explanation.get("detailed", "No explanation available.")
            else:
                return "No pattern information available for this question."
        except Exception as e:
            logger.error(f"Error generating question explanation: {e}")
            return "Unable to generate explanation due to an error."
    
    def generate_answer_explanation(
        self,
        question: BaseQuestion,
        user_answer: Any,
        is_correct: bool,
        validation_result: Optional[ValidationResult] = None
    ) -> Dict[str, Any]:
        """
        Generate explanation for an answer.
        
        Args:
            question: Question that was answered
            user_answer: User's answer
            is_correct: Whether the answer is correct
            validation_result: Optional validation result for enhanced explanations
            
        Returns:
            Dictionary with explanation text and additional data
        """
        try:
            from backend.assessments.candlestick_patterns.candlestick_pattern_recognition import CandlestickPatternQuestion
            
            # Convert to specific question type
            if not isinstance(question, CandlestickPatternQuestion):
                logger.error(f"Expected CandlestickPatternQuestion, got {type(question)}")
                return {"text": "Cannot generate explanation for this question type."}
            
            # Get pattern information
            pattern_info = question.get_pattern_explanation()
            correct_pattern = pattern_info.get("pattern", "unknown")
            
            # Process validation result for enhanced feedback
            confidence = validation_result.confidence if validation_result else 0.0
            explanation_context = self._get_explanation_context(validation_result) if validation_result else {}
            
            # Get base explanation
            base_explanation = self.generate_explanation(correct_pattern, UserLevel.INTERMEDIATE)
            key_points = base_explanation.get("key_points", [])
            
            # Determine the explanation template based on correctness and confidence
            if is_correct:
                # Correct answer
                template = random.choice(self.correct_templates)
                explanation_text = template.format(
                    pattern=correct_pattern,
                    explanation=base_explanation.get("concise", "")
                )
                
                # Add a market example for correct answers
                example = self._get_relevant_market_example(correct_pattern, explanation_context)
                if example:
                    explanation_text += f"\n\nReal market example: {example}"
                
            elif validation_result and validation_result.confidence > 0.4:
                # Partially correct/related answer
                template = random.choice(self.partial_templates)
                explanation_text = template.format(
                    user_answer=user_answer,
                    correct_pattern=correct_pattern,
                    explanation=base_explanation.get("concise", "")
                )
                
                # Add explanation of the difference
                if validation_result.explanation:
                    explanation_text += f"\n\n{validation_result.explanation}"
                
            else:
                # Incorrect answer
                template = random.choice(self.incorrect_templates)
                explanation_text = template.format(
                    user_answer=user_answer,
                    correct_pattern=correct_pattern,
                    explanation=base_explanation.get("concise", "")
                )
                
                # Add a hint for incorrect answers
                hint = random.choice(self.hint_templates)
                explanation_text += f"\n\n{hint}"
            
            # Add key characteristics
            if key_points:
                explanation_text += "\n\nKey characteristics:\n"
                for point in key_points[:3]:  # Limit to 3 points
                    explanation_text += f"- {point}\n"
            
            # Generate a detailed educational component for the correct pattern
            educational_content = self._generate_educational_content(
                correct_pattern, explanation_context, UserLevel.INTERMEDIATE
            )
            
            # Compile the final explanation
            result = {
                "text": explanation_text,
                "is_correct": is_correct,
                "confidence": confidence,
                "pattern": correct_pattern,
                "user_answer": user_answer,
                "educational_content": educational_content,
                "key_points": key_points,
                "visual_cues": base_explanation.get("visual_cues", [])
            }
            
            return result
        except Exception as e:
            logger.error(f"Error generating answer explanation: {e}")
            return {
                "text": f"The correct pattern is {correct_pattern}.",
                "is_correct": is_correct
            }
    
    def _get_explanation_context(self, validation_result: ValidationResult) -> Dict[str, Any]:
        """
        Extract context from validation result for enhanced explanations.
        
        Args:
            validation_result: The validation result
            
        Returns:
            Context dictionary for explanation generation
        """
        context = {}
        
        if validation_result:
            # Extract context features
            context = validation_result.context_features.copy() if validation_result.context_features else {}
            
            # Add validation metadata
            context["confidence"] = validation_result.confidence
            context["validation_tier"] = validation_result.validation_tier.value
            context["alternative_patterns"] = validation_result.alternative_patterns
            
        return context
    
    def _load_market_examples(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Load market examples database.
        
        Returns:
            Dictionary mapping pattern names to examples
        """
        try:
            # Try to load from file
            package_dir = os.path.dirname(os.path.abspath(__file__))
            examples_path = os.path.join(package_dir, "data", "market_examples.json")
            
            if os.path.exists(examples_path):
                with open(examples_path, "r") as f:
                    return json.load(f)
            
            # Default examples if file doesn't exist
            logger.info("Market examples file not found, creating default examples")
            return self._create_default_market_examples()
            
        except Exception as e:
            logger.error(f"Error loading market examples: {e}")
            return {}
    
    def _create_default_market_examples(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Create default market examples.
        
        Returns:
            Dictionary mapping pattern names to examples
        """
        examples = {
            "doji": [
                {
                    "symbol": "AAPL",
                    "date": "2022-05-15",
                    "description": "Classic doji showing market indecision after an uptrend in Apple stock",
                    "outcome": "Followed by a short-term reversal"
                }
            ],
            "hammer": [
                {
                    "symbol": "MSFT",
                    "date": "2022-03-20",
                    "description": "Bullish hammer at support level for Microsoft stock",
                    "outcome": "Started a 15% rally over the next two weeks"
                }
            ],
            "bullish_engulfing": [
                {
                    "symbol": "SPY",
                    "date": "2022-01-24",
                    "description": "Strong bullish engulfing pattern at market bottom for S&P 500 ETF",
                    "outcome": "Marked the beginning of a recovery rally"
                }
            ]
        }
        
        # Try to save for future use
        try:
            package_dir = os.path.dirname(os.path.abspath(__file__))
            os.makedirs(os.path.join(package_dir, "data"), exist_ok=True)
            examples_path = os.path.join(package_dir, "data", "market_examples.json")
            
            with open(examples_path, "w") as f:
                json.dump(examples, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving default market examples: {e}")
            
        return examples
    
    def _get_relevant_market_example(self, pattern: str, context: Dict[str, Any]) -> Optional[str]:
        """
        Get a relevant market example for the pattern.
        
        Args:
            pattern: Pattern name
            context: Context information
            
        Returns:
            Formatted market example string or None
        """
        try:
            pattern_lower = pattern.lower().replace(" ", "_")
            examples = self._market_examples.get(pattern_lower, [])
            
            if not examples:
                return None
            
            # Select a random example
            example = random.choice(examples)
            
            # Format the example
            text = (
                f"{example['symbol']} on {example['date']} - {example['description']}. "
                f"Outcome: {example['outcome']}."
            )
            
            return text
        except Exception as e:
            logger.error(f"Error getting market example for {pattern}: {e}")
            return None
    
    def generate_correct_answer_explanation(self, question: BaseQuestion) -> str:
        """
        Generate explanation for the correct answer.
        
        Args:
            question: Question to explain
            
        Returns:
            Correct answer explanation
        """
        try:
            from backend.assessments.candlestick_patterns.candlestick_pattern_recognition import CandlestickPatternQuestion
            
            # Convert to specific question type
            if not isinstance(question, CandlestickPatternQuestion):
                logger.error(f"Expected CandlestickPatternQuestion, got {type(question)}")
                return "Cannot generate explanation for this question type."
            
            # Get pattern and generate explanation
            pattern = question.get_pattern_type()
            if pattern:
                explanation = self.generate_explanation(pattern.value, UserLevel.INTERMEDIATE)
                return explanation.get("detailed", "No explanation available.")
            else:
                return "No pattern information available for this question."
        except Exception as e:
            logger.error(f"Error generating correct answer explanation: {e}")
            return "Unable to generate explanation due to an error."
    
    @lru_cache(maxsize=32)
    def generate_explanation(
        self, 
        pattern: str, 
        user_level: UserLevel = UserLevel.INTERMEDIATE
    ) -> Dict[str, Any]:
        """
        Generate explanation for a pattern.
        
        Args:
            pattern: Pattern to explain
            user_level: User knowledge level
            
        Returns:
            Dictionary with different levels of explanation detail
        """
        try:
            # Normalize pattern name
            pattern_lower = pattern.lower().replace("-", " ")
            
            # Try to get explanation from loaded data
            explanation = self._get_pattern_explanation(pattern_lower)
            
            if explanation:
                # Format explanation based on user level
                return self._format_explanation(explanation, user_level)
            else:
                # Generate default explanation if not found
                logger.info(f"No explanation found for {pattern}, generating default")
                default_explanation = self._generate_default_explanation(pattern_lower, user_level)
                return default_explanation
        except Exception as e:
            logger.error(f"Error generating explanation for {pattern}: {e}")
            return {
                "concise": f"Information about {pattern} pattern.",
                "detailed": f"Information about {pattern} pattern is currently unavailable.",
                "key_points": []
            }

    def _generate_educational_content(
        self, 
        pattern: str, 
        context: Dict[str, Any],
        user_level: UserLevel
    ) -> Dict[str, Any]:
        """
        Generate educational content based on pattern and context.
        
        Args:
            pattern: Pattern name
            context: Context information (market conditions, etc.)
            user_level: User knowledge level
            
        Returns:
            Dictionary with educational content
        """
        try:
            pattern_lower = pattern.lower().replace("-", " ")
            explanation = self._get_pattern_explanation(pattern_lower)
            
            if not explanation:
                explanation = self._generate_default_explanation_data(pattern_lower)
            
            # Base content from explanation
            content = {
                "what_is_it": explanation.description.get(user_level.value, ""),
                "how_to_identify": explanation.formation.get(user_level.value, ""),
                "market_psychology": explanation.psychology.get(user_level.value, ""),
                "trading_implications": "",
                "success_rate": explanation.reliability,
                "common_mistakes": "",
                "related_patterns": explanation.related_patterns
            }
            
            # Add trading implications based on pattern type
            if explanation.signal_type.lower() == "reversal":
                content["trading_implications"] = (
                    f"The {pattern} is typically a {explanation.signal_type.lower()} pattern. "
                    f"Traders often use it to identify potential trend changes. "
                    f"Consider confirming with volume and other indicators before trading."
                )
            elif explanation.signal_type.lower() == "continuation":
                content["trading_implications"] = (
                    f"The {pattern} is typically a {explanation.signal_type.lower()} pattern. "
                    f"It suggests the current trend is likely to continue. "
                    f"This can be used to add to existing positions or initiate new ones in the trend direction."
                )
            else:
                content["trading_implications"] = (
                    f"The {pattern} can provide insights about market psychology and potential price movements. "
                    f"Always use additional confirmation before making trading decisions based on this pattern."
                )
            
            # Add common mistakes based on context
            if "prior_trend" in context:
                trend = context.get("prior_trend", "")
                if trend == "uptrend":
                    content["common_mistakes"] = (
                        f"A common mistake is confusing this pattern with similar patterns that appear in uptrends. "
                        f"Always consider the prior trend when identifying patterns."
                    )
                elif trend == "downtrend":
                    content["common_mistakes"] = (
                        f"A common mistake is confusing this pattern with similar patterns that appear in downtrends. "
                        f"Always consider the prior trend when identifying patterns."
                    )
                else:
                    content["common_mistakes"] = (
                        f"A common mistake is focusing only on the pattern itself without considering the market context. "
                        f"Always look at the bigger picture including trend, support/resistance, and volume."
                    )
            else:
                content["common_mistakes"] = (
                    f"A common mistake is isolating the pattern from overall market context. "
                    f"Always confirm {pattern} patterns with other technical indicators and market conditions."
                )
            
            # Add visual cues for pattern identification
            content["visual_cues"] = self._generate_visual_cues(pattern_lower, explanation)
            
            return content
        except Exception as e:
            logger.error(f"Error generating educational content for {pattern}: {e}")
            return {
                "what_is_it": f"Information about {pattern} pattern.",
                "how_to_identify": "Look for the distinctive candlestick formation.",
                "common_mistakes": "Always confirm patterns with other indicators."
            }
    
    def _generate_visual_cues(self, pattern: str, explanation: PatternExplanation) -> List[str]:
        """
        Generate visual cues for pattern identification.
        
        Args:
            pattern: Pattern name
            explanation: Pattern explanation data
            
        Returns:
            List of visual cues
        """
        try:
            cues = []
            
            # Get visual characteristics from explanation
            char = explanation.visual_characteristics
            
            if char:
                # Use existing characteristics
                for level, description in char.items():
                    if description:
                        cues.append(description)
                
                return cues[:3]  # Limit to 3 cues
            
            # Default cues based on pattern type
            if "doji" in pattern:
                cues = [
                    "Look for a candle with very small or no body (open and close are at nearly the same level)",
                    "Upper and lower shadows may vary in length but are often significant",
                    "The overall shape resembles a cross or plus sign"
                ]
            elif "engulfing" in pattern:
                cues = [
                    "Look for two candles where the second candle's body completely engulfs the first candle's body",
                    "The colors of the two candles should be opposite (bullish: first red, second green; bearish: first green, second red)",
                    "The larger the second candle, the stronger the signal"
                ]
            elif "hammer" in pattern:
                cues = [
                    "Look for a small body at the upper end of the trading range",
                    "The lower shadow should be at least twice the size of the body",
                    "The upper shadow should be very small or non-existent"
                ]
            elif "shooting star" in pattern:
                cues = [
                    "Look for a small body at the lower end of the trading range",
                    "The upper shadow should be at least twice the size of the body",
                    "The lower shadow should be very small or non-existent"
                ]
            elif "star" in pattern:
                cues = [
                    "Look for a pattern of three candles with a small-bodied middle candle",
                    "The middle candle should show a gap from the first candle's close",
                    "The third candle should confirm the reversal with a strong move in the opposite direction"
                ]
            else:
                # Generic cues
                cues = [
                    f"Identify the key components of the {pattern} pattern in the price action",
                    "Consider the position of each candle relative to the others",
                    "Pay attention to the size relationship between bodies and shadows"
                ]
            
            return cues
        except Exception as e:
            logger.error(f"Error generating visual cues for {pattern}: {e}")
            return [
                "Look for the distinctive candlestick formation",
                "Consider the overall market context",
                "Confirm with other technical indicators"
            ]

    def _load_explanations(self) -> Dict[str, PatternExplanation]:
        """
        Load pattern explanations from file.
        
        Returns:
            Dictionary of pattern explanations
        """
        explanations = {}
        
        try:
            # Check if file exists
            if os.path.exists(self._explanations_path):
                with open(self._explanations_path, "r") as f:
                    data = json.load(f)
                
                # Convert to PatternExplanation objects
                for pattern_data in data:
                    explanation = PatternExplanation.from_dict(pattern_data)
                    explanations[explanation.pattern] = explanation
                
                logger.info(f"Loaded {len(explanations)} explanations from {self._explanations_path}")
            else:
                # Create explanations file with defaults
                logger.info(f"Explanations file not found, creating default file at {self._explanations_path}")
                self._create_default_explanations_file()
        except Exception as e:
            logger.error(f"Error loading explanations: {e}")
        
        return explanations
    
    def _create_default_explanations_file(self) -> None:
        """Create a default explanations file."""
        try:
            # Create directory if needed
            os.makedirs(os.path.dirname(self._explanations_path), exist_ok=True)
            
            # Generate default explanations
            default_explanations = []
            for pattern in PatternType:
                explanation = self._generate_default_explanation_data(pattern.name)
                default_explanations.append(explanation.to_dict())
            
            # Save to file
            with open(self._explanations_path, "w") as f:
                json.dump(default_explanations, f, indent=2)
                
            logger.info(f"Created default explanations file with {len(default_explanations)} patterns")
        except Exception as e:
            logger.error(f"Error creating default explanations file: {e}")
    
    def _get_pattern_explanation(self, pattern: str) -> Optional[PatternExplanation]:
        """
        Get explanation for a pattern.
        
        Args:
            pattern: Pattern name
            
        Returns:
            PatternExplanation instance if found, None otherwise
        """
        return self._explanations.get(pattern)
    
    def _generate_default_explanation_data(self, pattern: str) -> PatternExplanation:
        """
        Generate default explanation data for a pattern.
        
        Args:
            pattern: Pattern name
            
        Returns:
            PatternExplanation instance
        """
        try:
            # Get pattern type
            pattern_type = PatternType[pattern]
            
            # Determine signal type
            if pattern_type.is_bullish():
                signal_type = "Bullish"
            elif pattern_type.is_bearish():
                signal_type = "Bearish"
            else:
                signal_type = "Neutral"
            
            # Format name
            name = pattern.replace("_", " ").title()
            
            # Determine pattern category
            if "_" not in pattern and "STAR" not in pattern:
                category = "Single Candlestick Pattern"
            elif "MORNING_STAR" in pattern or "EVENING_STAR" in pattern:
                category = "Triple Candlestick Pattern"
            else:
                category = "Double Candlestick Pattern"
            
            # Create default descriptions
            descriptions = {
                "beginner": f"{name} is a {signal_type.lower()} {category.lower()}.",
                "intermediate": f"{name} is a {signal_type.lower()} {category.lower()} that indicates a potential reversal in price direction.",
                "advanced": f"{name} is a {signal_type.lower()} {category.lower()} that indicates a potential reversal in price direction, often seen after a sustained trend."
            }
            
            # Create default formation descriptions
            formations = {
                "beginner": f"The {name.lower()} pattern forms when specific candle shapes appear in sequence.",
                "intermediate": f"The {name.lower()} pattern forms when specific price action creates distinctive candle shapes.",
                "advanced": f"The {name.lower()} pattern forms through specific price action that reflects changing market sentiment."
            }
            
            # Create default psychology descriptions
            psychology = {
                "beginner": f"This pattern shows a shift in market sentiment from {'bearish to bullish' if pattern_type.is_bullish() else 'bullish to bearish' if pattern_type.is_bearish() else 'one direction to another'}.",
                "intermediate": f"The {name.lower()} pattern represents a psychological shift in market participants' sentiment, indicating {'buying pressure overwhelming selling pressure' if pattern_type.is_bullish() else 'selling pressure overwhelming buying pressure' if pattern_type.is_bearish() else 'a balance between buyers and sellers'}.",
                "advanced": f"This pattern psychologically represents {'buyers gaining control from sellers' if pattern_type.is_bullish() else 'sellers gaining control from buyers' if pattern_type.is_bearish() else 'indecision in the market'}, often at key price levels where market participants are reassessing value."
            }
            
            # Create explanation
            explanation = PatternExplanation(
                pattern=pattern,
                name=name,
                type=category,
                description=descriptions,
                formation=formations,
                psychology=psychology,
                signal_type=signal_type,
                reliability="Medium",
                time_frame="All time frames",
                examples=[],
                related_patterns=[]
                )
            
            return explanation
        except Exception as e:
            logger.error(f"Error generating default explanation data for {pattern}: {e}")
            # Return a minimal fallback explanation
            return PatternExplanation(
                pattern=pattern,
                name=pattern.replace("_", " ").title(),
                type="Candlestick Pattern",
                description={"intermediate": f"Information about {pattern} pattern."},
                formation={"intermediate": "Forms under specific market conditions."},
                psychology={"intermediate": "Represents a specific market sentiment."},
                signal_type="Neutral",
                reliability="Medium",
                time_frame="All time frames"
            )
    
    def _generate_default_explanation(
        self, 
        pattern: str, 
        user_level: UserLevel
    ) -> Dict[str, Any]:
        """
        Generate a default explanation for a pattern.
        
        Args:
            pattern: Pattern name
            user_level: User knowledge level
            
        Returns:
            Pattern explanation data
        """
        # Generate default explanation data
        explanation = self._generate_default_explanation_data(pattern)
            
        # Format based on user level
        return self._format_explanation(explanation, user_level)
    
    def _format_explanation(
        self, 
        explanation: PatternExplanation, 
        user_level: UserLevel
    ) -> Dict[str, Any]:
        """
        Format an explanation based on user level.
        
        Args:
            explanation: Pattern explanation
            user_level: User knowledge level
            
        Returns:
            Formatted explanation data
        """
        try:
            # Convert user level to string
            level_str = user_level.value
            
            # Get appropriate descriptions for user level
            description = explanation.description.get(level_str, explanation.description.get("intermediate", ""))
            formation = explanation.formation.get(level_str, explanation.formation.get("intermediate", ""))
            psychology = explanation.psychology.get(level_str, explanation.psychology.get("intermediate", ""))
            
            # Determine examples to include
            if user_level == UserLevel.BEGINNER:
                # Include only basic examples
                examples = explanation.examples[:1] if explanation.examples else []
            elif user_level == UserLevel.INTERMEDIATE:
                # Include a few examples
                examples = explanation.examples[:2] if explanation.examples else []
            else:
                # Include all examples
                examples = explanation.examples
            
            # Generate concise and detailed explanations
            concise = f"{description} {formation}"
            detailed = (
                f"{description}\n\n"
                f"Formation: {formation}\n\n"
                f"Market Psychology: {psychology}\n\n"
                f"Signal Type: {explanation.signal_type}\n"
                f"Reliability: {explanation.reliability}\n"
                f"Timeframe: {explanation.time_frame}"
            )
            
            # Generate key points
            key_points = [
                f"The {explanation.name} is a {explanation.signal_type.lower()} pattern.",
                f"{formation}",
                f"It is typically seen on {explanation.time_frame.lower()} charts."
            ]
            
            # Generate visual cues
            visual_cues = self._generate_visual_cues(explanation.pattern, explanation)
            
            # Create formatted explanation
            formatted = {
                "pattern": explanation.pattern,
                "name": explanation.name,
                "type": explanation.type,
                "description": description,
                "formation": formation,
                "psychology": psychology,
                "signal_type": explanation.signal_type,
                "reliability": explanation.reliability,
                "time_frame": explanation.time_frame,
                "examples": examples,
                "related_patterns": explanation.related_patterns,
                "user_level": level_str,
                "concise": concise,
                "detailed": detailed,
                "key_points": key_points,
                "visual_cues": visual_cues
            }
            
            return formatted
        except Exception as e:
            logger.error(f"Error formatting explanation: {e}")
            return {
                "pattern": explanation.pattern,
                "name": explanation.name,
                "concise": f"Information about {explanation.name} pattern.",
                "detailed": f"Information about {explanation.name} pattern. Please consult technical analysis resources for more details.",
                "key_points": ["Always confirm patterns with other indicators."],
                "user_level": user_level.value
            }

    async def explain_question(self, question: CandlestickPatternQuestion) -> str:
        """
        Generate an explanation of the question itself.
        
        Args:
            question: The question to explain
            
        Returns:
            Detailed explanation of the question
            
        Raises:
            ExplanationError: If explanation generation fails
        """
        try:
            # Get pattern and generate explanation
            pattern = question.correct_answer
            if not pattern:
                return "Unable to explain question: missing pattern information"
            
            # Get explanation for the pattern
            explanation = self.generate_explanation(pattern, UserLevel.INTERMEDIATE)
            
            # Format question explanation
            question_text = (
                f"This question asks you to identify a {explanation['type']} pattern. "
                f"\n\nThe pattern you're looking for typically {explanation['formation']}. "
                f"\n\nKey points to consider:\n"
            )
            
            # Add key points
            for point in explanation.get('key_points', []):
                question_text += f"- {point}\n"
            
            # Add visual cues if available
            if explanation.get('visual_cues'):
                question_text += "\nVisual cues to look for:\n"
                for cue in explanation['visual_cues']:
                    question_text += f"- {cue}\n"
            
            return question_text
            
        except Exception as e:
            logger.error(f"Error explaining question: {e}")
            return "Unable to generate question explanation due to an error."

    async def explain_correct_answer(self, question: CandlestickPatternQuestion) -> str:
        """
        Generate an explanation of the correct answer.
        
        Args:
            question: The question to explain the answer for
            
        Returns:
            Detailed explanation of the correct answer
            
        Raises:
            ExplanationError: If explanation generation fails
        """
        try:
            pattern = question.correct_answer
            if not pattern:
                return "Unable to explain answer: missing pattern information"
            
            # Get explanation for the pattern
            explanation = self.generate_explanation(pattern, UserLevel.INTERMEDIATE)
            
            # Format answer explanation
            answer_text = (
                f"The correct answer is {pattern}.\n\n"
                f"{explanation.get('detailed', '')}\n\n"
                f"Key characteristics:\n"
            )
            
            # Add key points
            for point in explanation.get('key_points', []):
                answer_text += f"- {point}\n"
            
            # Add market psychology if available
            if explanation.get('psychology'):
                answer_text += f"\nMarket Psychology:\n{explanation['psychology']}"
            
            # Add reliability information
            if explanation.get('reliability'):
                answer_text += f"\n\nReliability: {explanation['reliability']}"
            
            return answer_text
            
        except Exception as e:
            logger.error(f"Error explaining correct answer: {e}")
            return "Unable to generate answer explanation due to an error."

    async def explain_user_answer(
        self,
        question: CandlestickPatternQuestion,
        user_answer: Any,
        is_correct: bool,
        user_level: UserLevel = UserLevel.INTERMEDIATE
    ) -> str:
        """
        Generate an explanation of the user's answer.
        
        Args:
            question: The question that was answered
            user_answer: The user's answer
            is_correct: Whether the user's answer was correct
            user_level: The user's knowledge level
            
        Returns:
            Detailed explanation of why the user's answer was correct or incorrect
            
        Raises:
            ExplanationError: If explanation generation fails
        """
        try:
            pattern = question.correct_answer
            if not pattern:
                return "Unable to explain answer: missing pattern information"
            
            # Get explanations for both the correct pattern and user's answer
            correct_explanation = self.generate_explanation(pattern, user_level)
            user_pattern = str(user_answer).strip()
            
            if is_correct:
                explanation = (
                    f"Correct! You identified the {pattern} pattern.\n\n"
                    f"{correct_explanation.get('concise', '')}\n\n"
                    f"Key characteristics you correctly identified:\n"
                )
                
                # Add key points
                for point in correct_explanation.get('key_points', []):
                    explanation += f"- {point}\n"
                    
            else:
                # Try to get explanation for user's answer
                user_explanation = self.generate_explanation(user_pattern, user_level)
                
                explanation = (
                    f"Your answer '{user_pattern}' is incorrect. "
                    f"The correct pattern is {pattern}.\n\n"
                    f"The {pattern} pattern {correct_explanation.get('formation', '')}\n\n"
                    f"Key differences to note:\n"
                )
                
                # Compare characteristics
                correct_points = set(correct_explanation.get('key_points', []))
                user_points = set(user_explanation.get('key_points', []))
                
                # Add differences
                for point in correct_points - user_points:
                    explanation += f"- {point}\n"
                
                # Add learning suggestion
                explanation += (
                    f"\nTo improve, focus on these aspects of the {pattern} pattern:\n"
                    f"- {correct_explanation.get('formation', '')}\n"
                    f"- {correct_explanation.get('psychology', '')}"
                )
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error explaining user answer: {e}")
            return "Unable to generate answer explanation due to an error."

    async def explain_topic(
        self,
        topic: str,
        difficulty: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate an explanation of a topic.
        
        Args:
            topic: The topic to explain
            difficulty: Optional difficulty level to tailor the explanation
            
        Returns:
            Dictionary containing topic explanation and related resources
            
        Raises:
            ExplanationError: If explanation generation fails
        """
        try:
            # Convert difficulty to user level
            user_level = UserLevel.INTERMEDIATE
            if difficulty:
                if difficulty.lower() in ['beginner', 'basic']:
                    user_level = UserLevel.BEGINNER
                elif difficulty.lower() in ['advanced', 'expert']:
                    user_level = UserLevel.ADVANCED
            
            # Get all patterns related to the topic
            related_patterns = []
            for pattern, explanation in self._explanations.items():
                if topic.lower() in pattern.lower() or topic.lower() in explanation.type.lower():
                    related_patterns.append(explanation)
            
            if not related_patterns:
                return {
                    "topic": topic,
                    "explanation": f"No specific information available for topic: {topic}",
                    "patterns": [],
                    "resources": []
                }
            
            # Generate topic explanation
            topic_explanation = (
                f"Topic: {topic}\n\n"
                f"This topic includes the following candlestick patterns:\n"
            )
            
            # Add pattern summaries
            patterns_info = []
            for pattern in related_patterns:
                explanation = self.generate_explanation(pattern.pattern, user_level)
                patterns_info.append({
                    "name": pattern.name,
                    "type": pattern.type,
                    "description": explanation.get('concise', ''),
                    "reliability": pattern.reliability
                })
                topic_explanation += f"\n- {pattern.name}: {explanation.get('concise', '')}"
            
            # Generate learning resources
            resources = [
                {
                    "title": "Pattern Recognition Guide",
                    "type": "guide",
                    "description": f"Learn how to identify {topic} patterns"
                },
                {
                    "title": "Market Psychology",
                    "type": "article",
                    "description": f"Understanding the psychology behind {topic} patterns"
                },
                {
                    "title": "Practice Exercises",
                    "type": "exercises",
                    "description": f"Interactive exercises for {topic} pattern recognition"
                }
            ]
            
            return {
                "topic": topic,
                "explanation": topic_explanation,
                "patterns": patterns_info,
                "resources": resources
            }
            
        except Exception as e:
            logger.error(f"Error explaining topic: {e}")
            return {
                "topic": topic,
                "explanation": f"Unable to generate explanation for topic: {topic}",
                "patterns": [],
                "resources": []
            }

    async def generate_learning_resources(
        self,
        question: CandlestickPatternQuestion,
        was_correct: bool
    ) -> List[Dict[str, Any]]:
        """
        Generate learning resources related to a question.
        
        Args:
            question: The question to generate resources for
            was_correct: Whether the user answered correctly
            
        Returns:
            List of learning resources with titles and links/content
            
        Raises:
            ExplanationError: If resource generation fails
        """
        try:
            pattern = question.correct_answer
            if not pattern:
                return []
            
            # Get pattern explanation
            explanation = self.generate_explanation(pattern, UserLevel.INTERMEDIATE)
            
            # Base resources
            resources = [
                {
                    "title": f"Understanding the {pattern} Pattern",
                    "type": "article",
                    "description": explanation.get('detailed', ''),
                    "difficulty": "intermediate"
                }
            ]
            
            # Add specific resources based on correctness
            if not was_correct:
                # Add more detailed resources for incorrect answers
                resources.extend([
                    {
                        "title": f"Common Mistakes in {pattern} Recognition",
                        "type": "guide",
                        "description": "Learn how to avoid common identification errors",
                        "difficulty": "beginner"
                    },
                    {
                        "title": "Pattern Recognition Practice",
                        "type": "exercise",
                        "description": f"Interactive exercises focusing on {pattern} identification",
                        "difficulty": "intermediate"
                    }
                ])
            
            # Add advanced resources
            if explanation.get('related_patterns'):
                resources.append({
                    "title": "Related Pattern Analysis",
                    "type": "comparison",
                    "description": f"Compare {pattern} with similar patterns: " + 
                                 ", ".join(explanation['related_patterns']),
                    "difficulty": "advanced"
                })
            
            # Add market examples if available
            if self._market_examples.get(pattern):
                resources.append({
                    "title": "Real Market Examples",
                    "type": "examples",
                    "description": f"Historical examples of {pattern} in action",
                    "examples": self._market_examples[pattern][:3],  # Limit to 3 examples
                    "difficulty": "intermediate"
                })
            
            return resources
            
        except Exception as e:
            logger.error(f"Error generating learning resources: {e}")
            return [
                {
                    "title": "Basic Pattern Guide",
                    "type": "guide",
                    "description": "General guide to candlestick pattern recognition",
                    "difficulty": "beginner"
                }
            ] 