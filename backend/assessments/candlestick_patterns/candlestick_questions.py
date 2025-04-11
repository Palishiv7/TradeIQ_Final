"""
Candlestick Pattern Question Generation Module

This module provides functionality for generating questions about candlestick patterns
based on templates, user performance metrics, and pattern information.
"""

import json
import random
import hashlib
import time
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple, Set, TypeVar, Generic, ClassVar
from enum import Enum, auto
from datetime import datetime
from pydantic import BaseModel, Field
from bitarray import bitarray
import asyncio

from backend.common.logger import app_logger
from backend.common.cache import cache, async_cached
from backend.common.serialization import SerializableMixin
# Import existing base class for difficulty levels
from backend.assessments.base.models import QuestionDifficulty
from backend.assessments.candlestick_patterns.candlestick_config import (
    CANDLESTICK_PATTERNS, DIFFICULTY_LEVELS, ASSESSMENT_CONFIG, PatternCategory
)
from backend.assessments.candlestick_patterns.candlestick_utils import (
    CandlestickData, get_patterns_by_difficulty
)

# Module logger
logger = app_logger.getChild("candlestick_questions")

# Type variable for generics
T = TypeVar('T')

class QuestionType(str, Enum):
    """Enum for question types."""
    IDENTIFICATION = "identification"  # Identify the pattern
    PREDICTION = "prediction"  # Predict price movement
    CHARACTERISTIC = "characteristic"  # Identify characteristics of a pattern
    CONTEXT = "context"  # Identify market conditions
    COMPARISON = "comparison"  # Compare different patterns

class QuestionFormat(str, Enum):
    """Enum for question format."""
    MULTIPLE_CHOICE = "multiple_choice"
    TRUE_FALSE = "true_false"
    FILL_IN_BLANK = "fill_in_blank"
    MATCH_PAIRS = "match_pairs"

class QuestionTemplate(SerializableMixin):
    """Model for a question template."""
    
    def __init__(
        self,
        id: str,
        text: str,
        difficulty: QuestionDifficulty,
        question_type: QuestionType,
        format: QuestionFormat,
        requires_variables: List[str],
        tags: List[str] = []
    ):
        """Initialize a question template."""
        self.id = id
        self.text = text
        self.difficulty = difficulty
        self.question_type = question_type
        self.format = format
        self.requires_variables = requires_variables
        self.tags = tags
    
    def format_question(self, variables: Dict[str, Any]) -> str:
        """Format the question template with given variables."""
        formatted_text = self.text
        for var_name, var_value in variables.items():
            placeholder = f"{{{var_name}}}"
            if placeholder in formatted_text:
                formatted_text = formatted_text.replace(placeholder, str(var_value))
        return formatted_text
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "text": self.text,
            "difficulty": self.difficulty.value,
            "question_type": self.question_type.value,
            "format": self.format.value,
            "requires_variables": self.requires_variables,
            "tags": self.tags
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QuestionTemplate':
        """Create a template from a dictionary."""
        return cls(
            id=data["id"],
            text=data["text"],
            difficulty=QuestionDifficulty(data["difficulty"]),
            question_type=QuestionType(data["question_type"]),
            format=QuestionFormat(data["format"]),
            requires_variables=data["requires_variables"],
            tags=data.get("tags", [])
        )
    
    @property
    def serializable_fields(self) -> List[str]:
        """Get a list of fields that should be serialized."""
        return ["id", "text", "difficulty", "question_type", "format", "requires_variables", "tags"]


class TemplateIndex(Generic[T]):
    """Generic index for templates."""
    
    def __init__(self):
        """Initialize the index."""
        self.items: Dict[Any, List[T]] = {}
    
    def add(self, key: Any, item: T) -> None:
        """Add an item to the index."""
        if key not in self.items:
            self.items[key] = []
        self.items[key].append(item)
    
    def get(self, key: Any) -> List[T]:
        """Get items for a key."""
        return self.items.get(key, [])
    
    def keys(self) -> List[Any]:
        """Get all keys in the index."""
        return list(self.items.keys())
    
    def clear(self) -> None:
        """Clear the index."""
        self.items.clear()

class QuestionTemplateDatabase:
    """Database for question templates with efficient indexing."""
    
    def __init__(self):
        """Initialize the template database."""
        self.templates: Dict[str, QuestionTemplate] = {}
        self.by_difficulty = TemplateIndex[str]()
        self.by_type = TemplateIndex[str]()
        self.by_format = TemplateIndex[str]()
        self.by_tag = TemplateIndex[str]()
        
        # Load the predefined templates
        self._load_templates()
    
    def _load_templates(self):
        """Load predefined question templates."""
        try:
            # Load templates from our predefined list
            self._load_predefined_templates()
            
            logger.info(f"Loaded {len(self.templates)} question templates")
            for difficulty in QuestionDifficulty:
                count = len(self.by_difficulty.get(difficulty))
                logger.info(f"- {difficulty.value}: {count} templates")
        except Exception as e:
            logger.error(f"Error loading templates: {str(e)}")
            # Initialize with minimal templates as fallback
            self._initialize_minimal_templates()
    
    def _load_predefined_templates(self):
        """Load predefined question templates from configuration."""
        # Basic identification templates
        self._add_template(
            id="basic_identify_1",
            text="What candlestick pattern is shown in this chart?",
            difficulty=QuestionDifficulty.EASY,
            question_type=QuestionType.IDENTIFICATION,
            format=QuestionFormat.MULTIPLE_CHOICE,
            requires_variables=["pattern_name", "chart_data"],
            tags=["identification", "basic"]
        )

        self._add_template(
            id="basic_predict_1",
            text="Based on the {pattern_name} pattern shown, what is the most likely price movement to follow?",
            difficulty=QuestionDifficulty.EASY,
            question_type=QuestionType.PREDICTION,
            format=QuestionFormat.MULTIPLE_CHOICE,
            requires_variables=["pattern_name", "chart_data", "expected_movement"],
            tags=["prediction", "basic"]
        )

        # Intermediate templates
        self._add_template(
            id="intermediate_characteristic_1",
            text="Which of the following characteristics is essential for a valid {pattern_name} pattern?",
            difficulty=QuestionDifficulty.MEDIUM,
            question_type=QuestionType.CHARACTERISTIC,
            format=QuestionFormat.MULTIPLE_CHOICE,
            requires_variables=["pattern_name", "characteristics"],
            tags=["characteristics", "intermediate"]
        )

        self._add_template(
            id="intermediate_context_1",
            text="In what market context is a {pattern_name} pattern most significant?",
            difficulty=QuestionDifficulty.MEDIUM,
            question_type=QuestionType.CONTEXT,
            format=QuestionFormat.MULTIPLE_CHOICE,
            requires_variables=["pattern_name", "market_context"],
            tags=["context", "intermediate"]
        )

        # Advanced templates
        self._add_template(
            id="advanced_comparison_1",
            text="Compare the {pattern_name1} and {pattern_name2} patterns. Which statement is correct?",
            difficulty=QuestionDifficulty.HARD,
            question_type=QuestionType.COMPARISON,
            format=QuestionFormat.MULTIPLE_CHOICE,
            requires_variables=["pattern_name1", "pattern_name2", "comparison_points"],
            tags=["comparison", "advanced"]
        )

    def _initialize_minimal_templates(self):
        """Initialize with minimal set of templates as fallback."""
        # Clear existing templates
        self.templates.clear()
        self.by_difficulty.clear()
        self.by_type.clear()
        self.by_format.clear()
        self.by_tag.clear()

        # Add one basic template for each difficulty level
        self._add_template(
            id="minimal_easy",
            text="Identify the candlestick pattern shown in the chart.",
            difficulty=QuestionDifficulty.EASY,
            question_type=QuestionType.IDENTIFICATION,
            format=QuestionFormat.MULTIPLE_CHOICE,
            requires_variables=["chart_data"],
            tags=["basic"]
        )

        self._add_template(
            id="minimal_medium",
            text="What is the expected price movement after this {pattern_name} pattern?",
            difficulty=QuestionDifficulty.MEDIUM,
            question_type=QuestionType.PREDICTION,
            format=QuestionFormat.MULTIPLE_CHOICE,
            requires_variables=["pattern_name", "chart_data"],
            tags=["basic"]
        )

        self._add_template(
            id="minimal_hard",
            text="Analyze the characteristics of this {pattern_name} pattern and select the correct statement.",
            difficulty=QuestionDifficulty.HARD,
            question_type=QuestionType.CHARACTERISTIC,
            format=QuestionFormat.MULTIPLE_CHOICE,
            requires_variables=["pattern_name", "chart_data"],
            tags=["advanced"]
        )

        logger.warning("Initialized with minimal template set as fallback")
    
    def _add_template(self, id: str, text: str, difficulty: QuestionDifficulty, 
                     question_type: QuestionType, format: QuestionFormat,
                     requires_variables: List[str], tags: List[str] = []):
        """Add a template to the database."""
        template = QuestionTemplate(
            id=id,
            text=text,
            difficulty=difficulty,
            question_type=question_type,
            format=format,
            requires_variables=requires_variables,
            tags=tags
        )
        
        # Store template
        self.templates[id] = template
        
        # Index template
        self.by_difficulty.add(difficulty, id)
        self.by_type.add(question_type, id)
        self.by_format.add(format, id)
        
        for tag in tags:
            self.by_tag.add(tag, id)
    
    def get_template(self, template_id: str) -> Optional[QuestionTemplate]:
        """Get a template by ID."""
        return self.templates.get(template_id)
    
    def get_templates_by_difficulty(self, difficulty: QuestionDifficulty) -> List[QuestionTemplate]:
        """Get all templates of a specific difficulty."""
        template_ids = self.by_difficulty.get(difficulty)
        return [self.templates[tid] for tid in template_ids if tid in self.templates]
    
    def get_templates_by_type(self, question_type: QuestionType) -> List[QuestionTemplate]:
        """Get all templates of a specific type."""
        template_ids = self.by_type.get(question_type)
        return [self.templates[tid] for tid in template_ids if tid in self.templates]
    
    def get_templates_by_tag(self, tag: str) -> List[QuestionTemplate]:
        """Get all templates with a specific tag."""
        template_ids = self.by_tag.get(tag)
        return [self.templates[tid] for tid in template_ids if tid in self.templates]
    
    def get_random_template(self, difficulty: Optional[QuestionDifficulty] = None,
                           question_type: Optional[QuestionType] = None,
                           format: Optional[QuestionFormat] = None,
                           tag: Optional[str] = None) -> Optional[QuestionTemplate]:
        """Get a random template matching the given criteria."""
        # Start with all template IDs
        candidate_ids = set(self.templates.keys())
        
        # Apply filters
        if difficulty:
            candidate_ids &= set(self.by_difficulty.get(difficulty))
        
        if question_type:
            candidate_ids &= set(self.by_type.get(question_type))
        
        if format:
            candidate_ids &= set(self.by_format.get(format))
        
        if tag:
            candidate_ids &= set(self.by_tag.get(tag))
        
        # Return a random template from the candidates
        if candidate_ids:
            template_id = random.choice(list(candidate_ids))
            return self.templates.get(template_id)
            
        return None

class BloomFilter:
    """
    Optimized Bloom filter implementation for duplicate detection.
    
    This implementation is more memory-efficient and uses better hashing
    functions than the original implementation.
    """
    
    def __init__(self, size: int = 10000, hash_count: int = 5):
        """
        Initialize the Bloom filter.
        
        Args:
            size: Size of the bit array
            hash_count: Number of hash functions to use
        """
        self.size = size
        self.hash_count = hash_count
        self.bit_array = bitarray(size)
        self.bit_array.setall(0)
        self.logger = app_logger.getChild("bloom_filter")
        
    def _hash_fnv1a(self, data: str, seed: int = 0) -> int:
        """
        FNV-1a non-cryptographic hash function.
        
        Args:
            data: String to hash
            seed: Seed value for the hash
            
        Returns:
            Hash value
        """
        FNV_PRIME = 16777619
        OFFSET_BASIS = 2166136261 ^ seed
        
        hash_val = OFFSET_BASIS
        for byte in data.encode():
            hash_val = ((hash_val ^ byte) * FNV_PRIME) & 0xFFFFFFFF
        
        return hash_val % self.size
        
    def _get_hash_values(self, item: str) -> List[int]:
        """
        Get the hash values for an item.
        
        Args:
            item: The item to hash
            
        Returns:
            List of hash values
        """
        return [self._hash_fnv1a(item, i) for i in range(self.hash_count)]
    
    def add(self, item: str) -> None:
        """
        Add an item to the Bloom filter.
        
        Args:
            item: The item to add
        """
        for hash_val in self._get_hash_values(item):
            self.bit_array[hash_val] = 1
    
    def check(self, item: str) -> bool:
        """
        Check if an item might be in the Bloom filter.
        
        Args:
            item: The item to check
            
        Returns:
            True if the item might be in the filter, False if it's definitely not
        """
        return all(self.bit_array[hash_val] for hash_val in self._get_hash_values(item))
    

class SemanticSimilarityChecker:
    """
    Checks semantic similarity between questions using vector embeddings.
    Uses a more efficient caching mechanism and better similarity computation.
    """
    
    # Class-level constants
    DEFAULT_SIMILARITY_THRESHOLD: ClassVar[float] = 0.85
    MAX_CACHE_SIZE: ClassVar[int] = 1000
    
    def __init__(self, similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD):
        """
        Initialize the semantic similarity checker.
        
        Args:
            similarity_threshold: Threshold above which questions are considered similar
        """
        self.similarity_threshold = similarity_threshold
        self.embeddings_cache: Dict[str, List[float]] = {}
        self.logger = app_logger.getChild("similarity_checker")
        self.recent_questions: List[str] = [] 
        self.max_recent_questions = 50
        
    async def compute_embedding(self, text: str) -> List[float]:
        """
        Compute embedding for a text using a consistent hashing approach.
        
        Args:
            text: Text to compute embedding for
            
        Returns:
            Normalized embedding vector for the text
        """
        # Use cached embedding if available
        if text in self.embeddings_cache:
            return self.embeddings_cache[text]
        
        try:
            # Generate a deterministic embedding based on text content
            # This is a simplified version that could be replaced with a real embedding model
            text_normalized = text.lower().strip()
            embedding = []
            
            # Generate a 128-dimensional embedding
            for i in range(128):
                # Use different hash seeds for each dimension
                h = hashlib.sha256(f"{text_normalized}:{i}".encode()).digest()
                # Convert to float in [-1, 1]
                val = int.from_bytes(h[:4], byteorder='little') / (2**32) * 2 - 1
                embedding.append(val)
            
            # Normalize the embedding
            norm = sum(x*x for x in embedding) ** 0.5
            if norm > 0:
                embedding = [x / norm for x in embedding]
            
            # Manage cache size
            if len(self.embeddings_cache) >= self.MAX_CACHE_SIZE:
                # Remove a random item if cache is full
                if self.embeddings_cache:
                    key_to_remove = next(iter(self.embeddings_cache))
                    del self.embeddings_cache[key_to_remove]
            
            # Cache the embedding
            self.embeddings_cache[text] = embedding
            
            return embedding
            
        except Exception as e:
            self.logger.error(f"Error computing embedding: {str(e)}")
            # Return a zero vector as fallback
            return [0.0] * 128

class ContextAwareTemplateSelector:
    """
    Intelligent template selector that considers context and user performance.
    """
    
    def __init__(self, template_db: QuestionTemplateDatabase):
        """
        Initialize the template selector.
        
        Args:
            template_db: Database of question templates
        """
        self.template_db = template_db
        self.logger = app_logger.getChild("template_selector")
        
    def select_template(
        self,
        difficulty: Optional[QuestionDifficulty] = None,
        question_type: Optional[QuestionType] = None,
        format: Optional[QuestionFormat] = None,
        tag: Optional[str] = None
    ) -> Optional[QuestionTemplate]:
        """
        Select a template based on given criteria.
        
        Args:
            difficulty: Optional difficulty level
            question_type: Optional question type
            format: Optional question format
            tag: Optional tag to filter by
            
        Returns:
            Selected template or None if no matching template found
        """
        # Get candidate template IDs
        candidate_ids = set(self.template_db.templates.keys())
        
        # Apply filters
        if difficulty:
            candidate_ids &= set(self.template_db.by_difficulty.get(difficulty))
        
        if question_type:
            candidate_ids &= set(self.template_db.by_type.get(question_type))
        
        if format:
            candidate_ids &= set(self.template_db.by_format.get(format))
        
        if tag:
            candidate_ids &= set(self.template_db.by_tag.get(tag))
        
        # Return a random template from the candidates
        if candidate_ids:
            template_id = random.choice(list(candidate_ids))
            return self.template_db.templates.get(template_id)
            
        return None

class PromptEngineeringSystem:
    """
    System for engineering prompts and generating dynamic content for questions.
    Uses templates and pattern-specific information to create varied and engaging questions.
    """
    
    def __init__(self):
        """Initialize the prompt engineering system."""
        self.logger = app_logger.getChild("prompt_engineering")
        
    def generate_variables(
        self,
        template: QuestionTemplate,
        pattern_data: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate variables for a question template.
        
        Args:
            template: The question template to generate variables for
            pattern_data: Optional pattern-specific data
            context: Optional context data
            
        Returns:
            Dictionary of variable names to values
        """
        variables = {}
        
        try:
            # Handle required variables
            for var_name in template.requires_variables:
                if pattern_data and var_name in pattern_data:
                    variables[var_name] = pattern_data[var_name]
                elif context and var_name in context:
                    variables[var_name] = context[var_name]
                else:
                    # Generate placeholder value
                    variables[var_name] = self._generate_placeholder(var_name)
            
            # Add any additional dynamic content
            if template.question_type == QuestionType.PREDICTION:
                variables["timeframe"] = random.choice(["short-term", "medium-term", "long-term"])
            
            if template.question_type == QuestionType.CONTEXT:
                variables["market_condition"] = random.choice([
                    "bullish", "bearish", "ranging", "volatile"
                ])
            
        except Exception as e:
            self.logger.error(f"Error generating variables: {str(e)}")
            # Return minimal variables to prevent template errors
            return {var: "N/A" for var in template.requires_variables}
        
        return variables
    
    def _generate_placeholder(self, var_name: str) -> str:
        """
        Generate a placeholder value for a variable.
        
        Args:
            var_name: Name of the variable
            
        Returns:
            Placeholder value
        """
        # Map common variable names to sensible defaults
        defaults = {
            "pattern_name": "candlestick pattern",
            "description": "a technical analysis pattern",
            "trend": random.choice(["upward", "downward", "sideways"]),
            "timeframe": random.choice(["daily", "weekly", "monthly"]),
            "price_action": "price movement",
            "volume": random.choice(["high", "low", "average"]),
            "signal_type": random.choice(["bullish", "bearish", "neutral"])
        }
        
        return defaults.get(var_name, f"[{var_name}]")
    
    def enhance_question(self, base_text: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Enhance a question with additional context or clarifications.
        
        Args:
            base_text: The base question text
            context: Optional context information
            
        Returns:
            Enhanced question text
        """
        try:
            # Add context if available
            if context:
                if "market_condition" in context:
                    base_text = f"In a {context['market_condition']} market, {base_text.lower()}"
                if "timeframe" in context:
                    base_text = f"Considering the {context['timeframe']} timeframe, {base_text.lower()}"
            
            # Add clarity phrases randomly
            clarity_phrases = [
                "Based on the chart pattern,",
                "Looking at the price action,",
                "Given the market context,",
                "Analyzing the candlestick formation,"
            ]
            
            if random.random() < 0.3:  # 30% chance to add clarity phrase
                base_text = f"{random.choice(clarity_phrases)} {base_text.lower()}"
            
            return base_text[0].upper() + base_text[1:]  # Capitalize first letter
            
        except Exception as e:
            self.logger.error(f"Error enhancing question: {str(e)}")
            return base_text  # Return original text if enhancement fails