"""
Dynamic Question Generator for Candlestick Pattern Assessments

This module provides:
1. Advanced question generation and selection based on user performance
2. Integration with adaptive difficulty engine for optimal learning pace
3. Template-based generation with variability and uniqueness guarantees
4. Reinforcement learning-driven question selection strategies
"""

import asyncio
import random
import hashlib
import uuid
from typing import Dict, List, Any, Optional, Union, Tuple, Set, TypeVar, cast, ClassVar
from enum import Enum, auto
from datetime import datetime, timedelta

# Common utility imports
from backend.common.logger import app_logger, log_execution_time
from backend.common.cache import cache, async_cached
from backend.common.performance.tracker import PerformanceTracker, SkillLevel

# Base classes and interfaces
from backend.assessments.base.models import QuestionDifficulty
from backend.assessments.base.services import QuestionGenerator

# Candlestick pattern specific imports
from backend.assessments.candlestick_patterns.candlestick_questions import (
    QuestionTemplate, QuestionTemplateDatabase, QuestionType, QuestionFormat,
    ContextAwareTemplateSelector, PromptEngineeringSystem
)
from backend.assessments.candlestick_patterns.question_selection import (
    QuestionSelectionAlgorithm, EnhancedBloomFilter
)
from backend.assessments.candlestick_patterns.adaptive_difficulty import (
    AdaptiveDifficultyEngine
)
from backend.assessments.candlestick_patterns.candlestick_config import (
    CANDLESTICK_PATTERNS, DIFFICULTY_LEVELS, ASSESSMENT_CONFIG, PatternCategory
)
from backend.assessments.candlestick_patterns.candlestick_models import (
    CandlestickQuestion, CandlestickPatternData, CandlestickPattern, PatternType
)

# Module logger
logger = app_logger.getChild("question_generator")

# Type variable for generics
T = TypeVar('T', bound=CandlestickQuestion)

class PatternStrength(Enum):
    """Enum for pattern strength levels"""
    WEAK = "weak"
    MEDIUM = "medium"
    STRONG = "strong"

class LearningObjective(str, Enum):
    """Enum for question learning objectives."""
    PATTERN_RECOGNITION = "pattern_recognition"
    PATTERN_MEANING = "pattern_meaning" 
    MARKET_PSYCHOLOGY = "market_psychology"
    INDICATOR_CORRELATION = "indicator_correlation"
    ENTRY_EXIT_STRATEGY = "entry_exit_strategy"
    RISK_MANAGEMENT = "risk_management"


class AdaptiveDifficultyEngine:
    """Engine for adapting question difficulty based on user performance."""
    
    def __init__(self, user_id: str):
        """Initialize engine for a user."""
        self.user_id = user_id
        self.performance_tracker = PerformanceTracker(
            user_id=user_id,
            assessment_type="candlestick_patterns"
        )
        self.skill_level = SkillLevel.BEGINNER
        self.recent_performance = []  # List of recent correct/incorrect answers
        self.streak = 0  # Current streak of correct answers
        self.recent_performance_score = 0.5  # Default score for new users
    
    @classmethod
    def load(cls, user_id: str) -> 'AdaptiveDifficultyEngine':
        """Create or load engine instance for a user."""
        return cls(user_id)
    
    def generate_assessment_config(self) -> Dict[str, Any]:
        """Generate configuration for next assessment."""
        # Default patterns for each skill level
        beginner_patterns = ["DOJI", "HAMMER", "SHOOTING_STAR", "ENGULFING"]
        intermediate_patterns = ["HARAMI", "MORNING_STAR", "EVENING_STAR", "THREE_WHITE_SOLDIERS"]
        advanced_patterns = ["THREE_BLACK_CROWS", "PIERCING_LINE", "DARK_CLOUD_COVER", "RISING_THREE_METHODS"]
        
        # Select patterns based on skill level
        if self.skill_level == SkillLevel.BEGINNER:
            focus_patterns = beginner_patterns
        elif self.skill_level == SkillLevel.INTERMEDIATE:
            focus_patterns = intermediate_patterns + beginner_patterns[:2]  # Include some beginner patterns
        else:
            focus_patterns = advanced_patterns + intermediate_patterns[:2]  # Include some intermediate patterns
            
        # If user has pattern metrics, prioritize patterns they need practice on
        if hasattr(self.performance_tracker, 'pattern_metrics') and self.performance_tracker.pattern_metrics:
            # Sort patterns by mastery (ascending) to focus on ones needing improvement
            pattern_mastery = sorted(
                [(p, m.get('mastery', 0.0)) 
                 for p, m in self.performance_tracker.pattern_metrics.items()],
                key=lambda x: x[1]
            )
            # Add low mastery patterns to focus list
            focus_patterns.extend([p for p, _ in pattern_mastery[:3] if p not in focus_patterns])
        
        return {
            "skill_level": self.skill_level.value,
            "difficulty": self.skill_level.to_numeric() / 5.0,  # Normalize to 0-1
            "focus_patterns": focus_patterns,
            "streak": self.streak,
            "recent_performance": self.recent_performance_score
        }
    
    def update_performance(self, is_correct: bool, time_ms: float) -> None:
        """Update engine state based on user's answer."""
        # Update performance tracker
        self.performance_tracker.record_answer(
            topic="pattern_identification",
            is_correct=is_correct,
            time_ms=time_ms
        )
        
        # Update local state
        self.recent_performance.append(is_correct)
        if len(self.recent_performance) > 10:
            self.recent_performance.pop(0)
            
        # Update recent performance score (moving average)
        if self.recent_performance:
            self.recent_performance_score = sum(self.recent_performance) / len(self.recent_performance)
            
        if is_correct:
            self.streak += 1
        else:
            self.streak = 0
            
        # Update skill level based on performance tracker
        self.skill_level = self.performance_tracker.get_skill_level("pattern_identification")


class AdaptiveQuestionGenerator(QuestionGenerator[CandlestickQuestion]):
    """
    Advanced question generator that adapts to user performance and learning pace.
    
    Features:
    1. Multi-stage question generation pipeline
    2. Template-based generation with dynamic content
    3. Reinforcement learning for question selection
    4. Uniqueness guarantees to avoid repetition
    
    This class properly implements the abstract methods defined in the base
    QuestionGenerator class while extending functionality for candlestick patterns.
    """
    
    # Class constants
    DEFAULT_CACHE_TTL: ClassVar[int] = 7200  # 2 hours in seconds
    MAX_RECENT_PATTERNS: ClassVar[int] = 5
    MAX_DIFFICULTY_ENGINES: ClassVar[int] = 1000  # Limit number of difficulty engines to prevent memory leaks
    
    def __init__(
        self,
        template_db: Optional[QuestionTemplateDatabase] = None,
        selection_algorithm: Optional[QuestionSelectionAlgorithm] = None,
        template_selector: Optional[ContextAwareTemplateSelector] = None,
        prompt_system: Optional[PromptEngineeringSystem] = None,
        uniqueness_filter: Optional[EnhancedBloomFilter] = None
    ):
        """
        Initialize the adaptive question generator with dependency injection.
        
        Args:
            template_db: Optional question template database
            selection_algorithm: Optional question selection algorithm
            template_selector: Optional template selector
            prompt_system: Optional prompt engineering system
            uniqueness_filter: Optional uniqueness filter for deduplication
        """
        # Initialize components with dependency injection
        self.template_db = template_db or QuestionTemplateDatabase()
        self.selection_algorithm = selection_algorithm or QuestionSelectionAlgorithm()
        
        # Initialize template selector with the provided database or create new
        self.template_selector = template_selector or ContextAwareTemplateSelector(self.template_db)
        
        # Initialize prompt engineering system
        self.prompt_system = prompt_system or PromptEngineeringSystem()
        
        # Initialize uniqueness filter with better defaults
        self.uniqueness_filter = uniqueness_filter or EnhancedBloomFilter(
            capacity=100000,
            false_positive_rate=0.01,
            max_timestamps=50000  # Limit timestamp storage
        )
        
        # Difficulty engines by user with LRU caching to limit memory usage
        self.difficulty_engines: Dict[str, AdaptiveDifficultyEngine] = {}
        self._difficulty_engine_access_times: Dict[str, float] = {}
        
        # Performance metrics
        self.generation_metrics = {
            "total_generated": 0,
            "cache_hits": 0,
            "template_fallbacks": 0,
            "ai_generations": 0,
            "errors": 0
        }
        
        logger.info("Initialized AdaptiveQuestionGenerator with dependency injection")
    
    async def _get_difficulty_engine(self, user_id: str) -> AdaptiveDifficultyEngine:
        """
        Get or create a difficulty engine for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            AdaptiveDifficultyEngine instance
        """
        # Clean up old engines if we've reached the limit
        if len(self.difficulty_engines) >= self.MAX_DIFFICULTY_ENGINES:
            self._cleanup_oldest_difficulty_engines(int(self.MAX_DIFFICULTY_ENGINES * 0.2))
        
        # Record access time
        self._difficulty_engine_access_times[user_id] = datetime.now().timestamp()
        
        if user_id not in self.difficulty_engines:
            # Load from cache/persistence or create new
            self.difficulty_engines[user_id] = AdaptiveDifficultyEngine.load(user_id)
        
        return self.difficulty_engines[user_id]
    
    def _cleanup_oldest_difficulty_engines(self, count: int) -> None:
        """
        Remove the oldest difficulty engines to prevent memory leaks.
        
        Args:
            count: Number of engines to remove
        """
        if not self._difficulty_engine_access_times:
            return
            
        # Sort by access time (oldest first)
        sorted_items = sorted(
            self._difficulty_engine_access_times.items(),
            key=lambda x: x[1]
        )
        
        # Remove the oldest engines
        for user_id, _ in sorted_items[:count]:
            if user_id in self.difficulty_engines:
                del self.difficulty_engines[user_id]
            if user_id in self._difficulty_engine_access_times:
                del self._difficulty_engine_access_times[user_id]
                
        logger.info(f"Cleaned up {count} oldest difficulty engines")
    
    def _register_question(self, question: CandlestickQuestion) -> None:
        """
        Register a question with the selection algorithm.
        
        Args:
            question: The question to register
        """
        difficulty_value = question.difficulty.to_numeric() / 5.0  # Convert to 0.0-1.0 scale
        
        # Extract tags from question metadata if available
        tags = []
        if hasattr(question, 'metadata') and question.metadata:
            pattern_info = question.metadata.get('pattern_info', {})
            if pattern_info:
                if 'category' in pattern_info:
                    tags.append(pattern_info['category'])
                if pattern_info.get('is_reversal'):
                    tags.append('reversal')
                elif pattern_info.get('is_continuation'):
                    tags.append('continuation')
        
        # Register with the selection algorithm
        self.selection_algorithm.register_question(
            question.id, 
            difficulty_value,
            tags=tags
        )
    
    @log_execution_time()
    async def generate_question(
        self,
        difficulty: Optional[QuestionDifficulty] = None,
        topics: Optional[List[str]] = None,
        exclude_ids: Optional[List[str]] = None
    ) -> CandlestickQuestion:
        """
        Generate a single question based on criteria.
        
        This implements the abstract method from the base QuestionGenerator class.
        
        Args:
            difficulty: Optional difficulty level
            topics: Optional list of topics to choose from
            exclude_ids: Optional list of question IDs to exclude
            
        Returns:
            Generated question
        """
        try:
            # Use a default user ID if not generating for a specific user
            user_id = "default_user"
            
            # Get performance metrics (empty for default user)
            user_metrics = {
                "skill_level": 0.5,
                "recent_performance": 0.5,
                "streak": 0,
                "pattern_metrics": {}
            }
            
            # Create pattern diversity info
            pattern_diversity = {
                "recent_patterns": [],
                "recent_question_types": []
            }
            
            # Generate a question with default settings
            question_data = await self._generate_question_for_user(
                user_id=user_id,
                user_metrics=user_metrics,
                pattern_diversity=pattern_diversity,
                difficulty_override=difficulty,
                topics_override=topics,
                exclude_ids=exclude_ids
            )
            
            # Convert to CandlestickQuestion
            question = self._create_question_from_data(question_data)
            
            # Register with selection algorithm
            self._register_question(question)
            
            return question
        except Exception as e:
            logger.error(f"Error generating question: {str(e)}")
            # Create a basic fallback question
            return self._create_fallback_question(difficulty)
    
    def _create_fallback_question(
        self,
        difficulty: Optional[QuestionDifficulty] = None
    ) -> CandlestickQuestion:
        """
        Create a basic fallback question when generation fails.
        
        Args:
            difficulty: Optional difficulty level
            
        Returns:
            Basic fallback question
        """
        actual_difficulty = difficulty or QuestionDifficulty.MEDIUM
        
        # Create a basic identification question
        return CandlestickQuestion(
            id=str(uuid.uuid4()),
            question_text="What candlestick pattern is shown in the chart?",
            difficulty=actual_difficulty,
            question_type="identification",
            pattern="DOJI",  # Using a simple pattern
            pattern_strength=PatternStrength.MEDIUM,
            chart_data={},  # Empty chart data for fallback
            chart_image="",  # No image for fallback
            timeframe="1D",  # Default timeframe
            symbol="AAPL",  # Default symbol
            topics=["candlestick_patterns"],
            options=["Doji", "Hammer", "Bullish Engulfing", "Evening Star"],
            explanation="This is a fallback question.",
            metadata={"is_fallback": True}
        )
    
    @log_execution_time()
    async def generate_questions(
        self,
        count: int,
        difficulty: Optional[QuestionDifficulty] = None,
        topics: Optional[List[str]] = None,
        exclude_ids: Optional[List[str]] = None
    ) -> List[CandlestickQuestion]:
        """
        Generate multiple questions based on criteria.
        
        This implements the abstract method from the base QuestionGenerator class.
        
        Args:
            count: Number of questions to generate
            difficulty: Optional difficulty level
            topics: Optional list of topics to choose from
            exclude_ids: Optional list of question IDs to exclude
            
        Returns:
            List of generated questions
        """
        if count <= 0:
            logger.warning(f"Invalid question count: {count}")
            return []
            
        try:
            logger.info(f"Generating {count} questions with difficulty={difficulty}, topics={topics}")
            
            # Use a default user ID if not generating for a specific user
            user_id = "default_user"
            
            # Get performance metrics (empty for default user)
            user_metrics = {
                "skill_level": 0.5,
                "recent_performance": 0.5,
                "streak": 0,
                "pattern_metrics": {}
            }
            
            # Create pattern diversity info
            pattern_diversity = {
                "recent_patterns": [],
                "recent_question_types": []
            }
            
            # Generate questions in parallel
            questions = []
            exclude_set = set(exclude_ids or [])
            tasks = []
            
            # Create tasks for parallel question generation with batching for efficiency
            BATCH_SIZE = 10  # Process questions in batches of 10 for better performance
            for batch_start in range(0, count, BATCH_SIZE):
                batch_size = min(BATCH_SIZE, count - batch_start)
                batch_tasks = []
                
                for _ in range(batch_size):
                    batch_tasks.append(
                        self._generate_question_for_user(
                            user_id=user_id,
                            user_metrics=user_metrics,
                            pattern_diversity=pattern_diversity,
                            difficulty_override=difficulty,
                            topics_override=topics,
                            exclude_ids=list(exclude_set)
                        )
                    )
                
                # Execute batch tasks concurrently
                try:
                    batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                    
                    # Process results, handling any exceptions
                    for result in batch_results:
                        if isinstance(result, Exception):
                            logger.error(f"Error generating question in batch: {result}")
                            # Add a fallback question instead
                            questions.append(self._create_fallback_question(difficulty))
                        else:
                            # Convert to CandlestickQuestion
                            question = self._create_question_from_data(result)
                            
                            # Add to results and update exclusion set
                            questions.append(question)
                            exclude_set.add(question.id)
                            
                            # Update pattern diversity info
                            self._update_pattern_diversity(
                                pattern_diversity,
                                result.get("pattern", ""),
                                result.get("question_type", "")
                            )
                            
                            # Register with selection algorithm
                            self._register_question(question)
                    
                except Exception as batch_error:
                    logger.error(f"Fatal error processing question batch: {batch_error}")
                    # Add fallback questions for this batch
                    for _ in range(batch_size):
                        questions.append(self._create_fallback_question(difficulty))
                
                # Break early if we've generated enough questions
                if len(questions) >= count:
                    break
            
            # Return exactly the number of questions requested
            final_questions = questions[:count]
            logger.info(f"Successfully generated {len(final_questions)} questions")
            return final_questions
            
        except Exception as e:
            logger.error(f"Error generating questions batch: {str(e)}")
            # Return fallback questions
            return [self._create_fallback_question(difficulty) for _ in range(min(count, 3))]
    
    def _update_pattern_diversity(
        self, 
        pattern_diversity: Dict[str, List[str]],
        pattern: str,
        question_type: str
    ) -> None:
        """
        Update pattern diversity tracking information.
        
        Args:
            pattern_diversity: Dictionary tracking pattern diversity
            pattern: Pattern name to add
            question_type: Question type to add
        """
        if pattern:
            pattern_diversity["recent_patterns"].append(pattern)
            # Keep only most recent patterns
            if len(pattern_diversity["recent_patterns"]) > self.MAX_RECENT_PATTERNS:
                pattern_diversity["recent_patterns"] = pattern_diversity["recent_patterns"][-self.MAX_RECENT_PATTERNS:]
                
        if question_type:
            pattern_diversity["recent_question_types"].append(question_type)
            # Keep only most recent question types
            if len(pattern_diversity["recent_question_types"]) > self.MAX_RECENT_PATTERNS:
                pattern_diversity["recent_question_types"] = pattern_diversity["recent_question_types"][-self.MAX_RECENT_PATTERNS:]
    
    @log_execution_time()
    async def generate_for_user(
        self,
        user_id: str,
        count: int,
        topics: Optional[List[str]] = None
    ) -> List[CandlestickQuestion]:
        """
        Generate questions tailored for a specific user based on their performance.
        
        This implements the abstract method from the base QuestionGenerator class.
        It uses the user's performance history to generate appropriately challenging questions.
        
        Args:
            user_id: User identifier
            count: Number of questions to generate
            topics: Optional list of topics to choose from
            
        Returns:
            List of generated questions tailored to the user
        """
        try:
            logger.info(f"Generating {count} questions for user {user_id}")
            
            # Get the difficulty engine for this user
            difficulty_engine = await self._get_difficulty_engine(user_id)
            
            # Get assessment configuration
            assessment_config = difficulty_engine.generate_assessment_config()
            
            # Extract user metrics
            user_metrics = {
                "user_id": user_id,
                "skill_level": assessment_config["skill_level"],
                "recent_performance": assessment_config["recent_performance"],
                "streak": assessment_config["streak"],
                "pattern_metrics": difficulty_engine.performance_tracker.pattern_metrics
            }
            
            # Create pattern diversity info
            pattern_diversity = {
                "recent_patterns": [],
                "recent_question_types": []
            }
            
            # Use focus patterns from assessment config
            focus_patterns = assessment_config["focus_patterns"]
            
            # Determine how many questions to generate for each pattern
            pattern_counts = self._distribute_questions(count, focus_patterns)
            
            exclude_set = set()
            tasks = []
            
            # Create tasks for each pattern
            for pattern, pattern_count in pattern_counts.items():
                for _ in range(pattern_count):
                    # Create task for generating a question with this pattern
                    tasks.append(
                        self._generate_question_for_user(
                            user_id=user_id,
                            user_metrics=user_metrics,
                            pattern_diversity=pattern_diversity,
                            pattern_override=pattern,
                            topics_override=topics,
                            exclude_ids=list(exclude_set)
                        )
                    )
            
            # Execute tasks concurrently for performance
            question_data_list = await asyncio.gather(*tasks)
            
            # Process results
            questions = []
            for question_data in question_data_list:
                # Convert to CandlestickQuestion
                question = self._create_question_from_data(question_data)
                
                # Add to results
                questions.append(question)
                exclude_set.add(question.id)
                
                # Update pattern diversity info
                self._update_pattern_diversity(
                    pattern_diversity,
                    question_data.get("pattern", ""),
                    question_data.get("question_type", "")
                )
                
                # Register question with selection algorithm
                self._register_question(question)
                
                # Record that the user has seen this question
                self.selection_algorithm.record_question_seen(question.id, user_id)
            
            logger.info(f"Successfully generated {len(questions)} questions for user {user_id}")
            return questions
            
        except Exception as e:
            logger.error(f"Error generating questions for user {user_id}: {str(e)}")
            # Return fallback questions
            return [self._create_fallback_question() for _ in range(min(count, 3))]
    
    def _distribute_questions(self, count: int, patterns: List[str]) -> Dict[str, int]:
        """
        Distribute question count among patterns efficiently.
        
        Args:
            count: Total number of questions
            patterns: List of patterns
            
        Returns:
            Dictionary mapping patterns to question counts
        """
        result = {}
        
        if not patterns:
            logger.warning("No patterns provided for distribution")
            return result
        
        # Ensure we don't exceed the number of patterns
        pattern_count = len(patterns)
        
        # Give each pattern at least one question if possible
        if count >= pattern_count:
            for pattern in patterns:
                result[pattern] = 1
                
            # Distribute remaining questions proportionally
            remaining = count - pattern_count
            
            if remaining > 0:
                # Use round-robin distribution for fairness
                for i in range(remaining):
                    pattern = patterns[i % pattern_count]
                    result[pattern] = result.get(pattern, 0) + 1
        else:
            # Not enough questions for all patterns, select a subset
            selected_patterns = random.sample(patterns, count)
            for pattern in selected_patterns:
                result[pattern] = 1
        
        return result
    
    @async_cached(key_prefix="question_data", ttl=7200)
    async def _generate_question_for_user(
        self,
        user_id: str,
        user_metrics: Dict[str, Any],
        pattern_diversity: Dict[str, List[str]],
        pattern_override: Optional[str] = None,
        difficulty_override: Optional[QuestionDifficulty] = None,
        topics_override: Optional[List[str]] = None,
        exclude_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate a question for a specific user.
        
        This is the core question generation method that handles all the details
        of creating a question based on user metrics and pattern information.
        
        Args:
            user_id: User identifier
            user_metrics: User performance metrics
            pattern_diversity: Pattern diversity information
            pattern_override: Optional pattern override
            difficulty_override: Optional difficulty override
            topics_override: Optional topics override
            exclude_ids: Optional list of question IDs to exclude
            
        Returns:
            Dictionary containing question data
        """
        try:
            # Track generation
            self.generation_metrics["total_generated"] += 1
            
            # Determine pattern to use
            pattern = pattern_override
            if not pattern:
                # Select a pattern based on user metrics and diversity
                pattern = self._select_pattern(user_metrics, pattern_diversity["recent_patterns"])
            
            # Get pattern information
            pattern_info = self._get_pattern_info(pattern)
            
            # Apply topic filters if needed
            if topics_override and pattern_info["category"] not in topics_override:
                # If the pattern doesn't match the topic filter, try to find another pattern
                alternate_patterns = []
                for category, patterns in CANDLESTICK_PATTERNS.items():
                    if category in topics_override:
                        alternate_patterns.extend(patterns)
                
                if alternate_patterns:
                    pattern = random.choice(alternate_patterns)
                    pattern_info = self._get_pattern_info(pattern)
            
            # Determine difficulty
            difficulty = difficulty_override
            if not difficulty:
                # Get difficulty from user metrics
                skill_level = user_metrics.get("skill_level", 0.5)
                
                # Ensure skill_level is a float
                if isinstance(skill_level, str):
                    # Convert string skill levels to numeric values
                    if skill_level.lower() == "beginner":
                        skill_level = 0.2
                    elif skill_level.lower() == "intermediate":
                        skill_level = 0.5
                    elif skill_level.lower() == "advanced":
                        skill_level = 0.8
                    else:
                        # Default to medium if unknown
                        skill_level = 0.5
                        logger.warning(f"Unknown skill level string: {skill_level}, defaulting to 0.5")
                
                try:
                    # Convert to float and bound between 0.0 and 1.0
                    skill_level = float(skill_level)
                    skill_level = max(0.0, min(1.0, skill_level))
                except (ValueError, TypeError):
                    skill_level = 0.5
                    logger.warning(f"Could not convert skill_level to float, defaulting to 0.5")
                
                # Map skill level to difficulty (0.0-1.0 -> 1-5)
                difficulty_value = 1 + int(skill_level * 4)
                difficulty = QuestionDifficulty.from_numeric(difficulty_value)
            else:
                # Use the provided difficulty
                difficulty = difficulty_override
            
            # Select template
            template = self.template_selector.select_template(
                difficulty=difficulty,
                question_type=None,  # We're not specifying a question type
                format=None,         # We're not specifying a format
                tag=None             # We're not specifying a tag
            )
            
            if not template:
                # Fallback to a random template with the right difficulty
                self.generation_metrics["template_fallbacks"] += 1
                templates = self.template_db.get_templates_by_difficulty(difficulty)
                if templates:
                    template = random.choice(templates)
                else:
                    # Ultimate fallback: use a medium difficulty template
                    templates = self.template_db.get_templates_by_difficulty(QuestionDifficulty.MEDIUM)
                    if not templates:
                        raise ValueError("No templates available")
                    template = random.choice(templates)
            
            # Prepare variables for template
            variables = await self._prepare_template_variables(pattern_info, user_metrics)
            
            # Format question text
            question_text = template.format_question(variables)
            
            # Generate answer options if needed
            if template.format == QuestionFormat.MULTIPLE_CHOICE:
                # Add answer options
                answer_options = self._generate_answer_options(pattern, pattern_info, template.question_type)
            else:
                # Other question formats may not need options
                answer_options = None
            
            # Create question data
            question_id = str(uuid.uuid4())
            
            # Check if we need to exclude this ID
            while exclude_ids and question_id in exclude_ids:
                question_id = str(uuid.uuid4())
            
            # Create final question data
            return {
                "id": question_id,
                "question_text": question_text,
                "difficulty": difficulty.value,
                "question_type": template.question_type.value,
                "format": template.format.value,
                "pattern": pattern,
                "answer_options": answer_options,
                "correct_answer": pattern if template.question_type == QuestionType.IDENTIFICATION else None,
                "template_id": template.id,
                "created_at": datetime.now().isoformat(),
                "metadata": {
                    "user_id": user_id,
                    "user_skill_level": user_metrics.get("skill_level", 0.5),
                    "pattern_info": pattern_info,
                    "template_variables": variables
                }
            }
        except Exception as e:
            logger.error(f"Error in _generate_question_for_user: {str(e)}")
            # Return a basic fallback question data
            return self._create_fallback_question_data(user_id)
    
    def _create_fallback_question_data(self, user_id: str) -> Dict[str, Any]:
        """
        Create fallback question data when generation fails.
        
        Args:
            user_id: User identifier
            
        Returns:
            Basic fallback question data
        """
        question_id = str(uuid.uuid4())
        pattern = "Doji"  # Use a simple pattern as fallback
        
        return {
            "id": question_id,
            "question_text": "What candlestick pattern is shown in the chart?",
            "difficulty": QuestionDifficulty.MEDIUM.value,
            "question_type": QuestionType.IDENTIFICATION.value,
            "format": QuestionFormat.MULTIPLE_CHOICE.value,
            "pattern": pattern,
            "answer_options": {
                "A": "Doji",
                "B": "Hammer",
                "C": "Bullish Engulfing",
                "D": "Evening Star"
            },
            "correct_answer": "A",
            "template_id": "fallback_identification",
            "created_at": datetime.now().isoformat(),
            "metadata": {
                "user_id": user_id,
                "is_fallback": True
            }
        }
    
    def _select_pattern(
        self,
        user_metrics: Dict[str, Any],
        recent_patterns: List[str]
    ) -> str:
        """
        Intelligently select a pattern based on user metrics and diversity.
        
        Uses spaced repetition principles and mastery tracking to select
        patterns that will optimize learning.
        
        Args:
            user_metrics: User performance metrics
            recent_patterns: Recently used patterns
            
        Returns:
            Selected pattern name
        """
        # Get pattern metrics if available
        pattern_metrics = user_metrics.get("pattern_metrics", {})
        
        # Get all available patterns
        try:
            all_patterns = []
            for category, patterns in CANDLESTICK_PATTERNS.items():
                all_patterns.extend(patterns)
            
            if not all_patterns:
                logger.warning("No patterns available in CANDLESTICK_PATTERNS")
                return "Doji"  # Default fallback pattern
                
            # Filter out recent patterns to ensure diversity
            recent_set = set(recent_patterns[-self.MAX_RECENT_PATTERNS:] if recent_patterns else [])
            available_patterns = [p for p in all_patterns if p not in recent_set]
            
            # If all patterns are recent, reset and use any pattern
            if not available_patterns:
                logger.debug("All patterns were recently used, allowing repeats")
                available_patterns = all_patterns
            
            # Categorize patterns by mastery level
            pattern_categories = {
                "low_mastery": [],    # Mastery < 0.3
                "medium_mastery": [], # Mastery 0.3-0.7
                "high_mastery": []    # Mastery > 0.7
            }
            
            # Sort patterns into categories
            for pattern in available_patterns:
                if pattern in pattern_metrics:
                    mastery = pattern_metrics[pattern].get("mastery", 0.0)
                    if mastery < 0.3:
                        pattern_categories["low_mastery"].append(pattern)
                    elif mastery < 0.7:
                        pattern_categories["medium_mastery"].append(pattern)
                    else:
                        pattern_categories["high_mastery"].append(pattern)
                else:
                    # Unknown patterns go to medium mastery for exploration
                    pattern_categories["medium_mastery"].append(pattern)
            
            # Define selection weights based on learning strategy
            selection_weights = [
                (pattern_categories["low_mastery"], 0.6),     # 60% focus on low mastery
                (pattern_categories["medium_mastery"], 0.3),  # 30% focus on medium mastery
                (pattern_categories["high_mastery"], 0.1)     # 10% focus on high mastery for reinforcement
            ]
            
            # Filter out empty categories
            valid_selections = [(patterns, weight) for patterns, weight in selection_weights if patterns]
            
            if not valid_selections:
                # This should not happen as we've already checked for available_patterns,
                # but handle it gracefully just in case
                logger.warning("No valid pattern categories found, using random pattern")
                return random.choice(available_patterns)
            
            # Normalize weights
            patterns_list, weights = zip(*valid_selections)
            total_weight = sum(weights)
            normalized_weights = [w / total_weight for w in weights]
            
            # Select category based on weights
            selected_category = random.choices(
                population=patterns_list,
                weights=normalized_weights,
                k=1
            )[0]
            
            # Select a random pattern from the category
            return random.choice(selected_category)
            
        except Exception as e:
            logger.error(f"Error selecting pattern: {str(e)}", exc_info=True)
            # Fallback to a safe default pattern
            return "Doji"
    
    def _get_pattern_info(self, pattern: str) -> Dict[str, Any]:
        """
        Get comprehensive information about a candlestick pattern.
        
        Args:
            pattern: Pattern name
            
        Returns:
            Dictionary containing pattern information
        """
        # Find pattern category
        pattern_category = None
        for category, patterns in CANDLESTICK_PATTERNS.items():
            if pattern in patterns:
                pattern_category = category
                break
        
        if not pattern_category:
            logger.warning(f"Pattern '{pattern}' not found in any category")
            pattern_category = "single"  # Default fallback
        
        # Determine pattern characteristics based on name and category
        is_reversal = any(term in pattern.lower() for term in ["reversal", "hammer", "engulfing", "harami", "star"])
        is_continuation = any(term in pattern.lower() for term in ["continuation", "three methods", "three soldiers"])
        
        # Determine number of candles required
        candles_required = 1
        if "double" in pattern_category.lower() or "two" in pattern.lower():
            candles_required = 2
        elif "triple" in pattern_category.lower() or "three" in pattern.lower():
            candles_required = 3
        elif "complex" in pattern_category.lower():
            candles_required = 3
        
        # Create pattern info
        return {
            "name": pattern,
            "category": pattern_category,
            "is_reversal": is_reversal,
            "is_continuation": is_continuation,
            "candles_required": candles_required,
            "reliability": 0.7,  # Placeholder - would be data-driven in production
            "difficulty": 0.5    # Placeholder - would be data-driven in production
        }
    
    async def _prepare_template_variables(
        self,
        pattern_info: Dict[str, Any],
        user_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Prepare context-aware variables for template formatting.
        
        This method creates a rich set of variables that can be used
        to format question templates with personalized content.
        
        Args:
            pattern_info: Pattern information
            user_metrics: User performance metrics
            
        Returns:
            Dictionary of template variables
        """
        # Basic pattern information
        variables = {
            "pattern_name": pattern_info["name"],
            "pattern_category": pattern_info["category"],
        }
        
        # Add market context variables with proper randomization
        timeframes = ["daily", "4-hour", "1-hour", "30-minute", "15-minute"]
        market_conditions = ["bullish", "bearish", "consolidating", "volatile", "trending"]
        
        # User's skill level affects variable complexity
        skill_level = user_metrics.get("skill_level", 0.5)
        
        # Ensure skill_level is a float
        if isinstance(skill_level, str):
            # Convert string skill levels to numeric values
            if skill_level.lower() == "beginner":
                skill_level = 0.2
            elif skill_level.lower() == "intermediate":
                skill_level = 0.5
            elif skill_level.lower() == "advanced":
                skill_level = 0.8
            else:
                # Default to medium if unknown
                skill_level = 0.5
                logger.warning(f"Unknown skill level string: {skill_level}, defaulting to 0.5")
        
        try:
            # Convert to float and bound between 0.0 and 1.0
            skill_level = float(skill_level)
            skill_level = max(0.0, min(1.0, skill_level))
        except (ValueError, TypeError):
            skill_level = 0.5
            logger.warning(f"Could not convert skill_level to float, defaulting to 0.5")
        
        # More advanced timeframes for higher skill levels
        if skill_level > 0.7:
            timeframe_weights = [0.1, 0.2, 0.3, 0.2, 0.2]  # Bias toward shorter timeframes
        elif skill_level > 0.4:
            timeframe_weights = [0.2, 0.3, 0.3, 0.1, 0.1]  # Balanced
        else:
            timeframe_weights = [0.4, 0.3, 0.2, 0.1, 0.0]  # Bias toward longer timeframes
            
        # Select timeframe based on weights
        variables["timeframe"] = random.choices(timeframes, weights=timeframe_weights, k=1)[0]
        
        # Market conditions
        variables["market_condition"] = random.choice(market_conditions)
        
        # Add asset symbols with industry-appropriate choices
        variables["stock_symbol"] = random.choice(["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "JPM", "V", "WMT", "JNJ", "PG"])
        variables["crypto_symbol"] = random.choice(["BTC", "ETH", "SOL", "ADA", "DOT", "AVAX", "LINK", "XRP", "MATIC", "DOGE"])
        variables["forex_pair"] = random.choice(["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD", "USD/CHF", "NZD/USD", "EUR/GBP"])
        
        # Add pattern characteristics
        if pattern_info["is_reversal"]:
            variables["pattern_characteristic"] = "reversal"
            variables["pattern_signal"] = "potential trend reversal"
        elif pattern_info["is_continuation"]:
            variables["pattern_characteristic"] = "continuation"
            variables["pattern_signal"] = "likely trend continuation"
        else:
            variables["pattern_characteristic"] = "neutral"
            variables["pattern_signal"] = "market indecision"
            
        # Add trend direction based on pattern name
        if any(term in pattern_info["name"].lower() for term in ["bullish", "hammer", "white", "morning"]):
            variables["trend_direction"] = "bullish"
            variables["expected_outcome"] = "price increase"
        elif any(term in pattern_info["name"].lower() for term in ["bearish", "shooting", "black", "evening"]):
            variables["trend_direction"] = "bearish"
            variables["expected_outcome"] = "price decrease"
        else:
            variables["trend_direction"] = "neutral"
            variables["expected_outcome"] = "price consolidation"
            
        return variables
    
    def _generate_answer_options(
        self,
        correct_pattern: str,
        pattern_info: Dict[str, Any],
        question_type: QuestionType
    ) -> Dict[str, str]:
        """
        Generate appropriate answer options for multiple choice questions.
        
        This method creates answer options based on the question type, ensuring
        that the options are relevant and challenging.
        
        Args:
            correct_pattern: Correct pattern name
            pattern_info: Pattern information
            question_type: Question type
            
        Returns:
            Dictionary mapping option keys to values
        """
        try:
            if question_type == QuestionType.IDENTIFICATION:
                # For identification questions, generate pattern options
                return self._generate_pattern_options(correct_pattern, pattern_info)
            elif question_type == QuestionType.PREDICTION:
                # For prediction questions, generate outcome options
                return self._generate_prediction_options(correct_pattern, pattern_info)
            elif question_type == QuestionType.CHARACTERISTIC:
                # For characteristic questions, generate feature options
                return self._generate_characteristic_options(correct_pattern, pattern_info)
            else:
                # Default options for other question types
                return {
                    "A": "Option A",
                    "B": "Option B",
                    "C": "Option C",
                    "D": "Option D"
                }
        except Exception as e:
            logger.error(f"Error generating answer options: {str(e)}")
            # Return basic fallback options
            return {
                "A": "Option A (Fallback)",
                "B": "Option B (Fallback)",
                "C": "Option C (Fallback)",
                "D": "Option D (Fallback)"
            }
    
    def _generate_prediction_options(
        self,
        pattern: str,
        pattern_info: Dict[str, Any]
    ) -> Dict[str, str]:
        """
        Generate prediction options for prediction questions.
        
        Args:
            pattern: Pattern name
            pattern_info: Pattern information
            
        Returns:
            Dictionary mapping option keys to prediction outcomes
        """
        # Standard prediction options
        options = {
            "A": "The price will likely increase significantly",
            "B": "The price will likely decrease significantly",
            "C": "The price will likely continue in the same direction",
            "D": "The price will likely consolidate with minimal movement"
        }
        
        # Randomize the order
        items = list(options.items())
        random.shuffle(items)
        
        return {k: v for k, v in zip("ABCD", [v for _, v in items])}
    
    def _generate_characteristic_options(
        self,
        pattern: str,
        pattern_info: Dict[str, Any]
    ) -> Dict[str, str]:
        """
        Generate characteristic options for feature questions.
        
        Args:
            pattern: Pattern name
            pattern_info: Pattern information
            
        Returns:
            Dictionary mapping option keys to pattern characteristics
        """
        # Standard characteristic options
        options = {
            "A": f"It requires {pattern_info['candles_required']} candlesticks to form",
            "B": f"It is typically a {pattern_info['category']} pattern",
            "C": f"It signals a potential {'reversal' if pattern_info['is_reversal'] else 'continuation'} in the trend",
            "D": f"It forms during {'bullish' if 'bullish' in pattern.lower() else 'bearish' if 'bearish' in pattern.lower() else 'any'} market conditions"
        }
        
        # Randomize the order
        items = list(options.items())
        random.shuffle(items)
        
        return {k: v for k, v in zip("ABCD", [v for _, v in items])}
    
    def _generate_pattern_options(
        self,
        correct_pattern: str,
        pattern_info: Dict[str, Any]
    ) -> Dict[str, str]:
        """
        Generate pattern options for identification questions.
        
        This method selects challenging alternatives to the correct pattern,
        focusing on patterns that might be confused with the correct one.
        
        Args:
            correct_pattern: Correct pattern name
            pattern_info: Pattern information
            
        Returns:
            Dictionary mapping option keys to pattern names
        """
        try:
            # Get all patterns
            all_patterns = []
            for category, patterns in CANDLESTICK_PATTERNS.items():
                all_patterns.extend(patterns)
            
            if not all_patterns:
                logger.warning("No patterns available for options")
                return {"A": correct_pattern, "B": "Doji", "C": "Hammer", "D": "Engulfing"}
            
            # Remove correct pattern
            other_patterns = [p for p in all_patterns if p != correct_pattern]
            
            if not other_patterns:
                logger.warning("No alternative patterns available")
                return {"A": correct_pattern, "B": "Unknown Pattern 1", "C": "Unknown Pattern 2", "D": "Unknown Pattern 3"}
            
            # Prioritize confusing patterns from the same category
            same_category_patterns = []
            for category, patterns in CANDLESTICK_PATTERNS.items():
                if category == pattern_info["category"] and patterns:
                    same_category_patterns = [p for p in patterns if p != correct_pattern]
                    break
            
            # Select options with strategy
            other_options = []
            
            # Try to include at least one from the same category
            if same_category_patterns:
                other_options.append(random.choice(same_category_patterns))
            
            # Add similar sounding pattern if available
            similar_patterns = [p for p in other_patterns 
                               if any(term in p.lower() for term in correct_pattern.lower().split())
                               and p not in other_options]
            if similar_patterns:
                other_options.append(random.choice(similar_patterns))
                
            # Fill remaining spots with random patterns
            while len(other_options) < 3:
                candidate = random.choice(other_patterns)
                if candidate not in other_options:
                    other_options.append(candidate)
            
            # Shuffle all options
            all_options = [correct_pattern] + other_options[:3]  # Ensure we only take 3 other options
            random.shuffle(all_options)
            
            # Create options dictionary
            option_keys = ["A", "B", "C", "D"]
            return {key: value for key, value in zip(option_keys, all_options)}
            
        except Exception as e:
            logger.error(f"Error generating pattern options: {str(e)}")
            # Return basic fallback options with correct pattern
            fallback_options = ["Doji", "Hammer", "Engulfing", "Star"]
            if correct_pattern not in fallback_options:
                fallback_options[0] = correct_pattern
            
            return {key: value for key, value in zip(["A", "B", "C", "D"], fallback_options)}
    
    def _create_question_from_data(self, data: Dict[str, Any]) -> CandlestickQuestion:
        """
        Create a CandlestickQuestion from question data.
        
        This method converts the dictionary representation of a question
        into a proper CandlestickQuestion object.
        
        Args:
            data: Question data
            
        Returns:
            CandlestickQuestion instance
        """
        try:
            # Convert difficulty string to enum if needed
            if isinstance(data["difficulty"], str):
                difficulty = QuestionDifficulty(data["difficulty"])
            else:
                difficulty = data["difficulty"]
            
            # Parse created_at if it's a string
            if isinstance(data.get("created_at"), str):
                created_at = datetime.fromisoformat(data["created_at"])
            else:
                created_at = data.get("created_at", datetime.now())
                
            # Ensure there's at least one topic
            topics = data.get("topics", [])
            if not topics:
                # Use the pattern itself as a topic
                pattern = data.get("pattern")
                if pattern:
                    # Convert pattern to topic format (lowercase with underscores)
                    topics = [pattern.lower().replace(" ", "_")]
                else:
                    # Default topic if pattern is not available
                    topics = ["candlestick_patterns"]
                logger.debug(f"No topics provided, using pattern as topic: {topics}")
            
            # Create the question
            return CandlestickQuestion(
                id=data.get("id", str(uuid.uuid4())),
                question_type=data.get("question_type", "candlestick_pattern"),
                question_text=data["question_text"],
                difficulty=difficulty,
                topics=topics,
                pattern=data["pattern"],
                pattern_strength=PatternStrength.MEDIUM,
                chart_data=data.get("chart_data", {}),
                chart_image=data.get("chart_image", ""),
                timeframe=data.get("timeframe", ""),
                symbol=data.get("symbol", ""),
                options=data.get("options", []),
                metadata=data.get("metadata", {})
            )
        except Exception as e:
            logger.error(f"Error creating question from dictionary: {str(e)}")
            raise ValueError(f"Cannot create question from data: {str(e)}")
    
    async def generate_random_question(
        self,
        difficulty: Optional[str] = None,
        topic: Optional[str] = None
    ) -> CandlestickQuestion:
        """Generate a random question with optional constraints."""
        # Convert string difficulty to enum if provided
        difficulty_enum = QuestionDifficulty(difficulty) if difficulty else None
        
        # Convert topic to list for compatibility with existing method
        topics = [topic] if topic else None
        
        # Use existing generate_question method
        return await self.generate_question(
            difficulty=difficulty_enum,
            topics=topics
        )

    async def generate_questions_batch(
        self,
        count: int,
        difficulty: Optional[str] = None,
        topics: Optional[List[str]] = None,
        shuffle: bool = True
    ) -> List[CandlestickQuestion]:
        """Generate a batch of questions with optional constraints."""
        # Convert string difficulty to enum if provided
        difficulty_enum = QuestionDifficulty(difficulty) if difficulty else None
        
        # Use existing generate_questions method
        questions = await self.generate_questions(
            count=count,
            difficulty=difficulty_enum,
            topics=topics
        )
        
        # Shuffle if requested
        if shuffle and questions:
            random.shuffle(questions)
            
        return questions

    async def generate_adaptive_questions(
        self,
        user_id: str,
        count: int,
        topics: Optional[List[str]] = None
    ) -> List[CandlestickQuestion]:
        """Generate questions adapted to a user's skill level."""
        # Use existing generate_for_user method
        return await self.generate_for_user(
            user_id=user_id,
            count=count,
            topics=topics
        )

    async def generate_quiz(
        self,
        topic: str,
        question_count: int,
        difficulty: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate a complete quiz on a specific topic."""
        # Convert string difficulty to enum if provided
        difficulty_enum = QuestionDifficulty(difficulty) if difficulty else None
        
        # Generate questions for the quiz
        questions = await self.generate_questions_batch(
            count=question_count,
            difficulty=difficulty,
            topics=[topic]
        )
        
        # Create quiz structure
        return {
            "id": str(uuid.uuid4()),
            "topic": topic,
            "difficulty": difficulty or "adaptive",
            "question_count": len(questions),
            "questions": questions,
            "created_at": datetime.now().isoformat(),
            "metadata": {
                "generator": self.__class__.__name__,
                "template_stats": {
                    "total_templates": len(self.template_db.templates) if self.template_db else 0,
                    "generation_metrics": self.generation_metrics.copy()
                }
            }
        }

    async def generate_spaced_repetition_questions(
        self,
        user_id: str,
        count: int
    ) -> List[CandlestickQuestion]:
        """Generate questions following spaced repetition principles."""
        # Get the difficulty engine for this user
        difficulty_engine = await self._get_difficulty_engine(user_id)
        
        # Get patterns that are due for review based on spaced repetition
        patterns_for_review = difficulty_engine.get_patterns_for_review()
        
        if not patterns_for_review:
            # If no patterns are due for review, generate random questions
            return await self.generate_questions_batch(count=count)
        
        # Generate questions focusing on patterns due for review
        questions = []
        for pattern in patterns_for_review[:count]:
            try:
                # Generate a question specifically for this pattern
                question_data = await self._generate_question_for_user(
                    user_id=user_id,
                    user_metrics={"skill_level": 0.5},  # Default metrics
                    pattern_diversity={"recent_patterns": []},
                    pattern_override=pattern
                )
                questions.append(self._create_question_from_data(question_data))
            except Exception as e:
                logger.error(f"Error generating spaced repetition question for pattern {pattern}: {e}")
                continue
        
        # If we couldn't generate enough pattern-specific questions, fill with random ones
        remaining = count - len(questions)
        if remaining > 0:
            random_questions = await self.generate_questions_batch(count=remaining)
            questions.extend(random_questions)
        
        return questions

# Alias AdaptiveQuestionGenerator as CandlestickQuestionGenerator for backward compatibility
CandlestickQuestionGenerator = AdaptiveQuestionGenerator 