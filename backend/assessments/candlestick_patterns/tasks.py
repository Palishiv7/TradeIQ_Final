"""
Candlestick Pattern Assessment Tasks Module

This module contains task definitions specific to the candlestick pattern
recognition assessment, including pattern generation, validation, and evaluation.
"""

import datetime
import logging
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from backend.common.tasks.registry import task
from backend.common.finance.patterns import PatternType, PatternStrength, CandlestickPattern
from backend.assessments.base.models import QuestionDifficulty, BaseQuestion
from backend.assessments.base.tasks import (
    generate_session_questions,
    evaluate_answer,
    base_assessment_task
)

# Set up logging
logger = logging.getLogger(__name__)


@task(
    queue="assessments.patterns",
    tags=["assessment", "patterns", "generation"],
    description="Generate candlestick pattern chart for assessment",
    retry=True,
    max_retries=3
)
def generate_pattern_chart(
    pattern_type: str,
    difficulty: str,
    stock_symbol: Optional[str] = None,
    time_period: str = "daily",
    chart_size: Tuple[int, int] = (800, 600),
    with_indicators: bool = False
) -> Dict[str, Any]:
    """
    Generate a chart with a specific candlestick pattern for assessment.
    
    Args:
        pattern_type: Type of candlestick pattern to generate
        difficulty: Difficulty level (easy/intermediate/advanced)
        stock_symbol: Optional specific stock symbol to use
        time_period: Time period for the chart (daily/hourly/etc)
        chart_size: Size of the chart in pixels (width, height)
        with_indicators: Whether to include technical indicators
        
    Returns:
        Dictionary with chart details and metadata
    """
    logger.info(f"Generating {pattern_type} pattern chart at {difficulty} difficulty")
    
    # Validate pattern type
    try:
        pattern_enum = PatternType[pattern_type]
    except KeyError:
        logger.error(f"Invalid pattern type: {pattern_type}")
        raise ValueError(f"Invalid pattern type: {pattern_type}")
    
    # Validate difficulty
    try:
        difficulty_enum = QuestionDifficulty(difficulty)
    except ValueError:
        logger.error(f"Invalid difficulty: {difficulty}")
        raise ValueError(f"Invalid difficulty: {difficulty}")
    
    # In a real implementation, this would generate actual chart data
    # For demonstration, we'll return mock chart data
    
    # Mock chart generation with a simulated pattern
    chart_data = _generate_mock_chart_data(
        pattern_type=pattern_enum,
        difficulty=difficulty_enum,
        time_period=time_period,
        with_indicators=with_indicators
    )
    
    # Generate a simulated chart image (in a real implementation, this would create an actual image)
    image_path = f"/static/charts/{pattern_type.lower()}_{difficulty.lower()}_{int(datetime.datetime.now().timestamp())}.png"
    
    result = {
        "pattern_type": pattern_type,
        "difficulty": difficulty,
        "stock_symbol": stock_symbol or "EXAMPLE",
        "time_period": time_period,
        "chart_size": chart_size,
        "with_indicators": with_indicators,
        "image_path": image_path,
        "generated_at": datetime.datetime.now().isoformat(),
        "chart_data": chart_data,
        "pattern_start_index": chart_data.get("pattern_start_index", 0),
        "pattern_end_index": chart_data.get("pattern_end_index", 0)
    }
    
    logger.info(f"Generated {pattern_type} pattern chart with {len(chart_data.get('dates', []))} data points")
    return result


@task(
    queue="assessments.patterns",
    tags=["assessment", "patterns", "questions"],
    description="Generate candlestick pattern assessment questions",
    retry=True,
    max_retries=3
)
def generate_pattern_questions(
    session_id: str,
    difficulty: str = "adaptive",
    patterns: Optional[List[str]] = None,
    question_count: int = 10,
    with_indicators: bool = False
) -> Dict[str, Any]:
    """
    Generate candlestick pattern recognition questions for an assessment session.
    
    This implementation extends the base assessment question generation pattern.
    
    Args:
        session_id: The ID of the session
        difficulty: Difficulty level or "adaptive"
        patterns: Optional list of specific patterns to include
        question_count: Number of questions to generate
        with_indicators: Whether to include technical indicators in the charts
        
    Returns:
        Dictionary with generated questions
    """
    logger.info(f"Generating {question_count} pattern questions for session {session_id}")
    
    # Get all available patterns if not specified
    if patterns is None:
        patterns = [p.name for p in PatternType]
    
    # First use the base question generation mechanism
    base_questions = generate_session_questions(
        session_id=session_id,
        assessment_type="candlestick",
        question_count=question_count,
        difficulty=difficulty
    )
    
    # Then enhance it with candlestick-specific information
    questions = []
    for i, base_q in enumerate(base_questions.get("questions", [])):
        if i >= question_count:
            break
            
        # Select a random pattern from the list
        pattern_type = random.choice(patterns)
        
        # Generate a chart for the pattern
        chart_result = generate_pattern_chart(
            pattern_type=pattern_type,
            difficulty=difficulty if difficulty != "adaptive" else random.choice(["EASY", "INTERMEDIATE", "ADVANCED"]),
            with_indicators=with_indicators
        )
        
        # Create pattern options (correct + distractors)
        options = _create_pattern_options(pattern_type)
        
        # Create the question with base structure from base_q
        question = {
            "id": base_q.get("id"),
            "question_text": base_q.get("question_text", f"What pattern is shown in this chart?"),
            "difficulty": chart_result["difficulty"],
            "pattern_type": pattern_type,
            "options": options,
            "correct_option": pattern_type,
            "chart_data": chart_result,
            "created_at": datetime.datetime.now().isoformat()
        }
        questions.append(question)
    
    return {
        "session_id": session_id,
        "questions": questions,
        "question_count": len(questions),
        "difficulty": difficulty
    }


@task(
    queue="assessments.patterns",
    tags=["assessment", "patterns", "evaluation"],
    description="Evaluate answer to candlestick pattern question",
    retry=True,
    max_retries=3
)
def evaluate_pattern_answer(
    session_id: str,
    question_id: str,
    user_answer: Dict[str, Any],
    time_taken_ms: Optional[int] = None
) -> Dict[str, Any]:
    """
    Evaluate a user's answer to a pattern recognition question.
    
    This extends the base answer evaluation functionality.
    
    Args:
        session_id: Session identifier
        question_id: Question identifier
        user_answer: User's answer containing selected pattern
        time_taken_ms: Time taken to answer in milliseconds
        
    Returns:
        Dictionary with evaluation results
    """
    logger.info(f"Evaluating answer for question {question_id} in session {session_id}")
    
    # Use the base evaluation framework first
    base_result = evaluate_answer(
        session_id=session_id,
        question_id=question_id,
        user_answer=user_answer,
        time_taken_ms=time_taken_ms
    )
    
    # Add candlestick-specific evaluation details
    is_correct = False
    feedback = ""
    score = 0
    
    # Extract the selected pattern from the user answer
    selected_pattern = user_answer.get("selected_pattern")
    
    # Get the question from the repository
    from backend.assessments.candlestick_patterns.repository import CandlestickQuestionRepository
    
    # Create repository instance
    question_repo = CandlestickQuestionRepository()
    
    # Get the question
    question = question_repo.get_question(question_id)
    
    # Determine correctness
    if question:
        correct_pattern = question.pattern_type
        is_correct = selected_pattern == correct_pattern
        score = 10 if is_correct else 0
        
        if is_correct:
            feedback = f"Correct! That is a {correct_pattern} pattern."
        else:
            feedback = f"Incorrect. The pattern shown is a {correct_pattern}."
    
    # Combine with base result
    result = {
        **base_result,
        "is_correct": is_correct,
        "score": score,
        "feedback": feedback,
        "explanation": _generate_pattern_explanation(selected_pattern, question.pattern_type if question else None),
        "time_taken_ms": time_taken_ms,
        "evaluated_at": datetime.datetime.now().isoformat()
    }
    
    logger.info(f"Answer evaluated as {'correct' if is_correct else 'incorrect'} with score {score}")
    return result


@base_assessment_task(
    queue="assessments.patterns",
    tags=["assessment", "patterns", "analysis"],
    description="Analyze pattern recognition performance",
    retry=True,
    max_retries=3
)
def analyze_pattern_performance(
    session_id: str
) -> Dict[str, Any]:
    """
    Analyze pattern recognition performance for a session.
    
    Args:
        session_id: Session identifier
        
    Returns:
        Dictionary with performance analysis
    """
    logger.info(f"Analyzing pattern recognition performance for session {session_id}")
    
    # Get the session from the repository
    from backend.assessments.candlestick_patterns.repository import CandlestickSessionRepository
    
    # Create repository instance
    session_repo = CandlestickSessionRepository()
    
    # Get the session
    session = session_repo.get_session(session_id)
    
    if not session:
        logger.error(f"Session {session_id} not found")
        return {
            "error": "Session not found",
            "session_id": session_id
        }
    
    # Extract performance metrics
    total_questions = len(session.questions)
    answered_questions = len(session.answers)
    correct_answers = sum(1 for answer in session.answers.values() if answer.get("is_correct", False))
    
    # Calculate metrics
    accuracy = (correct_answers / answered_questions) * 100 if answered_questions > 0 else 0
    completion_rate = (answered_questions / total_questions) * 100 if total_questions > 0 else 0
    
    # Get pattern-specific metrics
    pattern_performance = {}
    for q_id, question in enumerate(session.questions):
        pattern_type = question.get("pattern_type")
        if pattern_type:
            if pattern_type not in pattern_performance:
                pattern_performance[pattern_type] = {
                    "total": 0,
                    "correct": 0,
                    "incorrect": 0,
                    "time_taken_ms": []
                }
            
            pattern_performance[pattern_type]["total"] += 1
            
            # Check if question was answered
            if str(q_id) in session.answers:
                answer = session.answers[str(q_id)]
                is_correct = answer.get("is_correct", False)
                time_taken = answer.get("time_taken_ms", 0)
                
                if is_correct:
                    pattern_performance[pattern_type]["correct"] += 1
                else:
                    pattern_performance[pattern_type]["incorrect"] += 1
                
                pattern_performance[pattern_type]["time_taken_ms"].append(time_taken)
    
    # Calculate average time per pattern
    for pattern, stats in pattern_performance.items():
        times = stats["time_taken_ms"]
        stats["avg_time_ms"] = sum(times) / len(times) if times else 0
        stats["accuracy"] = (stats["correct"] / stats["total"]) * 100 if stats["total"] > 0 else 0
    
    result = {
        "session_id": session_id,
        "total_questions": total_questions,
        "answered_questions": answered_questions,
        "correct_answers": correct_answers,
        "accuracy": accuracy,
        "completion_rate": completion_rate,
        "pattern_performance": pattern_performance,
        "analysis_date": datetime.datetime.now().isoformat()
    }
    
    logger.info(f"Performance analysis completed for session {session_id} with accuracy {accuracy:.2f}%")
    return result


# Helper Functions

def _create_pattern_options(correct_pattern: str, num_options: int = 4) -> List[str]:
    """
    Create a list of pattern options including the correct one.
    
    Args:
        correct_pattern: The correct pattern
        num_options: Number of options to generate
        
    Returns:
        List of pattern options
    """
    # Get all pattern types
    all_patterns = [p.name for p in PatternType]
    
    # Ensure the correct pattern is in the list
    if correct_pattern not in all_patterns:
        all_patterns.append(correct_pattern)
    
    # Remove the correct pattern from the list of all patterns
    other_patterns = [p for p in all_patterns if p != correct_pattern]
    
    # Randomly select additional options
    selected_others = random.sample(other_patterns, min(num_options - 1, len(other_patterns)))
    
    # Combine correct pattern with others and shuffle
    options = [correct_pattern] + selected_others
    random.shuffle(options)
    
    return options


def _generate_mock_chart_data(
    pattern_type: PatternType,
    difficulty: QuestionDifficulty,
    time_period: str,
    with_indicators: bool
) -> Dict[str, Any]:
    """
    Generate mock chart data for a pattern.
    
    Args:
        pattern_type: Type of pattern
        difficulty: Difficulty level
        time_period: Time period for the chart
        with_indicators: Whether to include indicators
        
    Returns:
        Dictionary with chart data
    """
    # This would be replaced with actual chart generation logic
    # For now, we'll return a mock structure
    
    # Generate dates
    now = datetime.datetime.now()
    dates = [(now - datetime.timedelta(days=i)).strftime("%Y-%m-%d") for i in range(30, 0, -1)]
    
    # Generate price data
    base_price = 100
    prices = []
    for i in range(30):
        price = base_price + random.uniform(-5, 5)
        prices.append(price)
    
    # Create candles
    candles = []
    for i, date in enumerate(dates):
        open_price = prices[i]
        close_price = open_price * (1 + random.uniform(-0.02, 0.02))
        high_price = max(open_price, close_price) * (1 + random.uniform(0, 0.01))
        low_price = min(open_price, close_price) * (1 - random.uniform(0, 0.01))
        
        candles.append({
            "date": date,
            "open": open_price,
            "high": high_price,
            "low": low_price,
            "close": close_price,
            "volume": random.randint(1000, 10000)
        })
    
    # Determine pattern start and end indices
    pattern_length = random.randint(1, 3)  # Most patterns are 1-3 candles
    pattern_end_index = random.randint(pattern_length, len(candles) - 5)
    pattern_start_index = pattern_end_index - pattern_length + 1
    
    # Create indicators if requested
    indicators = {}
    if with_indicators:
        # Simple moving averages
        sma_20 = [sum(candle["close"] for candle in candles[max(0, i-19):i+1]) / min(20, i+1) for i in range(len(candles))]
        sma_50 = [sum(candle["close"] for candle in candles[max(0, i-49):i+1]) / min(50, i+1) for i in range(len(candles))]
        
        indicators = {
            "sma_20": sma_20,
            "sma_50": sma_50
        }
    
    return {
        "pattern_type": pattern_type.name,
        "difficulty": difficulty.name,
        "dates": dates,
        "candles": candles,
        "pattern_start_index": pattern_start_index,
        "pattern_end_index": pattern_end_index,
        "indicators": indicators
    }


def _generate_pattern_explanation(selected_pattern: str, correct_pattern: Optional[str]) -> str:
    """
    Generate an explanation of the pattern.
    
    Args:
        selected_pattern: User selected pattern
        correct_pattern: Correct pattern
        
    Returns:
        Explanation text
    """
    # This would be enhanced to provide detailed pattern explanations
    # For now, we'll return a simple explanation
    
    if not correct_pattern:
        return "Pattern information not available."
    
    # Basic explanations for common patterns
    explanations = {
        "HAMMER": "A Hammer is a bullish pattern that forms during a downtrend, signaling a potential reversal. It's characterized by a small body near the top with a long lower shadow and little to no upper shadow.",
        "SHOOTING_STAR": "A Shooting Star is a bearish pattern that forms during an uptrend, signaling a potential reversal. It has a small body near the bottom with a long upper shadow and little to no lower shadow.",
        "DOJI": "A Doji shows market indecision with opening and closing prices that are virtually equal. The length of the shadows can vary.",
        "ENGULFING": "An Engulfing pattern occurs when the body of a candle completely engulfs the body of the previous candle. Bullish when a green candle engulfs a red one; bearish when a red candle engulfs a green one.",
        "MORNING_STAR": "A Morning Star is a bullish pattern that signals a potential reversal from a downtrend. It consists of three candles: a large bearish candle, a small-bodied candle, and a large bullish candle.",
        "EVENING_STAR": "An Evening Star is a bearish pattern that signals a potential reversal from an uptrend. It consists of three candles: a large bullish candle, a small-bodied candle, and a large bearish candle."
    }
    
    # Get the explanation for the correct pattern
    basic_explanation = explanations.get(correct_pattern, f"This is a {correct_pattern} pattern.")
    
    # Add comparison if user selected incorrectly
    if selected_pattern and selected_pattern != correct_pattern:
        selected_explanation = explanations.get(selected_pattern, f"The {selected_pattern} pattern")
        comparison = f"You selected {selected_pattern}. {selected_explanation}, which is different from the {correct_pattern} pattern shown here."
        return f"{basic_explanation}\n\n{comparison}"
    
    return basic_explanation 