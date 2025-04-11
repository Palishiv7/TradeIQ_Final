"""
Candlestick Pattern Database Synchronization

This module synchronizes data between Redis cache and PostgreSQL database:
- Initial load of pattern statistics
- Regular sync of assessment data
- Cache invalidation strategies
- Recovery mechanisms

This module complements the repository pattern in the base assessment architecture
by providing efficient data synchronization between the fast cache layer and the
persistent database storage. It ensures data consistency while maintaining
performance benefits of multi-level caching.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta

# Import from base assessment architecture
from backend.assessments.base.repositories import SessionRepository, QuestionRepository
from backend.assessments.base.models import AssessmentSession, BaseQuestion

from database.repositories.candlestick_repository import candlestick_repository
from database.seed_data.candlestick_patterns import get_initial_pattern_stats
from backend.assessments.candlestick_patterns.candlestick_db import candlestick_cache
from backend.common.logger import get_logger

from backend.common.tasks.registry import task, register_background_tasks
from backend.common.cache import get_cache_service
from backend.common.database import get_db_session
from backend.common.finance.market_data import MarketDataProvider
from backend.common.finance.patterns import PatternType

from backend.assessments.candlestick_patterns.repository import (
    session_repository, 
    question_repository, 
    performance_repository
)
from backend.assessments.candlestick_patterns.candlestick_pattern_identification import (
    PatternDetector,
    RuleBasedDetector,
    MLModelDetector,
    CompositeDetector
)
from backend.assessments.candlestick_patterns.candlestick_utils import (
    CandlestickData,
    Candle,
    generate_options
)

# Configure logging
logger = get_logger(__name__)

# Constants
SYNC_INTERVAL = 300  # 5 minutes
MAX_RETRY_ATTEMPTS = 3
RETRY_DELAY = 2  # seconds

SYNC_INTERVAL_SECONDS = 3600  # 1 hour
CACHE_TTL_SECONDS = 86400  # 24 hours
MARKET_SYMBOLS = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "BTC-USD", "ETH-USD"]
TIMEFRAMES = ["1d", "4h", "1h"]
QUESTIONS_PER_PATTERN = 5
MAX_CACHED_QUESTIONS = 1000


async def sync_pattern_statistics_to_db() -> bool:
    """
    Sync pattern statistics from Redis cache to PostgreSQL database.
    
    Returns:
        bool: True if successful, False otherwise
    """
    logger.info("Syncing pattern statistics from Redis to PostgreSQL")
    
    try:
        # Get statistics from Redis
        pattern_stats = await candlestick_cache.get_all_pattern_stats()
        
        if not pattern_stats:
            logger.warning("No pattern statistics found in Redis")
            return False
        
        # Update each pattern's statistics in database
        for pattern_name, stats in pattern_stats.items():
            # Find pattern in database
            db_pattern = await candlestick_repository.get_pattern_statistics(pattern_name)
            
            if db_pattern:
                # Update database with Redis stats
                db_pattern.total_attempts = stats.get('attempts', 0)
                db_pattern.correct_attempts = stats.get('correct', 0)
                db_pattern.avg_response_time_ms = stats.get('avg_response_time', 0)
                
                # Calculate success rate
                if db_pattern.total_attempts > 0:
                    db_pattern.success_rate = db_pattern.correct_attempts / db_pattern.total_attempts
                else:
                    db_pattern.success_rate = 0
            
        logger.info(f"Successfully synced statistics for {len(pattern_stats)} patterns")
        return True
    except Exception as e:
        logger.error(f"Error syncing pattern statistics to database: {e}")
        return False


async def sync_pattern_statistics_to_cache() -> bool:
    """
    Sync pattern statistics from PostgreSQL database to Redis cache.
    
    This is typically used during initialization or cache recovery.
    
    Returns:
        bool: True if successful, False otherwise
    """
    logger.info("Syncing pattern statistics from PostgreSQL to Redis")
    
    try:
        # Get all patterns from database
        db_patterns = await candlestick_repository.get_pattern_statistics()
        
        if not db_patterns:
            logger.warning("No pattern statistics found in database, using seed data")
            # Use seed data if database is empty
            stats_data = get_initial_pattern_stats()
        else:
            # Convert database patterns to cache format
            stats_data = {}
            for pattern in db_patterns:
                stats_data[pattern.pattern_name] = {
                    'attempts': pattern.total_attempts,
                    'correct': pattern.correct_attempts,
                    'avg_response_time': pattern.avg_response_time_ms,
                    'success_rate': pattern.success_rate
                }
        
        # Store in Redis
        success = await candlestick_cache.set_pattern_stats(stats_data)
        
        if success:
            logger.info(f"Successfully synced {len(stats_data)} patterns to Redis")
        else:
            logger.error("Failed to sync pattern statistics to Redis")
        
        return success
    except Exception as e:
        logger.error(f"Error syncing pattern statistics to cache: {e}")
        return False


async def sync_session_to_db(session_id: str) -> bool:
    """
    Sync a specific assessment session from Redis to PostgreSQL.
    
    Args:
        session_id: Session identifier
        
    Returns:
        bool: True if successful, False otherwise
    """
    logger.info(f"Syncing session {session_id} to database")
    
    try:
        # Get session data from Redis
        session_data = await candlestick_cache.get_session(session_id)
        
        if not session_data:
            logger.warning(f"No session data found in Redis for {session_id}")
            return False
        
        # Extract key information
        user_id = session_data.get('user_id')
        status = session_data.get('status', 'in_progress')
        
        # Check if assessment exists in database
        assessment_id = session_data.get('id', session_id)
        db_assessment = await candlestick_repository.get_assessment(assessment_id)
        
        if db_assessment:
            # Update existing assessment
            await candlestick_repository.update_assessment(
                assessment_id=assessment_id,
                status=status,
                end_difficulty=session_data.get('current_difficulty'),
                completed_questions=session_data.get('completed_questions', 0),
                correct_answers=session_data.get('correct_answers', 0),
                incorrect_answers=session_data.get('incorrect_answers', 0),
                avg_response_time_ms=session_data.get('avg_response_time_ms', 0),
                session_data=session_data
            )
        else:
            # Create new assessment record
            await candlestick_repository.create_assessment(
                user_id=user_id,
                difficulty=session_data.get('start_difficulty', 0.5),
                total_questions=session_data.get('total_questions', 10)
            )
        
        logger.info(f"Successfully synced session {session_id} to database")
        return True
    except Exception as e:
        logger.error(f"Error syncing session {session_id} to database: {e}")
        return False


async def initialize_cache_from_db() -> bool:
    """
    Initialize Redis cache with data from PostgreSQL database.
    
    This function is called during application startup to ensure
    the cache has the most up-to-date data.
    
    Returns:
        bool: True if successful, False otherwise
    """
    logger.info("Initializing Redis cache from PostgreSQL database")
    
    # Retry with exponential backoff
    for attempt in range(1, MAX_RETRY_ATTEMPTS + 1):
        try:
            # Sync pattern statistics
            success = await sync_pattern_statistics_to_cache()
            
            if success:
                logger.info("Successfully initialized cache from database")
                return True
            else:
                logger.warning(f"Failed to initialize cache (attempt {attempt}/{MAX_RETRY_ATTEMPTS})")
                
                if attempt < MAX_RETRY_ATTEMPTS:
                    retry_delay = RETRY_DELAY * (2 ** (attempt - 1))
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
        
        except Exception as e:
            logger.error(f"Error initializing cache from database (attempt {attempt}): {e}")
            
            if attempt < MAX_RETRY_ATTEMPTS:
                retry_delay = RETRY_DELAY * (2 ** (attempt - 1))
                logger.info(f"Retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
    
    logger.error("All attempts to initialize cache from database failed")
    return False


async def start_background_sync():
    """Start background task for regular Redis to PostgreSQL synchronization."""
    while True:
        try:
            logger.debug("Running scheduled Redis to PostgreSQL sync")
            await sync_pattern_statistics_to_db()
            
            # Wait for next sync interval
            await asyncio.sleep(SYNC_INTERVAL)
        except asyncio.CancelledError:
            # Handle task cancellation
            logger.info("Background sync task cancelled")
            break
        except Exception as e:
            logger.error(f"Error in background sync task: {e}")
            await asyncio.sleep(SYNC_INTERVAL)  # Wait before retrying


def register_background_tasks(app):
    """
    Register background sync tasks with the FastAPI application.
    
    Args:
        app: FastAPI application instance
    """
    @app.on_event("startup")
    async def start_sync_tasks():
        # Initialize cache from database on startup
        await initialize_cache_from_db()
        
        # Start background sync task
        app.state.background_tasks = asyncio.create_task(start_background_sync())
    
    @app.on_event("shutdown")
    async def stop_sync_tasks():
        # Cancel background task on shutdown
        if hasattr(app.state, "background_tasks"):
            app.state.background_tasks.cancel()
            try:
                await app.state.background_tasks
            except asyncio.CancelledError:
                pass


@task(
    queue="sync.patterns",
    tags=["sync", "patterns", "questions"],
    description="Synchronize candlestick pattern questions",
    retry=True,
    max_retries=3,
    scheduled=True,
    cron="0 */6 * * *"  # Run every 6 hours
)
async def sync_candlestick_questions():
    """
    Synchronize candlestick pattern questions between the database and cache.
    
    This task:
    1. Ensures all questions in the database are cached
    2. Removes stale questions from the cache
    3. Generates new questions if needed
    """
    logger.info("Starting candlestick pattern questions synchronization")
    
    # Get all pattern types
    pattern_types = [p.name for p in PatternType]
    
    # Get all difficulty levels
    difficulty_levels = [d.name for d in QuestionDifficulty]
    
    # Check question counts for each pattern type and difficulty
    for pattern_type in pattern_types:
        for difficulty in difficulty_levels:
            # Check how many questions we have for this combination
            questions = await question_repository.find({
                "pattern_type": pattern_type,
                "difficulty": difficulty
            }, limit=100)
            
            count = len(questions)
            logger.debug(f"Found {count} questions for pattern {pattern_type} at {difficulty} difficulty")
            
            # If we don't have enough questions, generate more
            if count < QUESTIONS_PER_PATTERN:
                needed = QUESTIONS_PER_PATTERN - count
                logger.info(f"Generating {needed} new questions for pattern {pattern_type} at {difficulty} difficulty")
                
                try:
                    await generate_pattern_questions(pattern_type, difficulty, needed)
                except Exception as e:
                    logger.error(f"Error generating questions for {pattern_type} at {difficulty} difficulty: {e}")
    
    # Remove old questions if we have too many
    try:
        total_questions = await _count_total_questions()
        if total_questions > MAX_CACHED_QUESTIONS:
            await _prune_old_questions(total_questions - MAX_CACHED_QUESTIONS)
    except Exception as e:
        logger.error(f"Error pruning old questions: {e}")
    
    logger.info("Candlestick pattern questions synchronization completed")


@task(
    queue="sync.patterns",
    tags=["sync", "patterns", "market_data"],
    description="Fetch and cache market data for pattern detection",
    retry=True,
    max_retries=3,
    scheduled=True,
    cron="0 */12 * * *"  # Run every 12 hours
)
async def sync_market_data():
    """
    Fetch and cache market data for pattern detection.
    
    This task:
    1. Fetches latest market data for configured symbols and timeframes
    2. Caches the data for pattern detection
    3. Identifies patterns in the new data
    """
    logger.info("Starting market data synchronization")
    
    # Get market data provider
    market_data_provider = MarketDataProvider()
    
    # Get cache service
    cache = get_cache_service()
    
    # Track total patterns found
    total_patterns = 0
    
    # Get pattern detector
    detector = CompositeDetector()
    
    # Fetch and process data for each symbol and timeframe
    for symbol in MARKET_SYMBOLS:
        for timeframe in TIMEFRAMES:
            logger.debug(f"Fetching market data for {symbol} {timeframe}")
            
            try:
                # Fetch data
                data = await market_data_provider.get_candlestick_data(symbol, timeframe, limit=100)
                
                # Create CandlestickData object
                candles = []
                for item in data.get("candles", []):
                    candle = Candle(
                        time=item.get("time") or int(datetime.fromisoformat(item.get("date", "")).timestamp()),
                        open=item.get("open", 0),
                        high=item.get("high", 0),
                        low=item.get("low", 0),
                        close=item.get("close", 0),
                        volume=item.get("volume", 0)
                    )
                    candles.append(candle)
                
                candlestick_data = CandlestickData(symbol, timeframe, candles)
                
                # Cache the data
                cache_key = f"market_data:{symbol}:{timeframe}"
                await cache.set(cache_key, candlestick_data.to_dict(), ttl=CACHE_TTL_SECONDS)
                
                # Detect patterns
                patterns = await detector.detect_patterns(candlestick_data)
                
                # Cache patterns
                for pattern in patterns:
                    pattern_key = f"pattern:{pattern.pattern_type}:{symbol}:{timeframe}:{pattern.pattern_id}"
                    await cache.set(pattern_key, pattern.to_dict(), ttl=CACHE_TTL_SECONDS)
                
                total_patterns += len(patterns)
                logger.debug(f"Found {len(patterns)} patterns in {symbol} {timeframe}")
                
                # Generate questions from some patterns
                if patterns:
                    selected_patterns = random.sample(patterns, min(3, len(patterns)))
                    for pattern in selected_patterns:
                        await generate_questions_from_pattern(pattern)
                
            except Exception as e:
                logger.error(f"Error processing market data for {symbol} {timeframe}: {e}")
    
    logger.info(f"Market data synchronization completed. Found {total_patterns} patterns")


@task(
    queue="sync.patterns",
    tags=["sync", "patterns", "leaderboard"],
    description="Update candlestick pattern leaderboard",
    retry=True,
    max_retries=3,
    scheduled=True,
    cron="0 */2 * * *"  # Run every 2 hours
)
async def update_leaderboard():
    """
    Update the candlestick pattern assessment leaderboard.
    
    This task:
    1. Calculates user scores and rankings
    2. Updates the leaderboard table
    3. Caches leaderboard data for quick access
    """
    logger.info("Starting leaderboard update")
    
    try:
        # Get database connection
        db = get_db_session()
        
        # Get cache service
        cache = get_cache_service()
        
        # Query for user performance data
        query = """
            SELECT s.user_id, 
                COUNT(DISTINCT s.session_id) as sessions_completed,
                AVG(a.score) as average_score,
                COUNT(a.attempt_id) as questions_answered,
                SUM(CASE WHEN a.is_correct THEN 1 ELSE 0 END) as correct_answers,
                SUM(a.score) as total_score
            FROM candlestick_sessions s
            JOIN candlestick_attempts a ON s.session_id = a.session_id
            WHERE s.completed_at IS NOT NULL
            GROUP BY s.user_id
            ORDER BY total_score DESC
        """
        
        result = await db.fetch_all(query)
        
        # Update leaderboard table
        for row in result:
            user_id = row['user_id']
            total_score = row['total_score']
            average_score = row['average_score']
            sessions_completed = row['sessions_completed']
            questions_answered = row['questions_answered']
            correct_answers = row['correct_answers']
            accuracy = (correct_answers / questions_answered * 100) if questions_answered > 0 else 0
            
            # Update the leaderboard table
            upsert_query = """
                INSERT INTO candlestick_leaderboard
                (user_id, total_score, average_score, sessions_completed, questions_answered, accuracy, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, NOW())
                ON CONFLICT (user_id)
                DO UPDATE SET
                    total_score = EXCLUDED.total_score,
                    average_score = EXCLUDED.average_score, 
                    sessions_completed = EXCLUDED.sessions_completed,
                    questions_answered = EXCLUDED.questions_answered,
                    accuracy = EXCLUDED.accuracy,
                    updated_at = NOW()
            """
            
            await db.execute(
                upsert_query, 
                (user_id, total_score, average_score, sessions_completed, questions_answered, accuracy)
            )
        
        # Cache the top 100 leaderboard entries for quick access
        top_leaderboard_query = """
            SELECT user_id, total_score, average_score, sessions_completed, questions_answered, accuracy
            FROM candlestick_leaderboard
            ORDER BY total_score DESC
            LIMIT 100
        """
        
        top_entries = await db.fetch_all(top_leaderboard_query)
        
        leaderboard_data = [dict(row) for row in top_entries]
        
        # Cache the leaderboard
        await cache.set("candlestick:leaderboard", leaderboard_data, ttl=CACHE_TTL_SECONDS)
        
        logger.info(f"Leaderboard updated with {len(leaderboard_data)} entries")
        
    except Exception as e:
        logger.error(f"Error updating leaderboard: {e}")


@task(
    queue="sync.patterns",
    tags=["sync", "patterns", "cleanup"],
    description="Clean up old candlestick pattern assessment data",
    retry=True,
    max_retries=3,
    scheduled=True,
    cron="0 2 * * *"  # Run at 2 AM every day
)
async def cleanup_old_data():
    """
    Clean up old assessment data to prevent database bloat.
    
    This task:
    1. Archives very old sessions (older than 90 days)
    2. Removes temporary data
    3. Cleans up expired cache entries
    """
    logger.info("Starting old data cleanup")
    
    try:
        # Get database connection
        db = get_db_session()
        
        # Archive sessions older than 90 days
        ninety_days_ago = datetime.now() - timedelta(days=90)
        
        # Count old sessions
        count_query = """
            SELECT COUNT(*) as count FROM candlestick_sessions
            WHERE created_at < %s
        """
        
        result = await db.fetch_one(count_query, (ninety_days_ago,))
        old_session_count = result['count'] if result else 0
        
        if old_session_count > 0:
            logger.info(f"Archiving {old_session_count} old sessions")
            
            # Archive sessions to candlestick_sessions_archive table
            archive_query = """
                INSERT INTO candlestick_sessions_archive
                SELECT * FROM candlestick_sessions
                WHERE created_at < %s
            """
            
            await db.execute(archive_query, (ninety_days_ago,))
            
            # Delete old sessions
            delete_query = """
                DELETE FROM candlestick_sessions
                WHERE created_at < %s
            """
            
            await db.execute(delete_query, (ninety_days_ago,))
        
        # Clean up orphaned attempts (where session no longer exists)
        cleanup_attempts_query = """
            DELETE FROM candlestick_attempts
            WHERE session_id NOT IN (SELECT session_id FROM candlestick_sessions)
        """
        
        result = await db.execute(cleanup_attempts_query)
        logger.info(f"Cleaned up orphaned attempts")
        
        # Clean up expired cache entries (done automatically by Redis TTL)
        
        logger.info("Old data cleanup completed")
        
    except Exception as e:
        logger.error(f"Error during old data cleanup: {e}")


# Helper functions

async def generate_pattern_questions(pattern_type: str, difficulty: str, count: int = 5):
    """
    Generate candlestick pattern questions for a specific pattern and difficulty.
    
    Args:
        pattern_type: Type of pattern
        difficulty: Difficulty level
        count: Number of questions to generate
    """
    # Get all available patterns for distractors
    all_patterns = [p.name for p in PatternType]
    
    # Get cache service
    cache = get_cache_service()
    
    # Try to find cached patterns of this type
    pattern_keys = await cache.keys(f"pattern:{pattern_type}:*")
    
    patterns = []
    for key in pattern_keys:
        pattern_data = await cache.get(key)
        if pattern_data:
            patterns.append(pattern_data)
    
    # If we don't have enough cached patterns, use mock data
    if len(patterns) < count:
        # Create new pattern data
        from backend.assessments.candlestick_patterns.tasks import _generate_mock_chart_data
        
        # Convert string difficulty to enum
        try:
            difficulty_enum = QuestionDifficulty[difficulty]
        except KeyError:
            difficulty_enum = QuestionDifficulty.INTERMEDIATE
        
        # Convert string pattern type to enum
        try:
            pattern_enum = PatternType[pattern_type]
        except KeyError:
            pattern_enum = PatternType.DOJI
        
        # Create mock charts
        for _ in range(count):
            chart_data = _generate_mock_chart_data(
                pattern_type=pattern_enum,
                difficulty=difficulty_enum,
                time_period="daily",
                with_indicators=False
            )
            patterns.append({
                "pattern_type": pattern_type,
                "chart_data": chart_data
            })
    
    # Generate questions from patterns
    for i, pattern in enumerate(patterns):
        if i >= count:
            break
        
        # Create options
        options = generate_options(pattern["pattern_type"], all_patterns)
        
        # Create question ID
        from uuid import uuid4
        question_id = str(uuid4())
        
        # Create question
        from backend.assessments.candlestick_patterns.candlestick_pattern_recognition import (
            CandlestickPatternQuestion, CandlestickQuestionData
        )
        
        question_data = CandlestickQuestionData(
            pattern_type=pattern["pattern_type"],
            chart_data=pattern["chart_data"],
            options=options,
            correct_option=pattern["pattern_type"]
        )
        
        question = CandlestickPatternQuestion(
            id=question_id,
            question_type="candlestick_pattern",
            question_text="Identify the candlestick pattern shown in this chart:",
            difficulty=QuestionDifficulty[difficulty],
            question_data=question_data
        )
        
        # Save to repository
        await question_repository.save_question(question)
        logger.debug(f"Generated question {question_id} for pattern {pattern_type} at {difficulty} difficulty")


async def generate_questions_from_pattern(pattern):
    """
    Generate questions from a detected pattern.
    
    Args:
        pattern: Pattern data
    """
    # Get all available patterns for distractors
    all_patterns = [p.name for p in PatternType]
    
    # Create question for each difficulty level
    for difficulty in ["EASY", "INTERMEDIATE", "ADVANCED"]:
        # Create options (different distractors for different difficulties)
        distractor_difficulty = 0.3 if difficulty == "EASY" else 0.6 if difficulty == "INTERMEDIATE" else 0.9
        options = generate_options(pattern.pattern_type.name, all_patterns, difficulty=distractor_difficulty)
        
        # Create question ID
        from uuid import uuid4
        question_id = str(uuid4())
        
        # Create chart data
        chart_data = {
            "pattern_type": pattern.pattern_type.name,
            "symbol": pattern.symbol,
            "timeframe": pattern.timeframe,
            "candles": pattern.candle_data,
            "pattern_start_index": 0,
            "pattern_end_index": len(pattern.candle_data) - 1
        }
        
        # Create question
        from backend.assessments.candlestick_patterns.candlestick_pattern_recognition import (
            CandlestickPatternQuestion, CandlestickQuestionData
        )
        
        question_data = CandlestickQuestionData(
            pattern_type=pattern.pattern_type.name,
            chart_data=chart_data,
            options=options,
            correct_option=pattern.pattern_type.name
        )
        
        question = CandlestickPatternQuestion(
            id=question_id,
            question_type="candlestick_pattern",
            question_text="What candlestick pattern is shown in this chart?",
            difficulty=QuestionDifficulty[difficulty],
            question_data=question_data
        )
        
        # Save to repository
        await question_repository.save_question(question)
        logger.debug(f"Generated question {question_id} from detected pattern {pattern.pattern_type.name}")


async def _count_total_questions():
    """Count the total number of candlestick pattern questions in the database."""
    db = get_db_session()
    result = await db.fetch_one("SELECT COUNT(*) as count FROM candlestick_questions")
    return result['count'] if result else 0


async def _prune_old_questions(count: int):
    """
    Remove the oldest questions from the database.
    
    Args:
        count: Number of questions to remove
    """
    db = get_db_session()
    
    # Get IDs of oldest questions
    query = """
        SELECT question_id FROM candlestick_questions
        ORDER BY created_at ASC
        LIMIT %s
    """
    
    results = await db.fetch_all(query, (count,))
    
    question_ids = [row['question_id'] for row in results]
    logger.info(f"Pruning {len(question_ids)} old questions")
    
    # Delete questions
    for question_id in question_ids:
        await question_repository.delete(question_id)
    
    logger.info(f"Pruned {len(question_ids)} old questions")


def register_background_tasks():
    """Register all background tasks."""
    return [
        sync_candlestick_questions,
        sync_market_data,
        update_leaderboard,
        cleanup_old_data
    ] 