"""
Candlestick Data Synchronization Tasks

This module defines background tasks for synchronizing market data
required for candlestick pattern assessments.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Set, Union, List, Final, TypedDict, cast

# Internal imports - common modules
from backend.common.finance.candlestick import CandlestickInterval
from backend.common.tasks.scheduler import task, schedule
from backend.common.cache.service import CacheService
from backend.common.db.connection import get_database_connection
from backend.common.config import Config

# Internal imports - assessment specific
from backend.assessments.candlestick_patterns.data_providers import (
    initialize_providers, 
    get_market_data_service
)

# Module constants and configuration
__version__ = "1.0.1"

# Configure module logger
logger = logging.getLogger(__name__)

# Type definitions for better type safety
class IntervalConfig(TypedDict):
    days_historical: int
    sync_frequency: int  # in hours
    
class SyncConfig(TypedDict):
    symbols: List[str]
    intervals: List[str]
    days_historical: Dict[str, int]
    sync_frequency: Dict[str, int]

# Default configuration for sync tasks
DEFAULT_CONFIG: Final[SyncConfig] = {
    # Symbols to sync data for
    "symbols": [
        # US Indices
        "^GSPC",    # S&P 500
        "^DJI",     # Dow Jones Industrial Average
        "^IXIC",    # NASDAQ Composite
        "^RUT",     # Russell 2000
        
        # Major stocks
        "AAPL",     # Apple
        "MSFT",     # Microsoft
        "AMZN",     # Amazon
        "GOOGL",    # Alphabet (Google)
        "META",     # Meta (Facebook)
        "TSLA",     # Tesla
        "NVDA",     # NVIDIA
        "JPM",      # JPMorgan Chase
        
        # ETFs
        "SPY",      # SPDR S&P 500 ETF
        "QQQ",      # Invesco QQQ (Nasdaq 100)
        "IWM",      # iShares Russell 2000 ETF
        "GLD",      # SPDR Gold Shares
    ],
    
    # Intervals to sync
    "intervals": [
        CandlestickInterval.ONE_DAY.value,
        CandlestickInterval.ONE_HOUR.value,
        CandlestickInterval.FIFTEEN_MINUTES.value,
    ],
    
    # How many days of historical data to fetch
    "days_historical": {
        CandlestickInterval.ONE_DAY.value: 365,   # 1 year
        CandlestickInterval.ONE_HOUR.value: 30,   # 1 month
        CandlestickInterval.FIFTEEN_MINUTES.value: 7,  # 1 week
    },
    
    # How frequently to sync each interval (in hours)
    "sync_frequency": {
        CandlestickInterval.ONE_DAY.value: 24,    # Daily
        CandlestickInterval.ONE_HOUR.value: 6,    # Every 6 hours
        CandlestickInterval.FIFTEEN_MINUTES.value: 2,  # Every 2 hours
    }
}

# Cache keys
CACHE_KEY_LAST_FULL_SYNC: Final[str] = "candlestick:last_full_sync"
CACHE_KEY_LAST_INCREMENTAL_SYNC: Final[str] = "candlestick:last_incremental_sync"
CACHE_KEY_LAST_PATTERN_UPDATE: Final[str] = "candlestick:last_pattern_update"

# TTL values in seconds
CACHE_TTL_WEEK: Final[int] = 86400 * 7  # 7 days
CACHE_TTL_DAY: Final[int] = 86400       # 1 day

def get_sync_config() -> SyncConfig:
    """
    Get the current sync configuration, allowing for overrides.
    
    Returns:
        SyncConfig object with current configuration 
    """
    # In the future, this could pull from environment or database
    # For now, just return the default config
    return DEFAULT_CONFIG


@task(name="sync_market_data_full")
async def sync_market_data_full() -> Dict[str, Any]:
    """
    Synchronize all market data for candlestick patterns.
    
    This task fetches data for all configured symbols and intervals,
    storing them in the database and cache for pattern detection.
    
    Returns:
        Dictionary with sync results
    """
    logger.info(f"Starting full market data synchronization (v{__version__})")
    
    results = {
        "success": False,
        "intervals": {},
        "symbols_processed": 0,
        "total_operations": 0,
        "successful_operations": 0,
        "failed_operations": 0,
        "errors": []
    }
    
    try:
        # Initialize providers if needed
        market_service = await initialize_providers()
        if not market_service:
            error_msg = "Failed to initialize market data providers"
            logger.error(error_msg)
            results["errors"].append(error_msg)
            return results
            
        # Get current configuration
        config = get_sync_config()
        symbols = config["symbols"]
        results["symbols_processed"] = len(symbols)
        
        # Sync all intervals
        for interval_name in config["intervals"]:
            try:
                interval = CandlestickInterval(interval_name)
                days_back = config["days_historical"].get(interval_name, 30)
                
                logger.info(f"Syncing {len(symbols)} symbols for interval {interval_name}, {days_back} days back")
                
                # Run the sync operation for this interval
                interval_results = await market_service.sync_market_data(
                    symbols=symbols,
                    intervals=[interval],
                    days_back=days_back
                )
                
                results["intervals"][interval_name] = _process_interval_results(interval_results)
                
                # Update totals
                interval_stats = results["intervals"][interval_name]
                results["total_operations"] += interval_stats["total"]
                results["successful_operations"] += interval_stats["success_count"]
                results["failed_operations"] += interval_stats["failure_count"]
                
                logger.info(
                    f"Sync complete for {interval_name}: "
                    f"{interval_stats['success_count']} succeeded, "
                    f"{interval_stats['failure_count']} failed"
                )
            except Exception as e:
                error_msg = f"Error syncing interval {interval_name}: {str(e)}"
                logger.error(error_msg)
                results["errors"].append(error_msg)
                results["intervals"][interval_name] = {
                    "success": False,
                    "error": str(e),
                    "total": 0,
                    "success_count": 0,
                    "failure_count": 0
                }
        
        # Update last sync timestamp
        cache = CacheService.get_instance()
        await cache.set(CACHE_KEY_LAST_FULL_SYNC, datetime.now().isoformat(), CACHE_TTL_WEEK)
        
        # Set overall success flag
        results["success"] = len(results["errors"]) == 0 and results["failed_operations"] < (results["total_operations"] * 0.2)
        
        logger.info(
            f"Full sync complete: {results['successful_operations']} operations succeeded, "
            f"{results['failed_operations']} failed"
        )
    except Exception as e:
        error_msg = f"Unexpected error in full market data sync: {str(e)}"
        logger.exception(error_msg)
        results["errors"].append(error_msg)
        results["success"] = False
    
    return results

def _process_interval_results(interval_results: Dict[str, Dict[str, bool]]) -> Dict[str, Any]:
    """
    Process the results from syncing one interval and generate statistics.
    
    Args:
        interval_results: Results dictionary from market service sync operation
        
    Returns:
        Dictionary with processed statistics
    """
    # Count successes and failures
    success_count = sum(
        1 for symbol_result in interval_results.values() 
        for status in symbol_result.values() 
        if status
    )
    failure_count = sum(
        1 for symbol_result in interval_results.values() 
        for status in symbol_result.values() 
        if not status
    )
    
    return {
        "success": failure_count == 0,
        "symbol_results": interval_results,
        "total": success_count + failure_count,
        "success_count": success_count,
        "failure_count": failure_count,
        "success_rate": (success_count / (success_count + failure_count)) if (success_count + failure_count) > 0 else 0
    }


@task(name="sync_market_data_incremental")
async def sync_market_data_incremental() -> Dict[str, Any]:
    """
    Synchronize recent market data increments.
    
    This task fetches only the most recent data for all configured symbols,
    optimizing for speed and API usage.
    
    Returns:
        Dictionary with sync results
    """
    logger.info(f"Starting incremental market data synchronization (v{__version__})")
    
    results = {
        "success": False,
        "intervals": {},
        "symbols_processed": 0,
        "total_operations": 0,
        "successful_operations": 0,
        "failed_operations": 0,
        "errors": []
    }
    
    try:
        # Get market data service
        market_service = get_market_data_service()
        if not market_service:
            error_msg = "Failed to access market data service"
            logger.error(error_msg)
            results["errors"].append(error_msg)
            return results
            
        # Get current configuration
        config = get_sync_config()
        symbols = config["symbols"]
        results["symbols_processed"] = len(symbols)
        
        # Sync all intervals
        for interval_name in config["intervals"]:
            try:
                interval = CandlestickInterval(interval_name)
                
                # For incremental sync, use shorter time ranges
                days_back = _get_incremental_days_back(interval)
                
                logger.info(f"Incrementally syncing {len(symbols)} symbols for interval {interval_name}, {days_back} days back")
                
                # Run the sync operation for this interval
                interval_results = await market_service.sync_market_data(
                    symbols=symbols,
                    intervals=[interval],
                    days_back=days_back
                )
                
                results["intervals"][interval_name] = _process_interval_results(interval_results)
                
                # Update totals
                interval_stats = results["intervals"][interval_name]
                results["total_operations"] += interval_stats["total"]
                results["successful_operations"] += interval_stats["success_count"]
                results["failed_operations"] += interval_stats["failure_count"]
                
                logger.info(
                    f"Incremental sync complete for {interval_name}: "
                    f"{interval_stats['success_count']} succeeded, "
                    f"{interval_stats['failure_count']} failed"
                )
            except Exception as e:
                error_msg = f"Error syncing interval {interval_name}: {str(e)}"
                logger.error(error_msg)
                results["errors"].append(error_msg)
                results["intervals"][interval_name] = {
                    "success": False,
                    "error": str(e),
                    "total": 0,
                    "success_count": 0,
                    "failure_count": 0
                }
        
        # Update last incremental sync timestamp
        cache = CacheService.get_instance()
        await cache.set(CACHE_KEY_LAST_INCREMENTAL_SYNC, datetime.now().isoformat(), CACHE_TTL_DAY)
        
        # Set overall success flag - for incremental, more tolerance for failures
        results["success"] = len(results["errors"]) == 0 and results["failed_operations"] < (results["total_operations"] * 0.3)
        
        logger.info(
            f"Incremental sync complete: {results['successful_operations']} operations succeeded, "
            f"{results['failed_operations']} failed"
        )
    except Exception as e:
        error_msg = f"Unexpected error in incremental market data sync: {str(e)}"
        logger.exception(error_msg)
        results["errors"].append(error_msg)
        results["success"] = False
    
    return results

def _get_incremental_days_back(interval: CandlestickInterval) -> int:
    """
    Determine how many days back to fetch for incremental sync based on interval.
    
    Args:
        interval: The candlestick interval
        
    Returns:
        Number of days to go back for incremental sync
    """
    if interval == CandlestickInterval.ONE_DAY:
        return 5  # Last 5 days
    elif interval == CandlestickInterval.ONE_HOUR:
        return 2  # Last 2 days
    else:
        return 1  # Last day


@task(name="update_pattern_database")
async def update_pattern_database() -> Dict[str, Any]:
    """
    Process synced market data and update the pattern database.
    
    This task detects patterns in the synced market data and updates
    the pattern database for assessment generation.
    
    Returns:
        Dictionary with update results
    """
    logger.info(f"Starting pattern database update (v{__version__})")
    
    results = {
        "success": False,
        "patterns_detected": 0,
        "patterns_added": 0,
        "symbols_processed": 0,
        "errors": [],
        "symbols_with_errors": [],
        "start_time": datetime.now().isoformat()
    }
    
    try:
        # Import required modules
        try:
            from backend.assessments.candlestick_patterns.candlestick_pattern_recognition import (
                recognize_patterns_in_series
            )
            from backend.assessments.candlestick_patterns.database_models import CandlestickPattern
        except ImportError as e:
            error_msg = f"Failed to import required modules: {str(e)}"
            logger.error(error_msg)
            results["errors"].append(error_msg)
            results["success"] = False
            return results
            
        # Get services
        market_service = get_market_data_service()
        if not market_service:
            error_msg = "Failed to access market data service"
            logger.error(error_msg)
            results["errors"].append(error_msg)
            return results
            
        # Get database connection
        try:
            db = await get_database_connection()
        except Exception as e:
            error_msg = f"Failed to connect to database: {str(e)}"
            logger.error(error_msg)
            results["errors"].append(error_msg)
            return results
        
        # Get configuration
        config = get_sync_config()
        symbols = config["symbols"]
        
        # Process each symbol
        async with db.session() as session:
            for symbol in symbols:
                try:
                    # Only process daily data for pattern detection
                    interval = CandlestickInterval.ONE_DAY
                    
                    # Get the candlestick data
                    series = await market_service.get_candlestick_data(
                        symbol=symbol,
                        interval=interval,
                        start_time=datetime.now() - timedelta(days=365),
                        end_time=None,
                        limit=None,
                        use_cache=True
                    )
                    
                    if not series or len(series) < 2:
                        logger.warning(f"Insufficient data for {symbol}, skipping pattern detection")
                        continue
                    
                    # Recognize patterns
                    pattern_results = recognize_patterns_in_series(series)
                    
                    # Store patterns in database
                    for pattern in pattern_results.patterns:
                        # Convert pattern to database model
                        db_pattern = CandlestickPattern(
                            pattern_type=pattern.pattern_type.value,
                            symbol=symbol,
                            timeframe=interval.value,
                            strength=pattern.strength.value,
                            timestamp=int(pattern.end_time.timestamp()),
                            candle_data={
                                "candles": [c.to_dict() for c in pattern.candles],
                                "start_time": pattern.start_time.isoformat(),
                                "end_time": pattern.end_time.isoformat(),
                                "confidence": pattern.confidence,
                                "trend_before": pattern.trend_before,
                                "expected_direction": pattern.expected_direction
                            }
                        )
                        
                        # Add to session
                        session.add(db_pattern)
                        results["patterns_added"] += 1
                    
                    results["patterns_detected"] += pattern_results.pattern_count
                    results["symbols_processed"] += 1
                    
                    logger.info(f"Processed {symbol}: detected {pattern_results.pattern_count} patterns")
                except Exception as e:
                    error_msg = f"Error processing {symbol}: {str(e)}"
                    logger.error(error_msg)
                    results["errors"].append(error_msg)
                    results["symbols_with_errors"].append(symbol)
            
            # Commit all changes
            try:
                await session.commit()
                logger.info(f"Successfully committed all pattern database updates")
            except Exception as e:
                error_msg = f"Error committing pattern database updates: {str(e)}"
                logger.error(error_msg)
                results["errors"].append(error_msg)
                raise  # Re-raise to be caught by outer try-except
    except Exception as e:
        error_msg = f"Error updating pattern database: {str(e)}"
        logger.exception(error_msg)  # Log with full traceback
        results["errors"].append(error_msg)
    
    # Calculate success based on errors and symbols processed
    results["success"] = (len(results["errors"]) == 0 or 
                         (len(results["symbols_with_errors"]) < len(config["symbols"]) * 0.2))
    
    # Update last update timestamp
    try:
        cache = CacheService.get_instance()
        await cache.set(CACHE_KEY_LAST_PATTERN_UPDATE, datetime.now().isoformat(), CACHE_TTL_DAY)
    except Exception as e:
        logger.error(f"Failed to update cache timestamp: {str(e)}")
    
    # Include end time and duration
    results["end_time"] = datetime.now().isoformat()
    
    logger.info(
        f"Pattern database update complete: {results['patterns_added']} patterns added, "
        f"{results['symbols_processed']}/{len(symbols)} symbols processed"
    )
    return results


@task(name="cleanup_old_data")
async def cleanup_old_data() -> Dict[str, Any]:
    """
    Clean up old market data and patterns.
    
    This task removes outdated market data and patterns to prevent
    database bloat and maintain performance.
    
    Returns:
        Dictionary with cleanup results
    """
    logger.info(f"Starting old data cleanup (v{__version__})")
    
    # Constants for retention periods
    SESSION_RETENTION_DAYS = 90
    PATTERN_RETENTION_DAYS = 730  # 2 years
    
    results = {
        "success": False,
        "patterns_removed": 0,
        "archived_sessions": 0,
        "errors": [],
        "start_time": datetime.now().isoformat()
    }
    
    try:
        # Import required modules
        try:
            from backend.assessments.candlestick_patterns.database_models import (
                CandlestickSession, CandlestickSessionArchive, CandlestickPattern
            )
        except ImportError as e:
            error_msg = f"Failed to import required modules: {str(e)}"
            logger.error(error_msg)
            results["errors"].append(error_msg)
            results["success"] = False
            return results
        
        # Get database connection
        try:
            db = await get_database_connection()
        except Exception as e:
            error_msg = f"Failed to connect to database: {str(e)}"
            logger.error(error_msg)
            results["errors"].append(error_msg)
            return results
        
        # Calculate cutoff timestamps
        sessions_cutoff = datetime.now() - timedelta(days=SESSION_RETENTION_DAYS)
        patterns_cutoff = datetime.now() - timedelta(days=PATTERN_RETENTION_DAYS)
        patterns_cutoff_timestamp = int(patterns_cutoff.timestamp())
        
        logger.info(f"Cleanup using cutoffs: sessions={sessions_cutoff.isoformat()}, patterns={patterns_cutoff.isoformat()}")
        
        async with db.session() as session:
            # Archive old sessions (older than 90 days)
            try:
                # Query for old sessions
                old_sessions_query = """
                    SELECT * FROM candlestick_sessions 
                    WHERE created_at < :cutoff_date
                """
                
                old_sessions_result = await session.execute(
                    old_sessions_query, 
                    {"cutoff_date": sessions_cutoff}
                )
                old_sessions = old_sessions_result.fetchall()
                
                logger.info(f"Found {len(old_sessions)} old sessions to archive")
                
                # Archive each session
                for session_row in old_sessions:
                    # Create archive record
                    archive = CandlestickSessionArchive(
                        session_id=session_row.session_id,
                        user_id=session_row.user_id,
                        assessment_type=session_row.assessment_type,
                        created_at=session_row.created_at,
                        completed_at=session_row.completed_at,
                        data=session_row.data,
                        updated_at=session_row.updated_at,
                        archived_at=datetime.now()
                    )
                    
                    session.add(archive)
                    results["archived_sessions"] += 1
                
                # Delete old sessions after archiving
                if results["archived_sessions"] > 0:
                    delete_sessions_query = """
                        DELETE FROM candlestick_sessions 
                        WHERE created_at < :cutoff_date
                    """
                    
                    await session.execute(
                        delete_sessions_query, 
                        {"cutoff_date": sessions_cutoff}
                    )
                    
                    logger.info(f"Archived and deleted {results['archived_sessions']} old sessions")
            except Exception as e:
                error_msg = f"Error archiving old sessions: {str(e)}"
                logger.error(error_msg)
                results["errors"].append(error_msg)
            
            # Remove old patterns (older than 2 years, except for educational examples)
            try:
                delete_patterns_query = """
                    DELETE FROM candlestick_patterns 
                    WHERE timestamp < :cutoff_timestamp
                    AND pattern_type != 'educational_example'
                """
                
                delete_result = await session.execute(
                    delete_patterns_query, 
                    {"cutoff_timestamp": patterns_cutoff_timestamp}
                )
                
                results["patterns_removed"] = delete_result.rowcount
                logger.info(f"Removed {results['patterns_removed']} old patterns")
            except Exception as e:
                error_msg = f"Error removing old patterns: {str(e)}"
                logger.error(error_msg)
                results["errors"].append(error_msg)
            
            # Commit all changes
            try:
                await session.commit()
                logger.info("Successfully committed all cleanup operations")
            except Exception as e:
                error_msg = f"Error committing cleanup operations: {str(e)}"
                logger.error(error_msg)
                results["errors"].append(error_msg)
                raise  # Re-raise to be caught by outer try-except
                
    except Exception as e:
        error_msg = f"Error during data cleanup: {str(e)}"
        logger.exception(error_msg)  # Log with full traceback
        results["errors"].append(error_msg)
    
    # Calculate success based on errors
    results["success"] = len(results["errors"]) == 0
    
    # Include end time
    results["end_time"] = datetime.now().isoformat()
    
    logger.info(
        f"Cleanup complete: {results['archived_sessions']} sessions archived, "
        f"{results['patterns_removed']} old patterns removed"
    )
    return results


# Cron schedule constants
CRON_WEEKLY_FULL_SYNC = "0 2 * * 0"  # Sunday at 2 AM
CRON_WEEKDAY_INCREMENTAL_SYNC = "0 */4 * * 1-5"  # Every 4 hours on weekdays
CRON_DAILY_PATTERN_UPDATE = "0 3 * * *"  # Every day at 3 AM
CRON_WEEKLY_CLEANUP = "0 1 * * 6"  # Saturday at 1 AM

def register_scheduled_tasks() -> None:
    """
    Register all market data sync tasks with the scheduler.
    
    This function registers the following tasks:
    1. Weekly full market data synchronization
    2. Incremental sync multiple times per day on weekdays
    3. Daily pattern database update
    4. Weekly data cleanup
    
    Returns:
        None
    """
    try:
        # Full synchronization - once per week (Sunday at 2 AM)
        schedule(
            task_name="sync_market_data_full",
            cron=CRON_WEEKLY_FULL_SYNC,
            description="Weekly full market data synchronization"
        )
        
        # Incremental synchronization - multiple times per day on weekdays
        schedule(
            task_name="sync_market_data_incremental",
            cron=CRON_WEEKDAY_INCREMENTAL_SYNC,
            description="Incremental market data synchronization (every 4 hours, weekdays)"
        )
        
        # Pattern database update - daily
        schedule(
            task_name="update_pattern_database",
            cron=CRON_DAILY_PATTERN_UPDATE,
            description="Daily pattern database update"
        )
        
        # Data cleanup - weekly
        schedule(
            task_name="cleanup_old_data",
            cron=CRON_WEEKLY_CLEANUP,
            description="Weekly data cleanup"
        )
        
        logger.info(
            f"Candlestick data sync tasks scheduled (v{__version__}): "
            f"full sync ({CRON_WEEKLY_FULL_SYNC}), "
            f"incremental sync ({CRON_WEEKDAY_INCREMENTAL_SYNC}), "
            f"pattern update ({CRON_DAILY_PATTERN_UPDATE}), "
            f"cleanup ({CRON_WEEKLY_CLEANUP})"
        )
    except Exception as e:
        logger.error(f"Failed to register scheduled tasks: {str(e)}")
        # We don't re-raise as this would prevent module import
        # Just log the error and continue


# Initialize tasks when module is imported
if __name__ != "__main__":  # Only register when imported as a module, not when run directly
    register_scheduled_tasks() 