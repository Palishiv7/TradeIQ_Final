"""
Market Data Provider

This module defines base classes and interfaces for market data providers,
providing a standardized way to fetch and manage market data from various sources.
"""

import asyncio
import datetime
import enum
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Union, Any, TypeVar, Generic

import pandas as pd
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from backend.common.finance.candlestick import Candlestick, CandlestickSeries, CandlestickInterval
from backend.common.finance.market import Asset, AssetPair, MarketData
from backend.common.serialization import SerializableMixin
from backend.common.cache.service import CacheService
from backend.common.db.connection import get_database_connection

# Type variable for return types
T = TypeVar('T')

# Module logger
logger = logging.getLogger(__name__)


class DataProviderType(enum.Enum):
    """Types of data providers."""
    
    ALPHA_VANTAGE = "alpha_vantage"
    YAHOO_FINANCE = "yahoo_finance"
    IEX_CLOUD = "iex_cloud"
    FINNHUB = "finnhub"
    POLYGON = "polygon"
    BINANCE = "binance"
    KRAKEN = "kraken"
    COINBASE = "coinbase"
    MOCK = "mock"
    CUSTOM = "custom"


class RateLimitPolicy(enum.Enum):
    """Rate limit policies for API calls."""
    
    STRICT = "strict"  # Strictly enforce rate limits, blocking if needed
    ADAPTIVE = "adaptive"  # Adaptively throttle based on response headers
    BEST_EFFORT = "best_effort"  # Try to respect limits but not guaranteed
    NONE = "none"  # No rate limiting


@dataclass
class DataProviderConfig(SerializableMixin):
    """
    Configuration for a data provider.
    
    Attributes:
        provider_type: The type of data provider
        api_key: API key for the provider (if required)
        api_secret: API secret for the provider (if required)
        base_url: Base URL for API requests
        rate_limit: Maximum requests per minute
        rate_limit_policy: Policy for rate limiting
        timeout: Request timeout in seconds
        max_retries: Maximum number of retries for failed requests
        cache_ttl: Time-to-live for cached data (in seconds)
        async_requests: Whether to use async requests
        params: Additional provider-specific parameters
    """
    provider_type: DataProviderType
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    base_url: Optional[str] = None
    rate_limit: int = 60  # requests per minute
    rate_limit_policy: RateLimitPolicy = RateLimitPolicy.STRICT
    timeout: int = 30  # seconds
    max_retries: int = 3
    cache_ttl: int = 300  # seconds
    async_requests: bool = True
    params: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the configuration to a dictionary."""
        return {
            "provider_type": self.provider_type.value,
            "api_key": self.api_key,
            "api_secret": self.api_secret,
            "base_url": self.base_url,
            "rate_limit": self.rate_limit,
            "rate_limit_policy": self.rate_limit_policy.value,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "cache_ttl": self.cache_ttl,
            "async_requests": self.async_requests,
            "params": self.params
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DataProviderConfig':
        """Create a configuration from a dictionary."""
        provider_type = DataProviderType(data["provider_type"])
        
        rate_limit_policy = RateLimitPolicy.STRICT
        if "rate_limit_policy" in data:
            rate_limit_policy = RateLimitPolicy(data["rate_limit_policy"])
        
        return cls(
            provider_type=provider_type,
            api_key=data.get("api_key"),
            api_secret=data.get("api_secret"),
            base_url=data.get("base_url"),
            rate_limit=data.get("rate_limit", 60),
            rate_limit_policy=rate_limit_policy,
            timeout=data.get("timeout", 30),
            max_retries=data.get("max_retries", 3),
            cache_ttl=data.get("cache_ttl", 300),
            async_requests=data.get("async_requests", True),
            params=data.get("params", {})
        )


class DataFetchError(Exception):
    """Exception raised when data fetching fails."""
    
    def __init__(self, message: str, provider: str, status_code: Optional[int] = None):
        self.provider = provider
        self.status_code = status_code
        super().__init__(f"[{provider}] {message}")


class BaseDataProvider(ABC):
    """
    Base class for market data providers.
    
    This abstract class defines the common interface for all data providers.
    Concrete implementations should inherit from this class and implement
    the abstract methods.
    """
    
    def __init__(self, config: DataProviderConfig):
        """
        Initialize the data provider with the given configuration.
        
        Args:
            config: Configuration for the data provider
        """
        self.config = config
        self.cache_service = CacheService.get_instance()
        self._last_request_time = 0
        self._request_count = 0
        self._request_lock = asyncio.Lock()
        
        # Setup logging
        self.logger = logger.getChild(self.__class__.__name__)
        self.logger.info(f"Initialized {self.__class__.__name__} with {config.provider_type.value}")
    
    async def _rate_limit(self) -> None:
        """
        Apply rate limiting based on the configured policy.
        
        This method ensures that the provider doesn't exceed the configured
        rate limit by introducing delays between requests if necessary.
        """
        if self.config.rate_limit_policy == RateLimitPolicy.NONE:
            return
        
        async with self._request_lock:
            # Calculate time since last request
            now = time.time()
            elapsed = now - self._last_request_time
            
            # Reset counter if more than a minute has passed
            if elapsed >= 60:
                self._request_count = 0
                self._last_request_time = now
                return
            
            # Check if we need to throttle
            if self._request_count >= self.config.rate_limit:
                # Sleep until a minute has passed since the first request
                wait_time = 60 - elapsed
                if wait_time > 0:
                    self.logger.debug(f"Rate limit reached, waiting {wait_time:.2f}s")
                    await asyncio.sleep(wait_time)
                    self._request_count = 0
                    self._last_request_time = time.time()
            else:
                self._request_count += 1
                if self._request_count == 1:
                    self._last_request_time = now
    
    def _get_cache_key(self, prefix: str, **params) -> str:
        """
        Generate a cache key for the given parameters.
        
        Args:
            prefix: Prefix for the cache key
            **params: Parameters to include in the key
            
        Returns:
            A unique cache key
        """
        provider = self.config.provider_type.value
        parts = [provider, prefix]
        
        # Sort parameters for consistent keys
        for key in sorted(params.keys()):
            value = params[key]
            # Handle special cases like timestamps
            if isinstance(value, (datetime.datetime, datetime.date)):
                value = value.isoformat()
            parts.append(f"{key}={value}")
        
        return ":".join(parts)
    
    async def _get_from_cache(self, key: str) -> Optional[Any]:
        """
        Get data from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached data if available, otherwise None
        """
        try:
            return await self.cache_service.get(key)
        except Exception as e:
            self.logger.warning(f"Cache error: {e}")
            return None
    
    async def _set_in_cache(self, key: str, data: Any, ttl: Optional[int] = None) -> None:
        """
        Store data in the cache.
        
        Args:
            key: Cache key
            data: Data to cache
            ttl: Time-to-live in seconds (if None, use config default)
        """
        if ttl is None:
            ttl = self.config.cache_ttl
            
        try:
            await self.cache_service.set(key, data, ttl)
        except Exception as e:
            self.logger.warning(f"Cache error: {e}")
    
    @abstractmethod
    async def get_assets(self) -> List[Asset]:
        """
        Get all available assets.
        
        Returns:
            List of assets
        """
        pass
    
    @abstractmethod
    async def get_asset(self, symbol: str) -> Optional[Asset]:
        """
        Get an asset by symbol.
        
        Args:
            symbol: Asset symbol
            
        Returns:
            Asset if found, otherwise None
        """
        pass
    
    @abstractmethod
    async def get_asset_pairs(self) -> List[AssetPair]:
        """
        Get all available asset pairs.
        
        Returns:
            List of asset pairs
        """
        pass
    
    @abstractmethod
    async def get_asset_pair(self, symbol: str) -> Optional[AssetPair]:
        """
        Get an asset pair by symbol.
        
        Args:
            symbol: Pair symbol
            
        Returns:
            Asset pair if found, otherwise None
        """
        pass
    
    @abstractmethod
    async def get_candlestick_data(
        self,
        symbol: str,
        interval: Union[CandlestickInterval, str],
        start_time: Optional[datetime.datetime] = None,
        end_time: Optional[datetime.datetime] = None,
        limit: Optional[int] = None,
        use_cache: bool = True
    ) -> CandlestickSeries:
        """
        Get candlestick data for a symbol.
        
        Args:
            symbol: Symbol to get data for
            interval: Candlestick interval (e.g., 1m, 1h, 1d)
            start_time: Start time for the data
            end_time: End time for the data
            limit: Maximum number of candles to retrieve
            use_cache: Whether to use cached data
            
        Returns:
            CandlestickSeries with the requested data
        """
        pass
    
    @abstractmethod
    async def get_latest_price(self, symbol: str, use_cache: bool = True) -> float:
        """
        Get the latest price for a symbol.
        
        Args:
            symbol: Symbol to get the price for
            use_cache: Whether to use cached data
            
        Returns:
            Latest price
        """
        pass


class DataProviderRegistry:
    """
    Registry for data providers.
    
    This class maintains a registry of available data providers and 
    allows retrieving providers by type.
    """
    
    _instance = None
    _providers: Dict[DataProviderType, BaseDataProvider] = {}
    _primary_provider: Optional[BaseDataProvider] = None
    _backup_providers: List[BaseDataProvider] = []
    
    @classmethod
    def get_instance(cls) -> 'DataProviderRegistry':
        """
        Get the singleton instance of the registry.
        
        Returns:
            DataProviderRegistry instance
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def register_provider(
        self, 
        provider: BaseDataProvider, 
        is_primary: bool = False,
        is_backup: bool = False
    ) -> None:
        """
        Register a data provider.
        
        Args:
            provider: The provider to register
            is_primary: Whether this is the primary provider
            is_backup: Whether this is a backup provider
        """
        provider_type = provider.config.provider_type
        self._providers[provider_type] = provider
        
        if is_primary:
            self._primary_provider = provider
            
        if is_backup:
            self._backup_providers.append(provider)
        
        logger.info(f"Registered {provider_type.value} provider")
    
    def get_provider(self, provider_type: Union[DataProviderType, str]) -> Optional[BaseDataProvider]:
        """
        Get a provider by type.
        
        Args:
            provider_type: Type of the provider
            
        Returns:
            The provider if registered, otherwise None
        """
        if isinstance(provider_type, str):
            provider_type = DataProviderType(provider_type)
            
        return self._providers.get(provider_type)
    
    def get_primary_provider(self) -> Optional[BaseDataProvider]:
        """
        Get the primary data provider.
        
        Returns:
            The primary provider if set, otherwise None
        """
        return self._primary_provider
    
    def get_backup_providers(self) -> List[BaseDataProvider]:
        """
        Get all backup data providers.
        
        Returns:
            List of backup providers
        """
        return self._backup_providers.copy()
    
    def get_all_providers(self) -> List[BaseDataProvider]:
        """
        Get all registered data providers.
        
        Returns:
            List of all providers
        """
        return list(self._providers.values())


class RedundantDataProvider(BaseDataProvider):
    """
    A data provider that uses multiple providers with failover.
    
    This provider tries the primary provider first, and if it fails,
    it falls back to backup providers in the order they were registered.
    """
    
    def __init__(self, registry: DataProviderRegistry):
        """
        Initialize with a provider registry.
        
        Args:
            registry: The provider registry to use
        """
        # Create a config for this meta-provider
        config = DataProviderConfig(
            provider_type=DataProviderType.CUSTOM,
            params={"name": "RedundantDataProvider"}
        )
        super().__init__(config)
        
        self.registry = registry
        self.primary = registry.get_primary_provider()
        self.backups = registry.get_backup_providers()
        
        if not self.primary:
            raise ValueError("No primary provider registered")
        
        self.logger.info(f"Using {self.primary.__class__.__name__} as primary provider")
        for backup in self.backups:
            self.logger.info(f"Using {backup.__class__.__name__} as backup provider")
    
    async def _execute_with_failover(self, method_name: str, *args, **kwargs) -> Any:
        """
        Execute a method with failover to backup providers.
        
        Args:
            method_name: Name of the method to execute
            *args: Positional arguments for the method
            **kwargs: Keyword arguments for the method
            
        Returns:
            Result of the method
            
        Raises:
            DataFetchError: If all providers fail
        """
        # Try primary provider first
        try:
            method = getattr(self.primary, method_name)
            return await method(*args, **kwargs)
        except Exception as e:
            self.logger.warning(f"Primary provider failed: {e}")
            
            # Try backup providers in order
            last_error = e
            for backup in self.backups:
                try:
                    method = getattr(backup, method_name)
                    result = await method(*args, **kwargs)
                    self.logger.info(f"Successfully used backup provider {backup.__class__.__name__}")
                    return result
                except Exception as backup_error:
                    self.logger.warning(f"Backup provider {backup.__class__.__name__} failed: {backup_error}")
                    last_error = backup_error
            
            # All providers failed
            raise DataFetchError(
                f"All providers failed for {method_name}",
                "RedundantDataProvider"
            ) from last_error
    
    async def get_assets(self) -> List[Asset]:
        """Get all available assets with failover."""
        return await self._execute_with_failover("get_assets")
    
    async def get_asset(self, symbol: str) -> Optional[Asset]:
        """Get an asset by symbol with failover."""
        return await self._execute_with_failover("get_asset", symbol)
    
    async def get_asset_pairs(self) -> List[AssetPair]:
        """Get all available asset pairs with failover."""
        return await self._execute_with_failover("get_asset_pairs")
    
    async def get_asset_pair(self, symbol: str) -> Optional[AssetPair]:
        """Get an asset pair by symbol with failover."""
        return await self._execute_with_failover("get_asset_pair", symbol)
    
    async def get_candlestick_data(
        self,
        symbol: str,
        interval: Union[CandlestickInterval, str],
        start_time: Optional[datetime.datetime] = None,
        end_time: Optional[datetime.datetime] = None,
        limit: Optional[int] = None,
        use_cache: bool = True
    ) -> CandlestickSeries:
        """Get candlestick data with failover."""
        return await self._execute_with_failover(
            "get_candlestick_data",
            symbol, interval, start_time, end_time, limit, use_cache
        )
    
    async def get_latest_price(self, symbol: str, use_cache: bool = True) -> float:
        """Get the latest price with failover."""
        return await self._execute_with_failover("get_latest_price", symbol, use_cache)


class MarketDataService:
    """
    Service for accessing market data from various providers.
    
    This service acts as a facade for data providers, offering a simplified
    interface and additional functionality like caching and persistence.
    """
    
    _instance = None
    
    @classmethod
    def get_instance(cls) -> 'MarketDataService':
        """
        Get the singleton instance of the service.
        
        Returns:
            MarketDataService instance
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        """Initialize the market data service."""
        self.registry = DataProviderRegistry.get_instance()
        self.provider = self.registry.get_primary_provider()
        self.redundant_provider = None
        self.cache_service = CacheService.get_instance()
        self.logger = logger.getChild(self.__class__.__name__)
        
        # Set up the redundant provider if there are backups
        if self.registry.get_backup_providers():
            self.redundant_provider = RedundantDataProvider(self.registry)
            self.logger.info("Using redundant data provider with failover")
        
        self.logger.info("MarketDataService initialized")
    
    async def get_candlestick_data(
        self,
        symbol: str,
        interval: Union[CandlestickInterval, str],
        start_time: Optional[datetime.datetime] = None,
        end_time: Optional[datetime.datetime] = None,
        limit: Optional[int] = None,
        use_cache: bool = True,
        use_redundancy: bool = True
    ) -> CandlestickSeries:
        """
        Get candlestick data for a symbol.
        
        Args:
            symbol: Symbol to get data for
            interval: Candlestick interval
            start_time: Start time for the data
            end_time: End time for the data
            limit: Maximum number of candles to retrieve
            use_cache: Whether to use cached data
            use_redundancy: Whether to use redundant providers
            
        Returns:
            CandlestickSeries with the requested data
        """
        try:
            provider = self.redundant_provider if use_redundancy and self.redundant_provider else self.provider
            return await provider.get_candlestick_data(
                symbol, interval, start_time, end_time, limit, use_cache
            )
        except Exception as e:
            self.logger.error(f"Failed to get candlestick data: {e}")
            raise
    
    async def save_candlestick_data(self, series: CandlestickSeries, db_session: Optional[AsyncSession] = None) -> None:
        """
        Save candlestick data to the database.
        
        Args:
            series: The candlestick series to save
            db_session: Database session to use (optional)
        """
        try:
            # Create a database session if not provided
            if db_session is None:
                db = await get_database_connection()
                async with db.session() as session:
                    await self._save_candlestick_data(series, session)
            else:
                await self._save_candlestick_data(series, db_session)
        except Exception as e:
            self.logger.error(f"Failed to save candlestick data: {e}")
            raise
    
    async def _save_candlestick_data(self, series: CandlestickSeries, db_session: AsyncSession) -> None:
        """
        Internal method to save candlestick data.
        
        Args:
            series: The candlestick series to save
            db_session: Database session to use
        """
        from backend.assessments.candlestick_patterns.database_models import CandlestickPattern
        
        # Convert to DataFrame for bulk operations
        df = series.to_dataframe()
        if df.empty:
            return
        
        # Create CandlestickPattern records from the series
        # This is a simplified example - the actual implementation should
        # match your database schema and requirements
        for _, row in df.iterrows():
            pattern = CandlestickPattern(
                pattern_type="raw_data",
                symbol=series.symbol,
                timeframe=series.interval.value,
                timestamp=int(row.name.timestamp()),
                candle_data={
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row["volume"]),
                }
            )
            db_session.add(pattern)
        
        await db_session.commit()
    
    async def get_latest_prices(self, symbols: List[str], use_redundancy: bool = True) -> Dict[str, float]:
        """
        Get latest prices for multiple symbols.
        
        Args:
            symbols: List of symbols to get prices for
            use_redundancy: Whether to use redundant providers
            
        Returns:
            Dictionary mapping symbols to their latest prices
        """
        provider = self.redundant_provider if use_redundancy and self.redundant_provider else self.provider
        
        results = {}
        for symbol in symbols:
            try:
                price = await provider.get_latest_price(symbol)
                results[symbol] = price
            except Exception as e:
                self.logger.warning(f"Failed to get price for {symbol}: {e}")
                results[symbol] = None
        
        return results
    
    async def sync_market_data(
        self,
        symbols: List[str],
        intervals: List[Union[CandlestickInterval, str]],
        days_back: int = 30
    ) -> Dict[str, Dict[str, bool]]:
        """
        Synchronize market data for multiple symbols and intervals.
        
        This method fetches and stores data for the specified symbols
        and intervals, for the given number of days back from now.
        
        Args:
            symbols: List of symbols to sync
            intervals: List of intervals to sync
            days_back: Number of days of historical data to sync
            
        Returns:
            Nested dictionary mapping symbols and intervals to success status
        """
        end_time = datetime.datetime.now()
        start_time = end_time - datetime.timedelta(days=days_back)
        
        results = {}
        for symbol in symbols:
            results[symbol] = {}
            for interval in intervals:
                try:
                    # Fetch data
                    series = await self.get_candlestick_data(
                        symbol, interval, start_time, end_time, use_cache=False
                    )
                    
                    # Save to database
                    await self.save_candlestick_data(series)
                    
                    # Update cache
                    cache_key = f"market_data:{symbol}:{interval}:synced"
                    await self.cache_service.set(cache_key, True, 86400)  # 24 hours TTL
                    
                    results[symbol][str(interval)] = True
                    self.logger.info(f"Synced {symbol} {interval} data")
                except Exception as e:
                    results[symbol][str(interval)] = False
                    self.logger.error(f"Failed to sync {symbol} {interval} data: {e}")
        
        return results 