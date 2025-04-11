"""
Alpha Vantage Data Provider

Implementation of the BaseDataProvider for Alpha Vantage data.
This serves as a fallback provider for redundancy.
"""

import asyncio
import datetime
import logging
import aiohttp
import pandas as pd
from typing import Dict, List, Optional, Union, Any

from backend.common.finance.candlestick import Candlestick, CandlestickSeries, CandlestickInterval
from backend.common.finance.market import Asset, AssetPair, AssetType
from backend.common.finance.data_provider import (
    BaseDataProvider, DataProviderConfig, DataProviderType, DataFetchError, RateLimitPolicy
)

logger = logging.getLogger(__name__)

# Alpha Vantage interval mapping
ALPHA_VANTAGE_INTERVALS = {
    CandlestickInterval.ONE_MINUTE: "1min",
    CandlestickInterval.FIVE_MINUTES: "5min",
    CandlestickInterval.FIFTEEN_MINUTES: "15min",
    CandlestickInterval.THIRTY_MINUTES: "30min",
    CandlestickInterval.ONE_HOUR: "60min",
    CandlestickInterval.ONE_DAY: "daily",
    CandlestickInterval.ONE_WEEK: "weekly",
    CandlestickInterval.ONE_MONTH: "monthly",
}

# Default configuration for Alpha Vantage
DEFAULT_CONFIG = DataProviderConfig(
    provider_type=DataProviderType.ALPHA_VANTAGE,
    base_url="https://www.alphavantage.co",
    rate_limit=5,  # Alpha Vantage free tier is limited to 5 requests per minute
    rate_limit_policy=RateLimitPolicy.STRICT,
    timeout=10,
    max_retries=3,
    cache_ttl=300,  # 5 minutes
    params={
        "api_key": "demo",  # Replace with your Alpha Vantage API key
    }
)


class AlphaVantageProvider(BaseDataProvider):
    """Alpha Vantage implementation of the BaseDataProvider."""
    
    def __init__(self, config: Optional[DataProviderConfig] = None):
        """
        Initialize the Alpha Vantage provider.
        
        Args:
            config: Configuration for the provider (optional)
        """
        if config is None:
            config = DEFAULT_CONFIG
            
        super().__init__(config)
        self._session = None
        self._initialize_lock = asyncio.Lock()
        self._initialized = False
        
    async def _ensure_initialized(self) -> None:
        """Ensure the provider is initialized with a session."""
        if self._initialized:
            return
            
        async with self._initialize_lock:
            if self._initialized:  # Check again in case another task initialized while waiting
                return
                
            if self._session is None:
                self._session = aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout)
                )
            
            self._initialized = True
    
    async def _make_request(self, params: Dict[str, Any]) -> Any:
        """
        Make a request to the Alpha Vantage API.
        
        Args:
            params: Query parameters
            
        Returns:
            Response data
        """
        await self._ensure_initialized()
        await self._rate_limit()
        
        url = f"{self.config.base_url}/query"
        
        # Add API key
        params["apikey"] = self.config.params["api_key"]
        
        for attempt in range(self.config.max_retries + 1):
            try:
                async with self._session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Check for API errors in the response
                        if "Error Message" in data:
                            error_msg = data["Error Message"]
                            self.logger.error(f"API error: {error_msg}")
                            
                            if attempt == self.config.max_retries:
                                raise DataFetchError(error_msg, "AlphaVantage")
                            
                            # Some errors are temporary, retry
                            wait_time = 2 ** attempt
                            self.logger.info(f"Retrying in {wait_time}s, attempt {attempt + 1}/{self.config.max_retries}")
                            await asyncio.sleep(wait_time)
                            continue
                        
                        # Check for rate limiting
                        if "Note" in data and "API call frequency" in data["Note"]:
                            wait_time = 60  # Wait for a minute
                            self.logger.warning(f"Rate limited, waiting {wait_time}s")
                            await asyncio.sleep(wait_time)
                            continue
                            
                        return data
                    elif response.status == 429:  # Too Many Requests
                        wait_time = 60  # Wait for a minute
                        self.logger.warning(f"Rate limited, waiting {wait_time}s")
                        await asyncio.sleep(wait_time)
                    else:
                        error_text = await response.text()
                        self.logger.error(f"API error: {response.status}, {error_text}")
                        
                        if attempt == self.config.max_retries:
                            raise DataFetchError(
                                f"Failed after {attempt + 1} attempts: {error_text}", 
                                "AlphaVantage", 
                                response.status
                            )
                        
                        # Exponential backoff
                        wait_time = 2 ** attempt
                        self.logger.info(f"Retrying in {wait_time}s, attempt {attempt + 1}/{self.config.max_retries}")
                        await asyncio.sleep(wait_time)
            except asyncio.TimeoutError:
                if attempt == self.config.max_retries:
                    raise DataFetchError("Request timed out", "AlphaVantage")
                    
                wait_time = 2 ** attempt
                self.logger.info(f"Timeout, retrying in {wait_time}s, attempt {attempt + 1}/{self.config.max_retries}")
                await asyncio.sleep(wait_time)
            except Exception as e:
                if attempt == self.config.max_retries:
                    raise DataFetchError(f"Request failed: {str(e)}", "AlphaVantage")
                    
                wait_time = 2 ** attempt
                self.logger.info(f"Error: {e}, retrying in {wait_time}s, attempt {attempt + 1}/{self.config.max_retries}")
                await asyncio.sleep(wait_time)
    
    async def get_assets(self) -> List[Asset]:
        """
        Get a list of available assets.
        
        Alpha Vantage doesn't provide a direct endpoint for this.
        We return a simplified list of common assets.
        
        Returns:
            List of assets
        """
        # Simple implementation for demonstration
        common_assets = [
            Asset(
                symbol="AAPL",
                name="Apple Inc.",
                asset_type=AssetType.STOCK,
                exchange="NASDAQ",
                currency="USD",
                sector="Technology",
                country="USA",
                description="Apple Inc. designs, manufactures, and markets smartphones, personal computers, tablets, wearables, and accessories worldwide."
            ),
            Asset(
                symbol="MSFT",
                name="Microsoft Corporation",
                asset_type=AssetType.STOCK,
                exchange="NASDAQ",
                currency="USD",
                sector="Technology",
                country="USA",
                description="Microsoft Corporation develops, licenses, and supports software, services, devices, and solutions worldwide."
            ),
            Asset(
                symbol="AMZN",
                name="Amazon.com, Inc.",
                asset_type=AssetType.STOCK,
                exchange="NASDAQ",
                currency="USD",
                sector="Consumer Cyclical",
                country="USA",
                description="Amazon.com, Inc. engages in the retail sale of consumer products and subscriptions in North America and internationally."
            ),
            Asset(
                symbol="GOOGL",
                name="Alphabet Inc.",
                asset_type=AssetType.STOCK,
                exchange="NASDAQ",
                currency="USD",
                sector="Communication Services",
                country="USA",
                description="Alphabet Inc. provides various products and platforms in the United States, Europe, the Middle East, Africa, the Asia-Pacific, Canada, and Latin America."
            ),
            Asset(
                symbol="BRK.A",
                name="Berkshire Hathaway Inc.",
                asset_type=AssetType.STOCK,
                exchange="NYSE",
                currency="USD",
                sector="Financial Services",
                country="USA",
                description="Berkshire Hathaway Inc. engages in the insurance, freight rail transportation, and utility businesses worldwide."
            )
        ]
        
        # Cache this result
        cache_key = self._get_cache_key("assets")
        await self._set_in_cache(cache_key, common_assets, 86400)  # 24 hours TTL
        
        return common_assets
    
    async def get_asset(self, symbol: str) -> Optional[Asset]:
        """
        Get details for a specific asset.
        
        Args:
            symbol: Asset symbol
            
        Returns:
            Asset details if found
        """
        cache_key = self._get_cache_key("asset", symbol=symbol)
        cached = await self._get_from_cache(cache_key)
        if cached:
            return cached
            
        # Fetch asset details
        try:
            params = {
                "function": "OVERVIEW",
                "symbol": symbol,
            }
            
            data = await self._make_request(params)
            
            # Check if we got valid data
            if not data or "Symbol" not in data:
                return None
                
            # Determine asset type
            asset_type = AssetType.STOCK
            if "ETF" in data.get("Name", ""):
                asset_type = AssetType.ETF
            
            asset = Asset(
                symbol=data.get("Symbol"),
                name=data.get("Name"),
                asset_type=asset_type,
                exchange=data.get("Exchange"),
                currency="USD",  # Alpha Vantage usually deals with USD
                sector=data.get("Sector"),
                country=data.get("Country"),
                description=data.get("Description"),
                metadata={
                    "industry": data.get("Industry"),
                    "market_cap": data.get("MarketCapitalization"),
                    "pe_ratio": data.get("PERatio"),
                    "dividend_yield": data.get("DividendYield"),
                    "book_value": data.get("BookValue"),
                    "eps": data.get("EPS"),
                    "52_week_high": data.get("52WeekHigh"),
                    "52_week_low": data.get("52WeekLow")
                }
            )
            
            # Cache the result
            await self._set_in_cache(cache_key, asset, 3600)  # 1 hour TTL
            
            return asset
        except Exception as e:
            self.logger.error(f"Error fetching asset {symbol}: {e}")
            return None
    
    async def get_asset_pairs(self) -> List[AssetPair]:
        """
        Get available asset pairs.
        
        Alpha Vantage doesn't have a direct concept of asset pairs.
        
        Returns:
            Empty list
        """
        return []
    
    async def get_asset_pair(self, symbol: str) -> Optional[AssetPair]:
        """
        Get details for a specific asset pair.
        
        Alpha Vantage doesn't have a direct concept of asset pairs.
        
        Args:
            symbol: Pair symbol
            
        Returns:
            None
        """
        return None
    
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
            interval: Candlestick interval
            start_time: Start time for the data
            end_time: End time for the data
            limit: Maximum number of candles to retrieve
            use_cache: Whether to use cached data
            
        Returns:
            CandlestickSeries with the requested data
        """
        # Convert interval to Alpha Vantage format
        if isinstance(interval, CandlestickInterval):
            alpha_interval = ALPHA_VANTAGE_INTERVALS.get(interval)
            if not alpha_interval:
                raise ValueError(f"Unsupported interval: {interval}")
        else:
            # Try to convert string to enum first
            try:
                enum_interval = CandlestickInterval(interval)
                alpha_interval = ALPHA_VANTAGE_INTERVALS.get(enum_interval)
                if not alpha_interval:
                    alpha_interval = interval  # Use as-is if not found
            except ValueError:
                alpha_interval = interval  # Use as-is if not a valid enum value
        
        # Check cache
        if use_cache:
            cache_key = self._get_cache_key(
                "candlestick",
                symbol=symbol,
                interval=alpha_interval
            )
            cached = await self._get_from_cache(cache_key)
            if cached:
                # Apply time filtering and limit after retrieving from cache
                series = cached
                filtered_candles = [
                    c for c in series.candlesticks
                    if (start_time is None or c.timestamp >= start_time) and
                       (end_time is None or c.timestamp <= end_time)
                ]
                
                if limit is not None and limit > 0 and len(filtered_candles) > limit:
                    filtered_candles = filtered_candles[-limit:]
                
                # Create a new series with the filtered candles
                filtered_series = CandlestickSeries(
                    symbol=series.symbol,
                    interval=series.interval,
                    candlesticks=filtered_candles,
                    source=series.source,
                    metadata=series.metadata.copy() if series.metadata else {}
                )
                
                return filtered_series
        
        # Determine the function to use based on the interval
        if alpha_interval in ["1min", "5min", "15min", "30min", "60min"]:
            function = "TIME_SERIES_INTRADAY"
            data_key = f"Time Series ({alpha_interval})"
            params = {
                "function": function,
                "symbol": symbol,
                "interval": alpha_interval,
                "outputsize": "full"
            }
        elif alpha_interval == "daily":
            function = "TIME_SERIES_DAILY"
            data_key = "Time Series (Daily)"
            params = {
                "function": function,
                "symbol": symbol,
                "outputsize": "full"
            }
        elif alpha_interval == "weekly":
            function = "TIME_SERIES_WEEKLY"
            data_key = "Weekly Time Series"
            params = {
                "function": function,
                "symbol": symbol
            }
        elif alpha_interval == "monthly":
            function = "TIME_SERIES_MONTHLY"
            data_key = "Monthly Time Series"
            params = {
                "function": function,
                "symbol": symbol
            }
        else:
            raise ValueError(f"Unsupported interval: {alpha_interval}")
        
        try:
            data = await self._make_request(params)
            
            # Check for errors
            if not data or data_key not in data:
                error_message = data.get("Error Message", "Unknown error")
                raise DataFetchError(f"Alpha Vantage API error: {error_message}", "AlphaVantage")
            
            # Extract the time series data
            time_series = data[data_key]
            
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(time_series, orient="index")
            
            # Rename columns to standard format
            df.rename(columns={
                "1. open": "open",
                "2. high": "high",
                "3. low": "low",
                "4. close": "close",
                "5. volume": "volume"
            }, inplace=True)
            
            # Convert types
            for col in ["open", "high", "low", "close"]:
                df[col] = pd.to_numeric(df[col])
            df["volume"] = pd.to_numeric(df["volume"])
            
            # Convert index to datetime
            df.index = pd.to_datetime(df.index)
            df.sort_index(inplace=True)
            
            # Filter by time range
            if start_time:
                df = df[df.index >= pd.Timestamp(start_time)]
            if end_time:
                df = df[df.index <= pd.Timestamp(end_time)]
            
            # Apply limit
            if limit is not None and limit > 0:
                df = df.tail(limit)
            
            # Create candlesticks
            candlesticks = []
            for timestamp, row in df.iterrows():
                candlesticks.append(Candlestick(
                    timestamp=timestamp.to_pydatetime(),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"])
                ))
            
            # Convert string interval back to enum if possible
            try:
                # Reverse lookup from Alpha Vantage to CandlestickInterval
                reverse_intervals = {v: k for k, v in ALPHA_VANTAGE_INTERVALS.items()}
                candlestick_interval = reverse_intervals.get(alpha_interval)
                if not candlestick_interval:
                    # Try direct conversion
                    candlestick_interval = CandlestickInterval(alpha_interval)
            except ValueError:
                # Fall back to a default if can't convert
                candlestick_interval = CandlestickInterval.ONE_DAY
            
            # Create the series
            series = CandlestickSeries(
                symbol=symbol,
                interval=candlestick_interval,
                candlesticks=candlesticks,
                source="AlphaVantage",
                metadata={
                    "query_interval": alpha_interval,
                    "function": function,
                    "last_refreshed": data.get("Meta Data", {}).get("3. Last Refreshed")
                }
            )
            
            # Cache the result
            if use_cache:
                ttl = 300  # 5 minutes default
                if alpha_interval in ["daily", "weekly", "monthly"]:
                    ttl = 3600  # 1 hour for daily and longer
                await self._set_in_cache(cache_key, series, ttl)
            
            return series
        except Exception as e:
            self.logger.error(f"Error fetching candlestick data: {e}")
            raise
    
    async def get_latest_price(self, symbol: str, use_cache: bool = True) -> float:
        """
        Get the latest price for a symbol.
        
        Args:
            symbol: Symbol to get the price for
            use_cache: Whether to use cached data
            
        Returns:
            Latest price
        """
        # Check cache
        if use_cache:
            cache_key = self._get_cache_key("latest_price", symbol=symbol)
            cached = await self._get_from_cache(cache_key)
            if cached is not None:
                return cached
        
        try:
            params = {
                "function": "GLOBAL_QUOTE",
                "symbol": symbol
            }
            
            data = await self._make_request(params)
            
            # Check for errors
            if not data or "Global Quote" not in data:
                error_message = data.get("Error Message", "Unknown error")
                raise DataFetchError(f"Alpha Vantage API error: {error_message}", "AlphaVantage")
            
            quote = data["Global Quote"]
            price_str = quote.get("05. price")
            
            if not price_str:
                raise DataFetchError(f"No price data for {symbol}", "AlphaVantage")
            
            price = float(price_str)
            
            # Cache the result with a short TTL
            if use_cache:
                await self._set_in_cache(cache_key, price, 60)  # 1 minute TTL
            
            return price
        except Exception as e:
            self.logger.error(f"Error fetching latest price for {symbol}: {e}")
            raise
    
    async def close(self) -> None:
        """Close the provider and release resources."""
        if self._session:
            await self._session.close()
            self._session = None
            self._initialized = False 