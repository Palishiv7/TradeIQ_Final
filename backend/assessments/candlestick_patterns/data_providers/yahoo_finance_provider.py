"""
Yahoo Finance Data Provider

Implementation of the BaseDataProvider for Yahoo Finance data.
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

# Yahoo Finance interval mapping
YAHOO_INTERVALS = {
    CandlestickInterval.ONE_MINUTE: "1m",
    CandlestickInterval.FIVE_MINUTES: "5m",
    CandlestickInterval.FIFTEEN_MINUTES: "15m",
    CandlestickInterval.THIRTY_MINUTES: "30m",
    CandlestickInterval.ONE_HOUR: "1h",
    CandlestickInterval.ONE_DAY: "1d",
    CandlestickInterval.ONE_WEEK: "1wk",
    CandlestickInterval.ONE_MONTH: "1mo",
}

# Default configuration for Yahoo Finance
DEFAULT_CONFIG = DataProviderConfig(
    provider_type=DataProviderType.YAHOO_FINANCE,
    base_url="https://query1.finance.yahoo.com",
    rate_limit=2000,  # Yahoo Finance allows ~2000 requests per hour
    rate_limit_policy=RateLimitPolicy.ADAPTIVE,
    timeout=10,
    max_retries=3,
    cache_ttl=300,  # 5 minutes
    params={
        "crumb": None,  # Will be fetched during initialization
        "session_cookie": None,  # Will be fetched during initialization
    }
)


class YahooFinanceProvider(BaseDataProvider):
    """Yahoo Finance implementation of the BaseDataProvider."""
    
    def __init__(self, config: Optional[DataProviderConfig] = None):
        """
        Initialize the Yahoo Finance provider.
        
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
        """Ensure the provider is initialized with a session and crumb."""
        if self._initialized:
            return
            
        async with self._initialize_lock:
            if self._initialized:  # Check again in case another task initialized while waiting
                return
                
            if self._session is None:
                self._session = aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout)
                )
            
            if not self.config.params.get("crumb"):
                try:
                    await self._fetch_crumb()
                except Exception as e:
                    self.logger.warning(f"Failed to fetch crumb: {e}")
                    # We can still proceed without the crumb for some endpoints
            
            self._initialized = True
    
    async def _fetch_crumb(self) -> None:
        """Fetch the crumb and session cookie required for some Yahoo Finance API calls."""
        url = f"{self.config.base_url}/v1/test/getcrumb"
        try:
            async with self._session.get(url) as response:
                if response.status == 200:
                    crumb = await response.text()
                    self.config.params["crumb"] = crumb
                    
                    # Get the cookie from the response
                    cookies = response.cookies
                    if "B" in cookies:
                        self.config.params["session_cookie"] = cookies["B"].value
                        
                    self.logger.info("Successfully fetched Yahoo Finance crumb")
                else:
                    self.logger.warning(f"Failed to fetch crumb, status: {response.status}")
        except Exception as e:
            self.logger.error(f"Error fetching crumb: {e}")
            raise
    
    async def _make_request(self, endpoint: str, params: Dict[str, Any]) -> Any:
        """
        Make a request to the Yahoo Finance API.
        
        Args:
            endpoint: API endpoint
            params: Query parameters
            
        Returns:
            Response data
        """
        await self._ensure_initialized()
        await self._rate_limit()
        
        url = f"{self.config.base_url}{endpoint}"
        
        # Add crumb if available
        if self.config.params.get("crumb"):
            params["crumb"] = self.config.params["crumb"]
            
        headers = {}
        if self.config.params.get("session_cookie"):
            headers["Cookie"] = f"B={self.config.params['session_cookie']}"
        
        for attempt in range(self.config.max_retries + 1):
            try:
                async with self._session.get(url, params=params, headers=headers) as response:
                    if response.status == 200:
                        return await response.json()
                    elif response.status == 429:  # Too Many Requests
                        wait_time = int(response.headers.get("Retry-After", 60))
                        self.logger.warning(f"Rate limited, waiting {wait_time}s")
                        await asyncio.sleep(wait_time)
                    else:
                        error_text = await response.text()
                        self.logger.error(f"API error: {response.status}, {error_text}")
                        
                        if attempt == self.config.max_retries:
                            raise DataFetchError(
                                f"Failed after {attempt + 1} attempts: {error_text}", 
                                "YahooFinance", 
                                response.status
                            )
                        
                        # Exponential backoff
                        wait_time = 2 ** attempt
                        self.logger.info(f"Retrying in {wait_time}s, attempt {attempt + 1}/{self.config.max_retries}")
                        await asyncio.sleep(wait_time)
            except asyncio.TimeoutError:
                if attempt == self.config.max_retries:
                    raise DataFetchError("Request timed out", "YahooFinance")
                    
                wait_time = 2 ** attempt
                self.logger.info(f"Timeout, retrying in {wait_time}s, attempt {attempt + 1}/{self.config.max_retries}")
                await asyncio.sleep(wait_time)
            except Exception as e:
                if attempt == self.config.max_retries:
                    raise DataFetchError(f"Request failed: {str(e)}", "YahooFinance")
                    
                wait_time = 2 ** attempt
                self.logger.info(f"Error: {e}, retrying in {wait_time}s, attempt {attempt + 1}/{self.config.max_retries}")
                await asyncio.sleep(wait_time)
    
    async def get_assets(self) -> List[Asset]:
        """
        Get a list of available assets.
        
        This is a simplified implementation that returns common indices.
        In a real implementation, this would fetch from an API.
        
        Returns:
            List of assets
        """
        # In a real implementation, this would fetch from an API
        # For simplicity, we just return a few common indices
        common_indices = [
            Asset(
                symbol="^GSPC",
                name="S&P 500",
                asset_type=AssetType.INDEX,
                exchange="SNP",
                currency="USD",
                country="USA",
                description="Standard & Poor's 500 Index"
            ),
            Asset(
                symbol="^DJI",
                name="Dow Jones Industrial Average",
                asset_type=AssetType.INDEX,
                exchange="DJI",
                currency="USD",
                country="USA",
                description="Dow Jones Industrial Average"
            ),
            Asset(
                symbol="^IXIC",
                name="NASDAQ Composite",
                asset_type=AssetType.INDEX,
                exchange="NASDAQ",
                currency="USD",
                country="USA",
                description="NASDAQ Composite Index"
            ),
            Asset(
                symbol="^FTSE",
                name="FTSE 100",
                asset_type=AssetType.INDEX,
                exchange="FTSE",
                currency="GBP",
                country="UK",
                description="Financial Times Stock Exchange 100 Index"
            ),
            Asset(
                symbol="^N225",
                name="Nikkei 225",
                asset_type=AssetType.INDEX,
                exchange="Nikkei",
                currency="JPY",
                country="Japan",
                description="Nikkei 225 Index"
            )
        ]
        
        # Cache this result
        cache_key = self._get_cache_key("assets")
        await self._set_in_cache(cache_key, common_indices, 86400)  # 24 hours TTL
        
        return common_indices
    
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
            data = await self._make_request("/v7/finance/quote", {"symbols": symbol})
            result = data.get("quoteResponse", {}).get("result", [])
            
            if not result:
                return None
                
            quote = result[0]
            
            # Determine asset type
            asset_type = AssetType.STOCK
            if quote.get("quoteType") == "ETF":
                asset_type = AssetType.ETF
            elif quote.get("quoteType") == "INDEX":
                asset_type = AssetType.INDEX
            
            asset = Asset(
                symbol=quote.get("symbol"),
                name=quote.get("longName", quote.get("shortName", quote.get("symbol"))),
                asset_type=asset_type,
                exchange=quote.get("exchange"),
                currency=quote.get("currency", "USD"),
                sector=quote.get("sector"),
                country=quote.get("country"),
                description=quote.get("longBusinessSummary"),
                metadata={
                    "market_cap": quote.get("marketCap"),
                    "fifty_two_week_high": quote.get("fiftyTwoWeekHigh"),
                    "fifty_two_week_low": quote.get("fiftyTwoWeekLow"),
                    "average_volume": quote.get("averageVolume"),
                    "exchange_timezone": quote.get("exchangeTimezoneName")
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
        
        For Yahoo Finance, we don't have a direct concept of asset pairs.
        This is more applicable to cryptocurrency or forex providers.
        
        Returns:
            Empty list (not applicable for Yahoo Finance)
        """
        # Not applicable for Yahoo Finance
        return []
    
    async def get_asset_pair(self, symbol: str) -> Optional[AssetPair]:
        """
        Get details for a specific asset pair.
        
        For Yahoo Finance, we don't have a direct concept of asset pairs.
        
        Args:
            symbol: Pair symbol
            
        Returns:
            None (not applicable for Yahoo Finance)
        """
        # Not applicable for Yahoo Finance
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
        # Convert interval to string if it's an enum
        if isinstance(interval, CandlestickInterval):
            yahoo_interval = YAHOO_INTERVALS.get(interval)
            if not yahoo_interval:
                raise ValueError(f"Unsupported interval: {interval}")
        else:
            # Try to convert string to enum first
            try:
                enum_interval = CandlestickInterval(interval)
                yahoo_interval = YAHOO_INTERVALS.get(enum_interval)
                if not yahoo_interval:
                    yahoo_interval = interval  # Use as-is if not found
            except ValueError:
                yahoo_interval = interval  # Use as-is if not a valid enum value
        
        # Set default times if not provided
        if end_time is None:
            end_time = datetime.datetime.now()
            
        if start_time is None:
            # Default to 30 days for daily candles, 7 days for intraday
            if yahoo_interval in ["1d", "1wk", "1mo"]:
                start_time = end_time - datetime.timedelta(days=30)
            else:
                start_time = end_time - datetime.timedelta(days=7)
        
        # Convert times to Unix timestamp (seconds)
        period1 = int(start_time.timestamp())
        period2 = int(end_time.timestamp())
        
        # Check cache
        if use_cache:
            cache_key = self._get_cache_key(
                "candlestick",
                symbol=symbol,
                interval=yahoo_interval,
                period1=period1,
                period2=period2
            )
            cached = await self._get_from_cache(cache_key)
            if cached:
                return cached
        
        # Fetch data
        params = {
            "symbol": symbol,
            "interval": yahoo_interval,
            "period1": period1,
            "period2": period2,
            "includePrePost": "false",
            "events": "div,split",
            "corsDomain": "finance.yahoo.com"
        }
        
        try:
            data = await self._make_request("/v8/finance/chart/" + symbol, params)
            
            # Check for errors
            error = data.get("chart", {}).get("error")
            if error:
                raise DataFetchError(
                    f"Yahoo Finance API error: {error.get('description', 'Unknown error')}",
                    "YahooFinance"
                )
            
            # Extract the results
            result = data.get("chart", {}).get("result", [])
            if not result:
                raise DataFetchError(
                    "No data returned from Yahoo Finance",
                    "YahooFinance"
                )
            
            chart_data = result[0]
            
            # Extract timestamps and values
            timestamps = chart_data.get("timestamp", [])
            quote = chart_data.get("indicators", {}).get("quote", [{}])[0]
            
            opens = quote.get("open", [])
            highs = quote.get("high", [])
            lows = quote.get("low", [])
            closes = quote.get("close", [])
            volumes = quote.get("volume", [])
            
            # Convert to DataFrame for easier manipulation
            df = pd.DataFrame({
                "timestamp": timestamps,
                "open": opens,
                "high": highs,
                "low": lows,
                "close": closes,
                "volume": volumes
            })
            
            # Drop rows with missing data
            df = df.dropna()
            
            # Convert timestamps to datetime
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
            
            # Apply limit if specified
            if limit is not None and limit > 0:
                df = df.tail(limit)
            
            # Create candlesticks from the DataFrame
            candlesticks = []
            for _, row in df.iterrows():
                candlesticks.append(Candlestick(
                    timestamp=row["timestamp"].to_pydatetime(),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"])
                ))
            
            # Convert string interval back to enum if possible
            try:
                # Reverse lookup from Yahoo to CandlestickInterval
                reverse_intervals = {v: k for k, v in YAHOO_INTERVALS.items()}
                candlestick_interval = reverse_intervals.get(yahoo_interval)
                if not candlestick_interval:
                    # Try direct conversion
                    candlestick_interval = CandlestickInterval(yahoo_interval)
            except ValueError:
                # Fall back to a default if can't convert
                candlestick_interval = CandlestickInterval.ONE_DAY
            
            # Create the series
            series = CandlestickSeries(
                symbol=symbol,
                interval=candlestick_interval,
                candlesticks=candlesticks,
                source="YahooFinance",
                metadata={
                    "query_interval": yahoo_interval,
                    "timezone": chart_data.get("meta", {}).get("exchangeTimezoneName")
                }
            )
            
            # Cache the result
            if use_cache:
                # TTL depends on interval - shorter for intraday
                ttl = 300  # 5 minutes default
                if yahoo_interval in ["1d", "1wk", "1mo"]:
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
            data = await self._make_request("/v7/finance/quote", {"symbols": symbol})
            result = data.get("quoteResponse", {}).get("result", [])
            
            if not result:
                raise DataFetchError(f"No data returned for {symbol}", "YahooFinance")
                
            quote = result[0]
            price = quote.get("regularMarketPrice")
            
            if price is None:
                raise DataFetchError(f"No price data for {symbol}", "YahooFinance")
            
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