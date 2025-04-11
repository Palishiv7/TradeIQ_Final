"""
Candlestick Data Models

This module defines data models for candlestick chart data, including
individual candlesticks and candlestick series with different time intervals.
"""

import datetime
import enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from backend.common.serialization import SerializableMixin


class TimeFrame(enum.Enum):
    """Time frames for market data."""
    
    INTRADAY = "intraday"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"


class CandlestickInterval(enum.Enum):
    """Intervals for candlestick data."""
    
    # Intraday intervals
    ONE_MINUTE = "1m"
    THREE_MINUTES = "3m"
    FIVE_MINUTES = "5m"
    FIFTEEN_MINUTES = "15m"
    THIRTY_MINUTES = "30m"
    ONE_HOUR = "1h"
    TWO_HOURS = "2h"
    FOUR_HOURS = "4h"
    
    # Daily intervals
    ONE_DAY = "1d"
    
    # Weekly intervals
    ONE_WEEK = "1w"
    
    # Monthly intervals
    ONE_MONTH = "1M"
    
    @property
    def seconds(self) -> int:
        """Get the interval duration in seconds."""
        if self == CandlestickInterval.ONE_MINUTE:
            return 60
        elif self == CandlestickInterval.THREE_MINUTES:
            return 3 * 60
        elif self == CandlestickInterval.FIVE_MINUTES:
            return 5 * 60
        elif self == CandlestickInterval.FIFTEEN_MINUTES:
            return 15 * 60
        elif self == CandlestickInterval.THIRTY_MINUTES:
            return 30 * 60
        elif self == CandlestickInterval.ONE_HOUR:
            return 60 * 60
        elif self == CandlestickInterval.TWO_HOURS:
            return 2 * 60 * 60
        elif self == CandlestickInterval.FOUR_HOURS:
            return 4 * 60 * 60
        elif self == CandlestickInterval.ONE_DAY:
            return 24 * 60 * 60
        elif self == CandlestickInterval.ONE_WEEK:
            return 7 * 24 * 60 * 60
        elif self == CandlestickInterval.ONE_MONTH:
            # Approximate as 30 days
            return 30 * 24 * 60 * 60
        else:
            raise ValueError(f"Unknown interval: {self}")
    
    @property
    def time_frame(self) -> TimeFrame:
        """Get the corresponding time frame for this interval."""
        if self in [
            CandlestickInterval.ONE_MINUTE,
            CandlestickInterval.THREE_MINUTES,
            CandlestickInterval.FIVE_MINUTES,
            CandlestickInterval.FIFTEEN_MINUTES,
            CandlestickInterval.THIRTY_MINUTES,
            CandlestickInterval.ONE_HOUR,
            CandlestickInterval.TWO_HOURS,
            CandlestickInterval.FOUR_HOURS,
        ]:
            return TimeFrame.INTRADAY
        elif self == CandlestickInterval.ONE_DAY:
            return TimeFrame.DAILY
        elif self == CandlestickInterval.ONE_WEEK:
            return TimeFrame.WEEKLY
        elif self == CandlestickInterval.ONE_MONTH:
            return TimeFrame.MONTHLY
        else:
            raise ValueError(f"Unknown interval: {self}")
    
    @classmethod
    def from_string(cls, value: str) -> 'CandlestickInterval':
        """Create an interval from a string representation."""
        for interval in cls:
            if interval.value == value:
                return interval
        raise ValueError(f"Unknown interval: {value}")


@dataclass
class Candlestick(SerializableMixin):
    """
    A single candlestick in a candlestick chart.
    
    Attributes:
        timestamp: The timestamp for this candlestick
        open: The opening price
        high: The highest price during the period
        low: The lowest price during the period
        close: The closing price
        volume: The trading volume during the period
        trades: The number of trades during the period (optional)
        vwap: Volume-weighted average price (optional)
    """
    timestamp: datetime.datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    trades: Optional[int] = None
    vwap: Optional[float] = None
    
    @property
    def body_size(self) -> float:
        """Get the absolute size of the candlestick body."""
        return abs(self.close - self.open)
    
    @property
    def upper_shadow(self) -> float:
        """Get the size of the upper shadow."""
        return self.high - max(self.open, self.close)
    
    @property
    def lower_shadow(self) -> float:
        """Get the size of the lower shadow."""
        return min(self.open, self.close) - self.low
    
    @property
    def is_bullish(self) -> bool:
        """Check if this is a bullish (green) candlestick."""
        return self.close > self.open
    
    @property
    def is_bearish(self) -> bool:
        """Check if this is a bearish (red) candlestick."""
        return self.close < self.open
    
    @property
    def is_doji(self) -> bool:
        """Check if this is a doji candlestick (open ≈ close)."""
        body_to_range_ratio = self.body_size / (self.high - self.low) if self.high != self.low else 0
        return body_to_range_ratio < 0.1
    
    @property
    def range(self) -> float:
        """Get the full range of the candlestick (high - low)."""
        return self.high - self.low
    
    @property
    def body_to_range_ratio(self) -> float:
        """Get the ratio of body size to full range."""
        if self.range == 0:
            return 0
        return self.body_size / self.range
    
    def to_dict(self) -> Dict:
        """Convert the candlestick to a dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "trades": self.trades,
            "vwap": self.vwap
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Candlestick':
        """Create a candlestick from a dictionary."""
        timestamp = data["timestamp"]
        if isinstance(timestamp, str):
            timestamp = datetime.datetime.fromisoformat(timestamp)
        
        return cls(
            timestamp=timestamp,
            open=data["open"],
            high=data["high"],
            low=data["low"],
            close=data["close"],
            volume=data["volume"],
            trades=data.get("trades"),
            vwap=data.get("vwap")
        )
    
    @property
    def serializable_fields(self) -> List[str]:
        """Get a list of fields that should be serialized."""
        return [
            "timestamp", "open", "high", "low", "close", 
            "volume", "trades", "vwap"
        ]


@dataclass
class CandlestickSeries(SerializableMixin):
    """
    A series of candlesticks forming a chart.
    
    Attributes:
        symbol: The symbol/ticker of the asset
        interval: The time interval between candlesticks
        candlesticks: List of candlesticks in the series
        start_time: The start time of the series
        end_time: The end time of the series
        source: The data source
        metadata: Additional metadata
    """
    symbol: str
    interval: CandlestickInterval
    candlesticks: List[Candlestick] = field(default_factory=list)
    start_time: Optional[datetime.datetime] = None
    end_time: Optional[datetime.datetime] = None
    source: Optional[str] = None
    metadata: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize the start and end times if not provided."""
        if self.candlesticks:
            if self.start_time is None:
                self.start_time = min(c.timestamp for c in self.candlesticks)
            if self.end_time is None:
                self.end_time = max(c.timestamp for c in self.candlesticks)
    
    def add_candlestick(self, candlestick: Candlestick) -> None:
        """
        Add a candlestick to the series.
        
        Args:
            candlestick: The candlestick to add
        """
        self.candlesticks.append(candlestick)
        
        # Update start and end times
        if self.start_time is None or candlestick.timestamp < self.start_time:
            self.start_time = candlestick.timestamp
        if self.end_time is None or candlestick.timestamp > self.end_time:
            self.end_time = candlestick.timestamp
    
    def get_candlestick_at(self, timestamp: datetime.datetime) -> Optional[Candlestick]:
        """
        Get the candlestick at a specific timestamp.
        
        Args:
            timestamp: The timestamp to look for
            
        Returns:
            The candlestick at the timestamp, or None if not found
        """
        for candlestick in self.candlesticks:
            if candlestick.timestamp == timestamp:
                return candlestick
        return None
    
    def slice(
        self, 
        start_time: Optional[datetime.datetime] = None,
        end_time: Optional[datetime.datetime] = None,
        count: Optional[int] = None
    ) -> 'CandlestickSeries':
        """
        Create a slice of this series.
        
        Args:
            start_time: The start time of the slice (inclusive)
            end_time: The end time of the slice (inclusive)
            count: The number of most recent candlesticks to include
            
        Returns:
            A new CandlestickSeries with the selected candlesticks
        """
        # Use the series start/end times if not provided
        if start_time is None:
            start_time = self.start_time
        if end_time is None:
            end_time = self.end_time
        
        # Filter candlesticks by time range
        filtered = [
            c for c in self.candlesticks
            if (start_time is None or c.timestamp >= start_time) and
               (end_time is None or c.timestamp <= end_time)
        ]
        
        # Sort by timestamp
        filtered.sort(key=lambda c: c.timestamp)
        
        # Apply count limit if specified
        if count is not None:
            filtered = filtered[-count:]
        
        # Create a new series
        return CandlestickSeries(
            symbol=self.symbol,
            interval=self.interval,
            candlesticks=filtered,
            source=self.source,
            metadata=self.metadata.copy()
        )
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert the candlestick series to a pandas DataFrame.
        
        Returns:
            A pandas DataFrame with the candlestick data
        """
        data = []
        for candle in self.candlesticks:
            data.append({
                "timestamp": candle.timestamp,
                "open": candle.open,
                "high": candle.high,
                "low": candle.low,
                "close": candle.close,
                "volume": candle.volume,
                "trades": candle.trades,
                "vwap": candle.vwap
            })
        
        df = pd.DataFrame(data)
        if not df.empty:
            df.set_index("timestamp", inplace=True)
            df.sort_index(inplace=True)
        
        return df
    
    @classmethod
    def from_dataframe(
        cls, 
        df: pd.DataFrame, 
        symbol: str, 
        interval: Union[CandlestickInterval, str],
        source: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> 'CandlestickSeries':
        """
        Create a candlestick series from a pandas DataFrame.
        
        Args:
            df: The DataFrame containing OHLCV data
            symbol: The symbol/ticker of the asset
            interval: The time interval between candlesticks
            source: The data source
            metadata: Additional metadata
            
        Returns:
            A new CandlestickSeries
        """
        # Convert interval string to enum if needed
        if isinstance(interval, str):
            interval = CandlestickInterval.from_string(interval)
        
        # Ensure the DataFrame has the required columns
        required_columns = ["open", "high", "low", "close", "volume"]
        df_columns = df.columns if not df.empty else []
        
        missing_columns = [col for col in required_columns if col not in df_columns]
        if missing_columns:
            raise ValueError(f"DataFrame missing required columns: {missing_columns}")
        
        # Create candlesticks from the DataFrame
        candlesticks = []
        for timestamp, row in df.iterrows():
            candlesticks.append(Candlestick(
                timestamp=timestamp,
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
                trades=int(row["trades"]) if "trades" in row else None,
                vwap=float(row["vwap"]) if "vwap" in row else None
            ))
        
        # Create the series
        return cls(
            symbol=symbol,
            interval=interval,
            candlesticks=candlesticks,
            source=source,
            metadata=metadata or {}
        )
    
    def to_dict(self) -> Dict:
        """Convert the candlestick series to a dictionary."""
        return {
            "symbol": self.symbol,
            "interval": self.interval.value,
            "candlesticks": [c.to_dict() for c in self.candlesticks],
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "source": self.source,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'CandlestickSeries':
        """Create a candlestick series from a dictionary."""
        # Convert interval string to enum
        interval = CandlestickInterval.from_string(data["interval"])
        
        # Convert timestamps
        start_time = data.get("start_time")
        if start_time and isinstance(start_time, str):
            start_time = datetime.datetime.fromisoformat(start_time)
        
        end_time = data.get("end_time")
        if end_time and isinstance(end_time, str):
            end_time = datetime.datetime.fromisoformat(end_time)
        
        # Convert candlesticks
        candlesticks = [
            Candlestick.from_dict(c) for c in data.get("candlesticks", [])
        ]
        
        return cls(
            symbol=data["symbol"],
            interval=interval,
            candlesticks=candlesticks,
            start_time=start_time,
            end_time=end_time,
            source=data.get("source"),
            metadata=data.get("metadata", {})
        )
    
    @property
    def serializable_fields(self) -> List[str]:
        """Get a list of fields that should be serialized."""
        return [
            "symbol", "interval", "candlesticks", "start_time", 
            "end_time", "source", "metadata"
        ] 