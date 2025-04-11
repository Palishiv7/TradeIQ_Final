"""
Market Data Models

This module defines data models for market data including assets, markets,
trading pairs, and historical market data.
"""

import datetime
import enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Union

from backend.common.serialization import SerializableMixin


class AssetType(enum.Enum):
    """Types of financial assets."""
    
    STOCK = "stock"
    ETF = "etf"
    INDEX = "index"
    FUTURE = "future"
    OPTION = "option"
    FOREX = "forex"
    CRYPTO = "crypto"
    COMMODITY = "commodity"
    BOND = "bond"


@dataclass
class Asset(SerializableMixin):
    """
    A financial asset that can be traded.
    
    Attributes:
        symbol: The ticker symbol of the asset
        name: The full name of the asset
        asset_type: The type of the asset
        exchange: The exchange where the asset is traded
        currency: The currency in which the asset is denominated
        sector: The sector of the asset (for stocks)
        country: The country of the asset
        description: A description of the asset
        metadata: Additional metadata about the asset
    """
    symbol: str
    name: str
    asset_type: AssetType
    exchange: Optional[str] = None
    currency: str = "USD"
    sector: Optional[str] = None
    country: Optional[str] = None
    description: Optional[str] = None
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert the asset to a dictionary."""
        return {
            "symbol": self.symbol,
            "name": self.name,
            "asset_type": self.asset_type.value,
            "exchange": self.exchange,
            "currency": self.currency,
            "sector": self.sector,
            "country": self.country,
            "description": self.description,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Asset':
        """Create an asset from a dictionary."""
        # Convert asset type
        asset_type = AssetType(data["asset_type"])
        
        return cls(
            symbol=data["symbol"],
            name=data["name"],
            asset_type=asset_type,
            exchange=data.get("exchange"),
            currency=data.get("currency", "USD"),
            sector=data.get("sector"),
            country=data.get("country"),
            description=data.get("description"),
            metadata=data.get("metadata", {})
        )
    
    @property
    def serializable_fields(self) -> List[str]:
        """Get a list of fields that should be serialized."""
        return [
            "symbol", "name", "asset_type", "exchange", "currency",
            "sector", "country", "description", "metadata"
        ]


@dataclass
class AssetPair(SerializableMixin):
    """
    A pair of assets that can be traded against each other.
    
    Attributes:
        symbol: The symbol of the pair (e.g., BTC/USD)
        base_asset: The base asset of the pair
        quote_asset: The quote asset of the pair
        exchange: The exchange where the pair is traded
        tick_size: The minimum price increment
        lot_size: The minimum quantity increment
        min_notional: The minimum notional value for an order
        metadata: Additional metadata about the pair
    """
    symbol: str
    base_asset: Asset
    quote_asset: Asset
    exchange: Optional[str] = None
    tick_size: float = 0.01
    lot_size: float = 1.0
    min_notional: float = 0.0
    metadata: Dict = field(default_factory=dict)
    
    @property
    def base_symbol(self) -> str:
        """Get the symbol of the base asset."""
        return self.base_asset.symbol
    
    @property
    def quote_symbol(self) -> str:
        """Get the symbol of the quote asset."""
        return self.quote_asset.symbol
    
    @property
    def currency(self) -> str:
        """Get the currency of the pair."""
        return self.quote_asset.currency
    
    def to_dict(self) -> Dict:
        """Convert the asset pair to a dictionary."""
        return {
            "symbol": self.symbol,
            "base_asset": self.base_asset.to_dict(),
            "quote_asset": self.quote_asset.to_dict(),
            "exchange": self.exchange,
            "tick_size": self.tick_size,
            "lot_size": self.lot_size,
            "min_notional": self.min_notional,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'AssetPair':
        """Create an asset pair from a dictionary."""
        # Convert assets
        base_asset = Asset.from_dict(data["base_asset"])
        quote_asset = Asset.from_dict(data["quote_asset"])
        
        return cls(
            symbol=data["symbol"],
            base_asset=base_asset,
            quote_asset=quote_asset,
            exchange=data.get("exchange"),
            tick_size=data.get("tick_size", 0.01),
            lot_size=data.get("lot_size", 1.0),
            min_notional=data.get("min_notional", 0.0),
            metadata=data.get("metadata", {})
        )
    
    @property
    def serializable_fields(self) -> List[str]:
        """Get a list of fields that should be serialized."""
        return [
            "symbol", "base_asset", "quote_asset", "exchange",
            "tick_size", "lot_size", "min_notional", "metadata"
        ]


@dataclass
class Market(SerializableMixin):
    """
    A market where assets and asset pairs are traded.
    
    Attributes:
        name: The name of the market
        description: A description of the market
        timezone: The timezone of the market
        currency: The primary currency of the market
        trading_hours: The trading hours of the market
        trading_days: The trading days of the market
        assets: Assets available on the market
        pairs: Asset pairs available on the market
        metadata: Additional metadata about the market
    """
    name: str
    description: Optional[str] = None
    timezone: str = "UTC"
    currency: str = "USD"
    trading_hours: Optional[str] = None
    trading_days: Optional[str] = None
    assets: List[Asset] = field(default_factory=list)
    pairs: List[AssetPair] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    
    def get_asset(self, symbol: str) -> Optional[Asset]:
        """
        Get an asset by its symbol.
        
        Args:
            symbol: The symbol to look for
            
        Returns:
            The asset with the given symbol, or None if not found
        """
        for asset in self.assets:
            if asset.symbol == symbol:
                return asset
        return None
    
    def get_pair(self, symbol: str) -> Optional[AssetPair]:
        """
        Get an asset pair by its symbol.
        
        Args:
            symbol: The symbol to look for
            
        Returns:
            The asset pair with the given symbol, or None if not found
        """
        for pair in self.pairs:
            if pair.symbol == symbol:
                return pair
        return None
    
    def get_assets_by_type(self, asset_type: Union[AssetType, str]) -> List[Asset]:
        """
        Get all assets of a specific type.
        
        Args:
            asset_type: The asset type to filter by
            
        Returns:
            List of assets of the specified type
        """
        # Convert string to enum if needed
        if isinstance(asset_type, str):
            asset_type = AssetType(asset_type)
            
        return [a for a in self.assets if a.asset_type == asset_type]
    
    def to_dict(self) -> Dict:
        """Convert the market to a dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "timezone": self.timezone,
            "currency": self.currency,
            "trading_hours": self.trading_hours,
            "trading_days": self.trading_days,
            "assets": [a.to_dict() for a in self.assets],
            "pairs": [p.to_dict() for p in self.pairs],
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Market':
        """Create a market from a dictionary."""
        # Convert assets and pairs
        assets = [Asset.from_dict(a) for a in data.get("assets", [])]
        pairs = [AssetPair.from_dict(p) for p in data.get("pairs", [])]
        
        return cls(
            name=data["name"],
            description=data.get("description"),
            timezone=data.get("timezone", "UTC"),
            currency=data.get("currency", "USD"),
            trading_hours=data.get("trading_hours"),
            trading_days=data.get("trading_days"),
            assets=assets,
            pairs=pairs,
            metadata=data.get("metadata", {})
        )
    
    @property
    def serializable_fields(self) -> List[str]:
        """Get a list of fields that should be serialized."""
        return [
            "name", "description", "timezone", "currency",
            "trading_hours", "trading_days", "assets", "pairs", "metadata"
        ]


@dataclass
class MarketData(SerializableMixin):
    """
    Market data for an asset or asset pair.
    
    Attributes:
        symbol: The symbol of the asset or pair
        data_type: The type of market data (e.g., ohlcv, trades, orderbook)
        start_time: The start time of the data
        end_time: The end time of the data
        data: The market data (format depends on data_type)
        source: The source of the data
        metadata: Additional metadata about the data
    """
    symbol: str
    data_type: str
    start_time: datetime.datetime
    end_time: datetime.datetime
    data: Dict
    source: Optional[str] = None
    metadata: Dict = field(default_factory=dict)
    
    @property
    def duration(self) -> datetime.timedelta:
        """Get the duration of the data."""
        return self.end_time - self.start_time
    
    def to_dict(self) -> Dict:
        """Convert the market data to a dictionary."""
        return {
            "symbol": self.symbol,
            "data_type": self.data_type,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "data": self.data,
            "source": self.source,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'MarketData':
        """Create market data from a dictionary."""
        # Convert timestamps
        start_time = datetime.datetime.fromisoformat(data["start_time"])
        end_time = datetime.datetime.fromisoformat(data["end_time"])
        
        return cls(
            symbol=data["symbol"],
            data_type=data["data_type"],
            start_time=start_time,
            end_time=end_time,
            data=data["data"],
            source=data.get("source"),
            metadata=data.get("metadata", {})
        )
    
    @property
    def serializable_fields(self) -> List[str]:
        """Get a list of fields that should be serialized."""
        return [
            "symbol", "data_type", "start_time", "end_time",
            "data", "source", "metadata"
        ] 