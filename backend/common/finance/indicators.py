"""
Technical Indicators Models

This module defines data models for technical indicators used in financial analysis,
such as moving averages, oscillators, and volatility measures.
"""

import datetime
import enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd

from backend.common.serialization import SerializableMixin
from backend.common.finance.candlestick import CandlestickSeries


class IndicatorType(enum.Enum):
    """Types of technical indicators."""
    
    # Trend indicators
    SMA = "simple_moving_average"
    EMA = "exponential_moving_average"
    WMA = "weighted_moving_average"
    MACD = "moving_average_convergence_divergence"
    ADX = "average_directional_index"
    
    # Momentum indicators
    RSI = "relative_strength_index"
    CCI = "commodity_channel_index"
    STOCHASTIC = "stochastic_oscillator"
    WILLIAMS_R = "williams_percent_r"
    MFI = "money_flow_index"
    
    # Volatility indicators
    BOLLINGER_BANDS = "bollinger_bands"
    ATR = "average_true_range"
    STANDARD_DEVIATION = "standard_deviation"
    
    # Volume indicators
    OBV = "on_balance_volume"
    ADI = "accumulation_distribution_index"
    VOLUME_SMA = "volume_simple_moving_average"
    CHAIKIN_MONEY_FLOW = "chaikin_money_flow"
    
    # Support/Resistance indicators
    PIVOT_POINTS = "pivot_points"
    FIBONACCI_RETRACEMENT = "fibonacci_retracement"
    ICHIMOKU_CLOUD = "ichimoku_cloud"
    
    # Other indicators
    CUSTOM = "custom"
    
    @property
    def is_trend_indicator(self) -> bool:
        """Check if this is a trend indicator."""
        trend_indicators = {
            IndicatorType.SMA,
            IndicatorType.EMA,
            IndicatorType.WMA,
            IndicatorType.MACD,
            IndicatorType.ADX
        }
        return self in trend_indicators
    
    @property
    def is_momentum_indicator(self) -> bool:
        """Check if this is a momentum indicator."""
        momentum_indicators = {
            IndicatorType.RSI,
            IndicatorType.CCI,
            IndicatorType.STOCHASTIC,
            IndicatorType.WILLIAMS_R,
            IndicatorType.MFI
        }
        return self in momentum_indicators
    
    @property
    def is_volatility_indicator(self) -> bool:
        """Check if this is a volatility indicator."""
        volatility_indicators = {
            IndicatorType.BOLLINGER_BANDS,
            IndicatorType.ATR,
            IndicatorType.STANDARD_DEVIATION
        }
        return self in volatility_indicators
    
    @property
    def is_volume_indicator(self) -> bool:
        """Check if this is a volume indicator."""
        volume_indicators = {
            IndicatorType.OBV,
            IndicatorType.ADI,
            IndicatorType.VOLUME_SMA,
            IndicatorType.CHAIKIN_MONEY_FLOW
        }
        return self in volume_indicators
    
    @property
    def is_support_resistance_indicator(self) -> bool:
        """Check if this is a support/resistance indicator."""
        support_resistance_indicators = {
            IndicatorType.PIVOT_POINTS,
            IndicatorType.FIBONACCI_RETRACEMENT,
            IndicatorType.ICHIMOKU_CLOUD
        }
        return self in support_resistance_indicators


@dataclass
class IndicatorValue(SerializableMixin):
    """
    A single value of a technical indicator at a specific time.
    
    Attributes:
        timestamp: The timestamp of the value
        value: The main indicator value
        additional_values: Additional values for multi-value indicators
    """
    timestamp: datetime.datetime
    value: float
    additional_values: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert the indicator value to a dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "value": self.value,
            "additional_values": self.additional_values
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'IndicatorValue':
        """Create an indicator value from a dictionary."""
        # Convert timestamp
        timestamp = datetime.datetime.fromisoformat(data["timestamp"])
        
        return cls(
            timestamp=timestamp,
            value=data["value"],
            additional_values=data.get("additional_values", {})
        )
    
    @property
    def serializable_fields(self) -> List[str]:
        """Get a list of fields that should be serialized."""
        return [
            "timestamp", "value", "additional_values"
        ]


@dataclass
class IndicatorSeries(SerializableMixin):
    """
    A series of technical indicator values over time.
    
    Attributes:
        symbol: The symbol/ticker the indicator is calculated for
        indicator_type: The type of indicator
        parameters: Parameters used to calculate the indicator
        values: The indicator values
        start_time: The start time of the series
        end_time: The end time of the series
        additional_series: Additional data series for multi-series indicators
        metadata: Additional metadata about the indicator
    """
    symbol: str
    indicator_type: IndicatorType
    parameters: Dict[str, Any]
    values: List[IndicatorValue] = field(default_factory=list)
    start_time: Optional[datetime.datetime] = None
    end_time: Optional[datetime.datetime] = None
    additional_series: Dict[str, List[IndicatorValue]] = field(default_factory=dict)
    metadata: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize the start and end times if not provided."""
        if self.values:
            if self.start_time is None:
                self.start_time = min(v.timestamp for v in self.values)
            if self.end_time is None:
                self.end_time = max(v.timestamp for v in self.values)
    
    @property
    def name(self) -> str:
        """Get a friendly name for the indicator series."""
        indicator_name = self.indicator_type.value
        
        # Add key parameters to the name
        param_str = ""
        if "period" in self.parameters:
            param_str += f"({self.parameters['period']})"
        
        return f"{indicator_name}{param_str}"
    
    def add_value(self, value: IndicatorValue) -> None:
        """
        Add an indicator value to the series.
        
        Args:
            value: The indicator value to add
        """
        self.values.append(value)
        
        # Update start and end times
        if self.start_time is None or value.timestamp < self.start_time:
            self.start_time = value.timestamp
        if self.end_time is None or value.timestamp > self.end_time:
            self.end_time = value.timestamp
    
    def get_value_at(self, timestamp: datetime.datetime) -> Optional[IndicatorValue]:
        """
        Get the indicator value at a specific timestamp.
        
        Args:
            timestamp: The timestamp to look for
            
        Returns:
            The indicator value at the timestamp, or None if not found
        """
        for value in self.values:
            if value.timestamp == timestamp:
                return value
        return None
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert the indicator series to a pandas DataFrame.
        
        Returns:
            A pandas DataFrame with the indicator values
        """
        # Create data for the main series
        data = []
        for value in self.values:
            row = {
                "timestamp": value.timestamp,
                "value": value.value
            }
            # Add additional values
            for key, val in value.additional_values.items():
                row[key] = val
            data.append(row)
        
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
        indicator_type: Union[IndicatorType, str],
        parameters: Dict[str, Any],
        value_column: str = "value",
        additional_value_columns: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict] = None
    ) -> 'IndicatorSeries':
        """
        Create an indicator series from a pandas DataFrame.
        
        Args:
            df: The DataFrame containing indicator data
            symbol: The symbol/ticker of the asset
            indicator_type: The type of indicator
            parameters: Parameters used to calculate the indicator
            value_column: The column containing the main indicator values
            additional_value_columns: Mapping of additional value names to column names
            metadata: Additional metadata
            
        Returns:
            A new IndicatorSeries
        """
        # Convert indicator type string to enum if needed
        if isinstance(indicator_type, str):
            indicator_type = IndicatorType(indicator_type)
        
        # Ensure DataFrame has the required columns
        if value_column not in df.columns:
            raise ValueError(f"DataFrame missing required column: {value_column}")
        
        # Create indicator values from the DataFrame
        values = []
        for timestamp, row in df.iterrows():
            additional_values = {}
            if additional_value_columns:
                for key, column in additional_value_columns.items():
                    if column in row:
                        additional_values[key] = row[column]
            
            values.append(IndicatorValue(
                timestamp=timestamp,
                value=float(row[value_column]),
                additional_values=additional_values
            ))
        
        # Create the series
        return cls(
            symbol=symbol,
            indicator_type=indicator_type,
            parameters=parameters,
            values=values,
            metadata=metadata or {}
        )
    
    def to_dict(self) -> Dict:
        """Convert the indicator series to a dictionary."""
        return {
            "symbol": self.symbol,
            "indicator_type": self.indicator_type.value,
            "parameters": self.parameters,
            "values": [v.to_dict() for v in self.values],
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "additional_series": {
                key: [v.to_dict() for v in values]
                for key, values in self.additional_series.items()
            },
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'IndicatorSeries':
        """Create an indicator series from a dictionary."""
        # Convert indicator type
        indicator_type = IndicatorType(data["indicator_type"])
        
        # Convert timestamps
        start_time = data.get("start_time")
        if start_time and isinstance(start_time, str):
            start_time = datetime.datetime.fromisoformat(start_time)
        
        end_time = data.get("end_time")
        if end_time and isinstance(end_time, str):
            end_time = datetime.datetime.fromisoformat(end_time)
        
        # Convert values
        values = [
            IndicatorValue.from_dict(v) for v in data.get("values", [])
        ]
        
        # Convert additional series
        additional_series = {}
        for key, series_values in data.get("additional_series", {}).items():
            additional_series[key] = [
                IndicatorValue.from_dict(v) for v in series_values
            ]
        
        return cls(
            symbol=data["symbol"],
            indicator_type=indicator_type,
            parameters=data["parameters"],
            values=values,
            start_time=start_time,
            end_time=end_time,
            additional_series=additional_series,
            metadata=data.get("metadata", {})
        )
    
    @property
    def serializable_fields(self) -> List[str]:
        """Get a list of fields that should be serialized."""
        return [
            "symbol", "indicator_type", "parameters", "values",
            "start_time", "end_time", "additional_series", "metadata"
        ]


@dataclass
class Indicator(SerializableMixin):
    """
    A technical indicator configuration.
    
    Attributes:
        name: The name of the indicator
        indicator_type: The type of indicator
        description: A description of the indicator
        parameters: Parameters for calculating the indicator
        formula: The formula used by the indicator
        interpretation: Guidelines for interpreting the indicator
        metadata: Additional metadata about the indicator
    """
    name: str
    indicator_type: IndicatorType
    description: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    formula: Optional[str] = None
    interpretation: Optional[str] = None
    metadata: Dict = field(default_factory=dict)
    
    def calculate(self, series: CandlestickSeries) -> IndicatorSeries:
        """
        Calculate the indicator values for a candlestick series.
        
        Args:
            series: The candlestick series to calculate the indicator for
            
        Returns:
            An indicator series with the calculated values
            
        Raises:
            NotImplementedError: This is a base class and doesn't implement calculation logic
        """
        raise NotImplementedError(
            "Indicator is a base class that doesn't implement calculation logic. "
            "Use a specific indicator implementation instead."
        )
    
    def to_dict(self) -> Dict:
        """Convert the indicator to a dictionary."""
        return {
            "name": self.name,
            "indicator_type": self.indicator_type.value,
            "description": self.description,
            "parameters": self.parameters,
            "formula": self.formula,
            "interpretation": self.interpretation,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Indicator':
        """Create an indicator from a dictionary."""
        # Convert indicator type
        indicator_type = IndicatorType(data["indicator_type"])
        
        return cls(
            name=data["name"],
            indicator_type=indicator_type,
            description=data.get("description"),
            parameters=data.get("parameters", {}),
            formula=data.get("formula"),
            interpretation=data.get("interpretation"),
            metadata=data.get("metadata", {})
        )
    
    @property
    def serializable_fields(self) -> List[str]:
        """Get a list of fields that should be serialized."""
        return [
            "name", "indicator_type", "description", "parameters",
            "formula", "interpretation", "metadata"
        ] 