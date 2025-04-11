"""
Data Providers Package

This package provides market data providers for candlestick pattern assessments.
"""

import asyncio
import logging
from typing import Dict, Any, Optional

from backend.common.finance.data_provider import (
    DataProviderRegistry, DataProviderConfig, DataProviderType, MarketDataService
)
from backend.assessments.candlestick_patterns.data_providers.yahoo_finance_provider import (
    YahooFinanceProvider, DEFAULT_CONFIG as YAHOO_DEFAULT_CONFIG
)
from backend.assessments.candlestick_patterns.data_providers.alpha_vantage_provider import (
    AlphaVantageProvider, DEFAULT_CONFIG as ALPHA_VANTAGE_DEFAULT_CONFIG
)

logger = logging.getLogger(__name__)


async def initialize_providers(
    yahoo_config: Optional[Dict[str, Any]] = None,
    alpha_vantage_config: Optional[Dict[str, Any]] = None
) -> MarketDataService:
    """
    Initialize and configure data providers.
    
    Args:
        yahoo_config: Configuration overrides for Yahoo Finance
        alpha_vantage_config: Configuration overrides for Alpha Vantage
        
    Returns:
        Configured MarketDataService
    """
    registry = DataProviderRegistry.get_instance()
    
    # Configure Yahoo Finance provider
    yahoo_provider_config = YAHOO_DEFAULT_CONFIG
    if yahoo_config:
        # Update with user-provided config
        for key, value in yahoo_config.items():
            if key == "params" and hasattr(yahoo_provider_config, "params"):
                yahoo_provider_config.params.update(value)
            else:
                setattr(yahoo_provider_config, key, value)
    
    # Configure Alpha Vantage provider
    alpha_provider_config = ALPHA_VANTAGE_DEFAULT_CONFIG
    if alpha_vantage_config:
        # Update with user-provided config
        for key, value in alpha_vantage_config.items():
            if key == "params" and hasattr(alpha_provider_config, "params"):
                alpha_provider_config.params.update(value)
            else:
                setattr(alpha_provider_config, key, value)
    
    # Create providers
    yahoo_provider = YahooFinanceProvider(yahoo_provider_config)
    alpha_provider = AlphaVantageProvider(alpha_provider_config)
    
    # Register providers with registry
    registry.register_provider(yahoo_provider, is_primary=True)
    registry.register_provider(alpha_provider, is_backup=True)
    
    logger.info("Data providers initialized: Yahoo Finance (primary), Alpha Vantage (backup)")
    
    # Return the market data service
    return MarketDataService.get_instance()


def get_market_data_service() -> MarketDataService:
    """
    Get the configured market data service.
    
    Returns:
        MarketDataService instance
    """
    return MarketDataService.get_instance()


__all__ = [
    'initialize_providers',
    'get_market_data_service',
    'YahooFinanceProvider',
    'AlphaVantageProvider',
] 