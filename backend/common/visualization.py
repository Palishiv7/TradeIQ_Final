"""
Visualization utilities for generating and encoding chart images.
"""

import base64
import io
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
from typing import List, Dict, Any, Optional

def generate_chart_image(data: List[Dict[str, Any]], 
                        pattern_regions: Optional[List[Dict[str, Any]]] = None,
                        width: int = 10,
                        height: int = 6,
                        style: str = 'charles') -> bytes:
    """
    Generate a candlestick chart image from OHLCV data.
    
    Args:
        data: List of dictionaries containing OHLCV data
        pattern_regions: Optional list of pattern regions to highlight
        width: Chart width in inches
        height: Chart height in inches
        style: MPLFinance style to use
        
    Returns:
        Bytes containing the PNG image data
    """
    # Convert data to pandas DataFrame
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    
    # Setup the plot
    mc = mpf.make_marketcolors(up='green',down='red',
                              edge='inherit',
                              wick='inherit',
                              volume='in')
    s  = mpf.make_mpf_style(base_mpf_style=style, marketcolors=mc)
    
    # Create a figure
    fig, ax = plt.subplots(figsize=(width, height))
    
    # Plot candlesticks
    mpf.plot(df, type='candle', style=s, ax=ax)
    
    # Add pattern regions if provided
    if pattern_regions:
        for region in pattern_regions:
            start_idx = df.index.get_loc(pd.to_datetime(region['start_date']))
            end_idx = df.index.get_loc(pd.to_datetime(region['end_date']))
            ax.axvspan(df.index[start_idx], df.index[end_idx], 
                      alpha=0.2, color=region.get('color', 'yellow'))
    
    # Save to bytes
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    
    return buf.getvalue()

def base64_encode_image(image_data: bytes) -> str:
    """
    Encode image data as base64 string.
    
    Args:
        image_data: Raw bytes of the image
        
    Returns:
        Base64 encoded string
    """
    return base64.b64encode(image_data).decode('utf-8') 