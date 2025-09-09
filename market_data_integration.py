"""
Real Market Data Integration Module

This module provides integration with real market data sources
for live trading and backtesting with historical data.
"""

import asyncio
import json
import websocket
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from abc import ABC, abstractmethod
import threading
import time
import requests
from urllib.parse import urljoin

@dataclass
class MarketData:
    """Market data point structure."""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    source: str = "unknown"

@dataclass
class MarketTick:
    """Real-time market tick data."""
    symbol: str
    timestamp: datetime
    price: float
    volume: float
    bid: Optional[float] = None
    ask: Optional[float] = None

class MarketDataProvider(ABC):
    """Abstract base class for market data providers."""
    
    @abstractmethod
    async def get_historical_data(
        self, 
        symbol: str, 
        interval: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> List[MarketData]:
        """Get historical market data."""
        pass
    
    @abstractmethod
    async def subscribe_realtime(
        self, 
        symbols: List[str], 
        callback: Callable[[MarketTick], None]
    ) -> None:
        """Subscribe to real-time market data."""
        pass
    
    @abstractmethod
    async def get_current_price(self, symbol: str) -> float:
        """Get current market price."""
        pass

class BinanceDataProvider(MarketDataProvider):
    """Binance market data provider implementation."""
    
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = "https://api.binance.com"
        self.ws_url = "wss://stream.binance.com:9443/ws/"
        self._ws_connections = {}
        self._callbacks = {}
    
    async def get_historical_data(
        self, 
        symbol: str, 
        interval: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> List[MarketData]:
        """Get historical data from Binance API."""
        try:
            # Convert interval to Binance format
            interval_map = {
                '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
                '1h': '1h', '4h': '4h', '1d': '1d', '1w': '1w'
            }
            binance_interval = interval_map.get(interval, '1h')
            
            # Prepare API request
            endpoint = "/api/v3/klines"
            params = {
                'symbol': symbol.upper(),
                'interval': binance_interval,
                'startTime': int(start_date.timestamp() * 1000),
                'endTime': int(end_date.timestamp() * 1000),
                'limit': 1000
            }
            
            response = requests.get(urljoin(self.base_url, endpoint), params=params)
            response.raise_for_status()
            
            data = response.json()
            market_data = []
            
            for kline in data:
                market_data.append(MarketData(
                    symbol=symbol,
                    timestamp=datetime.fromtimestamp(kline[0] / 1000),
                    open=float(kline[1]),
                    high=float(kline[2]),
                    low=float(kline[3]),
                    close=float(kline[4]),
                    volume=float(kline[5]),
                    source="binance"
                ))
            
            return market_data
            
        except Exception as e:
            print(f"Error fetching historical data: {e}")
            return []
    
    async def subscribe_realtime(
        self, 
        symbols: List[str], 
        callback: Callable[[MarketTick], None]
    ) -> None:
        """Subscribe to real-time data via WebSocket."""
        
        def on_message(ws, message):
            try:
                data = json.loads(message)
                if 's' in data:  # Symbol field exists
                    tick = MarketTick(
                        symbol=data['s'],
                        timestamp=datetime.fromtimestamp(data['E'] / 1000),
                        price=float(data['c']),
                        volume=float(data['v'])
                    )
                    callback(tick)
            except Exception as e:
                print(f"Error processing tick data: {e}")
        
        def on_error(ws, error):
            print(f"WebSocket error: {error}")
        
        def on_close(ws, close_status_code, close_msg):
            print("WebSocket connection closed")
        
        # Create stream names for symbols
        streams = [f"{symbol.lower()}@ticker" for symbol in symbols]
        stream_url = f"{self.ws_url}{'/'.join(streams)}"
        
        # Start WebSocket connection in separate thread
        def run_websocket():
            ws = websocket.WebSocketApp(
                stream_url,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close
            )
            ws.run_forever()
        
        ws_thread = threading.Thread(target=run_websocket, daemon=True)
        ws_thread.start()
    
    async def get_current_price(self, symbol: str) -> float:
        """Get current price from Binance API."""
        try:
            endpoint = "/api/v3/ticker/price"
            params = {'symbol': symbol.upper()}
            
            response = requests.get(urljoin(self.base_url, endpoint), params=params)
            response.raise_for_status()
            
            data = response.json()
            return float(data['price'])
            
        except Exception as e:
            print(f"Error fetching current price: {e}")
            return 0.0

class AlphaVantageDataProvider(MarketDataProvider):
    """Alpha Vantage market data provider for stocks."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
    
    async def get_historical_data(
        self, 
        symbol: str, 
        interval: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> List[MarketData]:
        """Get historical stock data from Alpha Vantage."""
        try:
            # Map intervals to Alpha Vantage format
            if interval in ['1m', '5m', '15m', '30m', '60m']:
                function = "TIME_SERIES_INTRADAY"
                params = {
                    'function': function,
                    'symbol': symbol,
                    'interval': interval,
                    'apikey': self.api_key,
                    'outputsize': 'full'
                }
            else:
                function = "TIME_SERIES_DAILY"
                params = {
                    'function': function,
                    'symbol': symbol,
                    'apikey': self.api_key,
                    'outputsize': 'full'
                }
            
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # Extract time series data
            if 'Time Series' in str(data):
                time_series_key = [k for k in data.keys() if 'Time Series' in k][0]
                time_series = data[time_series_key]
                
                market_data = []
                for timestamp_str, values in time_series.items():
                    timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S' if ' ' in timestamp_str else '%Y-%m-%d')
                    
                    # Filter by date range
                    if start_date <= timestamp <= end_date:
                        market_data.append(MarketData(
                            symbol=symbol,
                            timestamp=timestamp,
                            open=float(values['1. open']),
                            high=float(values['2. high']),
                            low=float(values['3. low']),
                            close=float(values['4. close']),
                            volume=float(values['5. volume']),
                            source="alphavantage"
                        ))
                
                return sorted(market_data, key=lambda x: x.timestamp)
            
            return []
            
        except Exception as e:
            print(f"Error fetching Alpha Vantage data: {e}")
            return []
    
    async def subscribe_realtime(
        self, 
        symbols: List[str], 
        callback: Callable[[MarketTick], None]
    ) -> None:
        """Alpha Vantage doesn't support real-time WebSocket, simulate with polling."""
        
        async def poll_prices():
            while True:
                for symbol in symbols:
                    try:
                        price = await self.get_current_price(symbol)
                        if price > 0:
                            tick = MarketTick(
                                symbol=symbol,
                                timestamp=datetime.now(),
                                price=price,
                                volume=0
                            )
                            callback(tick)
                    except Exception as e:
                        print(f"Error polling price for {symbol}: {e}")
                
                await asyncio.sleep(60)  # Poll every minute
        
        # Start polling task
        asyncio.create_task(poll_prices())
    
    async def get_current_price(self, symbol: str) -> float:
        """Get current stock price."""
        try:
            params = {
                'function': 'GLOBAL_QUOTE',
                'symbol': symbol,
                'apikey': self.api_key
            }
            
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if 'Global Quote' in data:
                return float(data['Global Quote']['05. price'])
            
            return 0.0
            
        except Exception as e:
            print(f"Error fetching current price: {e}")
            return 0.0

class MarketDataManager:
    """Central manager for market data from multiple sources."""
    
    def __init__(self):
        self.providers: Dict[str, MarketDataProvider] = {}
        self.subscriptions: Dict[str, List[Callable]] = {}
        self.cache: Dict[str, MarketData] = {}
        self.cache_ttl = timedelta(minutes=1)
        
    def add_provider(self, name: str, provider: MarketDataProvider):
        """Add a market data provider."""
        self.providers[name] = provider
    
    def get_provider(self, name: str) -> Optional[MarketDataProvider]:
        """Get a specific provider."""
        return self.providers.get(name)
    
    async def get_historical_data(
        self, 
        symbol: str, 
        interval: str, 
        days: int = 30,
        provider: str = "binance"
    ) -> List[MarketData]:
        """Get historical data from specified provider."""
        if provider not in self.providers:
            raise ValueError(f"Provider {provider} not found")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        return await self.providers[provider].get_historical_data(
            symbol, interval, start_date, end_date
        )
    
    async def get_current_price(self, symbol: str, provider: str = "binance") -> float:
        """Get current price with caching."""
        cache_key = f"{provider}_{symbol}"
        
        # Check cache
        if cache_key in self.cache:
            cached_data = self.cache[cache_key]
            if datetime.now() - cached_data.timestamp < self.cache_ttl:
                return cached_data.close
        
        # Fetch new data
        if provider not in self.providers:
            raise ValueError(f"Provider {provider} not found")
        
        price = await self.providers[provider].get_current_price(symbol)
        
        # Update cache
        self.cache[cache_key] = MarketData(
            symbol=symbol,
            timestamp=datetime.now(),
            open=price, high=price, low=price, close=price,
            volume=0, source=provider
        )
        
        return price
    
    def subscribe_to_symbol(
        self, 
        symbol: str, 
        callback: Callable[[MarketTick], None],
        provider: str = "binance"
    ):
        """Subscribe to real-time data for a symbol."""
        if provider not in self.providers:
            raise ValueError(f"Provider {provider} not found")
        
        subscription_key = f"{provider}_{symbol}"
        if subscription_key not in self.subscriptions:
            self.subscriptions[subscription_key] = []
        
        self.subscriptions[subscription_key].append(callback)
        
        # Start subscription if first callback for this symbol
        if len(self.subscriptions[subscription_key]) == 1:
            asyncio.create_task(
                self.providers[provider].subscribe_realtime([symbol], self._handle_tick)
            )
    
    def _handle_tick(self, tick: MarketTick):
        """Handle incoming tick data and distribute to callbacks."""
        for provider_name in self.providers:
            subscription_key = f"{provider_name}_{tick.symbol}"
            if subscription_key in self.subscriptions:
                for callback in self.subscriptions[subscription_key]:
                    try:
                        callback(tick)
                    except Exception as e:
                        print(f"Error in tick callback: {e}")

# Global market data manager instance
market_data_manager = MarketDataManager()

# Initialize with available providers
def initialize_market_data():
    """Initialize market data providers."""
    
    # Add Binance provider (for crypto)
    binance_provider = BinanceDataProvider()
    market_data_manager.add_provider("binance", binance_provider)
    
    # Add Alpha Vantage provider if API key is available
    import os
    alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY')
    if alpha_vantage_key:
        alpha_provider = AlphaVantageDataProvider(alpha_vantage_key)
        market_data_manager.add_provider("alphavantage", alpha_provider)

# Auto-initialize when module is imported
initialize_market_data()
