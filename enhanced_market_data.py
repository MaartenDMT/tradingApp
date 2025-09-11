"""
Enhanced Market Data Integration Module

This module provides enhanced integration with multiple market data sources
for live trading, backtesting, and advanced analytics.
"""

import asyncio
import json
import websocket
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Callable, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import threading
import time
import requests
from urllib.parse import urljoin
import yfinance as yf
import ccxt
import logging
from concurrent.futures import ThreadPoolExecutor
import warnings

from util.config_manager import get_config
from util.cache import get_global_cache

# Set up logging
logger = logging.getLogger(__name__)

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
    additional_data: Optional[Dict[str, Any]] = None

@dataclass
class MarketTick:
    """Real-time market tick data."""
    symbol: str
    timestamp: datetime
    price: float
    volume: float
    bid: Optional[float] = None
    ask: Optional[float] = None
    bid_volume: Optional[float] = None
    ask_volume: Optional[float] = None

@dataclass
class MarketDepth:
    """Market depth/order book data."""
    symbol: str
    timestamp: datetime
    bids: List[tuple]  # [(price, volume), ...]
    asks: List[tuple]  # [(price, volume), ...]
    last_update_id: Optional[int] = None

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
    
    @abstractmethod
    async def get_market_depth(self, symbol: str) -> MarketDepth:
        """Get market depth/order book data."""
        pass

class EnhancedBinanceDataProvider(MarketDataProvider):
    """Enhanced Binance market data provider implementation."""
    
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = "https://api.binance.com"
        self.ws_url = "wss://stream.binance.com:9443/ws/"
        self._ws_connections = {}
        self._callbacks = {}
        self._cache = None
        self._executor = ThreadPoolExecutor(max_workers=10)
        
        # Initialize CCXT exchange
        try:
            self.exchange = ccxt.binance({
                'apiKey': api_key,
                'secret': api_secret,
                'enableRateLimit': True,
            })
        except Exception as e:
            logger.warning(f"Failed to initialize CCXT Binance exchange: {e}")
            self.exchange = None
    
    async def get_historical_data(
        self, 
        symbol: str, 
        interval: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> List[MarketData]:
        """Get historical data from Binance API."""
        try:
            # Check cache first
            if self._cache is None:
                self._cache = await get_global_cache()
                
            cache_key = f"binance_historical_{symbol}_{interval}_{start_date.isoformat()}_{end_date.isoformat()}"
            cached_data = await self._cache.get(cache_key)
            if cached_data:
                logger.info(f"Retrieved historical data from cache for {symbol}")
                return [MarketData(**item) for item in cached_data]
            
            # Convert interval to Binance format
            interval_map = {
                '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
                '1h': '1h', '4h': '4h', '1d': '1d', '1w': '1w'
            }
            binance_interval = interval_map.get(interval, '1h')
            
            # Prepare API request
            endpoint = "/api/v3/klines"
            params = {
                'symbol': symbol.replace('/', ''),
                'interval': binance_interval,
                'startTime': int(start_date.timestamp() * 1000),
                'endTime': int(end_date.timestamp() * 1000),
                'limit': 1000  # Maximum limit per request
            }
            
            # Make API request
            url = urljoin(self.base_url, endpoint)
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            # Parse response
            data = response.json()
            market_data = []
            
            for item in data:
                market_data.append(MarketData(
                    symbol=symbol,
                    timestamp=datetime.fromtimestamp(item[0] / 1000),
                    open=float(item[1]),
                    high=float(item[2]),
                    low=float(item[3]),
                    close=float(item[4]),
                    volume=float(item[5]),
                    source="binance"
                ))
            
            # Cache the data
            if self._cache:
                cache_data = [asdict(item) for item in market_data]
                await self._cache.set(cache_key, cache_data, ttl=3600)  # 1 hour TTL
            
            logger.info(f"Retrieved {len(market_data)} historical data points for {symbol}")
            return market_data
            
        except Exception as e:
            logger.error(f"Failed to get historical data from Binance: {e}")
            return []
    
    async def subscribe_realtime(
        self, 
        symbols: List[str], 
        callback: Callable[[MarketTick], None]
    ) -> None:
        """Subscribe to real-time market data via WebSocket."""
        try:
            def ws_message_handler(ws, message):
                try:
                    data = json.loads(message)
                    if 'data' in data:
                        tick_data = data['data']
                        tick = MarketTick(
                            symbol=tick_data['s'],
                            timestamp=datetime.fromtimestamp(tick_data['E'] / 1000),
                            price=float(tick_data['c']),
                            volume=float(tick_data['v']),
                            bid=float(tick_data.get('b', 0)),
                            ask=float(tick_data.get('a', 0)),
                            bid_volume=float(tick_data.get('B', 0)),
                            ask_volume=float(tick_data.get('A', 0))
                        )
                        callback(tick)
                except Exception as e:
                    logger.error(f"Error processing WebSocket message: {e}")
            
            def ws_error_handler(ws, error):
                logger.error(f"WebSocket error: {error}")
            
            def ws_close_handler(ws, close_status_code, close_msg):
                logger.info(f"WebSocket closed: {close_status_code} - {close_msg}")
            
            # Create WebSocket connections for each symbol
            for symbol in symbols:
                stream_name = f"{symbol.lower()}@ticker"
                ws_url = f"{self.ws_url}{stream_name}"
                
                ws = websocket.WebSocketApp(
                    ws_url,
                    on_message=ws_message_handler,
                    on_error=ws_error_handler,
                    on_close=ws_close_handler
                )
                
                # Store connection and callback
                self._ws_connections[symbol] = ws
                self._callbacks[symbol] = callback
                
                # Run WebSocket in a separate thread
                def run_ws(websocket_app):
                    websocket_app.run_forever()
                
                thread = threading.Thread(target=run_ws, args=(ws,))
                thread.daemon = True
                thread.start()
                
                logger.info(f"Subscribed to real-time data for {symbol}")
                
        except Exception as e:
            logger.error(f"Failed to subscribe to real-time data: {e}")
    
    async def get_current_price(self, symbol: str) -> float:
        """Get current market price."""
        try:
            if self.exchange:
                ticker = self.exchange.fetch_ticker(symbol)
                return float(ticker['last'])
            else:
                # Fallback to REST API
                endpoint = "/api/v3/ticker/price"
                params = {'symbol': symbol.replace('/', '')}
                url = urljoin(self.base_url, endpoint)
                response = requests.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                return float(data['price'])
        except Exception as e:
            logger.error(f"Failed to get current price for {symbol}: {e}")
            return 0.0
    
    async def get_market_depth(self, symbol: str) -> MarketDepth:
        """Get market depth/order book data."""
        try:
            if self.exchange:
                orderbook = self.exchange.fetch_order_book(symbol)
                return MarketDepth(
                    symbol=symbol,
                    timestamp=datetime.now(),
                    bids=[(float(price), float(volume)) for price, volume in orderbook['bids'][:20]],
                    asks=[(float(price), float(volume)) for price, volume in orderbook['asks'][:20]],
                    last_update_id=orderbook.get('nonce')
                )
            else:
                # Fallback to REST API
                endpoint = "/api/v3/depth"
                params = {'symbol': symbol.replace('/', ''), 'limit': 20}
                url = urljoin(self.base_url, endpoint)
                response = requests.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                
                return MarketDepth(
                    symbol=symbol,
                    timestamp=datetime.now(),
                    bids=[(float(price), float(volume)) for price, volume in data['bids']],
                    asks=[(float(price), float(volume)) for price, volume in data['asks']],
                    last_update_id=data.get('lastUpdateId')
                )
        except Exception as e:
            logger.error(f"Failed to get market depth for {symbol}: {e}")
            return MarketDepth(symbol=symbol, timestamp=datetime.now(), bids=[], asks=[])

class YahooFinanceDataProvider(MarketDataProvider):
    """Yahoo Finance market data provider implementation."""
    
    def __init__(self):
        self._cache = None
    
    async def get_historical_data(
        self, 
        symbol: str, 
        interval: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> List[MarketData]:
        """Get historical data from Yahoo Finance."""
        try:
            # Check cache first
            if self._cache is None:
                self._cache = await get_global_cache()
                
            cache_key = f"yahoo_historical_{symbol}_{interval}_{start_date.isoformat()}_{end_date.isoformat()}"
            cached_data = await self._cache.get(cache_key)
            if cached_data:
                logger.info(f"Retrieved historical data from cache for {symbol}")
                return [MarketData(**item) for item in cached_data]
            
            # Map intervals to Yahoo Finance format
            interval_map = {
                '1d': '1d', '1w': '1wk', '1m': '1mo', '3m': '3mo'
            }
            yahoo_interval = interval_map.get(interval, '1d')
            
            # Fetch data using yfinance
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                start=start_date,
                end=end_date,
                interval=yahoo_interval
            )
            
            # Convert to MarketData objects
            market_data = []
            for index, row in df.iterrows():
                market_data.append(MarketData(
                    symbol=symbol,
                    timestamp=index.to_pydatetime(),
                    open=float(row['Open']),
                    high=float(row['High']),
                    low=float(row['Low']),
                    close=float(row['Close']),
                    volume=float(row['Volume']),
                    source="yahoo_finance"
                ))
            
            # Cache the data
            if self._cache:
                cache_data = [asdict(item) for item in market_data]
                await self._cache.set(cache_key, cache_data, ttl=3600)  # 1 hour TTL
            
            logger.info(f"Retrieved {len(market_data)} historical data points for {symbol}")
            return market_data
            
        except Exception as e:
            logger.error(f"Failed to get historical data from Yahoo Finance: {e}")
            return []
    
    async def subscribe_realtime(
        self, 
        symbols: List[str], 
        callback: Callable[[MarketTick], None]
    ) -> None:
        """Subscribe to real-time market data (not available in Yahoo Finance)."""
        logger.warning("Real-time data subscription not available for Yahoo Finance")
        # Yahoo Finance doesn't support real-time data via API
        # This would require a paid service or web scraping
    
    async def get_current_price(self, symbol: str) -> float:
        """Get current market price."""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return float(info.get('currentPrice', info.get('regularMarketPrice', 0)))
        except Exception as e:
            logger.error(f"Failed to get current price for {symbol}: {e}")
            return 0.0
    
    async def get_market_depth(self, symbol: str) -> MarketDepth:
        """Get market depth/order book data (not available in Yahoo Finance)."""
        logger.warning("Market depth not available for Yahoo Finance")
        return MarketDepth(symbol=symbol, timestamp=datetime.now(), bids=[], asks=[])

class MarketDataManager:
    """Manager for multiple market data providers."""
    
    def __init__(self):
        self.providers: Dict[str, MarketDataProvider] = {}
        self.config = get_config()
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize market data providers based on configuration."""
        # Initialize Binance provider if API keys are available
        binance_key = self.config.get_api_key('BINANCE')
        binance_secret = self.config.get_api_key('BINANCE', 'secret')
        if binance_key and binance_secret:
            self.providers['binance'] = EnhancedBinanceDataProvider(binance_key, binance_secret)
            logger.info("Initialized Binance data provider")
        
        # Always initialize Yahoo Finance provider
        self.providers['yahoo'] = YahooFinanceDataProvider()
        logger.info("Initialized Yahoo Finance data provider")
    
    async def get_historical_data(
        self, 
        symbol: str, 
        interval: str, 
        start_date: datetime, 
        end_date: datetime,
        provider: str = 'binance'
    ) -> List[MarketData]:
        """Get historical data from a specific provider."""
        if provider not in self.providers:
            raise ValueError(f"Provider {provider} not available")
        
        return await self.providers[provider].get_historical_data(
            symbol, interval, start_date, end_date
        )
    
    async def get_multiple_provider_data(
        self, 
        symbol: str, 
        interval: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> Dict[str, List[MarketData]]:
        """Get historical data from all available providers."""
        results = {}
        for provider_name, provider in self.providers.items():
            try:
                data = await provider.get_historical_data(
                    symbol, interval, start_date, end_date
                )
                results[provider_name] = data
            except Exception as e:
                logger.error(f"Failed to get data from {provider_name}: {e}")
                results[provider_name] = []
        
        return results
    
    async def subscribe_realtime(
        self, 
        symbols: List[str], 
        callback: Callable[[MarketTick], None],
        provider: str = 'binance'
    ) -> None:
        """Subscribe to real-time data from a specific provider."""
        if provider not in self.providers:
            raise ValueError(f"Provider {provider} not available")
        
        await self.providers[provider].subscribe_realtime(symbols, callback)
    
    async def get_current_price(
        self, 
        symbol: str, 
        provider: str = 'binance'
    ) -> float:
        """Get current price from a specific provider."""
        if provider not in self.providers:
            raise ValueError(f"Provider {provider} not available")
        
        return await self.providers[provider].get_current_price(symbol)
    
    async def get_market_depth(
        self, 
        symbol: str, 
        provider: str = 'binance'
    ) -> MarketDepth:
        """Get market depth from a specific provider."""
        if provider not in self.providers:
            raise ValueError(f"Provider {provider} not available")
        
        return await self.providers[provider].get_market_depth(symbol)
    
    def get_available_providers(self) -> List[str]:
        """Get list of available providers."""
        return list(self.providers.keys())

# Data processing utilities
class MarketDataProcessor:
    """Utilities for processing market data."""
    
    @staticmethod
    def calculate_technical_indicators(data: List[MarketData]) -> pd.DataFrame:
        """Calculate technical indicators from market data."""
        if not data:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df_data = [
            {
                'timestamp': item.timestamp,
                'open': item.open,
                'high': item.high,
                'low': item.low,
                'close': item.close,
                'volume': item.volume
            }
            for item in data
        ]
        df = pd.DataFrame(df_data)
        df.set_index('timestamp', inplace=True)
        
        # Calculate basic indicators
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        return df
    
    @staticmethod
    def resample_data(data: List[MarketData], new_interval: str) -> List[MarketData]:
        """Resample market data to a different interval."""
        if not data:
            return []
        
        # Convert to DataFrame
        df_data = [
            {
                'timestamp': item.timestamp,
                'open': item.open,
                'high': item.high,
                'low': item.low,
                'close': item.close,
                'volume': item.volume,
                'symbol': item.symbol,
                'source': item.source
            }
            for item in data
        ]
        df = pd.DataFrame(df_data)
        df.set_index('timestamp', inplace=True)
        
        # Resample based on new interval
        resampled = df.resample(new_interval).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'symbol': 'first',
            'source': 'first'
        }).dropna()
        
        # Convert back to MarketData objects
        resampled_data = []
        for timestamp, row in resampled.iterrows():
            resampled_data.append(MarketData(
                symbol=row['symbol'],
                timestamp=timestamp.to_pydatetime(),
                open=float(row['open']),
                high=float(row['high']),
                low=float(row['low']),
                close=float(row['close']),
                volume=float(row['volume']),
                source=row['source']
            ))
        
        return resampled_data

# Global market data manager instance
market_data_manager = MarketDataManager()

# Example usage
def run_market_data_demo():
    """Run a demonstration of the market data integration."""
    print("Enhanced Market Data Integration Module")
    print("=" * 40)
    
    # Initialize market data manager
    manager = market_data_manager
    
    print(f"Available providers: {manager.get_available_providers()}")
    
    # Get current price (example)
    async def get_price_demo():
        try:
            price = await manager.get_current_price("BTC/USDT", "binance")
            print(f"Current BTC/USDT price: ${price:.2f}")
        except Exception as e:
            print(f"Error getting price: {e}")
    
    # Run async function
    try:
        asyncio.run(get_price_demo())
    except Exception as e:
        print(f"Error in async demo: {e}")
    
    print("Market data integration module initialized successfully")

if __name__ == "__main__":
    run_market_data_demo()