"""
Async CCXT Client with WebSocket Support and Connection Optimization

This module provides optimized async CCXT client functionality with:
- Connection pooling and reuse
- WebSocket streaming for real-time data
- Request batching and rate limiting
- Caching mechanisms
- Error handling and resilience
"""

import asyncio
import logging
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import ccxt.pro as ccxtpro
from aiohttp import ClientSession, ClientTimeout, TCPConnector


@dataclass
class ClientConfig:
    """Configuration for async CCXT client"""
    # Connection settings
    max_connections: int = 20
    connection_timeout: float = 30.0
    request_timeout: float = 10.0

    # Rate limiting
    rate_limit_requests: int = 100
    rate_limit_window: float = 60.0

    # Caching
    cache_ttl: int = 300  # 5 minutes
    max_cache_size: int = 1000

    # WebSocket settings
    ws_auto_reconnect: bool = True
    ws_heartbeat: float = 30.0
    ws_max_retries: int = 5

    # Circuit breaker
    failure_threshold: int = 5
    recovery_timeout: float = 60.0


@dataclass
class CacheEntry:
    """Cache entry with timestamp and TTL"""
    data: Any
    timestamp: float
    ttl: int

    def is_expired(self) -> bool:
        return time.time() - self.timestamp > self.ttl


class RateLimiter:
    """Token bucket rate limiter"""

    def __init__(self, requests: int, window: float):
        self.requests = requests
        self.window = window
        self.tokens = requests
        self.last_update = time.time()
        self._lock = asyncio.Lock()

    async def acquire(self) -> bool:
        async with self._lock:
            now = time.time()
            elapsed = now - self.last_update
            self.tokens = min(self.requests, self.tokens + elapsed * (self.requests / self.window))
            self.last_update = now

            if self.tokens >= 1:
                self.tokens -= 1
                return True
            return False

    async def wait_for_token(self):
        """Wait until a token is available"""
        while not await self.acquire():
            await asyncio.sleep(0.1)


class CircuitBreaker:
    """Circuit breaker for error resilience"""

    def __init__(self, failure_threshold: int, recovery_timeout: float):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "closed"  # closed, open, half-open
        self._lock = asyncio.Lock()

    async def call(self, func: Callable, *args, **kwargs):
        async with self._lock:
            if self.state == "open":
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = "half-open"
                else:
                    raise Exception("Circuit breaker is open")

            try:
                result = await func(*args, **kwargs)
                if self.state == "half-open":
                    self.state = "closed"
                    self.failure_count = 0
                return result
            except Exception as e:
                self.failure_count += 1
                self.last_failure_time = time.time()

                if self.failure_count >= self.failure_threshold:
                    self.state = "open"

                raise e


class AsyncCache:
    """LRU cache with TTL support"""

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache: Dict[str, CacheEntry] = {}
        self.access_order = deque()
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[Any]:
        async with self._lock:
            if key in self.cache:
                entry = self.cache[key]
                if not entry.is_expired():
                    # Move to end (most recently used)
                    self.access_order.remove(key)
                    self.access_order.append(key)
                    return entry.data
                else:
                    # Remove expired entry
                    del self.cache[key]
                    self.access_order.remove(key)
            return None

    async def set(self, key: str, value: Any, ttl: int = 300):
        async with self._lock:
            # Remove oldest if at capacity
            while len(self.cache) >= self.max_size:
                oldest_key = self.access_order.popleft()
                del self.cache[oldest_key]

            # Remove existing entry if present
            if key in self.cache:
                self.access_order.remove(key)

            # Add new entry
            self.cache[key] = CacheEntry(value, time.time(), ttl)
            self.access_order.append(key)

    async def clear(self):
        async with self._lock:
            self.cache.clear()
            self.access_order.clear()


class AsyncCCXTClient:
    """
    Optimized async CCXT client with advanced features
    """

    def __init__(self, config: Optional[ClientConfig] = None):
        self.config = config or ClientConfig()
        self.logger = logging.getLogger(__name__)

        # Core components
        self.session: Optional[ClientSession] = None
        self.exchanges: Dict[str, ccxtpro.Exchange] = {}
        self.rate_limiters: Dict[str, RateLimiter] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.cache = AsyncCache(self.config.max_cache_size)

        # WebSocket connections
        self.ws_connections: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.ws_callbacks: Dict[str, Dict[str, List[Callable]]] = defaultdict(lambda: defaultdict(list))

        # Monitoring
        self.metrics = {
            'requests_made': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'errors': 0,
            'ws_reconnections': 0
        }

        # Cleanup tracking
        self._cleanup_tasks: List[asyncio.Task] = []

    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def initialize(self):
        """Initialize the async client"""
        # Create aiohttp session with optimized settings
        timeout = ClientTimeout(
            total=self.config.connection_timeout,
            connect=self.config.request_timeout
        )

        connector = TCPConnector(
            limit=self.config.max_connections,
            limit_per_host=10,
            ttl_dns_cache=300,
            use_dns_cache=True,
            keepalive_timeout=60,
            enable_cleanup_closed=True
        )

        self.session = ClientSession(
            timeout=timeout,
            connector=connector,
            trust_env=True
        )

        self.logger.info("AsyncCCXTClient initialized with optimized settings")

    async def close(self):
        """Clean up resources"""
        # Cancel cleanup tasks
        for task in self._cleanup_tasks:
            task.cancel()

        # Close WebSocket connections
        for exchange_id, connections in self.ws_connections.items():
            for symbol, connection in connections.items():
                if hasattr(connection, 'close'):
                    await connection.close()

        # Close exchanges
        for exchange in self.exchanges.values():
            if hasattr(exchange, 'close'):
                await exchange.close()

        # Close aiohttp session
        if self.session:
            await self.session.close()

        self.logger.info("AsyncCCXTClient closed successfully")

    def get_exchange(self, exchange_id: str, config: Dict[str, Any] = None) -> ccxtpro.Exchange:
        """Get or create exchange instance"""
        if exchange_id not in self.exchanges:
            exchange_class = getattr(ccxtpro, exchange_id)
            exchange_config = config or {}

            # Add session to config for connection reuse
            if self.session:
                exchange_config['session'] = self.session

            self.exchanges[exchange_id] = exchange_class(exchange_config)

            # Initialize rate limiter
            self.rate_limiters[exchange_id] = RateLimiter(
                self.config.rate_limit_requests,
                self.config.rate_limit_window
            )

            # Initialize circuit breaker
            self.circuit_breakers[exchange_id] = CircuitBreaker(
                self.config.failure_threshold,
                self.config.recovery_timeout
            )

            self.logger.info(f"Created exchange instance: {exchange_id}")

        return self.exchanges[exchange_id]

    async def fetch_ticker(self, exchange_id: str, symbol: str, use_cache: bool = True) -> Dict[str, Any]:
        """Fetch ticker with caching and rate limiting"""
        cache_key = f"{exchange_id}:ticker:{symbol}"

        # Check cache first
        if use_cache:
            cached_data = await self.cache.get(cache_key)
            if cached_data:
                self.metrics['cache_hits'] += 1
                return cached_data
            self.metrics['cache_misses'] += 1

        # Rate limiting
        await self.rate_limiters[exchange_id].wait_for_token()

        # Circuit breaker protection
        async def _fetch():
            exchange = self.get_exchange(exchange_id)
            return await exchange.fetch_ticker(symbol)

        try:
            ticker = await self.circuit_breakers[exchange_id].call(_fetch)

            # Cache the result
            if use_cache:
                await self.cache.set(cache_key, ticker, self.config.cache_ttl)

            self.metrics['requests_made'] += 1
            return ticker

        except Exception as e:
            self.metrics['errors'] += 1
            self.logger.error(f"Error fetching ticker {symbol} from {exchange_id}: {e}")
            raise

    async def fetch_order_book(self, exchange_id: str, symbol: str, limit: int = 100) -> Dict[str, Any]:
        """Fetch order book with optimizations"""
        cache_key = f"{exchange_id}:orderbook:{symbol}:{limit}"

        # Check cache (shorter TTL for order books)
        cached_data = await self.cache.get(cache_key)
        if cached_data:
            self.metrics['cache_hits'] += 1
            return cached_data

        # Rate limiting
        await self.rate_limiters[exchange_id].wait_for_token()

        async def _fetch():
            exchange = self.get_exchange(exchange_id)
            return await exchange.fetch_order_book(symbol, limit)

        try:
            order_book = await self.circuit_breakers[exchange_id].call(_fetch)

            # Cache with shorter TTL for order books
            await self.cache.set(cache_key, order_book, 10)  # 10 seconds

            self.metrics['requests_made'] += 1
            return order_book

        except Exception as e:
            self.metrics['errors'] += 1
            self.logger.error(f"Error fetching order book {symbol} from {exchange_id}: {e}")
            raise

    async def watch_ticker(self, exchange_id: str, symbol: str, callback: Callable):
        """Watch ticker updates via WebSocket"""
        exchange = self.get_exchange(exchange_id)

        if not exchange.has.get('watchTicker'):
            raise ValueError(f"Exchange {exchange_id} does not support watchTicker")

        # Register callback
        self.ws_callbacks[exchange_id]['ticker'].append(callback)

        # Start watching if not already
        ws_key = f"ticker:{symbol}"
        if ws_key not in self.ws_connections[exchange_id]:
            task = asyncio.create_task(self._watch_ticker_loop(exchange_id, symbol))
            self.ws_connections[exchange_id][ws_key] = task
            self._cleanup_tasks.append(task)

    async def _watch_ticker_loop(self, exchange_id: str, symbol: str):
        """WebSocket ticker watching loop with reconnection"""
        exchange = self.get_exchange(exchange_id)
        retries = 0

        while retries < self.config.ws_max_retries:
            try:
                while True:
                    ticker = await exchange.watch_ticker(symbol)

                    # Notify all callbacks
                    for callback in self.ws_callbacks[exchange_id]['ticker']:
                        try:
                            await callback(ticker)
                        except Exception as e:
                            self.logger.error(f"Error in ticker callback: {e}")

            except Exception as e:
                retries += 1
                self.metrics['ws_reconnections'] += 1
                self.logger.warning(f"WebSocket ticker error (retry {retries}): {e}")

                if retries < self.config.ws_max_retries:
                    await asyncio.sleep(min(2 ** retries, 30))  # Exponential backoff
                else:
                    self.logger.error(f"Max retries reached for ticker {symbol}")
                    break

    async def batch_fetch_tickers(self, exchange_id: str, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """Batch fetch multiple tickers efficiently"""
        exchange = self.get_exchange(exchange_id)

        if exchange.has.get('fetchTickers'):
            # Use native batch fetch if available
            await self.rate_limiters[exchange_id].wait_for_token()

            async def _fetch():
                return await exchange.fetch_tickers(symbols)

            return await self.circuit_breakers[exchange_id].call(_fetch)
        else:
            # Fall back to concurrent individual requests
            tasks = []
            for symbol in symbols:
                task = asyncio.create_task(self.fetch_ticker(exchange_id, symbol))
                tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)

            tickers = {}
            for symbol, result in zip(symbols, results):
                if not isinstance(result, Exception):
                    tickers[symbol] = result
                else:
                    self.logger.error(f"Error fetching ticker {symbol}: {result}")

            return tickers

    async def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        cache_hit_rate = (
            self.metrics['cache_hits'] /
            (self.metrics['cache_hits'] + self.metrics['cache_misses'])
            if (self.metrics['cache_hits'] + self.metrics['cache_misses']) > 0
            else 0
        )

        return {
            **self.metrics,
            'cache_hit_rate': cache_hit_rate,
            'active_exchanges': len(self.exchanges),
            'active_ws_connections': sum(len(conns) for conns in self.ws_connections.values()),
            'cache_size': len(self.cache.cache)
        }

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all components"""
        health = {
            'status': 'healthy',
            'session': self.session is not None and not self.session.closed,
            'exchanges': {},
            'circuit_breakers': {}
        }

        # Check exchanges
        for exchange_id, exchange in self.exchanges.items():
            try:
                # Simple connectivity test
                if hasattr(exchange, 'fetch_status'):
                    await exchange.fetch_status()
                health['exchanges'][exchange_id] = 'healthy'
            except Exception as e:
                health['exchanges'][exchange_id] = f'unhealthy: {str(e)}'
                health['status'] = 'degraded'

        # Check circuit breakers
        for exchange_id, cb in self.circuit_breakers.items():
            health['circuit_breakers'][exchange_id] = cb.state
            if cb.state == 'open':
                health['status'] = 'degraded'

        return health


# Factory functions for easy usage
async def create_client(config: Optional[ClientConfig] = None) -> AsyncCCXTClient:
    """Create and initialize async CCXT client"""
    client = AsyncCCXTClient(config)
    await client.initialize()
    return client


@asynccontextmanager
async def ccxt_client(config: Optional[ClientConfig] = None):
    """Context manager for async CCXT client"""
    client = await create_client(config)
    try:
        yield client
    finally:
        await client.close()


# Example usage configuration
DEFAULT_CONFIG = ClientConfig(
    max_connections=50,
    connection_timeout=30.0,
    request_timeout=10.0,
    rate_limit_requests=100,
    rate_limit_window=60.0,
    cache_ttl=300,
    max_cache_size=2000,
    ws_auto_reconnect=True,
    ws_heartbeat=30.0,
    ws_max_retries=5,
    failure_threshold=5,
    recovery_timeout=60.0
)
