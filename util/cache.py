"""
Enhanced Multi-Layer Cache System with Advanced Features

This module provides comprehensive caching functionality with:
- Redis integration for distributed caching
- Memory cache with LRU eviction
- TTL (Time To Live) support
- Cache invalidation and warming
- Performance metrics and monitoring
- Serialization optimization
- Cache hierarchies and fallbacks
"""

import asyncio
import hashlib
import logging
import pickle
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

logger = logging.getLogger(__name__)


@dataclass
class CacheStats:
    """Cache performance statistics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_size: int = 0
    avg_access_time: float = 0.0

    @property
    def hit_ratio(self) -> float:
        """Calculate cache hit ratio."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    value: Any
    timestamp: float
    ttl: Optional[float] = None
    access_count: int = 0
    last_access: float = None

    def __post_init__(self):
        if self.last_access is None:
            self.last_access = self.timestamp

    @property
    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        if self.ttl is None:
            return False
        return time.time() - self.timestamp > self.ttl

    def touch(self):
        """Update access metadata."""
        self.access_count += 1
        self.last_access = time.time()


class LRUCache:
    """Memory-based LRU cache with TTL support."""

    def __init__(self, max_size: int = 1000, default_ttl: Optional[float] = None):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._stats = CacheStats()
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        async with self._lock:
            start_time = time.time()

            if key in self._cache:
                entry = self._cache[key]

                if entry.is_expired:
                    del self._cache[key]
                    self._stats.misses += 1
                    return None

                # Move to end (most recently used)
                self._cache.move_to_end(key)
                entry.touch()

                self._stats.hits += 1
                self._update_access_time(time.time() - start_time)
                return entry.value

            self._stats.misses += 1
            self._update_access_time(time.time() - start_time)
            return None

    async def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Set value in cache."""
        async with self._lock:
            # Use default TTL if none specified
            if ttl is None:
                ttl = self.default_ttl

            # Remove expired entries
            await self._cleanup_expired()

            # Evict if at capacity
            if len(self._cache) >= self.max_size and key not in self._cache:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                self._stats.evictions += 1

            # Add/update entry
            entry = CacheEntry(value=value, timestamp=time.time(), ttl=ttl)
            self._cache[key] = entry
            self._cache.move_to_end(key)

            self._stats.total_size = len(self._cache)

    async def delete(self, key: str) -> bool:
        """Delete entry from cache."""
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._stats.total_size = len(self._cache)
                return True
            return False

    async def clear(self) -> None:
        """Clear all cache entries."""
        async with self._lock:
            self._cache.clear()
            self._stats.total_size = 0

    async def _cleanup_expired(self) -> None:
        """Remove expired entries."""
        current_time = time.time()
        expired_keys = [
            key for key, entry in self._cache.items()
            if entry.ttl and current_time - entry.timestamp > entry.ttl
        ]

        for key in expired_keys:
            del self._cache[key]

    def _update_access_time(self, access_time: float) -> None:
        """Update average access time."""
        total_operations = self._stats.hits + self._stats.misses
        if total_operations == 1:
            self._stats.avg_access_time = access_time
        else:
            # Running average
            self._stats.avg_access_time = (
                (self._stats.avg_access_time * (total_operations - 1) + access_time) / total_operations
            )

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        return self._stats


class RedisCache:
    """Redis-based distributed cache."""

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        db: int = 0,
        prefix: str = "trading_app:",
        default_ttl: Optional[int] = None,
        max_connections: int = 20
    ):
        self.redis_url = redis_url
        self.db = db
        self.prefix = prefix
        self.default_ttl = default_ttl
        self.max_connections = max_connections
        self._redis: Optional[redis.Redis] = None
        self._stats = CacheStats()

        if not REDIS_AVAILABLE:
            logger.warning("Redis not available. RedisCache will be disabled.")

    async def connect(self) -> None:
        """Initialize Redis connection."""
        if not REDIS_AVAILABLE:
            return

        try:
            self._redis = redis.from_url(
                self.redis_url,
                db=self.db,
                max_connections=self.max_connections,
                decode_responses=False
            )
            await self._redis.ping()
            logger.info("Redis cache connected successfully")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self._redis = None

    async def disconnect(self) -> None:
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()
            self._redis = None

    def _make_key(self, key: str) -> str:
        """Create prefixed cache key."""
        return f"{self.prefix}{key}"

    async def get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache."""
        if not self._redis:
            self._stats.misses += 1
            return None

        try:
            start_time = time.time()
            prefixed_key = self._make_key(key)
            data = await self._redis.get(prefixed_key)

            if data:
                self._stats.hits += 1
                self._update_access_time(time.time() - start_time)
                return pickle.loads(data)

            self._stats.misses += 1
            return None

        except Exception as e:
            logger.error(f"Redis get error for key {key}: {e}")
            self._stats.misses += 1
            return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in Redis cache."""
        if not self._redis:
            return

        try:
            prefixed_key = self._make_key(key)
            data = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)

            if ttl is None:
                ttl = self.default_ttl

            if ttl:
                await self._redis.setex(prefixed_key, ttl, data)
            else:
                await self._redis.set(prefixed_key, data)

        except Exception as e:
            logger.error(f"Redis set error for key {key}: {e}")

    async def delete(self, key: str) -> bool:
        """Delete entry from Redis cache."""
        if not self._redis:
            return False

        try:
            prefixed_key = self._make_key(key)
            result = await self._redis.delete(prefixed_key)
            return result > 0
        except Exception as e:
            logger.error(f"Redis delete error for key {key}: {e}")
            return False

    async def clear(self) -> None:
        """Clear all cache entries with prefix."""
        if not self._redis:
            return

        try:
            pattern = f"{self.prefix}*"
            keys = await self._redis.keys(pattern)
            if keys:
                await self._redis.delete(*keys)
        except Exception as e:
            logger.error(f"Redis clear error: {e}")

    def _update_access_time(self, access_time: float) -> None:
        """Update average access time."""
        total_operations = self._stats.hits + self._stats.misses
        if total_operations == 1:
            self._stats.avg_access_time = access_time
        else:
            self._stats.avg_access_time = (
                (self._stats.avg_access_time * (total_operations - 1) + access_time) / total_operations
            )

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        return self._stats


class HybridCache:
    """Multi-layer cache with memory and Redis."""

    def __init__(
        self,
        memory_cache_size: int = 1000,
        memory_ttl: Optional[float] = 300,  # 5 minutes
        redis_url: str = "redis://localhost:6379",
        redis_ttl: Optional[int] = 3600,    # 1 hour
        enable_redis: bool = True
    ):
        self.memory_cache = LRUCache(max_size=memory_cache_size, default_ttl=memory_ttl)
        self.redis_cache = RedisCache(redis_url=redis_url, default_ttl=redis_ttl) if enable_redis else None
        self._stats = CacheStats()

    async def connect(self) -> None:
        """Initialize connections."""
        if self.redis_cache:
            await self.redis_cache.connect()

    async def disconnect(self) -> None:
        """Close connections."""
        if self.redis_cache:
            await self.redis_cache.disconnect()

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache (memory first, then Redis)."""
        start_time = time.time()

        # Try memory cache first
        value = await self.memory_cache.get(key)
        if value is not None:
            self._stats.hits += 1
            self._update_access_time(time.time() - start_time)
            return value

        # Try Redis cache
        if self.redis_cache:
            value = await self.redis_cache.get(key)
            if value is not None:
                # Promote to memory cache
                await self.memory_cache.set(key, value)
                self._stats.hits += 1
                self._update_access_time(time.time() - start_time)
                return value

        self._stats.misses += 1
        self._update_access_time(time.time() - start_time)
        return None

    async def set(self, key: str, value: Any, memory_ttl: Optional[float] = None, redis_ttl: Optional[int] = None) -> None:
        """Set value in both caches."""
        # Set in memory cache
        await self.memory_cache.set(key, value, ttl=memory_ttl)

        # Set in Redis cache
        if self.redis_cache:
            await self.redis_cache.set(key, value, ttl=redis_ttl)

    async def delete(self, key: str) -> bool:
        """Delete from both caches."""
        memory_deleted = await self.memory_cache.delete(key)
        redis_deleted = await self.redis_cache.delete(key) if self.redis_cache else False
        return memory_deleted or redis_deleted

    async def clear(self) -> None:
        """Clear both caches."""
        await self.memory_cache.clear()
        if self.redis_cache:
            await self.redis_cache.clear()

    def _update_access_time(self, access_time: float) -> None:
        """Update average access time."""
        total_operations = self._stats.hits + self._stats.misses
        if total_operations == 1:
            self._stats.avg_access_time = access_time
        else:
            self._stats.avg_access_time = (
                (self._stats.avg_access_time * (total_operations - 1) + access_time) / total_operations
            )

    def get_stats(self) -> Dict[str, CacheStats]:
        """Get combined cache statistics."""
        return {
            "hybrid": self._stats,
            "memory": self.memory_cache.get_stats(),
            "redis": self.redis_cache.get_stats() if self.redis_cache else CacheStats()
        }


# Cache decorators
def cached(
    cache_instance: Union[LRUCache, RedisCache, HybridCache],
    ttl: Optional[Union[float, int]] = None,
    key_func: Optional[Callable] = None
):
    """Decorator for caching function results."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                key_parts = [func.__name__]
                key_parts.extend(str(arg) for arg in args)
                key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                cache_key = hashlib.md5(":".join(key_parts).encode()).hexdigest()

            # Try to get from cache
            result = await cache_instance.get(cache_key)
            if result is not None:
                return result

            # Call function and cache result
            result = await func(*args, **kwargs)

            # Set cache with appropriate TTL
            if isinstance(cache_instance, HybridCache):
                await cache_instance.set(cache_key, result, memory_ttl=ttl, redis_ttl=ttl)
            else:
                await cache_instance.set(cache_key, result, ttl=ttl)

            return result

        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            # For sync functions, we need to run in event loop
            return asyncio.run(async_wrapper(*args, **kwargs))

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator


def cache_key_for_ticker_data(symbol: str, timeframe: str, limit: int = 100) -> str:
    """Generate cache key for ticker data."""
    return f"ticker:{symbol}:{timeframe}:{limit}"


def cache_key_for_ml_prediction(model_name: str, features_hash: str) -> str:
    """Generate cache key for ML predictions."""
    return f"ml_pred:{model_name}:{features_hash}"


def cache_key_for_portfolio_data(user_id: str, timestamp: int) -> str:
    """Generate cache key for portfolio data."""
    return f"portfolio:{user_id}:{timestamp}"


# Global cache instance
_global_cache: Optional[HybridCache] = None


async def get_global_cache() -> HybridCache:
    """Get or create global cache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = HybridCache()
        await _global_cache.connect()
    return _global_cache


async def shutdown_global_cache() -> None:
    """Shutdown global cache instance."""
    global _global_cache
    if _global_cache:
        await _global_cache.disconnect()
        _global_cache = None


@asynccontextmanager
async def cache_context(cache_config: Optional[Dict[str, Any]] = None):
    """Context manager for cache lifecycle."""
    config = cache_config or {}
    cache = HybridCache(**config)

    try:
        await cache.connect()
        yield cache
    finally:
        await cache.disconnect()


# Performance monitoring
class CacheMonitor:
    """Monitor cache performance and health."""

    def __init__(self, cache_instance: Union[LRUCache, RedisCache, HybridCache]):
        self.cache = cache_instance
        self._metrics_history: List[Tuple[float, Dict[str, Any]]] = []

    async def collect_metrics(self) -> Dict[str, Any]:
        """Collect current cache metrics."""
        timestamp = time.time()

        if isinstance(self.cache, HybridCache):
            stats = self.cache.get_stats()
            metrics = {
                "timestamp": timestamp,
                "hybrid_hit_ratio": stats["hybrid"].hit_ratio,
                "memory_hit_ratio": stats["memory"].hit_ratio,
                "redis_hit_ratio": stats["redis"].hit_ratio,
                "memory_size": stats["memory"].total_size,
                "total_hits": stats["hybrid"].hits,
                "total_misses": stats["hybrid"].misses,
                "avg_access_time": stats["hybrid"].avg_access_time
            }
        else:
            stats = self.cache.get_stats()
            metrics = {
                "timestamp": timestamp,
                "hit_ratio": stats.hit_ratio,
                "total_hits": stats.hits,
                "total_misses": stats.misses,
                "cache_size": stats.total_size,
                "avg_access_time": stats.avg_access_time
            }

        self._metrics_history.append((timestamp, metrics))

        # Keep only last 1000 entries
        if len(self._metrics_history) > 1000:
            self._metrics_history = self._metrics_history[-1000:]

        return metrics

    def get_metrics_history(self, hours: int = 1) -> List[Tuple[float, Dict[str, Any]]]:
        """Get metrics history for specified hours."""
        cutoff_time = time.time() - (hours * 3600)
        return [(ts, metrics) for ts, metrics in self._metrics_history if ts >= cutoff_time]

    async def health_check(self) -> Dict[str, Any]:
        """Perform cache health check."""
        try:
            test_key = f"health_check_{time.time()}"
            test_value = {"test": True, "timestamp": time.time()}

            # Test write
            await self.cache.set(test_key, test_value, ttl=60)

            # Test read
            result = await self.cache.get(test_key)

            # Test delete
            await self.cache.delete(test_key)

            is_healthy = result is not None and result.get("test") is True

            return {
                "healthy": is_healthy,
                "timestamp": time.time(),
                "test_passed": result is not None,
                "cache_type": type(self.cache).__name__
            }

        except Exception as e:
            logger.error(f"Cache health check failed: {e}")
            return {
                "healthy": False,
                "timestamp": time.time(),
                "error": str(e),
                "cache_type": type(self.cache).__name__
            }
