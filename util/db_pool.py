"""
Async Database Connection Pool with Advanced Features

This module provides optimized async PostgreSQL connection pool with:
- Connection health monitoring and auto-recovery
- Connection pooling with statistics
- Retry logic with exponential backoff
- Connection lifecycle management
- Performance monitoring and metrics
- Graceful shutdown and cleanup
"""

import asyncio
import logging
import time
import weakref
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

try:
    import asyncpg
    import asyncpg.pool
    HAS_ASYNCPG = True
except ImportError:
    HAS_ASYNCPG = False
    # Fallback to psycopg2 for sync operations
    import psycopg2
    import psycopg2.pool
    from psycopg2.extras import RealDictCursor


class ConnectionState(Enum):
    """Connection state enumeration"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    RECOVERING = "recovering"


@dataclass
class PoolConfig:
    """Database pool configuration"""
    # Connection settings
    host: str = "localhost"
    port: int = 5432
    database: str = "trading"
    user: str = "postgres"
    password: str = ""

    # Pool settings
    min_size: int = 5
    max_size: int = 20
    max_queries: int = 50000
    max_inactive_connection_lifetime: float = 300.0

    # Health check settings
    health_check_interval: float = 60.0
    health_check_timeout: float = 5.0
    max_health_check_failures: int = 3

    # Retry settings
    max_retries: int = 3
    initial_retry_delay: float = 1.0
    max_retry_delay: float = 30.0
    retry_exponential_base: float = 2.0

    # Connection timeout settings
    command_timeout: float = 10.0
    connection_timeout: float = 30.0


@dataclass
class PoolMetrics:
    """Database pool metrics"""
    total_connections: int = 0
    active_connections: int = 0
    idle_connections: int = 0
    failed_connections: int = 0

    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0

    average_query_time: float = 0.0
    max_query_time: float = 0.0
    min_query_time: float = float('inf')

    pool_created_at: float = 0.0
    last_health_check: float = 0.0
    health_check_failures: int = 0

    def update_query_time(self, duration: float):
        """Update query timing statistics"""
        self.max_query_time = max(self.max_query_time, duration)
        self.min_query_time = min(self.min_query_time, duration)

        # Calculate running average
        total_time = self.average_query_time * self.total_queries
        self.total_queries += 1
        self.average_query_time = (total_time + duration) / self.total_queries


class HealthChecker:
    """Database connection health checker"""

    def __init__(self, config: PoolConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.failure_count = 0
        self.last_check = 0.0
        self.state = ConnectionState.HEALTHY

    async def check_connection_health(self, connection) -> bool:
        """Check if a connection is healthy"""
        try:
            if HAS_ASYNCPG:
                await asyncio.wait_for(
                    connection.fetchval("SELECT 1"),
                    timeout=self.config.health_check_timeout
                )
            else:
                # Sync health check for psycopg2
                cursor = connection.cursor()
                cursor.execute("SELECT 1")
                cursor.fetchone()
                cursor.close()
            return True
        except Exception as e:
            self.logger.warning(f"Health check failed: {e}")
            return False

    async def check_pool_health(self, pool) -> ConnectionState:
        """Check overall pool health"""
        try:
            if HAS_ASYNCPG:
                async with pool.acquire() as connection:
                    healthy = await self.check_connection_health(connection)
            else:
                with pool.getconn() as connection:
                    healthy = await self.check_connection_health(connection)
                    pool.putconn(connection)

            if healthy:
                self.failure_count = 0
                self.state = ConnectionState.HEALTHY
            else:
                self.failure_count += 1
                if self.failure_count >= self.config.max_health_check_failures:
                    self.state = ConnectionState.FAILED
                else:
                    self.state = ConnectionState.DEGRADED

        except Exception as e:
            self.failure_count += 1
            self.logger.error(f"Pool health check error: {e}")
            if self.failure_count >= self.config.max_health_check_failures:
                self.state = ConnectionState.FAILED
            else:
                self.state = ConnectionState.DEGRADED

        self.last_check = time.time()
        return self.state


class RetryHandler:
    """Exponential backoff retry handler"""

    def __init__(self, config: PoolConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger

    async def retry_with_backoff(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with exponential backoff retry"""
        last_exception = None

        for attempt in range(self.config.max_retries + 1):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt == self.config.max_retries:
                    break

                delay = min(
                    self.config.initial_retry_delay * (self.config.retry_exponential_base ** attempt),
                    self.config.max_retry_delay
                )

                self.logger.warning(f"Attempt {attempt + 1} failed, retrying in {delay:.2f}s: {e}")
                await asyncio.sleep(delay)

        raise last_exception


class AsyncDatabasePool:
    """
    Advanced async database connection pool with monitoring and resilience
    """

    def __init__(self, config: Optional[PoolConfig] = None, logger: Optional[logging.Logger] = None):
        self.config = config or PoolConfig()
        self.logger = logger or logging.getLogger(__name__)

        # Pool instances
        self.pool: Optional[Union[asyncpg.Pool, psycopg2.pool.ThreadedConnectionPool]] = None
        self.is_async = HAS_ASYNCPG

        # Components
        self.health_checker = HealthChecker(self.config, self.logger)
        self.retry_handler = RetryHandler(self.config, self.logger)
        self.metrics = PoolMetrics()

        # Monitoring
        self._health_check_task: Optional[asyncio.Task] = None
        self._cleanup_tasks: List[asyncio.Task] = []
        self._query_times: List[float] = []

        # Connection tracking
        self._active_connections: weakref.WeakSet = weakref.WeakSet()

        # State
        self._closed = False
        self._lock = asyncio.Lock()

    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def initialize(self):
        """Initialize the database pool"""
        try:
            if self.is_async:
                await self._initialize_async_pool()
            else:
                await self._initialize_sync_pool()

            # Start health monitoring
            await self._start_health_monitoring()

            self.metrics.pool_created_at = time.time()
            self.logger.info(f"Database pool initialized with {self.config.min_size}-{self.config.max_size} connections")

        except Exception as e:
            self.logger.error(f"Failed to initialize database pool: {e}")
            raise

    async def _initialize_async_pool(self):
        """Initialize asyncpg connection pool"""
        self.pool = await asyncpg.create_pool(
            host=self.config.host,
            port=self.config.port,
            database=self.config.database,
            user=self.config.user,
            password=self.config.password,
            min_size=self.config.min_size,
            max_size=self.config.max_size,
            max_queries=self.config.max_queries,
            max_inactive_connection_lifetime=self.config.max_inactive_connection_lifetime,
            command_timeout=self.config.command_timeout,
            connection_class=None  # Use default
        )

    async def _initialize_sync_pool(self):
        """Initialize psycopg2 connection pool (sync)"""
        dsn = f"host={self.config.host} port={self.config.port} dbname={self.config.database} user={self.config.user} password={self.config.password}"

        self.pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=self.config.min_size,
            maxconn=self.config.max_size,
            dsn=dsn,
            cursor_factory=RealDictCursor
        )

    async def _start_health_monitoring(self):
        """Start background health monitoring"""
        async def health_monitor():
            while not self._closed:
                try:
                    state = await self.health_checker.check_pool_health(self.pool)
                    self.metrics.last_health_check = time.time()
                    self.metrics.health_check_failures = self.health_checker.failure_count

                    if state == ConnectionState.FAILED:
                        self.logger.error("Database pool is in failed state, attempting recovery")
                        await self._attempt_recovery()

                    await asyncio.sleep(self.config.health_check_interval)

                except Exception as e:
                    self.logger.error(f"Health monitoring error: {e}")
                    await asyncio.sleep(self.config.health_check_interval)

        self._health_check_task = asyncio.create_task(health_monitor())
        self._cleanup_tasks.append(self._health_check_task)

    async def _attempt_recovery(self):
        """Attempt to recover failed pool"""
        try:
            self.logger.info("Attempting database pool recovery")
            await self.close()
            await asyncio.sleep(5)  # Wait before retry
            await self.initialize()
            self.logger.info("Database pool recovery successful")
        except Exception as e:
            self.logger.error(f"Pool recovery failed: {e}")

    @asynccontextmanager
    async def acquire(self):
        """Acquire a database connection with proper lifecycle management"""
        if self._closed:
            raise RuntimeError("Database pool is closed")

        connection = None
        start_time = time.time()

        try:
            if self.is_async:
                async with self.pool.acquire() as conn:
                    self._active_connections.add(conn)
                    self.metrics.active_connections += 1
                    yield conn
            else:
                # Sync connection handling
                connection = self.pool.getconn()
                self._active_connections.add(connection)
                self.metrics.active_connections += 1
                yield connection

        except Exception as e:
            self.metrics.failed_connections += 1
            self.logger.error(f"Connection error: {e}")
            raise

        finally:
            if connection and not self.is_async:
                self.pool.putconn(connection)

            self.metrics.active_connections = max(0, self.metrics.active_connections - 1)

            # Update connection timing
            duration = time.time() - start_time
            self._query_times.append(duration)

            # Keep only recent timings (last 1000)
            if len(self._query_times) > 1000:
                self._query_times = self._query_times[-1000:]

    async def execute(self, query: str, *args, **kwargs) -> Any:
        """Execute a query with retry logic and metrics"""
        start_time = time.time()

        async def _execute():
            async with self.acquire() as connection:
                if self.is_async:
                    return await connection.fetchval(query, *args, **kwargs)
                else:
                    cursor = connection.cursor()
                    cursor.execute(query, args)
                    result = cursor.fetchone()
                    cursor.close()
                    return result

        try:
            result = await self.retry_handler.retry_with_backoff(_execute)
            self.metrics.successful_queries += 1
            return result

        except Exception as e:
            self.metrics.failed_queries += 1
            self.logger.error(f"Query execution failed: {e}")
            raise

        finally:
            duration = time.time() - start_time
            self.metrics.update_query_time(duration)

    async def fetch(self, query: str, *args, **kwargs) -> List[Dict[str, Any]]:
        """Fetch multiple rows with retry logic"""
        start_time = time.time()

        async def _fetch():
            async with self.acquire() as connection:
                if self.is_async:
                    return await connection.fetch(query, *args, **kwargs)
                else:
                    cursor = connection.cursor()
                    cursor.execute(query, args)
                    results = cursor.fetchall()
                    cursor.close()
                    return [dict(row) for row in results]

        try:
            result = await self.retry_handler.retry_with_backoff(_fetch)
            self.metrics.successful_queries += 1
            return result

        except Exception as e:
            self.metrics.failed_queries += 1
            self.logger.error(f"Fetch query failed: {e}")
            raise

        finally:
            duration = time.time() - start_time
            self.metrics.update_query_time(duration)

    async def executemany(self, query: str, args_list: List[Tuple]) -> None:
        """Execute many queries efficiently"""
        start_time = time.time()

        async def _executemany():
            async with self.acquire() as connection:
                if self.is_async:
                    await connection.executemany(query, args_list)
                else:
                    cursor = connection.cursor()
                    cursor.executemany(query, args_list)
                    connection.commit()
                    cursor.close()

        try:
            await self.retry_handler.retry_with_backoff(_executemany)
            self.metrics.successful_queries += len(args_list)

        except Exception as e:
            self.metrics.failed_queries += len(args_list)
            self.logger.error(f"Executemany failed: {e}")
            raise

        finally:
            duration = time.time() - start_time
            self.metrics.update_query_time(duration)

    async def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive pool metrics"""
        if self.pool:
            if self.is_async:
                pool_size = self.pool.get_size()
                pool_idle = self.pool.get_idle_size()
            else:
                # Approximation for sync pool
                pool_size = self.config.max_size
                pool_idle = max(0, pool_size - len(self._active_connections))
        else:
            pool_size = pool_idle = 0

        return {
            'pool_size': pool_size,
            'idle_connections': pool_idle,
            'active_connections': len(self._active_connections),
            'total_queries': self.metrics.total_queries,
            'successful_queries': self.metrics.successful_queries,
            'failed_queries': self.metrics.failed_queries,
            'success_rate': (
                self.metrics.successful_queries / self.metrics.total_queries
                if self.metrics.total_queries > 0 else 0
            ),
            'average_query_time': self.metrics.average_query_time,
            'max_query_time': self.metrics.max_query_time,
            'min_query_time': self.metrics.min_query_time if self.metrics.min_query_time != float('inf') else 0,
            'health_state': self.health_checker.state.value,
            'health_check_failures': self.metrics.health_check_failures,
            'last_health_check': self.metrics.last_health_check,
            'uptime': time.time() - self.metrics.pool_created_at,
            'recent_query_times': self._query_times[-10:] if self._query_times else []
        }

    async def health_check(self) -> Dict[str, Any]:
        """Perform immediate health check"""
        state = await self.health_checker.check_pool_health(self.pool)
        return {
            'status': state.value,
            'pool_available': self.pool is not None and not self._closed,
            'connection_count': len(self._active_connections),
            'last_check': self.health_checker.last_check,
            'failure_count': self.health_checker.failure_count
        }

    async def close(self):
        """Close the database pool and cleanup resources"""
        if self._closed:
            return

        self._closed = True

        # Cancel monitoring tasks
        for task in self._cleanup_tasks:
            task.cancel()

        # Wait for tasks to complete
        if self._cleanup_tasks:
            await asyncio.gather(*self._cleanup_tasks, return_exceptions=True)

        # Close pool
        if self.pool:
            if self.is_async:
                await self.pool.close()
            else:
                self.pool.closeall()

        self.logger.info("Database pool closed successfully")


# Factory functions and utilities
async def create_pool(config: Optional[PoolConfig] = None, **kwargs) -> AsyncDatabasePool:
    """Create and initialize a database pool"""
    if config is None:
        config = PoolConfig(**kwargs)

    pool = AsyncDatabasePool(config)
    await pool.initialize()
    return pool


@asynccontextmanager
async def database_pool(config: Optional[PoolConfig] = None, **kwargs):
    """Context manager for database pool"""
    pool = await create_pool(config, **kwargs)
    try:
        yield pool
    finally:
        await pool.close()


# Configuration helper
def config_from_env() -> PoolConfig:
    """Create pool config from environment variables"""
    import os

    return PoolConfig(
        host=os.getenv('PGHOST', 'localhost'),
        port=int(os.getenv('PGPORT', '5432')),
        database=os.getenv('PGDATABASE', 'trading'),
        user=os.getenv('PGUSER', 'postgres'),
        password=os.getenv('PGPASSWORD', '')
    )
