# Utility Modules Documentation

This document provides detailed information about the utility modules in the trading application.

## Overview

The `util/` directory contains various utility modules that provide common functionality used throughout the application. These modules are designed to be reusable and follow best practices for Python development.

## Module Descriptions

### async_client.py
Provides an optimized async CCXT client with WebSocket support and connection optimization features:
- Connection pooling and reuse
- WebSocket streaming for real-time data
- Request batching and rate limiting
- Caching mechanisms
- Error handling and resilience

### async_trading.py
Handles asynchronous trading operations and order management:
- Async order placement and cancellation
- Position management
- Risk controls
- Performance monitoring

### cache.py
Implements an enhanced multi-layer cache system:
- Redis integration for distributed caching
- Memory cache with LRU eviction
- TTL (Time To Live) support
- Cache invalidation and warming
- Performance metrics and monitoring

### candlestick_parser.py
Provides candlestick pattern recognition and analysis:
- Pattern detection algorithms
- Technical indicator calculations
- Market sentiment analysis

### config_manager.py
Centralized configuration management system:
- Consolidates settings from multiple sources
- Type-safe access to configuration values
- Dynamic configuration reloading
- API key management

### db_pool.py
Database connection pooling and management:
- Connection pooling for PostgreSQL
- Query optimization
- Transaction management
- Error handling

### error_handling.py
Standardized error handling and exception management:
- Consistent error reporting
- Exception chaining
- Context-aware error messages

### loggers.py
Logging configuration and management:
- Multiple logger instances
- File and console handlers
- Log rotation

### ml_optimization.py
Machine learning model optimization utilities:
- Hyperparameter tuning
- Model selection
- Performance evaluation

### parallel.py
Parallel processing and concurrency utilities:
- Thread pools
- Process pools
- Async task management

### secure_config.py
Secure configuration handling and encryption:
- Configuration encryption
- Secure storage
- Access controls

### secure_credentials.py
Secure credential storage and retrieval:
- API key encryption
- Credential masking
- Secure storage mechanisms

### standardized_loggers.py
Enhanced logging with standardized formatting:
- Consistent log formats
- Multiple output destinations
- Performance optimized

### utils.py
General utility functions:
- Data conversion
- String manipulation
- File operations

### validation.py
Data validation and sanitization utilities:
- Input validation
- Data sanitization
- Schema validation

### websocket_util.py
WebSocket connection management and utilities:
- Connection management
- Message handling
- Error recovery

## Usage Examples

### Using the Configuration Manager

```python
from util.config_manager import get_config

# Get the global configuration instance
config = get_config()

# Access configuration values
window_size = config.window_size
api_key = config.get_api_key('binance')
```

### Using Secure Credentials

```python
from util.secure_credentials import encrypt_credential, decrypt_credential

# Encrypt a credential
encrypted = encrypt_credential("my_secret_key", "my_password")

# Decrypt a credential
decrypted = decrypt_credential("my_secret_key", encrypted)
```

### Using the Cache System

```python
from util.cache import get_global_cache

# Get the global cache instance
cache = await get_global_cache()

# Store a value in cache
await cache.set("key", "value", ttl=300)  # 5 minutes TTL

# Retrieve a value from cache
value = await cache.get("key")
```

## Best Practices

1. **Reusability**: Utility modules should be designed for reuse across the application
2. **Single Responsibility**: Each module should have a single, well-defined purpose
3. **Error Handling**: Utility modules should handle errors gracefully and provide meaningful error messages
4. **Performance**: Utility modules should be optimized for performance, especially those used frequently
5. **Security**: Utility modules that handle sensitive data should follow security best practices
6. **Testing**: Utility modules should be well-tested with comprehensive test coverage

## Testing

Utility modules are tested as part of the overall test suite. To run tests for utility modules:

```bash
python -m pytest test/test_utils.py
```

Or run the full test suite:

```bash
python test/run_tests.py
```