# Pandas-ta API Fixes Summary

## Issue Description
The trading application was encountering errors when calling pandas-ta indicators through the `cached_indicator` wrapper function:

```
ERROR - Error calculating ema_high: Trend.hlChannel.<locals>.<lambda>() got an unexpected keyword argument 'length'
```

## Root Cause
The issue was in how the lambda functions were defined in the cached_indicator calls. The lambda functions were not properly accepting and passing keyword arguments to the pandas-ta indicator functions.

## Files Modified
- `model/features.py` - Fixed multiple methods that use cached_indicator

## Specific Fixes Applied

### 1. hlChannel Method (Line ~826)
**Before:**
```python
ema55H = cached_indicator(
    lambda: ta.ema(self.data['high'], EMA_PERIODS["medium"]),
    self.data, 'ema_high', length=EMA_PERIODS["medium"]
)
```

**After:**
```python
ema55H = cached_indicator(
    lambda **kwargs: ta.ema(self.data['high'], **kwargs),
    self.data, 'ema_high', length=EMA_PERIODS["medium"]
)
```

### 2. ema Method (Line ~843)
**Before:**
```python
ema_200 = cached_indicator(
    lambda: ta.ema(self.data['close'], EMA_PERIODS["trend"]),
    self.data, 'ema_200', length=EMA_PERIODS["trend"]
)
```

**After:**
```python
ema_200 = cached_indicator(
    lambda **kwargs: ta.ema(self.data['close'], **kwargs),
    self.data, 'ema_200', length=EMA_PERIODS["trend"]
)
```

### 3. lsma_ Method (Line ~859)
**Before:**
```python
lsma = cached_indicator(
    lambda: ta.linreg(self.data['low'], 20, 8),
    self.data, 'lsma', length=20, offset=8
)
```

**After:**
```python
lsma = cached_indicator(
    lambda **kwargs: ta.linreg(self.data['low'], **kwargs),
    self.data, 'lsma', length=20, offset=8
)
```

### 4. vwap_ Method (Line ~875)
**Before:**
```python
vwma = cached_indicator(
    lambda: ta.vwma(self.data['close'], self.data['volume'], 14),
    self.data, 'vwma', length=14
)
```

**After:**
```python
vwma = cached_indicator(
    lambda **kwargs: ta.vwma(self.data['close'], self.data['volume'], **kwargs),
    self.data, 'vwma', length=14
)
```

### 5. stoploss Method (Line ~896)
**Before:**
```python
stop_loss = cached_indicator(
    lambda: ta.high_low_range(self.data.close, length),
    self.data, 'stop_loss', length=length
)
```

**After:**
```python
stop_loss = cached_indicator(
    lambda **kwargs: ta.high_low_range(self.data.close, **kwargs),
    self.data, 'stop_loss', length=length
)
```

### 6. waves Method (Screener class, Line ~985)
**Before:**
```python
ap = cached_indicator(
    lambda: ta.hlc3(df.high, df.low, df.close),
    df, 'hlc3_screener'
)
```

**After:**
```python
ap = cached_indicator(
    lambda **kwargs: ta.hlc3(df.high, df.low, df.close),
    df, 'hlc3_screener'
)
```

### 7. moneyflow Method (Line ~1026)
**Before:**
```python
mfi = cached_indicator(
    lambda: ta.mfi(self.data['high'], self.data['low'], self.data['close'], self.data['volume']),
    self.data, 'mfi'
)
```

**After:**
```python
mfi = cached_indicator(
    lambda **kwargs: ta.mfi(self.data['high'], self.data['low'], self.data['close'], self.data['volume'], **kwargs),
    self.data, 'mfi', length=14
)
```

## Verification
All fixes were verified with a test script that:
1. Creates sample trading data
2. Instantiates the required classes (Tradex_indicator, Trend)
3. Calls each method that was previously failing
4. Confirms all methods execute without errors

## Impact
- ✅ All pandas-ta indicator calls now work correctly
- ✅ Cached indicator functionality preserved
- ✅ No breaking changes to existing API
- ✅ All trading system features restored

The application should now run without the pandas-ta API errors.