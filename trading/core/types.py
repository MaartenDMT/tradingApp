"""
Trading System Type Definitions

Defines all enumerations, data classes, and type definitions used throughout
the trading system for consistency and type safety.
"""

import enum
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional, Union
from decimal import Decimal


class OrderType(enum.Enum):
    """Order type enumeration."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"
    OCO = "oco"  # One-Cancels-Other
    ICE = "ice"  # Immediate-or-Cancel-Extended


class OrderSide(enum.Enum):
    """Order side enumeration."""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(enum.Enum):
    """Order status enumeration."""
    PENDING = "pending"
    OPEN = "open"
    CLOSED = "closed"
    CANCELED = "canceled"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class TimeInForce(enum.Enum):
    """Time in force enumeration."""
    GTC = "gtc"  # Good Till Canceled
    IOC = "ioc"  # Immediate or Cancel
    FOK = "fok"  # Fill or Kill
    GTD = "gtd"  # Good Till Date
    DAY = "day"  # Day Order


class PositionType(enum.Enum):
    """Position type enumeration."""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


class RiskLevel(enum.Enum):
    """Risk level enumeration."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class TradeResult:
    """Result of a trade operation."""
    success: bool
    order_id: Optional[str] = None
    message: str = ""
    data: Optional[Dict[str, Any]] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class OrderParams:
    """Parameters for creating an order."""
    symbol: str
    side: OrderSide
    order_type: OrderType
    amount: Union[float, Decimal]
    price: Optional[Union[float, Decimal]] = None
    stop_price: Optional[Union[float, Decimal]] = None
    time_in_force: TimeInForce = TimeInForce.GTC
    stop_loss: Optional[Union[float, Decimal]] = None
    take_profit: Optional[Union[float, Decimal]] = None
    client_order_id: Optional[str] = None
    leverage: Optional[int] = None
    reduce_only: bool = False
    post_only: bool = False
    extra_params: Optional[Dict[str, Any]] = None


@dataclass
class Position:
    """Trading position representation."""
    symbol: str
    side: PositionType
    size: Union[float, Decimal]
    entry_price: Union[float, Decimal]
    current_price: Union[float, Decimal]
    unrealized_pnl: Union[float, Decimal]
    realized_pnl: Union[float, Decimal]
    timestamp: datetime
    leverage: Optional[int] = None
    liquidation_price: Optional[Union[float, Decimal]] = None
    
    @property
    def market_value(self) -> Union[float, Decimal]:
        """Calculate market value of the position."""
        return self.size * self.current_price
    
    @property
    def percentage_pnl(self) -> float:
        """Calculate percentage PnL."""
        if self.entry_price == 0:
            return 0.0
        return float((self.current_price - self.entry_price) / self.entry_price * 100)


@dataclass
class MarketData:
    """Market data snapshot."""
    symbol: str
    timestamp: datetime
    bid: Union[float, Decimal]
    ask: Union[float, Decimal]
    last: Union[float, Decimal]
    volume: Union[float, Decimal]
    high_24h: Optional[Union[float, Decimal]] = None
    low_24h: Optional[Union[float, Decimal]] = None
    change_24h: Optional[float] = None
    change_percentage_24h: Optional[float] = None
    
    @property
    def spread(self) -> Union[float, Decimal]:
        """Calculate bid-ask spread."""
        return self.ask - self.bid
    
    @property
    def spread_percentage(self) -> float:
        """Calculate spread as percentage of mid price."""
        mid_price = (self.bid + self.ask) / 2
        return float(self.spread / mid_price * 100)


@dataclass
class RiskMetrics:
    """Risk assessment metrics."""
    var_95: float  # Value at Risk at 95% confidence
    var_99: float  # Value at Risk at 99% confidence
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    beta: Optional[float] = None
    alpha: Optional[float] = None
    correlation_market: Optional[float] = None
    volatility: float = 0.0
    downside_deviation: float = 0.0


@dataclass
class PerformanceMetrics:
    """Trading performance metrics."""
    total_return: float
    total_return_percentage: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    max_consecutive_wins: int
    max_consecutive_losses: int
    largest_win: float
    largest_loss: float
    avg_trade_duration: float  # in hours
    risk_metrics: Optional[RiskMetrics] = None


@dataclass
class TradingSignal:
    """Trading signal representation."""
    symbol: str
    signal_type: OrderSide
    strength: float  # Signal strength from 0.0 to 1.0
    confidence: float  # Confidence level from 0.0 to 1.0
    entry_price: Optional[Union[float, Decimal]] = None
    stop_loss: Optional[Union[float, Decimal]] = None
    take_profit: Optional[Union[float, Decimal]] = None
    risk_reward_ratio: Optional[float] = None
    timeframe: Optional[str] = None
    source: str = "unknown"
    metadata: Optional[Dict[str, Any]] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class StrategyConfig:
    """Base configuration for trading strategies."""
    name: str
    enabled: bool = True
    risk_per_trade: float = 0.01  # 1% of capital per trade
    max_positions: int = 5
    min_signal_strength: float = 0.6
    min_confidence: float = 0.7
    timeframe: str = "1h"
    symbols: Optional[list] = None
    parameters: Optional[Dict[str, Any]] = None
