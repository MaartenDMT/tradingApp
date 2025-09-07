"""
Async Trading Operations with Advanced Portfolio Management

This module provides comprehensive async trading functionality with:
- Order management and execution
- Portfolio tracking and analysis
- Risk management and position sizing
- Performance metrics and monitoring
- Real-time market data streaming
- Trade execution optimization
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from .async_client import AsyncCCXTClient, MarketDataStream
from .cache import HybridCache, cache_key_for_portfolio_data
from .db_pool import AsyncDatabasePool

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Order types."""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    STOP_LIMIT = "stop_limit"


class OrderStatus(Enum):
    """Order status."""
    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class PositionSide(Enum):
    """Position side."""
    LONG = "long"
    SHORT = "short"


@dataclass
class Order:
    """Trading order."""
    id: str
    symbol: str
    order_type: OrderType
    side: str  # 'buy' or 'sell'
    amount: Decimal
    price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    status: OrderStatus = OrderStatus.PENDING
    timestamp: datetime = field(default_factory=datetime.now)
    filled_amount: Decimal = Decimal('0')
    remaining_amount: Optional[Decimal] = None
    average_price: Optional[Decimal] = None
    fees: Dict[str, Decimal] = field(default_factory=dict)
    info: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.remaining_amount is None:
            self.remaining_amount = self.amount

    @property
    def is_complete(self) -> bool:
        """Check if order is completely filled."""
        return self.filled_amount >= self.amount

    @property
    def fill_percentage(self) -> float:
        """Get fill percentage."""
        if self.amount == 0:
            return 0.0
        return float(self.filled_amount / self.amount * 100)


@dataclass
class Position:
    """Trading position."""
    symbol: str
    side: PositionSide
    size: Decimal
    entry_price: Decimal
    current_price: Decimal
    timestamp: datetime = field(default_factory=datetime.now)
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None
    unrealized_pnl: Decimal = Decimal('0')
    realized_pnl: Decimal = Decimal('0')
    fees: Decimal = Decimal('0')

    @property
    def market_value(self) -> Decimal:
        """Calculate current market value."""
        return self.size * self.current_price

    @property
    def pnl_percentage(self) -> float:
        """Calculate PnL percentage."""
        if self.entry_price == 0:
            return 0.0

        price_diff = self.current_price - self.entry_price
        if self.side == PositionSide.SHORT:
            price_diff = -price_diff

        return float((price_diff / self.entry_price) * 100)

    def update_price(self, new_price: Decimal) -> None:
        """Update current price and unrealized PnL."""
        self.current_price = new_price

        if self.side == PositionSide.LONG:
            self.unrealized_pnl = (new_price - self.entry_price) * self.size
        else:
            self.unrealized_pnl = (self.entry_price - new_price) * self.size


@dataclass
class Portfolio:
    """Portfolio management."""
    balance: Dict[str, Decimal] = field(default_factory=dict)
    positions: Dict[str, Position] = field(default_factory=dict)
    orders: Dict[str, Order] = field(default_factory=dict)
    total_value: Decimal = Decimal('0')
    total_pnl: Decimal = Decimal('0')
    daily_pnl: Decimal = Decimal('0')
    drawdown: Decimal = Decimal('0')
    max_drawdown: Decimal = Decimal('0')
    win_rate: float = 0.0
    sharpe_ratio: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)

    def add_position(self, position: Position) -> None:
        """Add or update position."""
        self.positions[position.symbol] = position
        self.last_updated = datetime.now()

    def remove_position(self, symbol: str) -> Optional[Position]:
        """Remove position."""
        position = self.positions.pop(symbol, None)
        if position:
            self.last_updated = datetime.now()
        return position

    def add_order(self, order: Order) -> None:
        """Add order."""
        self.orders[order.id] = order
        self.last_updated = datetime.now()

    def update_balance(self, currency: str, amount: Decimal) -> None:
        """Update balance."""
        self.balance[currency] = amount
        self.last_updated = datetime.now()

    def calculate_total_value(self) -> Decimal:
        """Calculate total portfolio value."""
        total = sum(self.balance.values())
        total += sum(pos.market_value for pos in self.positions.values())
        self.total_value = total
        return total

    def calculate_total_pnl(self) -> Decimal:
        """Calculate total PnL."""
        realized = sum(pos.realized_pnl for pos in self.positions.values())
        unrealized = sum(pos.unrealized_pnl for pos in self.positions.values())
        self.total_pnl = realized + unrealized
        return self.total_pnl


class RiskManager:
    """Risk management system."""

    def __init__(
        self,
        max_position_size: float = 0.1,  # 10% of portfolio
        max_daily_loss: float = 0.05,    # 5% daily loss limit
        max_drawdown: float = 0.15,      # 15% max drawdown
        max_leverage: float = 3.0,       # 3x max leverage
        stop_loss_percentage: float = 0.02  # 2% stop loss
    ):
        self.max_position_size = max_position_size
        self.max_daily_loss = max_daily_loss
        self.max_drawdown = max_drawdown
        self.max_leverage = max_leverage
        self.stop_loss_percentage = stop_loss_percentage

    def calculate_position_size(
        self,
        portfolio_value: Decimal,
        risk_percentage: float,
        entry_price: Decimal,
        stop_loss_price: Optional[Decimal] = None
    ) -> Decimal:
        """Calculate optimal position size based on risk."""
        risk_amount = portfolio_value * Decimal(str(risk_percentage))

        if stop_loss_price:
            price_risk = abs(entry_price - stop_loss_price)
            if price_risk > 0:
                position_size = risk_amount / price_risk
            else:
                position_size = risk_amount / entry_price
        else:
            # Use default stop loss percentage
            price_risk = entry_price * Decimal(str(self.stop_loss_percentage))
            position_size = risk_amount / price_risk

        # Apply maximum position size limit
        max_size = portfolio_value * Decimal(str(self.max_position_size))
        return min(position_size, max_size)

    def validate_order(self, order: Order, portfolio: Portfolio) -> Tuple[bool, str]:
        """Validate order against risk parameters."""
        # Check daily loss limit
        if portfolio.daily_pnl < -portfolio.total_value * Decimal(str(self.max_daily_loss)):
            return False, "Daily loss limit exceeded"

        # Check maximum drawdown
        if portfolio.drawdown > Decimal(str(self.max_drawdown)):
            return False, "Maximum drawdown exceeded"

        # Check position size
        if order.price:
            position_value = order.amount * order.price
            portfolio_value = portfolio.calculate_total_value()

            if position_value > portfolio_value * Decimal(str(self.max_position_size)):
                return False, "Position size exceeds maximum allowed"

        return True, "Order validated"

    def should_close_position(self, position: Position) -> Tuple[bool, str]:
        """Check if position should be closed due to risk limits."""
        # Check stop loss
        if position.stop_loss:
            if position.side == PositionSide.LONG and position.current_price <= position.stop_loss:
                return True, "Stop loss triggered"
            elif position.side == PositionSide.SHORT and position.current_price >= position.stop_loss:
                return True, "Stop loss triggered"

        # Check take profit
        if position.take_profit:
            if position.side == PositionSide.LONG and position.current_price >= position.take_profit:
                return True, "Take profit triggered"
            elif position.side == PositionSide.SHORT and position.current_price <= position.take_profit:
                return True, "Take profit triggered"

        return False, "Position within risk parameters"


class TradingEngine:
    """Async trading engine with portfolio management."""

    def __init__(
        self,
        ccxt_client: AsyncCCXTClient,
        db_pool: AsyncDatabasePool,
        cache: HybridCache,
        risk_manager: Optional[RiskManager] = None
    ):
        self.ccxt_client = ccxt_client
        self.db_pool = db_pool
        self.cache = cache
        self.risk_manager = risk_manager or RiskManager()
        self.portfolio = Portfolio()
        self.market_data_stream: Optional[MarketDataStream] = None
        self._running = False
        self._tasks: List[asyncio.Task] = []

        # Performance metrics
        self.trade_count = 0
        self.successful_trades = 0
        self.total_fees = Decimal('0')
        self.start_time = datetime.now()

    async def start(self) -> None:
        """Start the trading engine."""
        if self._running:
            return

        self._running = True
        logger.info("Starting trading engine")

        # Start background tasks
        self._tasks = [
            asyncio.create_task(self._monitor_positions()),
            asyncio.create_task(self._monitor_orders()),
            asyncio.create_task(self._update_portfolio()),
            asyncio.create_task(self._risk_monitoring())
        ]

        # Initialize market data stream
        symbols = list(self.portfolio.positions.keys())
        if symbols:
            self.market_data_stream = MarketDataStream(
                self.ccxt_client,
                symbols,
                timeframe='1m'
            )
            await self.market_data_stream.start()

    async def stop(self) -> None:
        """Stop the trading engine."""
        if not self._running:
            return

        self._running = False
        logger.info("Stopping trading engine")

        # Cancel all tasks
        for task in self._tasks:
            task.cancel()

        # Wait for tasks to complete
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

        # Stop market data stream
        if self.market_data_stream:
            await self.market_data_stream.stop()

        self._tasks.clear()

    async def place_order(
        self,
        symbol: str,
        order_type: OrderType,
        side: str,
        amount: Decimal,
        price: Optional[Decimal] = None,
        stop_price: Optional[Decimal] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> Order:
        """Place a trading order."""
        # Create order object
        order = Order(
            id=f"order_{int(time.time() * 1000)}",
            symbol=symbol,
            order_type=order_type,
            side=side,
            amount=amount,
            price=price,
            stop_price=stop_price
        )

        # Validate order with risk manager
        is_valid, reason = self.risk_manager.validate_order(order, self.portfolio)
        if not is_valid:
            order.status = OrderStatus.REJECTED
            logger.warning(f"Order rejected: {reason}")
            return order

        try:
            # Execute order through CCXT
            exchange_order = await self.ccxt_client.create_order(
                symbol=symbol,
                order_type=order_type.value,
                side=side,
                amount=float(amount),
                price=float(price) if price else None,
                params=params or {}
            )

            # Update order with exchange information
            order.id = exchange_order['id']
            order.status = OrderStatus.OPEN
            order.info = exchange_order

            # Add to portfolio
            self.portfolio.add_order(order)

            # Cache order
            cache_key = f"order:{order.id}"
            await self.cache.set(cache_key, order, ttl=3600)

            logger.info(f"Order placed: {order.id} - {symbol} {side} {amount}")

        except Exception as e:
            order.status = OrderStatus.REJECTED
            logger.error(f"Failed to place order: {e}")

        return order

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        try:
            order = self.portfolio.orders.get(order_id)
            if not order:
                return False

            # Cancel through exchange
            await self.ccxt_client.cancel_order(order_id, order.symbol)

            # Update order status
            order.status = OrderStatus.CANCELLED

            logger.info(f"Order cancelled: {order_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    async def close_position(self, symbol: str, percentage: float = 100.0) -> Optional[Order]:
        """Close a position."""
        position = self.portfolio.positions.get(symbol)
        if not position:
            return None

        # Calculate amount to close
        close_amount = position.size * Decimal(str(percentage / 100.0))

        # Determine side (opposite of position)
        side = "sell" if position.side == PositionSide.LONG else "buy"

        # Place market order to close
        order = await self.place_order(
            symbol=symbol,
            order_type=OrderType.MARKET,
            side=side,
            amount=close_amount
        )

        return order

    async def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get portfolio summary."""
        await self._update_portfolio_metrics()

        return {
            "total_value": float(self.portfolio.total_value),
            "total_pnl": float(self.portfolio.total_pnl),
            "daily_pnl": float(self.portfolio.daily_pnl),
            "positions_count": len(self.portfolio.positions),
            "open_orders": len([o for o in self.portfolio.orders.values() if o.status == OrderStatus.OPEN]),
            "win_rate": self.portfolio.win_rate,
            "sharpe_ratio": self.portfolio.sharpe_ratio,
            "max_drawdown": float(self.portfolio.max_drawdown),
            "trade_count": self.trade_count,
            "successful_trades": self.successful_trades,
            "total_fees": float(self.total_fees),
            "uptime": str(datetime.now() - self.start_time)
        }

    async def _monitor_positions(self) -> None:
        """Monitor positions for risk management."""
        while self._running:
            try:
                for symbol, position in self.portfolio.positions.items():
                    # Get current price
                    ticker = await self.ccxt_client.fetch_ticker(symbol)
                    current_price = Decimal(str(ticker['last']))

                    # Update position
                    position.update_price(current_price)

                    # Check risk limits
                    should_close, reason = self.risk_manager.should_close_position(position)
                    if should_close:
                        logger.info(f"Closing position {symbol}: {reason}")
                        await self.close_position(symbol)

                await asyncio.sleep(5)  # Check every 5 seconds

            except Exception as e:
                logger.error(f"Error monitoring positions: {e}")
                await asyncio.sleep(10)

    async def _monitor_orders(self) -> None:
        """Monitor open orders."""
        while self._running:
            try:
                for order_id, order in list(self.portfolio.orders.items()):
                    if order.status not in [OrderStatus.OPEN, OrderStatus.PENDING]:
                        continue

                    # Fetch order status from exchange
                    exchange_order = await self.ccxt_client.fetch_order(order_id, order.symbol)

                    # Update order status
                    if exchange_order['status'] == 'closed':
                        order.status = OrderStatus.FILLED
                        order.filled_amount = Decimal(str(exchange_order['filled']))
                        order.average_price = Decimal(str(exchange_order['average'] or 0))

                        # Update position if order is filled
                        await self._process_filled_order(order)

                    elif exchange_order['status'] == 'canceled':
                        order.status = OrderStatus.CANCELLED

                await asyncio.sleep(2)  # Check every 2 seconds

            except Exception as e:
                logger.error(f"Error monitoring orders: {e}")
                await asyncio.sleep(5)

    async def _process_filled_order(self, order: Order) -> None:
        """Process a filled order and update positions."""
        try:
            symbol = order.symbol
            position = self.portfolio.positions.get(symbol)

            if order.side == "buy":
                if position:
                    # Add to existing position or reduce short position
                    if position.side == PositionSide.LONG:
                        # Average price calculation
                        total_cost = (position.size * position.entry_price) + (order.filled_amount * order.average_price)
                        total_size = position.size + order.filled_amount
                        position.entry_price = total_cost / total_size
                        position.size = total_size
                    else:
                        # Reducing short position
                        position.size -= order.filled_amount
                        if position.size <= 0:
                            # Position closed or flipped
                            self.portfolio.remove_position(symbol)
                else:
                    # New long position
                    position = Position(
                        symbol=symbol,
                        side=PositionSide.LONG,
                        size=order.filled_amount,
                        entry_price=order.average_price,
                        current_price=order.average_price
                    )
                    self.portfolio.add_position(position)

            else:  # sell order
                if position:
                    if position.side == PositionSide.LONG:
                        # Reducing long position
                        position.size -= order.filled_amount
                        if position.size <= 0:
                            # Position closed
                            self.portfolio.remove_position(symbol)
                    else:
                        # Adding to short position
                        total_cost = (position.size * position.entry_price) + (order.filled_amount * order.average_price)
                        total_size = position.size + order.filled_amount
                        position.entry_price = total_cost / total_size
                        position.size = total_size
                else:
                    # New short position
                    position = Position(
                        symbol=symbol,
                        side=PositionSide.SHORT,
                        size=order.filled_amount,
                        entry_price=order.average_price,
                        current_price=order.average_price
                    )
                    self.portfolio.add_position(position)

            # Update trade statistics
            self.trade_count += 1
            if order.average_price and position and position.unrealized_pnl > 0:
                self.successful_trades += 1

            # Add fees
            if order.fees:
                self.total_fees += sum(order.fees.values())

            logger.info(f"Processed filled order: {order.id}")

        except Exception as e:
            logger.error(f"Error processing filled order {order.id}: {e}")

    async def _update_portfolio(self) -> None:
        """Update portfolio values and cache."""
        while self._running:
            try:
                await self._update_portfolio_metrics()

                # Cache portfolio data
                cache_key = cache_key_for_portfolio_data("main", int(time.time()))
                await self.cache.set(cache_key, self.portfolio, ttl=60)

                await asyncio.sleep(30)  # Update every 30 seconds

            except Exception as e:
                logger.error(f"Error updating portfolio: {e}")
                await asyncio.sleep(60)

    async def _update_portfolio_metrics(self) -> None:
        """Update portfolio performance metrics."""
        try:
            # Calculate total value and PnL
            self.portfolio.calculate_total_value()
            self.portfolio.calculate_total_pnl()

            # Calculate win rate
            if self.trade_count > 0:
                self.portfolio.win_rate = self.successful_trades / self.trade_count

            # TODO: Implement Sharpe ratio calculation
            # TODO: Implement drawdown calculation

        except Exception as e:
            logger.error(f"Error updating portfolio metrics: {e}")

    async def _risk_monitoring(self) -> None:
        """Monitor overall risk metrics."""
        while self._running:
            try:
                # Check daily loss limit
                max_daily_loss = self.portfolio.total_value * Decimal(str(self.risk_manager.max_daily_loss))
                if self.portfolio.daily_pnl < -max_daily_loss:
                    logger.critical("Daily loss limit exceeded - stopping trading")
                    await self.stop()
                    break

                # Check maximum drawdown
                if self.portfolio.drawdown > Decimal(str(self.risk_manager.max_drawdown)):
                    logger.critical("Maximum drawdown exceeded - stopping trading")
                    await self.stop()
                    break

                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                logger.error(f"Error in risk monitoring: {e}")
                await asyncio.sleep(60)


@asynccontextmanager
async def trading_engine_context(
    exchange_config: Dict[str, Any],
    db_config: Dict[str, Any],
    cache_config: Optional[Dict[str, Any]] = None,
    risk_config: Optional[Dict[str, Any]] = None
):
    """Context manager for trading engine lifecycle."""
    # Initialize components
    ccxt_client = AsyncCCXTClient(**exchange_config)
    db_pool = AsyncDatabasePool(**db_config)
    cache = HybridCache(**(cache_config or {}))
    risk_manager = RiskManager(**(risk_config or {}))

    # Create trading engine
    engine = TradingEngine(ccxt_client, db_pool, cache, risk_manager)

    try:
        # Connect all components
        await ccxt_client.connect()
        await db_pool.connect()
        await cache.connect()

        # Start trading engine
        await engine.start()

        yield engine

    finally:
        # Cleanup
        await engine.stop()
        await ccxt_client.disconnect()
        await db_pool.disconnect()
        await cache.disconnect()


# Export main classes
__all__ = [
    'OrderType', 'OrderStatus', 'PositionSide',
    'Order', 'Position', 'Portfolio',
    'RiskManager', 'TradingEngine',
    'trading_engine_context'
]
