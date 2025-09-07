"""
Enhanced Trading Engine for the reinforcement learning trading system.
Consolidates and optimizes trading execution logic from multiple files.
"""

from typing import Dict, Tuple

import util.loggers as loggers

logger = loggers.setup_loggers()
rl_trading = logger['rl_trading']
env_logger = logger['env']


class TradingEngine:
    """
    Unified trading engine that handles both spot and futures trading with improved performance.
    """

    def __init__(self,
                 initial_balance: float = 10000.0,
                 leverage: float = 1.0,
                 transaction_costs: float = 0.001,
                 max_position_ratio: float = 0.1,
                 trading_mode: str = 'spot'):
        """
        Initialize the trading engine.

        Args:
            initial_balance: Starting portfolio balance
            leverage: Leverage for futures trading
            transaction_costs: Transaction cost ratio (0.001 = 0.1%)
            max_position_ratio: Maximum position size as ratio of balance
            trading_mode: 'spot' or 'futures'
        """
        self.initial_balance = initial_balance
        self.leverage = leverage
        self.transaction_costs = self._validate_transaction_costs(transaction_costs)
        self.max_position_ratio = max_position_ratio
        self.trading_mode = trading_mode

        self.reset()

    def reset(self):
        """Reset the trading engine to initial state."""
        self.portfolio_balance = self.initial_balance
        self.max_portfolio_balance = self.initial_balance
        self.max_drawdown = 0.0

        # Position tracking
        self.positions = {
            'long': {'size': 0.0, 'entry_price': None, 'count': 0},
            'short': {'size': 0.0, 'entry_price': None, 'count': 0}
        }

        # Trading metrics
        self.trade_count = 0
        self.current_price = 0.0
        self.total_pnl = 0.0
        self.unrealized_pnl = 0.0

        # For spot trading
        self.stocks_held = 0.0

    def execute_trade(self, action: int, current_price: float) -> Tuple[float, float]:
        """
        Execute a trading action.

        Args:
            action: Trading action (0=sell, 1=hold, 2=buy, 3=buy_back, 4=sell_back)
            current_price: Current market price

        Returns:
            Tuple of (trade_result, updated_portfolio_balance)
        """
        if not self._validate_price(current_price):
            raise ValueError(f"Invalid price: {current_price}")

        self.current_price = current_price

        if action == 1:  # Hold
            return self._handle_hold()

        if self.trading_mode == 'futures':
            return self._execute_futures_trade(action, current_price)
        else:
            return self._execute_spot_trade(action, current_price)

    def _execute_futures_trade(self, action: int, current_price: float) -> Tuple[float, float]:
        """Execute futures trading action."""
        trade_result = 0.0

        if action == 0:  # Sell (open short)
            trade_result = self._open_short_position(current_price)
        elif action == 2:  # Buy (open long)
            trade_result = self._open_long_position(current_price)
        elif action == 3:  # Buy back (close short)
            trade_result = self._close_short_position(current_price)
        elif action == 4:  # Sell back (close long)
            trade_result = self._close_long_position(current_price)

        return self._finalize_trade(trade_result, action)

    def _execute_spot_trade(self, action: int, current_price: float) -> Tuple[float, float]:
        """Execute spot trading action."""
        trade_result = 0.0

        if action == 0:  # Sell
            trade_result = self._sell_stocks(current_price)
        elif action == 2:  # Buy
            trade_result = self._buy_stocks(current_price)
        elif action == 4:  # Sell all
            trade_result = self._sell_all_stocks(current_price)

        return self._finalize_trade(trade_result, action)

    def _open_long_position(self, current_price: float) -> float:
        """Open a long position in futures trading."""
        position_value = self.portfolio_balance * self.max_position_ratio
        position_size = position_value / current_price

        # Update position tracking
        if self.positions['long']['size'] == 0:
            self.positions['long']['entry_price'] = current_price
        else:
            # Calculate weighted average entry price
            total_size = self.positions['long']['size'] + position_size
            current_value = self.positions['long']['size'] * self.positions['long']['entry_price']
            new_value = position_size * current_price
            self.positions['long']['entry_price'] = (current_value + new_value) / total_size

        self.positions['long']['size'] += position_size
        self.positions['long']['count'] += 1

        rl_trading.info(f"Opened long position: {position_size:.4f} at {current_price:.2f}")
        return -position_value  # Negative because it's a cost

    def _open_short_position(self, current_price: float) -> float:
        """Open a short position in futures trading."""
        position_value = self.portfolio_balance * self.max_position_ratio
        position_size = position_value / current_price

        # Update position tracking
        if self.positions['short']['size'] == 0:
            self.positions['short']['entry_price'] = current_price
        else:
            # Calculate weighted average entry price
            total_size = self.positions['short']['size'] + position_size
            current_value = self.positions['short']['size'] * self.positions['short']['entry_price']
            new_value = position_size * current_price
            self.positions['short']['entry_price'] = (current_value + new_value) / total_size

        self.positions['short']['size'] += position_size
        self.positions['short']['count'] += 1

        rl_trading.info(f"Opened short position: {position_size:.4f} at {current_price:.2f}")
        return position_value  # Positive for short position

    def _close_long_position(self, current_price: float) -> float:
        """Close long position in futures trading."""
        if self.positions['long']['size'] <= 0:
            rl_trading.warning("No long position to close")
            return 0.0

        position_size = self.positions['long']['size']
        entry_price = self.positions['long']['entry_price']

        # Calculate PnL
        pnl = (current_price - entry_price) * position_size * self.leverage

        # Reset position
        self.positions['long'] = {'size': 0.0, 'entry_price': None, 'count': 0}

        rl_trading.info(f"Closed long position: {position_size:.4f}, PnL: {pnl:.2f}")
        return pnl

    def _close_short_position(self, current_price: float) -> float:
        """Close short position in futures trading."""
        if self.positions['short']['size'] <= 0:
            rl_trading.warning("No short position to close")
            return 0.0

        position_size = self.positions['short']['size']
        entry_price = self.positions['short']['entry_price']

        # Calculate PnL
        pnl = (entry_price - current_price) * position_size * self.leverage

        # Reset position
        self.positions['short'] = {'size': 0.0, 'entry_price': None, 'count': 0}

        rl_trading.info(f"Closed short position: {position_size:.4f}, PnL: {pnl:.2f}")
        return pnl

    def _buy_stocks(self, current_price: float) -> float:
        """Buy stocks in spot trading."""
        amount_to_invest = self.portfolio_balance * self.max_position_ratio

        if amount_to_invest > self.portfolio_balance:
            amount_to_invest = self.portfolio_balance * 0.95  # Leave some buffer

        stocks_to_buy = amount_to_invest / current_price

        if stocks_to_buy > 0:
            self.stocks_held += stocks_to_buy
            if self.positions['long']['size'] == 0:
                self.positions['long']['entry_price'] = current_price

            rl_trading.info(f"Bought {stocks_to_buy:.4f} stocks at {current_price:.2f}")
            return -amount_to_invest  # Negative because it's a cost

        return 0.0

    def _sell_stocks(self, current_price: float) -> float:
        """Sell stocks in spot trading."""
        if self.stocks_held <= 0:
            rl_trading.warning("No stocks to sell")
            return 0.0

        # Sell a portion of holdings
        stocks_to_sell = min(self.stocks_held, self.stocks_held * self.max_position_ratio)
        sale_value = stocks_to_sell * current_price

        self.stocks_held -= stocks_to_sell

        if self.stocks_held <= 0:
            self.positions['long']['entry_price'] = None

        rl_trading.info(f"Sold {stocks_to_sell:.4f} stocks at {current_price:.2f}")
        return sale_value

    def _sell_all_stocks(self, current_price: float) -> float:
        """Sell all stocks in spot trading."""
        if self.stocks_held <= 0:
            rl_trading.warning("No stocks to sell")
            return 0.0

        sale_value = self.stocks_held * current_price
        self.stocks_held = 0.0
        self.positions['long']['entry_price'] = None

        rl_trading.info(f"Sold all stocks: {sale_value:.2f}")
        return sale_value

    def _handle_hold(self) -> Tuple[float, float]:
        """Handle hold action."""
        return 0.0, self.portfolio_balance

    def _finalize_trade(self, trade_result: float, action: int) -> Tuple[float, float]:
        """Finalize trade by applying transaction costs and updating balance."""
        if trade_result != 0.0:
            # Apply transaction costs
            transaction_cost = abs(trade_result) * self.transaction_costs
            trade_result -= transaction_cost

            # Update portfolio balance
            self.portfolio_balance += trade_result

            # Update tracking metrics
            if action in [0, 2]:  # Only count actual trades
                self.trade_count += 1

            # Update max portfolio balance and drawdown
            self.max_portfolio_balance = max(self.max_portfolio_balance, self.portfolio_balance)
            self._update_drawdown()

        return trade_result, self.portfolio_balance

    def calculate_pnl(self) -> Dict[str, float]:
        """
        Calculate profit and loss for current positions.

        Returns:
            Dictionary with PnL information
        """
        unrealized_pnl = 0.0

        if self.trading_mode == 'futures':
            # Long position PnL
            if self.positions['long']['size'] > 0 and self.positions['long']['entry_price']:
                long_pnl = ((self.current_price - self.positions['long']['entry_price']) *
                           self.positions['long']['size'] * self.leverage)
                unrealized_pnl += long_pnl

            # Short position PnL
            if self.positions['short']['size'] > 0 and self.positions['short']['entry_price']:
                short_pnl = ((self.positions['short']['entry_price'] - self.current_price) *
                            self.positions['short']['size'] * self.leverage)
                unrealized_pnl += short_pnl
        else:
            # Spot trading PnL
            if self.stocks_held > 0 and self.positions['long']['entry_price']:
                spot_pnl = (self.current_price - self.positions['long']['entry_price']) * self.stocks_held
                unrealized_pnl += spot_pnl

        self.unrealized_pnl = unrealized_pnl

        return {
            'unrealized_pnl': unrealized_pnl,
            'unrealized_pnl_pct': (unrealized_pnl / self.initial_balance * 100) if self.initial_balance > 0 else 0.0,
            'total_value': self.portfolio_balance + unrealized_pnl,
            'total_return_pct': ((self.portfolio_balance + unrealized_pnl - self.initial_balance) /
                               self.initial_balance * 100) if self.initial_balance > 0 else 0.0
        }

    def get_position_info(self) -> Dict:
        """Get current position information."""
        return {
            'long_size': self.positions['long']['size'],
            'long_entry': self.positions['long']['entry_price'],
            'short_size': self.positions['short']['size'],
            'short_entry': self.positions['short']['entry_price'],
            'stocks_held': self.stocks_held,
            'portfolio_balance': self.portfolio_balance,
            'trade_count': self.trade_count,
            'max_drawdown': self.max_drawdown
        }

    def calculate_drawdown_penalty(self, threshold: float = 0.1) -> float:
        """Calculate penalty based on current drawdown."""
        if self.max_drawdown > threshold:
            return -(self.max_drawdown - threshold) * 10  # Penalty multiplier
        return 0.0

    def calculate_trading_penalty(self, max_trades: int = 100) -> float:
        """Calculate penalty for excessive trading."""
        if self.trade_count > max_trades:
            return -(self.trade_count - max_trades) / max_trades
        return 0.0

    def _update_drawdown(self):
        """Update maximum drawdown."""
        if self.max_portfolio_balance > 0:
            current_drawdown = 1 - (self.portfolio_balance / self.max_portfolio_balance)
            self.max_drawdown = max(self.max_drawdown, current_drawdown)

    def _validate_price(self, price: float) -> bool:
        """Validate if price is reasonable."""
        return isinstance(price, (int, float)) and price > 0

    def _validate_transaction_costs(self, costs: float) -> float:
        """Validate transaction costs are within reasonable range."""
        if not 0 <= costs <= 0.05:  # 0% to 5%
            rl_trading.warning(f"Transaction costs {costs} out of range (0-5%)")
            return max(0, min(costs, 0.05))  # Clamp to valid range
        return costs

    def get_metrics(self) -> Dict[str, float]:
        """Get comprehensive trading metrics."""
        pnl_info = self.calculate_pnl()

        return {
            'portfolio_balance': self.portfolio_balance,
            'total_value': pnl_info['total_value'],
            'total_return_pct': pnl_info['total_return_pct'],
            'unrealized_pnl_pct': pnl_info['unrealized_pnl_pct'],
            'max_drawdown': self.max_drawdown,
            'trade_count': self.trade_count,
            'drawdown_penalty': self.calculate_drawdown_penalty(),
            'trading_penalty': self.calculate_trading_penalty()
        }
