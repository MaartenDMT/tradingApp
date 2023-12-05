import random

import numpy as np

import util.loggers as loggers
from util.rl_util import sigmoid

logger = loggers.setup_loggers()
rl_trading = logger['rl_trading']
env_logger = logger['env']


class ActionSpace:
    """
    Represents the space of possible actions.
    Returns:
    0 sell. 1 hold. 2 buy. 3 buy back short. 4. sell back long
    """

    def __init__(self, n):
        self.n = n
        self.allowed: int = self.n - 1

    def sample(self):
        action = random.randint(0, self.allowed)
        if action > 4:
            env_logger.error(f'SAMPLE: not allowed action, action {action}')
            raise ValueError(
                f"Invalid action {action}. Action should be between 0 and 2.")
        env_logger.info(f"SAMPLE: random action generated: {action}")
        return action


class ObservationSpace:
    """Defines the observation space shape."""

    def __init__(self, n) -> None:
        self.shape = (n,)


class DynamicFeatureSelector:
    def __init__(self, initial_features):
        self.features = initial_features
        self.feature_performance = {feature: 0 for feature in initial_features}
        self.exempted_features = {
            'dots', 'l_wave', 'rsi14', 'rsi40', 'ema_200'}

    def evaluate_feature_performance(self, feature, trading_data, target_name):
        """
        Evaluate the performance of a feature.

        Parameters:
        - feature: The feature to be evaluated.
        - trading_data: DataFrame containing trading data.
        - target_name: The name of the target variable column.

        Returns:
        - performance_score: A score indicating the performance of the feature.
        """
        if feature not in trading_data.columns or target_name not in trading_data.columns:
            raise ValueError(
                f"Feature {feature} or target {target_name} not found in trading data")

        # Calculate correlation between the feature and the target
        correlation = trading_data[feature].corr(trading_data[target_name])
        performance_score = abs(correlation)
        self.feature_performance[feature] = performance_score
        return performance_score

    def update_feature_performance(self, feature, performance_impact):
        self.feature_performance[feature] = performance_impact

    def adjust_features(self, threshold=0.5):
        """
        Adjust the features list by removing underperforming features,
        except for those that are exempted.

        Parameters:
        - threshold: The minimum performance score to keep a feature.
        """
        self.features = [
            f for f in self.features
            if self.feature_performance[f] > threshold or f in self.exempted_features
        ]

        # Additional logic to potentially add new features can be included here.
        # Example:
        # if 'new_feature' not in self.features:
        #     self.features.append('new_feature')

    def get_features(self):
        return self.features


class TradingEnvironment:
    def __init__(self, initial_balance: float, leverage, transaction_costs, trade_limit, drawdown_threshold, symbol, trading_mode='futures'):
        self.trading_mode = trading_mode  # 'futures' or 'spot'
        self.initial_balance = initial_balance
        self.leverage = leverage
        self.transaction_costs = transaction_costs
        self.symbol = symbol
        self.trade_limit = trade_limit
        self.drawdown_threshold = drawdown_threshold
        self.pos_to_trade = 0.1
        self.reset()

    def reset(self):
        self.portfolio_balance = self.initial_balance
        self.open_positions = {'long': 0, 'short': 0}
        self.entry_prices = {'long': None, 'short': None}
        self.max_drawdown, self.max_portfolio_balance = 0, 0
        self.stocks_held = 0.0
        self.current_price, self.pnl = 0, 0.0
        self.trade_count = 0
        self.open_trade_pnl = 0  # Track ongoing trade PnL
        self.position_size_l = 0
        self.position_size_s = 0

    def future_trading(self, action, current_price):
        # Validate the action
        valid_actions = [0, 1, 2, 3, 4]
        if action not in valid_actions:
            raise ValueError(f"Invalid action: {action}")

        if not self.is_valid_price(current_price):
            raise ValueError(f"Invalid current price: {current_price}")

        if action == 1:  # Hold
            return self.handle_hold_action()

        self.update_current_price(current_price)

        try:
            if action == 0:  # Sell
                trade_result, portfolio_balance = self.handle_sell_action(
                    current_price, is_future=True)
            elif action == 2:  # Buy
                trade_result, portfolio_balance = self.handle_buy_action(
                    current_price, is_future=True)
            elif action == 3:  # Buy Back
                trade_result, portfolio_balance = self.handle_buy_back_action(
                    current_price, is_future=True)
            elif action == 4:  # Sell Back
                trade_result, portfolio_balance = self.handle_sell_back_action(
                    current_price, is_future=True)
            else:
                return self.handle_hold_action()
        except Exception as e:
            env_logger.error(f"Error during future trading: {e}")
            raise

        if action in [0, 2, 3, 4]:
            self.update_position_tracking(action, current_price)

        if action in [0, 2]:
            self.trade_count += 1

        return trade_result, portfolio_balance

    def spot_trading(self, action, current_price):
        # Validate the action
        valid_actions = [0, 1, 2, 4]
        if action not in valid_actions:
            raise ValueError(f"Invalid action: {action}")

        if not self.is_valid_price(current_price):
            raise ValueError(f"Invalid current price: {current_price}")

        if action == 1:  # Hold
            return self.handle_hold_action()

        self.update_current_price(current_price)

        try:
            if action == 0:  # Sell
                trade_result, portfolio_balance = self.handle_sell_action(
                    current_price, is_future=False)
            elif action == 2:  # Buy
                trade_result, portfolio_balance = self.handle_buy_action(
                    current_price, is_future=False)
            elif action == 4:  # Sell Back
                trade_result, portfolio_balance = self.handle_sell_back_action(
                    current_price, is_future=False)
            else:
                return self.handle_hold_action()

        except Exception as e:
            env_logger.error(f"Error during spot trading: {e}")
            raise

        if action in [0, 2, 3]:
            self.update_position_tracking(action, current_price)

        if action in [0, 2]:
            self.trade_count += 1

        return trade_result, portfolio_balance

    def finalize_trade(self, trade_result,  trading_volume, current_price, ):
        # Calculate the transaction cost as a percentage of the trade value
        trade_value = current_price * abs(trading_volume)
        transaction_cost = trade_value * self.transaction_costs

        # Apply transaction costs
        trade_result -= transaction_cost

        # Update portfolio balance
        self.portfolio_balance += trade_result

        # Update max portfolio balance
        self.max_portfolio_balance = max(
            self.max_portfolio_balance, self.portfolio_balance)

        # Calculate and update drawdown
        if self.max_portfolio_balance > 0:  # Add check to avoid division by zero
            current_drawdown = 1 - \
                (self.portfolio_balance / self.max_portfolio_balance)
            self.max_drawdown = max(self.max_drawdown, current_drawdown)
        else:
            # Set to 0 or an appropriate default value if max_portfolio_balance is 0
            current_drawdown = 0

        # Reset open_trade_pnl if all positions are closed
        if self.open_positions['long'] == 0 and self.open_positions['short'] == 0:
            self.open_trade_pnl = 0

        # Update tracking after trade
        self.update_current_price(current_price)

        rl_trading.info(
            f"Updated balance ({self.trading_mode}): ${self.portfolio_balance:.2f}, volume held: {trading_volume}")

        return trade_result, self.portfolio_balance

    def handle_hold_action(self):
        env_logger.info("Hold action taken. No change in portfolio.")
        return 0, self.portfolio_balance

    def handle_sell_action(self, current_price, is_future):
        trade_result: float = 0.0
        if is_future:
            # Logic for selling in futures trading
            self.entry_prices['short'] = current_price if self.entry_prices['short'] is None else self.entry_prices['short']
            trading_volume = self.portfolio_balance * self.pos_to_trade
            self.position_size_s += trading_volume
        else:
            rl_trading.info(
                f"Starting balance: ${self.portfolio_balance:.2f}, Stocks held: {self.stocks_held}")
            if self.stocks_held <= 0:
                raise ValueError("No stocks available to sell.")
            amount_to_trade = self.portfolio_balance * 0.1
            # Logic for selling in spot trading

            stocks_to_sell = min(
                self.stocks_held, amount_to_trade / current_price)
            trade_result = stocks_to_sell * \
                (current_price - self.entry_prices['long'])
            self.stocks_held -= stocks_to_sell
            rl_trading.info(
                f"Selling {stocks_to_sell:.2f} stocks at ${current_price:.2f} each.")

        trading_volume = self.stocks_held if not is_future else self.position_size_s
        return self.finalize_trade(trade_result, trading_volume, current_price)

    def handle_buy_action(self, current_price, is_future):
        trade_result: float = 0.0
        if is_future:
            # Logic for buying in futures trading
            self.entry_prices['long'] = current_price if self.entry_prices['long'] is None else self.entry_prices['long']
            trading_volume = self.portfolio_balance * self.pos_to_trade
            if trading_volume > self.portfolio_balance:
                raise ValueError("Trading amount exceeds available balance.")
            self.position_size_l += trading_volume
        else:
            rl_trading.info(
                f"Starting balance: ${self.portfolio_balance:.2f}, Stocks held: {self.stocks_held}")
            amount_to_trade = self.portfolio_balance * 0.1
            if amount_to_trade > self.portfolio_balance:
                raise ValueError("Trading amount exceeds available balance.")
            # Logic for buying in spot trading
            stocks_to_buy = min(amount_to_trade / current_price,
                                self.portfolio_balance / current_price)
            self.stocks_held += stocks_to_buy
            self.entry_prices['long'] = current_price if self.stocks_held > 0 else None
            rl_trading.info(
                f"Buying {stocks_to_buy:.2f} stocks at ${current_price:.2f} each.")
            trade_result = -amount_to_trade  # Negative because it's a cost

        trading_volume = self.stocks_held if not is_future else self.position_size_l
        return self.finalize_trade(trade_result, current_price, trading_volume)

    def handle_buy_back_action(self, current_price, is_future):
        trade_result: float = 0.0
        if is_future:
            # Logic for buying back in futures trading
            if self.open_positions['short'] <= 0:
                rl_trading.info(
                    "Attempted to close a short position when none were open.")
                return 0, self.portfolio_balance  # Return zero trade result and current balance
            trade_result = (
                self.entry_prices['short'] - current_price) * self.leverage
            self.position_size_s -= self.position_size_s

        trading_volume = self.stocks_held if not is_future else self.position_size_s
        return self.finalize_trade(trade_result, trading_volume, current_price)

    def handle_sell_back_action(self, current_price, is_future):
        trade_result: float = 0.0
        if is_future:
            if self.open_positions['long'] <= 0:
                rl_trading.info(
                    "Attempted to close a long position when none were open.")
                return 0, self.portfolio_balance  # Return zero trade result and current balance
            trade_result = (
                current_price - self.entry_prices['long']) * self.leverage
            self.position_size_l -= self.position_size_l
        else:
            # In spot trading, sell back would mean selling the stocks you hold
            if self.stocks_held <= 0:
                rl_trading.info("No stocks to sell back.")
                return 0, self.portfolio_balance  # No stocks to sell back

            # Calculate the number of stocks to sell
            stocks_to_sell = self.stocks_held

            # Calculate profit or loss from the sell-back action
            trade_result = stocks_to_sell * \
                (current_price - self.entry_prices['long'])
            self.stocks_held -= stocks_to_sell

            rl_trading.info(
                f"Sell back {stocks_to_sell:.2f} stocks at ${current_price:.2f} each.")

        trading_volume = self.stocks_held if not is_future else self.position_size_l
        return self.finalize_trade(trade_result, current_price, trading_volume)

    def longterm_pnl(self):
        """
        Calculate long-term profit and loss (PnL) for trades that are still open.

        Returns:
            float: The long-term PnL as a percentage of the initial balance.
        """
        # Check if there are any open trades
        if self.open_positions['long'] > 0 or self.open_positions['short'] > 0:
            # Calculate PnL based on current open positions
            pnl_long = ((self.current_price - self.entry_prices['long']) * self.open_positions['long']
                        if self.open_positions['long'] > 0 else 0) * self.leverage
            pnl_short = ((self.entry_prices['short'] - self.current_price) * self.open_positions['short']
                         if self.open_positions['short'] > 0 else 0) * self.leverage

            self.open_trade_pnl = pnl_long + pnl_short
        return (self.open_trade_pnl / self.initial_balance) * 100 if self.initial_balance != 0 else 0

    def shortterm_pnl(self):
        """
        Calculate short-term profit and loss (PnL) for each trade.

        Returns:
            float: The short-term PnL as a percentage of the initial balance.
        """
        if self.trading_mode == 'futures':
            return self.calculate_futures_pnl()
        else:  # spot trading
            return self.calculate_spot_pnl()

    def calculate_futures_pnl(self):
        """
        Calculate PnL for futures trading.

        PnL is calculated based on the difference between the current price and the entry price,
        multiplied by the position size and leverage.

        Returns:
            float: The PnL as a percentage of the initial balance.
        """
        pnl_long = 0
        pnl_short = 0

        if self.position_size_l > 0 and self.entry_prices['long'] is not None:
            pnl_long = (
                (self.current_price - self.entry_prices['long']) * self.position_size_l) * self.leverage

        if self.position_size_s < 0 and self.entry_prices['short'] is not None:
            pnl_short = (
                (self.entry_prices['short'] - self.current_price) * abs(self.position_size_s)) * self.leverage

        total_pnl = pnl_long + pnl_short
        return (total_pnl / self.initial_balance) * 100 if self.initial_balance != 0 else 0

    def calculate_spot_pnl(self):
        """
        Calculate PnL for spot trading.

        PnL is calculated based on the difference between the current price and the entry price,
        multiplied by the number of stocks held.

        Returns:
            float: The PnL as a percentage of the initial balance.
        """
        pnl = 0
        if self.stocks_held > 0 and self.entry_prices['long'] is not None:
            pnl += (self.current_price -
                    self.entry_prices['long']) * self.stocks_held
        elif self.stocks_held < 0 and self.entry_prices['short'] is not None:
            pnl += (self.entry_prices['short'] -
                    self.current_price) * abs(self.stocks_held)

        return (pnl / self.initial_balance) * 100 if self.initial_balance != 0 else 0

    def update_position_tracking(self, action, current_price):
        if action in [2, 4]:  # Buy or Sell Back (long positions)
            self.update_long_position_tracking(action, current_price)
        elif action in [0, 3]:  # Sell or Buy Back (short positions)
            self.update_short_position_tracking(action, current_price)

    def update_long_position_tracking(self, action, current_price):
        if action == 2:  # Buy
            self.entry_prices['long'] = self.calculate_new_entry_price(
                'long', current_price)
            self.open_positions['long'] += 1
        elif action == 4:  # Sell Back
            self.open_positions['long'] -= 1
            if self.open_positions['long'] == 0:
                self.entry_prices['long'] = 0

    def update_short_position_tracking(self, action, current_price):
        if action == 0:  # Sell
            self.entry_prices['short'] = self.calculate_new_entry_price(
                'short', current_price)
            self.open_positions['short'] += 1
        elif action == 3:  # Buy Back
            self.open_positions['short'] -= 1
            if self.open_positions['short'] == 0:
                self.entry_prices['short'] = 0

    def calculate_new_entry_price(self, position_type, current_price):
        if self.open_positions[position_type] == 0:
            # If there are no open positions, return the current price as the new entry price
            return current_price

        # If the entry price for the given position type is None, consider it as zero
        entry_price = self.entry_prices[position_type] if self.entry_prices[position_type] is not None else 0

        # Calculate the new average entry price
        total_cost = entry_price * self.open_positions[position_type]
        total_positions = self.open_positions[position_type] + 1
        return (total_cost + current_price) / total_positions

    def update_current_price(self, current_price):
        self.current_price = current_price

    def calculate_drawdown_penalty(self):
        if self.max_drawdown is None or self.drawdown_threshold is None:
            return 0  # Default penalty if values are not set

        # Calculate current drawdown
        current_drawdown = 1 - (self.portfolio_balance / self.initial_balance)

        # Check if drawdown exceeds threshold
        if current_drawdown > self.max_drawdown:
            # Calculate penalty (negative value)
            penalty = -abs(current_drawdown - self.max_drawdown)
        else:
            penalty = 0

        return penalty

    def calculate_trading_penalty(self):
        return -1 * max(0, self.trade_count - self.trade_limit) / self.trade_limit

    def execute_trade(self, action, current_price):
        if self.trading_mode == 'futures':
            return self.future_trading(action, current_price)
        elif self.trading_mode == 'spot':
            return self.spot_trading(action, current_price)
        else:
            raise ValueError("Invalid trading mode")

    def is_valid_price(self, price):
        """Validate the given price."""
        return price > 0

    def check_balance_limit(self, amount):
        """Check if the trading amount exceeds the balance limit."""
        if amount > self.portfolio_balance:
            raise ValueError("Trading amount exceeds available balance.")
