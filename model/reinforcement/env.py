import os
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_ta as ta

import model.reinforcement.indicators as ind
import util.loggers as loggers
from util.utils import convert_df, features, load_config, tradex_features

logger = loggers.setup_loggers()
env_logger = logger['env']
rl_logger = logger['rl']

# Constants
EARLY_STOP_REWARD_THRESHOLD = -5.0


class MultiAgentEnvironment:
    def __init__(self, num_agents, *args, **kwargs):
        self.num_agents = num_agents
        self.agents = [Environment(*args, **kwargs) for _ in range(num_agents)]

        self.action_spaces = [agent.action_space for agent in self.agents]
        self.observation_spaces = [
            agent.observation_space for agent in self.agents]
        self.look_backs = [agent.look_back for agent in self.agents]

        self.action_space = self.get_action_space()
        self.observation_space = self.get_observation_space()
        self.look_back = self.get_look_back()

    def reset(self):
        observations = [agent.reset() for agent in self.agents]
        return observations

    def step(self, actions):
        states, rewards, infos, dones = [], [], [], []
        for agent, action in zip(self.agents, actions):
            state, reward, info, done = agent.step(action)
            states.append(state)
            rewards.append(reward)
            infos.append(info)
            dones.append(done)
        return states, rewards, infos, dones

    def get_action_space(self):
        return self.action_spaces[0]

    def get_observation_space(self):
        return self.observation_spaces[0]

    def get_look_back(self):
        return self.look_backs[0]


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


class Environment:
    """The main trading environment."""

    def __init__(self, symbol, features, limit, time, actions, min_acc):
        self.config = load_config()
        self.symbol, self.features = symbol, features
        self.limit, self.time = limit, time
        self.action_space = ActionSpace(actions)
        self.min_accuracy = min_acc
        self.portfolio_balance = float(
            self.config['tradingenv']['portfolio_balance'])
        self._initialize_environment()

    def _initialize_environment(self):
        self._set_initial_parameters()
        self.tradingEnv = TradingEnvironment(
            self.portfolio_balance, 5, 0.05, self.trade_limit, self.threshold, self.symbol)

        self.original_data = self._get_data()
        self._create_features()
        self._setup_observation_space()

    def _set_initial_parameters(self):
        self.last_price, self.bar = None, 0
        self.trade_limit, self.threshold = 5, 0.2
        self.last_price, self.last_accuracy = None, None
        self.high_acc_counter, self.patience = 0, int(
            self.config['env']['patience'])
        self.wait, self.high_acc_threshold = 0, 5
        self.stocks_held = 0.0
        self.max_portfolio_balance, self.total_reward = 0, 0
        self.pnl, self.total_pnl = 0, 0
        self.accuracy, self.holding_time = 0, 0
        self.correct_actions, self.last_action = 0, None
        self.trade_result = 0

    def _setup_observation_space(self):
        # Get the number of features in the observation space
        num_features = len(self.env_data[self.features].columns)

        self.observation_space = ObservationSpace(num_features)
        self.look_back = 1

        env_logger.info(
            f"Environment initialized with symbol {self.symbol} and features {self.features}. Observation space shape: {self.observation_space.shape}, lookback: {self.look_back}")

    def _get_data(self):
        pickle_file_name = self.config['Path']['2020_30m_data']

        if not os.path.exists(pickle_file_name):
            env_logger.error('No data has been written')
            return pd.DataFrame()  # Return an empty DataFrame instead of None for consistency

        with open(pickle_file_name, 'rb') as f:
            data_ = pickle.load(f)

        if data_.empty:
            env_logger.error("Loaded data is empty.")
            return pd.DataFrame()

        data = convert_df(data_)

        if data.empty or data.isnull().values.any():
            env_logger.error("Converted data is empty or contains NaN values.")
            return pd.DataFrame()

        percentage_to_keep = float(self.config['Data']['percentage']) / 100.0
        rows_to_keep = int(len(data) * percentage_to_keep)
        data = data.head(rows_to_keep)

        rl_logger.info(f'Dataframe shape: {data.shape}')
        return data

    def _create_features(self):
        if self.original_data.empty:
            env_logger.error("Original data is empty.")
            return

        feature = features(self.original_data.copy())
        processed_features = tradex_features(
            self.symbol, self.original_data.copy())

        if feature.empty or processed_features.empty:
            env_logger.error(
                "Feature generation failed, resulting in empty data.")
            return

        self.original_data = pd.concat([processed_features, feature], axis=1)
        self.original_data.dropna(inplace=True)  # Remove rows with NaN values

        self.original_data['last_price'] = self.original_data['close'].shift(1)
        self.original_data['portfolio_balance'] = self.portfolio_balance
        self.env_data = self.original_data[self.features].copy()

        if self.env_data.isnull().values.any():
            env_logger.error("NaN values detected in environment data.")
            return

        # Ensure there are no division by zero or invalid operations
        self.original_data['r'] = np.log(self.env_data['close'] / self.env_data['close'].shift(
            int(self.config['env']['shifts'])).replace(0, np.nan))
        self.env_data['r'] = self.original_data['r'].fillna(
            0)  # Fill NaNs resulted from log operation

        self.env_data['d'] = self.compute_action(self.original_data)
        self.env_data['d'] = self.env_data['d'].fillna(
            method='ffill')  # Forward fill to handle NaNs

        # Normalize data, handle any NaNs that may arise from normalization
        self.env_data = (self.env_data - self.env_data.mean()
                         ) / self.env_data.std()
        # Replace any NaNs resulted from normalization
        self.env_data.fillna(0, inplace=True)

        self.env_data['action'] = 0

        env_logger.info("Features created successfully.")

    def get_feature_names(self):
        return self.env_data.columns.tolist()

    def _get_state(self):
        env_logger.info("STATE:get the state.")
        state = self._extract_state(self.bar)
        # print(f"the state shape {state.shape} & state {state}")
        return state

    def reset(self):
        env_logger.info("RESET:Environment reset.")

        self.env_data['action'] = 0
        self._set_initial_parameters()
        self.tradingEnv.reset()
        self.bar = self.look_back

        state = self._extract_state(self.bar)

        env_logger.info(
            f"RESET:the state shape {state.shape} & size is {state.size}")
        return state

    def _extract_state(self, bar):
        state_slice = self.env_data[self.features].iloc[bar -
                                                        self.look_back:bar].values
        return np.array(state_slice, dtype=np.float32)

    def step(self, action):
        env_logger.info(f"STEP: Action taken: {action}")

        # Validate index range
        if self.bar >= len(self.original_data):
            env_logger.error("Index out of range. Returning default state.")
            return np.zeros_like(self.env_data.iloc[0]), 0, {}, True

        df_row_raw = self.original_data.iloc[self.bar]
        env_row_raw = self.env_data.iloc[self.bar].copy()

        self.last_price = df_row_raw['close'] if pd.notna(
            df_row_raw['close']) else self.last_price
        self.current_price = df_row_raw['close'] if pd.notna(
            df_row_raw['close']) else self.last_price

        self.trade_result = self.execute_trade(action, self.current_price)
        env_row_raw['portfolio_balance'] = self.portfolio_balance
        self.pnl = self.tradingEnv.shortterm_pnl()

        reward = self.calculate_reward(
            action, df_row_raw) if pd.notna(df_row_raw).all() else 0

        correct = ind.is_correct_action(action, ind.get_optimal_action(reward))
        self._update_state_values(action, self.current_price, reward)
        done = self.is_episode_done()

        state = self._get_state() if pd.notna(
            self._get_state()).all() else np.zeros_like(self._get_state())
        info = {}

        self._log_step_details(
            reward, done, self._get_conditions(df_row_raw), correct)

        if correct:
            self.correct_actions += 1

        return state, reward, info, done

    def execute_trade(self, action, current_price):
        if self.action_space.n in [5]:
            trade_result, self.portfolio_balance = self.tradingEnv.future_trading(
                action, current_price)
        elif self.action_space.n == 4:
            # Unpack all returned values from spot_trading
            trade_result, self.portfolio_balance, self.max_drawdown, self.max_portfolio_balance, self.stocks_held = self.tradingEnv.spot_trading(
                action, current_price)
        else:
            env_logger.error(
                f'there is no environment with an action space of {self.action_space.n}')
            return None

        return trade_result

    def calculate_reward(self, action, df_row_raw):
        # Here, add logic to compute the reward
        market_condition_reward = self.compute_market_condition_reward(
            action, df_row_raw)
        financial_outcome_reward = self.compute_financial_outcome_reward(
            action)
        risk_adjusted_reward = self.compute_risk_adjusted_reward(
            self.env_data['r'])  # assuming self.env_data['r'] holds returns history
        drawdown_penalty = self.tradingEnv.calculate_drawdown_penalty()
        trading_penalty = self.tradingEnv.calculate_trading_penalty()

        return self.combined_reward(market_condition_reward, financial_outcome_reward,
                                    risk_adjusted_reward, drawdown_penalty, trading_penalty)

    def _get_conditions(self, df_row):
        """Helper method to centralize the conditions logic."""
        current_bar_index = self.bar

        # Adjusting to use self.original_data
        super_buy: bool = (df_row['dots'] == 1) & [df_row['l_wave'] >= -50]
        super_sell: bool = (df_row['dots'] == -1) & [df_row['l_wave'] >= 50]
        low_volatility: bool = (df_row['rsi14'] >= 45) & (
            df_row['rsi14'] <= 55)
        strong_upward_movement: bool = df_row['rsi14'] > 70
        strong_downward_movement: bool = df_row['rsi14'] < 30
        going_up_condition: bool = (df_row['close'] > df_row['last_price']) & (
            df_row['close'] > df_row['ema_200']) & (df_row['rsi40'] > 50)
        going_down_condition: bool = (df_row['close'] < df_row['last_price']) & (
            df_row['close'] < df_row['ema_200']) & (df_row['rsi40'] < 50)

        strong_buy_signal = strong_upward_movement & ~ind.is_increasing_trend(self.original_data,
                                                                              current_bar_index)
        strong_sell_signal = strong_downward_movement & ~ind.is_increasing_trend(self.original_data,
                                                                                 current_bar_index)  # ~ is the element-wise logical NOT

        # SHORT
        short_stochastic_signal = ~ind.short_stochastic_condition(
            self.original_data, current_bar_index)
        short_bollinger_outside = ~ind.short_bollinger_condition(
            self.original_data, current_bar_index)
        # LONG ONlY
        long_stochastic_signal = ~ind.long_stochastic_condition(
            self.original_data, current_bar_index)
        long_bollinger_outside = ~ind.long_bollinger_condition(
            self.original_data, current_bar_index)
        macd_buy = ~ind.macd_condition(self.original_data, current_bar_index)
        high_volatility = ~ind.atr_condition(
            self.original_data, current_bar_index)
        adx_signal = ~ind.adx_condition(self.original_data, current_bar_index)
        psar_signal = ~ind.parabolic_sar_condition(
            self.original_data, current_bar_index)
        cdl_pattern = ~ind.cdl_pattern(self.original_data, current_bar_index)
        volume_break = ~ind.volume_breakout(
            self.original_data, current_bar_index)
        resistance_break_signal = ~ind.resistance_break(
            self.original_data, current_bar_index)

        return short_stochastic_signal, short_bollinger_outside, long_stochastic_signal, long_bollinger_outside, low_volatility, going_up_condition, going_down_condition, strong_buy_signal, strong_sell_signal, super_buy, super_sell, macd_buy, high_volatility, adx_signal, psar_signal, cdl_pattern, volume_break, resistance_break_signal

    def compute_action(self, df):
        return df.apply(lambda row: self._compute_action_for_row(row), axis=1)

    def _compute_action_for_row(self, df_row):
        """Compute action based on the data row and current price."""
        short_stochastic_signal, short_bollinger_outside, long_stochastic_signal, long_bollinger_outside, low_volatility, going_up_condition, going_down_condition, strong_buy_signal, strong_sell_signal, super_buy, super_sell, macd_buy, high_volatility, adx_signal, psar_signal, cdl_pattern, volume_break, resistance_break_signal = self._get_conditions(
            df_row)

        bullish_conditions = [strong_buy_signal, super_buy, macd_buy, long_stochastic_signal, long_bollinger_outside,
                              high_volatility, adx_signal, psar_signal, cdl_pattern, volume_break, resistance_break_signal]
        bearish_conditions = [strong_sell_signal, super_sell,
                              short_stochastic_signal, short_bollinger_outside]
        neutral_conditions = [going_up_condition,
                              going_down_condition, low_volatility]

        if any(neutral_conditions):
            self.holding_time += 1
            return 1  # Neutral action
        elif any(bullish_conditions):
            self.holding_time = 0
            return 2  # Bullish action
        elif any(bearish_conditions):
            self.holding_time = 0
            return 0  # Bearish action
        else:
            # Default logic if none of the above conditions are met
            self.holding_time += 1
            return 1

    def compute_market_condition_reward(self, action, df_row):
        """Compute and return the reward based on the current action, data row, and current price."""
        short_stochastic_signal, short_bollinger_outside, long_stochastic_signal, long_bollinger_outside, low_volatility, going_up_condition, going_down_condition, strong_buy_signal, strong_sell_signal, super_buy, super_sell, macd_buy, high_volatility, adx_signal, psar_signal, cdl_pattern, volume_break, resistance_break_signal = self._get_conditions(
            df_row)

        bullish_conditions = [strong_buy_signal, super_buy, macd_buy, long_stochastic_signal, long_bollinger_outside,
                              high_volatility, adx_signal, psar_signal, cdl_pattern, volume_break, resistance_break_signal]
        bearish_conditions = [strong_sell_signal, super_sell,
                              short_stochastic_signal, short_bollinger_outside]
        neutral_conditions = [going_up_condition,
                              going_down_condition, low_volatility]

        if action == 2 and any(bullish_conditions):
            if super_buy:
                return 2  # Higher reward for correctly identifying a strong buy signal
            else:
                return 1  # Reward for other bullish conditions
        elif action == 0 and any(bearish_conditions):
            if super_sell:
                return 2  # Higher reward for a strong sell signal
            else:
                return 1  # Reward for other bearish conditions
        elif action == 1 and any(neutral_conditions):
            if low_volatility:
                return 0.2
            elif going_up_condition:
                return 0.5
            elif going_down_condition:
                return 0.5
        else:
            return -0.2  # Penalize when no specific condition is met

    def compute_financial_outcome_reward(self, action):
        """
        Compute the reward based on action and market data using the PnL from the TradingEnvironment.

        Parameters:
        - action: The action taken by the agent.

        Returns:
        - Calculated reward.
        """

        # Get the PnL from the TradingEnvironment
        pnl = self.tradingEnv.longterm_pnl()

        if action == 1 and pnl > 0:  # Reward for holding if in profit
            # Calculate holding reward based on holding time and PnL
            holding_reward = ind.calculate_holding_reward(
                self.holding_time, pnl)
            reward = holding_reward
        else:
            # For buy or sell actions, the reward is directly based on PnL
            reward = pnl

        return reward

    def compute_risk_adjusted_reward(self, returns_history):
        """
        Compute the reward based on action and risk.

        Parameters:
        - returns_history: Historical returns for risk calculation.

        Returns:
        - Calculated reward.
        """
        # Calculate Sharpe Ratio for risk-adjusted return
        sharpe_ratio = ind.calculate_sharpe_ratio(returns_history)

        # Combine profit/loss and risk-adjusted return
        # Here you can decide how to weigh these components
        reward = sharpe_ratio  # Simple additive model as an example

        return reward

    def combined_reward(self, market_condition_reward, financial_outcome_reward, risk_adjusted_reward, drawdown_penalty, trading_penalty, weights=(0.2, 0.4, 0.2, 0.1, 0.1)):
        """
        Combine different reward components into a single reward value.

        Parameters:
        - market_condition_reward: Reward based on market conditions.
        - financial_outcome_reward: Reward based on immediate financial outcome.
        - risk_adjusted_reward: Reward based on risk-adjusted returns.
        - weights: Tuple of weights for each component.

        Returns:
        - Combined reward.
        """
        self.market_condition_reward_weighted = weights[0] * \
            market_condition_reward

        self.financial_outcome_reward_weighted = weights[1] * \
            financial_outcome_reward

        self.risk_adjusted_reward_weighted = weights[2] * \
            risk_adjusted_reward
        self.drawdown_penalty_weighted = weights[3] * drawdown_penalty
        self.trading_penalty_weighted = weights[4] * trading_penalty

        rl_logger.info(
            f"Financial Outcome Reward: { self.financial_outcome_reward_weighted}")
        rl_logger.info(
            f"Risk Adjusted Reward: { self.risk_adjusted_reward_weighted}")
        rl_logger.info(f"Weights: {weights}")

        combined = (round(self.market_condition_reward_weighted, 2) +
                    round(self.financial_outcome_reward_weighted, 2) +
                    round(self.risk_adjusted_reward_weighted, 2) +
                    round(self.drawdown_penalty_weighted, 2) +
                    round(self.trading_penalty_weighted, 2))

        return combined

    def calculate_accuracy(self):
        """Calculate the accuracy of the agent's actions."""
        if self.bar - self.look_back > 0:
            return self.correct_actions / (self.bar - self.look_back)
        else:
            return 0

    def _update_state_values(self, action, current_price, reward):
        """Update internal state values based on the action, current price, and received reward."""

        # Log update information
        env_logger.info(
            f"Updating the state values based on action: {action}, current_price: {current_price}, reward: {reward}")

        # Update last action and price
        if action in [0, 2]:
            self.last_action = action
        self.last_price = current_price

        # Update data and bar counter
        self.env_data.loc[self.env_data.index[self.bar], 'action'] = action
        self.bar += 1

        # Update reward and accuracy
        self.total_reward = round(self.total_reward + reward, 2)
        self.accuracy = self.calculate_accuracy()

    def _log_step_details(self, reward, done, *conditions, correct=False):

        # Log reward component details
        env_logger.info(
            f"LOG: Weighted Market Condition Reward: {self.market_condition_reward_weighted:.2f}")
        env_logger.info(
            f"LOG: Weighted Financial Outcome Reward: {self.financial_outcome_reward_weighted:.2f}")
        env_logger.info(
            f"LOG: Weighted Risk Adjusted Reward: {self.risk_adjusted_reward_weighted:.2f}")
        env_logger.info(
            f"LOG: Weighted Drawdown Penalty: {self.drawdown_penalty_weighted:.2f}")
        env_logger.info(
            f"LOG: Weighted Trading Penalty: {self.trading_penalty_weighted:.2f}")

        portfolio_balance_str = f"${self.portfolio_balance:.2f}" if self.portfolio_balance is not None else "N/A"
        stocks_held_str = f"{self.stocks_held:.2f}" if self.stocks_held is not None else "N/A"
        pnl_str = f"{self.pnl:.2f}" if self.pnl is not None else "N/A"

        # Log general step details
        env_logger.info(
            f"LOG: Step {self.bar} | Action: {self.last_action} | Reward: {reward:.2f} | Correct: {correct}")
        env_logger.info(
            f"LOG: Portfolio Balance: {portfolio_balance_str} | Stocks Held: {stocks_held_str} | PNL: {pnl_str}")
        env_logger.info(
            f"LOG: Total Reward: {self.total_reward} | Accuracy: {self.accuracy:.2%}")
        if done:
            env_logger.info("LOG: Episode done.")

        # Log market condition details
        condition_names = ["Short Stochastic", "Short Bollinger", "Long Stochastic", "Long Bollinger", "Low Volatility",
                           "Going Up", "Going Down", "Strong Buy", "Strong Sell", "Super Buy", "Super Sell",
                           "MACD Buy", "High Volatility", "ADX Signal", "PSAR Signal", "CDL Pattern", "Volume Break", "Resistance Break"]
        condition_details = dict(zip(condition_names, conditions))
        for name, value in condition_details.items():
            env_logger.info(f"LOG: Condition - {name}: {value}")

        env_logger.info("LOG: ===========================================")

    def is_episode_done(self):
        if self.total_reward <= EARLY_STOP_REWARD_THRESHOLD:
            env_logger.warning("DONE: Early stopping due to less rewards.")
            return True
        if self.bar >= len(self.env_data) - 1:
            env_logger.warning("DONE: Early stopping due to no more bars.")
            return True
        if self.wait >= self.patience:
            self.wait = 0
            env_logger.warning(
                "DONE: Early stopping due to lack of improvement.")
            return True
        if self.high_acc_counter >= self.high_acc_threshold:
            env_logger.warning(
                f"DONE: Stopping early due to sustained high accuracy: {self.accuracy}")
            return True
        if self.accuracy < self.min_accuracy and self.bar > self.look_back + 50:
            self.wait = 0
            env_logger.warning(
                f"DONE: Stopping early due to low accuracy: {self.accuracy}")
            return True
        return False


class TradingEnvironment:
    def __init__(self, initial_balance: float, leverage, transaction_costs, trade_limit, drawdown_threshold, symbol, trading_mode='futures'):
        self.trading_mode = trading_mode  # 'futures' or 'spot'
        self.initial_balance = initial_balance
        self.leverage = leverage
        self.transaction_costs = transaction_costs
        self.symbol = symbol
        self.trade_limit = trade_limit
        self.drawdown_threshold = drawdown_threshold
        self.reset()

    def reset(self):
        self.portfolio_balance = self.initial_balance
        self.open_positions = {'long': 0, 'short': 0}
        self.entry_prices = {'long': None, 'short': None}
        self.max_drawdown, self.max_portfolio_balance = 0, 0
        self.stocks_held, self.position_size = 0.0, 0
        self.current_price, self.pnl = 0, 0.0
        self.trade_count = 0

    def future_trading(self, action, current_price):
        """
        Execute a trade based on the action.
        Actions: 0: Sell (open short), 2: Buy (open long), 
                 3: Buy Back (close short), 4: Sell Back (close long)

        Parameters:
        - action (int): Action taken.
        - current_price (float): Current stock price.

        Returns:
        - float: The profit or loss from the trade.
        - float: The updated portfolio balance.
        """
        # if action not in [0, 1, 2, 3, 4]:
        #     raise ValueError(
        #         "Invalid action. Action must be 0 (sell), 2 (buy), 3 (buy back), or 4 (sell back).")

        trade_result: float = 0.0
        # No action taken for 'Hold'
        if action == 1:
            env_logger.info("Hold action taken. No change in portfolio.")
            return 0, self.portfolio_balance
        elif action == 0:  # Sell (open a short position)
            self.open_positions['short'] += 1
            self.entry_prices['short'] = current_price if self.entry_prices['short'] is None else self.entry_prices['short']
            self.position_size += 1

        elif action == 2:  # Buy (open a long position)
            self.open_positions['long'] += 1
            self.entry_prices['long'] = current_price if self.entry_prices['long'] is None else self.entry_prices['long']
            self.position_size += 1

        elif action == 3:  # Buy Back (close a short position)
            if self.open_positions['short'] <= 0:
                env_logger.info(
                    "Attempted to close a short position when none were open.")
                return 0, self.portfolio_balance  # Return zero trade result and current balance
            trade_result = (
                self.entry_prices['short'] - current_price) * self.leverage
            self.open_positions['short'] -= 1
            self.position_size -= 1
        elif action == 4:  # Sell Back (close a long position)
            if self.open_positions['long'] <= 0:
                env_logger.info(
                    "Attempted to close a long position when none were open.")
                return 0, self.portfolio_balance  # Return zero trade result and current balance
            trade_result = (
                current_price - self.entry_prices['long']) * self.leverage
            self.open_positions['long'] -= 1
            self.position_size -= 1

        if action in [0, 2]:
            # Update trade count
            self.trade_count += 1

        # Calculate the transaction cost as a percentage of the trade value
        trade_value = current_price * self.position_size
        transaction_cost = trade_value * self.transaction_costs

        # Apply transaction costs
        trade_result -= transaction_cost

        # Update portfolio balance
        self.portfolio_balance += trade_result

        # Update max portfolio balance
        self.max_portfolio_balance = max(
            self.max_portfolio_balance, self.portfolio_balance)

        # Calculate and update drawdown
        if self.portfolio_balance < self.max_portfolio_balance:
            current_drawdown = (self.max_portfolio_balance -
                                self.portfolio_balance) / self.max_portfolio_balance
            self.max_drawdown = max(self.max_drawdown, current_drawdown)

        # Reset entry price if positions are closed
        if self.open_positions['long'] == 0:
            self.entry_prices['long'] = 0
        if self.open_positions['short'] == 0:
            self.entry_prices['short'] = 0

        # Update tracking after trade
        self.update_position_tracking(action, current_price)
        self.update_current_price(current_price)
        self.shortterm_pnl()
        self.longterm_pnl()

        return trade_result, self.portfolio_balance

    def spot_trading(self, action, current_price):
        """
        Update portfolio balance based on the action taken for spot trading.
        Actions: 0: Sell (or short sell), 1: Hold, 2: Buy, 5: Buy Back (close short position)
        Parameters:
        - action (int): Action taken.
        - current_price (float): Current stock price.
        Returns:
        - Tuple: Trade result, updated portfolio balance, max drawdown, max portfolio balance, and stocks held.
        """

        env_logger.info(
            f"Starting balance: ${self.portfolio_balance:.2f}, Stocks held: {self.stocks_held}")

        amount_to_trade = 0.1 * self.portfolio_balance
        trade_result: float = 0.0
        # No action taken for 'Hold'
        if action == 1:
            env_logger.info("Hold action taken. No change in portfolio.")
            return 0, self.portfolio_balance, self.max_drawdown, self.max_portfolio_balance, self.stocks_held

        elif action == 0:  # Sell or short sell
            if self.stocks_held > 0:  # Selling long position
                stocks_to_sell = min(
                    self.stocks_held, amount_to_trade / current_price)
                trade_result = stocks_to_sell * \
                    (current_price - self.entry_price['long'])
                self.stocks_held -= stocks_to_sell
                env_logger.info(
                    f"Selling {stocks_to_sell:.2f} stocks at ${current_price:.2f} each.")
            else:  # Short selling
                stocks_to_short = amount_to_trade / current_price
                self.stocks_held -= stocks_to_short  # Negative value for short positions
                self.entry_price['short'] = current_price
                env_logger.info(
                    f"Short selling {stocks_to_short:.2f} stocks at ${current_price:.2f} each.")

        elif action == 2:  # Buy
            stocks_to_buy = min(amount_to_trade / current_price,
                                self.portfolio_balance / current_price)
            self.stocks_held += stocks_to_buy
            self.entry_price['long'] = current_price if self.stocks_held > 0 else None
            env_logger.info(
                f"Buying {stocks_to_buy:.2f} stocks at ${current_price:.2f} each.")

        elif action == 3:  # Buy Back (closing short position)
            if self.stocks_held < 0:  # Has short positions
                stocks_to_buy_back = min(
                    abs(self.stocks_held), amount_to_trade / current_price)
                trade_result = stocks_to_buy_back * \
                    (self.entry_price['short'] - current_price)
                self.stocks_held += stocks_to_buy_back
                env_logger.info(
                    f"Buying back {stocks_to_buy_back:.2f} shorted stocks at ${current_price:.2f} each.")
            else:
                env_logger.info("No short positions to buy back.")

        if action in [0, 2]:
            # Update trade count
            self.trade_count += 1

        # Calculate transaction cost and update portfolio balance
        trade_value = current_price * \
            abs(stocks_to_sell if action == 0 else stocks_to_buy)
        transaction_cost = trade_value * self.transaction_costs
        trade_result -= transaction_cost
        self.portfolio_balance += trade_result

        # Update max portfolio balance and drawdown
        self.max_portfolio_balance = max(
            self.max_portfolio_balance, self.portfolio_balance)
        if self.portfolio_balance < self.max_portfolio_balance:
            current_drawdown = 1 - \
                (self.portfolio_balance / self.max_portfolio_balance)
            self.max_drawdown = max(self.max_drawdown, current_drawdown)

        env_logger.info(
            f"Updated balance: ${self.portfolio_balance:.2f}, Stocks held: {self.stocks_held}")

        # Update tracking after trade
        if action in [0, 2, 3]:
            self.update_position_tracking(action, current_price)

        self.update_current_price(current_price)
        self.shortterm_pnl()  # Update PnL after the trade
        self.longterm_pnl()

        return trade_result, self.portfolio_balance, self.max_drawdown, self.max_portfolio_balance, self.stocks_held

    def shortterm_pnl(self):
        if self.trading_mode == 'futures':
            trading_volume = self.position_size
        else:  # spot trading
            trading_volume = self.stocks_held

        pnl_long = (self.current_price -
                    self.entry_prices['long']) * trading_volume if trading_volume > 0 else 0
        pnl_short = (self.entry_prices['short'] - self.current_price) * \
            abs(trading_volume) if trading_volume < 0 else 0

        total_pnl = pnl_long + pnl_short
        pnl_percentage = (total_pnl / self.initial_balance) * \
            100 if self.initial_balance != 0 else 0
        return pnl_percentage

    def longterm_pnl(self):
        if self.trading_mode == 'futures':
            trading_volume = self.position_size
        else:  # spot trading
            trading_volume = self.stocks_held

        pnl = 0
        if trading_volume > 0 and self.entry_prices['long'] is not None:
            pnl += (self.current_price -
                    self.entry_prices['long']) * trading_volume
        elif trading_volume < 0 and self.entry_prices['short'] is not None:
            pnl += (self.entry_prices['short'] -
                    self.current_price) * abs(trading_volume)

        # Calculate PnL as a percentage of the initial balance
        pnl_percentage = (
            pnl / self.initial_balance) if self.initial_balance != 0 else 0

        self.pnl = pnl_percentage  # Update the pnl attribute to store the percentage
        return pnl_percentage

    def update_position_tracking(self, action, current_price):
        """
        Update tracking of open positions and entry prices.
        Parameters:
        - action (int): Action taken.
        - current_price (float): Current stock price.
        """
        # Handling long positions
        if action == 2:  # Buy
            self.entry_prices['long'] = current_price if self.open_positions['long'] == 0 else (
                (self.entry_prices['long'] * self.open_positions['long'] + current_price) /
                (self.open_positions['long'] + 1)
            )
            self.open_positions['long'] += 1

        elif action == 4:  # Sell Back (close long position)
            self.open_positions['long'] -= 1
            if self.open_positions['long'] == 0:
                self.entry_prices['long'] = 0

        # Handling short positions
        if action == 0:  # Sell (open short)
            self.entry_prices['short'] = current_price if self.open_positions['short'] == 0 else (
                (self.entry_prices['short'] * self.open_positions['short'] + current_price) /
                (self.open_positions['short'] + 1)
            )
            self.open_positions['short'] += 1

        elif action == 3:  # Buy Back (close short position)
            self.open_positions['short'] -= 1
            if self.open_positions['short'] == 0:
                self.entry_prices['short'] = 0

    def update_current_price(self, current_price):
        self.current_price = current_price

    def calculate_drawdown_penalty(self):
        return -1 * max(0, self.max_drawdown - self.drawdown_threshold)

    def calculate_trading_penalty(self):
        return -1 * max(0, self.trade_count - self.trade_limit) / self.trade_limit

    def execute_trade(self, action, current_price):
        if self.trading_mode == 'futures':
            return self.future_trading(action, current_price)
        elif self.trading_mode == 'spot':
            return self.spot_trading(action, current_price)
        else:
            raise ValueError("Invalid trading mode")
