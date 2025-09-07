import os
import pickle

import numpy as np
import pandas as pd

import model.reinforcement.indicators as ind
import util.loggers as loggers
from model.reinforcement.rl_env.env_uttils import (ActionSpace,
                                                   DynamicFeatureSelector,
                                                   ObservationSpace,
                                                   TradingEnvironment)
from util.utils import convert_df, features, load_config, tradex_features

logger = loggers.setup_loggers()
env_logger = logger['env']
rl_logger = logger['rl']


# Constants
EARLY_STOP_REWARD_THRESHOLD = -5.0


class MultiAgentEnvironment:
    def __init__(self, num_agents, *args, **kwargs):
        self.num_agents = num_agents
        self.single_agent_mode = num_agents == 1  # Check if only one agent
        self.agents = [Environment(*args, **kwargs) for _ in range(num_agents)]

        # Action space and observation space are set based on the first agent
        self.action_space = self.agents[0].action_space
        self.observation_space = self.agents[0].observation_space
        self.look_back = self.agents[0].look_back

    def reset(self):
        if self.single_agent_mode:
            return self.agents[0].reset()  # Single agent reset
        else:
            return [agent.reset() for agent in self.agents]

    def step(self, actions):
        if self.single_agent_mode:
            # Single action for single agent
            return self.agents[0].step(actions)
        else:
            # Handle multiple agents
            states, rewards, infos, dones = [], [], [], []
            for agent, action in zip(self.agents, actions):
                state, reward, info, done = agent.step(action)
                states.append(state)
                rewards.append(reward)
                infos.append(info)
                dones.append(done)
            return states, rewards, infos, dones

    # No changes needed for these methods as they already reference the first agent
    def get_action_space(self):
        return self.action_space

    def get_observation_space(self):
        return self.observation_space

    def get_look_back(self):
        return self.look_back

    def update_and_adjust_features(self):
        [agent.update_and_adjust_features() for agent in self.agents]


class Environment:
    """The main trading environment."""

    def __init__(self, symbol, features, limit, time, actions, min_acc):
        self.config = load_config()
        self.symbol, self.features = symbol, features
        self.limit, self.time = limit, time
        self.action_space = ActionSpace(actions)
        self.min_accuracy = min_acc

        self._initialize_environment()

    def _initialize_environment(self):
        self._set_initial_parameters()
        self.tradingEnv = TradingEnvironment(
            self.portfolio_balance, 5, 0.05, self.trade_limit, self.threshold, self.symbol)

        self.original_data = self._get_data()
        self.env_data = self.original_data
        self._create_features()
        self._setup_observation_space()

    def _set_initial_parameters(self):
        self.last_price, self.bar = None, 0
        self.trade_limit, self.threshold = 5, 0.2
        self.last_price, self.last_accuracy = None, None
        # int(self.config['Env']['patience'])
        self.high_acc_counter, self.patience = 0, 20
        self.wait, self.high_acc_threshold = 0, 5
        self.stocks_held = 0.0
        self.max_portfolio_balance, self.total_reward = 0, 0
        self.pnl, self.total_pnl = 0, 0
        self.accuracy, self.holding_time = 0, 0
        self.correct_actions, self.last_action = 0, None
        self.trade_result = 0
        # List to store recent trade results
        self.recent_trade_results = [0] * 5
        self.current_risk_level = 0     # Initialize current risk level
        self.market_volatility = 0      # Initialize market volatility
        # float(self.config['Tradingenv']['portfolio_balance'])
        self.portfolio_balance = 10_000

    def _setup_observation_space(self):
        if not all(feature in self.env_data.columns for feature in self.features):
            # Handle the missing columns case
            env_logger.error("Missing required features in env_data.")
            return

        # Calculate the original number of features in the observation space
        num_original_features = len(self.env_data[self.features].columns)

        # Calculate the size of the new elements added to the state
        num_recent_trade_results = 5  # Assuming we store the last 5 trade results
        num_additional_metrics = 2    # For current risk level and market volatility

        # Calculate the total number of features in the enhanced observation space
        total_features = num_original_features + \
            num_recent_trade_results + num_additional_metrics

        # Update the observation space to reflect the new shape
        self.observation_space = ObservationSpace(total_features)
        self.look_back = 1
        env_logger.info(
            f"Environment initialized with symbol {self.symbol}, features {self.features}, and enhanced observation space. Observation space shape: {self.observation_space.shape}, lookback: {self.look_back}")

    def _get_data(self):
        # self.config['Path']['2020_30m_data']
        pickle_file_name = 'data/pickle/all/30m_data_all.pkl'

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

        # float(self.config['Data']['percentage'])
        percentage_to_keep = 15 / 100.0
        rows_to_keep = int(len(data) * percentage_to_keep)
        data = data.head(rows_to_keep)

        rl_logger.info(f'Dataframe shape: {data.shape}')
        return data

    def _create_features(self):
        if self.original_data.empty:
            env_logger.error("Original data is empty.")
            return

        self.original_data['volume'] = self.original_data['volume'].astype(
            float).round(6)
        feature = features(self.original_data.copy())
        processed_features = tradex_features(
            self.symbol, self.original_data.copy())

        if feature.empty or processed_features.empty:
            env_logger.error(
                "Feature generation failed, resulting in empty data.")
            return

        self.original_data = pd.concat([processed_features, feature], axis=1)
        self.original_data.dropna(inplace=True)  # Remove rows with NaN values

        # Initialize DynamicFeatureSelector
        self.feature_selector = DynamicFeatureSelector(self.original_data)

        self.original_data['last_price'] = self.original_data['close'].shift(-1)
        self.original_data['portfolio_balance'] = self.portfolio_balance

        if self.env_data.isnull().values.any():
            env_logger.error("NaN values detected in environment data.")
            return

        # Ensure there are no division by zero or invalid operations
        self.original_data['r'] = np.log(
            self.original_data['close'] / self.original_data['close'].shift(5).replace(0, np.nan))

        if self.original_data.isnull().values.any():
            self.original_data.dropna(inplace=True)

        self.env_data = self.original_data[self.features].copy()
        self.env_data['r'] = self.original_data['r'].bfill()

        self.env_data['d'] = self.compute_action(self.original_data)
        # Forward fill to handle NaNs
        self.env_data['d'] = self.env_data['d']

        # Normalize data, handle any NaNs that may arise from normalization
        self.env_data = (self.env_data - self.env_data.mean()
                         ) / self.env_data.std()
        # Replace any NaNs resulted from normalization
        self.env_data.fillna(0, inplace=True)

        self.env_data['action'] = 0

        env_logger.info(self.original_data)
        env_logger.info(self.env_data)

        env_logger.info("Features created successfully.")

    def update_and_adjust_features(self):
        # This method should be called periodically to update and adjust features
        # Example: You can call this method after every trading action or at regular intervals

        # Evaluate feature performance based on your criteria (profitability, accuracy, etc.)
        for feature in self.feature_selector.get_features():
            performance_impact = self.feature_selector.evaluate_feature_performance(
                feature, self.original_data,  'r')
            self.feature_selector.update_feature_performance(
                feature, performance_impact)

        # Adjust features based on performance
        self.feature_selector.adjust_features()

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
        # Extract the original state slice based on existing features
        original_state_slice = self.env_data[self.features].iloc[bar -
                                                                 self.look_back:bar].values.flatten()

        # Combine all elements to form the complete state
        complete_state = np.concatenate([original_state_slice, self.recent_trade_results, [
                                        self.current_risk_level, self.market_volatility]])

        # Check for large values and normalize if necessary
        if np.any(np.abs(complete_state) > np.finfo(np.float32).max):
            env_logger.warning(
                "Overflow detected in state values, applying normalization.")
            complete_state = (
                complete_state - np.mean(complete_state)) / np.std(complete_state)

        return np.array(complete_state, dtype=np.float32)

    def step(self, action):
        env_logger.info(f"STEP: Action taken: {action}")

        # Validate index range
        if self.bar >= len(self.original_data):
            env_logger.error("Index out of range. Returning default state.")
            return np.zeros_like(self.env_data.iloc[0], 0, {}, True)

        df_row_raw = self.original_data.iloc[self.bar]
        env_row_raw = self.env_data.iloc[self.bar].copy()

        self.last_price = df_row_raw['close'] if pd.notna(
            df_row_raw['close']) else self.last_price
        self.current_price = df_row_raw['close'] if pd.notna(
            df_row_raw['close']) else self.last_price

        self.execute_trade(action, self.current_price)
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
        self.trade_result, self.portfolio_balance = self.tradingEnv.execute_trade(
            action, current_price)

    def _get_conditions(self, df_row):
        """Helper method to centralize the conditions logic."""
        current_bar_index = self.bar

        # Adjusting to use self.original_data
        super_buy = (df_row['dots'] == 1) & (df_row['l_wave'] >= -50)
        super_sell = (df_row['dots'] == -1) & (df_row['l_wave'] >= 50)
        low_volatility = (df_row['rsi14'] >= 45) & (df_row['rsi14'] <= 55)
        strong_upward_movement = df_row['rsi14'] > 70
        strong_downward_movement = df_row['rsi14'] < 30
        going_up_condition = (df_row['close'] > df_row['last_price']) & (
            df_row['close'] > df_row['ema_200']) & (df_row['rsi40'] > 50)
        going_down_condition = (df_row['close'] < df_row['last_price']) & (
            df_row['close'] < df_row['ema_200']) & (df_row['rsi40'] < 50)

        # Use parentheses for each condition to avoid ambiguity
        strong_buy_signal: bool = bool(strong_upward_movement & ~ind.is_increasing_trend(
            self.original_data, self.bar))
        strong_sell_signal: bool = (strong_downward_movement & ~ind.is_increasing_trend(
            self.original_data, self.bar))

        # SHORT
        short_stochastic_signal = ind.short_stochastic_condition(
            self.original_data, current_bar_index)
        short_bollinger_outside = ind.short_bollinger_condition(
            self.original_data, current_bar_index)
        # LONG ONlY
        long_stochastic_signal = ind.long_stochastic_condition(
            self.original_data, current_bar_index)
        long_bollinger_outside = ind.long_bollinger_condition(
            self.original_data, current_bar_index)
        macd_buy = ind.macd_condition(self.original_data, current_bar_index)
        high_volatility = ind.atr_condition(
            self.original_data, current_bar_index)
        adx_signal = ind.adx_condition(self.original_data, current_bar_index)
        psar_signal = ind.parabolic_sar_condition(
            self.original_data, current_bar_index)
        cdl_pattern = ind.cdl_pattern(self.original_data, current_bar_index)
        volume_break = ind.volume_breakout(
            self.original_data, current_bar_index)
        resistance_break_signal = ind.resistance_break(
            self.original_data, current_bar_index)

        return short_stochastic_signal, short_bollinger_outside, long_stochastic_signal, long_bollinger_outside, low_volatility, going_up_condition, going_down_condition, strong_buy_signal, strong_sell_signal, super_buy, super_sell, macd_buy, high_volatility, adx_signal, psar_signal, cdl_pattern, volume_break, resistance_break_signal

    def calculate_reward(self, action, df_row_raw):
        # Here, add logic to compute the reward
        market_condition_reward = self.compute_market_condition_reward(
            action, df_row_raw)
        financial_outcome_reward = self.compute_financial_outcome_reward(
            action)
        risk_adjusted_reward = ind.compute_risk_adjusted_reward(
            self.original_data['r'])  # assuming self.env_data['r'] holds returns history
        drawdown_penalty = self.tradingEnv.calculate_drawdown_penalty()
        trading_penalty = self.tradingEnv.calculate_trading_penalty()

        return self.combined_reward(market_condition_reward, financial_outcome_reward,
                                    risk_adjusted_reward, drawdown_penalty, trading_penalty)

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

        # Reward for holding if in profit
        if action == 1 and pnl > 0:
            holding_reward = ind.calculate_holding_reward(
                self.holding_time, pnl)
            reward = holding_reward
        elif action == 1 and pnl < 0:  # Penalize for holding if in loss
            holding_penalty = ind.calculate_holding_penalty(
                self.holding_time, pnl)
            reward = holding_penalty
        else:
            # For buy or sell actions, the reward is directly based on PnL
            reward = pnl

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

        # Update recent trade results
        self.recent_trade_results.append(self.trade_result)
        if len(self.recent_trade_results) > 5:  # Keep only the last 5 results
            self.recent_trade_results.pop(0)

        # Update current risk level
        self.current_risk_level = ind.calculate_current_risk_level(
            self.recent_trade_results)

        # Update market volatility
        self.market_volatility = ind.calculate_market_volatility(
            self.original_data, self.bar)

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

    def get_feature_names(self):
        return self.env_data.columns.tolist()
