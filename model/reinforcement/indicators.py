import numpy as np

from util.rl_util import sigmoid


def compute_market_condition_reward(action, df_row, bar, data):
    """Compute and return the reward based on the current action, data row, and current price."""
    short_stochastic_signal, short_bollinger_outside, long_stochastic_signal, long_bollinger_outside, low_volatility, going_up_condition, going_down_condition, strong_buy_signal, strong_sell_signal, super_buy, super_sell, macd_buy, high_volatility, adx_signal, psar_signal, cdl_pattern, volume_break, resistance_break_signal = get_conditions(
        df_row, bar, data)

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

    def get_conditions(df_row, bar, original_data):
        """Helper method to centralize the conditions logic."""
        current_bar_index = bar

        # Adjusting to use original_data
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

        strong_buy_signal = strong_upward_movement & ~is_increasing_trend(original_data,
                                                                          current_bar_index)
        strong_sell_signal = strong_downward_movement & ~is_increasing_trend(original_data,
                                                                             current_bar_index)  # ~ is the element-wise logical NOT

        # SHORT
        short_stochastic_signal = ~short_stochastic_condition(
            original_data, current_bar_index)
        short_bollinger_outside = ~short_bollinger_condition(
            original_data, current_bar_index)
        # LONG ONlY
        long_stochastic_signal = ~long_stochastic_condition(
            original_data, current_bar_index)
        long_bollinger_outside = ~long_bollinger_condition(
            original_data, current_bar_index)
        macd_buy = ~macd_condition(original_data, current_bar_index)
        high_volatility = ~atr_condition(
            original_data, current_bar_index)
        adx_signal = ~adx_condition(original_data, current_bar_index)
        psar_signal = ~parabolic_sar_condition(
            original_data, current_bar_index)
        cdl_pattern = ~cdl_pattern(original_data, current_bar_index)
        volume_break = ~volume_breakout(
            original_data, current_bar_index)
        resistance_break_signal = ~resistance_break(
            original_data, current_bar_index)

        return short_stochastic_signal, short_bollinger_outside, long_stochastic_signal, long_bollinger_outside, low_volatility, going_up_condition, going_down_condition, strong_buy_signal, strong_sell_signal, super_buy, super_sell, macd_buy, high_volatility, adx_signal, psar_signal, cdl_pattern, volume_break, resistance_break_signal


# SHORT ACTION
def short_bollinger_condition(df, current_bar_index):
    """Check for a short action signal using Bollinger Bands."""
    if current_bar_index < 21:
        return False
    bollinger = df.ta.bbands().iloc[current_bar_index]
    return df.iloc[current_bar_index]['close'] > bollinger['BBU_5_2.0']


def short_stochastic_condition(df, current_bar_index):
    """Check for a bearish crossover in stochastic oscillator for a short action."""
    if current_bar_index < 15:
        return False
    stochastic = df.ta.stoch().iloc[current_bar_index]
    return stochastic['STOCHk_14_3_3'] < stochastic['STOCHd_14_3_3'] and stochastic['STOCHk_14_3_3'] > 80


# LONG ACTION

def long_stochastic_condition(df, current_bar_index):
    """Check for a bullish crossover in stochastic oscillator for a long action."""
    if current_bar_index < 15:
        return False
    stochastic = df.ta.stoch().iloc[current_bar_index]
    return stochastic['STOCHk_14_3_3'] > stochastic['STOCHd_14_3_3'] and stochastic['STOCHk_14_3_3'] < 20


def long_bollinger_condition(df, current_bar_index):
    """Check for a long action signal using Bollinger Bands."""
    if current_bar_index < 21:
        return False
    bollinger = df.ta.bbands().iloc[current_bar_index]
    return df.iloc[current_bar_index]['close'] < bollinger['BBL_5_2.0']


def macd_condition(df, current_bar_index):
    """Checks if the MACD line is above the signal line."""
    if current_bar_index < 15:
        return False
    macd = df.ta.macd(
        fast=14, slow=28, signal=9).iloc[current_bar_index]
    return macd['MACD_14_28_9'] > macd['MACDs_14_28_9']


def atr_condition(df, current_bar_index):
    """Checks if the current ATR is greater than the mean ATR."""
    if current_bar_index < 15:
        return False
    atr = df.ta.atr().iloc[current_bar_index]
    return atr > atr.mean()


def adx_condition(df, current_bar_index):
    """Checks if the strength of the trend is strong with ADX > 25."""
    if current_bar_index < 15:
        return False
    adx = df.ta.adx().iloc[current_bar_index]
    return adx['ADX_14'] > 25


def parabolic_sar_condition(df, current_bar_index):
    """Checks if the close price is above the Parabolic SAR."""
    if current_bar_index < 21:
        return False
    psar = df.ta.psar().iloc[current_bar_index]
    return df.iloc[current_bar_index]['close'] > psar


def cdl_pattern(df, current_bar_index):
    """Checks for the presence of a cdl_pattern candlestick pattern."""
    if current_bar_index < 2:
        return False
    cdl_pattern = df.ta.cdl_pattern(
        name=["doji", "hammer"]).iloc[current_bar_index]
    return cdl_pattern != 0


def volume_breakout(df, current_bar_index):
    """Checks for the presence of a cdl_pattern candlestick pattern."""
    if current_bar_index < 21:
        return False
    # Average volume over the last 20 bars.
    avg_volume = df[
        'volume'].iloc[current_bar_index-20:current_bar_index].mean()
    # Current volume is 150% of the average volume.
    return df.iloc[current_bar_index]['volume'] > 1.5 * avg_volume


def resistance_break(df, current_bar_index):
    """Checks for the presence of a cdl_pattern candlestick pattern."""
    if current_bar_index < 21:
        return False
    # Maximum high over the last 20 bars.
    resistance = df[
        'high'].iloc[current_bar_index-20:current_bar_index].max()

    # Current close is above the resistance.
    return df.iloc[current_bar_index]['close'] > resistance


def is_increasing_trend(df, current_bar_index):
    """Checks if there's an increasing trend for the past 3 bars."""
    if current_bar_index < 2:
        return False
    return (df.iloc[current_bar_index]['close'] > df.iloc[current_bar_index - 1]['close']) and (df.iloc[current_bar_index - 1]['close'] > df.iloc[current_bar_index - 2]['close'])


# ========================================================


def calculate_holding_reward(holding_time, pnl):
    # Define a reward function based on holding time and PnL
    scaling_factor = pnl / 10000  # Adjust the scaling factor as needed
    return holding_time * pnl * scaling_factor

    # time-dependingScaling, s= 0.01 * (holding_time/ 10)
    # Performance-Based Scaling, s= pnl / 10000
    # Linear, s= 0.01


def is_correct_action(taken_action, optimal_action):
    """Compare taken action with the optimal action."""
    return optimal_action


def get_optimal_action(reward):
    """Determine the optimal action for a given state."""
    # Implement the logic to determine the optimal action
    # This can be environment specific
    if reward > 0.5:
        optimal_action = True
    else:
        optimal_action = False

    return optimal_action


# RISK REWARD

def compute_risk_adjusted_reward(returns_history):
    """
    Compute the reward based on action and risk.

    Parameters:
    - returns_history: Historical returns for risk calculation.

    Returns:
    - Calculated reward.
    """
    # Calculate Sharpe Ratio for risk-adjusted return
    sharpe_ratio = calculate_sharpe_ratio(returns_history)
    sortino_ratio = calculate_sortino_ratio(returns_history)

    # Combine profit/loss and risk-adjusted return
    # Here you can decide how to weigh these components
    reward = sharpe_ratio + sortino_ratio  # Simple additive model as an example

    # Scale the combined reward to be between 0 and 1
    normalized_reward = sigmoid(reward)

    return normalized_reward


def calculate_sortino_ratio(returns, risk_free_rate=0.02, data_frequency='daily'):
    """
    Compute the reward based on action and risk.

    Parameters:
    - returns_history: Historical returns for risk calculation.

    Returns:
    - Calculated reward.
    """

    # Convert annual risk-free rate to the correct frequency
    if data_frequency == 'daily':
        risk_free_rate = (1 + risk_free_rate) ** (1/252) - 1
    elif data_frequency == 'monthly':
        risk_free_rate = (1 + risk_free_rate) ** (1/12) - 1

    excess_returns = returns - risk_free_rate
    negative_returns = excess_returns[excess_returns < 0]

    downside_std_dev = negative_returns.std()

    if downside_std_dev > 0:
        return excess_returns.mean() / downside_std_dev
    else:
        return 0  # Avoid division by zero


def calculate_sharpe_ratio(returns, risk_free_rate=0.02, data_frequency='daily'):
    """
    Calculate the Sharpe Ratio for the given returns.

    Parameters:
    - returns: Array or list of returns.
    - risk_free_rate: Risk-free rate of return.

    Returns:
    - Sharpe Ratio.
    """

    # Convert annual risk-free rate to the correct frequency
    if data_frequency == 'daily':
        risk_free_rate = (1 + risk_free_rate) ** (1/252) - \
            1  # Assuming 252 trading days in a year
    elif data_frequency == 'monthly':
        risk_free_rate = (1 + risk_free_rate) ** (1/12) - 1
    # Other frequencies can be added as needed

    excess_returns = returns - risk_free_rate
    std_dev = excess_returns.std()

    if std_dev > 0:
        return excess_returns.mean() / std_dev
    else:
        return 0  # Avoid division by zero


def calculate_current_risk_level(recent_trade_results):
    # Implement your logic to calculate the current risk level
    # Example: Based on the volatility of the recent trade results
    return np.std(recent_trade_results)


def calculate_market_volatility(original_data, current_bar_index):
    # Define the window size for volatility calculation
    window_size = 20

    # Ensure that we have enough data to calculate volatility
    if current_bar_index >= window_size:
        # Slice the DataFrame to get the last 'window_size' rows up to the current index
        recent_data = original_data.iloc[max(
            0, current_bar_index - window_size):current_bar_index]

        # Calculate and return the standard deviation of the 'close' column
        return recent_data['close'].std()
    else:
        return 0
