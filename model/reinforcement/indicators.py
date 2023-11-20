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


def calculate_sharpe_ratio(returns, risk_free_rate=0):
    """
    Calculate the Sharpe Ratio for the given returns.

    Parameters:
    - returns: Array or list of returns.
    - risk_free_rate: Risk-free rate of return.

    Returns:
    - Sharpe Ratio.
    """
    excess_returns = returns - risk_free_rate
    return excess_returns.mean() / excess_returns.std()


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
