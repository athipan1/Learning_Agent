
from typing import List, Dict
from .models import Trade

def analyze_agent_accuracy(trade_history: List[Trade], lookback_window: int = 50) -> Dict[str, float]:
    """
    Analyzes the accuracy of each agent over a given lookback window.

    An agent's signal is considered "correct" if it aligns with a profitable
    trade action or advises against an unprofitable one. For example, a 'buy'
    signal is correct if the trade's PnL is positive. A 'sell' signal is
    also correct if the trade's PnL is positive (as it was a profitable short).
    - Hold signals are ignored.
    """
    agent_correct_counts = {}
    agent_signal_counts = {}

    # Consider only the most recent trades
    recent_trades = trade_history[-lookback_window:]

    for trade in recent_trades:
        is_profitable = trade.pnl_pct > 0
        for agent_name, signal in trade.agents.items():
            if agent_name not in agent_signal_counts:
                agent_signal_counts[agent_name] = 0
                agent_correct_counts[agent_name] = 0

            # We only measure accuracy on 'buy' or 'sell' signals
            if signal in ["buy", "sell"]:
                agent_signal_counts[agent_name] += 1
                # A signal is correct if it matches a profitable trade's action,
                # or is the opposite of an unprofitable trade's action.
                signal_matches_action = (signal == trade.action)
                if (is_profitable and signal_matches_action) or \
                   (not is_profitable and not signal_matches_action):
                    agent_correct_counts[agent_name] += 1

    agent_accuracies = {}
    for agent_name, total_signals in agent_signal_counts.items():
        if total_signals > 0:
            accuracy = agent_correct_counts[agent_name] / total_signals
            agent_accuracies[agent_name] = accuracy

    return agent_accuracies

def detect_trend_bias(trade_history: List[Trade], lookback_window: int = 50) -> float:
    """
    Detects if the system is overly biased towards trend-following.
    This can be inferred by analyzing the performance of consecutive trades
    in the same direction.
    """
    recent_trades = trade_history[-lookback_window:]

    if len(recent_trades) < 3: # Need at least 3 trades to detect a trend
        return 0.0

    long_trades = [t for t in recent_trades if t.action == 'buy']
    short_trades = [t for t in recent_trades if t.action == 'sell']

    if not long_trades or not short_trades:
        return 0.0 # Not enough data for comparison

    long_win_rate = sum(1 for t in long_trades if t.pnl_pct > 0) / len(long_trades)
    short_win_rate = sum(1 for t in short_trades if t.pnl_pct > 0) / len(short_trades)

    # A significant difference in win rates suggests a bias
    # The return value represents the magnitude of the bias
    return long_win_rate - short_win_rate

def detect_overtrading(trade_history: List[Trade], lookback_window: int = 50, trade_frequency_threshold: int = 10, win_rate_threshold: float = 0.4) -> bool:
    """
    Detects overtrading by checking for a high frequency of trades with a low
    win rate over a given lookback window.
    """
    recent_trades = trade_history[-lookback_window:]

    if len(recent_trades) < trade_frequency_threshold:
        return False

    win_count = sum(1 for trade in recent_trades if trade.pnl_pct > 0)
    current_win_rate = win_count / len(recent_trades)

    return current_win_rate < win_rate_threshold

def detect_drawdown_clustering(trade_history: List[Trade], lookback_window: int = 15, heavy_loss_threshold: float = -0.02, cluster_threshold: int = 3) -> bool:
    """
    Detects the clustering of significant losses in a recent, short window.
    """
    recent_trades = trade_history[-lookback_window:]
    heavy_losses = [
        trade for trade in recent_trades if trade.pnl_pct < heavy_loss_threshold
    ]

    return len(heavy_losses) >= cluster_threshold

def detect_confirmation_bias(agent_accuracies: Dict[str, float], agent_weights: Dict[str, float]) -> Dict[str, bool]:
    """
    Detects confirmation bias where a highly-weighted agent has low accuracy.
    """
    # A simple approach: flag agents with high weight but low accuracy
    # More sophisticated logic could be added later (e.g., comparing to median accuracy)

    bias_detected = {}
    # Find the median accuracy to establish a baseline
    if not agent_accuracies:
        return {}

    median_accuracy = sorted(agent_accuracies.values())[len(agent_accuracies) // 2]

    for agent_name, weight in agent_weights.items():
        accuracy = agent_accuracies.get(agent_name)

        # Flag if weight is high but accuracy is below median
        if accuracy is not None and weight > 0.4 and accuracy < median_accuracy:
            bias_detected[agent_name] = True
        else:
            bias_detected[agent_name] = False

    return bias_detected
