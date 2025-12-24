
from .models import LearningRequest, LearningResponse
from . import analysis
from typing import List
import statistics

# --- Constants for adjustments ---
MAX_WEIGHT_ADJUSTMENT = 0.05
MAX_RISK_ADJUSTMENT = 0.005

def run_learning_cycle(request: LearningRequest) -> LearningResponse:
    """
    Runs the full learning and recommendation cycle.
    """
    trade_history = request.trade_history
    portfolio_metrics = request.portfolio_metrics
    config = request.config

    response = LearningResponse(learning_state="active")
    reasoning = []

    # 1. Agent Weight Adjustments
    agent_accuracies = analysis.analyze_agent_accuracy(trade_history)

    if agent_accuracies:
        median_accuracy = statistics.median(agent_accuracies.values())

        for agent_name, accuracy in agent_accuracies.items():
            current_weight = config.agent_weights.get(agent_name, 0.0)

            # Reward agents performing better than median
            if accuracy > median_accuracy and current_weight < 1.0:
                adjustment = min(MAX_WEIGHT_ADJUSTMENT, 1.0 - current_weight)
                response.agent_weight_adjustments[agent_name] = adjustment
            # Penalize agents performing worse than median
            elif accuracy < median_accuracy and current_weight > 0.0:
                adjustment = -min(MAX_WEIGHT_ADJUSTMENT, current_weight)
                response.agent_weight_adjustments[agent_name] = adjustment

        if response.agent_weight_adjustments:
             reasoning.append("Adjusting agent weights based on recent performance.")

    # 2. Risk Adjustments (Asymmetric)
    is_clustering = analysis.detect_drawdown_clustering(trade_history)

    # Decrease risk quickly
    if portfolio_metrics.max_drawdown > 0.15 or is_clustering:
        response.risk_adjustments["risk_per_trade"] = -MAX_RISK_ADJUSTMENT
        if is_clustering:
            reasoning.append("Drawdown clustering detected. Reducing risk.")
        else:
            reasoning.append("High drawdown detected. Reducing risk.")

    # Increase risk slowly and conservatively
    elif portfolio_metrics.win_rate > 0.6 and \
         portfolio_metrics.sharpe_ratio > 1.2 and \
         portfolio_metrics.max_drawdown < 0.1:
        response.risk_adjustments["risk_per_trade"] = MAX_RISK_ADJUSTMENT
        reasoning.append("Strong performance metrics observed. Cautiously increasing risk.")

    # 3. Bias Detection
    trend_bias = analysis.detect_trend_bias(trade_history)
    if abs(trend_bias) > 0.3: # Threshold for significant trend bias
        # We suggest an adjustment in the opposite direction of the bias
        response.strategy_bias["trend_following"] = -trend_bias * 0.1
        reasoning.append(f"Trend bias detected. Suggesting adjustment.")

    if analysis.detect_overtrading(trade_history):
        # Overtrading is a complex issue. For now, we'll just flag it.
        # A more advanced agent might suggest specific guardrails.
        reasoning.append("Overtrading detected: high frequency of trades with low win rate.")

    confirmation_biases = analysis.detect_confirmation_bias(agent_accuracies, config.agent_weights)
    for agent_name, has_bias in confirmation_biases.items():
        if has_bias:
            reasoning.append(f"Confirmation bias detected for agent '{agent_name}': high weight but low accuracy.")

    response.reasoning = reasoning
    return response
