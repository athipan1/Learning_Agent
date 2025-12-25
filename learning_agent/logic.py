
from .models import LearningRequest, LearningResponse, Trade, PortfolioMetrics
from . import analysis
from typing import List
import statistics

def calculate_confidence_score(
    trade_history: List[Trade],
    portfolio_metrics: PortfolioMetrics
) -> float:
    """
    Calculates the confidence score based on a hybrid heuristic.
    """
    trade_count = len(trade_history)

    # Factor 1: Trade Count
    trade_count_factor = min(1.0, trade_count / 100)

    # Factor 2: Performance Consistency
    pnl_pcts = [trade.pnl_pct for trade in trade_history]
    if len(pnl_pcts) > 1:
        performance_consistency = 1 - statistics.stdev(pnl_pcts)
    else:
        performance_consistency = 1.0

    # Factor 3: Drawdown Penalty
    if portfolio_metrics.max_drawdown > 0.2:
        drawdown_penalty = 0.7
    elif portfolio_metrics.max_drawdown > 0.1:
        drawdown_penalty = 0.9
    else:
        drawdown_penalty = 1.0

    confidence = (
        trade_count_factor *
        performance_consistency *
        drawdown_penalty
    )

    return max(0.0, min(1.0, confidence))

# --- Constants for adjustments ---
MAX_WEIGHT_ADJUSTMENT = 0.05
MAX_RISK_ADJUSTMENT = 0.005

def run_learning_cycle(request: LearningRequest) -> LearningResponse:
    """
    Runs the full learning and recommendation cycle.
    """
    trade_history = request.trade_history
    portfolio_metrics = request.portfolio_metrics
    config = request.current_config
    price_history = request.price_history

    response = LearningResponse(learning_state="active")
    reasoning = []

    # 1. Analyze Market Regime
    market_regime = analysis.analyze_market_regime(price_history)

    # 2. Agent Weight Adjustments
    agent_accuracies = analysis.analyze_agent_accuracy(trade_history)
    if agent_accuracies:
        median_accuracy = statistics.median(agent_accuracies.values())

        for agent_name, accuracy in agent_accuracies.items():
            current_weight = config.agent_weights.get(agent_name, 0.0)

            if accuracy > median_accuracy and current_weight < 1.0:
                adjustment = min(MAX_WEIGHT_ADJUSTMENT, 1.0 - current_weight)
                response.policy_deltas.agent_weights[agent_name] = adjustment
            elif accuracy < median_accuracy and current_weight > 0.0:
                adjustment = -min(MAX_WEIGHT_ADJUSTMENT, current_weight)
                response.policy_deltas.agent_weights[agent_name] = adjustment

        if response.policy_deltas.agent_weights:
            reasoning.append("Adjusting agent weights based on recent performance.")

    # 3. Risk Adjustments
    is_clustering = analysis.detect_drawdown_clustering(trade_history)
    if portfolio_metrics.max_drawdown > 0.15 or is_clustering:
        response.policy_deltas.risk.risk_per_trade = -MAX_RISK_ADJUSTMENT
        if is_clustering:
            reasoning.append("Drawdown clustering detected. Reducing risk.")
        else:
            reasoning.append("High drawdown detected. Reducing risk.")

    elif portfolio_metrics.win_rate > 0.6 and portfolio_metrics.profit_factor > 1.2 and portfolio_metrics.max_drawdown < 0.1:
        response.policy_deltas.risk.risk_per_trade = MAX_RISK_ADJUSTMENT
        reasoning.append("Strong performance metrics observed. Cautiously increasing risk.")

    if market_regime == "volatile":
        response.policy_deltas.risk.max_position_pct = -0.05
        reasoning.append("Volatile market detected. Reducing max position size.")

    # 4. Bias Detection
    trend_bias = analysis.detect_trend_bias(trade_history)
    if abs(trend_bias) > 0.3:
        if trend_bias > 0 and market_regime != "trending":
            response.policy_deltas.strategy_bias["preferred_regime"] = "ranging"
            reasoning.append("Long bias underperforming in non-trending market.")
        elif trend_bias < 0 and market_regime == "trending":
            response.policy_deltas.strategy_bias["preferred_regime"] = "trending"
            reasoning.append("Short bias underperforming in trending market.")

    if analysis.detect_overtrading(trade_history):
        reasoning.append("Overtrading detected: high frequency of trades with low win rate.")

    confirmation_biases = analysis.detect_confirmation_bias(agent_accuracies, config.agent_weights)
    for agent_name, has_bias in confirmation_biases.items():
        if has_bias:
            reasoning.append(f"Confirmation bias detected for agent '{agent_name}': high weight but low accuracy.")

    # 5. Calculate Confidence Score
    response.confidence = calculate_confidence_score(trade_history, portfolio_metrics)

    response.reasoning = reasoning
    return response
