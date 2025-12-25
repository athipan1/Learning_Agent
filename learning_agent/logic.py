
from .models import LearningRequest, LearningResponse
from . import analysis
from typing import List
import statistics

# --- Constants for adjustments ---
MAX_WEIGHT_ADJUSTMENT = 0.10
MAX_RISK_PER_TRADE_ADJUSTMENT = 0.005
MAX_POSITION_PCT_ADJUSTMENT = 0.10

def calculate_confidence_score(
    performance_metrics: dict,
    trade_history: list
) -> float:
    """
    Calculates the confidence score based on performance metrics.
    """
    win_rate = performance_metrics["win_rate"]
    max_drawdown = performance_metrics["max_drawdown"]

    # Factor 1: Win Rate Score
    win_rate_score = win_rate / 0.7  # Target a 70% win rate for a score of 1.0

    # Factor 2: Drawdown Penalty
    drawdown_penalty = 1.0 - (max_drawdown / 0.5)  # Penalize linearly up to 50% drawdown

    # Factor 3: Consistency (variance of returns)
    pnl_pcts = [trade.pnl_pct for trade in trade_history]
    if len(pnl_pcts) > 1:
        consistency = 1 - statistics.stdev(pnl_pcts)
    else:
        consistency = 1.0

    # Combine factors
    confidence = (win_rate_score * 0.4) + (drawdown_penalty * 0.4) + (consistency * 0.2)

    return max(0.0, min(1.0, confidence))

def run_learning_cycle(request: LearningRequest) -> LearningResponse:
    """
    Runs the full learning and recommendation cycle.
    """
    response = LearningResponse(learning_state="active", learning_mode="macro")
    reasoning = []

    # 1. Analyze Performance
    performance_metrics = analysis.calculate_performance_metrics(request.trade_history)
    agent_accuracies = analysis.analyze_agent_accuracy(request.trade_history)

    # 2. Calculate Confidence Score
    confidence = calculate_confidence_score(performance_metrics, request.trade_history)
    response.confidence_score = confidence

    # If confidence is too low, recommend no changes
    if confidence < 0.6:
        response.reasoning.append("Confidence score is below threshold (0.6). Recommending no policy changes to ensure stability.")
        return response

    # 3. Agent Weight Adjustments (Zero-Sum)
    if len(agent_accuracies) > 1:
        best_agent = max(agent_accuracies, key=agent_accuracies.get)
        worst_agent = min(agent_accuracies, key=agent_accuracies.get)

        if best_agent != worst_agent:
            response.policy_deltas.agent_weights[best_agent] = MAX_WEIGHT_ADJUSTMENT
            response.policy_deltas.agent_weights[worst_agent] = -MAX_WEIGHT_ADJUSTMENT
            reasoning.append(f"Reallocating weight from underperforming agent '{worst_agent}' to outperforming agent '{best_agent}'.")

    # 4. Risk Adjustments
    if performance_metrics["max_drawdown"] > 0.20:
        response.policy_deltas.risk["risk_per_trade"] = -MAX_RISK_PER_TRADE_ADJUSTMENT
        reasoning.append(f"Max drawdown of {performance_metrics['max_drawdown']:.2%} exceeds threshold (20%). Reducing risk per trade.")

    # 5. Guardrail Recommendations
    if performance_metrics["max_drawdown"] > 0.25:
        response.policy_deltas.guardrails["max_total_drawdown_pct"] = 0.20
        reasoning.append("High drawdown detected. Recommending a max total drawdown guardrail of 20%.")

    # Analyze regime performance
    regime_performance = {}
    for trade in request.trade_history:
        if trade.market_regime not in regime_performance:
            regime_performance[trade.market_regime] = []
        regime_performance[trade.market_regime].append(trade.pnl_pct)

    for regime, pnl_list in regime_performance.items():
        if sum(pnl_list) / len(pnl_list) < 0: # Negative expectancy
            response.policy_deltas.strategy_bias["avoid_regime"] = [regime]
            reasoning.append(f"Negative expectancy detected in {regime} markets. Recommending to avoid this regime.")

    response.reasoning = reasoning
    return response
