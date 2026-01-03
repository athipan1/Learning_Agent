
from .models import LearningRequest, LearningResponse, PolicyDeltas, Trade
from typing import List, Dict
import numpy as np
from collections import defaultdict

# --- Constants for Asset-Aware Learning ---
ASSET_MIN_TRADES_WARMUP = 10
MAX_DRAWDOWN_THRESHOLD = 0.08
CONSECUTIVE_LOSS_THRESHOLD = 3
RECENT_TRADES_WINDOW = 10
RISK_PER_TRADE_ADJUSTMENT = -0.005

# --- Constants for Hybrid Scoring ---
PERFORMANCE_UPPER_THRESHOLD = 0.70
PERFORMANCE_LOWER_THRESHOLD = 0.45
BIAS_ADJUSTMENT_INCREMENT = 0.05

WEIGHT_WIN_RATE = 0.50
WEIGHT_MAX_DRAWDOWN = 0.35
WEIGHT_VOLATILITY = 0.15
MAX_ACCEPTABLE_DRAWDOWN = 0.20
MAX_ACCEPTABLE_VOLATILITY = 0.10

def _calculate_asset_performance(trades: List[Trade]) -> Dict:
    """
    Calculates performance metrics for a single asset.
    """
    pnl_pcts = [t.pnl_pct for t in trades]

    # Win Rate
    win_rate = len([p for p in pnl_pcts if p > 0]) / len(pnl_pcts) if pnl_pcts else 0

    # Max Drawdown
    if not pnl_pcts:
        max_drawdown = 0
    else:
        # Prepend an initial capital of 1.0 for accurate peak/drawdown calculation
        equity_curve = np.insert(np.cumprod([1 + p for p in pnl_pcts]), 0, 1.0)
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - peak) / peak
        # Ignore the initial 0.0 drawdown from the prepended capital
        max_drawdown = abs(np.min(drawdown[1:])) if len(drawdown) > 1 else 0

    # Volatility of Returns
    volatility = np.std(pnl_pcts) if len(pnl_pcts) > 1 else 0

    return {
        "win_rate": win_rate,
        "max_drawdown": max_drawdown,
        "volatility": volatility,
        "trade_count": len(trades)
    }

def run_learning_cycle(request: LearningRequest, correlation_id: str = "not-provided") -> LearningResponse:
    """
    Asset-aware learning cycle. Analyzes trade history grouped by asset
    and recommends policy adjustments.
    """
    print(f"[correlation_id={correlation_id}] learning cycle started")
    response = LearningResponse(learning_state="active", policy_deltas=PolicyDeltas())
    reasoning = []

    # --- Pre-filter trades to only include successfully executed ones ---
    executable_trades = [
        t for t in request.trade_history
        if t.executed and t.execution_status == 'success'
    ]

    if not executable_trades:
        response.learning_state = "insufficient_data"
        response.reasoning.append("No successfully executed trades found in the history.")
        print(f"[correlation_id={correlation_id}] learning cycle ended: no executable trades")
        return response

    trades_by_asset = defaultdict(list)
    for trade in executable_trades:
        trades_by_asset[trade.asset_id].append(trade)

    global_risk_adjustment_needed = False
    assets_in_warmup = 0

    for asset_id, trades in trades_by_asset.items():
        if len(trades) < ASSET_MIN_TRADES_WARMUP:
            assets_in_warmup += 1
            reasoning.append(f"Asset '{asset_id}' is in warmup ({len(trades)}/{ASSET_MIN_TRADES_WARMUP} trades). No bias will be applied.")
            continue

        # --- Performance and Scoring ---
        perf = _calculate_asset_performance(trades)

        # Normalize metrics to scores (higher is better)
        wr_score = perf["win_rate"]
        mdd_score = 1.0 - min(1.0, perf["max_drawdown"] / MAX_ACCEPTABLE_DRAWDOWN)
        vol_score = 1.0 - min(1.0, perf["volatility"] / MAX_ACCEPTABLE_VOLATILITY)

        performance_score = (WEIGHT_WIN_RATE * wr_score) + \
                            (WEIGHT_MAX_DRAWDOWN * mdd_score) + \
                            (WEIGHT_VOLATILITY * vol_score)

        bias_delta = 0.0
        if performance_score > PERFORMANCE_UPPER_THRESHOLD:
            bias_delta = BIAS_ADJUSTMENT_INCREMENT
            reasoning.append(f"Asset '{asset_id}' performance score ({performance_score:.2f}) is above {PERFORMANCE_UPPER_THRESHOLD}. Applying positive bias.")
        elif performance_score < PERFORMANCE_LOWER_THRESHOLD:
            bias_delta = -BIAS_ADJUSTMENT_INCREMENT
            reasoning.append(f"Asset '{asset_id}' performance score ({performance_score:.2f}) is below {PERFORMANCE_LOWER_THRESHOLD}. Applying negative bias.")

        if bias_delta != 0.0:
            response.policy_deltas.asset_biases[asset_id] = bias_delta

        # --- Drawdown Clustering Detection ---
        recent_trades = sorted(trades, key=lambda t: t.timestamp, reverse=True)[:RECENT_TRADES_WINDOW]
        recent_pnl = [t.pnl_pct for t in recent_trades]

        # Check for consecutive losses
        consecutive_losses = 0
        for pnl in recent_pnl:
            if pnl < 0:
                consecutive_losses += 1
                if consecutive_losses >= CONSECUTIVE_LOSS_THRESHOLD:
                    global_risk_adjustment_needed = True
                    reasoning.append(f"Asset '{asset_id}' has {consecutive_losses} consecutive losses. Flagging for risk review.")
                    break
            else:
                consecutive_losses = 0

        # Check for high recent drawdown
        recent_perf = _calculate_asset_performance(recent_trades)
        if recent_perf["max_drawdown"] > MAX_DRAWDOWN_THRESHOLD:
            global_risk_adjustment_needed = True
            reasoning.append(f"Asset '{asset_id}' has a high recent drawdown of {recent_perf['max_drawdown']:.2%}. Flagging for risk review.")

    if global_risk_adjustment_needed:
        response.policy_deltas.risk["risk_per_trade"] = RISK_PER_TRADE_ADJUSTMENT
        reasoning.append(f"Applying a global risk reduction of {RISK_PER_TRADE_ADJUSTMENT} due to drawdown clustering.")

    if assets_in_warmup == len(trades_by_asset):
        response.learning_state = "warmup"
    else:
        response.learning_state = "success"

    response.reasoning = reasoning
    print(f"[correlation_id={correlation_id}] policy updated")
    return response
