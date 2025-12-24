from typing import List, Dict
import numpy as np
from .schemas import Trade, AgentVote, OrchestratorConfig
import json

# --- Constants for Analysis ---
MIN_TRADES_FOR_ANALYSIS = 30
VOLATILE_MARKET_TRADING_THRESHOLD = 0.4
AGENT_MIN_TRADES_THRESHOLD = 10
AGENT_WIN_RATE_DIFF_THRESHOLD = 0.15
MARKET_BIAS_WIN_RATE_THRESHOLD = 0.6
LOW_WIN_RATE_THRESHOLD = 0.5
HIGH_DRAWDOWN_THRESHOLD = -15.0

# --- Constants for Adjustments ---
AGENT_WEIGHT_ADJUSTMENT = 0.05
RISK_PER_TRADE_ADJUSTMENT = -0.005
MAX_POSITION_PCT_ADJUSTMENT = -0.05


def process_trades_json(trades_json: str) -> str:
    """
    Parses a JSON string of trades, analyzes them, and returns recommendations as a JSON string.
    """
    trade_dicts = json.loads(trades_json)
    trades = []
    for trade_dict in trade_dicts:
        agent_votes = {name: AgentVote(**vote) for name, vote in trade_dict["agent_votes"].items()}
        orchestrator_config = OrchestratorConfig(**trade_dict["orchestrator_config"])

        trade = Trade(
            trade_id=trade_dict["trade_id"],
            ticker=trade_dict["ticker"],
            final_verdict=trade_dict["final_verdict"],
            executed=trade_dict["executed"],
            entry_price=trade_dict["entry_price"],
            exit_price=trade_dict["exit_price"],
            pnl_pct=trade_dict["pnl_pct"],
            holding_days=trade_dict["holding_days"],
            market_regime=trade_dict["market_regime"],
            agent_votes=agent_votes,
            orchestrator_config=orchestrator_config,
            timestamp=trade_dict["timestamp"]
        )
        trades.append(trade)

    recommendations = analyze_trades(trades)
    return json.dumps(recommendations, indent=2)


def analyze_trades(trades: List[Trade]) -> Dict:
    """
    Analyzes a list of trades and returns a dictionary with performance analysis.
    """
    if len(trades) < MIN_TRADES_FOR_ANALYSIS:
        return {
            "summary": {
                "overall_assessment": "stable",
                "key_issue": f"Monitoring period (less than {MIN_TRADES_FOR_ANALYSIS} trades)",
                "confidence_level": 0.5
            },
            "agent_weight_adjustments": {},
            "risk_adjustments": {},
            "strategy_bias": {},
            "guardrails": {},
            "reasoning": ["Initial monitoring period, no adjustments recommended."]
        }

    # Helper functions
    def calculate_win_rate(trade_list: List[Trade]) -> float:
        wins = sum(1 for trade in trade_list if trade.pnl_pct > 0)
        return wins / len(trade_list) if trade_list else 0.0

    def calculate_max_drawdown(trade_list: List[Trade]) -> float:
        returns = [trade.pnl_pct / 100 for trade in trade_list]
        cumulative_returns = np.cumprod([1 + r for r in returns])
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - peak) / peak
        return np.min(drawdown) * 100 if len(drawdown) > 0 else 0.0

    # 1. Agent Performance Analysis
    agent_performance = {}
    for trade in trades:
        for agent_name, vote in trade.agent_votes.items():
            if agent_name not in agent_performance:
                agent_performance[agent_name] = {"wins": 0, "losses": 0, "trades": 0}

            agent_performance[agent_name]["trades"] += 1
            is_win = trade.pnl_pct > 0
            correct_call = (vote.action == trade.final_verdict)
            if (is_win and correct_call) or (not is_win and not correct_call):
                agent_performance[agent_name]["wins"] += 1
            else:
                agent_performance[agent_name]["losses"] += 1

    # 2. Market Condition Analysis
    market_condition_performance = {regime: [] for regime in ["trending", "range", "volatile"]}
    for trade in trades:
        market_condition_performance[trade.market_regime].append(trade)

    market_win_rates = {r: calculate_win_rate(t) for r, t in market_condition_performance.items()}

    # 3. Behavioral Bias & Performance Metrics
    volatile_trades = market_condition_performance.get("volatile", [])
    overtrading_bias = len(volatile_trades) > (len(trades) * VOLATILE_MARKET_TRADING_THRESHOLD)
    overall_win_rate = calculate_win_rate(trades)
    max_drawdown = calculate_max_drawdown(trades)

    analysis_result = {
        "agent_performance": agent_performance,
        "market_win_rates": market_win_rates,
        "overtrading_bias": overtrading_bias,
        "overall_win_rate": overall_win_rate,
        "max_drawdown": max_drawdown
    }

    return generate_recommendations(analysis_result)


def generate_recommendations(analysis_result: Dict) -> Dict:
    """
    Generates recommendations based on the analysis result.
    """
    recommendations = {
        "summary": {"overall_assessment": "stable", "key_issue": "N/A", "confidence_level": 0.8},
        "agent_weight_adjustments": {"technical": 0.0, "fundamental": 0.0, "sentiment": 0.0, "macro": 0.0},
        "risk_adjustments": {"risk_per_trade": 0.0, "max_position_pct": 0.0, "stop_loss_pct": 0.0, "enable_technical_stop": True},
        "strategy_bias": {"preferred_action": "hold", "market_condition_bias": "none"},
        "guardrails": {"pause_trading_if": {"max_drawdown_pct": 10, "consecutive_losses": 5}},
        "reasoning": []
    }

    # Assess overall performance
    if analysis_result["overall_win_rate"] < LOW_WIN_RATE_THRESHOLD:
        recommendations["summary"]["overall_assessment"] = "degrading"
        recommendations["summary"]["key_issue"] = "Low overall win rate"
        recommendations["strategy_bias"]["preferred_action"] = "hold"
        recommendations["reasoning"].append(f"Overall win rate is below {LOW_WIN_RATE_THRESHOLD*100}% ({analysis_result['overall_win_rate']:.2f}). Shifting to a 'hold' bias.")

    # Adjust based on max drawdown
    if analysis_result["max_drawdown"] < HIGH_DRAWDOWN_THRESHOLD:
        recommendations["risk_adjustments"]["risk_per_trade"] = RISK_PER_TRADE_ADJUSTMENT
        recommendations["risk_adjustments"]["max_position_pct"] = MAX_POSITION_PCT_ADJUSTMENT
        recommendations["summary"]["key_issue"] = "High max drawdown"
        recommendations["reasoning"].append(f"Max drawdown of {analysis_result['max_drawdown']:.2f}% exceeds threshold. Reducing risk exposure.")

    # Adjust agent weights
    best_agent, worst_agent = None, None
    best_win_rate, worst_win_rate = 0, 1
    for agent, perf in analysis_result["agent_performance"].items():
        if perf["trades"] > AGENT_MIN_TRADES_THRESHOLD:
            win_rate = perf["wins"] / perf["trades"]
            if win_rate > best_win_rate:
                best_win_rate, best_agent = win_rate, agent
            if win_rate < worst_win_rate:
                worst_win_rate, worst_agent = win_rate, agent

    if best_agent and worst_agent and best_agent != worst_agent and (best_win_rate - worst_win_rate) > AGENT_WIN_RATE_DIFF_THRESHOLD:
        recommendations["agent_weight_adjustments"][best_agent] = AGENT_WEIGHT_ADJUSTMENT
        recommendations["agent_weight_adjustments"][worst_agent] = -AGENT_WEIGHT_ADJUSTMENT
        recommendations["reasoning"].append(f"Adjusting weights to favor {best_agent} ({best_win_rate:.2f}) over {worst_agent} ({worst_win_rate:.2f}).")

    # Strategy bias based on market conditions
    market_win_rates = analysis_result["market_win_rates"]
    if market_win_rates:
        best_market = max(market_win_rates, key=market_win_rates.get)
        if market_win_rates[best_market] > MARKET_BIAS_WIN_RATE_THRESHOLD:
            recommendations["strategy_bias"]["market_condition_bias"] = best_market
            recommendations["reasoning"].append(f"Performance is strongest in {best_market} markets. Biasing strategy.")

    return recommendations
