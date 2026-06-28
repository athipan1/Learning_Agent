from __future__ import annotations

from typing import Any, Dict, List

from .models import PerformanceLearningRequest, PerformanceLearningResponse, PerformanceSummaryPayload

MIN_PROFIT_FACTOR_FOR_INCREASE = 1.25
MAX_PROFIT_FACTOR_FOR_DECREASE = 0.90
MIN_EXPECTANCY_FOR_INCREASE = 0.0
MIN_WIN_RATE_FOR_INCREASE = 0.50
LOW_WIN_RATE_THRESHOLD = 0.40
STRATEGY_WEIGHT_STEP = 0.05
SYMBOL_BIAS_STEP = 0.03
RISK_REDUCTION_STEP = -0.0025


def _clamp(value: float, low: float = -0.25, high: float = 0.25) -> float:
    return max(low, min(high, value))


def _safe_profit_factor(value: float | None) -> float:
    if value is None:
        return 999.0
    return float(value)


def _performance_score(summary: PerformanceSummaryPayload) -> float:
    win_component = max(0.0, min(1.0, float(summary.win_rate)))
    return_component = 0.5 + max(-0.5, min(0.5, float(summary.return_pct)))
    pf = _safe_profit_factor(summary.profit_factor)
    pf_component = max(0.0, min(1.0, pf / 2.0))
    expectancy_component = 0.6 if summary.expectancy > 0 else 0.3 if summary.expectancy == 0 else 0.0
    return round((win_component * 0.35) + (return_component * 0.25) + (pf_component * 0.25) + (expectancy_component * 0.15), 4)


def _bucket_delta(metric: Dict[str, Any]) -> float:
    closed = int(metric.get("closed_plan_count") or 0)
    if closed <= 0:
        return 0.0
    win_rate = float(metric.get("win_rate") or 0.0)
    expectancy = float(metric.get("expectancy") or 0.0)
    net_pnl = float(metric.get("net_pnl") or 0.0)
    profit_factor = _safe_profit_factor(metric.get("profit_factor"))

    if net_pnl > 0 and expectancy > MIN_EXPECTANCY_FOR_INCREASE and win_rate >= MIN_WIN_RATE_FOR_INCREASE and profit_factor >= MIN_PROFIT_FACTOR_FOR_INCREASE:
        return STRATEGY_WEIGHT_STEP
    if net_pnl < 0 or expectancy < 0 or win_rate < LOW_WIN_RATE_THRESHOLD or profit_factor < MAX_PROFIT_FACTOR_FOR_DECREASE:
        return -STRATEGY_WEIGHT_STEP
    return 0.0


def _symbol_delta(metric: Dict[str, Any]) -> float:
    closed = int(metric.get("closed_plan_count") or 0)
    if closed <= 0:
        return 0.0
    expectancy = float(metric.get("expectancy") or 0.0)
    net_pnl = float(metric.get("net_pnl") or 0.0)
    win_rate = float(metric.get("win_rate") or 0.0)
    if net_pnl > 0 and expectancy > 0 and win_rate >= 0.50:
        return SYMBOL_BIAS_STEP
    if net_pnl < 0 or expectancy < 0 or win_rate < LOW_WIN_RATE_THRESHOLD:
        return -SYMBOL_BIAS_STEP
    return 0.0


def analyze_performance_summary(request: PerformanceLearningRequest) -> PerformanceLearningResponse:
    summary = request.performance_summary
    reasoning: List[str] = []
    bucket_metrics = {key: value.model_dump(mode="json") for key, value in summary.by_strategy_bucket.items()}
    symbol_metrics = {key: value.model_dump(mode="json") for key, value in summary.by_symbol.items()}
    policy_deltas: Dict[str, Any] = {
        "strategy_bucket_weights": {},
        "asset_biases": {},
        "risk": {},
        "guardrails": {
            "requires_human_review": True,
            "auto_apply": False,
            "source": "performance_summary_review",
        },
    }

    if summary.closed_plan_count < request.min_closed_plans:
        reasoning.append(
            f"Only {summary.closed_plan_count} closed TradePlan(s), below minimum {request.min_closed_plans}. No weight changes recommended."
        )
        return PerformanceLearningResponse(
            learning_state="warmup",
            learning_mode=request.learning_mode,
            confidence_score=0.25,
            reviewed_closed_plans=summary.closed_plan_count,
            performance_score=_performance_score(summary),
            bucket_metrics=bucket_metrics,
            symbol_metrics=symbol_metrics,
            policy_deltas=policy_deltas,
            reasoning=reasoning,
        )

    for bucket, metric in bucket_metrics.items():
        delta = _bucket_delta(metric)
        if delta != 0.0:
            policy_deltas["strategy_bucket_weights"][bucket] = _clamp(delta)
            direction = "increase" if delta > 0 else "decrease"
            reasoning.append(
                f"Recommend {direction} strategy bucket '{bucket}' by {abs(delta):.2f}: "
                f"net_pnl={metric.get('net_pnl')}, win_rate={metric.get('win_rate')}, "
                f"expectancy={metric.get('expectancy')}, profit_factor={metric.get('profit_factor')}."
            )

    for symbol, metric in symbol_metrics.items():
        delta = _symbol_delta(metric)
        if delta != 0.0:
            policy_deltas["asset_biases"][symbol] = _clamp(delta, -0.10, 0.10)
            direction = "positive" if delta > 0 else "negative"
            reasoning.append(
                f"Apply {direction} asset bias for {symbol}: net_pnl={metric.get('net_pnl')}, "
                f"win_rate={metric.get('win_rate')}, expectancy={metric.get('expectancy')}."
            )

    if summary.net_pnl < 0 or summary.profit_factor is not None and summary.profit_factor < MAX_PROFIT_FACTOR_FOR_DECREASE:
        policy_deltas["risk"]["risk_per_trade"] = RISK_REDUCTION_STEP
        reasoning.append("Overall TradePlan performance is weak; recommend reducing risk_per_trade.")

    if not policy_deltas["strategy_bucket_weights"] and not policy_deltas["asset_biases"] and not policy_deltas["risk"]:
        reasoning.append("Performance is mixed or neutral; no policy delta recommended.")

    confidence = 0.55 + min(0.35, summary.closed_plan_count / 100)
    if summary.warnings:
        confidence -= 0.10
        reasoning.append(f"Performance summary contained warnings: {summary.warnings}")

    return PerformanceLearningResponse(
        learning_state="success",
        learning_mode=request.learning_mode,
        confidence_score=round(max(0.0, min(1.0, confidence)), 4),
        reviewed_closed_plans=summary.closed_plan_count,
        performance_score=_performance_score(summary),
        bucket_metrics=bucket_metrics,
        symbol_metrics=symbol_metrics,
        policy_deltas=policy_deltas,
        reasoning=reasoning,
    )
