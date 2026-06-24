from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List

BUCKET_TARGET_WEIGHTS = {
    "core_dividend": 0.50,
    "value_rebound": 0.30,
    "news_momentum": 0.20,
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _bucket_from_row(row: Dict[str, Any]) -> str:
    return str(row.get("strategy_bucket") or row.get("bucket") or "unassigned")


def _symbol_from_row(row: Dict[str, Any]) -> str:
    return str(row.get("symbol") or "").upper()


def _approved(row: Dict[str, Any]) -> bool:
    return bool(row.get("approved")) or str(row.get("status") or "").lower() in {"approved", "executed", "submitted"}


def _pnl_pct(row: Dict[str, Any]) -> float:
    for key in ("return_pct", "pnl_pct", "realized_pnl_pct"):
        if key in row:
            return _safe_float(row.get(key))
    metadata = row.get("metadata") or {}
    for key in ("return_pct", "pnl_pct", "realized_pnl_pct"):
        if key in metadata:
            return _safe_float(metadata.get(key))
    return 0.0


def analyze_portfolio_audits(audits: List[Dict[str, Any]]) -> Dict[str, Any]:
    bucket_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        "selected_count": 0,
        "approved_count": 0,
        "executed_count": 0,
        "win_count": 0,
        "loss_count": 0,
        "total_return_pct": 0.0,
        "symbols": set(),
    })
    portfolio_count = len(audits)
    total_positions = 0
    total_approved = 0
    total_executed = 0

    for audit in audits:
        selected_positions = audit.get("selected_positions") or []
        risk_approvals = audit.get("risk_approvals") or []
        execution_orders = audit.get("execution_orders") or []

        for row in selected_positions:
            bucket = _bucket_from_row(row)
            symbol = _symbol_from_row(row)
            bucket_stats[bucket]["selected_count"] += 1
            if symbol:
                bucket_stats[bucket]["symbols"].add(symbol)
            total_positions += 1

        for row in risk_approvals:
            bucket = _bucket_from_row(row)
            symbol = _symbol_from_row(row)
            if symbol:
                bucket_stats[bucket]["symbols"].add(symbol)
            if _approved(row):
                bucket_stats[bucket]["approved_count"] += 1
                total_approved += 1

        for row in execution_orders:
            bucket = _bucket_from_row(row)
            symbol = _symbol_from_row(row)
            if symbol:
                bucket_stats[bucket]["symbols"].add(symbol)
            status = str(row.get("status") or "").lower()
            if status in {"executed", "submitted", "placed", "filled"}:
                bucket_stats[bucket]["executed_count"] += 1
                total_executed += 1
            pnl = _pnl_pct(row)
            bucket_stats[bucket]["total_return_pct"] += pnl
            if pnl > 0:
                bucket_stats[bucket]["win_count"] += 1
            elif pnl < 0:
                bucket_stats[bucket]["loss_count"] += 1

    normalized_buckets: Dict[str, Dict[str, Any]] = {}
    for bucket, stats in bucket_stats.items():
        executed_count = int(stats["executed_count"])
        selected_count = int(stats["selected_count"])
        approved_count = int(stats["approved_count"])
        decision_count = max(1, executed_count or approved_count or selected_count)
        avg_return = stats["total_return_pct"] / decision_count
        win_rate = stats["win_count"] / max(1, stats["win_count"] + stats["loss_count"])
        normalized_buckets[bucket] = {
            "selected_count": selected_count,
            "approved_count": approved_count,
            "executed_count": executed_count,
            "win_rate": round(win_rate, 4),
            "average_return_pct": round(avg_return, 6),
            "symbols": sorted(stats["symbols"]),
        }

    bucket_weight_deltas: Dict[str, float] = {}
    reasoning: List[str] = []
    for bucket, stats in sorted(normalized_buckets.items()):
        avg_return = stats["average_return_pct"]
        win_rate = stats["win_rate"]
        if stats["executed_count"] == 0 and stats["approved_count"] == 0:
            reasoning.append(f"Bucket {bucket} has no approved/executed orders yet; keep weight unchanged.")
            continue
        if avg_return > 0.01 and win_rate >= 0.5:
            bucket_weight_deltas[bucket] = 0.02
            reasoning.append(f"Bucket {bucket} is performing well; consider increasing target weight slightly.")
        elif avg_return < -0.01 or (stats["executed_count"] >= 2 and win_rate < 0.4):
            bucket_weight_deltas[bucket] = -0.02
            reasoning.append(f"Bucket {bucket} is underperforming; consider reducing target weight slightly.")
        else:
            reasoning.append(f"Bucket {bucket} is neutral; keep target weight unchanged.")

    if total_positions == 0:
        learning_state = "insufficient_data"
        confidence_score = 0.0
        reasoning.append("No selected positions found in portfolio audits.")
    elif portfolio_count < 3:
        learning_state = "warmup"
        confidence_score = min(0.45, 0.15 * portfolio_count + 0.1)
        reasoning.append("Portfolio audit sample is still small; treat deltas as low-confidence warmup feedback.")
    else:
        learning_state = "success"
        confidence_score = min(0.95, 0.5 + 0.05 * portfolio_count)

    approval_rate = total_approved / total_positions if total_positions else 0.0
    execution_rate = total_executed / total_approved if total_approved else 0.0

    return {
        "learning_state": learning_state,
        "confidence_score": round(confidence_score, 4),
        "portfolio_count": portfolio_count,
        "approval_rate": round(approval_rate, 4),
        "execution_rate": round(execution_rate, 4),
        "bucket_metrics": normalized_buckets,
        "policy_deltas": {
            "bucket_weight_deltas": bucket_weight_deltas,
            "risk": {"risk_per_trade": -0.0025} if any(delta < 0 for delta in bucket_weight_deltas.values()) else {},
            "guardrails": {
                "keep_core_satellite_structure": True,
                "target_weights": BUCKET_TARGET_WEIGHTS,
            },
        },
        "reasoning": reasoning,
    }
