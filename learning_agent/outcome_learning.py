from __future__ import annotations

from collections import defaultdict
from math import sqrt
from typing import Any, Dict, Iterable, List, Tuple

from .models import (
    EvidenceContribution,
    LearningOutcomeRecord,
    LearningOutcomeRequest,
    LearningOutcomeResponse,
)


LEARNING_OUTCOME_VERSION = "learning-outcome-v1"
SUPPORTED_EVIDENCE_VERSIONS = {
    "scanner": "scanner-bucket-hints-v2",
    "fundamental": "fundamental-evidence-v1",
    "technical": "technical-evidence-v1",
    "manager": "manager-analysis-evidence-v1",
}
SUPPORTED_MANAGER_CLASSIFIER_VERSION = "manager-strategy-bucket-v3"
SOURCE_WEIGHT_STEP = 0.03
BUCKET_THRESHOLD_INCREASE = 0.02
BUCKET_THRESHOLD_DECREASE = -0.01
RISK_REDUCTION_STEP = -0.0025


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _wilson_interval(
    wins: int,
    sample_count: int,
    z: float = 1.96,
) -> Tuple[float, float]:
    if sample_count <= 0:
        return 0.0, 0.0
    proportion = wins / sample_count
    denominator = 1.0 + (z * z / sample_count)
    centre = proportion + (z * z / (2.0 * sample_count))
    spread = z * sqrt(
        (proportion * (1.0 - proportion) / sample_count)
        + (z * z / (4.0 * sample_count * sample_count))
    )
    return (
        round(max(0.0, (centre - spread) / denominator), 4),
        round(min(1.0, (centre + spread) / denominator), 4),
    )


def _metric(rows: Iterable[LearningOutcomeRecord]) -> Dict[str, Any]:
    records = list(rows)
    sample_count = len(records)
    wins = sum(
        1 for record in records if float(record.return_pct) > 0.0
    )
    losses = sum(
        1 for record in records if float(record.return_pct) < 0.0
    )
    breakeven = sample_count - wins - losses
    win_rate = wins / sample_count if sample_count else 0.0
    net_pnl = sum(float(record.realized_pnl) for record in records)
    avg_return = (
        sum(float(record.return_pct) for record in records) / sample_count
        if sample_count
        else 0.0
    )
    expectancy = net_pnl / sample_count if sample_count else 0.0
    avg_holding = (
        sum(float(record.holding_period_days) for record in records)
        / sample_count
        if sample_count
        else 0.0
    )
    avg_confidence = (
        sum(float(record.bucket_confidence) for record in records)
        / sample_count
        if sample_count
        else 0.0
    )
    calibration_error = (
        sum(
            abs(
                float(record.bucket_confidence)
                - (1.0 if float(record.return_pct) > 0.0 else 0.0)
            )
            for record in records
        )
        / sample_count
        if sample_count
        else 0.0
    )
    ci_low, ci_high = _wilson_interval(wins, sample_count)
    return {
        "sample_count": sample_count,
        "wins": wins,
        "losses": losses,
        "breakeven": breakeven,
        "win_rate": round(win_rate, 4),
        "win_rate_ci_low": ci_low,
        "win_rate_ci_high": ci_high,
        "net_pnl": round(net_pnl, 6),
        "average_return_pct": round(avg_return, 6),
        "expectancy": round(expectancy, 6),
        "average_holding_period_days": round(avg_holding, 4),
        "average_bucket_confidence": round(avg_confidence, 4),
        "confidence_calibration_error": round(calibration_error, 4),
    }


def _source_metric(
    rows: Iterable[Tuple[LearningOutcomeRecord, EvidenceContribution]],
) -> Dict[str, Any]:
    pairs = list(rows)
    sample_count = len(pairs)
    manager_agreements = sum(
        1
        for record, contribution in pairs
        if contribution.supported_bucket == record.strategy_bucket
    )
    supported_pairs = [
        (record, contribution)
        for record, contribution in pairs
        if contribution.supported_bucket == record.strategy_bucket
    ]
    supported_count = len(supported_pairs)
    supported_wins = sum(
        1
        for record, _ in supported_pairs
        if float(record.return_pct) > 0.0
    )
    supported_win_rate = (
        supported_wins / supported_count if supported_count else 0.0
    )
    ci_low, ci_high = _wilson_interval(
        supported_wins,
        supported_count,
    )
    avg_confidence = (
        sum(float(contribution.confidence) for _, contribution in pairs)
        / sample_count
        if sample_count
        else 0.0
    )
    return {
        "sample_count": sample_count,
        "manager_agreement_count": manager_agreements,
        "manager_agreement_rate": round(
            manager_agreements / sample_count if sample_count else 0.0,
            4,
        ),
        "supported_trade_count": supported_count,
        "supported_wins": supported_wins,
        "supported_win_rate": round(supported_win_rate, 4),
        "supported_win_rate_ci_low": ci_low,
        "supported_win_rate_ci_high": ci_high,
        "average_confidence": round(avg_confidence, 4),
    }


def _validate_record(record: LearningOutcomeRecord) -> List[str]:
    issues: List[str] = []
    if record.outcome_version != LEARNING_OUTCOME_VERSION:
        issues.append(
            f"unsupported_outcome_version:{record.outcome_version}"
        )
    if record.outcome_status != "closed":
        issues.append("outcome_not_closed")
    if record.pnl_status != "realized":
        issues.append("pnl_not_realized")
    if not record.risk_approved:
        issues.append("risk_not_approved")
    if record.execution_status.lower() not in {
        "filled",
        "closed",
        "completed",
        "exited",
    }:
        issues.append(
            f"execution_not_complete:{record.execution_status}"
        )

    bucket_values = {
        "strategy": record.strategy_bucket,
        "manager": record.manager_bucket,
        "execution": record.execution_bucket,
        "database": record.database_bucket,
    }
    if len(set(bucket_values.values())) != 1:
        issues.append(
            "strategy_bucket_mismatch:"
            + ",".join(
                f"{name}={value}"
                for name, value in bucket_values.items()
            )
        )

    if (
        record.manager_classifier_version
        != SUPPORTED_MANAGER_CLASSIFIER_VERSION
    ):
        issues.append(
            "unsupported_manager_classifier_version:"
            f"{record.manager_classifier_version}"
        )

    for source, expected_version in SUPPORTED_EVIDENCE_VERSIONS.items():
        actual_version = record.evidence_versions.get(source)
        if actual_version != expected_version:
            issues.append(
                f"unsupported_{source}_evidence_version:"
                f"{actual_version or 'missing'}"
            )

    for source in ("scanner", "fundamental", "technical", "manager"):
        contribution = record.evidence_contributions.get(source)
        if contribution is None:
            issues.append(f"missing_{source}_evidence_contribution")
            continue
        expected_version = record.evidence_versions.get(source)
        if contribution.version != expected_version:
            issues.append(
                f"{source}_contribution_version_mismatch"
            )
        if contribution.evidence_status in {
            "insufficient",
            "invalid",
            "conflict",
        }:
            issues.append(
                f"{source}_evidence_status_not_learnable:"
                f"{contribution.evidence_status}"
            )
    return issues


def _recommend_source_weights(
    source_metrics: Dict[str, Dict[str, Any]],
    minimum_samples: int,
) -> Dict[str, float]:
    recommendations: Dict[str, float] = {}
    for source, metric in source_metrics.items():
        sample_count = int(metric.get("supported_trade_count") or 0)
        if sample_count < minimum_samples:
            continue
        win_rate = float(metric.get("supported_win_rate") or 0.0)
        ci_low = float(
            metric.get("supported_win_rate_ci_low") or 0.0
        )
        ci_high = float(
            metric.get("supported_win_rate_ci_high") or 0.0
        )
        if win_rate >= 0.60 and ci_low >= 0.50:
            recommendations[source] = SOURCE_WEIGHT_STEP
        elif win_rate <= 0.40 and ci_high <= 0.50:
            recommendations[source] = -SOURCE_WEIGHT_STEP
    return recommendations


def _recommend_bucket_thresholds(
    bucket_metrics: Dict[str, Dict[str, Any]],
    minimum_samples: int,
) -> Dict[str, float]:
    recommendations: Dict[str, float] = {}
    for bucket, metric in bucket_metrics.items():
        sample_count = int(metric.get("sample_count") or 0)
        if sample_count < minimum_samples:
            continue
        win_rate = float(metric.get("win_rate") or 0.0)
        expectancy = float(metric.get("expectancy") or 0.0)
        ci_low = float(metric.get("win_rate_ci_low") or 0.0)
        if win_rate <= 0.40 or expectancy < 0.0:
            recommendations[bucket] = BUCKET_THRESHOLD_INCREASE
        elif (
            win_rate >= 0.60
            and expectancy > 0.0
            and ci_low >= 0.50
        ):
            recommendations[bucket] = BUCKET_THRESHOLD_DECREASE
    return recommendations


def _empty_policy_recommendations() -> Dict[str, Dict[str, float]]:
    return {
        "agent_weight_deltas": {},
        "bucket_threshold_deltas": {},
        "risk_deltas": {},
    }


def analyze_learning_outcomes(
    request: LearningOutcomeRequest,
) -> LearningOutcomeResponse:
    accepted: List[LearningOutcomeRecord] = []
    rejected: List[Dict[str, Any]] = []
    duplicate_outcome_ids: List[str] = []
    seen_outcome_ids: set[str] = set()

    for record in request.outcomes:
        if record.outcome_id in seen_outcome_ids:
            duplicate_outcome_ids.append(record.outcome_id)
            rejected.append(
                {
                    "outcome_id": record.outcome_id,
                    "trade_plan_id": record.trade_plan_id,
                    "issues": ["duplicate_outcome_id"],
                }
            )
            continue
        seen_outcome_ids.add(record.outcome_id)
        issues = _validate_record(record)
        if issues:
            rejected.append(
                {
                    "outcome_id": record.outcome_id,
                    "trade_plan_id": record.trade_plan_id,
                    "issues": issues,
                }
            )
            continue
        accepted.append(record)

    by_bucket: Dict[str, List[LearningOutcomeRecord]] = defaultdict(list)
    source_rows: Dict[
        str,
        List[Tuple[LearningOutcomeRecord, EvidenceContribution]],
    ] = defaultdict(list)
    for record in accepted:
        by_bucket[record.strategy_bucket].append(record)
        for source, contribution in record.evidence_contributions.items():
            source_rows[source].append((record, contribution))

    bucket_metrics = {
        bucket: _metric(records)
        for bucket, records in by_bucket.items()
    }
    source_metrics = {
        source: _source_metric(rows)
        for source, rows in source_rows.items()
    }
    overall_metric = _metric(accepted)

    source_weight_deltas = _recommend_source_weights(
        source_metrics,
        request.min_source_samples,
    )
    bucket_threshold_deltas = _recommend_bucket_thresholds(
        bucket_metrics,
        request.min_bucket_samples,
    )
    risk_deltas: Dict[str, float] = {}
    if (
        len(accepted) >= request.min_total_samples
        and (
            float(overall_metric["net_pnl"]) < 0.0
            or float(overall_metric["win_rate"]) < 0.40
        )
    ):
        risk_deltas["risk_per_trade"] = RISK_REDUCTION_STEP

    policy_recommendations = {
        "agent_weight_deltas": source_weight_deltas,
        "bucket_threshold_deltas": bucket_threshold_deltas,
        "risk_deltas": risk_deltas,
    }
    if len(accepted) < request.min_total_samples:
        policy_recommendations = _empty_policy_recommendations()

    has_recommendation = any(
        bool(values) for values in policy_recommendations.values()
    )

    reasoning = [
        f"Accepted {len(accepted)} closed realized outcome(s).",
        (
            f"Rejected {len(rejected)} outcome(s) that failed "
            "learning guardrails."
        ),
    ]
    if not accepted:
        learning_state = "no_valid_outcomes"
        confidence = 0.0
        reasoning.append(
            "No outcome was eligible for attribution learning."
        )
    elif len(accepted) < request.min_total_samples:
        learning_state = "warmup"
        confidence = 0.25
        reasoning.append(
            "Accepted outcome count is below the total sample threshold; "
            "all policy recommendations were suppressed."
        )
    else:
        learning_state = (
            "review_ready" if has_recommendation else "stable"
        )
        confidence = _clamp(
            0.50 + (len(accepted) / 100.0),
            0.0,
            0.85,
        )
        if has_recommendation:
            reasoning.append(
                "Statistically guarded policy recommendations are ready "
                "for human review."
            )
        else:
            reasoning.append(
                "No source or bucket met the guarded policy-change "
                "thresholds."
            )

    return LearningOutcomeResponse(
        learning_state=learning_state,
        learning_mode=request.learning_mode,
        outcome_contract_version=LEARNING_OUTCOME_VERSION,
        confidence_score=round(confidence, 4),
        reviewed_outcomes=len(request.outcomes),
        accepted_outcomes=len(accepted),
        rejected_outcomes=len(rejected),
        duplicate_outcome_ids=list(
            dict.fromkeys(duplicate_outcome_ids)
        ),
        rejected_records=rejected,
        overall_metrics=overall_metric,
        bucket_metrics=bucket_metrics,
        source_attribution=source_metrics,
        confidence_calibration={
            "overall_error": overall_metric[
                "confidence_calibration_error"
            ],
            "by_bucket": {
                bucket: metric["confidence_calibration_error"]
                for bucket, metric in bucket_metrics.items()
            },
        },
        policy_recommendations=policy_recommendations,
        guardrails={
            "requires_human_review": True,
            "auto_apply": False,
            "minimum_total_samples": request.min_total_samples,
            "minimum_bucket_samples": request.min_bucket_samples,
            "minimum_source_samples": request.min_source_samples,
            "uses_realized_pnl_only": True,
            "requires_bucket_consistency": True,
            "supported_manager_classifier_version": (
                SUPPORTED_MANAGER_CLASSIFIER_VERSION
            ),
            "supported_evidence_versions": (
                SUPPORTED_EVIDENCE_VERSIONS
            ),
        },
        reasoning=reasoning,
    )
