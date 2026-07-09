from copy import deepcopy

from fastapi.testclient import TestClient

from learning_agent.main import app
from learning_agent.models import LearningOutcomeRequest
from learning_agent.outcome_learning import analyze_learning_outcomes


client = TestClient(app)

VERSIONS = {
    "scanner": "scanner-bucket-hints-v2",
    "fundamental": "fundamental-evidence-v1",
    "technical": "technical-evidence-v1",
    "manager": "manager-analysis-evidence-v1",
}


def _contribution(source, bucket="value_rebound", confidence=0.8):
    return {
        "version": VERSIONS[source],
        "supported_bucket": bucket,
        "confidence": confidence,
        "evidence_status": "complete",
        "reasons": [f"{source}_test_evidence"],
    }


def _outcome(
    index,
    *,
    profitable=True,
    bucket="value_rebound",
    scanner_bucket=None,
    technical_bucket=None,
):
    pnl = 100.0 if profitable else -75.0
    return {
        "outcome_version": "learning-outcome-v1",
        "outcome_id": f"outcome-{index}",
        "trade_plan_id": f"plan-{index}",
        "account_id": "paper-1",
        "symbol": f"SYM{index}",
        "strategy_bucket": bucket,
        "manager_bucket": bucket,
        "execution_bucket": bucket,
        "database_bucket": bucket,
        "manager_classifier_version": "manager-strategy-bucket-v3",
        "evidence_versions": dict(VERSIONS),
        "evidence_contributions": {
            "scanner": _contribution(
                "scanner",
                scanner_bucket if scanner_bucket is not None else bucket,
                0.82,
            ),
            "fundamental": _contribution("fundamental", bucket, 0.80),
            "technical": _contribution(
                "technical",
                technical_bucket if technical_bucket is not None else bucket,
                0.76,
            ),
            "manager": _contribution("manager", bucket, 0.84),
        },
        "classification_inputs": {
            "fundamental": {"valuation_score": 0.82},
            "technical": {"momentum_score": 0.64},
        },
        "bucket_confidence": 0.84,
        "entry_price": 100,
        "exit_price": 110 if profitable else 92.5,
        "realized_pnl": pnl,
        "return_pct": 0.10 if profitable else -0.075,
        "holding_period_days": 12,
        "exit_reason": "target" if profitable else "stop_loss",
        "risk_approved": True,
        "execution_status": "closed",
        "outcome_status": "closed",
        "pnl_status": "realized",
    }


def _request(outcomes, **overrides):
    payload = {
        "account_id": "paper-1",
        "learning_mode": "versioned_outcome_attribution",
        "outcomes": outcomes,
        "min_total_samples": 5,
        "min_bucket_samples": 5,
        "min_source_samples": 5,
    }
    payload.update(overrides)
    return LearningOutcomeRequest.model_validate(payload)


def test_outcome_learning_aggregates_bucket_and_source_attribution():
    outcomes = []
    for index in range(20):
        profitable = index < 16
        outcomes.append(
            _outcome(
                index,
                profitable=profitable,
                scanner_bucket=(
                    "value_rebound" if profitable else "news_momentum"
                ),
                technical_bucket=(
                    "news_momentum" if profitable else "value_rebound"
                ),
            )
        )

    response = analyze_learning_outcomes(
        _request(outcomes, min_source_samples=4)
    )

    assert response.learning_state == "review_ready"
    assert response.accepted_outcomes == 20
    assert response.rejected_outcomes == 0
    assert response.bucket_metrics["value_rebound"]["sample_count"] == 20
    assert response.bucket_metrics["value_rebound"]["win_rate"] == 0.8
    assert response.source_attribution["scanner"]["supported_trade_count"] == 16
    assert response.source_attribution["scanner"]["supported_win_rate"] == 1.0
    assert response.source_attribution["technical"]["supported_trade_count"] == 4
    assert response.source_attribution["technical"]["supported_win_rate"] == 0.0
    deltas = response.policy_recommendations["agent_weight_deltas"]
    assert deltas["scanner"] > 0
    assert deltas["technical"] < 0
    assert response.guardrails["requires_human_review"] is True
    assert response.guardrails["auto_apply"] is False


def test_weak_bucket_recommends_stricter_threshold():
    outcomes = [
        _outcome(
            index,
            profitable=index < 2,
            bucket="news_momentum",
            scanner_bucket="news_momentum",
            technical_bucket="news_momentum",
        )
        for index in range(10)
    ]

    response = analyze_learning_outcomes(_request(outcomes))

    threshold_deltas = response.policy_recommendations[
        "bucket_threshold_deltas"
    ]
    assert threshold_deltas["news_momentum"] > 0
    assert response.policy_recommendations["risk_deltas"][
        "risk_per_trade"
    ] < 0


def test_warmup_does_not_recommend_policy_changes():
    outcomes = [_outcome(index, profitable=False) for index in range(3)]

    response = analyze_learning_outcomes(
        _request(
            outcomes,
            min_total_samples=5,
            min_bucket_samples=1,
            min_source_samples=1,
        )
    )

    assert response.learning_state == "warmup"
    assert response.policy_recommendations == {
        "agent_weight_deltas": {},
        "bucket_threshold_deltas": {},
        "risk_deltas": {},
    }


def test_duplicate_outcome_is_rejected_once():
    first = _outcome(1)
    duplicate = deepcopy(first)

    response = analyze_learning_outcomes(_request([first, duplicate]))

    assert response.accepted_outcomes == 1
    assert response.rejected_outcomes == 1
    assert response.duplicate_outcome_ids == ["outcome-1"]
    assert response.rejected_records[0]["issues"] == [
        "duplicate_outcome_id"
    ]


def test_open_unrealized_and_bucket_mismatch_are_not_learned():
    invalid = _outcome(1)
    invalid["outcome_status"] = "open"
    invalid["pnl_status"] = "unrealized"
    invalid["database_bucket"] = "core_dividend"

    response = analyze_learning_outcomes(_request([invalid]))

    assert response.learning_state == "no_valid_outcomes"
    assert response.accepted_outcomes == 0
    issues = response.rejected_records[0]["issues"]
    assert "outcome_not_closed" in issues
    assert "pnl_not_realized" in issues
    assert any(issue.startswith("strategy_bucket_mismatch") for issue in issues)


def test_unsupported_evidence_version_is_rejected():
    invalid = _outcome(1)
    invalid["evidence_versions"]["technical"] = "technical-evidence-v999"

    response = analyze_learning_outcomes(_request([invalid]))

    assert response.accepted_outcomes == 0
    issues = response.rejected_records[0]["issues"]
    assert any(
        issue.startswith("unsupported_technical_evidence_version")
        for issue in issues
    )
    assert "technical_contribution_version_mismatch" in issues


def test_learn_outcomes_endpoint_preserves_correlation_and_guardrails():
    response = client.post(
        "/learn/outcomes",
        json={
            "account_id": "paper-1",
            "outcomes": [_outcome(index) for index in range(5)],
            "min_total_samples": 5,
            "min_bucket_samples": 5,
            "min_source_samples": 5,
        },
        headers={"X-Correlation-ID": "outcome-learning-test"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["version"] == "1.1.0"
    assert payload["correlation_id"] == "outcome-learning-test"
    assert payload["metadata"]["outcome_contract_version"] == (
        "learning-outcome-v1"
    )
    assert payload["metadata"]["auto_apply"] is False
    assert payload["data"]["accepted_outcomes"] == 5
    assert payload["data"]["guardrails"]["requires_human_review"] is True
