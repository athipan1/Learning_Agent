from fastapi.testclient import TestClient

from learning_agent.main import app
from learning_agent.shadow_review import (
    ShadowReviewRequest,
    evaluate_shadow_for_paper,
)


def _request(**overrides):
    evidence = {
        "strategy_id": "trend-v7",
        "observation_count": 60,
        "net_expectancy_pct": 0.012,
        "profit_factor": 1.35,
        "max_drawdown_pct": 0.05,
        "average_mfe_pct": 0.025,
        "average_mae_pct": -0.008,
        "execution_cost_calibrated": True,
        "final_holdout_passed": True,
    }
    evidence.update(overrides)
    return ShadowReviewRequest.model_validate({"evidence": evidence})


def test_shadow_review_requests_paper_review_only_after_all_gates_pass():
    result = evaluate_shadow_for_paper(_request())

    assert result.decision == "request_paper_review"
    assert result.failed_gates == []
    assert result.requires_human_review is True
    assert result.auto_promote is False
    assert result.risk_policy_change_authorized is False
    assert result.broker_order_authorized is False


def test_shadow_review_continues_shadow_when_sample_or_expectancy_fails():
    result = evaluate_shadow_for_paper(
        _request(observation_count=10, net_expectancy_pct=-0.002)
    )

    assert result.decision == "continue_shadow"
    assert "shadow_observation_count" in result.failed_gates
    assert "positive_net_expectancy" in result.failed_gates
    assert result.broker_order_authorized is False


def test_shadow_review_never_bypasses_final_holdout():
    result = evaluate_shadow_for_paper(_request(final_holdout_passed=False))

    assert result.decision == "continue_shadow"
    assert "final_holdout_passed" in result.failed_gates


def test_shadow_review_http_endpoint_is_advisory_only():
    client = TestClient(app)
    response = client.post(
        "/learn/shadow-paper-review",
        json=_request().model_dump(mode="json"),
        headers={"X-Correlation-ID": "corr-shadow-learning"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["data"]["decision"] == "request_paper_review"
    assert body["data"]["auto_promote"] is False
    assert body["data"]["broker_order_authorized"] is False
    assert body["metadata"]["advisory_only"] is True
    assert body["metadata"]["broker_order_authority"] is False
