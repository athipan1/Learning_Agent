from fastapi.testclient import TestClient

from learning_agent.champion_challenger import (
    ChampionChallengerRequest,
    evaluate_champion_challenger,
)
from learning_agent.main import app


def _request(**challenger_overrides):
    challenger = {
        "strategy_id": "challenger-v1",
        "research_profile": "strategy_research_v6",
        "pre_holdout_passed": True,
        "statistical_validation_passed": True,
        "robustness_validation_passed": True,
        "pbo_passed": True,
        "cost_stress_passed": True,
        "final_holdout_status": "opened_passed",
        "paper_shadow_observations": 60,
        "paper_net_expectancy_r": 0.30,
        "paper_profit_factor": 1.60,
        "paper_max_drawdown_pct": 0.04,
        "execution_cost_calibrated": True,
    }
    challenger.update(challenger_overrides)
    return ChampionChallengerRequest.model_validate(
        {
            "champion": {
                "strategy_id": "champion-v1",
                "observation_count": 100,
                "net_expectancy_r": 0.20,
                "profit_factor": 1.40,
                "max_drawdown_pct": 0.06,
                "execution_cost_calibrated": True,
            },
            "challenger": challenger,
            "policy": {
                "min_paper_shadow_observations": 50,
                "min_expectancy_improvement_r": 0.05,
                "max_challenger_drawdown_pct": 0.10,
                "require_execution_cost_calibration": True,
            },
        }
    )


def test_failed_research_gate_can_never_reach_holdout_or_promotion_review():
    result = evaluate_champion_challenger(
        _request(
            statistical_validation_passed=False,
            final_holdout_status="sealed_not_opened",
            paper_shadow_observations=0,
            paper_net_expectancy_r=None,
            paper_profit_factor=None,
            paper_max_drawdown_pct=None,
        )
    )

    assert result.decision == "continue_research"
    assert result.eligible_for_automatic_promotion is False
    assert result.final_holdout_open_authorized is False
    assert result.gate_changes_allowed is False


def test_pre_holdout_pass_can_only_request_review_not_open_holdout():
    result = evaluate_champion_challenger(
        _request(
            final_holdout_status="sealed_not_opened",
            paper_shadow_observations=0,
            paper_net_expectancy_r=None,
            paper_profit_factor=None,
            paper_max_drawdown_pct=None,
        )
    )

    assert result.decision == "request_final_holdout_review"
    assert result.final_holdout_open_authorized is False
    assert result.requires_human_review is True


def test_final_holdout_rejection_returns_to_research():
    result = evaluate_champion_challenger(
        _request(
            final_holdout_status="opened_rejected",
            paper_shadow_observations=0,
            paper_net_expectancy_r=None,
            paper_profit_factor=None,
            paper_max_drawdown_pct=None,
        )
    )
    assert result.decision == "continue_research"


def test_final_holdout_pass_requires_sufficient_paper_shadow_sample():
    result = evaluate_champion_challenger(
        _request(
            paper_shadow_observations=20,
            paper_net_expectancy_r=0.35,
            paper_profit_factor=1.8,
            paper_max_drawdown_pct=0.03,
        )
    )
    assert result.decision == "shadow_candidate"
    assert result.eligible_for_automatic_promotion is False


def test_missing_execution_cost_calibration_keeps_champion():
    result = evaluate_champion_challenger(
        _request(execution_cost_calibrated=False)
    )
    assert result.decision == "keep_champion"
    assert "execution-cost calibration" in result.reasons[0]


def test_challenger_that_does_not_dominate_keeps_champion():
    result = evaluate_champion_challenger(
        _request(
            paper_net_expectancy_r=0.22,
            paper_profit_factor=1.30,
            paper_max_drawdown_pct=0.08,
        )
    )
    assert result.decision == "keep_champion"
    assert result.evidence["comparison_gates"]["expectancy_improvement"] is False
    assert result.evidence["comparison_gates"]["profit_factor_not_worse"] is False
    assert result.evidence["comparison_gates"]["drawdown_not_worse"] is False


def test_dominating_challenger_only_reaches_human_promotion_review():
    result = evaluate_champion_challenger(_request())

    assert result.decision == "human_promotion_review"
    assert result.eligible_for_automatic_promotion is False
    assert result.auto_apply is False
    assert result.requires_human_review is True
    assert result.gate_changes_allowed is False
    assert result.final_holdout_open_authorized is False


def test_endpoint_is_explicitly_advisory_and_preserves_correlation_id():
    client = TestClient(app)
    payload = _request().model_dump(mode="json")
    response = client.post(
        "/learn/champion-challenger",
        headers={"X-Correlation-ID": "learning-cc-123"},
        json=payload,
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert body["correlation_id"] == "learning-cc-123"
    assert body["data"]["decision"] == "human_promotion_review"
    assert body["data"]["eligible_for_automatic_promotion"] is False
    assert body["metadata"]["advisory_only"] is True
    assert body["metadata"]["promotion_authority"] is False
    assert body["metadata"]["final_holdout_authority"] is False
    assert body["metadata"]["gate_change_authority"] is False
