from learning_agent.backtest_shadow_feedback import (
    BacktestShadowFeedbackRequest,
    evaluate_backtest_shadow_feedback,
)


def _request(*, observation_candidate=True, forward_ready=False):
    return BacktestShadowFeedbackRequest(
        symbol="TCOM",
        strategy_id="sma-crossover-balanced-v1",
        backtest_evidence={
            "observation_candidate": observation_candidate,
            "failed_candidate_oos_gates": ["candidate_oos_median_sharpe_ratio"],
        },
        forward_evidence={
            "forward_review_ready": forward_ready,
            "failed_gates": [] if forward_ready else ["minimum_observations"],
        },
    )


def test_unsafe_backtest_never_enters_shadow_feedback_lane():
    result = evaluate_backtest_shadow_feedback(
        _request(observation_candidate=False)
    )
    assert result.decision == "reject_challenger"
    assert result.broker_order_authorized is False


def test_safe_near_miss_waits_for_forward_evidence():
    result = evaluate_backtest_shadow_feedback(_request())
    assert result.decision == "continue_shadow"
    assert result.failed_gates == ["minimum_observations"]
    assert result.auto_promote is False


def test_strong_forward_evidence_only_requests_human_review():
    result = evaluate_backtest_shadow_feedback(_request(forward_ready=True))
    assert result.decision == "request_human_promotion_review"
    assert result.advisory_only is True
    assert result.risk_policy_change_authorized is False
    assert result.broker_order_authorized is False
