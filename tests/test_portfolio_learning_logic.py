from learning_agent.portfolio_learning import analyze_portfolio_audits


def make_audit(core_value, value_value, news_value):
    return {
        "account_id": "acc123",
        "selected_positions": [
            {"symbol": "KO", "strategy_bucket": "core_dividend"},
            {"symbol": "ACGL", "strategy_bucket": "value_rebound"},
            {"symbol": "MSFT", "strategy_bucket": "news_momentum"},
        ],
        "risk_approvals": [
            {"symbol": "KO", "strategy_bucket": "core_dividend", "approved": True},
            {"symbol": "ACGL", "strategy_bucket": "value_rebound", "approved": True},
            {"symbol": "MSFT", "strategy_bucket": "news_momentum", "approved": True},
        ],
        "execution_orders": [
            {"symbol": "KO", "strategy_bucket": "core_dividend", "status": "executed", "return_pct": core_value},
            {"symbol": "ACGL", "strategy_bucket": "value_rebound", "status": "executed", "return_pct": value_value},
            {"symbol": "MSFT", "strategy_bucket": "news_momentum", "status": "executed", "return_pct": news_value},
        ],
    }


def test_portfolio_bucket_learning_recommends_deltas():
    result = analyze_portfolio_audits([
        make_audit(0.02, -0.02, -0.03),
        make_audit(0.015, -0.01, -0.025),
        make_audit(0.03, 0.0, -0.02),
    ])

    assert result["learning_state"] == "success"
    assert result["portfolio_count"] == 3
    assert result["approval_rate"] == 1.0
    assert result["execution_rate"] == 1.0
    assert result["bucket_metrics"]["core_dividend"]["executed_count"] == 3
    assert result["policy_deltas"]["bucket_weight_deltas"]["core_dividend"] == 0.02
    assert result["policy_deltas"]["bucket_weight_deltas"]["news_momentum"] == -0.02


def test_portfolio_bucket_learning_handles_empty_input():
    result = analyze_portfolio_audits([])
    assert result["learning_state"] == "insufficient_data"
    assert result["confidence_score"] == 0.0
    assert result["bucket_metrics"] == {}
