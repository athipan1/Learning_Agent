from fastapi.testclient import TestClient

from learning_agent.main import app
from learning_agent.models import PerformanceLearningRequest, PerformanceSummaryPayload
from learning_agent.performance_learning import analyze_performance_summary

client = TestClient(app)


def summary_payload(closed=20):
    return PerformanceSummaryPayload(
        period="30d",
        trade_plan_count=closed,
        closed_plan_count=closed,
        winning_plans=12,
        losing_plans=8,
        win_rate=0.60,
        gross_profit=500,
        gross_loss=-200,
        net_pnl=300,
        return_pct=0.03,
        expectancy=15,
        profit_factor=2.5,
        average_win=41.67,
        average_loss=-25,
        best_strategy_bucket="value_rebound",
        worst_strategy_bucket="news_momentum",
        by_strategy_bucket={
            "value_rebound": {
                "trade_plan_count": 10,
                "closed_plan_count": 10,
                "win_rate": 0.70,
                "gross_profit": 400,
                "gross_loss": -100,
                "net_pnl": 300,
                "expectancy": 30,
                "profit_factor": 4.0,
            },
            "news_momentum": {
                "trade_plan_count": 10,
                "closed_plan_count": 10,
                "win_rate": 0.30,
                "gross_profit": 100,
                "gross_loss": -250,
                "net_pnl": -150,
                "expectancy": -15,
                "profit_factor": 0.4,
            },
        },
        by_symbol={
            "AAPL": {
                "trade_plan_count": 10,
                "closed_plan_count": 10,
                "win_rate": 0.70,
                "gross_profit": 300,
                "gross_loss": -100,
                "net_pnl": 200,
                "expectancy": 20,
                "profit_factor": 3.0,
            },
            "MSFT": {
                "trade_plan_count": 10,
                "closed_plan_count": 10,
                "win_rate": 0.30,
                "gross_profit": 50,
                "gross_loss": -150,
                "net_pnl": -100,
                "expectancy": -10,
                "profit_factor": 0.33,
            },
        },
    )


def test_analyze_performance_summary_recommends_bucket_and_symbol_deltas():
    request = PerformanceLearningRequest(
        account_id="1",
        performance_summary=summary_payload(),
        min_closed_plans=5,
    )

    response = analyze_performance_summary(request)

    assert response.learning_state == "success"
    assert response.reviewed_closed_plans == 20
    assert response.policy_deltas["strategy_bucket_weights"]["value_rebound"] > 0
    assert response.policy_deltas["strategy_bucket_weights"]["news_momentum"] < 0
    assert response.policy_deltas["asset_biases"]["AAPL"] > 0
    assert response.policy_deltas["asset_biases"]["MSFT"] < 0
    assert response.policy_deltas["guardrails"]["auto_apply"] is False
    assert response.confidence_score > 0.5


def test_analyze_performance_summary_warmup_when_insufficient_data():
    request = PerformanceLearningRequest(
        account_id="1",
        performance_summary=summary_payload(closed=2),
        min_closed_plans=5,
    )

    response = analyze_performance_summary(request)

    assert response.learning_state == "warmup"
    assert response.policy_deltas["strategy_bucket_weights"] == {}
    assert "below minimum" in response.reasoning[0]


def test_analyze_performance_summary_reduces_risk_for_weak_overall_performance():
    summary = summary_payload()
    summary.net_pnl = -250
    summary.profit_factor = 0.7
    summary.return_pct = -0.025
    request = PerformanceLearningRequest(account_id="1", performance_summary=summary, min_closed_plans=5)

    response = analyze_performance_summary(request)

    assert response.policy_deltas["risk"]["risk_per_trade"] < 0
    assert any("reducing risk_per_trade" in reason for reason in response.reasoning)


def test_learn_performance_endpoint():
    response = client.post(
        "/learn/performance",
        json={
            "account_id": "1",
            "learning_mode": "performance_summary_review",
            "min_closed_plans": 5,
            "performance_summary": summary_payload().model_dump(mode="json"),
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["data"]["learning_state"] == "success"
    assert payload["data"]["policy_deltas"]["strategy_bucket_weights"]["value_rebound"] > 0
