
import unittest
from unittest.mock import patch
from learning_agent.models import LearningRequest, Trade, AgentVote, CurrentPolicy, CurrentPolicyRisk, CurrentPolicyStrategyBias, PricePoint
from learning_agent.logic import run_learning_cycle
from learning_agent.analysis import analyze_agent_accuracy, calculate_performance_metrics

class TestAnalysisFunctions(unittest.TestCase):

    def test_analyze_agent_accuracy(self):
        trades = [
            Trade(trade_id="1", ticker="AAPL", final_verdict="buy", executed=True, pnl_pct=0.02, holding_days=2, market_regime="trending", agent_votes={"tech": AgentVote(action="buy", confidence=0.8), "fund": AgentVote(action="sell", confidence=0.6)}, timestamp="..."),
            Trade(trade_id="2", ticker="AAPL", final_verdict="buy", executed=True, pnl_pct=-0.01, holding_days=3, market_regime="ranging", agent_votes={"tech": AgentVote(action="buy", confidence=0.7), "fund": AgentVote(action="sell", confidence=0.5)}, timestamp="..."),
        ]
        accuracies = analyze_agent_accuracy(trades)
        self.assertAlmostEqual(accuracies["tech"], 0.5)
        self.assertAlmostEqual(accuracies["fund"], 0.5)

    def test_calculate_performance_metrics(self):
        trades = [
            Trade(trade_id="1", ticker="AAPL", final_verdict="buy", executed=True, pnl_pct=0.02, holding_days=2, market_regime="trending", agent_votes={}, timestamp="..."),
            Trade(trade_id="2", ticker="AAPL", final_verdict="buy", executed=True, pnl_pct=-0.01, holding_days=3, market_regime="ranging", agent_votes={}, timestamp="..."),
        ]
        metrics = calculate_performance_metrics(trades)
        self.assertAlmostEqual(metrics["win_rate"], 0.5)
        self.assertAlmostEqual(metrics["average_pnl_pct"], 0.005)
        self.assertGreater(metrics["max_drawdown"], 0)

class TestLogicFunctions(unittest.TestCase):

    def setUp(self):
        # Data that makes 'tech' agent clearly better than 'fund'
        tech_votes_win = AgentVote(action="buy", confidence=0.8)
        tech_votes_loss_correct = AgentVote(action="sell", confidence=0.8)
        tech_votes_loss_incorrect = AgentVote(action="buy", confidence=0.8)
        fund_votes = AgentVote(action="sell", confidence=0.6)

        self.trades = []
        for i in range(100):
            is_win = i % 2 == 0
            pnl = 0.02 if is_win else -0.01

            tech_vote = tech_votes_win if is_win else tech_votes_loss_correct if i % 4 == 1 else tech_votes_loss_incorrect

            self.trades.append(
                Trade(trade_id=str(i), ticker="AAPL", final_verdict="buy", executed=True, pnl_pct=pnl, holding_days=2, market_regime="trending", agent_votes={"tech": tech_vote, "fund": fund_votes}, timestamp="...")
            )

        self.price_history = {"AAPL": [PricePoint(timestamp="...", open=100, high=101, low=99, close=100, volume=1000)] * 20}
        self.current_policy = CurrentPolicy(agent_weights={"tech": 0.5, "fund": 0.5}, risk=CurrentPolicyRisk(risk_per_trade=0.01, max_position_pct=0.1, stop_loss_pct=0.05), strategy_bias=CurrentPolicyStrategyBias(preferred_regime="neutral"))
        self.request = LearningRequest(learning_mode="macro", window_size=100, trade_history=self.trades, price_history=self.price_history, current_policy=self.current_policy)

    def test_no_warmup_in_logic(self):
        request = self.request.model_copy(deep=True)
        request.trade_history = request.trade_history[:50]
        response = run_learning_cycle(request)
        self.assertEqual(response.learning_state, "active")

    def test_agent_weight_adjustment(self):
        response = run_learning_cycle(self.request)
        deltas = response.policy_deltas.agent_weights
        self.assertIn("tech", deltas)
        self.assertIn("fund", deltas)
        self.assertGreater(deltas["tech"], 0)
        self.assertLess(deltas["fund"], 0)
        self.assertAlmostEqual(sum(deltas.values()), 0.0)

    def test_risk_adjustment_on_drawdown(self):
        trades = (
            [Trade(trade_id=str(i), ticker="AAPL", final_verdict="buy", executed=True, pnl_pct=-0.08, holding_days=2, market_regime="volatile", agent_votes={}, timestamp="...") for i in range(5)] +
            [Trade(trade_id=str(i), ticker="AAPL", final_verdict="buy", executed=True, pnl_pct=0.01, holding_days=2, market_regime="trending", agent_votes={}, timestamp="...") for i in range(5, 100)]
        )
        request = self.request.model_copy(deep=True)
        request.trade_history = trades
        response = run_learning_cycle(request)
        self.assertIn("risk_per_trade", response.policy_deltas.risk)
        self.assertLess(response.policy_deltas.risk["risk_per_trade"], 0)

    def test_guardrail_recommendation(self):
        trades = (
            [Trade(trade_id=str(i), ticker="AAPL", final_verdict="buy", executed=True, pnl_pct=-0.1, holding_days=2, market_regime="volatile", agent_votes={}, timestamp="...") for i in range(5)] +
            [Trade(trade_id=str(i), ticker="AAPL", final_verdict="buy", executed=True, pnl_pct=0.01, holding_days=2, market_regime="trending", agent_votes={}, timestamp="...") for i in range(5, 100)]
        )
        request = self.request.model_copy(deep=True)
        request.trade_history = trades
        response = run_learning_cycle(request)
        self.assertIn("max_total_drawdown_pct", response.policy_deltas.guardrails)
        self.assertEqual(response.policy_deltas.guardrails["max_total_drawdown_pct"], 0.20)

if __name__ == '__main__':
    unittest.main()
