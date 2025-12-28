
import unittest
from learning_agent.models import LearningRequest, Trade, CurrentPolicy, CurrentPolicyRisk, CurrentPolicyStrategyBias
from learning_agent.logic import run_learning_cycle, _calculate_asset_performance

class TestAssetAwareLearning(unittest.TestCase):
    def setUp(self):
        """Set up mock trade data for tests."""
        self.trades = [
            # Asset A: Consistent performer
            Trade(asset_id="A", pnl_pct=0.02, timestamp="2023-01-01T00:00:00Z", trade_id="A1", final_verdict="buy", executed=True, holding_days=1, market_regime="trending", agent_votes={}),
            Trade(asset_id="A", pnl_pct=0.03, timestamp="2023-01-02T00:00:00Z", trade_id="A2", final_verdict="buy", executed=True, holding_days=1, market_regime="trending", agent_votes={}),
            Trade(asset_id="A", pnl_pct=0.01, timestamp="2023-01-03T00:00:00Z", trade_id="A3", final_verdict="buy", executed=True, holding_days=1, market_regime="trending", agent_votes={}),
            Trade(asset_id="A", pnl_pct=-0.01, timestamp="2023-01-04T00:00:00Z", trade_id="A4", final_verdict="buy", executed=True, holding_days=1, market_regime="trending", agent_votes={}),
            Trade(asset_id="A", pnl_pct=0.02, timestamp="2023-01-05T00:00:00Z", trade_id="A5", final_verdict="buy", executed=True, holding_days=1, market_regime="trending", agent_votes={}),
            Trade(asset_id="A", pnl_pct=0.02, timestamp="2023-01-06T00:00:00Z", trade_id="A6", final_verdict="buy", executed=True, holding_days=1, market_regime="trending", agent_votes={}),
            Trade(asset_id="A", pnl_pct=0.03, timestamp="2023-01-07T00:00:00Z", trade_id="A7", final_verdict="buy", executed=True, holding_days=1, market_regime="trending", agent_votes={}),
            Trade(asset_id="A", pnl_pct=0.01, timestamp="2023-01-08T00:00:00Z", trade_id="A8", final_verdict="buy", executed=True, holding_days=1, market_regime="trending", agent_votes={}),
            Trade(asset_id="A", pnl_pct=-0.01, timestamp="2023-01-09T00:00:00Z", trade_id="A9", final_verdict="buy", executed=True, holding_days=1, market_regime="trending", agent_votes={}),
            Trade(asset_id="A", pnl_pct=0.02, timestamp="2023-01-10T00:00:00Z", trade_id="A10", final_verdict="buy", executed=True, holding_days=1, market_regime="trending", agent_votes={}),

            # Asset B: Underperformer
            Trade(asset_id="B", pnl_pct=-0.05, timestamp="2023-01-01T00:00:00Z", trade_id="B1", final_verdict="buy", executed=True, holding_days=1, market_regime="ranging", agent_votes={}),
            Trade(asset_id="B", pnl_pct=-0.02, timestamp="2023-01-02T00:00:00Z", trade_id="B2", final_verdict="buy", executed=True, holding_days=1, market_regime="ranging", agent_votes={}),
            Trade(asset_id="B", pnl_pct=0.01, timestamp="2023-01-03T00:00:00Z", trade_id="B3", final_verdict="buy", executed=True, holding_days=1, market_regime="ranging", agent_votes={}),
            Trade(asset_id="B", pnl_pct=-0.04, timestamp="2023-01-04T00:00:00Z", trade_id="B4", final_verdict="buy", executed=True, holding_days=1, market_regime="ranging", agent_votes={}),
            Trade(asset_id="B", pnl_pct=-0.03, timestamp="2023-01-05T00:00:00Z", trade_id="B5", final_verdict="buy", executed=True, holding_days=1, market_regime="ranging", agent_votes={}),
            Trade(asset_id="B", pnl_pct=-0.05, timestamp="2023-01-06T00:00:00Z", trade_id="B6", final_verdict="buy", executed=True, holding_days=1, market_regime="ranging", agent_votes={}),
            Trade(asset_id="B", pnl_pct=-0.02, timestamp="2023-01-07T00:00:00Z", trade_id="B7", final_verdict="buy", executed=True, holding_days=1, market_regime="ranging", agent_votes={}),
            Trade(asset_id="B", pnl_pct=0.01, timestamp="2023-01-08T00:00:00Z", trade_id="B8", final_verdict="buy", executed=True, holding_days=1, market_regime="ranging", agent_votes={}),
            Trade(asset_id="B", pnl_pct=-0.04, timestamp="2023-01-09T00:00:00Z", trade_id="B9", final_verdict="buy", executed=True, holding_days=1, market_regime="ranging", agent_votes={}),
            Trade(asset_id="B", pnl_pct=-0.03, timestamp="2023-01-10T00:00:00Z", trade_id="B10", final_verdict="buy", executed=True, holding_days=1, market_regime="ranging", agent_votes={}),

            # Asset C: Warmup
            Trade(asset_id="C", pnl_pct=0.01, timestamp="2023-01-01T00:00:00Z", trade_id="C1", final_verdict="buy", executed=True, holding_days=1, market_regime="trending", agent_votes={}),
        ]
        self.current_policy = CurrentPolicy(
            agent_weights={'agent_a': 0.5, 'agent_b': 0.5},
            risk=CurrentPolicyRisk(risk_per_trade=0.01, max_position_pct=0.1, stop_loss_pct=0.05),
            strategy_bias=CurrentPolicyStrategyBias(preferred_regime="neutral")
        )
        self.request = LearningRequest(
            trade_history=self.trades,
            learning_mode="test",
            window_size=10,
            price_history={},
            current_policy=self.current_policy
        )

    def test_calculate_asset_performance(self):
        """Test the asset performance calculation."""
        asset_a_trades = [t for t in self.trades if t.asset_id == "A"]
        perf = _calculate_asset_performance(asset_a_trades)
        self.assertAlmostEqual(perf["win_rate"], 0.8)
        self.assertLess(perf["max_drawdown"], 0.02)
        self.assertGreater(perf["volatility"], 0)

    def test_warmup_phase(self):
        """Test that assets with insufficient trades are in warmup."""
        response = run_learning_cycle(self.request)
        self.assertNotIn("C", response.policy_deltas.asset_biases)
        self.assertIn("Asset 'C' is in warmup", "".join(response.reasoning))

    def test_asset_bias(self):
        """Test positive and negative bias recommendations."""
        response = run_learning_cycle(self.request)
        biases = response.policy_deltas.asset_biases
        self.assertGreater(biases.get("A", 0), 0)
        self.assertLess(biases.get("B", 0), 0)

    def test_drawdown_clustering_consecutive_losses(self):
        """Test risk adjustment from consecutive losses."""
        dd_trades = self.trades + [
            Trade(asset_id="D", pnl_pct=-0.02, timestamp="2023-01-01T00:00:00Z", trade_id="D1", final_verdict="buy", executed=True, holding_days=1, market_regime="volatile", agent_votes={}) for i in range(10)
        ]
        dd_trades[21].pnl_pct = -0.03
        dd_trades[22].pnl_pct = -0.04

        request = self.request.model_copy(deep=True)
        request.trade_history = dd_trades
        response = run_learning_cycle(request)
        self.assertIn("risk_per_trade", response.policy_deltas.risk)
        self.assertLess(response.policy_deltas.risk["risk_per_trade"], 0)

    def test_drawdown_clustering_high_recent_drawdown(self):
        """Test risk adjustment from high recent drawdown."""
        dd_trades = self.trades + [
            Trade(asset_id="E", pnl_pct=0.1, timestamp="2023-01-01T00:00:00Z", trade_id="E1", final_verdict="buy", executed=True, holding_days=1, market_regime="volatile", agent_votes={}),
            Trade(asset_id="E", pnl_pct=-0.15, timestamp="2023-01-02T00:00:00Z", trade_id="E2", final_verdict="buy", executed=True, holding_days=1, market_regime="volatile", agent_votes={}),
        ] * 5

        request = self.request.model_copy(deep=True)
        request.trade_history = dd_trades
        response = run_learning_cycle(request)
        self.assertIn("risk_per_trade", response.policy_deltas.risk)
        self.assertLess(response.policy_deltas.risk["risk_per_trade"], 0)

    def test_empty_trade_history(self):
        """Test that the service handles empty trade history."""
        request = self.request.model_copy(deep=True)
        request.trade_history = []
        response = run_learning_cycle(request)
        self.assertEqual(response.learning_state, "insufficient_data")
