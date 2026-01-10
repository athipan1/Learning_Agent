
import unittest
from decimal import Decimal
from learning_agent.models import LearningRequest, Trade, CurrentPolicy, CurrentPolicyRisk, CurrentPolicyStrategyBias, PricePoint
from learning_agent.logic import run_learning_cycle, _calculate_asset_performance

class TestAssetAwareLearning(unittest.TestCase):
    def setUp(self):
        """Set up mock data using the new, standardized Trade model."""
        self.trades = [
            # Asset A (profitable): 100% win rate
            *[Trade(trade_id=f"A{i}", asset_id="A", side="buy", quantity=Decimal("1"), entry_price=Decimal("100"), exit_price=Decimal("101"), timestamp=f"2024-01-{10+i:02d}T10:00:00Z", pnl_pct=Decimal("0.01")) for i in range(10)],

            # Asset B (losing): 100% loss rate
            *[Trade(trade_id=f"B{i}", asset_id="B", side="sell", quantity=Decimal("1"), entry_price=Decimal("200"), exit_price=Decimal("201"), timestamp=f"2024-01-{10+i:02d}T10:00:00Z", pnl_pct=Decimal("-0.005")) for i in range(10)],

            # Asset C (warmup): Not enough trades
            *[Trade(trade_id=f"C{i}", asset_id="C", side="buy", quantity=Decimal("1"), entry_price=Decimal("50"), exit_price=Decimal("49"), timestamp=f"2024-01-{10+i:02d}T10:00:00Z", pnl_pct=Decimal("-0.02")) for i in range(5)],
        ]

        self.price_history = {} # Price history is no longer used for PNL calculation in logic

        self.current_policy = CurrentPolicy(
            agent_weights={'agent_a': 0.5, 'agent_b': 0.5},
            risk=CurrentPolicyRisk(risk_per_trade=0.01, max_position_pct=0.1, stop_loss_pct=0.05),
            strategy_bias=CurrentPolicyStrategyBias(preferred_regime="neutral")
        )

        self.request = LearningRequest(
            learning_mode="test",
            window_size=10,
            trade_history=self.trades,
            price_history=self.price_history,
            current_policy=self.current_policy,
            execution_result=None,
        )
        self.bias_state = {} # Start with a neutral bias state

    def test_calculate_asset_performance(self):
        """Test the asset performance calculation."""
        asset_a_trades = [t for t in self.trades if t.asset_id == "A"]
        pnl_pcts = [float(t.pnl_pct) for t in asset_a_trades]
        perf = _calculate_asset_performance(asset_a_trades, pnl_pcts)

        self.assertAlmostEqual(perf["win_rate"], 1.0)
        self.assertAlmostEqual(perf["max_drawdown"], 0)
        self.assertGreater(perf["volatility"], 0)

    def test_warmup_phase(self):
        """Test that assets with insufficient trades are in warmup."""
        response = run_learning_cycle(self.request, self.bias_state)
        self.assertNotIn("C", response.policy_deltas.asset_biases)
        self.assertIn("Asset 'C' is in warmup", "".join(response.reasoning))

    def test_asset_bias(self):
        """Test positive and negative bias recommendations."""
        response = run_learning_cycle(self.request, self.bias_state)
        biases = response.policy_deltas.asset_biases
        self.assertGreater(biases.get("A", 0), 0) # Profitable asset gets positive bias
        self.assertLess(biases.get("B", 0), 0)    # Losing asset gets negative bias

    def test_drawdown_clustering_consecutive_losses(self):
        """Test risk adjustment from consecutive losses."""
        # Asset D is set up to have 10 consecutive losses
        dd_trades = self.trades + [
            Trade(trade_id=f"D{i}", asset_id="D", side="buy", quantity=Decimal("1"), entry_price=Decimal("100"), exit_price=Decimal("99"), timestamp=f"2024-01-1{i}T10:00:00Z", pnl_pct=Decimal("-0.01")) for i in range(10)
        ]

        request = self.request.model_copy(deep=True)
        request.trade_history = dd_trades

        response = run_learning_cycle(request, self.bias_state)
        self.assertIn("risk_per_trade", response.policy_deltas.risk)
        self.assertLess(response.policy_deltas.risk["risk_per_trade"], 0)
        self.assertTrue(any("consecutive losses" in r for r in response.reasoning))

    def test_drawdown_clustering_high_recent_drawdown(self):
        """Test risk adjustment from high recent drawdown."""
        # Asset E will have a drawdown > 8%
        dd_trades = self.trades + [
            Trade(trade_id="E1", asset_id="E", side="buy", quantity=Decimal("1"), entry_price=Decimal("100"), exit_price=Decimal("110"), timestamp="2024-01-11T10:00:00Z", pnl_pct=Decimal("0.10")),
            Trade(trade_id="E2", asset_id="E", side="buy", quantity=Decimal("1"), entry_price=Decimal("110"), exit_price=Decimal("101"), timestamp="2024-01-12T10:00:00Z", pnl_pct=Decimal("-0.0818")),
        ] * 5 # Creates 10 trades for asset E, with a significant loss

        request = self.request.model_copy(deep=True)
        request.trade_history = dd_trades

        response = run_learning_cycle(request, self.bias_state)
        self.assertIn("risk_per_trade", response.policy_deltas.risk)
        self.assertLess(response.policy_deltas.risk["risk_per_trade"], 0)
        self.assertTrue(any("high recent drawdown" in r for r in response.reasoning))

    def test_empty_trade_history(self):
        """Test that the service handles empty trade history."""
        request = self.request.model_copy(deep=True)
        request.trade_history = []
        response = run_learning_cycle(request, self.bias_state)
        self.assertEqual(response.learning_state, "insufficient_data")
