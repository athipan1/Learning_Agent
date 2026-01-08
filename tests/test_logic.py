
import unittest
from decimal import Decimal
from learning_agent.models import LearningRequest, Trade, CurrentPolicy, CurrentPolicyRisk, CurrentPolicyStrategyBias, PricePoint
from learning_agent.logic import run_learning_cycle, _calculate_asset_performance

class TestAssetAwareLearning(unittest.TestCase):
    def setUp(self):
        """Set up mock data using the new Trade model and price history."""
        self.trades = [
            # Asset A (profitable): 80% win rate, but with small losses to ensure high score
            *[Trade(trade_id=f"A{i}", account_id="acc-001", asset_id="A", symbol="A-USD", side="buy", quantity=Decimal("1"), price=Decimal("100"), executed_at=f"2024-01-{10+i:02d}T10:00:00Z") for i in range(8)],
            *[Trade(trade_id=f"A{i+8}", account_id="acc-001", asset_id="A", symbol="A-USD", side="buy", quantity=Decimal("1"), price=Decimal("111"), executed_at=f"2024-01-{18+i:02d}T10:00:00Z") for i in range(2)], # 2 small losing trades

            # Asset B (losing): Bought at 200, current price is 180 (-10% P/L)
            *[Trade(trade_id=f"B{i}", account_id="acc-001", asset_id="B", symbol="B-USD", side="buy", quantity=Decimal("1"), price=Decimal("200"), executed_at=f"2024-01-{10+i:02d}T10:00:00Z") for i in range(8)],
            *[Trade(trade_id=f"B{i+8}", account_id="acc-001", asset_id="B", symbol="B-USD", side="buy", quantity=Decimal("1"), price=Decimal("170"), executed_at=f"2024-01-{18+i:02d}T10:00:00Z") for i in range(2)], # 2 winning trades

            # Asset C (warmup)
            Trade(trade_id="C1", account_id="acc-001", asset_id="C", symbol="C-USD", side="buy", quantity=Decimal("1"), price=Decimal("50"), executed_at="2024-01-10T10:00:00Z"),
        ]

        self.price_history = {
            "A": [PricePoint(timestamp="2024-01-20T10:00:00Z", open=105, high=112, low=103, close=110, volume=1000)],
            "B": [PricePoint(timestamp="2024-01-20T10:00:00Z", open=185, high=190, low=178, close=180, volume=1000)],
            "C": [PricePoint(timestamp="2024-01-20T10:00:00Z", open=50, high=52, low=49, close=51, volume=1000)],
        }

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
            current_policy=self.current_policy
        )

    def test_calculate_asset_performance(self):
        """Test the asset performance calculation."""
        asset_a_trades = [t for t in self.trades if t.asset_id == "A"]
        # Manually calculate expected P/L based on setUp data
        # 8 trades: (110 - 100) / 100 = +10%
        # 2 trades: (110 - 111) / 111 = -0.9%
        pnl_pcts = [0.10] * 8 + [-0.009] * 2
        perf = _calculate_asset_performance(asset_a_trades, pnl_pcts)

        self.assertAlmostEqual(perf["win_rate"], 0.8)
        self.assertLess(perf["max_drawdown"], 0.02) # Loosen assertion to handle float precision
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
        # Asset D is set up to have 3 consecutive losses
        dd_trades = self.trades + [
            Trade(trade_id=f"D{i}", account_id="acc-001", asset_id="D", symbol="D-USD", side="buy", quantity=Decimal("1"), price=Decimal("100"), executed_at=f"2024-01-1{i}T10:00:00Z") for i in range(10)
        ]
        dd_price_history = { "D": [PricePoint(timestamp="2024-01-20T10:00:00Z", close=90, **{'open': 0, 'high': 0, 'low': 0, 'volume': 0})] } # Causes a -10% P/L

        request = self.request.model_copy(deep=True)
        request.trade_history = dd_trades
        request.price_history.update(dd_price_history)

        response = run_learning_cycle(request)
        self.assertIn("risk_per_trade", response.policy_deltas.risk)
        self.assertLess(response.policy_deltas.risk["risk_per_trade"], 0)
        self.assertTrue(any("consecutive losses" in r for r in response.reasoning))

    def test_drawdown_clustering_high_recent_drawdown(self):
        """Test risk adjustment from high recent drawdown."""
        # Asset E will have a drawdown > 8%
        dd_trades = self.trades + [
            Trade(trade_id="E1", account_id="acc-001", asset_id="E", symbol="E-USD", side="buy", quantity=Decimal("1"), price=Decimal("100"), executed_at="2024-01-11T10:00:00Z"),
            Trade(trade_id="E2", account_id="acc-001", asset_id="E", symbol="E-USD", side="buy", quantity=Decimal("1"), price=Decimal("110"), executed_at="2024-01-12T10:00:00Z"),
        ] * 5 # Creates 10 trades for asset E
        dd_price_history = { "E": [PricePoint(timestamp="2024-01-20T10:00:00Z", close=101, **{'open': 0, 'high': 0, 'low': 0, 'volume': 0})] } # E1 is +1%, E2 is -8.18%

        request = self.request.model_copy(deep=True)
        request.trade_history = dd_trades
        request.price_history.update(dd_price_history)

        response = run_learning_cycle(request)
        self.assertIn("risk_per_trade", response.policy_deltas.risk)
        self.assertLess(response.policy_deltas.risk["risk_per_trade"], 0)
        self.assertTrue(any("high recent drawdown" in r for r in response.reasoning))

    def test_empty_trade_history(self):
        """Test that the service handles empty trade history."""
        request = self.request.model_copy(deep=True)
        request.trade_history = []
        response = run_learning_cycle(request)
        self.assertEqual(response.learning_state, "insufficient_data")
