
import unittest
from fastapi.testclient import TestClient
from learning_agent.main import app
from learning_agent.models import LearningRequest, Trade
from decimal import Decimal

class TestMain(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health_check(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_update_biases_single(self):
        request_body = {
            "asset_id": "AAPL",
            "bias_delta": {
                "bull_bias": 0.1,
                "bear_bias": -0.05,
                "vol_bias": 0.02
            },
            "source": "execution",
            "timestamp": "2026-01-08T08:23:00Z"
        }
        response = self.client.post("/learning/update-biases", json=request_body)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["asset_id"], "AAPL")
        self.assertEqual(data[0]["current_bias"]["bull_bias"], 0.1)
        self.assertTrue(data[0]["updated"])

    def test_update_biases_batch(self):
        request_body = [
            {
                "asset_id": "GOOG",
                "bias_delta": {"bull_bias": 0.2},
                "source": "simulation",
                "timestamp": "2026-01-08T08:25:00Z"
            },
            {
                "asset_id": "TSLA",
                "bias_delta": {"bear_bias": 0.15},
                "source": "backtest",
                "timestamp": "2026-01-08T08:26:00Z"
            }
        ]
        response = self.client.post("/learning/update-biases", json=request_body)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(len(data), 2)
        self.assertEqual(data[0]["asset_id"], "GOOG")
        self.assertEqual(data[0]["current_bias"]["bull_bias"], 0.2)
        self.assertEqual(data[1]["asset_id"], "TSLA")
        self.assertEqual(data[1]["current_bias"]["bear_bias"], 0.15)

    def test_bias_clamping(self):
        # First, push the bias to the limit
        for _ in range(12):
             self.client.post("/learning/update-biases", json={
                "asset_id": "NVDA",
                "bias_delta": {"bull_bias": 0.1},
                "source": "execution",
                "timestamp": "2026-01-08T08:30:00Z"
            })

        # Check if it's clamped at 1.0
        response = self.client.post("/learning/update-biases", json={
            "asset_id": "NVDA",
            "bias_delta": {"bull_bias": 0.0}, # Send a zero delta to just get the current state
            "source": "execution",
            "timestamp": "2026-01-08T08:31:00Z"
        })
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data[0]["current_bias"]["bull_bias"], 1.0)

    def _create_dummy_learning_request(self, trades):
        return {
            "learning_mode": "test",
            "window_size": 10,
            "trade_history": [t.model_dump() for t in trades],
            "price_history": {},
            "current_policy": {
                "agent_weights": {},
                "risk": {"risk_per_trade": 0.01, "max_position_pct": 0.1, "stop_loss_pct": 0.05},
                "strategy_bias": {"preferred_regime": "any"}
            }
        }

    def test_learn_endpoint_with_new_schema(self):
        trades = [
            Trade(trade_id=str(i), asset_id="BTC-USD", side="buy", entry_price=Decimal("50000"),
                  exit_price=Decimal("51000"), quantity=Decimal("1"), timestamp="2026-01-08T09:00:00Z",
                  pnl_pct=Decimal("0.02")) for i in range(10)
        ]
        request_body = self._create_dummy_learning_request(trades)
        # Manually convert Decimals to strings for JSON serialization
        for trade in request_body["trade_history"]:
            trade["entry_price"] = str(trade["entry_price"])
            trade["exit_price"] = str(trade["exit_price"])
            trade["quantity"] = str(trade["quantity"])
            trade["pnl_pct"] = str(trade["pnl_pct"])

        response = self.client.post("/learn", json=request_body)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["learning_state"], "success")

    def test_bias_integration_in_learn_endpoint(self):
        # Setup: Give BTC-USD a strong positive bull_bias
        self.client.post("/learning/update-biases", json={
            "asset_id": "BTC-USD", "bias_delta": {"bull_bias": 0.5},
            "source": "execution", "timestamp": "2026-01-08T08:50:00Z"
        })

        # Create a series of profitable trades that should be pushed over the threshold by the bias
        trades = [
            Trade(trade_id=str(i), asset_id="BTC-USD", side="buy", entry_price=Decimal("50000"),
                  exit_price=Decimal("51000"), quantity=Decimal("1"), timestamp="2026-01-08T09:00:00Z",
                  pnl_pct=Decimal("0.02")) for i in range(10) # 100% win rate
        ]
        request_body = self._create_dummy_learning_request(trades)
        for trade in request_body["trade_history"]:
            trade["entry_price"] = str(trade["entry_price"])
            trade["exit_price"] = str(trade["exit_price"])
            trade["quantity"] = str(trade["quantity"])
            trade["pnl_pct"] = str(trade["pnl_pct"])

        response = self.client.post("/learn", json=request_body)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        # The positive bias should result in a positive bias_delta in the response
        self.assertIn("BTC-USD", data["policy_deltas"]["asset_biases"])
        self.assertGreater(data["policy_deltas"]["asset_biases"]["BTC-USD"], 0)

    def test_vol_bias_integration_in_learn_endpoint(self):
        # This test verifies that vol_bias can correctly turn a negative outcome into a neutral one.

        # Part 1: Establish a baseline scenario that produces a negative bias.
        # These trades have a 50% win rate and high volatility, which penalizes the score
        # enough to push it into negative territory. The max drawdown is small.
        trades = []
        for i in range(5):
            trades.append(Trade(trade_id=f"W{i}", asset_id="ETH-USD", side="buy", entry_price=Decimal("100"), exit_price=Decimal("112"), quantity=Decimal("1"), timestamp=f"2026-01-08T{10+i}:00:00Z", pnl_pct=Decimal("0.12")))
            trades.append(Trade(trade_id=f"L{i}", asset_id="ETH-USD", side="buy", entry_price=Decimal("112"), exit_price=Decimal("100.8"), quantity=Decimal("1"), timestamp=f"2026-01-08T{10+i}:30:00Z", pnl_pct=Decimal("-0.10")))

        request_body = self._create_dummy_learning_request(trades)
        for trade in request_body["trade_history"]:
            for key, value in trade.items():
                if isinstance(value, Decimal):
                    trade[key] = str(value)

        response_without_bias = self.client.post("/learn", json=request_body)
        self.assertEqual(response_without_bias.status_code, 200)
        data_without_bias = response_without_bias.json()
        self.assertIn("ETH-USD", data_without_bias["policy_deltas"]["asset_biases"])
        self.assertEqual(data_without_bias["policy_deltas"]["asset_biases"]["ETH-USD"], -0.05)

        # Part 2: Apply a positive vol_bias and verify the outcome is now neutral.
        self.client.post("/learning/update-biases", json={
            "asset_id": "ETH-USD", "bias_delta": {"vol_bias": 0.4}, # A bias strong enough to push score over lower threshold
            "source": "execution", "timestamp": "2026-01-08T09:00:00Z"
        })

        response_with_bias = self.client.post("/learn", json=request_body)
        self.assertEqual(response_with_bias.status_code, 200)
        data_with_bias = response_with_bias.json()
        # The bias should push the score into the neutral zone (no delta recommended)
        self.assertNotIn("ETH-USD", data_with_bias["policy_deltas"]["asset_biases"])


if __name__ == "__main__":
    unittest.main()
