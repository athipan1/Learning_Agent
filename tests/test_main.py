import unittest
import json
from learning_agent.main import process_trades_json
from learning_agent.schemas import Trade, AgentVote, OrchestratorConfig

class TestLearningAgent(unittest.TestCase):

    def setUp(self):
        self.sample_trades = [
            {
                "trade_id": f"trade_{i}",
                "ticker": "AAPL",
                "final_verdict": "buy" if i % 2 == 0 else "sell",
                "executed": True,
                "entry_price": 150.0,
                "exit_price": 155.0 if i % 2 == 0 else 145.0,
                "pnl_pct": 3.33 if i % 2 == 0 else -3.33,
                "holding_days": 5,
                "market_regime": "trending" if i % 3 == 0 else "range",
                "agent_votes": {
                    "technical": {
                        "action": "buy",
                        "confidence_score": 0.8,
                        "version": "1.0"
                    },
                    "fundamental": {
                        "action": "hold",
                        "confidence_score": 0.6,
                        "version": "1.0"
                    }
                },
                "orchestrator_config": {
                    "agent_weights": {
                        "technical": 0.7,
                        "fundamental": 0.3
                    },
                    "confidence_threshold": {
                        "buy": 0.6,
                        "sell": 0.6
                    }
                },
                "timestamp": "2023-10-27T10:00:00Z"
            } for i in range(35)
        ]

    def test_monitoring_period(self):
        trades_json = json.dumps(self.sample_trades[:20])
        result_json = process_trades_json(trades_json)
        result = json.loads(result_json)
        self.assertEqual(result["summary"]["key_issue"], "Monitoring period (less than 30 trades)")

    def test_agent_weight_adjustment(self):
        trades_json = json.dumps(self.sample_trades)
        result_json = process_trades_json(trades_json)
        result = json.loads(result_json)
        # In this scenario, the technical agent always makes the right call, and fundamental always makes the wrong one
        self.assertGreater(result["agent_weight_adjustments"]["technical"], 0)
        self.assertLess(result["agent_weight_adjustments"]["fundamental"], 0)

    def test_high_drawdown_risk_adjustment(self):
        # Create a scenario with high drawdown by forcing a long losing streak
        trades = self.sample_trades.copy()
        for i in range(15):
            trades[i]['pnl_pct'] = -5.0
            trades[i]['exit_price'] = 142.5

        trades_json = json.dumps(trades)
        result_json = process_trades_json(trades_json)
        result = json.loads(result_json)

        self.assertEqual(result['summary']['key_issue'], 'High max drawdown')
        self.assertLess(result['risk_adjustments']['risk_per_trade'], 0)
        self.assertLess(result['risk_adjustments']['max_position_pct'], 0)


if __name__ == '__main__':
    unittest.main()
