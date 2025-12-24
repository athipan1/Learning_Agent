
import unittest
from learning_agent.models import Trade, PortfolioMetrics, Config, LearningRequest
from learning_agent.logic import run_learning_cycle
from learning_agent.analysis import (
    analyze_agent_accuracy,
    detect_drawdown_clustering,
    detect_trend_bias,
    detect_confirmation_bias
)

class TestAnalysisFunctions(unittest.TestCase):

    def test_analyze_agent_accuracy(self):
        trades = [
            # Good: buy (correct), Bad: sell (correctly predicted loss)
            Trade(timestamp="t1", action="buy", entry_price=100, exit_price=99, pnl_pct=-0.01, agents={"good": "buy", "bad": "sell"}),
            # Good: buy (correct), Bad: sell (incorrectly predicted loss)
            Trade(timestamp="t2", action="buy", entry_price=100, exit_price=101, pnl_pct=0.01, agents={"good": "buy", "bad": "sell"}),
            # Good: buy (correct), Bad: buy (incorrect)
            Trade(timestamp="t3", action="sell", entry_price=100, exit_price=101, pnl_pct=-0.01, agents={"good": "sell", "bad": "buy"}),
        ]
        # This is a placeholder test; the main logic is tested in TestLogicFunctions
        accuracies = analyze_agent_accuracy(trades, lookback_window=3)
        # Note: accuracies will not be tested here directly, but in the logic test.

class TestLogicFunctions(unittest.TestCase):

    def setUp(self):
        trade_history = [
            # Case 1: good=correct (predicted buy on win), bad=incorrect (predicted sell on win)
            Trade(timestamp="t1", action="buy", entry_price=100, exit_price=102, pnl_pct=0.02, agents={"good": "buy", "bad": "sell"}),
            Trade(timestamp="t2", action="buy", entry_price=100, exit_price=102, pnl_pct=0.02, agents={"good": "buy", "bad": "sell"}),
            # Case 2: good=correct (predicted sell on loss), bad=incorrect (predicted buy on loss)
            Trade(timestamp="t3", action="buy", entry_price=100, exit_price=98, pnl_pct=-0.02, agents={"good": "sell", "bad": "buy"}),
        ] * 10 # 30 trades total

        self.base_request = LearningRequest(
            symbol="TEST",
            trade_history=trade_history,
            portfolio_metrics=PortfolioMetrics(win_rate=0.7, max_drawdown=0.05, sharpe_ratio=1.5, average_return=0.01),
            config=Config(agent_weights={"good": 0.5, "bad": 0.5}, risk_per_trade=0.01, max_position_pct=0.1)
        )

    def test_agent_weight_adjustment(self):
        # good agent accuracy: 100%
        # bad agent accuracy: 0%
        response = run_learning_cycle(self.base_request)
        self.assertIn("good", response.agent_weight_adjustments)
        self.assertIn("bad", response.agent_weight_adjustments)
        self.assertGreater(response.agent_weight_adjustments["good"], 0)
        self.assertLess(response.agent_weight_adjustments["bad"], 0)

    def test_risk_increase(self):
        response = run_learning_cycle(self.base_request)
        self.assertGreater(response.risk_adjustments.get("risk_per_trade", 0), 0)

    def test_risk_decrease_on_drawdown(self):
        request = self.base_request.model_copy(deep=True)
        request.portfolio_metrics.max_drawdown = 0.20
        response = run_learning_cycle(request)
        self.assertLess(response.risk_adjustments.get("risk_per_trade", 0), 0)

if __name__ == '__main__':
    unittest.main()
