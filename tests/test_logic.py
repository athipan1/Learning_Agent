
import unittest
from unittest.mock import patch, PropertyMock
import pandas as pd
from learning_agent.models import (
    Trade,
    PortfolioMetrics,
    Config,
    LearningRequest,
    AgentVote
)
from learning_agent.logic import run_learning_cycle
from learning_agent.analysis import analyze_market_regime, analyze_agent_accuracy

class TestAnalysisFunctions(unittest.TestCase):

    @patch('pandas.DataFrame.ta', new_callable=PropertyMock)
    def test_analyze_market_regime(self, mock_ta):
        price_history = [{'timestamp': f'2023-01-{i+1:02d}', 'close': 100} for i in range(30)]

        # Test for ranging market
        mock_ta.return_value.adx.return_value = pd.DataFrame({'ADX_14': [15.0]})
        mock_ta.return_value.atr.return_value = pd.Series([1.0])
        self.assertEqual(analyze_market_regime(price_history), "ranging")

        # Test for trending market
        mock_ta.return_value.adx.return_value = pd.DataFrame({'ADX_14': [30.0]})
        self.assertEqual(analyze_market_regime(price_history), "trending")

        # Test for volatile market
        mock_ta.return_value.atr.return_value = pd.Series([6.0])
        self.assertEqual(analyze_market_regime(price_history), "volatile")


    def test_analyze_agent_accuracy(self):
        trades = [
            Trade(timestamp="t1", action="buy", entry_price=100, exit_price=101, pnl_pct=0.01, agent_votes={"good": AgentVote(action="buy", confidence=0.8), "bad": AgentVote(action="sell", confidence=0.8)}),
            Trade(timestamp="t2", action="sell", entry_price=100, exit_price=99, pnl_pct=0.01, agent_votes={"good": AgentVote(action="sell", confidence=0.8), "bad": AgentVote(action="buy", confidence=0.8)}),
            Trade(timestamp="t3", action="buy", entry_price=100, exit_price=99, pnl_pct=-0.01, agent_votes={"good": AgentVote(action="sell", confidence=0.8), "bad": AgentVote(action="buy", confidence=0.8)}),
        ]
        accuracies = analyze_agent_accuracy(trades)
        self.assertEqual(accuracies["good"], 1.0)
        self.assertEqual(accuracies["bad"], 0.0)

class TestLogicFunctions(unittest.TestCase):

    def setUp(self):
        self.price_history = [{'timestamp': f'2023-01-{i+1:02d}', 'open': 100, 'high': 102, 'low': 98, 'close': 101, 'volume': 1000} for i in range(30)]
        trade_history = [
            Trade(timestamp="t1", action="buy", entry_price=100, exit_price=102, pnl_pct=0.02, agent_votes={"good": AgentVote(action="buy", confidence=0.8), "bad": AgentVote(action="sell", confidence=0.8)})
        ] * 30

        self.base_request = LearningRequest(
            trade_history=trade_history,
            price_history=self.price_history,
            portfolio_metrics=PortfolioMetrics(equity_curve=[100, 102, 101], max_drawdown=0.05, win_rate=0.7, profit_factor=1.5),
            current_config=Config(agent_weights={"good": 0.5, "bad": 0.5}, risk_per_trade=0.01, stop_loss_pct=0.02, max_position_pct=0.1, enable_technical_stop=True)
        )

    @patch('learning_agent.logic.analysis.analyze_market_regime')
    def test_agent_weight_adjustment(self, mock_analyze_market_regime):
        mock_analyze_market_regime.return_value = "ranging"
        response = run_learning_cycle(self.base_request)
        deltas = response.policy_deltas.agent_weights
        self.assertIn("good", deltas)
        self.assertIn("bad", deltas)
        self.assertGreater(deltas["good"], 0)
        self.assertLess(deltas["bad"], 0)

    @patch('learning_agent.logic.analysis.analyze_market_regime')
    def test_risk_increase(self, mock_analyze_market_regime):
        mock_analyze_market_regime.return_value = "ranging"
        response = run_learning_cycle(self.base_request)
        self.assertIsNotNone(response.policy_deltas.risk.risk_per_trade)
        self.assertGreater(response.policy_deltas.risk.risk_per_trade, 0)

    @patch('learning_agent.logic.analysis.analyze_market_regime')
    def test_risk_decrease_on_drawdown(self, mock_analyze_market_regime):
        mock_analyze_market_regime.return_value = "ranging"
        request = self.base_request.model_copy(deep=True)
        request.portfolio_metrics.max_drawdown = 0.20
        response = run_learning_cycle(request)
        self.assertIsNotNone(response.policy_deltas.risk.risk_per_trade)
        self.assertLess(response.policy_deltas.risk.risk_per_trade, 0)

    @patch('learning_agent.logic.analysis.analyze_market_regime')
    def test_confidence_score(self, mock_analyze_market_regime):
        mock_analyze_market_regime.return_value = "ranging"
        response = run_learning_cycle(self.base_request)
        self.assertGreaterEqual(response.confidence, 0.0)
        self.assertLessEqual(response.confidence, 1.0)

    def test_calculate_confidence_score_calculation(self):
        trade_history = [
            Trade(timestamp="t1", action="buy", entry_price=100, exit_price=102, pnl_pct=0.02, agent_votes={})
        ] * 50
        portfolio_metrics = PortfolioMetrics(equity_curve=[], max_drawdown=0.15, win_rate=0.7, profit_factor=1.5)

        # Expected calculation:
        # trade_count_factor = 50 / 100 = 0.5
        # performance_consistency = 1 - stdev([0.02]*50) = 1.0
        # drawdown_penalty = 0.9
        # confidence = 0.5 * 1.0 * 0.9 = 0.45
        from learning_agent.logic import calculate_confidence_score
        confidence = calculate_confidence_score(trade_history, portfolio_metrics)
        self.assertAlmostEqual(confidence, 0.45, places=2)


if __name__ == '__main__':
    unittest.main()
