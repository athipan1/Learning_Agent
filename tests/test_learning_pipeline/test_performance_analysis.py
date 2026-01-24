
import pytest
from learning_agent.learning_pipeline.performance_analysis import PerformanceAnalyzer, Trade

# --- Test Fixtures ---

@pytest.fixture
def sample_trades_profitable():
    """A series of trades representing a profitable strategy."""
    return [
        Trade(entry_price=100, exit_price=110, side='LONG'),  # +10%
        Trade(entry_price=110, exit_price=105, side='LONG'),  # -4.5%
        Trade(entry_price=105, exit_price=120, side='LONG'),  # +14.3%
        Trade(entry_price=120, exit_price=115, side='SHORT'), # +4.2%
    ]

@pytest.fixture
def sample_trades_unprofitable():
    """A series of trades representing an unprofitable strategy."""
    return [
        Trade(entry_price=100, exit_price=90, side='LONG'),   # -10%
        Trade(entry_price=90, exit_price=95, side='LONG'),    # +5.6%
        Trade(entry_price=95, exit_price=85, side='LONG'),    # -10.5%
    ]

@pytest.fixture
def trades_all_wins():
    """All winning trades."""
    return [
        Trade(entry_price=100, exit_price=110, side='LONG'),
        Trade(entry_price=120, exit_price=110, side='SHORT'),
    ]

@pytest.fixture
def trades_no_trades():
    """An empty list of trades."""
    return []

# --- Tests for Win Rate ---

def test_win_rate_mixed(sample_trades_profitable):
    analyzer = PerformanceAnalyzer(sample_trades_profitable)
    assert analyzer.calculate_win_rate() == pytest.approx(75.0)

def test_win_rate_all_wins(trades_all_wins):
    analyzer = PerformanceAnalyzer(trades_all_wins)
    assert analyzer.calculate_win_rate() == pytest.approx(100.0)

def test_win_rate_no_trades(trades_no_trades):
    analyzer = PerformanceAnalyzer(trades_no_trades)
    assert analyzer.calculate_win_rate() == 0.0

# --- Tests for Profit Factor ---

def test_profit_factor_profitable(sample_trades_profitable):
    analyzer = PerformanceAnalyzer(sample_trades_profitable)
    # Gross Profit: 0.10 + 0.1428 + 0.0416 = 0.2844
    # Gross Loss: 0.0454
    # Profit Factor: 0.2844 / 0.0454 = 6.26
    assert analyzer.calculate_profit_factor() == pytest.approx(6.26, rel=1e-2)

def test_profit_factor_unprofitable(sample_trades_unprofitable):
    analyzer = PerformanceAnalyzer(sample_trades_unprofitable)
    # Gross Profit: 0.0555
    # Gross Loss: 0.10 + 0.1052 = 0.2052
    # Profit Factor: 0.0555 / 0.2052 = 0.270
    assert analyzer.calculate_profit_factor() == pytest.approx(0.270, rel=1e-2)

def test_profit_factor_all_wins(trades_all_wins):
    analyzer = PerformanceAnalyzer(trades_all_wins)
    assert analyzer.calculate_profit_factor() == float('inf')

# --- Tests for Max Drawdown ---

def test_max_drawdown_typical_case():
    trades = [
        Trade(100, 110, 'LONG'), # PnL: +0.10, Equity: 1.10
        Trade(110, 100, 'LONG'), # PnL: -0.09, Equity: 1.01
        Trade(100, 120, 'LONG'), # PnL: +0.20, Equity: 1.21 (New Peak)
        Trade(120, 90, 'LONG'),  # PnL: -0.25, Equity: 0.96
        Trade(90, 110, 'LONG'),   # PnL: +0.22, Equity: 1.18
    ]
    analyzer = PerformanceAnalyzer(trades)
    # Equity Curve: 1.0, 1.10, 1.01, 1.21 (Peak), 0.96 (Trough), 1.18
    # Max Drawdown: (1.21 - 0.96) / 1.21 = 0.2066
    assert analyzer.calculate_max_drawdown() == pytest.approx(20.66, rel=1e-2)

def test_max_drawdown_no_drawdown(trades_all_wins):
    analyzer = PerformanceAnalyzer(trades_all_wins)
    # Equity curve is always increasing
    assert analyzer.calculate_max_drawdown() == 0.0

def test_max_drawdown_immediate_loss():
    trades = [Trade(100, 90, 'LONG')] # -10%
    analyzer = PerformanceAnalyzer(trades)
    # Equity Curve: 1.0 (Peak), 0.90 (Trough)
    # Max Drawdown: (1.0 - 0.90) / 1.0 = 0.10
    assert analyzer.calculate_max_drawdown() == pytest.approx(10.0)

def test_max_drawdown_no_trades(trades_no_trades):
    analyzer = PerformanceAnalyzer(trades_no_trades)
    assert analyzer.calculate_max_drawdown() == 0.0
