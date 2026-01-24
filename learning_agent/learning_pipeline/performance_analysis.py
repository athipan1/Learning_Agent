
import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Data Model (Pydantic-like structure for clarity) ---
# In a real scenario, this might be a Pydantic model shared across the project.
class Trade:
    def __init__(self, entry_price: float, exit_price: float, side: str):
        self.entry_price = entry_price
        self.exit_price = exit_price
        self.side = side.upper()  # 'LONG' or 'SHORT'

    def calculate_pnl_pct(self) -> float:
        if self.side == 'LONG':
            return (self.exit_price - self.entry_price) / self.entry_price
        elif self.side == 'SHORT':
            return (self.entry_price - self.exit_price) / self.entry_price
        else:
            return 0.0

class PerformanceAnalyzer:
    """
    Calculates key performance metrics from a history of trades.
    """
    def __init__(self, trades: List[Trade]):
        if not trades:
            logger.warning("No trades provided for analysis. Metrics will be zero.")
            self.trades = []
            self.pnl_pcts = []
        else:
            self.trades = trades
            self.pnl_pcts = [t.calculate_pnl_pct() for t in self.trades]

    def calculate_win_rate(self) -> float:
        """Calculates the percentage of trades that were profitable."""
        if not self.pnl_pcts:
            return 0.0
        wins = sum(1 for pnl in self.pnl_pcts if pnl > 0)
        return (wins / len(self.pnl_pcts)) * 100

    def calculate_profit_factor(self) -> float:
        """
        Calculates the ratio of gross profits to gross losses.
        A value > 1 indicates a profitable system.
        """
        if not self.pnl_pcts:
            return 0.0

        gross_profit = sum(pnl for pnl in self.pnl_pcts if pnl > 0)
        gross_loss = abs(sum(pnl for pnl in self.pnl_pcts if pnl < 0))

        if gross_loss == 0:
            return float('inf')  # All trades were profitable

        return gross_profit / gross_loss

    def calculate_max_drawdown(self) -> float:
        """
        Calculates the largest peak-to-trough decline in the equity curve.
        Returns the value as a percentage (e.g., 15.5 for 15.5%).
        """
        if not self.pnl_pcts:
            return 0.0

        # Create an equity curve starting with a hypothetical 1.0 unit of capital
        equity_curve = [1.0] + [1.0 + sum(self.pnl_pcts[:i+1]) for i, _ in enumerate(self.pnl_pcts)]

        peak = equity_curve[0]
        max_drawdown = 0.0

        for value in equity_curve:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        return max_drawdown * 100

    def run_analysis(self) -> Dict[str, Any]:
        """
        Runs all performance calculations and returns a summary dictionary.
        """
        logger.info(f"Running performance analysis on {len(self.trades)} trades.")

        summary = {
            "total_trades": len(self.trades),
            "win_rate_pct": self.calculate_win_rate(),
            "profit_factor": self.calculate_profit_factor(),
            "max_drawdown_pct": self.calculate_max_drawdown(),
        }

        logger.info(f"Analysis complete: {summary}")
        return summary

# --- Example Usage ---
if __name__ == '__main__':
    # Sample trade data for demonstration
    sample_trades = [
        Trade(entry_price=100, exit_price=105, side='LONG'),   # Win
        Trade(entry_price=105, exit_price=102, side='LONG'),   # Loss
        Trade(entry_price=102, exit_price=110, side='LONG'),   # Win
        Trade(entry_price=110, exit_price=100, side='SHORT'),  # Loss
        Trade(entry_price=100, exit_price=95, side='SHORT'),   # Win
        Trade(entry_price=95, exit_price=98, side='SHORT'),    # Loss
        Trade(entry_price=98, exit_price=115, side='LONG'),   # Win
    ]

    analyzer = PerformanceAnalyzer(sample_trades)
    performance_summary = analyzer.run_analysis()

    print("\n--- Performance Summary ---")
    print(performance_summary)
