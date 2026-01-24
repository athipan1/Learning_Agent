
import pandas as pd
import pandas_ta as ta
import logging

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IndicatorEngine:
    """
    A modular engine for calculating technical indicators on market data.
    It leverages the pandas-ta library for efficient calculations.
    """
    def __init__(self, ohlcv_df: pd.DataFrame):
        if not isinstance(ohlcv_df, pd.DataFrame) or not {'open', 'high', 'low', 'close', 'volume'}.issubset(ohlcv_df.columns):
            raise ValueError("Input must be a pandas DataFrame with OHLCV columns.")
        self.df = ohlcv_df.copy()

    def add_sma(self, length: int = 50):
        """Calculates and adds the Simple Moving Average (SMA)."""
        self.df.ta.sma(length=length, append=True)
        return self

    def add_ema(self, length: int = 200):
        """Calculates and adds the Exponential Moving Average (EMA)."""
        self.df.ta.ema(length=length, append=True)
        return self

    def add_rsi(self, length: int = 14):
        """Calculates and adds the Relative Strength Index (RSI)."""
        self.df.ta.rsi(length=length, append=True)
        return self

    def add_macd(self, fast: int = 12, slow: int = 26, signal: int = 9):
        """Calculates and adds the Moving Average Convergence Divergence (MACD)."""
        self.df.ta.macd(fast=fast, slow=slow, signal=signal, append=True)
        return self

    def add_atr(self, length: int = 14):
        """Calculates and adds the Average True Range (ATR)."""
        self.df.ta.atr(length=length, append=True)
        return self

    def add_adx(self, length: int = 14):
        """Calculates and adds the Average Directional Index (ADX)."""
        self.df.ta.adx(length=length, append=True)
        return self

    def add_volatility(self, length: int = 20):
        """Calculates and adds price volatility (standard deviation of returns)."""
        # pandas-ta doesn't have a direct 'volatility' indicator in this form,
        # so we calculate it manually.
        self.df[f'VOLATILITY_{length}'] = self.df['close'].pct_change().rolling(window=length).std() * (252**0.5) # Annualized
        return self

    def add_all_indicators(self):
        """A convenience method to add a standard set of indicators."""
        logger.info("Calculating all standard indicators...")
        self.add_sma().add_ema().add_rsi().add_macd().add_atr().add_adx().add_volatility()
        logger.info("Finished calculating indicators.")
        return self

    def get_data(self) -> pd.DataFrame:
        """Returns the DataFrame with all calculated indicators."""
        return self.df

# --- Example Usage ---
if __name__ == '__main__':
    # This block demonstrates how to use the IndicatorEngine.
    # It creates a sample OHLCV DataFrame and calculates indicators on it.

    data = {
        'open': [100, 102, 101, 103, 105, 104, 106, 108, 107, 109],
        'high': [103, 104, 103, 105, 106, 106, 108, 109, 109, 111],
        'low': [99, 101, 100, 102, 104, 103, 105, 107, 106, 108],
        'close': [102, 103, 102, 104, 105, 105, 107, 108, 108, 110],
        'volume': [1000, 1200, 1100, 1300, 1400, 1250, 1500, 1600, 1450, 1700]
    }
    sample_df = pd.DataFrame(data)

    # Initialize the engine
    engine = IndicatorEngine(sample_df)

    # Add all indicators
    engine.add_all_indicators()

    # Get the resulting DataFrame
    final_df = engine.get_data()

    print("DataFrame with Indicators:")
    print(final_df)
