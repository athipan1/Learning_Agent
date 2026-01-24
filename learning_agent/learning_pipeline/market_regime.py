
import pandas as pd
import pandas_ta as ta
import logging
from typing import Dict, Any, Tuple

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def classify_market_regime(historical_data: pd.DataFrame) -> Tuple[str, float]:
    """
    Analyzes historical market data to classify the current market regime.

    Args:
        historical_data: A pandas DataFrame with OHLCV columns, indexed by timestamp.

    Returns:
        A tuple containing the regime string (e.g., "uptrend") and a confidence score (0.0 to 1.0).
    """
    if len(historical_data) < 200:
        logger.warning("Insufficient data for market regime classification (< 200 points).")
        return "undefined", 0.0

    df = historical_data.copy()

    # --- 1. Calculate Indicators ---
    try:
        df.ta.ema(length=200, append=True)
        df.ta.adx(length=14, append=True)
        df.ta.atr(length=14, append=True)

        # Drop rows with NaN values resulting from indicator calculations
        df.dropna(inplace=True)

        if len(df) < 20: # Need some runway for slope and ratio calculations
             return "undefined", 0.0

    except Exception as e:
        logger.error(f"Error calculating indicators for market regime: {e}")
        return "undefined", 0.0

    # --- 2. Extract Latest Indicator Values ---
    latest = df.iloc[-1]
    prev = df.iloc[-6] # 5 periods ago

    latest_price = latest['close']
    latest_ema_200 = latest['EMA_200']
    latest_adx = latest['ADX_14']
    adx_5_periods_ago = prev['ADX_14']

    # Calculate EMA slope
    ema_slope = df['EMA_200'].iloc[-1] - df['EMA_200'].iloc[-3]
    ema_slope_3_periods_ago = df['EMA_200'].iloc[-4] - df['EMA_200'].iloc[-6]

    # Calculate ATR ratio for volatility
    atr_mean_20 = df['ATRr_14'].rolling(window=20).mean().iloc[-1]
    atr_ratio = latest['ATRr_14'] / atr_mean_20 if atr_mean_20 > 0 else 1.0

    # --- 3. Apply Scoring Logic ---
    scores = { "uptrend": 0.0, "downtrend": 0.0, "ranging": 0.0, "volatile": 0.0 }

    # Uptrend
    if latest_adx > 25 and ema_slope > 0 and latest_price > latest_ema_200:
        scores["uptrend"] = 1.0

    # Downtrend
    if latest_adx > 25 and ema_slope < 0 and latest_price < latest_ema_200:
        scores["downtrend"] = 1.0

    # Ranging
    price_proximity_pct = abs(latest_price - latest_ema_200) / latest_ema_200
    if latest_adx < 20 and price_proximity_pct < 0.02:
        scores["ranging"] = 1.0

    # Volatile
    adx_accelerating = latest_adx > (adx_5_periods_ago + 5)
    ema_flipped = (ema_slope > 0 and ema_slope_3_periods_ago < 0) or (ema_slope < 0 and ema_slope_3_periods_ago > 0)
    if atr_ratio >= 1.5 or adx_accelerating or ema_flipped:
        scores["volatile"] = 1.0

    # --- 4. Determine Final Regime ---
    # Simplified logic: find the highest score. If tied or low, it's "undefined".
    sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)

    winner, runner_up = sorted_scores[0], sorted_scores[1]

    if winner[1] == 0: # No regime scored
        return "undefined", 0.0

    if winner[1] == runner_up[1]: # A tie
        return "undefined", 0.5

    final_regime = winner[0]
    confidence = winner[1] # Simple confidence for now

    return final_regime, confidence
