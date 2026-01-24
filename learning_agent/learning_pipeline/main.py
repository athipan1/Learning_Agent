
import logging
import uuid
import json
import os
from datetime import datetime
from sqlalchemy.orm import Session
from ..core.database import get_db_session
from ..core.schemas import Strategy, LearningResult, LearningLog
from .data_ingestion import run_data_ingestion
from .performance_analysis import PerformanceAnalyzer, Trade

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from .trade_history import fetch_trades_from_db

# --- Core Learning Logic ---
# This is a placeholder for the strategy adjustment logic.
# It will evolve to be more sophisticated.
def generate_strategy_adjustment(performance: dict, regime: str) -> dict:
    """
    Analyzes performance and market regime to determine bias adjustments.
    """
    bias_adjustment = 0.0
    risk_level = "MEDIUM"
    confidence = 0.65

    # Example Logic: If profit factor is great, increase bias. If drawdown is high, reduce risk.
    if performance.get("profit_factor", 0) > 1.5:
        bias_adjustment = 0.1
        confidence = 0.75
    elif performance.get("profit_factor", 0) < 1.0:
        bias_adjustment = -0.1
        confidence = 0.70

    if performance.get("max_drawdown_pct", 0) > 20.0:
        risk_level = "LOW"
        bias_adjustment -= 0.05 # Further penalize risky strategies
    elif performance.get("max_drawdown_pct", 0) < 10.0:
        risk_level = "HIGH"

    return {
        "bias_adjustment": bias_adjustment,
        "risk_level": risk_level,
        "market_regime": regime, # For now, just passing it through
        "confidence": confidence
    }

def run_learning_for_strategy(db: Session, strategy: Strategy, run_id: str):
    """
    Executes the full learning pipeline for a single strategy.
    """
    logger.info(f"--- Starting learning cycle for strategy: {strategy.strategy_name} ---")

    # 1. Data Ingestion (for the assets this strategy trades)
    # The config_json should contain the list of symbols.
    symbols = strategy.config_json.get("symbols", [])
    if not symbols:
        logger.error(f"Strategy '{strategy.strategy_name}' has no symbols in config. Skipping.")
        return

    # For now, we assume we need data for all symbols for the analysis.
    # In a real scenario, we might fetch data per-symbol.
    run_data_ingestion(symbols, days_to_fetch=365) # Fetches and saves to DB

    # 2. Performance Analysis
    # Fetch trade history for the primary symbol of the strategy.
    trades = fetch_trades_from_db(strategy.strategy_name, symbols[0])
    analyzer = PerformanceAnalyzer(trades)
    performance_summary = analyzer.run_analysis()

    # 3. Market Regime Detection
    # Fetch the historical data we just ingested for the primary symbol.
    # A more robust implementation would query the DB. For now, we'll re-run ingestion logic for simplicity.
    from .data_ingestion import AlpacaDataFetcher
    from ..core.schemas import HistoricalMarketData
    import pandas as pd
    from datetime import datetime, timedelta

    # In a real scenario, you'd query the 'historical_market_data' table.
    # For now, we'll re-fetch to get a DataFrame.
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    fetcher = AlpacaDataFetcher(os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY"))
    market_data_df = fetcher.fetch_historical_data([symbols[0]], start_date, end_date)


    from .market_regime import classify_market_regime
    market_regime, regime_confidence = classify_market_regime(market_data_df)


    # 4. Generate Adjustments
    adjustment = generate_strategy_adjustment(performance_summary, market_regime)
    # Carry over the confidence from the regime detection
    adjustment['confidence'] = regime_confidence

    # 5. Store Results
    # Store the main, actionable result
    learning_result = LearningResult(
        strategy_id=strategy.id,
        **adjustment
    )
    db.add(learning_result)

    # Store the detailed log for analysis
    learning_log = LearningLog(
        run_id=run_id,
        strategy_id=strategy.id,
        log_data={
            "performance_summary": performance_summary,
            "inputs": {
                "market_regime": market_regime,
                "total_trades": len(trades)
            },
            "outputs": adjustment
        }
    )
    db.add(learning_log)

    db.commit()
    logger.info(f"Successfully saved learning results for strategy: {strategy.strategy_name}")


def learning_pipeline_main():
    """
    The main entry point for the entire learning pipeline.
    """
    run_id = str(uuid.uuid4())
    logger.info(f"Starting Learning Pipeline Run ID: {run_id}")

    with get_db_session() as db:
        # Get all active strategies from the database
        active_strategies = db.query(Strategy).filter(Strategy.is_active == True).all()

        if not active_strategies:
            logger.warning("No active strategies found in the database. Exiting.")
            # Let's add one if none exist, for demonstration purposes.
            logger.info("Adding a default 'EMA_CROSS' strategy for demonstration.")
            default_strategy = Strategy(
                strategy_name="EMA_CROSS",
                is_active=True,
                config_json={"symbols": ["BTC/USD", "ETH/USD"], "params": {"fast": 10, "slow": 50}}
            )
            db.add(default_strategy)
            db.commit()
            active_strategies = [default_strategy]

        logger.info(f"Found {len(active_strategies)} active strategies to process.")

        for strategy in active_strategies:
            try:
                run_learning_for_strategy(db, strategy, run_id)
            except Exception as e:
                db.rollback()
                logger.critical(f"CRITICAL ERROR processing strategy {strategy.strategy_name}: {e}")

    logger.info(f"Learning Pipeline Run ID: {run_id} finished.")


if __name__ == "__main__":
    # Allows the pipeline to be run directly.
    # The GitHub Action will call this script.
    learning_pipeline_main()
