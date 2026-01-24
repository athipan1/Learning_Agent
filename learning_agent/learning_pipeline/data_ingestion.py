
import os
import logging
import pandas as pd
from datetime import datetime, timedelta
from alpaca_trade_api.rest import REST, TimeFrame
from sqlalchemy.orm import Session
from ..core.database import get_db_session
from ..core.schemas import HistoricalMarketData

# --- Configuration ---
# It's crucial to use environment variables for API keys in a real application.
# The GitHub Action will inject these as secrets.
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_PAPER_TRADING = os.getenv("ALPACA_PAPER_TRADING", "true").lower() == "true"

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlpacaDataFetcher:
    """
    Handles fetching historical market data from the Alpaca API.
    """
    def __init__(self, api_key: str, secret_key: str, paper: bool = True):
        if not api_key or not secret_key:
            raise ValueError("Alpaca API key and secret key must be provided.")
        try:
            self.api = REST(key_id=api_key, secret_key=secret_key, paper=paper)
            logger.info("Successfully connected to Alpaca API.")
        except Exception as e:
            logger.critical(f"Failed to connect to Alpaca API: {e}")
            raise

    def fetch_historical_data(
        self,
        symbols: list[str],
        start_date: datetime,
        end_date: datetime,
        timeframe: TimeFrame = TimeFrame.Day
    ) -> pd.DataFrame:
        """
        Fetches historical OHLCV data for multiple symbols and combines them.
        """
        try:
            logger.info(f"Fetching data for {symbols} from {start_date} to {end_date}.")
            # Note: Alpaca's get_bars is inclusive of the start and end dates.
            df = self.api.get_bars(
                symbols,
                timeframe,
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d')
            ).df
            logger.info(f"Successfully fetched {len(df)} data points.")
            return df
        except Exception as e:
            logger.error(f"Error fetching historical data from Alpaca: {e}")
            return pd.DataFrame() # Return empty dataframe on error

def save_market_data(db: Session, data: pd.DataFrame):
    """
    Saves the fetched market data into the historical_market_data table.
    Performs an "upsert" to avoid duplicate entries.
    """
    if data.empty:
        logger.warning("Dataframe is empty. Nothing to save.")
        return

    # The Alpaca SDK multi-symbol get_bars returns a multi-index DataFrame.
    # We reset the index to make 'symbol' and 'timestamp' regular columns.
    if isinstance(data.index, pd.MultiIndex):
        data = data.reset_index()

    records = data.to_dict(orient='records')

    # This is a simple insert. For a production system, a more efficient
    # bulk upsert mechanism (like ON CONFLICT DO UPDATE) would be better.
    # SQLAlchemy's ORM doesn't have a universal "upsert" command, so we
    # perform a simple insert and rely on the DB's unique constraint.
    try:
        db.bulk_insert_mappings(HistoricalMarketData, records)
        db.commit()
        logger.info(f"Successfully saved {len(records)} records to the database.")
    except Exception as e:
        db.rollback()
        logger.error(f"Error saving market data to database. Potential duplicates. Error: {e}")


def run_data_ingestion(symbols: list[str], days_to_fetch: int = 365):
    """
    Main entry point for the data ingestion process.
    """
    logger.info("Starting data ingestion pipeline...")
    fetcher = AlpacaDataFetcher(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_PAPER_TRADING)

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_to_fetch)

    market_data = fetcher.fetch_historical_data(symbols, start_date, end_date)

    if not market_data.empty:
        with get_db_session() as db:
            save_market_data(db, market_data)

    logger.info("Data ingestion pipeline finished.")

if __name__ == '__main__':
    # This allows running the ingestion script directly for testing or manual runs.
    # Example usage:
    # ALPACA_API_KEY="YOUR_KEY" ALPACA_SECRET_KEY="YOUR_SECRET" python -m learning_agent.learning_pipeline.data_ingestion

    # In a real run (e.g., from the main pipeline), we'd get symbols from the 'strategies' table.
    test_symbols = ['AAPL', 'GOOGL', 'MSFT']
    run_data_ingestion(test_symbols, days_to_fetch=7)
