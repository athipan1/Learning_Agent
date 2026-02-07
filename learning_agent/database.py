
import os
from datetime import datetime, timezone
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base
from .schemas import Base, BiasState, Order
from .models import Trade
from typing import Dict, List, Optional, Union
from collections import defaultdict
import logging

# --- Database Configuration ---
USE_SQLITE = os.getenv("USE_SQLITE", "false").lower() in ("true", "1", "t")

if USE_SQLITE:
    DATABASE_URL = "sqlite:///./learning_agent.db"
    logging.info(f"Using SQLite database: {DATABASE_URL}")
else:
    DATABASE_URL = os.getenv("DATABASE_URL")

if not USE_SQLITE and not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is not set and USE_SQLITE is false.")

SQLALCHEMY_ECHO = os.getenv("SQLALCHEMY_ECHO", "False").lower() in ("true", "1", "t")

# SQLite needs check_same_thread=False for multi-threaded access in FastAPI
if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False}, echo=SQLALCHEMY_ECHO)
else:
    engine = create_engine(DATABASE_URL, echo=SQLALCHEMY_ECHO)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def check_db_connection() -> bool:
    """
    Checks if the database is reachable.
    """
    db = SessionLocal()
    try:
        # For both SQLite and PostgreSQL, this simple query should work.
        db.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logging.error(f"Database connection check failed: {e}")
        return False
    finally:
        db.close()

def get_historical_trades(account_id: Union[int, str], asset_id: Optional[str] = None) -> List[Trade]:
    """
    Fetches executed trades directly from the PostgreSQL database.
    This replaces the need for an external API call to the Database Agent for trade history.
    """
    db = SessionLocal()
    try:
        # PostgreSQL-compatible query to fetch executed orders for the given account
        query = db.query(Order).filter(Order.account_id == int(account_id), Order.status == 'executed')

        if asset_id:
            query = query.filter(Order.symbol == asset_id)

        # Order by timestamp descending to get the most recent trades first
        db_orders = query.order_by(Order.timestamp.desc()).all()

        trades = []
        for order in db_orders:
            # Map the database Order model to the Pydantic Trade model used by the logic.
            # Field mapping: 'symbol' from DB becomes 'asset_id' in the model.
            trades.append(Trade(
                trade_id=str(order.order_id),
                account_id=str(order.account_id),
                asset_id=order.symbol,
                side=order.order_type.lower(),
                quantity=order.quantity,
                entry_price=order.price,
                exit_price=order.price, # The orders table doesn't distinguish entry/exit prices
                executed_at=order.timestamp.isoformat() if order.timestamp else datetime.now(timezone.utc).isoformat(),
                pnl_pct=0.0 # PnL percentage is not stored in the basic orders table
            ))

        logging.info(f"Directly fetched {len(trades)} trades from PostgreSQL for account {account_id}, asset {asset_id}.")
        return trades
    except Exception as e:
        logging.error(f"Failed to fetch trades from PostgreSQL: {e}")
        return []
    finally:
        db.close()

def init_db():
    """
    Initializes the database by creating tables based on the SQLAlchemy models.
    """
    try:
        print("Initializing database...")
        Base.metadata.create_all(bind=engine)
        print("Database initialized successfully.")
    except Exception as e:
        logging.error(f"Error initializing database: {e}")
        raise

def load_bias_state() -> Dict[str, Dict[str, float]]:
    """
    Loads the entire BIAS_STATE from the PostgreSQL database.
    Returns a defaultdict compatible with the application's in-memory state.
    """
    db = SessionLocal()
    try:
        states = db.query(BiasState).all()

        # Use defaultdict to maintain the same behavior as the original in-memory state
        loaded_state = defaultdict(lambda: {
            "bull_bias": 0.0,
            "bear_bias": 0.0,
            "vol_bias": 0.0
        })

        for state in states:
            loaded_state[state.asset_id] = state.to_dict()

        logging.info(f"Loaded bias state for {len(loaded_state)} assets from the database.")
        return loaded_state
    except Exception as e:
        logging.error(f"Failed to load bias state from database: {e}")
        # Return a fresh, empty state to ensure the application can start even if the DB is unavailable
        return defaultdict(lambda: {"bull_bias": 0.0, "bear_bias": 0.0, "vol_bias": 0.0})
    finally:
        db.close()

def save_bias_state(bias_state: Dict[str, Dict[str, float]]):
    """
    Saves the provided BIAS_STATE to the PostgreSQL database.
    This function performs an "upsert" operation for each asset.
    """
    db = SessionLocal()
    try:
        for asset_id, biases in bias_state.items():
            # Check if the record already exists
            db_state = db.query(BiasState).filter(BiasState.asset_id == asset_id).first()

            if db_state:
                # Update existing record
                db_state.bull_bias = biases["bull_bias"]
                db_state.bear_bias = biases["bear_bias"]
                db_state.vol_bias = biases["vol_bias"]
            else:
                # Create new record
                db_state = BiasState(
                    asset_id=asset_id,
                    bull_bias=biases["bull_bias"],
                    bear_bias=biases["bear_bias"],
                    vol_bias=biases["vol_bias"]
                )
                db.add(db_state)

        db.commit()
        logging.info(f"Successfully saved bias state for {len(bias_state)} assets.")
    except Exception as e:
        logging.error(f"Failed to save bias state to database: {e}")
        db.rollback()
        raise
    finally:
        db.close()
