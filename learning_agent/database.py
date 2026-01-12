
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from .schemas import Base, BiasState
from typing import Dict
from collections import defaultdict
import logging

# --- Database Configuration ---
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost/learning_agent_db")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is not set.")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

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
