
import logging
from fastapi import FastAPI, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import Optional
from contextlib import asynccontextmanager

from ..core.database import get_db_session, init_db
from ..core.schemas import LearningResult, Strategy
from .models import BiasResponse, HealthResponse

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Lifespan Management ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles application startup and shutdown events.
    """
    logger.info("--- Starting Learning Agent API ---")
    try:
        init_db()
        logger.info("Database initialized successfully.")
    except Exception as e:
        logger.critical(f"CRITICAL: Failed to initialize database on startup: {e}")
    yield
    logger.info("--- Shutting down Learning Agent API ---")


# --- FastAPI App ---
app = FastAPI(
    title="Learning Agent API",
    description="Provides access to the latest pre-computed learning results and strategy biases.",
    version="2.0.0",
    lifespan=lifespan
)

@app.get(
    "/bias",
    response_model=BiasResponse,
    summary="Get Latest Strategy Bias",
    description="Retrieves the most recent learning result for a specified strategy."
)
def get_bias(
    strategy: str = Query(..., description="The name of the strategy, e.g., 'EMA_CROSS'"),
    db: Session = Depends(get_db_session)
):
    """
    This endpoint fetches the latest learning result from the database for a given strategy.
    It's a lightweight, read-only operation.
    """
    logger.info(f"Received request for strategy: {strategy}")

    # First, find the strategy ID from its name
    strategy_obj = db.query(Strategy).filter(Strategy.strategy_name == strategy).first()

    if not strategy_obj:
        logger.warning(f"Strategy '{strategy}' not found in the database.")
        raise HTTPException(status_code=404, detail=f"Strategy '{strategy}' not found.")

    # Then, find the most recent LearningResult for that strategy ID
    latest_result = db.query(LearningResult)\
        .filter(LearningResult.strategy_id == strategy_obj.id)\
        .order_by(LearningResult.created_at.desc())\
        .first()

    if not latest_result:
        logger.warning(f"No learning results found for strategy '{strategy}'.")
        raise HTTPException(status_code=404, detail=f"No learning results found for strategy '{strategy}'.")

    # The Pydantic model will automatically map the SQLAlchemy object to a JSON response
    return BiasResponse(
        strategy_name=strategy_obj.strategy_name,
        bias_adjustment=latest_result.bias_adjustment,
        risk_level=latest_result.risk_level,
        market_regime=latest_result.market_regime,
        confidence=latest_result.confidence,
        last_updated=latest_result.created_at
    )

@app.get("/health", response_model=HealthResponse, summary="Health Check")
def health():
    """A simple health check endpoint."""
    return {"status": "ok"}
