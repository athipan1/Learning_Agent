from collections import defaultdict
from typing import Dict, List, Union
import logging

from fastapi import FastAPI, Request

from .database import check_db_connection, init_db, load_bias_state, save_bias_state
from .logic import run_learning_cycle
from .market_regime import classify_market_regime
from .models import (
    BiasUpdateRequest,
    BiasUpdateResponse,
    CurrentBias,
    HealthData,
    LearningRequest,
    LearningResponse,
    MarketRegimeRequest,
    MarketRegimeResponse,
    PerformanceLearningRequest,
    PerformanceLearningResponse,
    PortfolioLearningRequest,
    PortfolioLearningResponse,
    StandardAgentResponse,
)
from .performance_learning import analyze_performance_summary
from .portfolio_learning import analyze_portfolio_audits
from .system_contract import router as system_contract_router

# --- Global State ---
# This will be populated from the database on startup.
BIAS_STATE: Dict[str, Dict[str, float]] = {}

app = FastAPI(
    title="Macro Learning Agent",
    description=(
        "An analytical AI responsible for strategic, long-horizon learning "
        "in an automated trading system."
    ),
    version="1.1.0",
)
app.include_router(system_contract_router)


@app.on_event("startup")
def on_startup():
    """
    Initialize the database and load the initial BIAS_STATE on application startup.
    """
    global BIAS_STATE
    try:
        init_db()
        BIAS_STATE = load_bias_state()
        logging.info("Successfully initialized and loaded bias state.")
    except Exception as e:
        logging.critical(
            "CRITICAL: Failed to initialize database or load state on startup: %s",
            e,
        )
        # If loading fails, start with a fresh defaultdict so the app can still run.
        BIAS_STATE = defaultdict(
            lambda: {"bull_bias": 0.0, "bear_bias": 0.0, "vol_bias": 0.0}
        )


@app.post("/learn", response_model=StandardAgentResponse[LearningResponse])
async def learn(
    request: LearningRequest,
    req: Request,
) -> StandardAgentResponse[LearningResponse]:
    """
    Analyzes trade history and portfolio metrics to generate incremental
    policy adjustments.
    """
    correlation_id = req.headers.get("X-Correlation-ID")
    # The learning cycle now uses the globally loaded and persisted BIAS_STATE.
    learning_result = await run_learning_cycle(
        request,
        BIAS_STATE,
        correlation_id=correlation_id,
    )
    return StandardAgentResponse(status="success", data=learning_result)


@app.post("/learn/portfolio", response_model=StandardAgentResponse[PortfolioLearningResponse])
async def learn_portfolio(
    request: PortfolioLearningRequest,
    req: Request,
) -> StandardAgentResponse[PortfolioLearningResponse]:
    """
    Learns from Database_Agent portfolio audit trails and recommends bucket-level
    allocation/risk adjustments for the core-satellite portfolio.
    """
    correlation_id = req.headers.get("X-Correlation-ID")
    audit_payloads = [
        audit.model_dump(mode="json") for audit in request.portfolio_audits
    ]
    result = analyze_portfolio_audits(audit_payloads)
    result["learning_mode"] = request.learning_mode
    logging.info(
        "[correlation_id=%s] portfolio learning reviewed %s audit(s)",
        correlation_id,
        result.get("portfolio_count", 0),
    )
    return StandardAgentResponse(
        status="success",
        data=PortfolioLearningResponse(**result),
    )


@app.post("/learn/performance", response_model=StandardAgentResponse[PerformanceLearningResponse])
async def learn_performance(
    request: PerformanceLearningRequest,
    req: Request,
) -> StandardAgentResponse[PerformanceLearningResponse]:
    """
    Learns from Performance_Agent TradePlan summaries and recommends guarded
    strategy-bucket, symbol-bias, and risk policy deltas.
    """
    correlation_id = req.headers.get("X-Correlation-ID")
    result = analyze_performance_summary(request)
    logging.info(
        "[correlation_id=%s] performance learning reviewed %s closed TradePlan(s)",
        correlation_id,
        result.reviewed_closed_plans,
    )
    return StandardAgentResponse(status="success", data=result)


@app.post("/market-regime", response_model=StandardAgentResponse[MarketRegimeResponse])
async def market_regime(
    request: MarketRegimeRequest,
) -> StandardAgentResponse[MarketRegimeResponse]:
    """
    Analyzes price history to determine the current market regime.
    """
    result = classify_market_regime(request.price_history)
    return StandardAgentResponse(status="success", data=result)


@app.post("/learning/update-biases", response_model=StandardAgentResponse[List[BiasUpdateResponse]])
async def update_biases(
    request: Union[List[BiasUpdateRequest], BiasUpdateRequest],
) -> StandardAgentResponse[List[BiasUpdateResponse]]:
    """
    Receives feedback from the Manager to update the agent's internal biases,
    and persists the new state to the database. Supports both single and batch updates.
    """
    updates = request if isinstance(request, list) else [request]
    responses = []

    for update in updates:
        asset_id = update.asset_id
        # Safely handle new assets by checking existence first.
        if asset_id not in BIAS_STATE:
            BIAS_STATE[asset_id] = {
                "bull_bias": 0.0,
                "bear_bias": 0.0,
                "vol_bias": 0.0,
            }
        current_asset_bias = BIAS_STATE[asset_id]

        current_asset_bias["bull_bias"] += update.bias_delta.bull_bias
        current_asset_bias["bear_bias"] += update.bias_delta.bear_bias
        current_asset_bias["vol_bias"] += update.bias_delta.vol_bias

        current_asset_bias["bull_bias"] = max(
            -1.0,
            min(1.0, current_asset_bias["bull_bias"]),
        )
        current_asset_bias["bear_bias"] = max(
            -1.0,
            min(1.0, current_asset_bias["bear_bias"]),
        )
        current_asset_bias["vol_bias"] = max(
            -1.0,
            min(1.0, current_asset_bias["vol_bias"]),
        )

        response = BiasUpdateResponse(
            asset_id=asset_id,
            current_bias=CurrentBias(**current_asset_bias),
            updated=True,
        )
        responses.append(response)

    try:
        save_bias_state(dict(BIAS_STATE))
        logging.info("Persisted updated bias state for %s asset(s).", len(updates))
    except Exception as e:
        logging.error("Failed to persist bias state after update: %s", e)

    return StandardAgentResponse(status="success", data=responses)


@app.get("/health", response_model=StandardAgentResponse[HealthData])
def health():
    try:
        db_connected = check_db_connection()
        database_status = "connected" if db_connected else "disconnected"
    except Exception as e:
        logging.warning("Health check database error: %s", e)
        database_status = "disconnected"

    data = HealthData(status="healthy", database=database_status)
    return StandardAgentResponse(status="success", data=data)
