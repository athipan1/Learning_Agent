from collections import defaultdict
from typing import Dict, List, Union
import logging

from fastapi import FastAPI, Request

from .champion_challenger_routes import router as champion_challenger_router
from .database import check_db_connection, init_db, load_bias_state, save_bias_state
from .logic import run_learning_cycle
from .market_regime import classify_market_regime
from .models import (
    BiasUpdateRequest,
    BiasUpdateResponse,
    CurrentBias,
    HealthData,
    LEARNING_OUTCOME_VERSION,
    LEARNING_SERVICE_VERSION,
    LearningOutcomeRequest,
    LearningOutcomeResponse,
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
from .outcome_learning import analyze_learning_outcomes
from .performance_learning import analyze_performance_summary
from .portfolio_learning import analyze_portfolio_audits
from .shadow_review_routes import router as shadow_review_router
from .system_contract import router as system_contract_router

BIAS_STATE: Dict[str, Dict[str, float]] = {}

app = FastAPI(
    title="Macro Learning Agent",
    description=(
        "An analytical AI responsible for strategic, long-horizon learning "
        "in an automated trading system."
    ),
    version=LEARNING_SERVICE_VERSION,
)
app.include_router(system_contract_router)
app.include_router(champion_challenger_router)
app.include_router(shadow_review_router)


@app.on_event("startup")
def on_startup():
    """Initialize the database and load the persisted bias state."""
    global BIAS_STATE
    try:
        init_db()
        BIAS_STATE = load_bias_state()
        logging.info("Successfully initialized and loaded bias state.")
    except Exception as exc:
        logging.critical(
            "CRITICAL: Failed to initialize database or load state on startup: %s",
            exc,
        )
        BIAS_STATE = defaultdict(
            lambda: {"bull_bias": 0.0, "bear_bias": 0.0, "vol_bias": 0.0}
        )


@app.post("/learn", response_model=StandardAgentResponse[LearningResponse])
async def learn(
    request: LearningRequest,
    req: Request,
) -> StandardAgentResponse[LearningResponse]:
    correlation_id = req.headers.get("X-Correlation-ID")
    learning_result = await run_learning_cycle(
        request,
        BIAS_STATE,
        correlation_id=correlation_id,
    )
    return StandardAgentResponse(
        status="success",
        data=learning_result,
        correlation_id=correlation_id,
    )


@app.post(
    "/learn/portfolio",
    response_model=StandardAgentResponse[PortfolioLearningResponse],
)
async def learn_portfolio(
    request: PortfolioLearningRequest,
    req: Request,
) -> StandardAgentResponse[PortfolioLearningResponse]:
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
        correlation_id=correlation_id,
    )


@app.post(
    "/learn/performance",
    response_model=StandardAgentResponse[PerformanceLearningResponse],
)
async def learn_performance(
    request: PerformanceLearningRequest,
    req: Request,
) -> StandardAgentResponse[PerformanceLearningResponse]:
    correlation_id = req.headers.get("X-Correlation-ID")
    result = analyze_performance_summary(request)
    logging.info(
        "[correlation_id=%s] performance learning reviewed %s closed TradePlan(s)",
        correlation_id,
        result.reviewed_closed_plans,
    )
    return StandardAgentResponse(
        status="success",
        data=result,
        correlation_id=correlation_id,
    )


@app.post(
    "/learn/outcomes",
    response_model=StandardAgentResponse[LearningOutcomeResponse],
)
async def learn_outcomes(
    request: LearningOutcomeRequest,
    req: Request,
) -> StandardAgentResponse[LearningOutcomeResponse]:
    """Review closed realized outcomes with versioned evidence attribution."""
    correlation_id = req.headers.get("X-Correlation-ID")
    result = analyze_learning_outcomes(request)
    logging.info(
        "[correlation_id=%s] outcome learning accepted=%s rejected=%s",
        correlation_id,
        result.accepted_outcomes,
        result.rejected_outcomes,
    )
    return StandardAgentResponse(
        status="success",
        data=result,
        correlation_id=correlation_id,
        confidence_score=result.confidence_score,
        metadata={
            "outcome_contract_version": LEARNING_OUTCOME_VERSION,
            "requires_human_review": True,
            "auto_apply": False,
        },
    )


@app.post(
    "/market-regime",
    response_model=StandardAgentResponse[MarketRegimeResponse],
)
async def market_regime(
    request: MarketRegimeRequest,
) -> StandardAgentResponse[MarketRegimeResponse]:
    result = classify_market_regime(request.price_history)
    return StandardAgentResponse(status="success", data=result)


@app.post(
    "/learning/update-biases",
    response_model=StandardAgentResponse[List[BiasUpdateResponse]],
)
async def update_biases(
    request: Union[List[BiasUpdateRequest], BiasUpdateRequest],
) -> StandardAgentResponse[List[BiasUpdateResponse]]:
    updates = request if isinstance(request, list) else [request]
    responses = []

    for update in updates:
        asset_id = update.asset_id
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
        responses.append(
            BiasUpdateResponse(
                asset_id=asset_id,
                current_bias=CurrentBias(**current_asset_bias),
                updated=True,
            )
        )

    try:
        save_bias_state(dict(BIAS_STATE))
        logging.info("Persisted updated bias state for %s asset(s).", len(updates))
    except Exception as exc:
        logging.error("Failed to persist bias state after update: %s", exc)

    return StandardAgentResponse(status="success", data=responses)


@app.get("/health", response_model=StandardAgentResponse[HealthData])
def health():
    try:
        db_connected = check_db_connection()
        database_status = "connected" if db_connected else "disconnected"
    except Exception as exc:
        logging.warning("Health check database error: %s", exc)
        database_status = "disconnected"

    data = HealthData(
        status="healthy",
        database=database_status,
        outcome_contract_version=LEARNING_OUTCOME_VERSION,
    )
    return StandardAgentResponse(status="success", data=data)
