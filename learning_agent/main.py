
from fastapi import FastAPI, Request
from .models import (
    LearningRequest, LearningResponse, MarketRegimeRequest, MarketRegimeResponse,
    BiasUpdateRequest, BiasUpdateResponse, CurrentBias
)
from .logic import run_learning_cycle
from .market_regime import classify_market_regime
from typing import Dict, List, Union
from collections import defaultdict

# --- In-Memory State ---
# Stores the current biases for each asset.
# Using defaultdict to provide a neutral default bias for unseen assets.
BIAS_STATE: Dict[str, Dict[str, float]] = defaultdict(lambda: {
    "bull_bias": 0.0,
    "bear_bias": 0.0,
    "vol_bias": 0.0
})


app = FastAPI(
    title="Macro Learning Agent",
    description="An analytical AI responsible for strategic, long-horizon learning in an automated trading system.",
    version="1.0.0"
)

@app.post("/learn", response_model=LearningResponse)
async def learn(request: LearningRequest, req: Request) -> LearningResponse:
    """
    Analyzes trade history and portfolio metrics to generate incremental
    policy adjustments.
    """
    correlation_id = req.headers.get("X-Correlation-ID")
    return run_learning_cycle(request, BIAS_STATE, correlation_id=correlation_id)

@app.post("/market-regime", response_model=MarketRegimeResponse)
async def market_regime(request: MarketRegimeRequest) -> MarketRegimeResponse:
    """
    Analyzes price history to determine the current market regime.
    """
    return classify_market_regime(request.price_history)

@app.post("/learning/update-biases", response_model=List[BiasUpdateResponse])
async def update_biases(request: Union[List[BiasUpdateRequest], BiasUpdateRequest]) -> List[BiasUpdateResponse]:
    """
    Receives feedback from the Manager to update the agent's internal biases.
    Supports both single and batch updates.
    """
    updates = request if isinstance(request, list) else [request]
    responses = []

    for update in updates:
        asset_id = update.asset_id
        current_asset_bias = BIAS_STATE[asset_id]

        # Apply the deltas
        current_asset_bias["bull_bias"] += update.bias_delta.bull_bias
        current_asset_bias["bear_bias"] += update.bias_delta.bear_bias
        current_asset_bias["vol_bias"] += update.bias_delta.vol_bias

        # Clamp the values to a reasonable range, e.g., [-1.0, 1.0] to prevent runaway feedback loops
        current_asset_bias["bull_bias"] = max(-1.0, min(1.0, current_asset_bias["bull_bias"]))
        current_asset_bias["bear_bias"] = max(-1.0, min(1.0, current_asset_bias["bear_bias"]))
        current_asset_bias["vol_bias"] = max(-1.0, min(1.0, current_asset_bias["vol_bias"]))

        BIAS_STATE[asset_id] = current_asset_bias

        response = BiasUpdateResponse(
            asset_id=asset_id,
            current_bias=CurrentBias(**current_asset_bias),
            updated=True
        )
        responses.append(response)

    return responses

@app.get("/health")
def health():
    return {"status": "ok"}
