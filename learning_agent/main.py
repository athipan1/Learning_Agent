
from fastapi import FastAPI
from .models import LearningRequest, LearningResponse, MarketRegimeInput, MarketRegimeOutput
from .logic import run_learning_cycle
from .regime_logic import run_regime_analysis

app = FastAPI(
    title="Macro Learning Agent",
    description="An analytical AI responsible for strategic, long-horizon learning in an automated trading system.",
    version="1.0.0"
)

@app.post("/learn", response_model=LearningResponse)
async def learn(request: LearningRequest) -> LearningResponse:
    """
    Analyzes trade history and portfolio metrics to generate incremental
    policy adjustments.
    """
    return run_learning_cycle(request)

@app.post("/market-regime", response_model=MarketRegimeOutput)
async def market_regime(request: MarketRegimeInput) -> MarketRegimeOutput:
    """
    Analyzes price history to determine the current market regime.
    """
    return run_regime_analysis(request)
