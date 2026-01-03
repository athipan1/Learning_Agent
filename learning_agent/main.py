
from fastapi import FastAPI, Request
from .models import LearningRequest, LearningResponse, MarketRegimeRequest, MarketRegimeResponse
from .logic import run_learning_cycle
from .market_regime import classify_market_regime

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
    return run_learning_cycle(request, correlation_id=correlation_id)

@app.post("/market-regime", response_model=MarketRegimeResponse)
async def market_regime(request: MarketRegimeRequest) -> MarketRegimeResponse:
    """
    Analyzes price history to determine the current market regime.
    """
    return classify_market_regime(request.price_history)

@app.get("/health")
def health():
    return {"status": "ok"}
