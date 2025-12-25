
from fastapi import FastAPI
from .models import LearningRequest, LearningResponse
from .logic import run_learning_cycle

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
    # 1. Validate Learning Readiness
    if len(request.trade_history) < request.window_size:
        return LearningResponse(
            learning_state="warmup",
            reasoning=[f"Requires at least {request.window_size} trades for analysis, but received {len(request.trade_history)}."]
        )

    # 2. Run the full learning cycle
    return run_learning_cycle(request)
