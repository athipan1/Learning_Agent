
from fastapi import FastAPI
from .models import LearningRequest, LearningResponse
from .logic import run_learning_cycle

app = FastAPI(
    title="Auto-Learning Agent",
    description="An AI agent that learns from historical trade outcomes to recommend policy adjustments.",
    version="1.0.0"
)

@app.post("/learn", response_model=LearningResponse)
async def learn(request: LearningRequest) -> LearningResponse:
    """
    Analyzes trade history and portfolio metrics to generate incremental
    policy adjustments.
    """
    # Determine the learning phase
    trade_count = len(request.trade_history)
    if trade_count < 20:
        return LearningResponse(
            learning_state="warmup",
            reasoning=["Insufficient trade samples to evaluate policy effectiveness."]
        )

    # Run the full learning cycle
    return run_learning_cycle(request)
