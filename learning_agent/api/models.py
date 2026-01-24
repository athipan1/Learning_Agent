
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime

# --- API Response Models ---

class BiasResponse(BaseModel):
    """
    The data structure for the primary API endpoint (/bias).
    It provides the latest learned adjustments for a given strategy.
    This model directly corresponds to the LearningResult schema in the database.
    """
    strategy_name: str
    bias_adjustment: float
    risk_level: str
    market_regime: str
    confidence: float
    last_updated: datetime = Field(..., description="The timestamp when this result was generated.")

    model_config = ConfigDict(from_attributes=True)


class HealthResponse(BaseModel):
    """
    A simple model for the /health endpoint.
    """
    status: str = "ok"
