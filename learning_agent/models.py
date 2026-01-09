
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, Literal
from decimal import Decimal

# --- Input Contract Models ---

class AgentVote(BaseModel):
    """Represents a single agent's vote in a trade."""
    action: str
    confidence: float

class Trade(BaseModel):
    """Represents a single historical trade, as received from the Manager."""
    trade_id: str # uuid
    account_id: str # uuid
    asset_id: str
    symbol: str
    side: Literal["buy", "sell"]
    quantity: Decimal
    price: Decimal
    executed_at: str # ISO-8601 timestamp
    agents: Dict[str, str] = Field(default_factory=dict)
    pnl_pct: Optional[Decimal] = None
    entry_price: Optional[Decimal] = None
    exit_price: Optional[Decimal] = None

class PricePoint(BaseModel):
    """Represents a single price point in history."""
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: int

class CurrentPolicyRisk(BaseModel):
    risk_per_trade: float
    max_position_pct: float
    stop_loss_pct: float

class CurrentPolicyStrategyBias(BaseModel):
    preferred_regime: str

class CurrentPolicy(BaseModel):
    agent_weights: Dict[str, float]
    risk: CurrentPolicyRisk
    strategy_bias: CurrentPolicyStrategyBias

class LearningRequest(BaseModel):
    """The complete input data structure for the /learn endpoint."""
    learning_mode: str
    window_size: int
    trade_history: List[Trade]
    price_history: Dict[str, List[PricePoint]]
    current_policy: CurrentPolicy
    execution_result: Optional[dict] = None

# --- Output Contract Models ---

class PolicyDeltas(BaseModel):
    agent_weights: Dict[str, float] = Field(default_factory=dict)
    risk: Dict[str, float] = Field(default_factory=dict)
    strategy_bias: Dict[str, Any] = Field(default_factory=dict)
    guardrails: Dict[str, Any] = Field(default_factory=dict)
    asset_biases: Dict[str, float] = Field(default_factory=dict)


class LearningResponse(BaseModel):
    """The complete output data structure for the /learn endpoint."""
    learning_state: str
    learning_mode: Optional[str] = None
    confidence_score: float = 0.0
    policy_deltas: PolicyDeltas = Field(default_factory=PolicyDeltas)
    reasoning: List[str] = Field(default_factory=list)

# --- Market Regime Analysis Models ---

class MarketRegimeRequest(BaseModel):
    """The input data structure for the /market-regime endpoint."""
    price_history: List[PricePoint] = Field(..., min_length=200)


class MarketRegimeResponse(BaseModel):
    """The output data structure for the /market-regime endpoint."""
    regime: str
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    explanation: str
