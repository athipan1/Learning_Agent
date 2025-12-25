
from pydantic import BaseModel, Field
from typing import List, Dict, Optional

# --- Input Contract Models ---

class AgentVote(BaseModel):
    """Represents a single agent's vote in a trade."""
    action: str
    confidence: float

class Trade(BaseModel):
    """Represents a single historical trade."""
    timestamp: str
    action: str
    entry_price: float
    exit_price: float
    pnl_pct: float
    agent_votes: Dict[str, AgentVote]

class PortfolioMetrics(BaseModel):
    """Represents the overall performance metrics of the portfolio."""
    equity_curve: List[float]
    max_drawdown: float
    win_rate: float
    profit_factor: float

class Config(BaseModel):
    """Represents the current configuration of the trading system."""
    agent_weights: Dict[str, float]
    risk_per_trade: float
    stop_loss_pct: float
    max_position_pct: float
    enable_technical_stop: bool

class LearningRequest(BaseModel):
    """The complete input data structure for the /learn endpoint."""
    trade_history: List[Trade]
    price_history: List[Dict]
    portfolio_metrics: PortfolioMetrics
    current_config: Config

# --- Output Contract Models ---

class RiskAdjustments(BaseModel):
    risk_per_trade: Optional[float] = None
    stop_loss_pct: Optional[float] = None
    max_position_pct: Optional[float] = None
    enable_technical_stop: Optional[bool] = None

class PolicyDeltas(BaseModel):
    agent_weights: Dict[str, float] = Field(default_factory=dict)
    risk: RiskAdjustments = Field(default_factory=RiskAdjustments)
    strategy_bias: Dict[str, str] = Field(default_factory=dict)


class LearningResponse(BaseModel):
    """The complete output data structure for the /learn endpoint."""
    learning_state: str
    confidence: float = 0.0
    policy_deltas: PolicyDeltas = Field(default_factory=PolicyDeltas)
    reasoning: List[str] = Field(default_factory=list)
