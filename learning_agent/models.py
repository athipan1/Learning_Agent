
from pydantic import BaseModel, Field
from typing import List, Dict

# --- Input Contract Models ---

class Trade(BaseModel):
    """Represents a single historical trade."""
    timestamp: str
    action: str
    entry_price: float
    exit_price: float
    pnl_pct: float
    agents: Dict[str, str]

class PortfolioMetrics(BaseModel):
    """Represents the overall performance metrics of the portfolio."""
    win_rate: float
    average_return: float
    max_drawdown: float
    sharpe_ratio: float

class Config(BaseModel):
    """Represents the current configuration of the trading system."""
    agent_weights: Dict[str, float]
    risk_per_trade: float
    max_position_pct: float

class LearningRequest(BaseModel):
    """The complete input data structure for the /learn endpoint."""
    symbol: str
    trade_history: List[Trade]
    portfolio_metrics: PortfolioMetrics
    config: Config

# --- Output Contract Models ---

class LearningResponse(BaseModel):
    """The complete output data structure for the /learn endpoint."""
    status: str = "ok"
    learning_state: str
    agent_weight_adjustments: Dict[str, float] = Field(default_factory=dict)
    risk_adjustments: Dict[str, float] = Field(default_factory=dict)
    strategy_bias: Dict[str, float] = Field(default_factory=dict)
    guardrails: Dict[str, float] = Field(default_factory=dict)
    reasoning: List[str] = Field(default_factory=list)
