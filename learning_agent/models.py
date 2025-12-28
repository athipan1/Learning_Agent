
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any

# --- Input Contract Models ---

class AgentVote(BaseModel):
    """Represents a single agent's vote in a trade."""
    action: str
    confidence: float

class Trade(BaseModel):
    """Represents a single historical trade."""
    trade_id: str
    asset_id: str
    final_verdict: str
    executed: bool
    pnl_pct: float
    holding_days: int
    market_regime: str
    agent_votes: Dict[str, AgentVote]
    timestamp: str

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

class IndicatorSettings(BaseModel):
    """Indicator settings for market regime analysis."""
    ema_fast: int
    ema_slow: int
    adx_period: int
    atr_period: int
    rsi_period: int

class MarketRegimeInput(BaseModel):
    """The input data structure for the /market-regime endpoint."""
    symbol: str
    timeframe: str
    price_history: List[PricePoint]
    indicators: IndicatorSettings

class LearnedPatterns(BaseModel):
    """Learned patterns from the market regime analysis."""
    trend_character: Optional[str] = None
    false_breakout_risk: Optional[str] = None
    best_strategy_fit: List[str] = Field(default_factory=list)

class MarketRegimeOutput(BaseModel):
    """The output data structure for the /market-regime endpoint."""
    learning_state: str = "success"
    market_regime: Optional[str] = None
    confidence: float
    explanation: List[str] = Field(default_factory=list)
    learned_patterns: Optional[LearnedPatterns] = None
    risk_notes: List[str] = Field(default_factory=list)
    reasoning: List[str] = Field(default_factory=list)
