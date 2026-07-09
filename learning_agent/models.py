from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, Dict, Generic, List, Literal, Optional, TypeVar, Union

from pydantic import BaseModel, Field

T = TypeVar("T")

LEARNING_AGENT_VERSION = "1.1.0"
LEARNING_SERVICE_VERSION = "1.2.0"
LEARNING_OUTCOME_VERSION = "learning-outcome-v1"
SCHEMA_VERSION = "1.0"
StrategyBucket = Literal[
    "core_dividend",
    "value_rebound",
    "news_momentum",
]


class StandardAgentResponse(BaseModel, Generic[T]):
    """Standardized response format for all agents."""

    status: Literal["success", "error"]
    agent_type: str = "learning"
    version: str = LEARNING_AGENT_VERSION
    schema_version: str = SCHEMA_VERSION
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    correlation_id: Optional[str] = None
    data: Optional[T] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[Dict[str, Any]] = None
    confidence_score: Optional[float] = None


class AgentVote(BaseModel):
    """Represents a single agent's vote in a trade."""

    action: str
    confidence: float


class Trade(BaseModel):
    """Represents a single, standardized historical trade."""

    trade_id: Union[int, str]
    account_id: Union[int, str]
    asset_id: str
    side: Literal["buy", "sell"]
    entry_price: Optional[Decimal] = Field(default=Decimal("0"))
    exit_price: Optional[Decimal] = Field(default=Decimal("0"))
    quantity: Decimal
    executed_at: str
    pnl_pct: Optional[Decimal] = Field(default=Decimal("0"))


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
    preferred_regime: str = "neutral"


class CurrentPolicy(BaseModel):
    agent_weights: Dict[str, float]
    risk: CurrentPolicyRisk
    strategy_bias: CurrentPolicyStrategyBias = Field(
        default_factory=CurrentPolicyStrategyBias
    )


class LearningRequest(BaseModel):
    """The complete input data structure for the /learn endpoint."""

    account_id: Union[int, str]
    learning_mode: str
    window_size: int
    trade_history: List[Trade] = Field(default_factory=list)
    price_history: Dict[str, List[PricePoint]] = Field(default_factory=dict)
    current_policy: CurrentPolicy
    execution_result: Optional[dict] = None


class PortfolioAuditRecord(BaseModel):
    portfolio_audit_id: Optional[str] = None
    account_id: Union[int, str]
    correlation_id: Optional[str] = None
    policy_name: Optional[str] = None
    mode: str = "portfolio_allocation"
    status: str = "created"
    allocation_plan: Dict[str, Any] = Field(default_factory=dict)
    portfolio_snapshot: Dict[str, Any] = Field(default_factory=dict)
    selected_positions: List[Dict[str, Any]] = Field(default_factory=list)
    risk_approvals: List[Dict[str, Any]] = Field(default_factory=list)
    execution_orders: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PortfolioLearningRequest(BaseModel):
    account_id: Union[int, str]
    learning_mode: str = "portfolio_bucket_review"
    portfolio_audits: List[PortfolioAuditRecord] = Field(default_factory=list)
    current_policy: Optional[Dict[str, Any]] = None


class PerformanceGroupMetric(BaseModel):
    trade_plan_count: int = 0
    closed_plan_count: int = 0
    win_rate: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    net_pnl: float = 0.0
    expectancy: float = 0.0
    profit_factor: Optional[float] = None


class PerformanceSummaryPayload(BaseModel):
    period: str = "all"
    trade_plan_count: int = 0
    closed_plan_count: int = 0
    open_plan_count: int = 0
    winning_plans: int = 0
    losing_plans: int = 0
    win_rate: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    net_pnl: float = 0.0
    return_pct: float = 0.0
    expectancy: float = 0.0
    profit_factor: Optional[float] = None
    average_win: float = 0.0
    average_loss: float = 0.0
    best_strategy_bucket: Optional[str] = None
    worst_strategy_bucket: Optional[str] = None
    by_strategy_bucket: Dict[str, PerformanceGroupMetric] = Field(
        default_factory=dict
    )
    by_symbol: Dict[str, PerformanceGroupMetric] = Field(default_factory=dict)
    plan_results: List[Dict[str, Any]] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class PerformanceLearningRequest(BaseModel):
    account_id: Union[int, str]
    learning_mode: str = "performance_summary_review"
    performance_summary: PerformanceSummaryPayload
    current_policy: Optional[Dict[str, Any]] = None
    min_closed_plans: int = Field(default=5, ge=1)


class EvidenceContribution(BaseModel):
    """One source's non-binding contribution to the final Manager decision."""

    version: str
    supported_bucket: Optional[StrategyBucket] = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence_status: str = "complete"
    reasons: List[str] = Field(default_factory=list)


class LearningOutcomeRecord(BaseModel):
    """A closed, realized TradePlan outcome with complete attribution context."""

    outcome_version: Literal["learning-outcome-v1"] = LEARNING_OUTCOME_VERSION
    outcome_id: str = Field(min_length=1)
    trade_plan_id: str = Field(min_length=1)
    account_id: Union[int, str]
    symbol: str = Field(min_length=1)
    strategy_bucket: StrategyBucket
    manager_bucket: StrategyBucket
    execution_bucket: StrategyBucket
    database_bucket: StrategyBucket
    manager_classifier_version: str
    evidence_versions: Dict[str, str]
    evidence_contributions: Dict[str, EvidenceContribution]
    classification_inputs: Dict[str, Any] = Field(default_factory=dict)
    bucket_confidence: float = Field(ge=0.0, le=1.0)
    entry_price: Decimal = Field(ge=0)
    exit_price: Decimal = Field(ge=0)
    realized_pnl: Decimal
    return_pct: Decimal
    holding_period_days: float = Field(default=0.0, ge=0.0)
    exit_reason: str = "unspecified"
    risk_approved: bool
    execution_status: str
    outcome_status: Literal["closed", "open"] = "closed"
    pnl_status: Literal["realized", "unrealized"] = "realized"


class LearningOutcomeRequest(BaseModel):
    account_id: Union[int, str]
    learning_mode: str = "versioned_outcome_attribution"
    outcomes: List[LearningOutcomeRecord] = Field(default_factory=list)
    current_policy: Optional[Dict[str, Any]] = None
    min_total_samples: int = Field(default=5, ge=1)
    min_bucket_samples: int = Field(default=5, ge=1)
    min_source_samples: int = Field(default=5, ge=1)


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


class PortfolioLearningResponse(BaseModel):
    learning_state: str
    learning_mode: str = "portfolio_bucket_review"
    confidence_score: float = 0.0
    portfolio_count: int = 0
    approval_rate: float = 0.0
    execution_rate: float = 0.0
    bucket_metrics: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    policy_deltas: Dict[str, Any] = Field(default_factory=dict)
    reasoning: List[str] = Field(default_factory=list)


class PerformanceLearningResponse(BaseModel):
    learning_state: str
    learning_mode: str = "performance_summary_review"
    confidence_score: float = 0.0
    reviewed_closed_plans: int = 0
    performance_score: float = 0.0
    bucket_metrics: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    symbol_metrics: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    policy_deltas: Dict[str, Any] = Field(default_factory=dict)
    reasoning: List[str] = Field(default_factory=list)


class LearningOutcomeResponse(BaseModel):
    learning_state: str
    learning_mode: str = "versioned_outcome_attribution"
    outcome_contract_version: str = LEARNING_OUTCOME_VERSION
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    reviewed_outcomes: int = 0
    accepted_outcomes: int = 0
    rejected_outcomes: int = 0
    duplicate_outcome_ids: List[str] = Field(default_factory=list)
    rejected_records: List[Dict[str, Any]] = Field(default_factory=list)
    overall_metrics: Dict[str, Any] = Field(default_factory=dict)
    bucket_metrics: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    source_attribution: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict
    )
    confidence_calibration: Dict[str, Any] = Field(default_factory=dict)
    policy_recommendations: Dict[str, Any] = Field(default_factory=dict)
    guardrails: Dict[str, Any] = Field(default_factory=dict)
    reasoning: List[str] = Field(default_factory=list)


class BiasDelta(BaseModel):
    """Represents the delta changes for different bias types."""

    bull_bias: float = 0.0
    bear_bias: float = 0.0
    vol_bias: float = 0.0


class BiasUpdateRequest(BaseModel):
    """The input data structure for a single bias update."""

    asset_id: str
    bias_delta: BiasDelta
    source: Literal["execution", "simulation", "backtest"]
    timestamp: str


class CurrentBias(BaseModel):
    """Represents the current bias state for an asset."""

    bull_bias: float
    bear_bias: float
    vol_bias: float


class BiasUpdateResponse(BaseModel):
    """The output data structure for the bias update endpoint."""

    asset_id: str
    current_bias: CurrentBias
    updated: bool


class MarketRegimeRequest(BaseModel):
    """The input data structure for the market-regime endpoint."""

    price_history: List[PricePoint] = Field(..., min_length=1)


class MarketRegimeResponse(BaseModel):
    """The output data structure for the market-regime endpoint."""

    regime: str
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    explanation: str


class HealthData(BaseModel):
    status: str
    database: str
    outcome_contract_version: str = LEARNING_OUTCOME_VERSION


LearningAgentResponseData = Union[
    LearningResponse,
    PortfolioLearningResponse,
    PerformanceLearningResponse,
    LearningOutcomeResponse,
    MarketRegimeResponse,
    List[BiasUpdateResponse],
    HealthData,
]
