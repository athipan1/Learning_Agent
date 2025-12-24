from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class AgentVote:
    action: str
    confidence_score: float
    version: str

@dataclass
class OrchestratorConfig:
    agent_weights: Dict[str, float]
    confidence_threshold: Dict[str, float]

@dataclass
class Trade:
    trade_id: str
    ticker: str
    final_verdict: str
    executed: bool
    entry_price: float
    exit_price: float
    pnl_pct: float
    holding_days: int
    market_regime: str
    agent_votes: Dict[str, AgentVote]
    orchestrator_config: OrchestratorConfig
    timestamp: str
