from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


SHADOW_REVIEW_SCHEMA_VERSION = "learning-shadow-paper-review.v1"


class ShadowEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy_id: str = Field(min_length=1, max_length=128)
    observation_count: int = Field(ge=0)
    net_expectancy_pct: Optional[float] = None
    profit_factor: Optional[float] = Field(default=None, ge=0)
    max_drawdown_pct: Optional[float] = Field(default=None, ge=0, le=1)
    average_mfe_pct: Optional[float] = None
    average_mae_pct: Optional[float] = None
    execution_cost_calibrated: bool = False
    final_holdout_passed: bool = False


class ShadowReviewPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_observations: int = Field(default=50, ge=10, le=100000)
    min_net_expectancy_pct: float = 0.0
    min_profit_factor: float = Field(default=1.10, ge=0)
    max_drawdown_pct: float = Field(default=0.10, gt=0, le=1)
    require_execution_cost_calibration: bool = True


class ShadowReviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence: ShadowEvidence
    policy: ShadowReviewPolicy = Field(default_factory=ShadowReviewPolicy)


class ShadowReviewResponse(BaseModel):
    schema_version: Literal["learning-shadow-paper-review.v1"] = SHADOW_REVIEW_SCHEMA_VERSION
    strategy_id: str
    decision: Literal["continue_shadow", "request_paper_review"]
    failed_gates: list[str] = Field(default_factory=list)
    requires_human_review: Literal[True] = True
    auto_promote: Literal[False] = False
    risk_policy_change_authorized: Literal[False] = False
    broker_order_authorized: Literal[False] = False
    reasons: list[str] = Field(default_factory=list)


def evaluate_shadow_for_paper(request: ShadowReviewRequest) -> ShadowReviewResponse:
    evidence = request.evidence
    policy = request.policy
    gates = {
        "final_holdout_passed": evidence.final_holdout_passed,
        "shadow_observation_count": evidence.observation_count >= policy.min_observations,
        "positive_net_expectancy": (
            evidence.net_expectancy_pct is not None
            and evidence.net_expectancy_pct > policy.min_net_expectancy_pct
        ),
        "profit_factor": (
            evidence.profit_factor is not None
            and evidence.profit_factor >= policy.min_profit_factor
        ),
        "max_drawdown": (
            evidence.max_drawdown_pct is not None
            and evidence.max_drawdown_pct <= policy.max_drawdown_pct
        ),
        "execution_cost_calibrated": (
            evidence.execution_cost_calibrated
            or not policy.require_execution_cost_calibration
        ),
    }
    failed = [name for name, passed in gates.items() if not passed]
    if failed:
        return ShadowReviewResponse(
            strategy_id=evidence.strategy_id,
            decision="continue_shadow",
            failed_gates=failed,
            reasons=["Shadow evidence is not yet sufficient for Paper review."],
        )
    return ShadowReviewResponse(
        strategy_id=evidence.strategy_id,
        decision="request_paper_review",
        reasons=[
            "Shadow evidence passed the advisory gates. Manager/Risk and human review remain required before any Paper order."
        ],
    )
