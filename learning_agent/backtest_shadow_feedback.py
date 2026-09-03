from __future__ import annotations

from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field


SCHEMA_VERSION = "learning-backtest-shadow-feedback.v1"


class BacktestShadowFeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str = Field(min_length=1, max_length=20)
    strategy_id: str = Field(min_length=1, max_length=128)
    backtest_evidence: dict[str, Any]
    forward_evidence: dict[str, Any]


class BacktestShadowFeedbackResponse(BaseModel):
    schema_version: Literal["learning-backtest-shadow-feedback.v1"] = SCHEMA_VERSION
    symbol: str
    strategy_id: str
    decision: Literal[
        "reject_challenger",
        "continue_shadow",
        "request_human_promotion_review",
    ]
    failed_gates: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    advisory_only: Literal[True] = True
    auto_promote: Literal[False] = False
    risk_policy_change_authorized: Literal[False] = False
    broker_order_authorized: Literal[False] = False


def evaluate_backtest_shadow_feedback(
    request: BacktestShadowFeedbackRequest,
) -> BacktestShadowFeedbackResponse:
    """Join Backtest near-miss evidence with independent Shadow evidence.

    Learning is advisory only. Even a successful review cannot authorize Risk,
    Execution, broker mutation, or threshold changes.
    """
    backtest: Mapping[str, Any] = request.backtest_evidence
    forward: Mapping[str, Any] = request.forward_evidence

    if backtest.get("observation_candidate") is not True:
        return BacktestShadowFeedbackResponse(
            symbol=request.symbol.upper(),
            strategy_id=request.strategy_id,
            decision="reject_challenger",
            failed_gates=list(backtest.get("failed_candidate_oos_gates") or []),
            reasons=["Backtest evidence is not safe enough for the challenger lane."],
        )

    if forward.get("forward_review_ready") is not True:
        return BacktestShadowFeedbackResponse(
            symbol=request.symbol.upper(),
            strategy_id=request.strategy_id,
            decision="continue_shadow",
            failed_gates=list(forward.get("failed_gates") or []),
            reasons=[
                "Backtest near-miss is observation-safe, but independent forward evidence is not sufficient yet."
            ],
        )

    return BacktestShadowFeedbackResponse(
        symbol=request.symbol.upper(),
        strategy_id=request.strategy_id,
        decision="request_human_promotion_review",
        reasons=[
            "Backtest challenger evidence and independent forward evidence passed advisory review. Production Backtest, Manager, Risk, and human promotion controls remain authoritative."
        ],
    )
