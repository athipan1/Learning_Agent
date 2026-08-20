from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


CHAMPION_CHALLENGER_SCHEMA_VERSION = "learning-champion-challenger.v1"


class ChampionEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy_id: str = Field(min_length=1, max_length=128)
    observation_count: int = Field(ge=0)
    net_expectancy_r: float
    profit_factor: float = Field(ge=0)
    max_drawdown_pct: float = Field(ge=0, le=1)
    execution_cost_calibrated: bool = False


class ChallengerEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy_id: str = Field(min_length=1, max_length=128)
    research_profile: str = Field(min_length=1, max_length=128)
    pre_holdout_passed: bool
    statistical_validation_passed: bool
    robustness_validation_passed: bool
    pbo_passed: bool
    cost_stress_passed: bool
    final_holdout_status: Literal[
        "sealed_not_opened",
        "opened_passed",
        "opened_rejected",
    ]
    paper_shadow_observations: int = Field(default=0, ge=0)
    paper_net_expectancy_r: Optional[float] = None
    paper_profit_factor: Optional[float] = Field(default=None, ge=0)
    paper_max_drawdown_pct: Optional[float] = Field(default=None, ge=0, le=1)
    execution_cost_calibrated: bool = False

    @model_validator(mode="after")
    def validate_paper_evidence(self) -> "ChallengerEvidence":
        paper_metrics = (
            self.paper_net_expectancy_r,
            self.paper_profit_factor,
            self.paper_max_drawdown_pct,
        )
        if self.paper_shadow_observations > 0 and any(
            value is None for value in paper_metrics
        ):
            raise ValueError(
                "paper shadow observations require expectancy, profit factor, "
                "and drawdown evidence"
            )
        if self.paper_shadow_observations == 0 and any(
            value is not None for value in paper_metrics
        ):
            raise ValueError(
                "paper metrics cannot be supplied without paper shadow observations"
            )
        return self


class ChampionChallengerPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_paper_shadow_observations: int = Field(default=50, ge=10, le=10000)
    min_expectancy_improvement_r: float = Field(default=0.05, ge=0, le=10)
    max_challenger_drawdown_pct: float = Field(default=0.10, gt=0, le=1)
    require_execution_cost_calibration: bool = True


class ChampionChallengerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    champion: ChampionEvidence
    challenger: ChallengerEvidence
    policy: ChampionChallengerPolicy = Field(
        default_factory=ChampionChallengerPolicy
    )


class ChampionChallengerResponse(BaseModel):
    schema_version: Literal["learning-champion-challenger.v1"] = (
        CHAMPION_CHALLENGER_SCHEMA_VERSION
    )
    decision: Literal[
        "continue_research",
        "request_final_holdout_review",
        "shadow_candidate",
        "keep_champion",
        "human_promotion_review",
    ]
    champion_strategy_id: str
    challenger_strategy_id: str
    eligible_for_automatic_promotion: Literal[False] = False
    requires_human_review: Literal[True] = True
    auto_apply: Literal[False] = False
    gate_changes_allowed: Literal[False] = False
    final_holdout_open_authorized: Literal[False] = False
    reasons: list[str] = Field(default_factory=list)
    evidence: dict = Field(default_factory=dict)


def evaluate_champion_challenger(
    request: ChampionChallengerRequest,
) -> ChampionChallengerResponse:
    """Compare a challenger without granting promotion or holdout authority.

    Learning_Agent may recommend the next review stage only. It cannot open a
    sealed final holdout, change research thresholds, promote a strategy, change
    Risk policy, or create an Execution order.
    """

    champion = request.champion
    challenger = request.challenger
    policy = request.policy
    reasons: list[str] = []

    pre_holdout_gates = {
        "pre_holdout": challenger.pre_holdout_passed,
        "statistical_validation": challenger.statistical_validation_passed,
        "robustness_validation": challenger.robustness_validation_passed,
        "pbo": challenger.pbo_passed,
        "cost_stress": challenger.cost_stress_passed,
    }
    failed_pre_holdout = [
        name for name, passed in pre_holdout_gates.items() if not passed
    ]
    if failed_pre_holdout:
        reasons.append(
            "Challenger failed required research gates: "
            + ", ".join(failed_pre_holdout)
        )
        return ChampionChallengerResponse(
            decision="continue_research",
            champion_strategy_id=champion.strategy_id,
            challenger_strategy_id=challenger.strategy_id,
            reasons=reasons,
            evidence={"pre_holdout_gates": pre_holdout_gates},
        )

    if challenger.final_holdout_status == "opened_rejected":
        reasons.append("Challenger was rejected by the sealed final holdout.")
        return ChampionChallengerResponse(
            decision="continue_research",
            champion_strategy_id=champion.strategy_id,
            challenger_strategy_id=challenger.strategy_id,
            reasons=reasons,
            evidence={"pre_holdout_gates": pre_holdout_gates},
        )

    if challenger.final_holdout_status == "sealed_not_opened":
        reasons.append(
            "All supplied pre-holdout gates passed; Backtest authority and human "
            "review are still required before the sealed holdout can be opened."
        )
        return ChampionChallengerResponse(
            decision="request_final_holdout_review",
            champion_strategy_id=champion.strategy_id,
            challenger_strategy_id=challenger.strategy_id,
            reasons=reasons,
            evidence={"pre_holdout_gates": pre_holdout_gates},
        )

    if (
        challenger.paper_shadow_observations
        < policy.min_paper_shadow_observations
    ):
        reasons.append(
            "Final holdout passed, but Paper shadow evidence is below the "
            "minimum observation requirement."
        )
        return ChampionChallengerResponse(
            decision="shadow_candidate",
            champion_strategy_id=champion.strategy_id,
            challenger_strategy_id=challenger.strategy_id,
            reasons=reasons,
            evidence={
                "paper_shadow_observations": challenger.paper_shadow_observations,
                "required_paper_shadow_observations": (
                    policy.min_paper_shadow_observations
                ),
            },
        )

    if policy.require_execution_cost_calibration and not (
        champion.execution_cost_calibrated
        and challenger.execution_cost_calibrated
    ):
        reasons.append(
            "Paper sample is large enough, but execution-cost calibration is "
            "required before a promotion review."
        )
        return ChampionChallengerResponse(
            decision="keep_champion",
            champion_strategy_id=champion.strategy_id,
            challenger_strategy_id=challenger.strategy_id,
            reasons=reasons,
            evidence={"execution_cost_calibrated": False},
        )

    challenger_expectancy = challenger.paper_net_expectancy_r
    challenger_pf = challenger.paper_profit_factor
    challenger_dd = challenger.paper_max_drawdown_pct
    assert challenger_expectancy is not None
    assert challenger_pf is not None
    assert challenger_dd is not None

    required_expectancy = (
        champion.net_expectancy_r + policy.min_expectancy_improvement_r
    )
    comparison_gates = {
        "expectancy_improvement": challenger_expectancy >= required_expectancy,
        "profit_factor_not_worse": challenger_pf >= champion.profit_factor,
        "drawdown_within_policy": (
            challenger_dd <= policy.max_challenger_drawdown_pct
        ),
        "drawdown_not_worse": challenger_dd <= champion.max_drawdown_pct,
    }
    failed_comparison = [
        name for name, passed in comparison_gates.items() if not passed
    ]
    if failed_comparison:
        reasons.append(
            "Challenger does not dominate the Champion under the configured "
            "Paper evidence policy: "
            + ", ".join(failed_comparison)
        )
        return ChampionChallengerResponse(
            decision="keep_champion",
            champion_strategy_id=champion.strategy_id,
            challenger_strategy_id=challenger.strategy_id,
            reasons=reasons,
            evidence={"comparison_gates": comparison_gates},
        )

    reasons.append(
        "Challenger cleared research, final-holdout, Paper sample, execution-cost, "
        "expectancy, profit-factor, and drawdown checks. Human promotion review "
        "is required; Learning_Agent cannot promote it."
    )
    return ChampionChallengerResponse(
        decision="human_promotion_review",
        champion_strategy_id=champion.strategy_id,
        challenger_strategy_id=challenger.strategy_id,
        reasons=reasons,
        evidence={
            "comparison_gates": comparison_gates,
            "required_expectancy_r": required_expectancy,
            "challenger_expectancy_r": challenger_expectancy,
        },
    )
