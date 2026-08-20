from __future__ import annotations

from fastapi import APIRouter, Request

from .champion_challenger import (
    CHAMPION_CHALLENGER_SCHEMA_VERSION,
    ChampionChallengerRequest,
    ChampionChallengerResponse,
    evaluate_champion_challenger,
)
from .models import StandardAgentResponse


router = APIRouter(tags=["research-learning"])


@router.post(
    "/learn/champion-challenger",
    response_model=StandardAgentResponse[ChampionChallengerResponse],
)
async def learn_champion_challenger(
    payload: ChampionChallengerRequest,
    request: Request,
) -> StandardAgentResponse[ChampionChallengerResponse]:
    correlation_id = request.headers.get("X-Correlation-ID")
    result = evaluate_champion_challenger(payload)
    return StandardAgentResponse(
        status="success",
        data=result,
        correlation_id=correlation_id,
        metadata={
            "schema_version": CHAMPION_CHALLENGER_SCHEMA_VERSION,
            "advisory_only": True,
            "requires_human_review": True,
            "auto_apply": False,
            "promotion_authority": False,
            "final_holdout_authority": False,
            "gate_change_authority": False,
        },
    )
