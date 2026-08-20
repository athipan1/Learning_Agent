from __future__ import annotations

from fastapi import APIRouter, Request

from .models import StandardAgentResponse
from .shadow_review import (
    SHADOW_REVIEW_SCHEMA_VERSION,
    ShadowReviewRequest,
    ShadowReviewResponse,
    evaluate_shadow_for_paper,
)


router = APIRouter(tags=["research-learning"])


@router.post(
    "/learn/shadow-paper-review",
    response_model=StandardAgentResponse[ShadowReviewResponse],
)
async def learn_shadow_paper_review(
    payload: ShadowReviewRequest,
    request: Request,
) -> StandardAgentResponse[ShadowReviewResponse]:
    correlation_id = request.headers.get("X-Correlation-ID")
    result = evaluate_shadow_for_paper(payload)
    return StandardAgentResponse(
        status="success",
        data=result,
        correlation_id=correlation_id,
        metadata={
            "schema_version": SHADOW_REVIEW_SCHEMA_VERSION,
            "advisory_only": True,
            "requires_human_review": True,
            "auto_promote": False,
            "risk_policy_change_authority": False,
            "broker_order_authority": False,
        },
    )
