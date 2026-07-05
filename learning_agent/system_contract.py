from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Dict

from fastapi import APIRouter

LEARNING_AGENT_TYPE = "learning"
LEARNING_AGENT_VERSION = "1.0.0"
LEARNING_SERVICE_VERSION = "1.1.0"
SCHEMA_VERSION = "1.0"

router = APIRouter()


def utc_timestamp() -> str:
    return datetime.now(UTC).isoformat()


def contract_response(
    *,
    status: str,
    data: Dict[str, Any] | None = None,
    metadata: Dict[str, Any] | None = None,
    error: Dict[str, Any] | None = None,
    confidence_score: float | None = None,
) -> Dict[str, Any]:
    return {
        "status": status,
        "agent_type": LEARNING_AGENT_TYPE,
        "version": LEARNING_AGENT_VERSION,
        "schema_version": SCHEMA_VERSION,
        "timestamp": utc_timestamp(),
        "correlation_id": None,
        "data": data,
        "metadata": metadata or {},
        "error": error,
        "confidence_score": confidence_score,
    }


@router.get("/version")
def version() -> Dict[str, Any]:
    return contract_response(
        status="success",
        data={
            "agent_type": LEARNING_AGENT_TYPE,
            "version": LEARNING_AGENT_VERSION,
            "service_version": LEARNING_SERVICE_VERSION,
            "schema_version": SCHEMA_VERSION,
            "api_contract": "multi-agent-trading-api-contract",
        },
        metadata={
            "required_operational_endpoints": ["/health", "/ready", "/version"],
        },
    )


@router.get("/ready")
def ready() -> Dict[str, Any]:
    return contract_response(
        status="success",
        data={
            "ready": True,
            "learn_endpoint": "/learn",
            "portfolio_learning_endpoint": "/learn/portfolio",
            "performance_learning_endpoint": "/learn/performance",
            "market_regime_endpoint": "/market-regime",
            "bias_update_endpoint": "/learning/update-biases",
            "persistence": "database-backed-bias-state",
        },
        metadata={
            "contract_source": "learning-agent-runtime-contract",
        },
        confidence_score=1.0,
    )
