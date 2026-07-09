from fastapi.testclient import TestClient

from learning_agent.main import app


REQUIRED_CONTRACT_FIELDS = {
    "status",
    "agent_type",
    "version",
    "schema_version",
    "timestamp",
    "correlation_id",
    "data",
    "metadata",
    "error",
    "confidence_score",
}


def assert_contract_response(payload):
    assert REQUIRED_CONTRACT_FIELDS.issubset(payload.keys())
    assert payload["agent_type"] == "learning"
    assert payload["version"] == "1.1.0"
    assert payload["schema_version"] == "1.0"


def test_version_endpoint_uses_contract_response():
    client = TestClient(app)
    response = client.get("/version")

    assert response.status_code == 200
    payload = response.json()
    assert_contract_response(payload)
    assert payload["data"]["api_contract"] == "multi-agent-trading-api-contract"
    assert payload["data"]["schema_version"] == "1.0"
    assert payload["data"]["service_version"] == "1.2.0"
    assert payload["data"]["outcome_contract_version"] == (
        "learning-outcome-v1"
    )
    assert payload["metadata"]["learning_policy"] == "human-review-only"


def test_ready_endpoint_uses_contract_response():
    client = TestClient(app)
    response = client.get("/ready")

    assert response.status_code == 200
    payload = response.json()
    assert_contract_response(payload)
    assert payload["data"]["ready"] is True
    assert payload["data"]["outcome_learning_endpoint"] == "/learn/outcomes"
    assert payload["data"]["outcome_contract_version"] == (
        "learning-outcome-v1"
    )
    assert payload["data"]["auto_apply"] is False
    assert payload["data"]["requires_human_review"] is True
    assert payload["metadata"]["contract_source"] == (
        "learning-agent-runtime-contract"
    )


def test_existing_health_endpoint_still_works():
    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert_contract_response(payload)
    assert payload["data"]["status"] == "healthy"
    assert payload["data"]["outcome_contract_version"] == (
        "learning-outcome-v1"
    )
