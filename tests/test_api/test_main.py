
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from learning_agent.api.main import app, get_db_session
from learning_agent.core.schemas import Base, Strategy, LearningResult
from datetime import datetime, timezone

# --- Final, Direct Test Fixture ---

@pytest.fixture(scope="function")
def api_test_setup():
    """
    This fixture provides a fully isolated test environment for the API by
    directly controlling the database setup and dependency overrides.
    """
    # 1. Create a completely isolated in-memory SQLite database for the test
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    # 2. Explicitly create all tables on this engine
    Base.metadata.create_all(bind=engine)

    # 3. Define the dependency override
    def override_get_db():
        session = TestingSessionLocal()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db_session] = override_get_db

    # 4. Yield the client and a session factory for test data setup
    yield TestClient(app), TestingSessionLocal

    # 5. Clean up after the test
    Base.metadata.drop_all(bind=engine)
    del app.dependency_overrides[get_db_session]


# --- Tests ---

def test_get_bias_success(api_test_setup):
    """Test the /bias endpoint for a successful response."""
    client, Session = api_test_setup
    db = Session()

    strategy = Strategy(id="strat1", strategy_name="EMA_CROSS", is_active=True)
    result = LearningResult(strategy_id="strat1", bias_adjustment=0.15, risk_level="HIGH",
                            market_regime="TRENDING", confidence=0.85, created_at=datetime.now(timezone.utc))
    db.add(strategy)
    db.add(result)
    db.commit()

    response = client.get("/bias?strategy=EMA_CROSS")

    assert response.status_code == 200
    data = response.json()
    assert data["strategy_name"] == "EMA_CROSS"
    db.close()

def test_get_bias_strategy_not_found(api_test_setup):
    """Test the /bias endpoint for a strategy that doesn't exist."""
    client, _ = api_test_setup
    response = client.get("/bias?strategy=NON_EXISTENT")
    assert response.status_code == 404

def test_get_bias_no_results_for_strategy(api_test_setup):
    """Test for a strategy that exists but has no learning results."""
    client, Session = api_test_setup
    db = Session()

    strategy = Strategy(id="strat2", strategy_name="RSI_STRATEGY", is_active=True)
    db.add(strategy)
    db.commit()

    response = client.get("/bias?strategy=RSI_STRATEGY")
    assert response.status_code == 404
    db.close()

def test_health_check(api_test_setup):
    """Test the /health endpoint."""
    client, _ = api_test_setup
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
