# Learning Agent Audit Report

### 1. Overall Architecture
*   **Issue:** The current design combines a real-time API with heavy analytical processing in a single service. This is not scalable and can lead to slow API response times.
*   **Recommendation:** Separate the system into two distinct components as discussed: a lightweight **API** for serving results and an offline **Learning Pipeline** for performing intensive calculations on a schedule.

### 2. Folder Structure & Separation of Concerns
*   **Issue:** All logic files (`logic.py`, `market_regime.py`, `analysis.py`) are mixed in the root `learning_agent` directory. This makes it difficult to distinguish between API-specific code, database models, and analytical code.
*   **Recommendation:** Create a more organized folder structure that clearly separates concerns:
    *   `learning_agent/api/`: For FastAPI endpoints and API-specific models.
    *   `learning_agent/learning_pipeline/`: For all offline data fetching, analysis, and learning logic.
    *   `learning_agent/core/`: For shared components like database schemas and connections.

### 3. Data Ingestion & Persistence
*   **Issue:** The agent is purely reactive; it does not ingest or store its own historical data. It relies entirely on data being sent to it via API requests. The current database interaction is limited to a simple key-value `BIAS_STATE`.
*   **Recommendation:** Implement a dedicated data ingestion module to fetch from Alpaca and design a proper relational database schema to store historical market data, indicator snapshots, and learning results over time.

### 4. Testability
*   **Issue:** The current tests are primarily for the API endpoints. The core analytical logic within functions like `run_learning_cycle` is complex and not easily unit-testable in isolation.
*   **Recommendation:** The new modular structure will promote testability. The Indicator Engine, Performance Analysis, and Data Ingestion modules should all have dedicated unit tests, independent of the API.

### 5. Configuration Management
*   **Issue:** The system lacks a clear way to manage different strategies. Adding a new strategy would require code changes.
*   **Recommendation:** Implement a database-driven approach for strategy configuration, allowing strategies to be added or toggled without redeployment.

### Recommended Folder Structure
```
.
├── AUDIT_REPORT.md
├── learning_agent/
│   ├── __init__.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   └── models.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── database.py
│   │   └── schemas.py
│   └── learning_pipeline/
│       ├── __init__.py
│       ├── data_ingestion.py
│       ├── indicator_engine.py
│       ├── main.py
│       └── performance_analysis.py
├── tests/
│   ├── __init__.py
│   ├── test_api/
│   │   └── __init__.py
│   └── test_learning_pipeline/
│       └── __init__.py
├── Dockerfile
├── README.md
├── docker-compose.yml
└── requirements.txt
```