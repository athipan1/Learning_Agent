# Learning_Agent API Contract

This document defines the API contract for `Learning_Agent`.

`Learning_Agent` reviews trade history, portfolio audits, performance summaries, market regimes, and versioned closed-trade outcomes. It only recommends guarded policy changes; it never submits orders or automatically changes production policy.

## Versions

```text
agent_version   = 1.1.0
service_version = 1.2.0
schema_version  = 1.0
outcome_contract = learning-outcome-v1
```

## Standard Headers

```http
Content-Type: application/json
X-Correlation-ID: <uuid>
X-API-KEY: <learning-agent-api-key>
```

## Standard Response Envelope

```json
{
  "status": "success",
  "agent_type": "learning",
  "version": "1.1.0",
  "schema_version": "1.0",
  "timestamp": "2026-07-09T00:00:00Z",
  "correlation_id": null,
  "data": {},
  "metadata": {},
  "error": null,
  "confidence_score": null
}
```

## Operational Endpoints

```http
GET /health
GET /ready
GET /version
```

## Learning Endpoints

```http
POST /learn
POST /learn/portfolio
POST /learn/performance
POST /learn/outcomes
POST /market-regime
POST /learning/update-biases
```

## Versioned Outcome Learning

`POST /learn/outcomes` accepts `learning-outcome-v1` records. Each record must describe a closed TradePlan with realized PnL and matching strategy buckets from Manager, Execution, and Database.

Supported upstream versions:

```text
scanner-bucket-hints-v2
fundamental-evidence-v1
technical-evidence-v1
manager-analysis-evidence-v1
manager-strategy-bucket-v3
```

The endpoint returns:

- accepted and rejected outcome counts
- rejection reasons
- bucket-level performance metrics
- Scanner/Fundamental/Technical/Manager attribution metrics
- win-rate confidence intervals
- confidence calibration error
- guarded source-weight, bucket-threshold, and risk recommendations

## Safety Rules

1. Only `closed` outcomes with `realized` PnL are learnable.
2. Manager, Execution, Database, and final strategy buckets must match.
3. Unsupported evidence or classifier versions are rejected.
4. Duplicate outcome IDs are rejected within each request.
5. Failed, invalid, conflicting, or insufficient evidence is not learned.
6. Minimum total, bucket, and source sample sizes are mandatory.
7. All recommendations include `requires_human_review=true`.
8. `auto_apply` is always `false`.
9. Learning_Agent does not mutate historical buckets or place orders.
