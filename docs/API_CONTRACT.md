# Learning_Agent API Contract

This document defines the baseline API contract for `Learning_Agent`.

`Learning_Agent` provides learning reviews for trade history, portfolio audits, performance summaries, market regime classification, and bias-state updates.

## Standard Headers

```http
Content-Type: application/json
X-Correlation-ID: <uuid>
X-API-KEY: <learning-agent-api-key>
```

## Standard Response Envelope

Operational contract endpoints return this envelope:

```json
{
  "status": "success",
  "agent_type": "learning",
  "version": "1.0.0",
  "schema_version": "1.0",
  "timestamp": "2026-07-04T00:00:00Z",
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
POST /market-regime
POST /learning/update-biases
```

## Notes

1. This service returns learning and policy-review output for other agents.
2. Runtime readiness is reported through `/ready`.
3. Version and schema metadata are reported through `/version`.
4. Existing learning endpoints keep their current response models.
