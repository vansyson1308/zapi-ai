# 2api.ai - Specification Index

> **Single Source of Truth** for all 2api.ai platform specifications.
> 
> Version: 0.1.0 | Last Updated: 2026-02

## Quick Links

| Spec | Description | Status |
|------|-------------|--------|
| [Architecture](./ARCHITECTURE.md) | System design, components, data flow | ✅ Stable |
| [OpenAPI](./openapi.yaml) | REST API specification | ✅ Stable |
| [Streaming](./STREAMING_SPEC.md) | SSE protocol, chunk format, error handling | ✅ Stable |
| [Tool Calling](./TOOL_CALLING_SPEC.md) | Function calling, cross-provider normalization | ✅ Stable |
| [Retry & Fallback](./RETRY_FALLBACK_POLICY.md) | Error handling, circuit breaker, semantic safety | ✅ Stable |
| [Error Taxonomy](./ERROR_TAXONOMY.md) | Error classification, codes, response format | ✅ Stable |
| [Multi-Tenant](./MULTI_TENANT_DESIGN.md) | Isolation, billing, rate limiting | ✅ Stable |

---

## Spec by Feature

### 🔌 API Surface

| Feature | Primary Spec | Related |
|---------|--------------|---------|
| REST Endpoints | [OpenAPI](./openapi.yaml) | [Architecture](./ARCHITECTURE.md) |
| Authentication | [Multi-Tenant](./MULTI_TENANT_DESIGN.md#3-api-key-system) | [Error Taxonomy](./ERROR_TAXONOMY.md#41-authentication-errors) |
| Rate Limiting | [Multi-Tenant](./MULTI_TENANT_DESIGN.md#5-per-tenant-rate-limiting) | [Error Taxonomy](./ERROR_TAXONOMY.md#33-rate-limit-errors) |

### 🌊 Streaming

| Feature | Primary Spec | Related |
|---------|--------------|---------|
| SSE Protocol | [Streaming](./STREAMING_SPEC.md#1-protocol) | - |
| Chunk Format | [Streaming](./STREAMING_SPEC.md#3-unified-chunk-schema) | [OpenAPI](./openapi.yaml) |
| Tool Call Streaming | [Streaming](./STREAMING_SPEC.md#22-tool-call-delta-chunk) | [Tool Calling](./TOOL_CALLING_SPEC.md) |
| Partial Failure | [Streaming](./STREAMING_SPEC.md#5-error-during-stream) | [Retry Policy](./RETRY_FALLBACK_POLICY.md#5-partial-failure-handling) |
| Keep-Alive | [Streaming](./STREAMING_SPEC.md#9-keep-alive--heartbeat) | - |
| Reconnection | [Streaming](./STREAMING_SPEC.md#10-reconnect-semantics) | - |
| Backpressure | [Streaming](./STREAMING_SPEC.md#11-backpressure-handling) | - |

### 🔧 Tool Calling

| Feature | Primary Spec | Related |
|---------|--------------|---------|
| Tool Schema | [Tool Calling](./TOOL_CALLING_SPEC.md#2-unified-tool-schema) | [OpenAPI](./openapi.yaml) |
| Provider Normalization | [Tool Calling](./TOOL_CALLING_SPEC.md#3-provider-normalization) | - |
| Execution Loop | [Tool Calling](./TOOL_CALLING_SPEC.md#4-execution-loop-ownership) | - |
| Guardrails | [Tool Calling](./TOOL_CALLING_SPEC.md#7-guardrails--limits) | - |
| Content Semantics | [Tool Calling](./TOOL_CALLING_SPEC.md#8-tool-result-content-semantics) | - |

### 🔄 Reliability

| Feature | Primary Spec | Related |
|---------|--------------|---------|
| Retry Policy | [Retry & Fallback](./RETRY_FALLBACK_POLICY.md#3-retry-policy) | - |
| Fallback Chain | [Retry & Fallback](./RETRY_FALLBACK_POLICY.md#4-fallback-policy) | [Architecture](./ARCHITECTURE.md#32-router-service) |
| Circuit Breaker | [Retry & Fallback](./RETRY_FALLBACK_POLICY.md#6-circuit-breaker) | - |
| Retry Budget | [Retry & Fallback](./RETRY_FALLBACK_POLICY.md#7-retry-budget-per-tenant) | [Multi-Tenant](./MULTI_TENANT_DESIGN.md) |
| Semantic Safety | [Retry & Fallback](./RETRY_FALLBACK_POLICY.md#1-core-principle-semantic-safety) | - |

### ❌ Error Handling

| Feature | Primary Spec | Related |
|---------|--------------|---------|
| Error Classification | [Error Taxonomy](./ERROR_TAXONOMY.md#1-error-classification-philosophy) | - |
| Error Response Format | [Error Taxonomy](./ERROR_TAXONOMY.md#2-unified-error-response-format) | [OpenAPI](./openapi.yaml) |
| Infra Errors | [Error Taxonomy](./ERROR_TAXONOMY.md#3-infra-error-codes) | [Retry Policy](./RETRY_FALLBACK_POLICY.md) |
| Semantic Errors | [Error Taxonomy](./ERROR_TAXONOMY.md#4-semantic-error-codes) | - |
| Response Headers | [Error Taxonomy](./ERROR_TAXONOMY.md#5-error-response-headers) | - |
| Tracing | [Error Taxonomy](./ERROR_TAXONOMY.md#21-trace-id-architecture) | - |

### 👥 Multi-Tenancy

| Feature | Primary Spec | Related |
|---------|--------------|---------|
| Tenant Isolation | [Multi-Tenant](./MULTI_TENANT_DESIGN.md#2-tenant-isolation-model) | - |
| API Keys | [Multi-Tenant](./MULTI_TENANT_DESIGN.md#3-api-key-system) | [Error Taxonomy](./ERROR_TAXONOMY.md) |
| Key Rotation | [Multi-Tenant](./MULTI_TENANT_DESIGN.md#4-api-key-lifecycle-management) | - |
| Rate Limiting | [Multi-Tenant](./MULTI_TENANT_DESIGN.md#5-per-tenant-rate-limiting) | - |
| Usage Tracking | [Multi-Tenant](./MULTI_TENANT_DESIGN.md#6-usage-tracking) | - |
| Billing | [Multi-Tenant](./MULTI_TENANT_DESIGN.md#7-billing) | - |

---

## Spec Versioning

### Version Policy

- **Major version** (1.x → 2.x): Breaking changes
- **Minor version** (1.0 → 1.1): New features, backward compatible
- **Spec status**: Draft → Stable → Deprecated

### Changelog

| Version | Date | Changes |
|---------|------|---------|
| 0.1.0 | 2026-02 | Initial release: Core specs for Phase 1 |

---

## Contract Tests

All specs have corresponding contract tests in [`/tests/test_contracts.py`](../tests/test_contracts.py):

```bash
# Run all contract tests
pytest tests/test_contracts.py -v

# Run specific spec tests
pytest tests/test_contracts.py::TestStreamingContract -v
pytest tests/test_contracts.py::TestToolCallingContract -v
pytest tests/test_contracts.py::TestErrorTaxonomyContract -v
pytest tests/test_contracts.py::TestMultiTenantContract -v
```

---

## Implementation Status

| Component | Spec | Implementation | Tests |
|-----------|------|----------------|-------|
| Core Models | ✅ | `src/core/models.py` | ✅ |
| Error Handling | ✅ | `src/core/errors.py` | ✅ |
| OpenAI Adapter | ✅ | `src/adapters/openai_adapter.py` | 🔄 |
| Anthropic Adapter | ✅ | `src/adapters/anthropic_adapter.py` | 🔄 |
| Google Adapter | ✅ | `src/adapters/google_adapter.py` | 🔄 |
| Router | ✅ | `src/routing/router.py` | 🔄 |
| API Server | ✅ | `src/server.py` | 🔄 |
| Python SDK | ✅ | `src/sdk/python/twoapi.py` | 🔄 |
| JS SDK | ✅ | `src/sdk/javascript/twoapi.ts` | 🔄 |

Legend: ✅ Complete | 🔄 In Progress | ❌ Not Started

---

## How to Use This Index

1. **Building a feature?** → Find the relevant spec in "Spec by Feature"
2. **Debugging an error?** → Check [Error Taxonomy](./ERROR_TAXONOMY.md)
3. **Implementing streaming?** → Start with [Streaming Spec](./STREAMING_SPEC.md)
4. **Adding a provider?** → See [Architecture](./ARCHITECTURE.md) + [Tool Calling](./TOOL_CALLING_SPEC.md)
5. **Validating implementation?** → Run contract tests

---

## Contributing to Specs

1. All changes must update relevant contract tests
2. Breaking changes require major version bump
3. New features require examples in spec
4. All specs must be reviewed before merge
