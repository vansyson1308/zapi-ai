# 2api.ai - Error Taxonomy

## 1. Error Classification Philosophy

Errors are classified into two distinct categories:

| Category | Description | Client Action |
|----------|-------------|---------------|
| **Infra Error** | Platform/network issues, not client's fault | Retry or wait |
| **Semantic Error** | Request/content issues, client must fix | Modify request |

## 2. Unified Error Response Format

```typescript
interface ErrorResponse {
  error: {
    // Core fields (always present)
    code: string;                    // Machine-readable error code
    message: string;                 // Human-readable message
    type: "infra_error" | "semantic_error";
    
    // Context fields (when applicable)
    provider?: string;               // Which provider caused error
    param?: string;                  // Which parameter was invalid
    
    // Trace fields (always present)
    request_id: string;              // 2api.ai request ID (format: req_{ulid})
    trace_id: string;                // Distributed trace ID (format: trace_{ulid})
    provider_request_id?: string;    // Upstream provider's request ID
    
    // Recovery fields (when applicable)
    retryable: boolean;              // Can client retry?
    retry_after?: number;            // Seconds to wait before retry
    fallback_attempted?: boolean;    // Did 2api.ai try fallback?
    partial_content?: string;        // Content before failure (streaming)
    
    // Debug fields (when applicable)
    details?: Record<string, any>;   // Additional context
  };
}
```

### 2.1 Trace ID Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Request Tracing                                  │
│                                                                          │
│  Client Request                                                          │
│       │                                                                  │
│       ▼                                                                  │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ 2api.ai Gateway                                                  │    │
│  │                                                                  │    │
│  │  request_id: req_01HQ3K5V8YCJN4XWZP0QRMVT6B                     │    │
│  │  trace_id:   trace_01HQ3K5V8YCJN4XWZP0QRMVT6B (same or linked) │    │
│  │  span_id:    span_gateway_01HQ3K5V...                           │    │
│  └───────────────────────┬─────────────────────────────────────────┘    │
│                          │                                               │
│                          ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ Provider Call                                                    │    │
│  │                                                                  │    │
│  │  provider_request_id: chatcmpl-abc123... (from OpenAI)          │    │
│  │  span_id: span_openai_01HQ3K5V...                               │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  All logs, metrics, errors include: request_id + trace_id               │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 ID Formats

| ID Type | Format | Example | Source |
|---------|--------|---------|--------|
| `request_id` | `req_{ulid}` | `req_01HQ3K5V8YCJN4XWZP0QRMVT6B` | 2api.ai generated |
| `trace_id` | `trace_{ulid}` or W3C | `trace_01HQ3K5V8Y...` or `00-abc123...` | 2api.ai or propagated |
| `provider_request_id` | Provider-specific | `chatcmpl-abc123xyz` | From provider response |
| `span_id` | Internal | `span_router_01HQ...` | For observability |

### 2.3 Trace Propagation

2api.ai supports W3C Trace Context:

```http
# Incoming request (optional)
traceparent: 00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01

# 2api.ai will:
# 1. Use provided trace_id if valid
# 2. Generate new trace_id if not provided
# 3. Create child span for provider calls
```

## 3. Infra Error Codes

### 3.1 Network Errors

| Code | HTTP | Message | Retryable |
|------|------|---------|-----------|
| `connection_timeout` | 504 | Failed to connect to provider | ✅ Yes |
| `read_timeout` | 504 | Provider did not respond in time | ✅ Yes |
| `connection_reset` | 502 | Connection was reset | ✅ Yes |
| `dns_error` | 502 | Failed to resolve provider hostname | ✅ Yes |
| `tls_error` | 502 | TLS handshake failed | ✅ Yes |
| `network_error` | 502 | Generic network failure | ✅ Yes |

**Example**:
```json
{
  "error": {
    "code": "connection_timeout",
    "message": "Failed to connect to OpenAI API within 10 seconds",
    "type": "infra_error",
    "provider": "openai",
    "request_id": "req_abc123",
    "retryable": true,
    "retry_after": 5
  }
}
```

### 3.2 Provider Errors

| Code | HTTP | Message | Retryable |
|------|------|---------|-----------|
| `upstream_500` | 502 | Provider returned internal error | ✅ Yes |
| `upstream_502` | 502 | Provider's upstream failed | ✅ Yes |
| `upstream_503` | 503 | Provider temporarily unavailable | ✅ Yes |
| `upstream_504` | 504 | Provider gateway timeout | ✅ Yes |
| `upstream_overloaded` | 503 | Provider is overloaded | ✅ Yes |
| `provider_down` | 503 | Provider appears to be down | ✅ Fallback |
| `provider_error` | 502 | Unknown provider error | ⚠️ Maybe |

**Example**:
```json
{
  "error": {
    "code": "upstream_503",
    "message": "Anthropic API is temporarily unavailable",
    "type": "infra_error",
    "provider": "anthropic",
    "request_id": "req_abc123",
    "provider_request_id": "req-ant-xyz789",
    "retryable": true,
    "retry_after": 30,
    "fallback_attempted": true
  }
}
```

### 3.3 Rate Limit Errors

| Code | HTTP | Message | Retryable |
|------|------|---------|-----------|
| `rate_limited` | 429 | Rate limit exceeded | ⏳ Wait |
| `tpm_exceeded` | 429 | Tokens per minute limit exceeded | ⏳ Wait |
| `rpm_exceeded` | 429 | Requests per minute limit exceeded | ⏳ Wait |
| `daily_quota_exceeded` | 429 | Daily quota exceeded | ❌ No (today) |
| `tenant_rate_limited` | 429 | Your 2api.ai rate limit exceeded | ⏳ Wait |

**Example**:
```json
{
  "error": {
    "code": "rate_limited",
    "message": "OpenAI rate limit exceeded. Retry after 45 seconds.",
    "type": "infra_error",
    "provider": "openai",
    "request_id": "req_abc123",
    "retryable": true,
    "retry_after": 45,
    "details": {
      "limit_type": "tokens_per_minute",
      "limit": 90000,
      "used": 92500,
      "reset_at": "2024-01-15T10:30:00Z"
    }
  }
}
```

### 3.4 Stream Errors

| Code | HTTP | Message | Retryable |
|------|------|---------|-----------|
| `stream_timeout` | 504 | Stream timed out | ❌ No* |
| `stream_interrupted` | 502 | Stream was interrupted | ❌ No* |
| `stream_connection_lost` | 502 | Lost connection during stream | ❌ No* |

*Not retryable if content was already produced (semantic drift risk)

**Example**:
```json
{
  "error": {
    "code": "stream_interrupted",
    "message": "Connection lost after receiving partial content",
    "type": "infra_error",
    "provider": "openai",
    "request_id": "req_abc123",
    "retryable": false,
    "partial_content": "The weather in Tokyo is currently"
  }
}
```

### 3.5 Internal Errors

| Code | HTTP | Message | Retryable |
|------|------|---------|-----------|
| `internal_error` | 500 | Unexpected internal error | ⚠️ Maybe |
| `configuration_error` | 500 | Server misconfiguration | ❌ No |
| `database_error` | 500 | Database unavailable | ✅ Yes |
| `router_error` | 500 | Routing decision failed | ✅ Yes |

## 4. Semantic Error Codes

### 4.1 Authentication Errors

| Code | HTTP | Message | Action |
|------|------|---------|--------|
| `invalid_api_key` | 401 | Invalid API key | Check key |
| `expired_api_key` | 401 | API key has expired | Renew key |
| `missing_api_key` | 401 | No API key provided | Add header |
| `insufficient_permissions` | 403 | API key lacks permission | Upgrade plan |
| `ip_not_allowed` | 403 | Request from disallowed IP | Check allowlist |

**Example**:
```json
{
  "error": {
    "code": "invalid_api_key",
    "message": "The API key provided is invalid. Check that it starts with '2api_'",
    "type": "semantic_error",
    "request_id": "req_abc123",
    "retryable": false,
    "details": {
      "key_prefix": "sk_",
      "expected_prefix": "2api_"
    }
  }
}
```

### 4.2 Request Validation Errors

| Code | HTTP | Message | Action |
|------|------|---------|--------|
| `invalid_request` | 400 | Malformed request body | Fix JSON |
| `missing_required_field` | 400 | Required field missing | Add field |
| `invalid_field_type` | 400 | Field has wrong type | Fix type |
| `invalid_field_value` | 400 | Field value out of range | Fix value |
| `invalid_model` | 400 | Model identifier invalid | Check model |
| `model_not_found` | 404 | Model does not exist | Use valid model |

**Example**:
```json
{
  "error": {
    "code": "missing_required_field",
    "message": "'messages' is required",
    "type": "semantic_error",
    "param": "messages",
    "request_id": "req_abc123",
    "retryable": false
  }
}
```

### 4.3 Content Errors

| Code | HTTP | Message | Action |
|------|------|---------|--------|
| `content_filtered` | 400 | Content violates usage policy | Modify content |
| `content_too_long` | 400 | Input exceeds context window | Reduce input |
| `invalid_image_format` | 400 | Image format not supported | Convert image |
| `image_too_large` | 400 | Image exceeds size limit | Resize image |
| `unsupported_media_type` | 415 | Media type not supported | Change format |

**Example**:
```json
{
  "error": {
    "code": "content_filtered",
    "message": "Your request was flagged by content moderation",
    "type": "semantic_error",
    "provider": "openai",
    "request_id": "req_abc123",
    "retryable": false,
    "details": {
      "filter_reason": "violence",
      "flagged_content": "[REDACTED]"
    }
  }
}
```

### 4.4 Tool Calling Errors

| Code | HTTP | Message | Action |
|------|------|---------|--------|
| `tool_schema_invalid` | 400 | Tool schema is invalid | Fix schema |
| `tool_schema_incompatible` | 400 | Schema incompatible with provider | Simplify schema |
| `tool_not_found` | 400 | Referenced tool doesn't exist | Check tool name |
| `tool_call_id_invalid` | 400 | Invalid tool_call_id | Check ID |
| `tool_result_too_large` | 400 | Tool result exceeds limit | Reduce result |

**Example**:
```json
{
  "error": {
    "code": "tool_schema_incompatible",
    "message": "Tool 'search_db' uses 'oneOf' which is not supported by Google Gemini",
    "type": "semantic_error",
    "param": "tools[0].function.parameters",
    "request_id": "req_abc123",
    "retryable": false,
    "details": {
      "tool_name": "search_db",
      "unsupported_features": ["oneOf", "anyOf"],
      "provider": "google"
    }
  }
}
```

### 4.5 Routing Errors

| Code | HTTP | Message | Action |
|------|------|---------|--------|
| `no_provider_available` | 503 | All providers unavailable | Wait/retry |
| `routing_policy_violation` | 400 | Request violates routing policy | Check policy |
| `budget_exceeded` | 402 | Request would exceed budget | Increase budget |
| `model_capability_mismatch` | 400 | Model doesn't support feature | Use compatible model |

## 5. Error Response Headers

All responses (success and error) include these headers:

### 5.1 Required Headers (Always Present)

```http
X-Request-Id: req_01HQ3K5V8YCJN4XWZP0QRMVT6B
X-Trace-Id: trace_01HQ3K5V8YCJN4XWZP0QRMVT6B
```

### 5.2 Conditional Headers

```http
# When request hit a provider
X-Provider: openai
X-Provider-Request-Id: chatcmpl-xyz789

# When error occurred
X-Error-Type: infra_error
X-Error-Code: rate_limited

# When rate limited
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 0
X-RateLimit-Reset: 1700000060
Retry-After: 30

# When fallback was attempted
X-Fallback-Attempted: true
X-Original-Provider: openai
X-Actual-Provider: anthropic
```

### 5.3 Header Reference Table

| Header | When Present | Description |
|--------|--------------|-------------|
| `X-Request-Id` | Always | Unique request identifier |
| `X-Trace-Id` | Always | Distributed trace ID |
| `X-Provider` | After routing | Provider that handled request |
| `X-Provider-Request-Id` | If provider returned one | Provider's request ID |
| `X-Error-Type` | On error | `infra_error` or `semantic_error` |
| `X-Error-Code` | On error | Machine-readable error code |
| `X-RateLimit-Limit` | Always | Requests allowed per minute |
| `X-RateLimit-Remaining` | Always | Requests remaining |
| `X-RateLimit-Reset` | Always | Unix timestamp of reset |
| `Retry-After` | On 429 | Seconds until retry allowed |
| `X-Fallback-Attempted` | On fallback | Whether fallback was tried |
| `X-Latency-Ms` | Always | Total request latency |
| `X-Tokens-Used` | On success | Total tokens consumed |
| `X-Cost-Usd` | On success | Estimated cost |

### 5.4 CORS Headers (for browser clients)

```http
Access-Control-Allow-Origin: *
Access-Control-Expose-Headers: X-Request-Id, X-Trace-Id, X-RateLimit-Limit, X-RateLimit-Remaining, X-RateLimit-Reset, Retry-After, X-Error-Type, X-Error-Code
```

## 6. Error Handling Best Practices

### 6.1 Client-Side Handling

```python
from twoapi import TwoAPIError, InfraError, SemanticError, RateLimitError

try:
    response = client.chat("Hello")
    
except RateLimitError as e:
    # Wait and retry
    time.sleep(e.retry_after)
    response = client.chat("Hello")
    
except InfraError as e:
    if e.retryable:
        # Retry with exponential backoff
        for attempt in range(3):
            try:
                response = client.chat("Hello")
                break
            except InfraError:
                time.sleep(2 ** attempt)
    else:
        # Handle partial content if available
        if e.partial_content:
            log.warning(f"Partial response: {e.partial_content}")
        raise
        
except SemanticError as e:
    # Client must fix the request
    log.error(f"Invalid request: {e.message}, param: {e.param}")
    raise
```

### 6.2 Logging

```python
# All errors should log these fields
{
    "timestamp": "2024-01-15T10:30:00Z",
    "request_id": "req_abc123",
    "provider_request_id": "chatcmpl-xyz789",
    "error_code": "rate_limited",
    "error_type": "infra_error",
    "provider": "openai",
    "retryable": True,
    "tenant_id": "tenant_abc",
    "model": "openai/gpt-4o"
}
```

## 7. Error Code Reference Table

| Code | Type | HTTP | Retryable | Fallback |
|------|------|------|-----------|----------|
| `connection_timeout` | Infra | 504 | ✅ | ✅ |
| `read_timeout` | Infra | 504 | ✅ | ✅ |
| `upstream_500` | Infra | 502 | ✅ | ✅ |
| `upstream_503` | Infra | 503 | ✅ | ✅ |
| `rate_limited` | Infra | 429 | ⏳ | ✅ |
| `stream_interrupted` | Infra | 502 | ❌ | ❌ |
| `invalid_api_key` | Semantic | 401 | ❌ | ❌ |
| `invalid_request` | Semantic | 400 | ❌ | ❌ |
| `content_filtered` | Semantic | 400 | ❌ | ❌ |
| `tool_schema_invalid` | Semantic | 400 | ❌ | ❌ |
| `model_not_found` | Semantic | 404 | ❌ | ⚠️ |
| `budget_exceeded` | Semantic | 402 | ❌ | ❌ |
