# 2api.ai - Retry & Fallback Policy

## 1. Core Principle: Semantic Safety

> **Golden Rule**: Never retry or fallback if doing so could cause semantic drift.

**Semantic drift** occurs when:
- Model has already produced content
- Retrying would generate different content
- User receives inconsistent/merged responses

## 2. Error Classification

### 2.1 Retryable Errors (Infra-level)

| Error | Code | Retry? | Fallback? | Rationale |
|-------|------|--------|-----------|-----------|
| Connection timeout | `connection_timeout` | ✅ Yes (3x) | ✅ Yes | No content produced |
| DNS resolution failed | `dns_error` | ✅ Yes (2x) | ✅ Yes | No content produced |
| TCP connection reset | `connection_reset` | ✅ Yes (2x) | ✅ Yes | No content produced |
| TLS handshake failed | `tls_error` | ✅ Yes (2x) | ✅ Yes | No content produced |
| HTTP 500 Internal Error | `upstream_500` | ✅ Yes (2x) | ✅ Yes | Server error, no content |
| HTTP 502 Bad Gateway | `upstream_502` | ✅ Yes (2x) | ✅ Yes | Proxy error |
| HTTP 503 Service Unavailable | `upstream_503` | ✅ Yes (2x) | ✅ Yes | Temporary overload |
| HTTP 504 Gateway Timeout | `upstream_504` | ✅ Yes (2x) | ✅ Yes | Upstream timeout |
| HTTP 529 Overloaded (Anthropic) | `upstream_overloaded` | ✅ Yes (3x) | ✅ Yes | Temporary |

### 2.2 Non-Retryable Errors (Semantic-level)

| Error | Code | Retry? | Fallback? | Rationale |
|-------|------|--------|-----------|-----------|
| HTTP 400 Bad Request | `invalid_request` | ❌ No | ❌ No | Client error, fix request |
| HTTP 401 Unauthorized | `authentication_error` | ❌ No | ❌ No | Invalid API key |
| HTTP 403 Forbidden | `permission_denied` | ❌ No | ❌ No | Access denied |
| HTTP 404 Not Found | `model_not_found` | ❌ No | ⚠️ Maybe | Model doesn't exist |
| HTTP 422 Unprocessable | `validation_error` | ❌ No | ❌ No | Schema error |
| Content filtered | `content_filtered` | ❌ No | ❌ No | Policy violation |
| Context length exceeded | `context_length_exceeded` | ❌ No | ⚠️ Maybe | Try smaller model |
| Tool schema invalid | `tool_schema_invalid` | ❌ No | ❌ No | Fix schema |

### 2.3 Rate Limit Errors (Special Handling)

| Error | Code | Retry? | Fallback? | Rationale |
|-------|------|--------|-----------|-----------|
| HTTP 429 Rate Limited | `rate_limited` | ⏳ Wait | ✅ Yes | Respect Retry-After |
| Tokens per minute | `tpm_exceeded` | ⏳ Wait | ✅ Yes | Provider quota |
| Requests per minute | `rpm_exceeded` | ⏳ Wait | ✅ Yes | Provider quota |
| Daily quota | `quota_exceeded` | ❌ No | ✅ Yes | Switch provider |

### 2.4 Mid-Stream Errors (Critical)

| Error | Code | Retry? | Fallback? | Rationale |
|-------|------|--------|-----------|-----------|
| Stream timeout (content started) | `stream_timeout` | ❌ NO | ❌ NO | **Semantic drift** |
| Provider crash mid-stream | `stream_interrupted` | ❌ NO | ❌ NO | **Semantic drift** |
| Rate limit mid-stream | `stream_rate_limited` | ❌ NO | ❌ NO | **Semantic drift** |
| Content filter mid-stream | `stream_content_filtered` | ❌ NO | ❌ NO | **Semantic drift** |

## 3. Retry Policy

### 3.1 Retry Configuration

```typescript
interface RetryConfig {
  maxRetries: number;           // Default: 3
  initialDelayMs: number;       // Default: 1000
  maxDelayMs: number;           // Default: 30000
  backoffMultiplier: number;    // Default: 2
  jitterFactor: number;         // Default: 0.1
}
```

### 3.2 Retry Algorithm

```python
def should_retry(error: Error, attempt: int, config: RetryConfig) -> bool:
    # Never retry semantic errors
    if error.type == "semantic_error":
        return False
    
    # Never retry if content was produced
    if error.partial_content:
        return False
    
    # Check retry count
    if attempt >= config.max_retries:
        return False
    
    # Check if error is retryable
    return error.code in RETRYABLE_ERRORS

def calculate_delay(attempt: int, config: RetryConfig) -> float:
    delay = config.initial_delay_ms * (config.backoff_multiplier ** attempt)
    delay = min(delay, config.max_delay_ms)
    
    # Add jitter to prevent thundering herd
    jitter = delay * config.jitter_factor * random.uniform(-1, 1)
    return delay + jitter
```

### 3.3 Rate Limit Handling

```python
def handle_rate_limit(error: RateLimitError) -> Action:
    if error.retry_after:
        # Provider specified wait time
        if error.retry_after <= 60:
            return Action.WAIT(error.retry_after)
        else:
            return Action.FALLBACK
    else:
        # Default: exponential backoff
        return Action.WAIT(calculate_delay(attempt, config))
```

## 4. Fallback Policy

### 4.1 Fallback Triggers

| Trigger | Fallback Behavior |
|---------|-------------------|
| Primary provider down | Try next in `fallback_chain` |
| Rate limited (long wait) | Try next in `fallback_chain` |
| Model not found | Try equivalent model |
| Max retries exceeded | Try next in `fallback_chain` |
| Context too long | Try model with larger context |

### 4.2 Fallback Chain Configuration

```json
{
  "routing": {
    "fallback": [
      "anthropic/claude-3-5-sonnet",
      "google/gemini-1.5-pro",
      "openai/gpt-4o-mini"
    ]
  }
}
```

### 4.3 Fallback Algorithm

```python
def execute_with_fallback(request: Request, fallback_chain: List[str]) -> Response:
    last_error = None
    
    for model in [request.model] + fallback_chain:
        try:
            response = call_provider(request, model)
            
            # Record if fallback was used
            if model != request.model:
                response._2api.fallback_used = True
                response._2api.original_model = request.model
                response._2api.actual_model = model
            
            return response
            
        except RetryableError as e:
            last_error = e
            # Retry with same model first
            for attempt in range(MAX_RETRIES):
                if not should_retry(e, attempt):
                    break
                await sleep(calculate_delay(attempt))
                try:
                    return call_provider(request, model)
                except RetryableError as retry_error:
                    last_error = retry_error
                    continue
            
            # All retries failed, try next in fallback chain
            continue
            
        except SemanticError as e:
            # Don't fallback for semantic errors
            raise e
    
    # All fallbacks exhausted
    raise AllProvidersFailedError(last_error)
```

### 4.4 Fallback Restrictions

**NEVER fallback when**:
1. Streaming has started and content was produced
2. Error is `semantic_error` (client must fix request)
3. All providers in chain have been tried
4. Request is non-idempotent (e.g., has side effects)

## 5. Partial Failure Handling

### 5.1 Stream Interrupted

```
┌─────────────────────────────────────────────────────────────┐
│ Timeline: Stream with interruption                          │
│                                                             │
│  T0 ──────► T1 ──────► T2 ──────► T3                       │
│  │          │          │          │                         │
│  Start     "Hello"    "world"    TIMEOUT                   │
│  stream    chunk      chunk      (provider dies)           │
│                                                             │
│  Action: Send error chunk with partial_content              │
│          DO NOT retry or fallback                           │
└─────────────────────────────────────────────────────────────┘
```

**Response**:
```json
{
  "error": {
    "code": "stream_interrupted",
    "message": "Provider connection lost during streaming",
    "type": "infra_error",
    "partial_content": "Hello world",
    "recoverable": false
  }
}
```

### 5.2 Client Recovery Options

```python
# Option 1: Accept partial content
if error.partial_content:
    return f"(Partial response) {error.partial_content}"

# Option 2: Retry full request (user-initiated)
if user_confirms_retry:
    response = client.chat(original_messages)  # Fresh request

# Option 3: Continue from partial
continuation_messages = [
    *original_messages,
    {"role": "assistant", "content": error.partial_content},
    {"role": "user", "content": "Please continue from where you left off."}
]
response = client.chat(continuation_messages)
```

## 6. Circuit Breaker

### 6.1 Configuration

```typescript
interface CircuitBreakerConfig {
  failureThreshold: number;      // Default: 5 failures in window
  failureWindowMs: number;       // Default: 60000 (1 minute)
  successThreshold: number;      // Default: 2 successes to close
  openDurationMs: number;        // Default: 30000 (30 seconds)
  halfOpenRequests: number;      // Default: 1 test request
}
```

### 6.2 States

```
     ┌──────────────┐
     │    CLOSED    │◄──────── Normal operation
     └──────┬───────┘
            │ failures >= threshold within window
            ▼
     ┌──────────────┐
     │     OPEN     │◄──────── All requests fail immediately
     └──────┬───────┘          (return cached error or fallback)
            │ openDuration expires
            ▼
     ┌──────────────┐
     │  HALF-OPEN   │◄──────── Test with limited requests
     └──────┬───────┘
            │
      ┌─────┴─────┐
      │           │
  success      failure
      │           │
      ▼           ▼
   CLOSED       OPEN
```

### 6.3 Per-Provider Circuit Breaker Rules

| Provider State | Behavior | Metrics Tracked |
|----------------|----------|-----------------|
| CLOSED | Normal routing | failures, latency |
| OPEN | Skip provider, use fallback | time since open |
| HALF-OPEN | Send 1 test request | test result |

### 6.4 Circuit Breaker Implementation

```python
class ProviderCircuitBreaker:
    def __init__(self, provider: str, config: CircuitBreakerConfig):
        self.provider = provider
        self.config = config
        self.state = CircuitState.CLOSED
        self.failures: List[float] = []  # Timestamps of failures
        self.last_state_change = time.time()
        self.half_open_successes = 0
    
    def record_success(self):
        # Clear old failures outside window
        self._cleanup_old_failures()
        
        if self.state == CircuitState.HALF_OPEN:
            self.half_open_successes += 1
            if self.half_open_successes >= self.config.success_threshold:
                self._transition_to(CircuitState.CLOSED)
        elif self.state == CircuitState.CLOSED:
            # Reset failure count on success
            self.failures.clear()
    
    def record_failure(self, error: Exception):
        self._cleanup_old_failures()
        self.failures.append(time.time())
        
        if self.state == CircuitState.HALF_OPEN:
            # Any failure in half-open -> back to open
            self._transition_to(CircuitState.OPEN)
        elif self.state == CircuitState.CLOSED:
            if len(self.failures) >= self.config.failure_threshold:
                self._transition_to(CircuitState.OPEN)
                emit_metric("circuit_breaker.opened", provider=self.provider)
    
    def can_execute(self) -> bool:
        if self.state == CircuitState.CLOSED:
            return True
        
        if self.state == CircuitState.OPEN:
            # Check if open duration has passed
            elapsed = time.time() - self.last_state_change
            if elapsed >= self.config.open_duration_ms / 1000:
                self._transition_to(CircuitState.HALF_OPEN)
                return True
            return False
        
        if self.state == CircuitState.HALF_OPEN:
            # Only allow configured number of test requests
            return self.half_open_successes < self.config.half_open_requests
        
        return False
    
    def _transition_to(self, new_state: CircuitState):
        old_state = self.state
        self.state = new_state
        self.last_state_change = time.time()
        
        if new_state == CircuitState.HALF_OPEN:
            self.half_open_successes = 0
        elif new_state == CircuitState.CLOSED:
            self.failures.clear()
        
        log.info(f"Circuit breaker {self.provider}: {old_state} -> {new_state}")
    
    def _cleanup_old_failures(self):
        cutoff = time.time() - (self.config.failure_window_ms / 1000)
        self.failures = [f for f in self.failures if f > cutoff]
```

### 6.5 Circuit Breaker Metrics

```python
# Emit these metrics for monitoring
emit_metric("circuit_breaker.state", state=self.state, provider=self.provider)
emit_metric("circuit_breaker.failures", count=len(self.failures), provider=self.provider)
emit_metric("circuit_breaker.opened", provider=self.provider)  # Event
emit_metric("circuit_breaker.closed", provider=self.provider)  # Event
```

## 7. Retry Budget (Per-Tenant)

### 7.1 Why Retry Budget?

Unlimited retries can:
- Burn through tenant's token quota
- Amplify costs during outages
- Create "retry storms" that worsen provider issues

### 7.2 Retry Budget Configuration

```typescript
interface RetryBudgetConfig {
  maxRetriesPerMinute: number;     // Default: 100
  maxRetryTokensPerMinute: number; // Default: 500,000
  maxRetryCostPerMinute: number;   // Default: $10.00
  budgetWindowMs: number;          // Default: 60000 (1 minute)
}
```

### 7.3 Retry Budget Enforcement

```python
class TenantRetryBudget:
    def __init__(self, tenant_id: str, config: RetryBudgetConfig):
        self.tenant_id = tenant_id
        self.config = config
        # Redis keys for tracking
        self.retry_count_key = f"retry_budget:{tenant_id}:count"
        self.retry_tokens_key = f"retry_budget:{tenant_id}:tokens"
        self.retry_cost_key = f"retry_budget:{tenant_id}:cost"
    
    async def can_retry(self, estimated_tokens: int, estimated_cost: float) -> bool:
        """Check if tenant has retry budget remaining."""
        pipe = redis.pipeline()
        pipe.get(self.retry_count_key)
        pipe.get(self.retry_tokens_key)
        pipe.get(self.retry_cost_key)
        results = await pipe.execute()
        
        current_count = int(results[0] or 0)
        current_tokens = int(results[1] or 0)
        current_cost = float(results[2] or 0)
        
        if current_count >= self.config.max_retries_per_minute:
            log.warning(f"Tenant {self.tenant_id} retry count budget exhausted")
            return False
        
        if current_tokens + estimated_tokens > self.config.max_retry_tokens_per_minute:
            log.warning(f"Tenant {self.tenant_id} retry token budget exhausted")
            return False
        
        if current_cost + estimated_cost > self.config.max_retry_cost_per_minute:
            log.warning(f"Tenant {self.tenant_id} retry cost budget exhausted")
            return False
        
        return True
    
    async def record_retry(self, tokens: int, cost: float):
        """Record a retry against budget."""
        pipe = redis.pipeline()
        
        # Increment counters with 60s TTL
        pipe.incr(self.retry_count_key)
        pipe.expire(self.retry_count_key, 60)
        
        pipe.incrbyfloat(self.retry_tokens_key, tokens)
        pipe.expire(self.retry_tokens_key, 60)
        
        pipe.incrbyfloat(self.retry_cost_key, cost)
        pipe.expire(self.retry_cost_key, 60)
        
        await pipe.execute()
```

### 7.4 Retry Budget Exhaustion Response

When budget is exhausted, return error instead of retry:

```json
{
  "error": {
    "code": "retry_budget_exhausted",
    "message": "Retry budget exceeded. Please wait before retrying.",
    "type": "infra_error",
    "retryable": true,
    "retry_after": 60,
    "details": {
      "budget_type": "cost",
      "budget_limit": 10.00,
      "budget_used": 10.50,
      "reset_in_seconds": 45
    }
  }
}
```

### 7.5 Retry Budget Tiers

| Plan | Retries/min | Tokens/min | Cost/min |
|------|-------------|------------|----------|
| Free | 10 | 50,000 | $1.00 |
| Starter | 50 | 200,000 | $5.00 |
| Pro | 100 | 500,000 | $10.00 |
| Enterprise | Custom | Custom | Custom |

## 8. Observability

### 7.1 Metrics to Track

| Metric | Description |
|--------|-------------|
| `retry_count` | Total retries per provider |
| `retry_success_rate` | % of retries that succeeded |
| `fallback_count` | Total fallbacks triggered |
| `fallback_success_rate` | % of fallbacks that succeeded |
| `circuit_breaker_opens` | Times circuit opened |
| `partial_failure_count` | Streams interrupted mid-way |

### 7.2 Response Headers

```http
X-2api-Request-Id: req_abc123
X-2api-Retry-Count: 2
X-2api-Fallback-Used: true
X-2api-Original-Model: openai/gpt-4o
X-2api-Actual-Model: anthropic/claude-3-5-sonnet
X-2api-Provider-Request-Id: chatcmpl-xyz789
```

## 8. Summary Table

| Error Type | Retry | Fallback | Wait | Rationale |
|------------|-------|----------|------|-----------|
| Connection error | ✅ 3x | ✅ | - | Transient |
| 5xx errors | ✅ 2x | ✅ | - | Server error |
| 429 Rate limit | ✅ | ✅ | Retry-After | Respect limits |
| 400 Bad request | ❌ | ❌ | - | Client error |
| 401/403 Auth | ❌ | ❌ | - | Fix credentials |
| Content filtered | ❌ | ❌ | - | Policy violation |
| **Mid-stream error** | ❌ | ❌ | - | **Semantic drift** |
