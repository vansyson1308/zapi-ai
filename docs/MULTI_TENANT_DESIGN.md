# 2api.ai - Multi-Tenant Design

## 1. Overview

2api.ai is designed as a **multi-tenant SaaS platform** where each customer (tenant) has completely isolated data, usage tracking, and billing.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           2api.ai Platform                               │
│                                                                          │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐                   │
│  │  Tenant A   │   │  Tenant B   │   │  Tenant C   │                   │
│  │             │   │             │   │             │                   │
│  │ • API Keys  │   │ • API Keys  │   │ • API Keys  │   ...             │
│  │ • Usage     │   │ • Usage     │   │ • Usage     │                   │
│  │ • Billing   │   │ • Billing   │   │ • Billing   │                   │
│  │ • Settings  │   │ • Settings  │   │ • Settings  │                   │
│  └─────────────┘   └─────────────┘   └─────────────┘                   │
│         │                 │                 │                           │
│         └─────────────────┼─────────────────┘                           │
│                           │                                              │
│                    ┌──────▼──────┐                                      │
│                    │   Router    │                                      │
│                    └──────┬──────┘                                      │
│                           │                                              │
│         ┌─────────────────┼─────────────────┐                           │
│         ▼                 ▼                 ▼                           │
│    ┌─────────┐      ┌─────────┐      ┌─────────┐                       │
│    │ OpenAI  │      │ Claude  │      │ Gemini  │                       │
│    └─────────┘      └─────────┘      └─────────┘                       │
└─────────────────────────────────────────────────────────────────────────┘
```

## 2. Tenant Isolation Model

### 2.1 Data Isolation Layers

| Layer | Isolation Method | Implementation |
|-------|------------------|----------------|
| API Keys | Tenant-scoped | `api_keys.tenant_id` FK |
| Usage Records | Tenant-scoped | `usage_records.tenant_id` FK |
| Billing | Tenant-scoped | `billing_records.tenant_id` FK |
| Rate Limits | Per-tenant | Redis key prefix |
| Provider Keys | Optional per-tenant | `provider_configs.tenant_id` |
| Logs | Tenant-tagged | Structured logging |

### 2.2 Tenant ID Propagation

```
Request Flow:
                                                          
  API Key ──► Authentication ──► tenant_id ──► All Operations
       │                              │
       │                              ├──► Usage Tracking
       │                              ├──► Rate Limiting
       │                              ├──► Cost Calculation
       │                              └──► Logging
```

Every operation in the system carries `tenant_id`:

```python
@dataclass
class RequestContext:
    request_id: str
    tenant_id: str
    api_key_id: str
    user_agent: str
    ip_address: str
    timestamp: datetime

# Middleware injects context
async def authenticate(request: Request) -> RequestContext:
    api_key = extract_api_key(request)
    key_record = await validate_api_key(api_key)
    
    return RequestContext(
        request_id=generate_request_id(),
        tenant_id=key_record.tenant_id,
        api_key_id=key_record.id,
        user_agent=request.headers.get("User-Agent"),
        ip_address=request.client.host,
        timestamp=datetime.utcnow()
    )
```

## 3. API Key System

### 3.1 Key Format

```
2api_{tenant_prefix}_{random_string}
│     │               │
│     │               └─ 24 random chars (base62)
│     └─ 4 char tenant prefix
└─ Platform identifier

Example: 2api_acme_x7k9m2p4q8r1s3t5u6v0w2y
```

### 3.2 Key Storage

```sql
CREATE TABLE api_keys (
    id UUID PRIMARY KEY,
    tenant_id UUID NOT NULL REFERENCES tenants(id),
    
    -- Key identification
    key_hash VARCHAR(64) NOT NULL,      -- SHA-256(full_key)
    key_prefix VARCHAR(16) NOT NULL,    -- "2api_acme_x7k9" for lookup
    
    -- Metadata
    name VARCHAR(255) DEFAULT 'Default Key',
    description TEXT,
    
    -- Permissions
    permissions JSONB DEFAULT '["*"]',  -- ["chat", "embeddings"] or ["*"]
    allowed_models JSONB DEFAULT '["*"]', -- ["openai/*", "anthropic/claude-3-5-sonnet"]
    allowed_ips JSONB DEFAULT '[]',     -- ["192.168.1.0/24"]
    
    -- Limits
    rate_limit_rpm INTEGER DEFAULT 60,   -- Requests per minute
    rate_limit_tpm INTEGER DEFAULT 100000, -- Tokens per minute
    monthly_budget_usd DECIMAL(10,2),    -- Optional spending cap
    
    -- Status
    is_active BOOLEAN DEFAULT TRUE,
    expires_at TIMESTAMP WITH TIME ZONE,
    last_used_at TIMESTAMP WITH TIME ZONE,
    
    -- Audit
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by UUID REFERENCES users(id)
);

-- Index for key lookup (hash-based)
CREATE INDEX idx_api_keys_hash ON api_keys(key_hash);
-- Index for prefix lookup (for UI display)
CREATE INDEX idx_api_keys_prefix ON api_keys(key_prefix);
```

### 3.3 Key Validation Flow

```python
async def validate_api_key(key: str) -> APIKeyRecord:
    # 1. Basic format check
    if not key.startswith("2api_"):
        raise InvalidAPIKeyError("Key must start with '2api_'")
    
    # 2. Hash the key
    key_hash = hashlib.sha256(key.encode()).hexdigest()
    
    # 3. Look up in database
    record = await db.fetch_one(
        "SELECT * FROM api_keys WHERE key_hash = $1",
        key_hash
    )
    
    if not record:
        raise InvalidAPIKeyError("Invalid API key")
    
    # 4. Check if active
    if not record.is_active:
        raise InvalidAPIKeyError("API key is deactivated")
    
    # 5. Check expiration
    if record.expires_at and record.expires_at < datetime.utcnow():
        raise InvalidAPIKeyError("API key has expired")
    
    # 6. Update last_used_at (async, non-blocking)
    asyncio.create_task(update_last_used(record.id))
    
    return record
```

## 4. API Key Lifecycle Management

### 4.1 Key Rotation Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       Key Rotation Process                               │
│                                                                          │
│  1. Create new key (overlap period starts)                              │
│     └─► New key: 2api_acme_NEW123... (active)                           │
│     └─► Old key: 2api_acme_OLD456... (active, grace period)            │
│                                                                          │
│  2. Update applications to use new key                                  │
│     └─► Deploy with new key                                             │
│     └─► Both keys work during transition                                │
│                                                                          │
│  3. Monitor old key usage                                               │
│     └─► Dashboard shows last_used_at for old key                        │
│     └─► Verify no recent usage                                          │
│                                                                          │
│  4. Revoke old key                                                      │
│     └─► Set is_active = false                                           │
│     └─► Old key immediately stops working                               │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Key Rotation API

```yaml
# Create new key (for rotation)
POST /v1/api-keys
{
  "name": "Production Key v2",
  "permissions": ["*"],
  "rate_limit_rpm": 1000,
  "expires_at": "2025-12-31T23:59:59Z"  # Optional
}

Response:
{
  "id": "key_01HQ3K5V8Y...",
  "key": "2api_acme_x7k9m2p4q8r1s3t5u6v0w2y",  # Only shown once!
  "key_prefix": "2api_acme_x7k9",
  "created_at": "2024-01-15T10:00:00Z"
}

# List keys (to see which to rotate)
GET /v1/api-keys

Response:
{
  "data": [
    {
      "id": "key_01HQ3K5V8Y...",
      "key_prefix": "2api_acme_x7k9",
      "name": "Production Key v2",
      "last_used_at": "2024-01-15T10:30:00Z",
      "is_active": true
    },
    {
      "id": "key_01HQ2J4U7X...",
      "key_prefix": "2api_acme_m3n5",
      "name": "Production Key v1 (old)",
      "last_used_at": "2024-01-14T23:45:00Z",
      "is_active": true
    }
  ]
}

# Revoke old key
DELETE /v1/api-keys/{key_id}
# or
PATCH /v1/api-keys/{key_id}
{ "is_active": false }
```

### 4.3 Emergency Key Revocation

For compromised keys:

```python
async def emergency_revoke_key(key_id: str, reason: str):
    """Immediately revoke a key and log the event."""
    
    # 1. Deactivate in database
    await db.execute("""
        UPDATE api_keys 
        SET is_active = false, 
            revoked_at = NOW(),
            revoke_reason = $2
        WHERE id = $1
    """, key_id, reason)
    
    # 2. Invalidate in cache (immediate effect)
    await redis.delete(f"api_key:{key_id}")
    
    # 3. Add to revocation list (for distributed cache)
    await redis.sadd("revoked_keys", key_id)
    await redis.expire("revoked_keys", 86400)  # 24h TTL
    
    # 4. Audit log
    await log_audit_event(
        action="api_key_emergency_revoked",
        resource_id=key_id,
        reason=reason,
        severity="critical"
    )
    
    # 5. Alert tenant
    await send_alert(
        tenant_id=key.tenant_id,
        type="key_revoked",
        message=f"API key {key.key_prefix}... was revoked: {reason}"
    )
```

### 4.4 Key Validation with Revocation Check

```python
async def validate_api_key_with_revocation(key: str) -> APIKeyRecord:
    key_hash = hashlib.sha256(key.encode()).hexdigest()
    
    # Check revocation list first (fast path for compromised keys)
    key_id = await get_key_id_by_hash(key_hash)
    if key_id and await redis.sismember("revoked_keys", key_id):
        raise InvalidAPIKeyError("API key has been revoked")
    
    # Normal validation
    return await validate_api_key(key)
```

## 5. Per-Tenant Rate Limiting

### 5.1 Rate Limit Keys (Redis)

```
rl:{tenant_id}:rpm         # Requests per minute
rl:{tenant_id}:tpm         # Tokens per minute
rl:{tenant_id}:{key_id}:rpm  # Per API key RPM
rl:{tenant_id}:{key_id}:tpm  # Per API key TPM
```

### 5.2 Token Bucket Algorithm

2api.ai uses **Token Bucket** algorithm for rate limiting:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       Token Bucket Algorithm                             │
│                                                                          │
│  Bucket Capacity: 1000 tokens (burst limit)                             │
│  Refill Rate: 60 tokens/second (sustained rate)                         │
│                                                                          │
│  ┌─────────────────────────┐                                            │
│  │ ████████████████░░░░░░░ │  Current: 800 tokens                       │
│  │ ████████████████░░░░░░░ │  Capacity: 1000 tokens                     │
│  └─────────────────────────┘                                            │
│            ▲                                                             │
│            │ +60 tokens/second (refill)                                 │
│            │                                                             │
│  Request arrives:                                                        │
│  - Cost: 1 token (for RPM) or N tokens (for TPM)                        │
│  - If bucket has enough → Allow, subtract tokens                        │
│  - If bucket empty → Reject with 429, include retry_after               │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.3 Rate Limiting Implementation

```python
class TokenBucket:
    """Redis-based token bucket rate limiter."""
    
    def __init__(
        self,
        key: str,
        capacity: int,
        refill_rate: float,  # tokens per second
        redis: Redis
    ):
        self.key = key
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.redis = redis
    
    async def consume(self, tokens: int = 1) -> Tuple[bool, dict]:
        """
        Try to consume tokens from bucket.
        
        Returns:
            (allowed, info) where info contains:
            - remaining: tokens remaining
            - limit: bucket capacity
            - reset: seconds until full refill
            - retry_after: seconds to wait (if rejected)
        """
        now = time.time()
        
        # Lua script for atomic token bucket operation
        lua_script = """
        local key = KEYS[1]
        local capacity = tonumber(ARGV[1])
        local refill_rate = tonumber(ARGV[2])
        local requested = tonumber(ARGV[3])
        local now = tonumber(ARGV[4])
        
        -- Get current state
        local data = redis.call('HMGET', key, 'tokens', 'last_update')
        local tokens = tonumber(data[1]) or capacity
        local last_update = tonumber(data[2]) or now
        
        -- Calculate refill
        local elapsed = now - last_update
        local refill = elapsed * refill_rate
        tokens = math.min(capacity, tokens + refill)
        
        -- Try to consume
        local allowed = 0
        if tokens >= requested then
            tokens = tokens - requested
            allowed = 1
        end
        
        -- Update state
        redis.call('HMSET', key, 'tokens', tokens, 'last_update', now)
        redis.call('EXPIRE', key, 120)  -- 2 minute TTL
        
        -- Calculate retry_after if rejected
        local retry_after = 0
        if allowed == 0 then
            retry_after = math.ceil((requested - tokens) / refill_rate)
        end
        
        return {allowed, tokens, retry_after}
        """
        
        result = await self.redis.eval(
            lua_script,
            keys=[self.key],
            args=[self.capacity, self.refill_rate, tokens, now]
        )
        
        allowed = result[0] == 1
        remaining = int(result[1])
        retry_after = int(result[2])
        
        return allowed, {
            "remaining": remaining,
            "limit": self.capacity,
            "reset": int(self.capacity / self.refill_rate),
            "retry_after": retry_after if not allowed else None
        }


class TenantRateLimiter:
    """Rate limiter for a tenant with multiple limits."""
    
    def __init__(self, tenant_id: str, config: RateLimitConfig, redis: Redis):
        self.tenant_id = tenant_id
        
        # Requests per minute bucket
        self.rpm_bucket = TokenBucket(
            key=f"rl:{tenant_id}:rpm",
            capacity=config.rpm_burst,      # e.g., 100 (burst)
            refill_rate=config.rpm / 60,    # e.g., 60/60 = 1/sec
            redis=redis
        )
        
        # Tokens per minute bucket
        self.tpm_bucket = TokenBucket(
            key=f"rl:{tenant_id}:tpm",
            capacity=config.tpm_burst,      # e.g., 200000 (burst)
            refill_rate=config.tpm / 60,    # e.g., 100000/60 = 1666/sec
            redis=redis
        )
    
    async def check_and_consume(
        self,
        estimated_tokens: int
    ) -> Tuple[bool, dict]:
        """
        Check rate limits and consume if allowed.
        
        Returns:
            (allowed, headers) for rate limit response headers
        """
        # Check RPM first
        rpm_allowed, rpm_info = await self.rpm_bucket.consume(1)
        if not rpm_allowed:
            return False, {
                "X-RateLimit-Limit": str(self.rpm_bucket.capacity),
                "X-RateLimit-Remaining": str(rpm_info["remaining"]),
                "X-RateLimit-Reset": str(int(time.time()) + rpm_info["reset"]),
                "Retry-After": str(rpm_info["retry_after"]),
                "X-RateLimit-Type": "rpm"
            }
        
        # Check TPM
        tpm_allowed, tpm_info = await self.tpm_bucket.consume(estimated_tokens)
        if not tpm_allowed:
            # Refund the RPM token we consumed
            await self.rpm_bucket.consume(-1)
            
            return False, {
                "X-RateLimit-Limit": str(self.tpm_bucket.capacity),
                "X-RateLimit-Remaining": str(tpm_info["remaining"]),
                "X-RateLimit-Reset": str(int(time.time()) + tpm_info["reset"]),
                "Retry-After": str(tpm_info["retry_after"]),
                "X-RateLimit-Type": "tpm"
            }
        
        # Both passed
        return True, {
            "X-RateLimit-Limit": str(self.rpm_bucket.capacity),
            "X-RateLimit-Remaining": str(rpm_info["remaining"]),
            "X-RateLimit-Reset": str(int(time.time()) + rpm_info["reset"])
        }
```

### 5.4 Rate Limit Tiers

| Plan | RPM | RPM Burst | TPM | TPM Burst |
|------|-----|-----------|-----|-----------|
| Free | 20 | 30 | 40,000 | 60,000 |
| Starter | 60 | 100 | 100,000 | 150,000 |
| Pro | 300 | 500 | 500,000 | 750,000 |
| Enterprise | 1000+ | Custom | 2,000,000+ | Custom |

### 5.5 Rate Limit Response

```json
HTTP/1.1 429 Too Many Requests
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 0
X-RateLimit-Reset: 1700000060
X-RateLimit-Type: rpm
Retry-After: 45

{
  "error": {
    "code": "rate_limited",
    "message": "Rate limit exceeded. Please retry after 45 seconds.",
    "type": "infra_error",
    "request_id": "req_01HQ3K5V8Y...",
    "trace_id": "trace_01HQ3K5V8Y...",
    "retryable": true,
    "retry_after": 45,
    "details": {
      "limit_type": "rpm",
      "limit": 60,
      "remaining": 0,
      "reset_at": "2024-01-15T10:31:00Z"
    }
  }
}
```

## 5. Usage Tracking

### 5.1 Usage Record Schema

```sql
CREATE TABLE usage_records (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Tenant context (required)
    tenant_id UUID NOT NULL REFERENCES tenants(id),
    api_key_id UUID REFERENCES api_keys(id),
    
    -- Request identification
    request_id VARCHAR(64) NOT NULL UNIQUE,
    provider_request_id VARCHAR(64),
    
    -- What was called
    provider VARCHAR(50) NOT NULL,
    model VARCHAR(100) NOT NULL,
    operation VARCHAR(50) NOT NULL,  -- chat, embedding, image
    
    -- Token usage
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    total_tokens INTEGER NOT NULL DEFAULT 0,
    
    -- Cost (calculated at request time)
    cost_usd DECIMAL(12, 8) NOT NULL DEFAULT 0,
    
    -- Performance
    latency_ms INTEGER,
    time_to_first_token_ms INTEGER,  -- For streaming
    
    -- Status
    status VARCHAR(20) NOT NULL,  -- success, error, timeout, filtered
    error_code VARCHAR(50),
    
    -- Routing info
    routing_strategy VARCHAR(50),
    fallback_used BOOLEAN DEFAULT FALSE,
    original_model VARCHAR(100),  -- If fallback changed model
    
    -- Metadata
    metadata JSONB DEFAULT '{}',
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for common queries
CREATE INDEX idx_usage_tenant_date ON usage_records(tenant_id, created_at DESC);
CREATE INDEX idx_usage_tenant_model ON usage_records(tenant_id, model);
CREATE INDEX idx_usage_billing ON usage_records(tenant_id, created_at, cost_usd);
```

### 5.2 Usage Recording

```python
async def record_request(
    ctx: RequestContext,
    request: ChatCompletionRequest,
    response: ChatCompletionResponse,
    latency_ms: int
) -> None:
    # Calculate cost
    cost = calculate_cost(
        model=response.model,
        input_tokens=response.usage.prompt_tokens,
        output_tokens=response.usage.completion_tokens
    )
    
    # Insert usage record
    await db.execute("""
        INSERT INTO usage_records (
            tenant_id, api_key_id, request_id, provider_request_id,
            provider, model, operation,
            input_tokens, output_tokens, total_tokens, cost_usd,
            latency_ms, status, routing_strategy, fallback_used
        ) VALUES (
            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15
        )
    """, 
        ctx.tenant_id,
        ctx.api_key_id,
        ctx.request_id,
        response.id,
        response.provider,
        response.model,
        "chat",
        response.usage.prompt_tokens,
        response.usage.completion_tokens,
        response.usage.total_tokens,
        cost,
        latency_ms,
        "success",
        response._2api.routing_decision.strategy_used if response._2api else None,
        response._2api.routing_decision.fallback_used if response._2api else False
    )
    
    # Update rate limit counters
    await record_usage(ctx, response.usage.total_tokens)
```

## 6. Cost Calculation

### 6.1 Price Table (Versioned)

```sql
CREATE TABLE pricing (
    id UUID PRIMARY KEY,
    version INTEGER NOT NULL,           -- Price version
    effective_from DATE NOT NULL,       -- When this price takes effect
    effective_to DATE,                  -- When this price expires
    
    provider VARCHAR(50) NOT NULL,
    model VARCHAR(100) NOT NULL,
    
    -- Token pricing (per 1M tokens)
    input_price_per_1m DECIMAL(10, 4) NOT NULL,
    output_price_per_1m DECIMAL(10, 4) NOT NULL,
    
    -- Image pricing (per image)
    image_price_standard DECIMAL(10, 4),
    image_price_hd DECIMAL(10, 4),
    
    -- Embedding pricing (per 1M tokens)
    embedding_price_per_1m DECIMAL(10, 4),
    
    -- Markup
    markup_percent DECIMAL(5, 2) DEFAULT 0,  -- 2api.ai markup
    
    UNIQUE(version, provider, model)
);

-- Current prices view
CREATE VIEW current_pricing AS
SELECT * FROM pricing
WHERE effective_from <= CURRENT_DATE
  AND (effective_to IS NULL OR effective_to > CURRENT_DATE);
```

### 6.2 Cost Calculation Function

```python
# In-memory price cache (refreshed every 5 min)
PRICE_CACHE: Dict[str, ModelPricing] = {}

@dataclass
class ModelPricing:
    provider: str
    model: str
    input_per_1m: Decimal
    output_per_1m: Decimal
    markup_percent: Decimal
    effective_from: date

async def get_pricing(provider: str, model: str) -> ModelPricing:
    cache_key = f"{provider}/{model}"
    
    if cache_key in PRICE_CACHE:
        return PRICE_CACHE[cache_key]
    
    pricing = await db.fetch_one("""
        SELECT * FROM current_pricing
        WHERE provider = $1 AND model = $2
    """, provider, model)
    
    if not pricing:
        # Fallback to provider-level default
        pricing = await db.fetch_one("""
            SELECT * FROM current_pricing
            WHERE provider = $1 AND model = 'default'
        """, provider)
    
    if not pricing:
        raise PricingNotFoundError(f"No pricing for {provider}/{model}")
    
    model_pricing = ModelPricing(**pricing)
    PRICE_CACHE[cache_key] = model_pricing
    return model_pricing

def calculate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int
) -> Decimal:
    provider, model_name = model.split("/", 1)
    pricing = get_pricing(provider, model_name)
    
    input_cost = (Decimal(input_tokens) / 1_000_000) * pricing.input_per_1m
    output_cost = (Decimal(output_tokens) / 1_000_000) * pricing.output_per_1m
    
    subtotal = input_cost + output_cost
    markup = subtotal * (pricing.markup_percent / 100)
    
    return (subtotal + markup).quantize(Decimal("0.00000001"))
```

## 7. Billing

### 7.1 Billing Aggregation (Daily Job)

```python
async def aggregate_daily_billing(date: date) -> None:
    """Run daily to aggregate usage into billing records."""
    
    # Get all tenants with usage on this date
    tenants = await db.fetch_all("""
        SELECT DISTINCT tenant_id FROM usage_records
        WHERE DATE(created_at) = $1
    """, date)
    
    for tenant in tenants:
        # Aggregate by provider and model
        usage = await db.fetch_all("""
            SELECT 
                provider,
                model,
                SUM(input_tokens) as input_tokens,
                SUM(output_tokens) as output_tokens,
                SUM(total_tokens) as total_tokens,
                SUM(cost_usd) as cost,
                COUNT(*) as requests
            FROM usage_records
            WHERE tenant_id = $1 AND DATE(created_at) = $2
            GROUP BY provider, model
        """, tenant.tenant_id, date)
        
        # Upsert billing record
        await db.execute("""
            INSERT INTO billing_records (
                tenant_id, period_start, period_end,
                total_requests, total_tokens, 
                total_input_tokens, total_output_tokens,
                total_cost_usd, cost_by_provider, cost_by_model
            ) VALUES ($1, $2, $2, $3, $4, $5, $6, $7, $8, $9)
            ON CONFLICT (tenant_id, period_start, period_end) 
            DO UPDATE SET
                total_requests = EXCLUDED.total_requests,
                total_tokens = EXCLUDED.total_tokens,
                total_cost_usd = EXCLUDED.total_cost_usd,
                cost_by_provider = EXCLUDED.cost_by_provider,
                cost_by_model = EXCLUDED.cost_by_model,
                updated_at = NOW()
        """, tenant.tenant_id, date, ...)
```

### 7.2 Budget Alerts

```python
async def check_budget_alerts(ctx: RequestContext, cost: Decimal) -> None:
    tenant = await get_tenant(ctx.tenant_id)
    
    if not tenant.budget_limit_usd:
        return
    
    # Get current month spending
    current_spending = await db.fetch_val("""
        SELECT COALESCE(SUM(total_cost_usd), 0)
        FROM billing_records
        WHERE tenant_id = $1
          AND period_start >= DATE_TRUNC('month', CURRENT_DATE)
    """, ctx.tenant_id)
    
    projected = current_spending + cost
    
    # Check thresholds
    if projected > tenant.budget_limit_usd:
        raise BudgetExceededError(
            f"Request would exceed monthly budget of ${tenant.budget_limit_usd}"
        )
    
    # Send alerts at 50%, 80%, 90%
    for threshold in [0.5, 0.8, 0.9]:
        if current_spending < tenant.budget_limit_usd * threshold <= projected:
            await send_budget_alert(tenant, threshold, projected)
```

## 8. Dashboard API

### 8.1 Usage Endpoints

```yaml
# GET /v1/usage?start_date=2024-01-01&end_date=2024-01-31&group_by=day
Response:
  {
    "data": [
      {
        "date": "2024-01-01",
        "requests": 1523,
        "tokens": 2500000,
        "cost_usd": 45.23,
        "by_provider": {
          "openai": {"requests": 800, "tokens": 1500000, "cost": 30.00},
          "anthropic": {"requests": 723, "tokens": 1000000, "cost": 15.23}
        }
      }
    ],
    "summary": {
      "total_requests": 45690,
      "total_tokens": 75000000,
      "total_cost_usd": 1357.89
    }
  }

# GET /v1/usage/realtime
Response:
  {
    "current_rpm": 45,
    "current_tpm": 85000,
    "limit_rpm": 1000,
    "limit_tpm": 1000000,
    "today": {
      "requests": 5234,
      "tokens": 8500000,
      "cost_usd": 255.67
    }
  }
```

## 9. Security Considerations

### 9.1 Data Access Rules

```python
# ALL database queries MUST include tenant_id filter
async def get_usage(tenant_id: str, filters: dict) -> List[UsageRecord]:
    return await db.fetch_all("""
        SELECT * FROM usage_records
        WHERE tenant_id = $1  -- ALWAYS FIRST CONDITION
          AND created_at BETWEEN $2 AND $3
    """, tenant_id, filters.start, filters.end)

# NEVER expose other tenants' data
# NEVER allow tenant_id to come from user input
# ALWAYS derive tenant_id from authenticated API key
```

### 9.2 Audit Logging

```python
# All sensitive operations are logged
await log_audit_event(
    tenant_id=ctx.tenant_id,
    action="api_key_created",
    actor_id=ctx.user_id,
    resource_type="api_key",
    resource_id=new_key.id,
    details={"name": new_key.name, "permissions": new_key.permissions}
)
```

## 10. Summary

| Component | Isolation Method |
|-----------|------------------|
| Data | `tenant_id` foreign key |
| API Keys | Hashed, tenant-scoped |
| Rate Limits | Redis keys with tenant prefix |
| Usage | Per-request, tenant-tagged |
| Billing | Daily aggregation per tenant |
| Logs | Structured with tenant_id |
| Alerts | Per-tenant thresholds |
