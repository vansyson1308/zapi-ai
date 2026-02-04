# 2api.ai - Architecture Document

## 1. Tổng Quan Hệ Thống

### 1.1 Mục Đích
2api.ai là một **AI Infrastructure Platform** cung cấp lớp abstraction thống nhất để gọi nhiều AI providers (OpenAI, Anthropic Claude, Google Gemini) thông qua một API/SDK duy nhất.

### 1.2 Yêu Cầu Chính
- **Scale**: 100,000+ requests/ngày
- **Multi-tenant**: Data isolation giữa các customers
- **Providers**: OpenAI, Claude, Gemini
- **Features**: Chat, Streaming, Tool Calling, Vision, Embeddings, Image Generation
- **Routing**: Fallback, Load Balancing, Cost-based, Latency-based
- **Billing**: Token-based pricing
- **Infrastructure**: Google Cloud

---

## 2. Kiến Trúc Tổng Thể

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CLIENTS                                         │
│                    (Python SDK / JavaScript SDK / REST API)                  │
└─────────────────────────────────────────────┬───────────────────────────────┘
                                              │
                                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           API GATEWAY (Cloud Run)                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │    Auth     │  │ Rate Limit  │  │  Routing    │  │  Request Validation │ │
│  │  (API Key)  │  │  (Redis)    │  │             │  │                     │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────┬───────────────────────────────┘
                                              │
                                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CORE SERVICES (Cloud Run)                            │
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                        ROUTER SERVICE                                  │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌──────────────┐  │  │
│  │  │  Fallback   │  │    Load     │  │    Cost     │  │   Latency    │  │  │
│  │  │   Router    │  │  Balancer   │  │   Router    │  │    Router    │  │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └──────────────┘  │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                     ABSTRACTION LAYER                                  │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │  │
│  │  │                    Unified AI Interface                          │  │  │
│  │  │  • Normalize Request  • Normalize Response  • Error Handling    │  │  │
│  │  └─────────────────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                      PROVIDER ADAPTERS                                 │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                    │  │
│  │  │   OpenAI    │  │  Anthropic  │  │   Google    │                    │  │
│  │  │   Adapter   │  │   Adapter   │  │   Adapter   │                    │  │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘                    │  │
│  └─────────┼────────────────┼────────────────┼───────────────────────────┘  │
└────────────┼────────────────┼────────────────┼──────────────────────────────┘
             │                │                │
             ▼                ▼                ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│   OpenAI API    │ │  Anthropic API  │ │   Gemini API    │
│   (External)    │ │   (External)    │ │   (External)    │
└─────────────────┘ └─────────────────┘ └─────────────────┘

                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SUPPORT SERVICES                                     │
│                                                                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐  │
│  │  Usage Tracker  │  │  Tenant Manager │  │     Observability           │  │
│  │  (Token Count)  │  │  (Isolation)    │  │  (Logging, Metrics, Traces) │  │
│  └────────┬────────┘  └────────┬────────┘  └──────────────┬──────────────┘  │
└───────────┼────────────────────┼─────────────────────────┼──────────────────┘
            │                    │                         │
            ▼                    ▼                         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA LAYER                                         │
│                                                                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐  │
│  │   Cloud SQL     │  │    Redis        │  │     Cloud Storage           │  │
│  │  (PostgreSQL)   │  │  (Memorystore)  │  │     (Logs, Files)           │  │
│  │                 │  │                 │  │                             │  │
│  │  • Users        │  │  • Rate Limits  │  │  • Request Logs             │  │
│  │  • API Keys     │  │  • Cache        │  │  • Analytics Data           │  │
│  │  • Usage Data   │  │  • Sessions     │  │                             │  │
│  │  • Tenants      │  │                 │  │                             │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Components Chi Tiết

### 3.1 API Gateway
**Công nghệ**: Cloud Run + Cloud Endpoints

**Chức năng**:
- Authentication (API Key validation)
- Rate Limiting (per tenant, per endpoint)
- Request validation
- Request routing

**Endpoints chính**:
```
POST /v1/chat/completions      # Chat với AI
POST /v1/embeddings            # Tạo embeddings
POST /v1/images/generations    # Tạo hình ảnh
GET  /v1/models                # List available models
GET  /v1/usage                 # Xem usage statistics
```

### 3.2 Router Service
**Logic routing theo thứ tự ưu tiên**:

1. **Explicit Model**: Nếu user chỉ định model cụ thể → dùng model đó
2. **Policy-based**: Áp dụng routing policy của tenant
3. **Fallback**: Nếu primary provider fail → chuyển sang backup

**Routing Policies**:
```python
{
    "strategy": "cost_optimized",  # hoặc "latency_optimized", "quality_optimized"
    "fallback_chain": ["openai", "anthropic", "google"],
    "constraints": {
        "max_latency_ms": 5000,
        "max_cost_per_request": 0.10
    }
}
```

### 3.3 Provider Adapters
Mỗi adapter chịu trách nhiệm:
- Convert request từ unified format → provider-specific format
- Convert response từ provider format → unified format
- Handle provider-specific errors
- Implement streaming

### 3.4 Usage Tracker
- Count tokens (input + output) cho mỗi request
- Aggregate usage per tenant
- Support billing reports
- Real-time usage alerts

### 3.5 Tenant Manager
- Tenant isolation (data, API keys, usage)
- Tenant-specific configurations
- Tenant-specific rate limits

---

## 4. Data Models

### 4.1 Core Entities

```sql
-- Tenants (Customers)
CREATE TABLE tenants (
    id UUID PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    plan VARCHAR(50) DEFAULT 'free',
    settings JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- API Keys
CREATE TABLE api_keys (
    id UUID PRIMARY KEY,
    tenant_id UUID REFERENCES tenants(id),
    key_hash VARCHAR(64) NOT NULL,  -- SHA-256 hash
    key_prefix VARCHAR(8) NOT NULL,  -- First 8 chars for identification
    name VARCHAR(255),
    permissions JSONB DEFAULT '["*"]',
    rate_limit INTEGER DEFAULT 1000,  -- requests per minute
    is_active BOOLEAN DEFAULT TRUE,
    last_used_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Usage Records
CREATE TABLE usage_records (
    id UUID PRIMARY KEY,
    tenant_id UUID REFERENCES tenants(id),
    api_key_id UUID REFERENCES api_keys(id),
    provider VARCHAR(50) NOT NULL,
    model VARCHAR(100) NOT NULL,
    operation VARCHAR(50) NOT NULL,  -- chat, embedding, image
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    total_tokens INTEGER DEFAULT 0,
    cost_usd DECIMAL(10, 6) DEFAULT 0,
    latency_ms INTEGER,
    status VARCHAR(20),  -- success, error, timeout
    error_code VARCHAR(50),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Provider Health (for routing decisions)
CREATE TABLE provider_health (
    provider VARCHAR(50) PRIMARY KEY,
    is_healthy BOOLEAN DEFAULT TRUE,
    avg_latency_ms INTEGER,
    error_rate DECIMAL(5, 4),
    last_check_at TIMESTAMP,
    updated_at TIMESTAMP DEFAULT NOW()
);
```

### 4.2 Unified Request/Response Schema

```typescript
// Unified Chat Request
interface ChatRequest {
    model: string;                    // "openai/gpt-4" hoặc "auto"
    messages: Message[];
    temperature?: number;             // 0-2, default 1
    max_tokens?: number;
    stream?: boolean;                 // default false
    tools?: Tool[];                   // for function calling
    tool_choice?: "auto" | "none" | ToolChoice;
    
    // 2api.ai specific
    routing?: {
        strategy?: "cost" | "latency" | "quality";
        fallback?: string[];
        max_latency_ms?: number;
        max_cost?: number;
    };
    metadata?: Record<string, string>;
}

interface Message {
    role: "system" | "user" | "assistant" | "tool";
    content: string | ContentPart[];
    name?: string;
    tool_call_id?: string;
    tool_calls?: ToolCall[];
}

interface ContentPart {
    type: "text" | "image_url";
    text?: string;
    image_url?: {
        url: string;        // URL hoặc base64
        detail?: "low" | "high" | "auto";
    };
}

// Unified Chat Response
interface ChatResponse {
    id: string;
    object: "chat.completion";
    created: number;
    model: string;                    // Actual model used
    provider: string;                 // "openai" | "anthropic" | "google"
    
    choices: [{
        index: number;
        message: {
            role: "assistant";
            content: string | null;
            tool_calls?: ToolCall[];
        };
        finish_reason: "stop" | "length" | "tool_calls" | "error";
    }];
    
    usage: {
        prompt_tokens: number;
        completion_tokens: number;
        total_tokens: number;
    };
    
    // 2api.ai specific
    _2api: {
        request_id: string;
        latency_ms: number;
        cost_usd: number;
        routing_decision: {
            strategy_used: string;
            candidates_evaluated: string[];
            fallback_used: boolean;
        };
    };
}
```

---

## 5. Tech Stack

| Component | Technology | Rationale |
|-----------|------------|-----------|
| **API Gateway** | Cloud Run | Auto-scaling, pay-per-use |
| **Core Services** | Python (FastAPI) | AI ecosystem, async support |
| **Database** | Cloud SQL (PostgreSQL) | ACID, JSON support, reliability |
| **Cache** | Memorystore (Redis) | Low latency, rate limiting |
| **Queue** | Cloud Pub/Sub | Async processing, reliability |
| **Storage** | Cloud Storage | Logs, large files |
| **Monitoring** | Cloud Monitoring + Logging | Native integration |
| **SDK** | Python + TypeScript | Most common AI dev languages |

---

## 6. Security

### 6.1 Authentication
- API Key based authentication
- Keys are hashed (SHA-256) before storage
- Key prefix stored for identification
- Support for key rotation

### 6.2 Tenant Isolation
- All queries filtered by tenant_id
- Separate rate limits per tenant
- No cross-tenant data access

### 6.3 Provider Keys
- Provider API keys stored encrypted
- Support for customer's own keys (BYOK)
- Key rotation without downtime

---

## 7. Scalability Considerations

### 7.1 Horizontal Scaling
- All services stateless → easy to scale
- Cloud Run auto-scaling based on requests
- Redis for shared state (rate limits, cache)

### 7.2 Performance Targets
| Metric | Target |
|--------|--------|
| P50 Latency (overhead) | < 50ms |
| P99 Latency (overhead) | < 200ms |
| Availability | 99.9% |
| Max concurrent requests | 10,000 |

### 7.3 Caching Strategy
- Model metadata: Cache indefinitely
- Provider health: Cache 30 seconds
- Rate limit counters: Redis with TTL

---

## 8. Deployment

### 8.1 Environments
- **Development**: Local Docker Compose
- **Staging**: GCP Project (staging)
- **Production**: GCP Project (prod)

### 8.2 CI/CD Pipeline
```
Push to main → Build → Test → Deploy to Staging → Manual approval → Deploy to Prod
```

### 8.3 Infrastructure as Code
- Terraform for GCP resources
- Docker for application containers
- Cloud Build for CI/CD

---

## 9. Monitoring & Observability

### 9.1 Metrics
- Request count (by provider, model, tenant)
- Latency distribution
- Error rate
- Token usage
- Cost per tenant

### 9.2 Logging
- Structured JSON logs
- Request/Response logging (with PII masking)
- Error tracking

### 9.3 Alerting
- Provider outage
- High error rate
- Unusual usage patterns
- Cost threshold exceeded

---

## 10. Roadmap Milestones

### Phase 1 (0-3 months) - MVP
- [x] Core abstraction layer
- [x] 3 provider adapters (OpenAI, Claude, Gemini)
- [x] Basic routing (explicit model + fallback)
- [x] API Key authentication
- [x] Usage tracking
- [x] Python + JS SDK

### Phase 2 (3-6 months) - Production Ready
- [ ] Advanced routing (cost, latency optimization)
- [ ] Dashboard UI
- [ ] Billing integration
- [ ] Multi-region deployment
- [ ] SOC 2 compliance

### Phase 3 (6-12 months) - Scale
- [ ] More providers (Mistral, Cohere, etc.)
- [ ] Custom model support
- [ ] Advanced analytics
- [ ] Enterprise features (SSO, audit logs)
