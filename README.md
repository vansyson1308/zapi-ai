# 2api.ai - Unified AI Infrastructure Platform

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![License](https://img.shields.io/badge/license-Proprietary-red)
![Python](https://img.shields.io/badge/python-3.10+-green)
![Tests](https://img.shields.io/badge/tests-420%20passed-brightgreen)

**2api.ai** is an AI infrastructure platform that provides a unified API to access multiple AI providers (OpenAI, Anthropic Claude, Google Gemini) through a single interface with intelligent routing, automatic fallback, cost optimization, and full observability.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Your Application                             │
│                               │                                      │
│                               ▼                                      │
│         ┌───────────────────────────────────────────┐               │
│         │              2api.ai Gateway               │               │
│         │  ┌─────────┬─────────┬─────────────────┐  │               │
│         │  │ Routing │ Fallback│  Observability  │  │               │
│         │  │ Engine  │ Handler │ (Metrics/Traces)│  │               │
│         │  └─────────┴─────────┴─────────────────┘  │               │
│         └───────────────────┬───────────────────────┘               │
│                             │                                        │
│             ┌───────────────┼───────────────┐                       │
│             ▼               ▼               ▼                       │
│        ┌─────────┐    ┌──────────┐    ┌─────────┐                  │
│        │ OpenAI  │    │ Anthropic│    │ Google  │                  │
│        │ GPT-4o  │    │  Claude  │    │ Gemini  │                  │
│        └─────────┘    └──────────┘    └─────────┘                  │
└─────────────────────────────────────────────────────────────────────┘
```

## Features

### Core Platform
- **Unified API**: Single OpenAI-compatible interface for all AI providers
- **Smart Routing**: Automatic model selection based on cost, latency, or quality
- **Automatic Fallback**: Seamless failover with circuit breaker pattern
- **Multi-tenant**: Complete data isolation between customers
- **Token-based Billing**: Pay only for what you use
- **Streaming Support**: Real-time response streaming with SSE
- **Tool Calling**: Function calling normalized across all providers
- **Vision Support**: Image understanding for supported models

### Observability (EPIC H)
- **Prometheus Metrics**: Request counts, latencies, token usage, costs at `/metrics`
- **OpenTelemetry Tracing**: Distributed tracing with W3C traceparent propagation
- **Structured Logging**: JSON-formatted logs with correlation IDs and sensitive data redaction
- **Health Endpoints**: `/health`, `/ready` for orchestration

### SDKs (EPIC I)
- **Python SDK**: Simple and OpenAI-compatible API surfaces with async support
- **JavaScript/TypeScript SDK**: Full TypeScript support with streaming iterators
- **Tool Calling Helpers**: Automatic tool execution loop with `ToolRunner`
- **Built-in Retries**: Exponential backoff with jitter for infrastructure errors
- **Error Taxonomy**: Distinguish retryable (infra) vs non-retryable (semantic) errors

---

## Quick Start

### Using Python SDK

```python
from twoapi import TwoAPI

client = TwoAPI(api_key="2api_your_key")

# Simple chat - one liner
response = client.chat("Hello!")
print(response.content)

# OpenAI-compatible API
response = client.chat.completions.create(
    model="openai/gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)

# Streaming
for chunk in client.chat_stream("Tell me a story"):
    print(chunk, end="", flush=True)

# With routing strategy
from twoapi import RoutingConfig
response = client.chat(
    "Simple question",
    routing=RoutingConfig(strategy="cost")
)
```

### Tool Calling with Python

```python
from twoapi import TwoAPI, ToolRunner, tool

@tool(description="Get weather for a location")
def get_weather(location: str) -> str:
    return f"Weather in {location}: sunny, 22C"

client = TwoAPI()
runner = ToolRunner([get_weather], max_iterations=5)

# Automatic tool execution loop
result = runner.run(client, "What's the weather in Tokyo?")
print(result.response.content)    # Natural language response
print(result.tool_calls_made)     # List of executed tool calls
print(result.total_iterations)    # Number of LLM calls
```

### Using JavaScript/TypeScript SDK

```typescript
import { TwoAPI, ToolRunner, tool } from 'twoapi';

const client = new TwoAPI({ apiKey: '2api_your_key' });

// Simple chat
const response = await client.chat('Hello!');
console.log(response.content);

// OpenAI-compatible API
const completion = await client.chat.completions.create({
  model: 'openai/gpt-4o',
  messages: [{ role: 'user', content: 'Hello!' }]
});
console.log(completion.choices[0].message.content);

// Streaming
for await (const chunk of client.chatStream('Tell me a story')) {
  process.stdout.write(chunk.content);
}

// Tool calling with automatic execution
const weatherTool = tool({
  name: 'get_weather',
  description: 'Get weather for a location',
  parameters: {
    type: 'object',
    properties: { location: { type: 'string' } }
  },
  execute: async (args) => `Weather in ${args.location}: sunny`
});

const runner = new ToolRunner([weatherTool]);
const result = await runner.run(client, 'What is the weather in Tokyo?');
console.log(result.response.content);
```

### Using REST API

```bash
curl https://api.2api.ai/v1/chat/completions \
  -H "Authorization: Bearer 2api_your_key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "messages": [{"role": "user", "content": "Hello!"}],
    "routing": {"strategy": "cost"}
  }'
```

---

## Running Locally with Docker

### 1. Create environment file

```bash
cp .env.example .env
```

Edit `.env` with at least one provider key:
```bash
OPENAI_API_KEY=sk-...
# or
ANTHROPIC_API_KEY=sk-ant-...
# or
GOOGLE_API_KEY=AI...
```

### 2. Build and Run

```bash
docker build -t 2api-ai .
docker run --rm -p 8080:8080 --env-file .env 2api-ai
```

### 3. Verify

```bash
# Health check
curl http://localhost:8080/health

# Prometheus metrics
curl http://localhost:8080/metrics

# List models
curl http://localhost:8080/v1/models \
  -H "Authorization: Bearer 2api_test"

# Chat completion
curl http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer 2api_test" \
  -H "Content-Type: application/json" \
  -d '{"model":"openai/gpt-4o-mini","messages":[{"role":"user","content":"Hello!"}]}'
```

---

## Project Structure

```
2api-ai/
├── src/
│   ├── server.py                 # FastAPI application entry
│   ├── core/
│   │   ├── models.py             # Unified data models
│   │   ├── errors.py             # Error definitions
│   │   └── http_client.py        # HTTP client utilities
│   ├── adapters/
│   │   ├── base.py               # Base adapter interface
│   │   ├── openai_adapter.py     # OpenAI provider adapter
│   │   ├── anthropic_adapter.py  # Anthropic provider adapter
│   │   └── google_adapter.py     # Google provider adapter
│   ├── routing/
│   │   ├── router.py             # Main routing service
│   │   ├── strategies.py         # Cost/latency/quality strategies
│   │   ├── fallback.py           # Fallback handling
│   │   ├── circuit_breaker.py    # Circuit breaker pattern
│   │   └── health.py             # Provider health checks
│   ├── streaming/
│   │   ├── normalizer.py         # SSE stream normalization
│   │   ├── tool_calls.py         # Streaming tool call handling
│   │   └── errors.py             # Stream error handling
│   ├── tools/
│   │   ├── normalizer.py         # Tool format normalization
│   │   ├── validator.py          # Tool schema validation
│   │   └── parallel.py           # Parallel tool execution
│   ├── observability/            # EPIC H - Observability Stack
│   │   ├── metrics.py            # Prometheus metrics collector
│   │   ├── tracing.py            # OpenTelemetry distributed tracing
│   │   ├── logging.py            # Structured JSON logging
│   │   └── middleware.py         # Observability middleware
│   ├── usage/
│   │   ├── tracker.py            # Usage tracking
│   │   ├── pricing.py            # Pricing calculations
│   │   └── limits.py             # Rate limiting
│   ├── api/
│   │   ├── routes/               # API route handlers
│   │   ├── middleware.py         # Request/response middleware
│   │   └── dependencies.py       # FastAPI dependencies
│   ├── auth/
│   │   ├── middleware.py         # Authentication middleware
│   │   └── config.py             # Auth configuration
│   ├── db/
│   │   ├── models.py             # Database models
│   │   └── services.py           # Database services
│   └── sdk/                      # EPIC I - SDKs
│       ├── python/
│       │   └── twoapi/
│       │       ├── client.py         # Main sync client
│       │       ├── async_client.py   # Async client
│       │       ├── openai_compat.py  # OpenAI-compatible API
│       │       ├── tools.py          # Tool calling helpers (@tool, ToolRunner)
│       │       ├── errors.py         # Error classes with taxonomy
│       │       ├── retry.py          # Retry logic with backoff
│       │       └── models.py         # Data models
│       └── javascript/
│           └── src/
│               ├── client.ts         # Main client
│               ├── openai-compat.ts  # OpenAI-compatible types
│               ├── tools.ts          # Tool calling helpers
│               ├── errors.ts         # Error classes
│               ├── retry.ts          # Retry logic
│               └── types.ts          # TypeScript types
├── tests/                        # 420 tests
│   ├── test_observability.py     # Observability tests (21)
│   ├── test_sdk_system.py        # SDK tests (55)
│   └── ...                       # Other test files
├── docs/
│   ├── ARCHITECTURE.md           # System architecture
│   ├── ERROR_TAXONOMY.md         # Error classification
│   ├── STREAMING_SPEC.md         # Streaming protocol
│   ├── TOOL_CALLING_SPEC.md      # Tool calling spec
│   ├── RETRY_FALLBACK_POLICY.md  # Retry policies
│   └── MULTI_TENANT_DESIGN.md    # Multi-tenant design
├── Dockerfile
├── docker-compose.yaml
├── requirements.txt
└── README.md
```

---

## API Reference

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/v1/chat/completions` | Create chat completion |
| POST | `/v1/embeddings` | Create embeddings |
| POST | `/v1/images/generations` | Generate images |
| GET | `/v1/models` | List available models |
| GET | `/v1/usage` | Get usage statistics |
| GET | `/health` | Health check (no auth) |
| GET | `/ready` | Readiness probe (no auth) |
| GET | `/metrics` | Prometheus metrics (no auth) |

### Model Naming

Models are specified as `provider/model-name`:

```
openai/gpt-4o
openai/gpt-4o-mini
anthropic/claude-3-5-sonnet
anthropic/claude-3-opus
google/gemini-1.5-pro
google/gemini-1.5-flash
auto  # Let 2api.ai choose based on routing strategy
```

### Routing Strategies

| Strategy | Description |
|----------|-------------|
| `cost` | Choose the cheapest model for the request |
| `latency` | Choose the fastest responding model |
| `quality` | Choose the most capable model |

---

## SDK Features

### Error Handling

Both SDKs provide taxonomy-aware error classes:

```python
# Python
from twoapi import (
    TwoAPIError,           # Base error
    AuthenticationError,   # 401 - Invalid/missing API key
    RateLimitError,        # 429 - Rate limit exceeded (retryable)
    InvalidRequestError,   # 400 - Bad request parameters
    ProviderError,         # 5xx - Provider-side errors (retryable)
    TimeoutError,          # Request timeout (retryable)
    ConnectionError,       # Network issues (retryable)
    is_retryable_error,    # Check if error can be retried
)

try:
    response = client.chat("Hello")
except RateLimitError as e:
    print(f"Rate limited. Retry after {e.retry_after} seconds")
except AuthenticationError:
    print("Check your API key")
except ProviderError as e:
    print(f"Provider {e.provider} is having issues")
```

```typescript
// JavaScript/TypeScript
import {
  TwoAPIError,
  AuthenticationError,
  RateLimitError,
  InvalidRequestError,
  ProviderError,
  TimeoutError,
  ConnectionError,
  isRetryableError
} from 'twoapi';
```

### Retry Configuration

```python
# Python - Custom retry configuration
from twoapi import TwoAPI

client = TwoAPI(
    api_key="2api_xxx",
    max_retries=5,
    on_retry=lambda attempt, error, delay: print(f"Retry {attempt}: {error}")
)

# Or with full control
from twoapi import TwoAPI, RetryConfig

client = TwoAPI(
    retry_config=RetryConfig(
        max_retries=5,
        initial_delay=1.0,
        max_delay=30.0,
        exponential_base=2.0,
        retry_on_status=[429, 500, 502, 503, 504]
    )
)
```

---

## Observability

### Prometheus Metrics

Available at `/metrics`:

| Metric | Type | Description |
|--------|------|-------------|
| `twoapi_requests_total` | Counter | Total requests by provider, model, status |
| `twoapi_request_duration_seconds` | Histogram | Request latency |
| `twoapi_time_to_first_token_seconds` | Histogram | Streaming TTFT |
| `twoapi_tokens_total` | Counter | Token usage (input/output) |
| `twoapi_cost_usd_total` | Counter | Cost in USD |
| `twoapi_active_requests` | Gauge | Currently processing requests |
| `twoapi_circuit_breaker_state` | Gauge | Circuit breaker state (0=closed, 1=open, 2=half-open) |
| `twoapi_rate_limit_hits_total` | Counter | Rate limit hits |

### OpenTelemetry Tracing

Configure via environment variables:

```bash
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
OTEL_CONSOLE_EXPORT=false
```

All requests include:
- `request_id`: Unique request identifier (X-Request-ID header)
- `trace_id`: Distributed trace ID
- W3C `traceparent` header propagation

### Structured Logging

JSON-formatted logs with automatic context injection:

```json
{
  "timestamp": "2024-01-15T10:30:00.000Z",
  "level": "INFO",
  "logger": "2api",
  "message": "Request completed",
  "request_id": "req_abc123",
  "trace_id": "trace_xyz789",
  "provider": "openai",
  "model": "gpt-4o",
  "duration_ms": 1234,
  "tokens": {"input": 100, "output": 50}
}
```

Sensitive fields (api_key, password, authorization) are automatically redacted.

---

## Environment Variables

### Required (at least one provider)

```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AI...
```

### Optional

```bash
# Server
PORT=8000
MODE=local  # local or prod

# Database & Cache
DATABASE_URL=postgresql://user:pass@localhost:5432/twoapi
REDIS_URL=redis://localhost:6379

# Logging
LOG_LEVEL=INFO        # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FORMAT=json       # json or text

# Observability
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
OTEL_CONSOLE_EXPORT=false

# Usage Tracking
USAGE_BUFFER_SIZE=100
USAGE_FLUSH_INTERVAL=5.0
```

---

## Development

### Prerequisites

- Python 3.10+
- Node.js 18+ (for JavaScript SDK)
- Docker & Docker Compose

### Setup

```bash
git clone https://github.com/2api-ai/2api
cd 2api

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v
```

### Running Tests

```bash
# All tests (420 tests)
pytest tests/ -v

# Specific test file
pytest tests/test_observability.py -v
pytest tests/test_sdk_system.py -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [Architecture](docs/ARCHITECTURE.md) | System design & components |
| [Error Taxonomy](docs/ERROR_TAXONOMY.md) | Error codes & classification |
| [Streaming Spec](docs/STREAMING_SPEC.md) | SSE protocol, chunk format |
| [Tool Calling Spec](docs/TOOL_CALLING_SPEC.md) | Function calling normalization |
| [Retry & Fallback](docs/RETRY_FALLBACK_POLICY.md) | Circuit breaker, retry policies |
| [Multi-Tenant Design](docs/MULTI_TENANT_DESIGN.md) | Tenant isolation, billing |

---

## Security Notes

- **Never commit secrets**: `.env`, provider keys, or credentials
- Keep `.env` in `.gitignore`
- Use `--env-file .env` instead of `-e KEY=...` to avoid shell history leaks
- Use a secrets manager in production
- If a key is exposed, **rotate it immediately**
- Sensitive data is automatically redacted in logs

---

## License

Proprietary - All rights reserved.
