# 2api.ai - Unified AI Infrastructure Platform

![Version](https://img.shields.io/badge/version-0.1.0-blue)
![License](https://img.shields.io/badge/license-Proprietary-red)
![Status](https://img.shields.io/badge/status-Phase%201%20Alpha-yellow)

**2api.ai** is an AI infrastructure platform that provides a unified API to access multiple AI providers (OpenAI, Anthropic Claude, Google Gemini) through a single interface with intelligent routing, automatic fallback, and cost optimization.

## 🎯 Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      Your Application                        │
│                            │                                 │
│                            ▼                                 │
│                    ┌──────────────┐                         │
│                    │   2api.ai    │                         │
│                    │  Unified API │                         │
│                    └──────┬───────┘                         │
│                           │                                  │
│           ┌───────────────┼───────────────┐                 │
│           ▼               ▼               ▼                 │
│      ┌─────────┐    ┌──────────┐    ┌─────────┐            │
│      │ OpenAI  │    │ Anthropic│    │ Google  │            │
│      │ GPT-4o  │    │  Claude  │    │ Gemini  │            │
│      └─────────┘    └──────────┘    └─────────┘            │
└─────────────────────────────────────────────────────────────┘
```

## ✨ Features

- **Unified API**: Single interface for all AI providers
- **Smart Routing**: Automatic model selection based on cost, latency, or quality
- **Automatic Fallback**: Seamless failover when providers are down
- **Multi-tenant**: Complete data isolation between customers
- **Token-based Billing**: Pay only for what you use
- **Streaming Support**: Real-time response streaming
- **Tool Calling**: Function calling across all providers
- **Vision Support**: Image understanding for supported models

---

## 🚀 Quick Start (Run the API server locally with Docker)

### 1) Create environment file (recommended)
Create a `.env` file at repo root with **at least one** provider key:

**Linux/macOS**
```bash
cp .env.example .env
```

**Windows CMD**
```bat
copy .env.example .env
```

Edit `.env`:
```bash
OPENAI_API_KEY=sk-...
# or
ANTHROPIC_API_KEY=sk-ant-...
# or
GOOGLE_API_KEY=AI...
```

> Note: If **no** provider keys are set, the server can still start, but `/v1/models` will return an empty list and routing to providers is effectively disabled.

### 2) Build
```bash
docker build -t 2api-ai .
```

### 3) Run
```bash
docker run --rm -p 8080:8080 --env-file .env 2api-ai
```

### 4) Verify
Health endpoint (no auth):
```bash
curl http://localhost:8080/health
```

List models (requires Authorization):
```bash
curl http://localhost:8080/v1/models ^
  -H "Authorization: Bearer 2api_test"
```

Chat completion (example):
```bash
curl http://localhost:8080/v1/chat/completions ^
  -H "Authorization: Bearer 2api_test" ^
  -H "Content-Type: application/json" ^
  -d "{\"model\":\"openai/gpt-4o-mini\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello from 2api.ai\"}]}"
```

> In the current alpha, `Authorization: Bearer 2api_*` is validated by format/prefix. Production typically replaces this with DB-backed API key validation.

---

## 🔧 Environment

### Required (at least one)
Provider API keys (used by the server to call upstream providers):
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `GOOGLE_API_KEY`

### Required for calling 2api.ai endpoints
2api API key (sent by client in request headers):
- `Authorization: Bearer 2api_<your_key>`

### Optional / runtime
- `PORT` (default: `8080` in Docker)
- `PYTHONUNBUFFERED=1`, `PYTHONDONTWRITEBYTECODE=1`
- `DATABASE_URL`, `REDIS_URL` (for production-grade billing/usage, cache, rate limiting, etc.)

---

## 🔒 Security Notes

- **Never commit secrets**: `.env`, provider keys, service account JSON, or any credentials.
- Add (or keep) `.env` in `.gitignore`. Prefer `--env-file .env` over `-e KEY=...` to avoid leaking keys in shell history.
- Use a secrets manager in production (e.g., Cloud Run env vars wired from Secret Manager).
- If a key is ever exposed (logs, screenshots, git history), **rotate it immediately**.
- Avoid logging sensitive headers and request bodies by default.

---

## 🚀 Quick Start (SDK / Client Usage)

### Using Python SDK

```python
from twoapi import TwoAPI

client = TwoAPI(api_key="2api_your_key")

# Simple chat
response = client.chat("Hello!")
print(response.content)

# With specific model
response = client.chat(
    "Explain quantum computing",
    model="anthropic/claude-3-5-sonnet"
)

# Cost-optimized routing
response = client.chat(
    "Simple question",
    routing_strategy="cost"
)
```

### Using JavaScript SDK

```javascript
import TwoAPI from 'twoapi';

const client = new TwoAPI({ apiKey: '2api_your_key' });

// Simple chat
const response = await client.chat('Hello!');
console.log(response.content);

// Streaming
for await (const chunk of client.chatStream('Tell me a story')) {
  process.stdout.write(chunk);
}
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

## 📁 Project Structure

```
2api-ai/
├── docs/
│   ├── SPEC_INDEX.md        # 📋 Start here - links to all specs
│   ├── ARCHITECTURE.md      # System architecture
│   ├── openapi.yaml         # API specification (OpenAPI 3.0)
│   ├── STREAMING_SPEC.md    # Streaming protocol & semantics
│   ├── TOOL_CALLING_SPEC.md # Tool calling normalization
│   ├── RETRY_FALLBACK_POLICY.md # Retry, fallback, circuit breaker
│   ├── ERROR_TAXONOMY.md    # Error classification & codes
│   └── MULTI_TENANT_DESIGN.md # Tenant isolation & billing
├── src/
│   ├── core/
│   │   ├── models.py        # Unified data models
│   │   └── errors.py        # Error definitions
│   ├── adapters/
│   │   ├── base.py          # Base adapter interface
│   │   ├── openai_adapter.py
│   │   ├── anthropic_adapter.py
│   │   └── google_adapter.py
│   ├── routing/
│   │   └── router.py        # Intelligent routing service
│   ├── server.py            # FastAPI server
│   └── sdk/
│       ├── python/          # Python SDK
│       └── javascript/      # JavaScript/TypeScript SDK
├── tests/
│   └── test_contracts.py    # Contract tests for specs
├── scripts/
│   └── init-db.sql          # Database schema
├── .github/
│   └── workflows/
│       └── ci.yml           # CI pipeline
├── Dockerfile
├── docker-compose.yaml
├── requirements.txt
└── README.md
```

## 📚 Documentation

### Specifications (Start Here)

| Document | Description |
|----------|-------------|
| [**Spec Index**](docs/SPEC_INDEX.md) | 📋 Master index - start here |
| [Architecture](docs/ARCHITECTURE.md) | System design & components |
| [OpenAPI Spec](docs/openapi.yaml) | REST API definition |
| [Streaming Spec](docs/STREAMING_SPEC.md) | SSE protocol, chunk format |
| [Tool Calling Spec](docs/TOOL_CALLING_SPEC.md) | Function calling across providers |
| [Retry & Fallback](docs/RETRY_FALLBACK_POLICY.md) | Error handling, circuit breaker |
| [Error Taxonomy](docs/ERROR_TAXONOMY.md) | Error codes & classification |
| [Multi-Tenant Design](docs/MULTI_TENANT_DESIGN.md) | Isolation, billing, rate limits |

### Contract Tests

```bash
# Validate all specs are correctly implemented
pytest tests/test_contracts.py -v
```

## 🔧 Development Setup

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- API keys for OpenAI, Anthropic, and/or Google

### Local Development

1. **Clone and setup**
```bash
git clone https://github.com/2api-ai/2api
cd 2api

# Create environment file
cp .env.example .env
# Edit .env with your API keys
```

2. **Start services**
```bash
# Start all services (API, PostgreSQL, Redis)
docker-compose up -d

# Or with development tools (Adminer, Redis Commander)
docker-compose --profile dev up -d
```

3. **Verify**
```bash
# docker-compose may map to 8000 depending on docker-compose.yaml
curl http://localhost:8000/health
```

### Environment Variables

```bash
# Required: At least one provider API key
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AI...

# Optional: Database and cache
DATABASE_URL=postgresql://user:pass@localhost:5432/twoapi
REDIS_URL=redis://localhost:6379
```

## 📊 API Reference

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/v1/chat/completions` | Create chat completion |
| POST | `/v1/embeddings` | Create embeddings |
| POST | `/v1/images/generations` | Generate images |
| GET | `/v1/models` | List available models |
| GET | `/v1/usage` | Get usage statistics |
| GET | `/health` | Health check |

### Model Naming

Models are specified as `provider/model-name`:

```
openai/gpt-4o
openai/gpt-4o-mini
anthropic/claude-3-5-sonnet
anthropic/claude-3-opus
google/gemini-1.5-pro
google/gemini-1.5-flash
auto  # Let 2api.ai choose
```

### Routing Strategies

| Strategy | Description |
|----------|-------------|
| `cost` | Choose the cheapest model |
| `latency` | Choose the fastest model |
| `quality` | Choose the most capable model |

## 🏗️ Architecture Highlights

- **Stateless API**: Horizontal scaling with Cloud Run
- **Multi-tenant**: Complete data isolation via tenant_id
- **High Availability**: Automatic failover between providers
- **Observability**: Structured logging, metrics, and traces
- **Security**: API key hashing, encrypted provider keys
