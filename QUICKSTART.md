# 2api.ai Quickstart Guide

Get up and running with 2api.ai in 5 minutes.

## Table of Contents

1. [Installation](#installation)
2. [Your First Request](#your-first-request)
3. [Using the SDKs](#using-the-sdks)
4. [Streaming](#streaming)
5. [Tool Calling](#tool-calling)
6. [Error Handling](#error-handling)
7. [Observability](#observability)
8. [Next Steps](#next-steps)

---

## Installation

### Python SDK

```bash
pip install twoapi
```

Or install from source:
```bash
cd src/sdk/python
pip install -e .
```

### JavaScript/TypeScript SDK

```bash
npm install twoapi
# or
yarn add twoapi
```

### Running the Server Locally

**Option 1: Docker (Recommended)**

```bash
# Clone the repository
git clone https://github.com/2api-ai/2api
cd 2api

# Setup environment
cp .env.example .env
# Edit .env with at least one provider key

# Run with Docker Compose
docker-compose up -d

# Test
curl http://localhost:8000/health
```

**Option 2: Python Local**

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set API keys
export OPENAI_API_KEY=sk-your-openai-key
export ANTHROPIC_API_KEY=sk-ant-your-anthropic-key
export GOOGLE_API_KEY=your-google-key

# Run server (from repo root, NOT cd into src/)
python -m uvicorn src.server:app --host 0.0.0.0 --port 8000

# Test (another terminal)
curl http://localhost:8000/health
```

---

## Your First Request

### Using curl

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer 2api_your_key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### Using Python

```python
from twoapi import TwoAPI

client = TwoAPI(api_key="2api_your_key")

# Simple one-liner
response = client.chat("Hello!")
print(response.content)
```

### Using JavaScript

```javascript
import { TwoAPI } from 'twoapi';

const client = new TwoAPI({ apiKey: '2api_your_key' });

const response = await client.chat('Hello!');
console.log(response.content);
```

---

## Using the SDKs

### Simple API vs OpenAI-Compatible API

Both SDKs support two API styles:

#### Simple API (Recommended for new projects)

```python
# Python
response = client.chat("Hello!")
response = client.chat("Hello!", model="openai/gpt-4o", temperature=0.7)
```

```javascript
// JavaScript
const response = await client.chat('Hello!');
const response = await client.chat('Hello!', { model: 'openai/gpt-4o' });
```

#### OpenAI-Compatible API (For migration from OpenAI)

```python
# Python
response = client.chat.completions.create(
    model="openai/gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

```javascript
// JavaScript
const response = await client.chat.completions.create({
    model: 'openai/gpt-4o',
    messages: [{ role: 'user', content: 'Hello!' }]
});
console.log(response.choices[0].message.content);
```

### Specifying Models

Models use the format `provider/model-name`:

```python
# Use specific models
response = client.chat("Hello!", model="openai/gpt-4o")
response = client.chat("Hello!", model="anthropic/claude-3-5-sonnet")
response = client.chat("Hello!", model="google/gemini-1.5-pro")

# Let 2api.ai choose based on strategy
response = client.chat("Hello!", model="auto")  # Default
```

### Routing Strategies

```python
from twoapi import RoutingConfig

# Cost-optimized (cheapest model)
response = client.chat("Simple question", routing=RoutingConfig(strategy="cost"))

# Latency-optimized (fastest response)
response = client.chat("Quick answer needed", routing=RoutingConfig(strategy="latency"))

# Quality-optimized (most capable model)
response = client.chat("Complex analysis", routing=RoutingConfig(strategy="quality"))
```

---

## Streaming

### Python Streaming

```python
# Simple streaming
for chunk in client.chat_stream("Tell me a story"):
    print(chunk, end="", flush=True)

# OpenAI-compatible streaming
stream = client.chat.completions.create(
    model="openai/gpt-4o",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True
)
for chunk in stream:
    content = chunk.choices[0].delta.content or ""
    print(content, end="", flush=True)
```

### JavaScript Streaming

```javascript
// Simple streaming
for await (const chunk of client.chatStream('Tell me a story')) {
    process.stdout.write(chunk.content);
}

// OpenAI-compatible streaming
const stream = await client.chat.completions.create({
    model: 'openai/gpt-4o',
    messages: [{ role: 'user', content: 'Tell me a story' }],
    stream: true
});
for await (const chunk of stream) {
    const content = chunk.choices[0]?.delta?.content || '';
    process.stdout.write(content);
}
```

### Async Python Streaming

```python
from twoapi import AsyncTwoAPI

async_client = AsyncTwoAPI()

async for chunk in async_client.chat_stream("Tell me a story"):
    print(chunk, end="", flush=True)
```

---

## Tool Calling

### Defining Tools

```python
from twoapi import tool

@tool(description="Get the current weather for a location")
def get_weather(location: str, unit: str = "celsius") -> str:
    # Your implementation here
    return f"Weather in {location}: 22{unit[0].upper()}, sunny"

@tool(description="Search the web")
def search(query: str) -> str:
    return f"Search results for: {query}"
```

```javascript
import { tool } from 'twoapi';

const getWeather = tool({
    name: 'get_weather',
    description: 'Get the current weather for a location',
    parameters: {
        type: 'object',
        properties: {
            location: { type: 'string' },
            unit: { type: 'string', enum: ['celsius', 'fahrenheit'] }
        },
        required: ['location']
    },
    execute: async (args) => `Weather in ${args.location}: sunny`
});
```

### Using ToolRunner

The `ToolRunner` automatically executes tool calls in a loop:

```python
from twoapi import TwoAPI, ToolRunner, tool

@tool(description="Get weather")
def get_weather(location: str) -> str:
    return f"Weather in {location}: sunny, 22C"

client = TwoAPI()
runner = ToolRunner(
    tools=[get_weather],
    max_iterations=5,  # Prevent infinite loops
    on_tool_call=lambda name, args, result: print(f"Called {name}")
)

result = runner.run(client, "What's the weather in Tokyo and New York?")

print(result.response.content)     # Final natural language response
print(result.tool_calls_made)      # List of executed tool calls
print(result.total_iterations)     # Number of LLM calls
print(result.messages)             # Full conversation history
```

```javascript
import { TwoAPI, ToolRunner, tool } from 'twoapi';

const runner = new ToolRunner([getWeather], {
    maxIterations: 5,
    onToolCall: (name, args, result) => console.log(`Called ${name}`)
});

const result = await runner.run(client, 'What is the weather in Tokyo?');
console.log(result.response.content);
```

---

## Error Handling

### Error Types

| Error | Status | Retryable | Description |
|-------|--------|-----------|-------------|
| `AuthenticationError` | 401 | No | Invalid or missing API key |
| `RateLimitError` | 429 | Yes | Too many requests |
| `InvalidRequestError` | 400 | No | Bad request parameters |
| `ProviderError` | 5xx | Yes | Provider-side issues |
| `TimeoutError` | - | Yes | Request timeout |
| `ConnectionError` | - | Yes | Network issues |

### Python Error Handling

```python
from twoapi import (
    TwoAPI,
    AuthenticationError,
    RateLimitError,
    ProviderError,
    is_retryable_error
)

client = TwoAPI()

try:
    response = client.chat("Hello")
except AuthenticationError:
    print("Check your API key")
except RateLimitError as e:
    print(f"Rate limited. Retry after {e.retry_after} seconds")
except ProviderError as e:
    print(f"Provider {e.provider} is having issues")
    if is_retryable_error(e):
        # Safe to retry
        pass
```

### JavaScript Error Handling

```javascript
import {
    TwoAPI,
    AuthenticationError,
    RateLimitError,
    ProviderError,
    isRetryableError
} from 'twoapi';

try {
    const response = await client.chat('Hello');
} catch (error) {
    if (error instanceof AuthenticationError) {
        console.log('Check your API key');
    } else if (error instanceof RateLimitError) {
        console.log(`Rate limited. Retry after ${error.retryAfter}s`);
    } else if (isRetryableError(error)) {
        // Safe to retry
    }
}
```

### Custom Retry Configuration

```python
from twoapi import TwoAPI, RetryConfig

client = TwoAPI(
    api_key="2api_xxx",
    max_retries=5,
    on_retry=lambda attempt, error, delay: print(f"Retry {attempt} in {delay}s")
)

# Or with full configuration
client = TwoAPI(
    retry_config=RetryConfig(
        max_retries=5,
        initial_delay=1.0,
        max_delay=30.0,
        exponential_base=2.0
    )
)
```

---

## Observability

### Prometheus Metrics

```bash
curl http://localhost:8000/metrics
```

Key metrics available:

| Metric | Type | Description |
|--------|------|-------------|
| `twoapi_requests_total` | Counter | Total requests by endpoint, method, status |
| `twoapi_request_duration_seconds` | Histogram | Request latency distribution |
| `twoapi_tokens_total` | Counter | Token usage by type and model |
| `twoapi_provider_requests_total` | Counter | Requests per provider |
| `twoapi_active_requests` | Gauge | Currently processing requests |

### Health Checks

```bash
# Liveness - is the service running?
curl http://localhost:8000/health

# Readiness - is the service ready to accept traffic?
curl http://localhost:8000/ready
```

---

## Next Steps

### Learn More

- [Architecture](docs/ARCHITECTURE.md) - System design and components
- [Error Taxonomy](docs/ERROR_TAXONOMY.md) - Complete error reference
- [Streaming Spec](docs/STREAMING_SPEC.md) - SSE protocol details
- [Tool Calling Spec](docs/TOOL_CALLING_SPEC.md) - Tool format normalization

### Examples

Check out the example files:

**Python:**
- `src/sdk/python/examples/basic_chat.py`
- `src/sdk/python/examples/streaming.py`
- `src/sdk/python/examples/tool_calling.py`
- `src/sdk/python/examples/error_handling.py`
- `src/sdk/python/examples/async_usage.py`

**JavaScript:**
- `src/sdk/javascript/examples/basic-chat.ts`
- `src/sdk/javascript/examples/streaming.ts`
- `src/sdk/javascript/examples/tool-calling.ts`
- `src/sdk/javascript/examples/error-handling.ts`

### Run Tests

```bash
# Run all 420 tests
cd 2api-ai
pip install pytest pytest-asyncio
pytest tests/ -v
```

### Need Help?

- Check the [README](README.md) for complete documentation
- Review the [API Reference](README.md#api-reference)
- Look at the test files for more usage examples

---

## Troubleshooting

### Import Error
```bash
# IMPORTANT: Run from repo root, NOT cd into src/
cd 2api-ai
python -m uvicorn src.server:app --port 8000
```

### Port Already in Use
```bash
# Use different port
python -m uvicorn src.server:app --port 8001
```

### Missing Dependencies
```bash
pip install fastapi uvicorn httpx pydantic prometheus-client opentelemetry-api
```
