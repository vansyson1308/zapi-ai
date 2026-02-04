# 2api.ai Python SDK

A simple, unified interface to access multiple AI providers (OpenAI, Anthropic, Google) through a single API.

## Installation

```bash
pip install twoapi
```

## Quick Start

```python
from twoapi import TwoAPI

# Initialize client
client = TwoAPI(api_key="2api_your_key_here")

# Simple chat
response = client.chat("Hello! How are you?")
print(response.content)
```

## Features

### Chat Completions

```python
# Basic chat
response = client.chat("What is 2+2?")

# With specific model
response = client.chat(
    "Explain quantum computing",
    model="anthropic/claude-3-5-sonnet"
)

# With system prompt
response = client.chat(
    "What should I cook?",
    system="You are a helpful chef assistant."
)

# With parameters
response = client.chat(
    "Write a poem about AI",
    temperature=0.9,
    max_tokens=500
)
```

### Streaming

```python
for chunk in client.chat_stream("Tell me a story"):
    print(chunk, end="", flush=True)
```

### Smart Routing

```python
# Cost-optimized routing
response = client.chat(
    "Simple question",
    routing_strategy="cost"
)

# With fallback
response = client.chat(
    "Important task",
    model="openai/gpt-4o",
    fallback=["anthropic/claude-3-5-sonnet", "google/gemini-1.5-pro"]
)
```

### Embeddings

```python
response = client.embed("Hello world")
print(f"Embedding dimension: {len(response.embeddings[0])}")

# Batch embeddings
response = client.embed(["Hello", "World", "AI"])
```

### Image Generation

```python
response = client.generate_image(
    "A cat wearing a top hat, digital art",
    size="1024x1024"
)
print(response.urls[0])
```

### List Models

```python
# All models
models = client.list_models()

# Filter by provider
openai_models = client.list_models(provider="openai")

# Filter by capability
vision_models = client.list_models(capability="vision")
```

## Available Models

### OpenAI
- `openai/gpt-4o` - Most capable, multimodal
- `openai/gpt-4o-mini` - Fast and affordable
- `openai/gpt-4-turbo` - Previous flagship
- `openai/text-embedding-3-small` - Embeddings
- `openai/dall-e-3` - Image generation

### Anthropic (Claude)
- `anthropic/claude-3-5-sonnet` - Best balance of speed and capability
- `anthropic/claude-3-opus` - Most capable
- `anthropic/claude-3-haiku` - Fastest

### Google (Gemini)
- `google/gemini-1.5-pro` - 2M context window
- `google/gemini-1.5-flash` - Fast and efficient
- `google/text-embedding-004` - Embeddings

## Response Objects

### ChatResponse
```python
response.content       # The generated text
response.model         # Model used
response.provider      # Provider used
response.usage         # Token usage
response.routing       # Routing information (latency, cost, strategy)
```

### Usage
```python
response.usage.prompt_tokens      # Input tokens
response.usage.completion_tokens  # Output tokens
response.usage.total_tokens       # Total tokens
```

### RoutingInfo
```python
response.routing.strategy_used   # "cost", "latency", "quality", or "explicit"
response.routing.fallback_used   # Whether fallback was triggered
response.routing.latency_ms      # Request latency
response.routing.cost_usd        # Estimated cost
```

## Error Handling

```python
from twoapi import TwoAPI, TwoAPIError, RateLimitError, AuthenticationError

try:
    response = client.chat("Hello")
except AuthenticationError:
    print("Invalid API key")
except RateLimitError as e:
    print(f"Rate limited. Retry after {e.retry_after} seconds")
except TwoAPIError as e:
    print(f"Error: {e.message} (code: {e.code})")
```

## Environment Variables

```bash
export TWOAPI_API_KEY="2api_your_key_here"
export TWOAPI_BASE_URL="https://api.2api.ai/v1"  # Optional
```

## License

MIT
