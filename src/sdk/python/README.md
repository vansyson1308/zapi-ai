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

### Async Support

```python
from twoapi import AsyncTwoAPI
import asyncio

async def main():
    async with AsyncTwoAPI(api_key="2api_xxx") as client:
        # Async chat
        response = await client.chat("Hello!")
        print(response.content)

        # Async streaming
        async for chunk in client.chat_stream("Tell me a story"):
            print(chunk, end="", flush=True)

asyncio.run(main())
```

### Streaming

```python
# Sync streaming
for chunk in client.chat_stream("Tell me a story"):
    print(chunk, end="", flush=True)

# Async streaming
async for chunk in async_client.chat_stream("Tell me a story"):
    print(chunk, end="", flush=True)
```

### Tool Calling (Function Calling)

```python
from twoapi import TwoAPI, Tool, Message

client = TwoAPI(api_key="2api_xxx")

# Define tools
tools = [
    Tool.create(
        name="get_weather",
        description="Get the current weather for a location",
        parameters={
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name"
                }
            },
            "required": ["location"]
        }
    )
]

# Chat with tools
response = client.chat(
    "What's the weather in Tokyo?",
    tools=tools
)

# Check for tool calls
if response.has_tool_calls:
    for tool_call in response.tool_calls:
        print(f"Function: {tool_call.function.name}")
        print(f"Arguments: {tool_call.function.arguments}")

        # Execute function and continue conversation
        result = execute_function(tool_call)  # Your function

        # Send result back
        messages = [
            Message.user("What's the weather in Tokyo?"),
            Message.assistant(tool_calls=response.tool_calls),
            Message.tool(tool_call.id, result)
        ]
        final_response = client.chat(messages)
        print(final_response.content)
```

### Smart Routing

```python
from twoapi import RoutingConfig

# Cost-optimized routing
response = client.chat(
    "Simple question",
    routing=RoutingConfig(strategy="cost")
)

# With fallback providers
response = client.chat(
    "Important task",
    model="openai/gpt-4o",
    routing=RoutingConfig(
        fallback=["anthropic/claude-3-5-sonnet", "google/gemini-1.5-pro"],
        max_latency_ms=5000
    )
)
```

### Automatic Retry

```python
from twoapi import TwoAPI, RetryConfig

# Configure retry behavior
client = TwoAPI(
    api_key="2api_xxx",
    retry_config=RetryConfig(
        max_retries=5,
        initial_delay=0.5,
        max_delay=30.0,
        exponential_base=2.0
    )
)

# Requests will automatically retry on transient errors
response = client.chat("Hello!")
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
response.tool_calls    # List of tool calls (if any)
response.has_tool_calls # Boolean check for tool calls
response.finish_reason # Why generation stopped
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
from twoapi import (
    TwoAPI,
    TwoAPIError,
    RateLimitError,
    AuthenticationError,
    InvalidRequestError,
    ProviderError,
    TimeoutError,
    ConnectionError,
)

try:
    response = client.chat("Hello")
except AuthenticationError:
    print("Invalid API key")
except RateLimitError as e:
    print(f"Rate limited. Retry after {e.retry_after} seconds")
except InvalidRequestError as e:
    print(f"Invalid request: {e.message}, param: {e.param}")
except ProviderError as e:
    print(f"Provider {e.provider} error: {e.message}")
except TimeoutError:
    print("Request timed out")
except ConnectionError:
    print("Failed to connect to API")
except TwoAPIError as e:
    print(f"Error: {e.message} (code: {e.code})")
    if e.retryable:
        print("This error can be retried")
```

## Environment Variables

```bash
export TWOAPI_API_KEY="2api_your_key_here"
export TWOAPI_BASE_URL="https://api.2api.ai/v1"  # Optional
```

## Convenience Functions

For quick one-off calls without creating a client:

```python
import twoapi

# Set API key once
twoapi.set_api_key("2api_xxx")

# Use convenience functions
response = twoapi.chat("Hello!")
embedding = twoapi.embed("Hello world")
image = twoapi.generate_image("A sunset")

# Streaming
for chunk in twoapi.chat_stream("Tell me a story"):
    print(chunk, end="")
```

## License

MIT
