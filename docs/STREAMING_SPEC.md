# 2api.ai - Streaming Specification

## 1. Protocol

2api.ai sử dụng **Server-Sent Events (SSE)** theo chuẩn OpenAI-compatible.

```
Content-Type: text/event-stream
Cache-Control: no-cache
Connection: keep-alive
```

## 2. Chunk Format

### 2.1 Standard Text Chunk

```
data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1700000000,"model":"openai/gpt-4o","provider":"openai","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"},"finish_reason":null}]}\n\n
```

### 2.2 Tool Call Delta Chunk

Tool calls được stream theo **delta mode** (không phải final-only):

```
data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_abc","type":"function","function":{"name":"get_weather","arguments":""}}]},"finish_reason":null}]}\n\n

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"lo"}}]},"finish_reason":null}]}\n\n

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"cation"}}]},"finish_reason":null}]}\n\n

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\":\"Tokyo\"}"}}]},"finish_reason":null}]}\n\n
```

### 2.3 Final Chunk

```
data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n

data: [DONE]\n\n
```

## 3. Unified Chunk Schema

```typescript
interface StreamChunk {
  id: string;                    // Request ID, consistent across all chunks
  object: "chat.completion.chunk";
  created: number;               // Unix timestamp
  model: string;                 // "provider/model-name"
  provider: string;              // "openai" | "anthropic" | "google"
  
  choices: [{
    index: number;               // Always 0 for single completion
    delta: {
      role?: "assistant";        // Only in first chunk
      content?: string;          // Text content delta
      tool_calls?: ToolCallDelta[];
    };
    finish_reason: "stop" | "length" | "tool_calls" | "content_filter" | null;
  }];
  
  // 2api.ai metadata (only in final chunk with finish_reason)
  usage?: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
  _2api?: {
    request_id: string;
    latency_ms: number;
    cost_usd: number;
  };
}

interface ToolCallDelta {
  index: number;                 // Tool call index (for multiple parallel calls)
  id?: string;                   // Only in first delta for this tool
  type?: "function";             // Only in first delta
  function: {
    name?: string;               // Only in first delta
    arguments: string;           // Accumulated JSON string delta
  };
}
```

## 4. Provider Normalization

| Provider | Native Format | 2api.ai Normalization |
|----------|---------------|----------------------|
| OpenAI | SSE `data: {json}` | Pass-through với thêm `provider` field |
| Anthropic | SSE với `event:` types | Convert sang OpenAI-style delta |
| Google | SSE với nested structure | Flatten sang OpenAI-style delta |

### 4.1 Anthropic Conversion

```
# Anthropic native:
event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}

# 2api.ai normalized:
data: {"id":"msg_xxx","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}
```

### 4.2 Google Conversion

```
# Google native:
data: {"candidates":[{"content":{"parts":[{"text":"Hello"}]}}]}

# 2api.ai normalized:
data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}
```

## 5. Error During Stream

Khi có lỗi trong quá trình streaming:

```
data: {"error":{"code":"provider_error","message":"Upstream timeout","type":"infra_error","provider":"openai","partial_content":"Hello, I can help you with","recoverable":true}}\n\n

data: [DONE]\n\n
```

### 5.1 Error Schema trong Stream

```typescript
interface StreamError {
  error: {
    code: string;
    message: string;
    type: "infra_error" | "semantic_error";
    provider?: string;
    partial_content?: string;    // Content đã stream được trước khi lỗi
    recoverable: boolean;        // Client có thể retry không
  };
}
```

## 6. Client Implementation Requirements

### 6.1 Accumulation

Client MUST accumulate:
- `content` strings → full response text
- `tool_calls[i].function.arguments` → complete JSON

### 6.2 Tool Call Reconstruction

```javascript
const toolCalls = {};

for (const chunk of stream) {
  for (const tc of chunk.choices[0].delta.tool_calls || []) {
    if (!toolCalls[tc.index]) {
      toolCalls[tc.index] = { id: tc.id, type: tc.type, function: { name: tc.function?.name, arguments: '' } };
    }
    if (tc.function?.arguments) {
      toolCalls[tc.index].function.arguments += tc.function.arguments;
    }
  }
}

// Parse complete arguments
for (const tc of Object.values(toolCalls)) {
  tc.function.parsedArguments = JSON.parse(tc.function.arguments);
}
```

## 7. Partial Failure Handling

| Scenario | Behavior |
|----------|----------|
| Provider timeout mid-stream | Send error chunk với `partial_content`, close stream |
| Rate limit mid-stream | Send error chunk, close stream, DO NOT fallback (semantic drift) |
| Network error mid-stream | Send error chunk nếu có thể, close connection |
| Content filter mid-stream | Send error chunk với `finish_reason: "content_filter"` |

**Critical Rule**: Không bao giờ fallback sang provider khác khi đã bắt đầu stream content. Điều này gây semantic drift không thể recover.

## 8. Timeouts

| Phase | Timeout | Behavior |
|-------|---------|----------|
| Connection | 10s | Error, có thể retry |
| First byte | 30s | Error, có thể retry |
| Between chunks | 60s | Error với partial_content |
| Total stream | 5 min | Graceful close |

## 9. Keep-Alive & Heartbeat

### 9.1 Server-Side Keep-Alive

To prevent proxies/load balancers from closing idle connections:

```
# Every 15 seconds during active stream, if no content chunk sent:
data: {"type":"heartbeat","timestamp":1700000000}\n\n
```

**Heartbeat Schema**:
```typescript
interface HeartbeatChunk {
  type: "heartbeat";
  timestamp: number;
}
```

**Client Behavior**: Clients MUST ignore heartbeat chunks (they are not content).

### 9.2 Connection Timeouts

| Component | Timeout | Purpose |
|-----------|---------|---------|
| Proxy (nginx/cloudflare) | 60s | Idle connection |
| 2api.ai heartbeat | 15s | Keep connection alive |
| Client read timeout | 120s | Max wait between chunks |

## 10. Reconnect Semantics

### 10.1 Client Reconnection Strategy

When client loses connection mid-stream:

```
┌─────────────────────────────────────────────────────────────┐
│  Client disconnects at chunk N                              │
│                                                             │
│  Option A: Accept partial content                           │
│  └─► Use accumulated content, mark as incomplete            │
│                                                             │
│  Option B: Retry from beginning (NEW request)               │
│  └─► WARNING: Will get DIFFERENT content (non-deterministic)│
│                                                             │
│  Option C: Continue from partial (recommended)              │
│  └─► Send new request with partial content as context       │
└─────────────────────────────────────────────────────────────┘
```

### 10.2 Continuation Request Pattern

```python
# Client accumulated "The weather in Tokyo is" before disconnect

continuation_request = {
    "model": "openai/gpt-4o",
    "messages": [
        {"role": "user", "content": "What's the weather in Tokyo?"},
        {"role": "assistant", "content": "The weather in Tokyo is"},  # Partial
        {"role": "user", "content": "Please continue your response."}
    ]
}
```

### 10.3 Server Does NOT Support Resume

**Important**: 2api.ai does NOT support "resume from offset" semantics.

- No `Last-Event-ID` header support
- No chunk sequence numbers for resumption
- Each request is independent

**Rationale**: LLM outputs are non-deterministic. Resuming mid-stream would create semantic inconsistency.

## 11. Backpressure Handling

### 11.1 Slow Client Detection

If client reads slower than server produces:

| Scenario | Server Behavior |
|----------|-----------------|
| Buffer < 64KB | Continue streaming |
| Buffer 64KB-256KB | Slow down upstream reads |
| Buffer > 256KB | Pause upstream, wait for client |
| Client stalled > 60s | Close connection with error |

### 11.2 Implementation Note

```python
async def stream_with_backpressure(upstream: AsyncIterator, response: StreamingResponse):
    buffer = asyncio.Queue(maxsize=100)  # ~100 chunks max buffer
    
    async def producer():
        async for chunk in upstream:
            try:
                await asyncio.wait_for(buffer.put(chunk), timeout=60)
            except asyncio.TimeoutError:
                # Client too slow, abort
                raise StreamBackpressureError("Client not consuming fast enough")
    
    async def consumer():
        while True:
            chunk = await buffer.get()
            if chunk is None:
                break
            yield chunk
    
    # Run producer and consumer concurrently
    await asyncio.gather(producer(), stream_response(consumer()))
```

### 11.3 Client Recommendations

- Process chunks immediately, don't buffer excessively
- Use async/streaming HTTP clients
- If processing is slow, accumulate to string incrementally

## 12. Contract Tests

```python
def test_stream_chunk_has_required_fields():
    """Every chunk must have id, object, choices."""
    for chunk in stream_response:
        assert "id" in chunk
        assert chunk["object"] == "chat.completion.chunk"
        assert "choices" in chunk
        assert len(chunk["choices"]) > 0

def test_stream_ends_with_done():
    """Stream must end with [DONE]."""
    chunks = list(stream_response)
    assert chunks[-1] == "[DONE]"

def test_finish_reason_only_in_final():
    """finish_reason must be null except in final content chunk."""
    chunks = list(stream_response)
    for chunk in chunks[:-2]:  # Exclude final chunk and [DONE]
        assert chunk["choices"][0]["finish_reason"] is None

def test_tool_calls_accumulate_correctly():
    """Tool call arguments must form valid JSON when accumulated."""
    accumulated = ""
    for chunk in stream_response:
        delta = chunk["choices"][0]["delta"]
        if "tool_calls" in delta:
            accumulated += delta["tool_calls"][0]["function"].get("arguments", "")
    
    if accumulated:
        json.loads(accumulated)  # Must not raise

def test_provider_normalized():
    """All chunks must have consistent provider field."""
    providers = set()
    for chunk in stream_response:
        providers.add(chunk.get("provider"))
    assert len(providers) == 1
```
