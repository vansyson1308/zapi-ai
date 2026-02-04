# 2api.ai - Tool Calling Specification

## 1. Overview

Tool calling cho phép AI models gọi external functions. 2api.ai cung cấp unified interface hoạt động nhất quán across all providers.

## 2. Unified Tool Schema

### 2.1 Tool Definition (Request)

```typescript
interface Tool {
  type: "function";
  function: {
    name: string;                    // ^[a-zA-Z0-9_-]{1,64}$
    description: string;             // Max 1024 chars
    parameters: JSONSchema;          // JSON Schema object
    strict?: boolean;                // Enforce schema validation (default: false)
  };
}

// Example
{
  "type": "function",
  "function": {
    "name": "get_weather",
    "description": "Get current weather for a location",
    "parameters": {
      "type": "object",
      "properties": {
        "location": {
          "type": "string",
          "description": "City name, e.g., 'Tokyo'"
        },
        "unit": {
          "type": "string",
          "enum": ["celsius", "fahrenheit"],
          "default": "celsius"
        }
      },
      "required": ["location"]
    }
  }
}
```

### 2.2 Tool Call (Response)

```typescript
interface ToolCall {
  id: string;                        // Unique ID, format: "call_{random}"
  type: "function";
  function: {
    name: string;
    arguments: string;               // JSON string (may be malformed!)
  };
}

// Example
{
  "id": "call_abc123xyz",
  "type": "function",
  "function": {
    "name": "get_weather",
    "arguments": "{\"location\":\"Tokyo\",\"unit\":\"celsius\"}"
  }
}
```

### 2.3 Tool Result (Follow-up Request)

```typescript
interface ToolResultMessage {
  role: "tool";
  tool_call_id: string;              // Must match ToolCall.id
  content: string;                   // Tool execution result (string or JSON string)
}

// Example
{
  "role": "tool",
  "tool_call_id": "call_abc123xyz",
  "content": "{\"temperature\": 22, \"condition\": \"sunny\"}"
}
```

## 3. Provider Normalization

### 3.1 Request Normalization (2api → Provider)

| 2api.ai | OpenAI | Anthropic | Google |
|---------|--------|-----------|--------|
| `tools[].function.name` | `tools[].function.name` | `tools[].name` | `tools[].functionDeclarations[].name` |
| `tools[].function.description` | `tools[].function.description` | `tools[].description` | `tools[].functionDeclarations[].description` |
| `tools[].function.parameters` | `tools[].function.parameters` | `tools[].input_schema` | `tools[].functionDeclarations[].parameters` |
| `tool_choice: "auto"` | `tool_choice: "auto"` | `tool_choice: {type: "auto"}` | (default behavior) |
| `tool_choice: "none"` | `tool_choice: "none"` | `tool_choice: {type: "none"}` | (not supported) |
| `tool_choice: "required"` | `tool_choice: "required"` | `tool_choice: {type: "any"}` | (not supported) |
| `tool_choice: {function: {name}}` | `tool_choice: {type: "function", function: {name}}` | `tool_choice: {type: "tool", name}` | (not supported) |

### 3.2 Response Normalization (Provider → 2api)

| Provider Response | 2api.ai Normalized |
|-------------------|-------------------|
| OpenAI `tool_calls[].id` | `tool_calls[].id` (pass-through) |
| Anthropic `content[].id` (tool_use block) | `tool_calls[].id` |
| Google `functionCall` (no ID) | `tool_calls[].id` = generated `call_{timestamp}_{name}` |
| OpenAI `tool_calls[].function.arguments` | `tool_calls[].function.arguments` |
| Anthropic `content[].input` (object) | `tool_calls[].function.arguments` = `JSON.stringify(input)` |
| Google `functionCall.args` (object) | `tool_calls[].function.arguments` = `JSON.stringify(args)` |

### 3.3 Tool Result Normalization (2api → Provider)

| 2api.ai | OpenAI | Anthropic | Google |
|---------|--------|-----------|--------|
| `role: "tool"` | `role: "tool"` | `role: "user"` + `tool_result` block | `role: "user"` + `functionResponse` |
| `tool_call_id` | `tool_call_id` | `content[].tool_use_id` | `functionResponse.name` |
| `content` | `content` | `content[].content` | `functionResponse.response` |

## 4. Execution Loop Ownership

### 4.1 Architecture Decision: Client-Side Execution

```
┌──────────────────────────────────────────────────────────────┐
│                         CLIENT                                │
│                                                              │
│  1. Send request with tools                                  │
│  2. Receive response with tool_calls                         │
│  3. Execute tools locally          ◄── CLIENT RESPONSIBILITY │
│  4. Send tool results                                        │
│  5. Receive final response                                   │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│                        2api.ai                               │
│                                                              │
│  - Route requests                                            │
│  - Normalize tool schemas                                    │
│  - Normalize tool_calls responses                            │
│  - Track usage                                               │
│  - DO NOT execute tools        ◄── SERVER DOES NOT EXECUTE   │
└──────────────────────────────────────────────────────────────┘
```

**Rationale**:
- Security: Server không có access tới client's systems
- Flexibility: Client controls execution environment
- Simplicity: No need for webhook/callback infrastructure

### 4.2 Client Execution Flow

```python
# Python SDK example
response = client.chat(
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
    tools=[get_weather_tool]
)

# Loop until no more tool calls
while response.tool_calls:
    tool_results = []
    
    for tool_call in response.tool_calls:
        # CLIENT executes the tool
        result = execute_tool(tool_call.function.name, tool_call.function.arguments)
        tool_results.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(result)
        })
    
    # Send results back
    response = client.chat(
        messages=[
            {"role": "user", "content": "What's the weather in Tokyo?"},
            {"role": "assistant", "tool_calls": response.tool_calls},
            *tool_results
        ],
        tools=[get_weather_tool]
    )

print(response.content)  # Final text response
```

## 5. Safe Defaults & Edge Cases

### 5.1 tool_choice Defaults

| Scenario | Default Behavior |
|----------|------------------|
| `tools` provided, no `tool_choice` | `tool_choice: "auto"` |
| `tools` empty or null | No tool calling |
| `tool_choice: "required"` but model doesn't call | Error response |

### 5.2 Unhandled Tool Calls

**Problem**: Model returns `tool_calls` but client doesn't handle them.

**Policy**:
```typescript
// Response includes warning
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": null,
      "tool_calls": [...]
    },
    "finish_reason": "tool_calls"
  }],
  "_2api": {
    "warning": "Response contains tool_calls. Client must execute tools and send results to continue conversation."
  }
}
```

### 5.3 Malformed Tool Arguments

**Problem**: Model generates invalid JSON in `arguments`.

**Policy**:
- 2api.ai **DOES NOT** validate or fix `arguments`
- Client MUST handle `JSON.parse()` errors
- Recommended: Wrap parse in try-catch, ask model to retry on failure

```python
try:
    args = json.loads(tool_call.function.arguments)
except json.JSONDecodeError:
    # Option 1: Ask model to fix
    response = client.chat(
        messages=[
            *previous_messages,
            {"role": "assistant", "tool_calls": [tool_call]},
            {"role": "tool", "tool_call_id": tool_call.id, 
             "content": "Error: Invalid JSON in arguments. Please retry with valid JSON."}
        ]
    )
```

### 5.4 Tool Schema Mismatch

**Problem**: Tool schema differs between providers (Google doesn't support some JSON Schema features).

**Policy**:
- 2api.ai validates schema compatibility before routing
- If schema incompatible with target provider → return `semantic_error`

```json
{
  "error": {
    "code": "tool_schema_incompatible",
    "message": "Tool 'complex_query' uses 'oneOf' which is not supported by google/gemini-1.5-pro",
    "type": "semantic_error",
    "incompatible_features": ["oneOf", "anyOf"]
  }
}
```

### 5.5 Parallel Tool Calls

Some models (GPT-4, Claude) can return multiple tool calls in one response.

**Guaranteed Behavior**:
- `tool_calls` is always an array (even for single call)
- `tool_calls[].index` indicates order for parallel calls
- Client SHOULD execute in parallel when possible
- Tool results can be sent in any order (matched by `tool_call_id`)

## 6. Provider Capability Matrix

| Feature | OpenAI | Anthropic | Google |
|---------|--------|-----------|--------|
| Basic tool calling | ✅ | ✅ | ✅ |
| Parallel tool calls | ✅ | ✅ | ❌ |
| `tool_choice: "none"` | ✅ | ✅ | ❌ |
| `tool_choice: "required"` | ✅ | ✅ (as "any") | ❌ |
| `tool_choice: {specific}` | ✅ | ✅ | ❌ |
| Streaming tool calls | ✅ (delta) | ✅ (delta) | ⚠️ (final only) |
| Nested object params | ✅ | ✅ | ✅ |
| `oneOf`/`anyOf` | ✅ | ✅ | ❌ |
| `$ref` | ✅ | ❌ | ❌ |

## 7. Guardrails & Limits

### 7.1 Request Limits

| Limit | Default | Max | Configurable |
|-------|---------|-----|--------------|
| Tools per request | 128 | 128 | No |
| Tool name length | 64 chars | 64 chars | No |
| Tool description length | 1024 chars | 4096 chars | Per-tenant |
| Parameters schema depth | 5 levels | 10 levels | No |
| Arguments size | 64KB | 256KB | Per-tenant |
| Tool calls per response | 20 | 50 | Per-tenant |

### 7.2 Tool Name Allowlist (Optional)

Tenants can configure allowed tool names:

```json
{
  "tenant_settings": {
    "tool_allowlist": ["get_weather", "search_db", "send_email"],
    "tool_allowlist_mode": "enforce"  // "enforce" | "warn" | "off"
  }
}
```

**Behavior**:
- `enforce`: Reject requests with tools not in allowlist
- `warn`: Log warning but allow
- `off`: No restriction

### 7.3 Arguments Validation

2api.ai optionally validates tool call arguments against schema:

```json
{
  "request_options": {
    "validate_tool_arguments": true  // Default: false
  }
}
```

When enabled:
- Invalid arguments → Error response (not forwarded to client as tool_call)
- Prevents malformed JSON from reaching client execution

### 7.4 Rate Limiting for Tool Calls

Tool calls count towards rate limits:
- 1 request with 5 tool_calls = 5 "tool call units"
- Separate `tool_calls_per_minute` limit available

## 8. Tool Result Content Semantics

### 8.1 Content-Type Handling

Tool result `content` is always a **string**, but semantics vary:

| Content Pattern | Interpretation | Best Practice |
|-----------------|----------------|---------------|
| Plain text | Human-readable result | Simple responses |
| JSON string | Structured data | Data retrieval tools |
| Error message | Tool execution failed | Prefix with "Error:" |
| Empty string | Tool executed, no output | Acknowledge actions |

### 8.2 Recommended Patterns

**Successful data retrieval**:
```json
{
  "role": "tool",
  "tool_call_id": "call_abc",
  "content": "{\"temperature\": 22, \"unit\": \"celsius\", \"condition\": \"sunny\"}"
}
```

**Successful action (no return data)**:
```json
{
  "role": "tool",
  "tool_call_id": "call_abc",
  "content": "Email sent successfully to user@example.com"
}
```

**Tool execution error**:
```json
{
  "role": "tool",
  "tool_call_id": "call_abc",
  "content": "Error: Database connection failed. Please try again later."
}
```

**Tool not found / permission denied**:
```json
{
  "role": "tool",
  "tool_call_id": "call_abc",
  "content": "Error: Tool 'delete_all_data' is not available. Permission denied."
}
```

### 8.3 Large Tool Results

For results > 64KB:

```python
# Option 1: Truncate with notice
content = result[:60000] + "\n\n[Truncated: Result exceeded 64KB limit]"

# Option 2: Summarize
content = f"Query returned {len(rows)} rows. First 100: {json.dumps(rows[:100])}"

# Option 3: Reference external storage
content = json.dumps({
    "status": "success",
    "result_url": "https://storage.example.com/results/abc123",
    "result_size_bytes": 1500000,
    "preview": rows[:10]
})
```

## 9. Contract Tests

```python
def test_tool_call_has_required_fields():
    """Every tool_call must have id, type, function."""
    response = client.chat(messages=[...], tools=[...])
    for tc in response.tool_calls or []:
        assert tc.id is not None
        assert tc.type == "function"
        assert tc.function.name is not None
        assert tc.function.arguments is not None

def test_tool_call_id_format():
    """Tool call ID must be non-empty string."""
    response = client.chat(messages=[...], tools=[...])
    for tc in response.tool_calls or []:
        assert isinstance(tc.id, str)
        assert len(tc.id) > 0

def test_tool_result_matches_call():
    """Tool result must reference valid tool_call_id."""
    # This is a client-side validation

def test_finish_reason_tool_calls():
    """When tool_calls present, finish_reason must be 'tool_calls'."""
    response = client.chat(messages=[...], tools=[...])
    if response.tool_calls:
        assert response.finish_reason == "tool_calls"

def test_arguments_is_string():
    """arguments must always be string (JSON), never parsed object."""
    response = client.chat(messages=[...], tools=[...])
    for tc in response.tool_calls or []:
        assert isinstance(tc.function.arguments, str)
```
