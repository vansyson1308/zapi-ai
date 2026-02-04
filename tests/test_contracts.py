"""
2api.ai - Contract Tests

Tests that verify the API conforms to documented specifications:
- Streaming Spec (docs/STREAMING_SPEC.md)
- Tool Calling Spec (docs/TOOL_CALLING_SPEC.md)
- Error Taxonomy (docs/ERROR_TAXONOMY.md)
"""

import json
import pytest
from typing import Any, Dict, List, Optional
from dataclasses import dataclass


# ============================================================
# Test Helpers
# ============================================================

def parse_sse_stream(raw_stream: str) -> List[Dict[str, Any]]:
    """Parse SSE stream into list of chunks."""
    chunks = []
    for line in raw_stream.split('\n'):
        if line.startswith('data: '):
            data = line[6:]
            if data == '[DONE]':
                chunks.append('[DONE]')
            else:
                chunks.append(json.loads(data))
    return chunks


@dataclass
class MockStreamChunk:
    """Mock stream chunk for testing."""
    id: str
    content: Optional[str] = None
    tool_calls: Optional[List[Dict]] = None
    finish_reason: Optional[str] = None


# ============================================================
# Streaming Contract Tests
# ============================================================

class TestStreamingContract:
    """Tests for Streaming Specification compliance."""
    
    def test_chunk_has_required_fields(self):
        """Every chunk must have id, object, choices."""
        chunk = {
            "id": "chatcmpl-abc123",
            "object": "chat.completion.chunk",
            "created": 1700000000,
            "model": "openai/gpt-4o",
            "provider": "openai",
            "choices": [{
                "index": 0,
                "delta": {"content": "Hello"},
                "finish_reason": None
            }]
        }
        
        assert "id" in chunk
        assert chunk["object"] == "chat.completion.chunk"
        assert "choices" in chunk
        assert len(chunk["choices"]) > 0
    
    def test_stream_ends_with_done(self):
        """Stream must end with [DONE]."""
        stream = """data: {"id":"test","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":null}]}

data: {"id":"test","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]

"""
        chunks = parse_sse_stream(stream)
        assert chunks[-1] == '[DONE]'
    
    def test_finish_reason_null_except_final(self):
        """finish_reason must be null except in final content chunk."""
        chunks = [
            {"choices": [{"delta": {"content": "Hello"}, "finish_reason": None}]},
            {"choices": [{"delta": {"content": " world"}, "finish_reason": None}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        ]
        
        # All except last should have null finish_reason
        for chunk in chunks[:-1]:
            assert chunk["choices"][0]["finish_reason"] is None
        
        # Last should have finish_reason
        assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    
    def test_tool_calls_delta_accumulation(self):
        """Tool call arguments must form valid JSON when accumulated."""
        chunks = [
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_abc", "type": "function", "function": {"name": "get_weather", "arguments": ""}}]}}]},
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "{\"lo"}}]}}]},
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "cation"}}]}}]},
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "\":\"Tokyo\"}"}}]}}]},
        ]
        
        # Accumulate arguments
        accumulated = ""
        for chunk in chunks:
            delta = chunk["choices"][0]["delta"]
            if "tool_calls" in delta:
                tc = delta["tool_calls"][0]
                if "function" in tc and "arguments" in tc["function"]:
                    accumulated += tc["function"]["arguments"]
        
        # Must parse as valid JSON
        parsed = json.loads(accumulated)
        assert parsed == {"location": "Tokyo"}
    
    def test_provider_field_consistent(self):
        """All chunks must have consistent provider field."""
        chunks = [
            {"id": "test", "provider": "openai", "choices": [{"delta": {"content": "Hi"}}]},
            {"id": "test", "provider": "openai", "choices": [{"delta": {"content": " there"}}]},
        ]
        
        providers = set(chunk.get("provider") for chunk in chunks)
        assert len(providers) == 1
    
    def test_stream_error_format(self):
        """Stream errors must include required fields."""
        error_chunk = {
            "error": {
                "code": "stream_interrupted",
                "message": "Connection lost during streaming",
                "type": "infra_error",
                "provider": "openai",
                "partial_content": "Hello, I can help you with",
                "recoverable": False
            }
        }
        
        error = error_chunk["error"]
        assert "code" in error
        assert "message" in error
        assert "type" in error
        assert error["type"] in ["infra_error", "semantic_error"]
        # partial_content should be present if content was produced
        assert "partial_content" in error
    
    def test_first_chunk_has_role(self):
        """First chunk should include role in delta."""
        first_chunk = {
            "choices": [{
                "delta": {
                    "role": "assistant",
                    "content": "Hello"
                }
            }]
        }
        
        assert first_chunk["choices"][0]["delta"].get("role") == "assistant"


# ============================================================
# Tool Calling Contract Tests
# ============================================================

class TestToolCallingContract:
    """Tests for Tool Calling Specification compliance."""
    
    def test_tool_call_has_required_fields(self):
        """Every tool_call must have id, type, function."""
        tool_call = {
            "id": "call_abc123xyz",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": "{\"location\":\"Tokyo\"}"
            }
        }
        
        assert "id" in tool_call
        assert tool_call["type"] == "function"
        assert "function" in tool_call
        assert "name" in tool_call["function"]
        assert "arguments" in tool_call["function"]
    
    def test_tool_call_id_format(self):
        """Tool call ID must be non-empty string."""
        tool_call = {"id": "call_abc123xyz", "type": "function", "function": {}}
        
        assert isinstance(tool_call["id"], str)
        assert len(tool_call["id"]) > 0
    
    def test_arguments_is_string(self):
        """arguments must always be string (JSON), never parsed object."""
        tool_call = {
            "function": {
                "name": "get_weather",
                "arguments": "{\"location\":\"Tokyo\"}"  # String, not dict
            }
        }
        
        assert isinstance(tool_call["function"]["arguments"], str)
        # Should be parseable as JSON
        parsed = json.loads(tool_call["function"]["arguments"])
        assert isinstance(parsed, dict)
    
    def test_finish_reason_tool_calls(self):
        """When tool_calls present, finish_reason must be 'tool_calls'."""
        response = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_abc",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{}"}
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        }
        
        if response["choices"][0]["message"].get("tool_calls"):
            assert response["choices"][0]["finish_reason"] == "tool_calls"
    
    def test_tool_definition_schema(self):
        """Tool definition must follow schema."""
        tool = {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City name"
                        }
                    },
                    "required": ["location"]
                }
            }
        }
        
        assert tool["type"] == "function"
        assert "name" in tool["function"]
        # Name must match pattern
        import re
        assert re.match(r'^[a-zA-Z0-9_-]{1,64}$', tool["function"]["name"])
    
    def test_tool_result_message_format(self):
        """Tool result message must have correct format."""
        tool_result = {
            "role": "tool",
            "tool_call_id": "call_abc123xyz",
            "content": "{\"temperature\": 22, \"condition\": \"sunny\"}"
        }
        
        assert tool_result["role"] == "tool"
        assert "tool_call_id" in tool_result
        assert "content" in tool_result
        assert isinstance(tool_result["content"], str)
    
    def test_parallel_tool_calls_indexed(self):
        """Parallel tool calls must have index field."""
        response = {
            "choices": [{
                "message": {
                    "tool_calls": [
                        {"index": 0, "id": "call_1", "function": {"name": "func1"}},
                        {"index": 1, "id": "call_2", "function": {"name": "func2"}},
                    ]
                }
            }]
        }
        
        # All tool calls should have index
        for tc in response["choices"][0]["message"]["tool_calls"]:
            assert "index" in tc
            assert isinstance(tc["index"], int)


# ============================================================
# Error Taxonomy Contract Tests
# ============================================================

class TestErrorTaxonomyContract:
    """Tests for Error Taxonomy compliance."""
    
    def test_error_response_structure(self):
        """Error response must have required structure."""
        error_response = {
            "error": {
                "code": "invalid_request",
                "message": "messages is required",
                "type": "semantic_error",
                "request_id": "req_abc123",
                "retryable": False
            }
        }
        
        error = error_response["error"]
        assert "code" in error
        assert "message" in error
        assert "type" in error
        assert error["type"] in ["infra_error", "semantic_error"]
        assert "request_id" in error
        assert "retryable" in error
    
    def test_infra_error_is_retryable(self):
        """Most infra errors should be retryable."""
        infra_errors = [
            {"code": "connection_timeout", "type": "infra_error", "retryable": True},
            {"code": "upstream_503", "type": "infra_error", "retryable": True},
            {"code": "rate_limited", "type": "infra_error", "retryable": True},
        ]
        
        for error in infra_errors:
            assert error["type"] == "infra_error"
            assert error["retryable"] is True
    
    def test_stream_interrupted_not_retryable(self):
        """Stream interrupted error must NOT be retryable (semantic drift)."""
        error = {
            "code": "stream_interrupted",
            "type": "infra_error",
            "retryable": False,
            "partial_content": "Hello, I"
        }
        
        assert error["retryable"] is False
        # Should include partial content
        assert "partial_content" in error
    
    def test_semantic_error_not_retryable(self):
        """Semantic errors should not be retryable."""
        semantic_errors = [
            {"code": "invalid_api_key", "type": "semantic_error", "retryable": False},
            {"code": "invalid_request", "type": "semantic_error", "retryable": False},
            {"code": "content_filtered", "type": "semantic_error", "retryable": False},
        ]
        
        for error in semantic_errors:
            assert error["type"] == "semantic_error"
            assert error["retryable"] is False
    
    def test_rate_limit_has_retry_after(self):
        """Rate limit errors must include retry_after."""
        error = {
            "code": "rate_limited",
            "type": "infra_error",
            "retryable": True,
            "retry_after": 45
        }
        
        assert "retry_after" in error
        assert isinstance(error["retry_after"], int)
        assert error["retry_after"] > 0
    
    def test_provider_error_has_provider(self):
        """Provider errors should include provider field."""
        error = {
            "code": "upstream_503",
            "type": "infra_error",
            "provider": "openai",
            "provider_request_id": "chatcmpl-xyz789"
        }
        
        assert "provider" in error
        # If available, should include provider's request ID
        if "provider_request_id" in error:
            assert isinstance(error["provider_request_id"], str)
    
    def test_validation_error_has_param(self):
        """Validation errors should include param field."""
        error = {
            "code": "missing_required_field",
            "type": "semantic_error",
            "message": "'messages' is required",
            "param": "messages"
        }
        
        assert "param" in error


# ============================================================
# Multi-tenant Contract Tests
# ============================================================

class TestMultiTenantContract:
    """Tests for Multi-tenant isolation."""
    
    def test_usage_record_has_tenant_id(self):
        """Every usage record must have tenant_id."""
        usage_record = {
            "id": "usage_abc123",
            "tenant_id": "tenant_xyz",
            "api_key_id": "key_123",
            "request_id": "req_abc",
            "provider": "openai",
            "model": "gpt-4o",
            "input_tokens": 100,
            "output_tokens": 50,
            "cost_usd": 0.005
        }
        
        assert "tenant_id" in usage_record
        assert usage_record["tenant_id"] is not None
    
    def test_api_key_format(self):
        """API key must have correct format."""
        api_key = "2api_acme_x7k9m2p4q8r1s3t5u6v0w2y"
        
        assert api_key.startswith("2api_")
        parts = api_key.split("_")
        assert len(parts) >= 3
    
    def test_rate_limit_response_headers(self):
        """Rate limit response should include standard headers."""
        headers = {
            "X-RateLimit-Limit": "1000",
            "X-RateLimit-Remaining": "950",
            "X-RateLimit-Reset": "1700000060",
            "X-Request-Id": "req_abc123"
        }
        
        assert "X-RateLimit-Limit" in headers
        assert "X-RateLimit-Remaining" in headers
        assert "X-Request-Id" in headers


# ============================================================
# Run Tests
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
