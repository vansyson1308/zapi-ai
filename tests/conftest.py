"""
2api.ai - Pytest Configuration

Configures:
- Integration test markers (skip by default)
- Smoke test handling (skip with SKIP_SMOKE=1)
- Mock providers for unit tests
"""

import os
import pytest
import logging
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock


# ============================================================
# Environment Configuration
# ============================================================

def _is_truthy(value: Optional[str]) -> bool:
    """Check if environment variable is truthy."""
    if value is None:
        return False
    return value.lower() in ("1", "true", "yes", "on")


RUN_INTEGRATION = _is_truthy(os.getenv("RUN_INTEGRATION"))
SKIP_SMOKE = _is_truthy(os.getenv("SKIP_SMOKE"))


# ============================================================
# Pytest Markers
# ============================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "integration: mark test as integration test (requires RUN_INTEGRATION=1)"
    )
    config.addinivalue_line(
        "markers",
        "smoke: mark test as smoke test (skip with SKIP_SMOKE=1)"
    )
    config.addinivalue_line(
        "markers",
        "requires_provider(name): mark test as requiring specific provider"
    )


def pytest_collection_modifyitems(config, items):
    """
    Modify test collection to handle integration and smoke tests.

    - Integration tests: Skip unless RUN_INTEGRATION=1
    - Smoke tests: Skip if SKIP_SMOKE=1
    """
    skip_integration = pytest.mark.skip(
        reason="Integration test - set RUN_INTEGRATION=1 to run"
    )
    skip_smoke = pytest.mark.skip(
        reason="Smoke test skipped - SKIP_SMOKE=1"
    )

    for item in items:
        # Handle integration tests
        if "integration" in item.keywords:
            if not RUN_INTEGRATION:
                item.add_marker(skip_integration)

        # Handle smoke tests
        if "smoke" in item.keywords:
            if SKIP_SMOKE:
                item.add_marker(skip_smoke)


# ============================================================
# Mock Providers (for unit tests)
# ============================================================

@pytest.fixture
def mock_openai_response():
    """Standard mock OpenAI chat response."""
    return {
        "id": "chatcmpl-test123",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "gpt-4o",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello! I'm a mock response."
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 8,
            "total_tokens": 18
        }
    }


@pytest.fixture
def mock_anthropic_response():
    """Standard mock Anthropic response."""
    return {
        "id": "msg-test123",
        "type": "message",
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": "Hello! I'm a mock Claude response."
            }
        ],
        "model": "claude-3-5-sonnet-20241022",
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": 10,
            "output_tokens": 8
        }
    }


@pytest.fixture
def mock_error_500():
    """Mock 500 error response."""
    return {
        "error": {
            "code": "internal_error",
            "message": "Internal server error",
            "type": "server_error"
        }
    }


@pytest.fixture
def mock_error_429():
    """Mock 429 rate limit response."""
    return {
        "error": {
            "code": "rate_limit_exceeded",
            "message": "Rate limit exceeded",
            "type": "rate_limit_error"
        }
    }


# ============================================================
# Mock HTTP Client
# ============================================================

class MockHttpResponse:
    """Mock HTTP response for testing."""

    def __init__(
        self,
        status_code: int = 200,
        json_data: Optional[Dict[str, Any]] = None,
        text: str = "",
        headers: Optional[Dict[str, str]] = None
    ):
        self.status_code = status_code
        self._json_data = json_data
        self.text = text or (str(json_data) if json_data else "")
        self.headers = headers or {}

    def json(self):
        if self._json_data is None:
            raise ValueError("No JSON data")
        return self._json_data


@pytest.fixture
def mock_http_client():
    """
    Mock HTTP client that doesn't make real network calls.

    Usage:
        def test_something(mock_http_client):
            mock_http_client.set_response(200, {"data": "test"})
            # Now any HTTP call will return the mocked response
    """
    class MockClient:
        def __init__(self):
            self.responses = []
            self.requests = []
            self.default_response = MockHttpResponse(200, {"status": "ok"})

        def set_response(
            self,
            status_code: int = 200,
            json_data: Optional[Dict] = None,
            headers: Optional[Dict] = None
        ):
            """Set the next response to return."""
            self.responses.append(MockHttpResponse(
                status_code=status_code,
                json_data=json_data,
                headers=headers
            ))

        def set_responses(self, responses: list):
            """Set multiple responses (for retry testing)."""
            for r in responses:
                self.set_response(**r)

        async def request(self, method: str, url: str, **kwargs):
            """Mock request method."""
            self.requests.append({
                "method": method,
                "url": url,
                **kwargs
            })
            if self.responses:
                return self.responses.pop(0)
            return self.default_response

        def get_requests(self):
            """Get all recorded requests."""
            return self.requests

        def clear(self):
            """Clear all responses and requests."""
            self.responses = []
            self.requests = []

    return MockClient()


# ============================================================
# Fake Provider Server (for unit tests)
# ============================================================

@pytest.fixture
def fake_provider_responses():
    """
    Configurable fake provider responses.

    Usage:
        def test_chat(fake_provider_responses):
            fake_provider_responses["openai"] = {"id": "test", ...}
    """
    return {
        "openai": None,
        "anthropic": None,
        "google": None
    }


# ============================================================
# Logging Configuration
# ============================================================

@pytest.fixture(autouse=True)
def configure_test_logging():
    """Configure logging for tests."""
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    yield


# ============================================================
# Skip Helpers
# ============================================================

requires_integration = pytest.mark.skipif(
    not RUN_INTEGRATION,
    reason="Requires RUN_INTEGRATION=1"
)

skip_if_no_openai = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="Requires OPENAI_API_KEY"
)

skip_if_no_anthropic = pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="Requires ANTHROPIC_API_KEY"
)

skip_if_no_google = pytest.mark.skipif(
    not os.getenv("GOOGLE_API_KEY"),
    reason="Requires GOOGLE_API_KEY"
)
