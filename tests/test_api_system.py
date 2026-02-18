"""
2api.ai - API Layer Tests

Comprehensive tests for:
- Request/Response models
- API dependencies
- Middleware
- Route handlers
"""

import pytest
import json
import time
import asyncio
from typing import Any, Dict, List, Optional
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from pydantic import ValidationError


# ============================================================
# Test API Models
# ============================================================

class TestMessageInput:
    """Tests for MessageInput model."""

    def test_create_simple_message(self):
        """Test creating a simple message."""
        from src.api.models import MessageInput, RoleEnum

        msg = MessageInput(role=RoleEnum.USER, content="Hello")
        assert msg.role == RoleEnum.USER
        assert msg.content == "Hello"
        assert msg.name is None

    def test_create_system_message(self):
        """Test creating a system message."""
        from src.api.models import MessageInput, RoleEnum

        msg = MessageInput(role=RoleEnum.SYSTEM, content="You are helpful")
        assert msg.role == RoleEnum.SYSTEM
        assert msg.content == "You are helpful"

    def test_create_assistant_message(self):
        """Test creating an assistant message."""
        from src.api.models import MessageInput, RoleEnum

        msg = MessageInput(role=RoleEnum.ASSISTANT, content="Hello!")
        assert msg.role == RoleEnum.ASSISTANT
        assert msg.content == "Hello!"

    def test_create_message_with_name(self):
        """Test message with name field."""
        from src.api.models import MessageInput, RoleEnum

        msg = MessageInput(role=RoleEnum.USER, content="Hi", name="john")
        assert msg.name == "john"

    def test_create_tool_message(self):
        """Test creating a tool result message."""
        from src.api.models import MessageInput, RoleEnum

        msg = MessageInput(
            role=RoleEnum.TOOL,
            content='{"result": 42}',
            tool_call_id="call_123"
        )
        assert msg.role == RoleEnum.TOOL
        assert msg.tool_call_id == "call_123"


class TestChatCompletionRequest:
    """Tests for ChatCompletionRequest model."""

    def test_minimal_request(self):
        """Test minimal valid request."""
        from src.api.models import ChatCompletionRequest, MessageInput, RoleEnum

        request = ChatCompletionRequest(
            model="openai/gpt-4o",
            messages=[MessageInput(role=RoleEnum.USER, content="Hi")]
        )
        assert request.model == "openai/gpt-4o"
        assert len(request.messages) == 1
        assert request.stream is False

    def test_request_with_temperature(self):
        """Test request with temperature."""
        from src.api.models import ChatCompletionRequest, MessageInput, RoleEnum

        request = ChatCompletionRequest(
            model="openai/gpt-4o",
            messages=[MessageInput(role=RoleEnum.USER, content="Hi")],
            temperature=0.7
        )
        assert request.temperature == 0.7

    def test_request_with_max_tokens(self):
        """Test request with max_tokens."""
        from src.api.models import ChatCompletionRequest, MessageInput, RoleEnum

        request = ChatCompletionRequest(
            model="openai/gpt-4o",
            messages=[MessageInput(role=RoleEnum.USER, content="Hi")],
            max_tokens=100
        )
        assert request.max_tokens == 100

    def test_request_with_streaming(self):
        """Test streaming request."""
        from src.api.models import ChatCompletionRequest, MessageInput, RoleEnum

        request = ChatCompletionRequest(
            model="openai/gpt-4o",
            messages=[MessageInput(role=RoleEnum.USER, content="Hi")],
            stream=True
        )
        assert request.stream is True

    def test_request_with_tools(self):
        """Test request with tools."""
        from src.api.models import (
            ChatCompletionRequest,
            MessageInput,
            RoleEnum,
            ToolDefinition,
            FunctionDefinition,
        )

        request = ChatCompletionRequest(
            model="openai/gpt-4o",
            messages=[MessageInput(role=RoleEnum.USER, content="Hi")],
            tools=[
                ToolDefinition(
                    type="function",
                    function=FunctionDefinition(
                        name="get_weather",
                        description="Get current weather",
                        parameters={"type": "object", "properties": {}}
                    )
                )
            ]
        )
        assert len(request.tools) == 1
        assert request.tools[0].function.name == "get_weather"

    def test_request_with_routing(self):
        """Test request with routing config."""
        from src.api.models import (
            ChatCompletionRequest,
            MessageInput,
            RoleEnum,
            RoutingConfig,
        )

        request = ChatCompletionRequest(
            model="openai/gpt-4o",
            messages=[MessageInput(role=RoleEnum.USER, content="Hi")],
            routing=RoutingConfig(
                strategy="cost",
                fallback=["anthropic/claude-3-sonnet"],
                max_latency_ms=5000
            )
        )
        assert request.routing.strategy == "cost"
        assert len(request.routing.fallback) == 1


class TestEmbeddingRequest:
    """Tests for EmbeddingRequest model."""

    def test_single_input(self):
        """Test embedding with single string input."""
        from src.api.models import EmbeddingRequest

        request = EmbeddingRequest(
            model="openai/text-embedding-3-small",
            input="Hello world"
        )
        assert request.model == "openai/text-embedding-3-small"
        assert request.input == "Hello world"

    def test_batch_input(self):
        """Test embedding with multiple inputs."""
        from src.api.models import EmbeddingRequest

        request = EmbeddingRequest(
            model="openai/text-embedding-3-small",
            input=["Hello", "World"]
        )
        assert isinstance(request.input, list)
        assert len(request.input) == 2

    def test_with_dimensions(self):
        """Test embedding with custom dimensions."""
        from src.api.models import EmbeddingRequest

        request = EmbeddingRequest(
            model="openai/text-embedding-3-small",
            input="Hello",
            dimensions=512
        )
        assert request.dimensions == 512


class TestImageGenerationRequest:
    """Tests for ImageGenerationRequest model."""

    def test_minimal_request(self):
        """Test minimal image generation request."""
        from src.api.models import ImageGenerationRequest

        request = ImageGenerationRequest(
            prompt="A cat"
        )
        assert request.prompt == "A cat"
        assert request.model == "dall-e-3"  # default
        assert request.n == 1

    def test_with_size(self):
        """Test request with size."""
        from src.api.models import ImageGenerationRequest

        request = ImageGenerationRequest(
            prompt="A cat",
            size="1024x1792"
        )
        assert request.size == "1024x1792"

    def test_with_quality(self):
        """Test request with quality setting."""
        from src.api.models import ImageGenerationRequest

        request = ImageGenerationRequest(
            prompt="A cat",
            quality="hd"
        )
        assert request.quality == "hd"


class TestModelInfo:
    """Tests for ModelInfo response model."""

    def test_create_model_info(self):
        """Test creating model info."""
        from src.api.models import ModelInfo, ModelPricing

        info = ModelInfo(
            id="openai/gpt-4o",
            object="model",
            provider="openai",
            name="GPT-4o",
            capabilities=["chat", "vision"],
            context_window=128000,
            max_output_tokens=4096,
            pricing=ModelPricing(
                input_per_1m_tokens=2.5,
                output_per_1m_tokens=10.0
            )
        )
        assert info.id == "openai/gpt-4o"
        assert info.provider == "openai"
        assert "chat" in info.capabilities


# ============================================================
# Test API Dependencies
# ============================================================

class TestGetRouter:
    """Tests for get_router dependency."""

    def test_router_not_initialized(self):
        """Test error when router not initialized."""
        from src.api.dependencies import get_router, _router_instance_getter
        from src.core.errors import InfraError

        # Temporarily clear the getter
        import src.api.dependencies as deps
        original = deps._router_instance_getter
        deps._router_instance_getter = None

        try:
            with pytest.raises(InfraError) as exc_info:
                get_router()
            assert exc_info.value.error.code == "service_unavailable"
        finally:
            deps._router_instance_getter = original


class TestAddStandardHeaders:
    """Tests for add_standard_headers helper."""

    def _create_auth_context(self, request_id="req_test123", trace_id="trace_test123"):
        """Helper to create a valid AuthContext."""
        from src.db.models import AuthContext, Tenant, APIKey
        import uuid
        from datetime import datetime

        tenant_id = uuid.uuid4()
        api_key_id = uuid.uuid4()

        tenant = Tenant(
            id=tenant_id,
            name="Test Tenant",
            email="test@example.com",
            plan="free",
            is_active=True,
            created_at=datetime.utcnow()
        )
        api_key = APIKey(
            id=api_key_id,
            tenant_id=tenant_id,
            key_hash="hash123",
            key_prefix="2api_test",
            name="Test Key",
            permissions=["*"],
            rate_limit_per_minute=100,
            is_active=True,
            created_at=datetime.utcnow()
        )

        return AuthContext(
            tenant_id=tenant_id,
            api_key_id=api_key_id,
            tenant=tenant,
            api_key=api_key,
            permissions=["*"],
            rate_limit_per_minute=100,
            request_id=request_id,
            trace_id=trace_id
        )

    def test_adds_request_id(self):
        """Test that request ID is added."""
        from src.api.dependencies import add_standard_headers

        auth = self._create_auth_context()
        headers = add_standard_headers({}, auth)
        assert headers["X-Request-Id"] == "req_test123"
        assert headers["X-Trace-Id"] == "trace_test123"

    def test_adds_extra_headers(self):
        """Test that extra headers are added."""
        from src.api.dependencies import add_standard_headers

        auth = self._create_auth_context()
        headers = add_standard_headers(
            {},
            auth,
            **{"X-Provider": "openai", "X-Latency-Ms": 100}
        )
        assert headers["X-Provider"] == "openai"
        assert headers["X-Latency-Ms"] == "100"


class TestRequestTimer:
    """Tests for RequestTimer context manager."""

    @pytest.mark.asyncio
    async def test_measures_elapsed_time(self):
        """Test that elapsed time is measured."""
        from src.api.dependencies import RequestTimer

        async with RequestTimer() as timer:
            await asyncio.sleep(0.01)

        assert timer.elapsed_ms >= 10  # At least 10ms

    @pytest.mark.asyncio
    async def test_records_first_token(self):
        """Test first token timing."""
        from src.api.dependencies import RequestTimer

        async with RequestTimer() as timer:
            await asyncio.sleep(0.005)
            timer.record_first_token()
            await asyncio.sleep(0.005)

        assert timer.ttft_ms is not None
        assert timer.ttft_ms >= 5


class TestStartRequestTracking:
    """Tests for start_request_tracking helper."""

    def _create_auth_context(self, request_id="req_test123"):
        """Helper to create a valid AuthContext."""
        from src.db.models import AuthContext, Tenant, APIKey
        import uuid
        from datetime import datetime

        tenant_id = uuid.uuid4()
        api_key_id = uuid.uuid4()

        tenant = Tenant(
            id=tenant_id,
            name="Test Tenant",
            email="test@example.com",
            plan="free",
            is_active=True,
            created_at=datetime.utcnow()
        )
        api_key = APIKey(
            id=api_key_id,
            tenant_id=tenant_id,
            key_hash="hash123",
            key_prefix="2api_test",
            name="Test Key",
            permissions=["*"],
            rate_limit_per_minute=100,
            is_active=True,
            created_at=datetime.utcnow()
        )

        return AuthContext(
            tenant_id=tenant_id,
            api_key_id=api_key_id,
            tenant=tenant,
            api_key=api_key,
            permissions=["*"],
            rate_limit_per_minute=100,
            request_id=request_id,
            trace_id=f"trace_{request_id[4:]}"
        )

    def test_creates_tracker(self):
        """Test that a tracker is created."""
        from src.api.dependencies import start_request_tracking
        from src.usage import OperationType

        auth = self._create_auth_context()
        tracker = start_request_tracking(auth, "openai/gpt-4o", OperationType.CHAT)
        assert tracker.request_id == "req_test123"
        assert tracker.model == "openai/gpt-4o"
        assert tracker.provider == "openai"


class TestValidateModelAccess:
    """Tests for validate_model_access helper."""

    def _create_auth_context(self, request_id="req_test123"):
        """Helper to create a valid AuthContext."""
        from src.db.models import AuthContext, Tenant, APIKey
        import uuid
        from datetime import datetime

        tenant_id = uuid.uuid4()
        api_key_id = uuid.uuid4()

        tenant = Tenant(
            id=tenant_id,
            name="Test Tenant",
            email="test@example.com",
            plan="free",
            is_active=True,
            created_at=datetime.utcnow()
        )
        api_key = APIKey(
            id=api_key_id,
            tenant_id=tenant_id,
            key_hash="hash123",
            key_prefix="2api_test",
            name="Test Key",
            permissions=["*"],
            rate_limit_per_minute=100,
            is_active=True,
            created_at=datetime.utcnow()
        )

        return AuthContext(
            tenant_id=tenant_id,
            api_key_id=api_key_id,
            tenant=tenant,
            api_key=api_key,
            permissions=["*"],
            rate_limit_per_minute=100,
            request_id=request_id,
            trace_id=f"trace_{request_id[4:]}"
        )

    def test_invalid_provider_raises(self):
        """Test that invalid provider raises error."""
        from src.api.dependencies import validate_model_access
        from src.core.errors import SemanticError

        auth = self._create_auth_context()

        # Create mock router
        mock_router = Mock()
        mock_router.adapters = {}

        with pytest.raises(SemanticError) as exc_info:
            validate_model_access("invalid/model", auth, mock_router)
        assert "Unknown provider" in exc_info.value.error.message


# ============================================================
# Test API Middleware
# ============================================================

class TestRequestIdMiddleware:
    """Tests for RequestIdMiddleware."""

    def test_middleware_creates_request_id(self):
        """Test that middleware creates request ID if missing."""
        from src.api.middleware import RequestIdMiddleware

        app = FastAPI()
        app.add_middleware(RequestIdMiddleware)

        @app.get("/test")
        async def test_endpoint(request: Request):
            return {"request_id": request.state.request_id}

        client = TestClient(app)
        response = client.get("/test")

        assert response.status_code == 200
        data = response.json()
        assert "request_id" in data
        assert data["request_id"].startswith("req_")


class TestMetricsMiddleware:
    """Tests for MetricsMiddleware."""

    def test_middleware_tracks_timing(self):
        """Test that middleware tracks request timing."""
        from src.api.middleware import MetricsMiddleware

        app = FastAPI()
        app.add_middleware(MetricsMiddleware)

        @app.get("/test")
        async def test_endpoint():
            return {"status": "ok"}

        client = TestClient(app)
        response = client.get("/test")

        assert response.status_code == 200
        # Metrics are collected internally


# ============================================================
# Test Chat Routes
# ============================================================

class TestChatRouteConversion:
    """Tests for chat route message conversion."""

    def test_convert_message(self):
        """Test message conversion."""
        from src.api.routes.chat import convert_message
        from src.api.models import MessageInput, RoleEnum

        msg = MessageInput(role=RoleEnum.USER, content="Hello")
        internal = convert_message(msg)

        assert internal.role.value == "user"
        assert internal.content == "Hello"

    def test_convert_tool(self):
        """Test tool conversion."""
        from src.api.routes.chat import convert_tool
        from src.api.models import ToolDefinition, FunctionDefinition

        tool = ToolDefinition(
            type="function",
            function=FunctionDefinition(
                name="get_weather",
                description="Get weather",
                parameters={"type": "object"}
            )
        )
        internal = convert_tool(tool)

        assert internal.function.name == "get_weather"
        assert internal.function.description == "Get weather"

    def test_convert_routing(self):
        """Test routing conversion."""
        from src.api.routes.chat import convert_routing
        from src.api.models import RoutingConfig

        routing = RoutingConfig(
            strategy="cost",
            fallback=["anthropic/claude-3-sonnet"],
            max_latency_ms=5000
        )
        internal = convert_routing(routing)

        assert internal.strategy.value == "cost"
        assert internal.max_latency_ms == 5000


class TestGetAdapterForModel:
    """Tests for _get_adapter_for_model helper."""

    def test_extracts_provider_from_model(self):
        """Test provider extraction from model string."""
        from src.api.routes.chat import _get_adapter_for_model
        from src.core.models import Provider
        from src.core.errors import SemanticError

        # Create mock router
        mock_router = Mock()
        mock_adapter = Mock()
        mock_router.adapters = {Provider.OPENAI: mock_adapter}

        result = _get_adapter_for_model("openai/gpt-4o", mock_router)
        assert result == mock_adapter

    def test_defaults_to_openai(self):
        """Test that it defaults to OpenAI for unknown models."""


# ============================================================
# API Correctness Regression Tests
# ============================================================

class TestApiCorrectnessRegression:
    """Regression tests for route correctness conflicts."""

    @staticmethod
    def _mock_auth_context():
        """Create a deterministic auth context for route tests."""
        from datetime import datetime
        from uuid import UUID
        from src.db.models import AuthContext, Tenant, APIKey

        tenant_id = UUID("11111111-1111-1111-1111-111111111111")
        api_key_id = UUID("22222222-2222-2222-2222-222222222222")

        tenant = Tenant(
            id=tenant_id,
            name="Test Tenant",
            email="test@example.com",
            plan="pro",
            is_active=True,
            created_at=datetime.utcnow(),
        )
        api_key = APIKey(
            id=api_key_id,
            tenant_id=tenant_id,
            key_hash="hash",
            key_prefix="2api_test",
            name="Test Key",
            permissions=["*"],
            rate_limit_per_minute=100,
            is_active=True,
            created_at=datetime.utcnow(),
        )

        return AuthContext(
            tenant_id=tenant_id,
            api_key_id=api_key_id,
            tenant=tenant,
            api_key=api_key,
            permissions=["*"],
            rate_limit_per_minute=100,
            request_id="req_test_models_compare",
            trace_id="trace_test_models_compare",
        )

    def test_models_compare_route_not_captured_by_model_path(self):
        """/v1/models/compare must resolve to compare handler, not model lookup."""
        from src.api.routes.models import router as models_router
        from src.auth.middleware import get_auth_context
        from src.api.dependencies import get_router

        app = FastAPI()
        app.include_router(models_router)
        app.dependency_overrides[get_auth_context] = self._mock_auth_context

        mock_router = Mock()
        mock_router.list_all_models.return_value = []
        app.dependency_overrides[get_router] = lambda: mock_router

        client = TestClient(app)
        response = client.get(
            "/v1/models/compare",
            params={"input_tokens": 1000, "output_tokens": 500, "capability": "chat"},
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["object"] == "list"
        assert "query" in payload
        assert payload["query"]["input_tokens"] == 1000

    def test_usage_route_contract_is_unique_on_server_app(self):
        """Server app should expose exactly one /v1/usage route with stable contract."""
        from src.server import app

        usage_routes = [
            route
            for route in app.router.routes
            if getattr(route, "path", None) == "/v1/usage"
        ]

        assert len(usage_routes) == 1

        admin_usage_routes = [
            route
            for route in app.router.routes
            if getattr(route, "path", None) == "/v1/admin/usage"
        ]
        assert len(admin_usage_routes) == 1
        from src.api.routes.chat import _get_adapter_for_model
        from src.core.models import Provider

        mock_router = Mock()
        mock_adapter = Mock()
        mock_router.adapters = {Provider.OPENAI: mock_adapter}

        result = _get_adapter_for_model("gpt-4o", mock_router)
        assert result == mock_adapter

    def test_raises_when_no_providers(self):
        """Test error when no providers configured."""
        from src.api.routes.chat import _get_adapter_for_model
        from src.core.errors import SemanticError

        mock_router = Mock()
        mock_router.adapters = {}

        with pytest.raises(SemanticError) as exc_info:
            _get_adapter_for_model("openai/gpt-4o", mock_router)
        assert exc_info.value.error.code == "no_providers"


# ============================================================
# Test Model Routes
# ============================================================

class TestModelsRouteHelpers:
    """Tests for model route helpers."""

    def test_provider_extraction(self):
        """Test extracting provider from model string."""
        from src.api.routes.embeddings import _get_provider_from_model

        assert _get_provider_from_model("openai/text-embedding-3-small") == "openai"
        assert _get_provider_from_model("anthropic/claude-3-sonnet") == "anthropic"
        assert _get_provider_from_model("gpt-4o") == "openai"  # default


# ============================================================
# Test Response Models
# ============================================================

class TestChatCompletionResponse:
    """Tests for ChatCompletionResponse model."""

    def test_create_response(self):
        """Test creating a response."""
        from src.api.models import (
            ChatCompletionResponse,
            Choice,
            MessageOutput,
            UsageInfo,
            RoleEnum,
        )

        response = ChatCompletionResponse(
            id="chatcmpl-123",
            object="chat.completion",
            created=1234567890,
            model="gpt-4o",
            choices=[
                Choice(
                    index=0,
                    message=MessageOutput(role=RoleEnum.ASSISTANT, content="Hello!"),
                    finish_reason="stop"
                )
            ],
            usage=UsageInfo(
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15
            )
        )

        assert response.id == "chatcmpl-123"
        assert len(response.choices) == 1
        assert response.choices[0].message.content == "Hello!"


class TestUsageInfo:
    """Tests for UsageInfo model."""

    def test_create_usage_info(self):
        """Test creating usage info."""
        from src.api.models import UsageInfo

        usage = UsageInfo(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150
        )
        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50
        assert usage.total_tokens == 150


# ============================================================
# Test Integration
# ============================================================

class TestAPIIntegration:
    """Integration tests for API layer."""

    def test_request_model_validation(self):
        """Test that request models validate properly."""
        from src.api.models import ChatCompletionRequest, MessageInput, RoleEnum

        # Valid request
        request = ChatCompletionRequest(
            model="openai/gpt-4o",
            messages=[MessageInput(role=RoleEnum.USER, content="Hi")]
        )
        assert request.model == "openai/gpt-4o"

    def test_response_model_serialization(self):
        """Test that response models serialize properly."""
        from src.api.models import (
            ChatCompletionResponse,
            Choice,
            MessageOutput,
            UsageInfo,
            RoleEnum,
        )

        response = ChatCompletionResponse(
            id="chatcmpl-123",
            object="chat.completion",
            created=1234567890,
            model="gpt-4o",
            choices=[
                Choice(
                    index=0,
                    message=MessageOutput(role=RoleEnum.ASSISTANT, content="Hello!"),
                    finish_reason="stop"
                )
            ],
            usage=UsageInfo(
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15
            )
        )

        data = response.model_dump()
        assert data["id"] == "chatcmpl-123"
        assert data["choices"][0]["message"]["content"] == "Hello!"

    def test_tool_definition_format(self):
        """Test tool definition matches OpenAI format."""
        from src.api.models import ToolDefinition, FunctionDefinition

        tool = ToolDefinition(
            type="function",
            function=FunctionDefinition(
                name="get_weather",
                description="Get the current weather",
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city name"
                        }
                    },
                    "required": ["location"]
                }
            )
        )

        data = tool.model_dump()
        assert data["type"] == "function"
        assert data["function"]["name"] == "get_weather"
        assert "properties" in data["function"]["parameters"]


class TestAPIModuleExports:
    """Test that API module exports all expected symbols."""

    def test_routers_exported(self):
        """Test that routers are exported."""
        from src.api import (
            management_router,
            chat_router,
            embeddings_router,
            images_router,
            models_router,
        )
        assert management_router is not None
        assert chat_router is not None
        assert embeddings_router is not None
        assert images_router is not None
        assert models_router is not None

    def test_models_exported(self):
        """Test that models are exported."""
        from src.api import (
            ChatCompletionRequest,
            EmbeddingRequest,
            ImageGenerationRequest,
            ChatCompletionResponse,
            EmbeddingResponse,
            ImageGenerationResponse,
        )
        assert ChatCompletionRequest is not None
        assert EmbeddingRequest is not None
        assert ImageGenerationRequest is not None

    def test_middleware_exported(self):
        """Test that middleware is exported."""
        from src.api import (
            RequestLoggingMiddleware,
            MetricsMiddleware,
            RequestIdMiddleware,
            ErrorTrackingMiddleware,
        )
        assert RequestLoggingMiddleware is not None
        assert MetricsMiddleware is not None

    def test_dependencies_exported(self):
        """Test that dependencies are exported."""
        from src.api import (
            get_router,
            add_standard_headers,
            start_request_tracking,
            RequestTimer,
        )
        assert get_router is not None
        assert add_standard_headers is not None
        assert RequestTimer is not None


class TestHelperFunctions:
    """Test utility functions."""

    def test_generate_request_id(self):
        """Test request ID generation."""
        from src.api.dependencies import generate_request_id

        id1 = generate_request_id()
        id2 = generate_request_id()

        assert id1.startswith("req_")
        assert id2.startswith("req_")
        assert id1 != id2

    def test_generate_trace_id(self):
        """Test trace ID generation from request ID."""
        from src.api.dependencies import generate_trace_id

        request_id = "req_abc123"
        trace_id = generate_trace_id(request_id)

        assert trace_id == "trace_abc123"
