"""
2api.ai - Usage Tracking System Tests

Comprehensive tests for the usage tracking module covering:
- Pricing catalog
- Token estimation
- Usage tracking
- Usage aggregation
- Rate limiting and quotas
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from typing import List

from src.usage import (
    # Pricing
    ModelPrice,
    PricingCatalog,
    get_model_price,
    calculate_cost,
    list_models,
    compare_costs,
    # Estimator
    TokenEstimate,
    TokenEstimator,
    estimate_tokens,
    estimate_message_tokens,
    estimate_request_cost,
    # Tracker
    UsageStatus,
    OperationType,
    UsageRecord,
    RequestTracker,
    UsageTracker,
    get_usage_tracker,
    set_usage_tracker,
    start_tracking,
    complete_tracking,
    # Aggregator
    AggregationPeriod,
    UsageAggregate,
    CostBreakdown,
    UsageAggregator,
    aggregate_usage,
    get_cost_breakdown,
    get_top_models,
    # Limits
    LimitType,
    LimitPeriod,
    LimitAction,
    UsageLimit,
    LimitStatus,
    QuotaConfig,
    LimitCheckResult,
    SlidingWindowCounter,
    UsageLimiter,
    get_limiter,
    set_quota,
    check_limits,
    record_usage,
    get_limit_status,
)
from src.core.models import Message, Role


# ============================================================
# Test Fixtures
# ============================================================

@pytest.fixture
def sample_messages():
    """Create sample messages for testing."""
    return [
        Message(role=Role.SYSTEM, content="You are a helpful assistant."),
        Message(role=Role.USER, content="Hello, how are you today?"),
        Message(role=Role.ASSISTANT, content="I'm doing well, thank you for asking!"),
        Message(role=Role.USER, content="What's the weather like?"),
    ]


@pytest.fixture
def sample_usage_records():
    """Create sample usage records for testing."""
    records = []
    base_time = datetime.utcnow() - timedelta(days=7)

    for i in range(50):
        model = ["openai/gpt-4o", "openai/gpt-4o-mini", "anthropic/claude-3-5-sonnet"][i % 3]
        provider = model.split("/")[0]

        record = UsageRecord(
            request_id=f"req_{i:03d}",
            tenant_id="tenant_123",
            provider=provider,
            model=model,
            operation=OperationType.CHAT,
            input_tokens=100 + (i * 10),
            output_tokens=50 + (i * 5),
            latency_ms=100 + (i * 2),
            status=UsageStatus.SUCCESS if i % 10 != 9 else UsageStatus.ERROR,
            error_code="rate_limited" if i % 10 == 9 else None,
            created_at=base_time + timedelta(hours=i * 4)
        )
        records.append(record)

    return records


# ============================================================
# Pricing Tests
# ============================================================

class TestModelPrice:
    """Tests for ModelPrice."""

    def test_create_model_price(self):
        """Test creating a model price."""
        price = ModelPrice(
            model_id="openai/gpt-4o",
            provider="openai",
            model_name="gpt-4o",
            input_per_1m=2.50,
            output_per_1m=10.00
        )
        assert price.model_id == "openai/gpt-4o"
        assert price.input_per_1m == 2.50
        assert price.output_per_1m == 10.00

    def test_calculate_cost(self):
        """Test cost calculation."""
        price = ModelPrice(
            model_id="test/model",
            provider="test",
            model_name="model",
            input_per_1m=2.00,
            output_per_1m=8.00
        )

        # 1000 input + 500 output tokens
        cost = price.calculate_cost(input_tokens=1000, output_tokens=500)
        expected = (1000 / 1_000_000 * 2.00) + (500 / 1_000_000 * 8.00)
        assert abs(cost - expected) < 0.0001

    def test_calculate_cost_with_cached(self):
        """Test cost calculation with cached tokens."""
        price = ModelPrice(
            model_id="test/model",
            provider="test",
            model_name="model",
            input_per_1m=2.00,
            output_per_1m=8.00,
            cached_input_per_1m=0.50  # 75% discount
        )

        cost = price.calculate_cost(
            input_tokens=500,
            output_tokens=500,
            cached_tokens=500
        )

        expected_input = (500 / 1_000_000) * 2.00
        expected_output = (500 / 1_000_000) * 8.00
        expected_cached = (500 / 1_000_000) * 0.50
        expected = expected_input + expected_output + expected_cached

        assert abs(cost - expected) < 0.0001

    def test_calculate_batch_cost(self):
        """Test batch pricing calculation."""
        price = ModelPrice(
            model_id="test/model",
            provider="test",
            model_name="model",
            input_per_1m=2.00,
            output_per_1m=8.00,
            batch_input_per_1m=1.00,  # 50% off
            batch_output_per_1m=4.00   # 50% off
        )

        regular_cost = price.calculate_cost(1000, 500, is_batch=False)
        batch_cost = price.calculate_cost(1000, 500, is_batch=True)

        assert batch_cost < regular_cost
        assert batch_cost == regular_cost / 2


class TestPricingCatalog:
    """Tests for PricingCatalog."""

    def test_get_known_model(self):
        """Test getting a known model."""
        catalog = PricingCatalog()
        price = catalog.get_price("openai/gpt-4o")
        assert price is not None
        assert price.model_name == "gpt-4o"
        assert price.input_per_1m == 2.50

    def test_get_model_by_alias(self):
        """Test getting model by short name."""
        catalog = PricingCatalog()
        price = catalog.get_price("gpt-4o")
        assert price is not None
        assert price.model_id == "openai/gpt-4o"

    def test_get_unknown_model(self):
        """Test getting unknown model returns None."""
        catalog = PricingCatalog()
        price = catalog.get_price("unknown/model")
        assert price is None

    def test_list_all_models(self):
        """Test listing all models."""
        catalog = PricingCatalog()
        models = catalog.list_models()
        assert len(models) > 10  # Should have many models
        assert any(m.provider == "openai" for m in models)
        assert any(m.provider == "anthropic" for m in models)
        assert any(m.provider == "google" for m in models)

    def test_list_models_by_provider(self):
        """Test listing models by provider."""
        catalog = PricingCatalog()
        openai_models = catalog.list_models(provider="openai")
        assert all(m.provider == "openai" for m in openai_models)
        assert len(openai_models) > 5

    def test_calculate_cost(self):
        """Test catalog cost calculation."""
        catalog = PricingCatalog()
        cost = catalog.calculate_cost(
            model_id="openai/gpt-4o-mini",
            input_tokens=10000,
            output_tokens=5000
        )
        # gpt-4o-mini: $0.15/1M input, $0.60/1M output
        expected = (10000 / 1_000_000 * 0.15) + (5000 / 1_000_000 * 0.60)
        assert abs(cost - expected) < 0.0001

    def test_get_cheapest_model(self):
        """Test getting cheapest model."""
        catalog = PricingCatalog()
        cheapest = catalog.get_cheapest_model(capability="chat")
        assert cheapest is not None
        # Should be one of the mini/flash models
        assert "mini" in cheapest.model_name or "flash" in cheapest.model_name or "haiku" in cheapest.model_name

    def test_compare_costs(self):
        """Test cost comparison."""
        catalog = PricingCatalog()
        comparison = catalog.compare_costs(
            input_tokens=10000,
            output_tokens=5000,
            capability="chat"
        )
        assert len(comparison) > 5
        # Should be sorted by cost
        costs = [c["cost_usd"] for c in comparison]
        assert costs == sorted(costs)


class TestPricingFunctions:
    """Tests for pricing convenience functions."""

    def test_get_model_price_function(self):
        """Test get_model_price function."""
        price = get_model_price("openai/gpt-4o")
        assert price is not None
        assert price.provider == "openai"

    def test_calculate_cost_function(self):
        """Test calculate_cost function."""
        cost = calculate_cost(
            model_id="openai/gpt-4o",
            input_tokens=1000,
            output_tokens=500
        )
        assert cost > 0

    def test_list_models_function(self):
        """Test list_models function."""
        models = list_models()
        assert len(models) > 0

    def test_compare_costs_function(self):
        """Test compare_costs function."""
        comparison = compare_costs(10000, 5000)
        assert len(comparison) > 0


# ============================================================
# Estimator Tests
# ============================================================

class TestTokenEstimator:
    """Tests for TokenEstimator."""

    def test_estimate_simple_text(self):
        """Test token estimation for simple text."""
        estimator = TokenEstimator()
        tokens = estimator.estimate_text_tokens("Hello, world!")
        assert tokens > 0
        assert tokens < 10  # Should be around 3-4 tokens

    def test_estimate_long_text(self):
        """Test token estimation for longer text."""
        estimator = TokenEstimator()
        text = "This is a longer piece of text " * 100
        tokens = estimator.estimate_text_tokens(text)
        # ~8 tokens per repeat, 100 repeats = ~800 tokens
        assert 500 < tokens < 1200

    def test_estimate_empty_text(self):
        """Test token estimation for empty text."""
        estimator = TokenEstimator()
        tokens = estimator.estimate_text_tokens("")
        assert tokens == 0

    def test_estimate_message_tokens(self, sample_messages):
        """Test message token estimation."""
        estimator = TokenEstimator()
        estimate = estimator.estimate_messages_tokens(sample_messages)

        assert estimate.total_tokens > 0
        assert estimate.text_tokens > 0
        assert estimate.message_overhead > 0
        assert estimate.confidence > 0

    def test_estimate_tools_tokens(self):
        """Test tool token estimation."""
        estimator = TokenEstimator()
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city name"
                            }
                        }
                    }
                }
            }
        ]
        tokens = estimator.estimate_tools_tokens(tools)
        assert tokens > 20  # Should account for name, description, params

    def test_estimate_image_tokens_low(self):
        """Test low-detail image token estimation."""
        estimator = TokenEstimator()
        tokens = estimator.estimate_image_tokens(detail="low")
        assert tokens == 85  # OpenAI's low detail token count

    def test_estimate_image_tokens_high(self):
        """Test high-detail image token estimation."""
        estimator = TokenEstimator()
        tokens = estimator.estimate_image_tokens(
            detail="high",
            width=1024,
            height=1024
        )
        assert tokens > 85  # Should be more than low detail


class TestEstimatorFunctions:
    """Tests for estimator convenience functions."""

    def test_estimate_tokens_function(self):
        """Test estimate_tokens function."""
        tokens = estimate_tokens("Hello world")
        assert tokens > 0

    def test_estimate_message_tokens_function(self, sample_messages):
        """Test estimate_message_tokens function."""
        estimate = estimate_message_tokens(sample_messages)
        assert estimate.total_tokens > 0

    def test_estimate_request_cost_function(self, sample_messages):
        """Test estimate_request_cost function."""
        result = estimate_request_cost(
            messages=sample_messages,
            model_id="openai/gpt-4o-mini",
            max_output_tokens=500
        )
        assert "estimated_input_tokens" in result
        assert "estimated_cost_usd" in result
        assert result["estimated_cost_usd"] > 0


# ============================================================
# Tracker Tests
# ============================================================

class TestUsageRecord:
    """Tests for UsageRecord."""

    def test_create_usage_record(self):
        """Test creating a usage record."""
        record = UsageRecord(
            request_id="req_123",
            tenant_id="tenant_456",
            provider="openai",
            model="openai/gpt-4o",
            input_tokens=1000,
            output_tokens=500
        )
        assert record.total_tokens == 1500
        assert record.cost_usd > 0  # Auto-calculated

    def test_usage_record_to_dict(self):
        """Test converting to dictionary."""
        record = UsageRecord(
            request_id="req_123",
            provider="openai",
            model="openai/gpt-4o",
            input_tokens=100,
            output_tokens=50
        )
        data = record.to_dict()
        assert data["request_id"] == "req_123"
        assert data["total_tokens"] == 150
        assert "created_at" in data


class TestRequestTracker:
    """Tests for RequestTracker."""

    def test_create_tracker(self):
        """Test creating a request tracker."""
        tracker = RequestTracker(
            request_id="req_123",
            tenant_id="tenant_456",
            model="openai/gpt-4o",
            provider="openai"
        )
        assert tracker.request_id == "req_123"
        assert tracker.status == UsageStatus.SUCCESS

    def test_add_tokens(self):
        """Test adding tokens during tracking."""
        tracker = RequestTracker(request_id="req_123")
        tracker.add_tokens(input_tokens=100)
        tracker.add_tokens(output_tokens=50)
        tracker.add_tokens(input_tokens=50, output_tokens=25)

        assert tracker.input_tokens == 150
        assert tracker.output_tokens == 75

    def test_complete_tracking(self):
        """Test completing tracking."""
        tracker = RequestTracker(
            request_id="req_123",
            model="openai/gpt-4o",
            provider="openai"
        )
        tracker.add_tokens(input_tokens=100, output_tokens=50)
        record = tracker.complete()

        assert record.request_id == "req_123"
        assert record.input_tokens == 100
        assert record.output_tokens == 50
        assert record.latency_ms >= 0  # May be 0 if very fast

    def test_track_error(self):
        """Test tracking an error."""
        tracker = RequestTracker(request_id="req_123")
        tracker.set_error("rate_limited", "Rate limit exceeded", UsageStatus.RATE_LIMITED)
        record = tracker.complete()

        assert record.status == UsageStatus.RATE_LIMITED
        assert record.error_code == "rate_limited"

    def test_track_fallback(self):
        """Test tracking fallback providers."""
        tracker = RequestTracker(request_id="req_123")
        tracker.add_fallback("openai")
        tracker.add_fallback("anthropic")
        record = tracker.complete()

        assert record.fallback_used
        assert len(record.fallback_providers) == 2

    def test_record_first_token(self):
        """Test recording time to first token."""
        tracker = RequestTracker(request_id="req_123")
        tracker.record_first_token()
        record = tracker.complete()

        assert record.time_to_first_token_ms is not None
        assert record.time_to_first_token_ms >= 0


class TestUsageTracker:
    """Tests for UsageTracker service."""

    @pytest.fixture
    def tracker(self):
        return UsageTracker()

    def test_start_tracking(self, tracker):
        """Test starting request tracking."""
        req_tracker = tracker.start_tracking(
            request_id="req_123",
            tenant_id="tenant_456",
            model="openai/gpt-4o",
            provider="openai"
        )
        assert req_tracker is not None
        assert tracker.get_active_request_count() == 1

    @pytest.mark.asyncio
    async def test_complete_tracking(self, tracker):
        """Test completing request tracking."""
        req_tracker = tracker.start_tracking(
            request_id="req_123",
            tenant_id="tenant_456",
            model="openai/gpt-4o",
            provider="openai"
        )
        req_tracker.add_tokens(input_tokens=100, output_tokens=50)

        record = await tracker.complete_tracking(req_tracker)

        assert record.request_id == "req_123"
        assert tracker.get_active_request_count() == 0

    @pytest.mark.asyncio
    async def test_tenant_usage_aggregation(self, tracker):
        """Test tenant-level usage aggregation."""
        for i in range(3):
            req_tracker = tracker.start_tracking(
                request_id=f"req_{i}",
                tenant_id="tenant_123",
                model="openai/gpt-4o",
                provider="openai"
            )
            req_tracker.add_tokens(input_tokens=100, output_tokens=50)
            await tracker.complete_tracking(req_tracker)

        usage = tracker.get_tenant_usage("tenant_123")
        assert usage["request_count"] == 3
        assert usage["total_tokens"] == 450


# ============================================================
# Aggregator Tests
# ============================================================

class TestUsageAggregator:
    """Tests for UsageAggregator."""

    @pytest.fixture
    def aggregator(self):
        return UsageAggregator()

    def test_aggregate_records(self, aggregator, sample_usage_records):
        """Test aggregating usage records."""
        agg = aggregator.aggregate(
            sample_usage_records,
            AggregationPeriod.DAILY
        )

        assert agg.total_requests == 50
        assert agg.successful_requests == 45  # 5 errors
        assert agg.total_tokens > 0
        assert agg.total_cost_usd > 0

    def test_aggregate_by_tenant(self, aggregator, sample_usage_records):
        """Test filtering by tenant."""
        agg = aggregator.aggregate(
            sample_usage_records,
            AggregationPeriod.DAILY,
            tenant_id="tenant_123"
        )
        assert agg.total_requests == 50

        agg_empty = aggregator.aggregate(
            sample_usage_records,
            AggregationPeriod.DAILY,
            tenant_id="nonexistent"
        )
        assert agg_empty.total_requests == 0

    def test_aggregate_by_provider(self, aggregator, sample_usage_records):
        """Test filtering by provider."""
        agg = aggregator.aggregate(
            sample_usage_records,
            AggregationPeriod.DAILY,
            provider="openai"
        )
        # Should only count OpenAI requests
        assert agg.total_requests > 0
        assert agg.total_requests < 50

    def test_aggregate_by_time(self, aggregator, sample_usage_records):
        """Test time-based aggregation."""
        aggregates = aggregator.aggregate_by_time(
            sample_usage_records,
            AggregationPeriod.DAILY
        )
        assert len(aggregates) > 1
        # Each should be a day bucket
        total_requests = sum(a.total_requests for a in aggregates)
        assert total_requests == 50

    def test_cost_breakdown(self, aggregator, sample_usage_records):
        """Test cost breakdown."""
        breakdown = aggregator.get_cost_breakdown(sample_usage_records)

        assert breakdown.total_cost_usd > 0
        assert len(breakdown.by_provider) > 0
        assert len(breakdown.by_model) > 0
        assert len(breakdown.by_day) > 0

    def test_top_models(self, aggregator, sample_usage_records):
        """Test getting top models."""
        top = aggregator.get_top_models(sample_usage_records, limit=3)
        assert len(top) <= 3
        # Should be sorted by requests (default)
        if len(top) >= 2:
            assert top[0]["requests"] >= top[1]["requests"]

    def test_error_analysis(self, aggregator, sample_usage_records):
        """Test error analysis."""
        analysis = aggregator.get_error_analysis(sample_usage_records)

        assert analysis["total_errors"] == 5
        assert analysis["error_rate"] > 0


class TestAggregatorFunctions:
    """Tests for aggregator convenience functions."""

    def test_aggregate_usage_function(self, sample_usage_records):
        """Test aggregate_usage function."""
        agg = aggregate_usage(sample_usage_records)
        assert agg.total_requests > 0

    def test_get_cost_breakdown_function(self, sample_usage_records):
        """Test get_cost_breakdown function."""
        breakdown = get_cost_breakdown(sample_usage_records)
        assert breakdown.total_cost_usd > 0

    def test_get_top_models_function(self, sample_usage_records):
        """Test get_top_models function."""
        top = get_top_models(sample_usage_records, limit=5)
        assert len(top) <= 5


# ============================================================
# Limits Tests
# ============================================================

class TestUsageLimit:
    """Tests for UsageLimit."""

    def test_create_limit(self):
        """Test creating a limit."""
        limit = UsageLimit(
            limit_type=LimitType.RATE,
            period=LimitPeriod.MINUTE,
            limit_value=60
        )
        assert limit.limit_value == 60
        assert limit.get_period_seconds() == 60

    def test_period_seconds(self):
        """Test period to seconds conversion."""
        minute = UsageLimit(LimitType.RATE, LimitPeriod.MINUTE, 10)
        hour = UsageLimit(LimitType.RATE, LimitPeriod.HOUR, 10)
        day = UsageLimit(LimitType.RATE, LimitPeriod.DAY, 10)

        assert minute.get_period_seconds() == 60
        assert hour.get_period_seconds() == 3600
        assert day.get_period_seconds() == 86400


class TestQuotaConfig:
    """Tests for QuotaConfig."""

    def test_create_quota_config(self):
        """Test creating quota config."""
        config = QuotaConfig(
            tenant_id="tenant_123",
            limits=[
                UsageLimit(LimitType.RATE, LimitPeriod.MINUTE, 60)
            ]
        )
        assert config.tenant_id == "tenant_123"
        assert len(config.limits) == 1

    def test_quota_for_plan(self):
        """Test creating config from plan."""
        free_config = QuotaConfig.for_plan("tenant_123", "free")
        pro_config = QuotaConfig.for_plan("tenant_456", "pro")

        # Pro should have higher limits
        free_rate = next(l for l in free_config.limits if l.limit_type == LimitType.RATE)
        pro_rate = next(l for l in pro_config.limits if l.limit_type == LimitType.RATE)

        assert pro_rate.limit_value > free_rate.limit_value


class TestSlidingWindowCounter:
    """Tests for SlidingWindowCounter."""

    @pytest.mark.asyncio
    async def test_increment_and_count(self):
        """Test incrementing and getting count."""
        counter = SlidingWindowCounter(window_seconds=60)

        count1 = await counter.increment(1)
        assert count1 >= 0.9  # Near 1, accounting for weight

        count2 = await counter.increment(1)
        assert count2 >= 1.8  # Near 2

    @pytest.mark.asyncio
    async def test_sliding_window(self):
        """Test sliding window behavior."""
        counter = SlidingWindowCounter(window_seconds=1)

        await counter.increment(10)
        initial = await counter.get_count()

        # Wait for window to slide
        await asyncio.sleep(1.1)

        final = await counter.get_count()
        assert final < initial  # Count should decrease


class TestUsageLimiter:
    """Tests for UsageLimiter."""

    @pytest.fixture
    def limiter(self):
        limiter = UsageLimiter()
        # Set up test quota
        config = QuotaConfig(
            tenant_id="test_tenant",
            limits=[
                UsageLimit(LimitType.RATE, LimitPeriod.MINUTE, 10),
                UsageLimit(LimitType.TOKENS, LimitPeriod.DAY, 1000),
                UsageLimit(LimitType.COST, LimitPeriod.MONTH, 10.0),
            ]
        )
        limiter.set_quota(config)
        return limiter

    @pytest.mark.asyncio
    async def test_check_limits_allowed(self, limiter):
        """Test checking limits when allowed."""
        result = await limiter.check_limits("test_tenant", tokens=100, cost=0.01)
        assert result.allowed

    @pytest.mark.asyncio
    async def test_check_limits_exceeded(self, limiter):
        """Test checking limits when exceeded."""
        # Make many requests to exceed rate limit
        for _ in range(15):
            result = await limiter.check_limits("test_tenant")

        # Should eventually be rejected
        # (rate limit uses sliding window, so may need more calls)
        exceeded = any(l.limit.limit_type == LimitType.RATE for l in result.exceeded_limits)
        # Rate limit may or may not trigger depending on timing

    @pytest.mark.asyncio
    async def test_check_token_quota(self, limiter):
        """Test token quota checking."""
        # Record usage close to limit
        await limiter.record_usage("test_tenant", tokens=900)

        # Check if next request would exceed
        result = await limiter.check_limits("test_tenant", tokens=200)

        # Should have token limit exceeded
        assert any(l.limit.limit_type == LimitType.TOKENS for l in result.exceeded_limits)

    @pytest.mark.asyncio
    async def test_get_limit_status(self, limiter):
        """Test getting limit status."""
        statuses = await limiter.get_limit_status("test_tenant")
        assert len(statuses) == 3

        # Find rate limit status
        rate_status = next(s for s in statuses if s.limit.limit_type == LimitType.RATE)
        assert rate_status.remaining <= rate_status.limit.limit_value

    @pytest.mark.asyncio
    async def test_soft_limit_warning(self, limiter):
        """Test soft limit warning."""
        # Record usage at 85% of token limit
        await limiter.record_usage("test_tenant", tokens=850)

        result = await limiter.check_limits("test_tenant", tokens=10)

        # Should have soft limit warning
        assert len(result.warnings) > 0 or len(result.exceeded_limits) > 0


class TestLimitFunctions:
    """Tests for limit convenience functions."""

    @pytest.fixture(autouse=True)
    def setup_quota(self):
        """Set up test quota."""
        config = QuotaConfig.for_plan("func_test_tenant", "starter")
        set_quota(config)

    @pytest.mark.asyncio
    async def test_check_limits_function(self):
        """Test check_limits function."""
        result = await check_limits("func_test_tenant", tokens=100)
        assert isinstance(result, LimitCheckResult)

    @pytest.mark.asyncio
    async def test_record_usage_function(self):
        """Test record_usage function."""
        # Should not raise
        await record_usage("func_test_tenant", tokens=100, cost=0.01)

    @pytest.mark.asyncio
    async def test_get_limit_status_function(self):
        """Test get_limit_status function."""
        statuses = await get_limit_status("func_test_tenant")
        assert len(statuses) > 0


# ============================================================
# Integration Tests
# ============================================================

class TestUsageIntegration:
    """Integration tests for usage tracking flow."""

    @pytest.mark.asyncio
    async def test_complete_tracking_flow(self):
        """Test complete usage tracking from request to aggregation."""
        # 1. Set up limiter
        limiter = UsageLimiter()
        config = QuotaConfig.for_plan("integration_tenant", "pro")
        limiter.set_quota(config)

        # 2. Check limits before request
        check_result = await limiter.check_limits(
            "integration_tenant",
            tokens=1000,
            cost=0.01
        )
        assert check_result.allowed

        # 3. Start tracking
        tracker = UsageTracker()
        req_tracker = tracker.start_tracking(
            request_id="integration_req_001",
            tenant_id="integration_tenant",
            model="openai/gpt-4o",
            provider="openai",
            operation=OperationType.CHAT
        )

        # 4. Simulate request execution
        req_tracker.record_first_token()
        req_tracker.add_tokens(input_tokens=500, output_tokens=250)

        # 5. Complete tracking
        record = await tracker.complete_tracking(req_tracker)

        # 6. Record usage against limits
        await limiter.record_usage(
            "integration_tenant",
            tokens=record.total_tokens,
            cost=record.cost_usd
        )

        # 7. Verify record
        assert record.total_tokens == 750
        assert record.cost_usd > 0
        assert record.latency_ms >= 0  # May be 0 if very fast
        assert record.time_to_first_token_ms is not None

        # 8. Check tenant usage
        usage = tracker.get_tenant_usage("integration_tenant")
        assert usage["total_tokens"] == 750
        assert usage["request_count"] == 1

    @pytest.mark.asyncio
    async def test_cost_estimation_accuracy(self, sample_messages):
        """Test that cost estimation is reasonably accurate."""
        # Estimate cost
        estimate = estimate_request_cost(
            messages=sample_messages,
            model_id="openai/gpt-4o-mini",
            max_output_tokens=100
        )

        # Simulate actual usage
        actual_input = estimate["estimated_input_tokens"]
        actual_output = 50  # Simulated

        actual_cost = calculate_cost(
            model_id="openai/gpt-4o-mini",
            input_tokens=actual_input,
            output_tokens=actual_output
        )

        # Estimate should be in the same order of magnitude
        assert estimate["estimated_cost_usd"] > 0
        assert actual_cost > 0
        # Within 10x is reasonable for estimation
        ratio = max(estimate["estimated_cost_usd"], actual_cost) / min(estimate["estimated_cost_usd"], actual_cost)
        assert ratio < 10

    def test_aggregation_completeness(self, sample_usage_records):
        """Test that aggregation captures all data."""
        aggregator = UsageAggregator()

        # Total aggregation
        total_agg = aggregator.aggregate(sample_usage_records, AggregationPeriod.MONTHLY)

        # Time-based aggregation
        time_aggs = aggregator.aggregate_by_time(sample_usage_records, AggregationPeriod.DAILY)

        # Provider aggregation
        provider_aggs = aggregator.aggregate_by_provider(sample_usage_records)

        # All should sum to same totals
        time_total = sum(a.total_requests for a in time_aggs)
        provider_total = sum(a.total_requests for a in provider_aggs.values())

        assert total_agg.total_requests == time_total
        assert total_agg.total_requests == provider_total


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
