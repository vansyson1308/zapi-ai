"""
2api.ai - Usage Aggregation

Aggregates and reports usage data for analytics and billing.

Features:
- Time-based aggregation (hourly, daily, monthly)
- Provider-level stats
- Model-level stats
- Cost breakdown
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from collections import defaultdict
from enum import Enum

from .tracker import UsageRecord, UsageStatus, OperationType


class AggregationPeriod(str, Enum):
    """Time periods for aggregation."""
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


@dataclass
class UsageAggregate:
    """
    Aggregated usage statistics.

    Contains summarized data for a time period.
    """
    # Period info
    period: AggregationPeriod
    start_time: datetime
    end_time: datetime

    # Scope
    tenant_id: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None

    # Request counts
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    rate_limited_requests: int = 0

    # Token usage
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    total_cached_tokens: int = 0

    # Cost
    total_cost_usd: float = 0.0

    # Performance
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    avg_ttft_ms: Optional[float] = None  # Time to first token

    # Breakdown by operation type
    operation_breakdown: Dict[str, int] = field(default_factory=dict)

    # Breakdown by status
    status_breakdown: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "period": self.period.value,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "tenant_id": self.tenant_id,
            "provider": self.provider,
            "model": self.model,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "rate_limited_requests": self.rate_limited_requests,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "total_cached_tokens": self.total_cached_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "p50_latency_ms": round(self.p50_latency_ms, 2),
            "p95_latency_ms": round(self.p95_latency_ms, 2),
            "p99_latency_ms": round(self.p99_latency_ms, 2),
            "avg_ttft_ms": round(self.avg_ttft_ms, 2) if self.avg_ttft_ms else None,
            "operation_breakdown": self.operation_breakdown,
            "status_breakdown": self.status_breakdown,
            "success_rate": round(
                self.successful_requests / max(self.total_requests, 1) * 100, 2
            )
        }


@dataclass
class CostBreakdown:
    """
    Detailed cost breakdown by various dimensions.
    """
    total_cost_usd: float = 0.0
    by_provider: Dict[str, float] = field(default_factory=dict)
    by_model: Dict[str, float] = field(default_factory=dict)
    by_operation: Dict[str, float] = field(default_factory=dict)
    by_day: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_cost_usd": round(self.total_cost_usd, 6),
            "by_provider": {k: round(v, 6) for k, v in self.by_provider.items()},
            "by_model": {k: round(v, 6) for k, v in self.by_model.items()},
            "by_operation": {k: round(v, 6) for k, v in self.by_operation.items()},
            "by_day": {k: round(v, 6) for k, v in self.by_day.items()}
        }


class UsageAggregator:
    """
    Aggregates usage records into statistics.

    Provides various views of usage data for
    analytics and billing purposes.
    """

    def aggregate(
        self,
        records: List[UsageRecord],
        period: AggregationPeriod,
        tenant_id: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None
    ) -> UsageAggregate:
        """
        Aggregate a list of usage records.

        Args:
            records: Usage records to aggregate
            period: Aggregation time period
            tenant_id: Filter by tenant
            provider: Filter by provider
            model: Filter by model

        Returns:
            Aggregated statistics
        """
        # Filter records
        filtered = records
        if tenant_id:
            filtered = [r for r in filtered if r.tenant_id == tenant_id]
        if provider:
            filtered = [r for r in filtered if r.provider == provider]
        if model:
            filtered = [r for r in filtered if r.model == model]

        if not filtered:
            now = datetime.utcnow()
            return UsageAggregate(
                period=period,
                start_time=now,
                end_time=now,
                tenant_id=tenant_id,
                provider=provider,
                model=model
            )

        # Calculate time range
        times = [r.created_at for r in filtered]
        start_time = min(times)
        end_time = max(times)

        # Count by status
        status_breakdown: Dict[str, int] = defaultdict(int)
        operation_breakdown: Dict[str, int] = defaultdict(int)

        # Collect metrics
        latencies: List[int] = []
        ttfts: List[int] = []

        total_input = 0
        total_output = 0
        total_cached = 0
        total_cost = 0.0
        successful = 0
        failed = 0
        rate_limited = 0

        for record in filtered:
            # Counts
            status_breakdown[record.status.value] += 1
            operation_breakdown[record.operation.value] += 1

            # Tokens
            total_input += record.input_tokens
            total_output += record.output_tokens
            total_cached += record.cached_tokens

            # Cost
            total_cost += record.cost_usd

            # Status counts
            if record.status == UsageStatus.SUCCESS:
                successful += 1
            elif record.status == UsageStatus.RATE_LIMITED:
                rate_limited += 1
            else:
                failed += 1

            # Latency
            if record.latency_ms > 0:
                latencies.append(record.latency_ms)

            if record.time_to_first_token_ms:
                ttfts.append(record.time_to_first_token_ms)

        # Calculate percentiles
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        p50_latency = self._percentile(latencies, 50)
        p95_latency = self._percentile(latencies, 95)
        p99_latency = self._percentile(latencies, 99)

        avg_ttft = sum(ttfts) / len(ttfts) if ttfts else None

        return UsageAggregate(
            period=period,
            start_time=start_time,
            end_time=end_time,
            tenant_id=tenant_id,
            provider=provider,
            model=model,
            total_requests=len(filtered),
            successful_requests=successful,
            failed_requests=failed,
            rate_limited_requests=rate_limited,
            total_input_tokens=total_input,
            total_output_tokens=total_output,
            total_tokens=total_input + total_output,
            total_cached_tokens=total_cached,
            total_cost_usd=total_cost,
            avg_latency_ms=avg_latency,
            p50_latency_ms=p50_latency,
            p95_latency_ms=p95_latency,
            p99_latency_ms=p99_latency,
            avg_ttft_ms=avg_ttft,
            operation_breakdown=dict(operation_breakdown),
            status_breakdown=dict(status_breakdown)
        )

    def aggregate_by_time(
        self,
        records: List[UsageRecord],
        period: AggregationPeriod
    ) -> List[UsageAggregate]:
        """
        Aggregate records into time buckets.

        Args:
            records: Usage records
            period: Bucket size

        Returns:
            List of aggregates, one per time bucket
        """
        if not records:
            return []

        # Group by time bucket
        buckets: Dict[str, List[UsageRecord]] = defaultdict(list)

        for record in records:
            bucket_key = self._get_bucket_key(record.created_at, period)
            buckets[bucket_key].append(record)

        # Aggregate each bucket
        results = []
        for bucket_key, bucket_records in sorted(buckets.items()):
            agg = self.aggregate(bucket_records, period)
            results.append(agg)

        return results

    def aggregate_by_provider(
        self,
        records: List[UsageRecord]
    ) -> Dict[str, UsageAggregate]:
        """
        Aggregate records by provider.

        Returns:
            Dict mapping provider to aggregate
        """
        # Group by provider
        by_provider: Dict[str, List[UsageRecord]] = defaultdict(list)
        for record in records:
            by_provider[record.provider].append(record)

        # Aggregate each
        results = {}
        for provider, provider_records in by_provider.items():
            results[provider] = self.aggregate(
                provider_records,
                AggregationPeriod.DAILY,
                provider=provider
            )

        return results

    def aggregate_by_model(
        self,
        records: List[UsageRecord]
    ) -> Dict[str, UsageAggregate]:
        """
        Aggregate records by model.

        Returns:
            Dict mapping model to aggregate
        """
        # Group by model
        by_model: Dict[str, List[UsageRecord]] = defaultdict(list)
        for record in records:
            by_model[record.model].append(record)

        # Aggregate each
        results = {}
        for model, model_records in by_model.items():
            results[model] = self.aggregate(
                model_records,
                AggregationPeriod.DAILY,
                model=model
            )

        return results

    def get_cost_breakdown(
        self,
        records: List[UsageRecord]
    ) -> CostBreakdown:
        """
        Get detailed cost breakdown.

        Args:
            records: Usage records

        Returns:
            Cost breakdown by various dimensions
        """
        breakdown = CostBreakdown()

        by_provider: Dict[str, float] = defaultdict(float)
        by_model: Dict[str, float] = defaultdict(float)
        by_operation: Dict[str, float] = defaultdict(float)
        by_day: Dict[str, float] = defaultdict(float)

        for record in records:
            breakdown.total_cost_usd += record.cost_usd
            by_provider[record.provider] += record.cost_usd
            by_model[record.model] += record.cost_usd
            by_operation[record.operation.value] += record.cost_usd
            day_key = record.created_at.strftime("%Y-%m-%d")
            by_day[day_key] += record.cost_usd

        breakdown.by_provider = dict(by_provider)
        breakdown.by_model = dict(by_model)
        breakdown.by_operation = dict(by_operation)
        breakdown.by_day = dict(by_day)

        return breakdown

    def get_top_models(
        self,
        records: List[UsageRecord],
        limit: int = 10,
        metric: str = "requests"  # requests, tokens, cost
    ) -> List[Dict[str, Any]]:
        """
        Get top models by a metric.

        Args:
            records: Usage records
            limit: Number of models to return
            metric: Metric to sort by

        Returns:
            List of models with their metrics
        """
        # Aggregate by model
        model_stats: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"requests": 0, "tokens": 0, "cost": 0.0}
        )

        for record in records:
            model_stats[record.model]["requests"] += 1
            model_stats[record.model]["tokens"] += record.total_tokens
            model_stats[record.model]["cost"] += record.cost_usd

        # Convert to list and sort
        results = [
            {"model": model, **stats}
            for model, stats in model_stats.items()
        ]

        results.sort(key=lambda x: x[metric], reverse=True)

        return results[:limit]

    def get_error_analysis(
        self,
        records: List[UsageRecord]
    ) -> Dict[str, Any]:
        """
        Analyze errors in usage records.

        Returns:
            Error statistics and breakdown
        """
        errors = [r for r in records if r.status != UsageStatus.SUCCESS]

        if not errors:
            return {
                "total_errors": 0,
                "error_rate": 0.0,
                "by_code": {},
                "by_provider": {},
                "by_model": {}
            }

        by_code: Dict[str, int] = defaultdict(int)
        by_provider: Dict[str, int] = defaultdict(int)
        by_model: Dict[str, int] = defaultdict(int)

        for error in errors:
            if error.error_code:
                by_code[error.error_code] += 1
            by_provider[error.provider] += 1
            by_model[error.model] += 1

        return {
            "total_errors": len(errors),
            "error_rate": round(len(errors) / len(records) * 100, 2),
            "by_status": {
                status.value: len([e for e in errors if e.status == status])
                for status in UsageStatus
                if status != UsageStatus.SUCCESS
            },
            "by_code": dict(by_code),
            "by_provider": dict(by_provider),
            "by_model": dict(by_model)
        }

    def _percentile(self, values: List[int], p: int) -> float:
        """Calculate percentile of a list of values."""
        if not values:
            return 0.0

        sorted_values = sorted(values)
        k = (len(sorted_values) - 1) * (p / 100)
        f = int(k)
        c = f + 1 if f + 1 < len(sorted_values) else f

        if f == c:
            return float(sorted_values[f])

        return sorted_values[f] + (sorted_values[c] - sorted_values[f]) * (k - f)

    def _get_bucket_key(self, dt: datetime, period: AggregationPeriod) -> str:
        """Get bucket key for a datetime."""
        if period == AggregationPeriod.HOURLY:
            return dt.strftime("%Y-%m-%d-%H")
        elif period == AggregationPeriod.DAILY:
            return dt.strftime("%Y-%m-%d")
        elif period == AggregationPeriod.WEEKLY:
            # Monday of the week
            monday = dt - timedelta(days=dt.weekday())
            return monday.strftime("%Y-%m-%d")
        elif period == AggregationPeriod.MONTHLY:
            return dt.strftime("%Y-%m")
        return dt.strftime("%Y-%m-%d")


# Global instance
_aggregator = UsageAggregator()


def aggregate_usage(
    records: List[UsageRecord],
    period: AggregationPeriod = AggregationPeriod.DAILY,
    **filters
) -> UsageAggregate:
    """Aggregate usage records."""
    return _aggregator.aggregate(records, period, **filters)


def get_cost_breakdown(records: List[UsageRecord]) -> CostBreakdown:
    """Get cost breakdown from records."""
    return _aggregator.get_cost_breakdown(records)


def get_top_models(
    records: List[UsageRecord],
    limit: int = 10,
    metric: str = "requests"
) -> List[Dict[str, Any]]:
    """Get top models by metric."""
    return _aggregator.get_top_models(records, limit, metric)
