"""
2api.ai - Parallel Tool Call Support

Handles parallel tool call execution and result aggregation.

Key Features:
- Tracks multiple concurrent tool calls
- Aggregates results for batch submission
- Handles partial failures gracefully
- Provider-specific result formatting
"""

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set

from .schema import (
    ToolCallSchema,
    ToolResultSchema,
    FunctionCallSchema,
)
from .normalizer import ToolNormalizer


class ToolCallStatus(str, Enum):
    """Status of a tool call."""
    PENDING = "pending"       # Waiting to be executed
    RUNNING = "running"       # Currently executing
    COMPLETED = "completed"   # Successfully completed
    FAILED = "failed"         # Execution failed
    TIMEOUT = "timeout"       # Execution timed out
    CANCELLED = "cancelled"   # Cancelled by user


@dataclass
class ToolCallExecution:
    """
    Tracks execution of a single tool call.

    Contains all information about a tool call from
    start to finish.
    """
    tool_call: ToolCallSchema
    status: ToolCallStatus = ToolCallStatus.PENDING
    result: Optional[ToolResultSchema] = None
    error: Optional[str] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None

    @property
    def duration_ms(self) -> Optional[int]:
        """Get execution duration in milliseconds."""
        if self.started_at and self.completed_at:
            return int((self.completed_at - self.started_at) * 1000)
        return None

    def start(self):
        """Mark execution as started."""
        self.status = ToolCallStatus.RUNNING
        self.started_at = time.time()

    def complete(self, result: ToolResultSchema):
        """Mark execution as completed with result."""
        self.status = ToolCallStatus.COMPLETED
        self.result = result
        self.completed_at = time.time()

    def fail(self, error: str):
        """Mark execution as failed with error."""
        self.status = ToolCallStatus.FAILED
        self.error = error
        self.completed_at = time.time()

    def timeout(self):
        """Mark execution as timed out."""
        self.status = ToolCallStatus.TIMEOUT
        self.error = "Execution timed out"
        self.completed_at = time.time()

    def cancel(self):
        """Mark execution as cancelled."""
        self.status = ToolCallStatus.CANCELLED
        self.completed_at = time.time()


@dataclass
class ParallelToolCallTracker:
    """
    Tracks multiple parallel tool calls.

    Manages the lifecycle of concurrent tool executions
    and aggregates results.
    """
    request_id: str
    executions: Dict[str, ToolCallExecution] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def add_tool_call(self, tool_call: ToolCallSchema) -> str:
        """
        Add a tool call to track.

        Returns:
            The tool call ID
        """
        execution = ToolCallExecution(tool_call=tool_call)
        self.executions[tool_call.id] = execution
        return tool_call.id

    def add_tool_calls(self, tool_calls: List[ToolCallSchema]) -> List[str]:
        """Add multiple tool calls."""
        return [self.add_tool_call(tc) for tc in tool_calls]

    def get_execution(self, tool_call_id: str) -> Optional[ToolCallExecution]:
        """Get execution by tool call ID."""
        return self.executions.get(tool_call_id)

    def start_execution(self, tool_call_id: str):
        """Mark a tool call as started."""
        if tool_call_id in self.executions:
            self.executions[tool_call_id].start()

    def complete_execution(self, tool_call_id: str, content: str, is_error: bool = False):
        """
        Mark a tool call as completed.

        Args:
            tool_call_id: The tool call ID
            content: Result content (or error message)
            is_error: Whether this is an error result
        """
        if tool_call_id not in self.executions:
            return

        execution = self.executions[tool_call_id]
        result = ToolResultSchema(
            tool_call_id=tool_call_id,
            content=content,
            name=execution.tool_call.function.name,
            is_error=is_error
        )

        if is_error:
            execution.fail(content)
            execution.result = result
        else:
            execution.complete(result)

    def fail_execution(self, tool_call_id: str, error: str):
        """Mark a tool call as failed."""
        self.complete_execution(tool_call_id, error, is_error=True)

    def timeout_execution(self, tool_call_id: str):
        """Mark a tool call as timed out."""
        if tool_call_id in self.executions:
            self.executions[tool_call_id].timeout()

    def cancel_execution(self, tool_call_id: str):
        """Cancel a tool call."""
        if tool_call_id in self.executions:
            self.executions[tool_call_id].cancel()

    @property
    def pending_count(self) -> int:
        """Get number of pending executions."""
        return sum(
            1 for e in self.executions.values()
            if e.status == ToolCallStatus.PENDING
        )

    @property
    def running_count(self) -> int:
        """Get number of running executions."""
        return sum(
            1 for e in self.executions.values()
            if e.status == ToolCallStatus.RUNNING
        )

    @property
    def completed_count(self) -> int:
        """Get number of completed executions (success or failure)."""
        return sum(
            1 for e in self.executions.values()
            if e.status in {
                ToolCallStatus.COMPLETED,
                ToolCallStatus.FAILED,
                ToolCallStatus.TIMEOUT
            }
        )

    @property
    def is_all_complete(self) -> bool:
        """Check if all executions are complete."""
        return all(
            e.status in {
                ToolCallStatus.COMPLETED,
                ToolCallStatus.FAILED,
                ToolCallStatus.TIMEOUT,
                ToolCallStatus.CANCELLED
            }
            for e in self.executions.values()
        )

    @property
    def has_failures(self) -> bool:
        """Check if any executions failed."""
        return any(
            e.status in {ToolCallStatus.FAILED, ToolCallStatus.TIMEOUT}
            for e in self.executions.values()
        )

    def get_results(self) -> List[ToolResultSchema]:
        """
        Get all results.

        Returns results for completed executions.
        Failed executions return error results.
        """
        results = []

        for execution in self.executions.values():
            if execution.result:
                results.append(execution.result)
            elif execution.status == ToolCallStatus.TIMEOUT:
                results.append(ToolResultSchema(
                    tool_call_id=execution.tool_call.id,
                    content="Tool execution timed out",
                    name=execution.tool_call.function.name,
                    is_error=True
                ))
            elif execution.status == ToolCallStatus.CANCELLED:
                results.append(ToolResultSchema(
                    tool_call_id=execution.tool_call.id,
                    content="Tool execution was cancelled",
                    name=execution.tool_call.function.name,
                    is_error=True
                ))

        return results

    def get_successful_results(self) -> List[ToolResultSchema]:
        """Get only successful results."""
        return [
            e.result for e in self.executions.values()
            if e.status == ToolCallStatus.COMPLETED and e.result
        ]


# Type for tool executor functions
ToolExecutor = Callable[[ToolCallSchema], Coroutine[Any, Any, str]]


class ParallelToolExecutor:
    """
    Executes multiple tool calls in parallel.

    Provides:
    - Concurrent execution with configurable limits
    - Timeout handling
    - Error isolation (one failure doesn't affect others)
    - Result aggregation
    """

    def __init__(
        self,
        max_concurrent: int = 10,
        default_timeout: float = 30.0
    ):
        """
        Initialize parallel executor.

        Args:
            max_concurrent: Maximum concurrent executions
            default_timeout: Default timeout per tool call (seconds)
        """
        self.max_concurrent = max_concurrent
        self.default_timeout = default_timeout
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def execute_all(
        self,
        tool_calls: List[ToolCallSchema],
        executor: ToolExecutor,
        timeout: Optional[float] = None,
        request_id: Optional[str] = None
    ) -> ParallelToolCallTracker:
        """
        Execute all tool calls in parallel.

        Args:
            tool_calls: List of tool calls to execute
            executor: Async function that executes a single tool call
            timeout: Timeout per tool call (uses default if not specified)
            request_id: Request ID for tracking

        Returns:
            Tracker with all execution results
        """
        timeout = timeout or self.default_timeout
        tracker = ParallelToolCallTracker(
            request_id=request_id or f"parallel_{uuid.uuid4().hex[:8]}"
        )

        # Add all tool calls to tracker
        tracker.add_tool_calls(tool_calls)

        # Execute all in parallel
        tasks = [
            self._execute_single(tc, executor, timeout, tracker)
            for tc in tool_calls
        ]

        await asyncio.gather(*tasks, return_exceptions=True)

        return tracker

    async def _execute_single(
        self,
        tool_call: ToolCallSchema,
        executor: ToolExecutor,
        timeout: float,
        tracker: ParallelToolCallTracker
    ):
        """Execute a single tool call with semaphore and timeout."""
        async with self._semaphore:
            tracker.start_execution(tool_call.id)

            try:
                # Execute with timeout
                result = await asyncio.wait_for(
                    executor(tool_call),
                    timeout=timeout
                )
                tracker.complete_execution(tool_call.id, result)

            except asyncio.TimeoutError:
                tracker.timeout_execution(tool_call.id)

            except Exception as e:
                tracker.fail_execution(tool_call.id, str(e))


class ToolCallBatcher:
    """
    Batches tool results for submission to providers.

    Different providers have different ways of handling
    multiple tool results:
    - OpenAI: Multiple tool messages
    - Anthropic: Single user message with multiple tool_result blocks
    - Google: Single user message with multiple functionResponse parts
    """

    def __init__(self):
        self._normalizer = ToolNormalizer()

    def batch_for_openai(
        self,
        results: List[ToolResultSchema]
    ) -> List[Dict[str, Any]]:
        """
        Batch results for OpenAI.

        OpenAI wants separate messages for each tool result.
        """
        return [
            self._normalizer.create_tool_result_message_for_openai(r)
            for r in results
        ]

    def batch_for_anthropic(
        self,
        results: List[ToolResultSchema]
    ) -> Dict[str, Any]:
        """
        Batch results for Anthropic.

        Anthropic wants a single user message with multiple
        tool_result content blocks.
        """
        return self._normalizer.normalize_multiple_tool_results_for_anthropic(results)

    def batch_for_google(
        self,
        results: List[ToolResultSchema]
    ) -> Dict[str, Any]:
        """
        Batch results for Google.

        Google wants a single user message with multiple
        functionResponse parts.
        """
        return self._normalizer.normalize_multiple_tool_results_for_google(results)

    def batch_for_provider(
        self,
        results: List[ToolResultSchema],
        provider: str
    ) -> Any:
        """
        Batch results for a specific provider.

        Args:
            results: Tool results to batch
            provider: Target provider

        Returns:
            Provider-specific batch format
        """
        if provider == "openai":
            return self.batch_for_openai(results)
        elif provider == "anthropic":
            return self.batch_for_anthropic(results)
        elif provider == "google":
            return self.batch_for_google(results)
        else:
            return self.batch_for_openai(results)


# Global instances
_executor = ParallelToolExecutor()
_batcher = ToolCallBatcher()


async def execute_tools_parallel(
    tool_calls: List[ToolCallSchema],
    executor: ToolExecutor,
    timeout: Optional[float] = None,
    max_concurrent: int = 10
) -> ParallelToolCallTracker:
    """
    Execute multiple tool calls in parallel.

    Convenience function using default executor.
    """
    # Create executor with specified concurrency
    parallel_executor = ParallelToolExecutor(
        max_concurrent=max_concurrent,
        default_timeout=timeout or 30.0
    )

    return await parallel_executor.execute_all(
        tool_calls=tool_calls,
        executor=executor,
        timeout=timeout
    )


def batch_tool_results(
    results: List[ToolResultSchema],
    provider: str
) -> Any:
    """
    Batch tool results for a provider.

    Convenience function using default batcher.
    """
    return _batcher.batch_for_provider(results, provider)
