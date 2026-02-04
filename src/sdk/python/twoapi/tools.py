"""
2api.ai SDK - Tool Calling Helpers

Utilities for implementing tool calling loops with the SDK.

Example:
    from twoapi import TwoAPI, Tool
    from twoapi.tools import ToolRunner, tool

    # Define tools using decorator
    @tool(description="Get the current weather")
    def get_weather(location: str, unit: str = "celsius") -> str:
        return f"Weather in {location}: 22{unit[0].upper()}"

    @tool(description="Calculate a math expression")
    def calculate(expression: str) -> str:
        return str(eval(expression))

    # Create runner with tools
    runner = ToolRunner([get_weather, calculate])

    # Run conversation with automatic tool execution
    client = TwoAPI()
    response = runner.run(
        client,
        "What's the weather in Tokyo and what's 2+2?"
    )
    print(response.content)  # Final response after tool calls
"""

from __future__ import annotations

import json
import inspect
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Union,
    TypeVar,
    get_type_hints,
)
from dataclasses import dataclass, field
from functools import wraps

from .models import (
    Message,
    Tool,
    ToolCall,
    FunctionDefinition,
    ChatResponse,
)


T = TypeVar("T")


# ============================================================
# Tool Definition Helpers
# ============================================================

@dataclass
class ToolFunction:
    """
    A callable function registered as a tool.

    Wraps a Python function with its Tool definition for use
    with the 2api.ai API.
    """
    func: Callable[..., Any]
    tool: Tool
    name: str
    description: str
    parameters: Dict[str, Any]

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call the underlying function."""
        return self.func(*args, **kwargs)

    def execute(self, arguments: Union[str, Dict[str, Any]]) -> str:
        """
        Execute the tool with the given arguments.

        Args:
            arguments: JSON string or dict of arguments

        Returns:
            String result of the function call
        """
        if isinstance(arguments, str):
            try:
                args_dict = json.loads(arguments)
            except json.JSONDecodeError:
                return f"Error: Invalid JSON arguments: {arguments}"
        else:
            args_dict = arguments

        try:
            result = self.func(**args_dict)
            if isinstance(result, str):
                return result
            return json.dumps(result, default=str)
        except Exception as e:
            return f"Error executing {self.name}: {str(e)}"


def _python_type_to_json_schema(python_type: Any) -> Dict[str, Any]:
    """Convert Python type hints to JSON Schema types."""
    if python_type is None or python_type is type(None):
        return {"type": "null"}
    elif python_type is str:
        return {"type": "string"}
    elif python_type is int:
        return {"type": "integer"}
    elif python_type is float:
        return {"type": "number"}
    elif python_type is bool:
        return {"type": "boolean"}
    elif python_type is list or (hasattr(python_type, "__origin__") and python_type.__origin__ is list):
        return {"type": "array"}
    elif python_type is dict:
        return {"type": "object"}
    else:
        # Default to string for unknown types
        return {"type": "string"}


def _generate_parameters_schema(func: Callable) -> Dict[str, Any]:
    """Generate JSON Schema for function parameters."""
    sig = inspect.signature(func)
    hints = get_type_hints(func) if hasattr(func, "__annotations__") else {}

    properties: Dict[str, Any] = {}
    required: List[str] = []

    for name, param in sig.parameters.items():
        if name in ("self", "cls"):
            continue

        # Get type from hints or default to string
        param_type = hints.get(name, str)
        schema = _python_type_to_json_schema(param_type)

        # Add description from docstring if available
        properties[name] = schema

        # Check if required (no default value)
        if param.default is inspect.Parameter.empty:
            required.append(name)

    return {
        "type": "object",
        "properties": properties,
        "required": required if required else []
    }


def tool(
    name: Optional[str] = None,
    description: str = "",
    parameters: Optional[Dict[str, Any]] = None
) -> Callable[[Callable[..., T]], ToolFunction]:
    """
    Decorator to convert a function into a Tool.

    Args:
        name: Tool name (defaults to function name)
        description: Tool description for the AI model
        parameters: JSON Schema for parameters (auto-generated if not provided)

    Returns:
        Decorated function as a ToolFunction

    Example:
        @tool(description="Get current weather for a location")
        def get_weather(location: str, unit: str = "celsius") -> str:
            return f"22 degrees {unit}"

        # Use with client
        client.chat("What's the weather?", tools=[get_weather.tool])
    """
    def decorator(func: Callable[..., T]) -> ToolFunction:
        tool_name = name or func.__name__
        tool_description = description or func.__doc__ or ""
        tool_parameters = parameters or _generate_parameters_schema(func)

        tool_def = Tool(
            function=FunctionDefinition(
                name=tool_name,
                description=tool_description,
                parameters=tool_parameters
            )
        )

        return ToolFunction(
            func=func,
            tool=tool_def,
            name=tool_name,
            description=tool_description,
            parameters=tool_parameters
        )

    return decorator


def create_tool(
    func: Callable[..., Any],
    name: Optional[str] = None,
    description: Optional[str] = None,
    parameters: Optional[Dict[str, Any]] = None
) -> ToolFunction:
    """
    Create a ToolFunction from a regular function.

    Alternative to the @tool decorator for programmatic tool creation.

    Args:
        func: The function to wrap
        name: Tool name (defaults to function name)
        description: Tool description
        parameters: JSON Schema for parameters

    Returns:
        ToolFunction instance
    """
    tool_name = name or func.__name__
    tool_description = description or func.__doc__ or ""
    tool_parameters = parameters or _generate_parameters_schema(func)

    tool_def = Tool(
        function=FunctionDefinition(
            name=tool_name,
            description=tool_description,
            parameters=tool_parameters
        )
    )

    return ToolFunction(
        func=func,
        tool=tool_def,
        name=tool_name,
        description=tool_description,
        parameters=tool_parameters
    )


# ============================================================
# Tool Execution Runner
# ============================================================

@dataclass
class ToolResult:
    """Result of a tool execution."""
    tool_call_id: str
    name: str
    result: str
    error: Optional[str] = None


@dataclass
class RunResult:
    """Result of a complete tool calling run."""
    response: ChatResponse
    tool_calls_made: List[ToolResult] = field(default_factory=list)
    total_iterations: int = 0
    messages: List[Message] = field(default_factory=list)


class ToolRunner:
    """
    Manages tool calling loops with the 2api.ai API.

    Handles the complete flow:
    1. Send user message
    2. If response has tool_calls, execute tools
    3. Send tool results back
    4. Repeat until final response

    Example:
        runner = ToolRunner([get_weather, calculate])

        # Simple run
        result = runner.run(client, "What's 2+2?")
        print(result.response.content)

        # With conversation history
        messages = [Message.user("Remember I like metric units")]
        result = runner.run(client, "What's the weather?", messages=messages)

        # Async
        result = await runner.run_async(async_client, "Tell me the weather")
    """

    def __init__(
        self,
        tools: List[Union[ToolFunction, Callable]],
        max_iterations: int = 10,
        on_tool_call: Optional[Callable[[str, str, str], None]] = None
    ):
        """
        Initialize the tool runner.

        Args:
            tools: List of ToolFunction objects or decorated functions
            max_iterations: Maximum tool calling iterations to prevent infinite loops
            on_tool_call: Optional callback called for each tool execution
                         with (tool_name, arguments, result)
        """
        self.tools_map: Dict[str, ToolFunction] = {}
        self.tools_list: List[Tool] = []
        self.max_iterations = max_iterations
        self.on_tool_call = on_tool_call

        for t in tools:
            if isinstance(t, ToolFunction):
                self.tools_map[t.name] = t
                self.tools_list.append(t.tool)
            elif hasattr(t, "tool") and hasattr(t, "execute"):
                # Duck typing for ToolFunction-like objects
                self.tools_map[t.name] = t
                self.tools_list.append(t.tool)
            else:
                # Assume it's a decorated function
                raise ValueError(
                    f"Invalid tool: {t}. Use @tool decorator or ToolFunction."
                )

    def execute_tool_call(self, tool_call: ToolCall) -> ToolResult:
        """Execute a single tool call."""
        name = tool_call.function.name
        arguments = tool_call.function.arguments

        if name not in self.tools_map:
            return ToolResult(
                tool_call_id=tool_call.id,
                name=name,
                result="",
                error=f"Unknown tool: {name}"
            )

        tool_func = self.tools_map[name]

        try:
            result = tool_func.execute(arguments)

            if self.on_tool_call:
                self.on_tool_call(name, arguments, result)

            return ToolResult(
                tool_call_id=tool_call.id,
                name=name,
                result=result
            )
        except Exception as e:
            return ToolResult(
                tool_call_id=tool_call.id,
                name=name,
                result="",
                error=str(e)
            )

    def run(
        self,
        client: "TwoAPI",
        message: Union[str, Message],
        messages: Optional[List[Message]] = None,
        model: str = "auto",
        system: Optional[str] = None,
        **kwargs
    ) -> RunResult:
        """
        Run a complete tool calling conversation.

        Args:
            client: TwoAPI client instance
            message: User message or Message object
            messages: Optional existing conversation history
            model: Model to use
            system: Optional system message
            **kwargs: Additional arguments for chat()

        Returns:
            RunResult with final response and tool call history
        """
        # Build initial messages
        conversation: List[Message] = []
        if messages:
            conversation.extend(messages)

        if isinstance(message, str):
            conversation.append(Message.user(message))
        else:
            conversation.append(message)

        tool_calls_made: List[ToolResult] = []
        iterations = 0

        while iterations < self.max_iterations:
            iterations += 1

            # Make API call
            response = client.chat(
                message=[m.to_dict() for m in conversation],
                model=model,
                system=system if iterations == 1 else None,
                tools=self.tools_list,
                **kwargs
            )

            # Check if response has tool calls
            if not response.has_tool_calls:
                # Final response
                return RunResult(
                    response=response,
                    tool_calls_made=tool_calls_made,
                    total_iterations=iterations,
                    messages=conversation
                )

            # Add assistant message with tool calls
            conversation.append(Message.assistant(
                content=response.content,
                tool_calls=response.tool_calls
            ))

            # Execute tool calls
            for tool_call in response.tool_calls:
                result = self.execute_tool_call(tool_call)
                tool_calls_made.append(result)

                # Add tool result message
                content = result.result if not result.error else f"Error: {result.error}"
                conversation.append(Message.tool(
                    tool_call_id=tool_call.id,
                    content=content
                ))

        # Max iterations reached
        return RunResult(
            response=response,
            tool_calls_made=tool_calls_made,
            total_iterations=iterations,
            messages=conversation
        )

    async def run_async(
        self,
        client: "AsyncTwoAPI",
        message: Union[str, Message],
        messages: Optional[List[Message]] = None,
        model: str = "auto",
        system: Optional[str] = None,
        **kwargs
    ) -> RunResult:
        """
        Run a complete tool calling conversation asynchronously.

        Same as run() but for AsyncTwoAPI client.
        """
        # Build initial messages
        conversation: List[Message] = []
        if messages:
            conversation.extend(messages)

        if isinstance(message, str):
            conversation.append(Message.user(message))
        else:
            conversation.append(message)

        tool_calls_made: List[ToolResult] = []
        iterations = 0

        while iterations < self.max_iterations:
            iterations += 1

            # Make API call
            response = await client.chat(
                message=[m.to_dict() for m in conversation],
                model=model,
                system=system if iterations == 1 else None,
                tools=self.tools_list,
                **kwargs
            )

            # Check if response has tool calls
            if not response.has_tool_calls:
                return RunResult(
                    response=response,
                    tool_calls_made=tool_calls_made,
                    total_iterations=iterations,
                    messages=conversation
                )

            # Add assistant message with tool calls
            conversation.append(Message.assistant(
                content=response.content,
                tool_calls=response.tool_calls
            ))

            # Execute tool calls
            for tool_call in response.tool_calls:
                result = self.execute_tool_call(tool_call)
                tool_calls_made.append(result)

                content = result.result if not result.error else f"Error: {result.error}"
                conversation.append(Message.tool(
                    tool_call_id=tool_call.id,
                    content=content
                ))

        return RunResult(
            response=response,
            tool_calls_made=tool_calls_made,
            total_iterations=iterations,
            messages=conversation
        )


# Type imports for type hints
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .client import TwoAPI
    from .async_client import AsyncTwoAPI
