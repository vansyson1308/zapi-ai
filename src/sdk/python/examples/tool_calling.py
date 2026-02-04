"""
2api.ai Python SDK - Tool Calling Example

Demonstrates how to use tools (function calling) with the SDK.
"""

import json
from twoapi import TwoAPI, ToolRunner, tool

# ============================================================
# Define Tools
# ============================================================

@tool(
    description="Get the current weather for a location",
    parameters={
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": 'City name (e.g., "Tokyo", "New York")',
            },
            "unit": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "Temperature unit",
            },
        },
        "required": ["location"],
    },
)
def get_weather(location: str, unit: str = "celsius") -> str:
    """Get weather for a location."""
    # Simulated weather data
    weather_data = {
        "Tokyo": {"temp": 22, "condition": "sunny"},
        "New York": {"temp": 15, "condition": "cloudy"},
        "London": {"temp": 12, "condition": "rainy"},
    }

    data = weather_data.get(location, {"temp": 20, "condition": "unknown"})
    temp = data["temp"]

    if unit == "fahrenheit":
        temp = round(temp * 9 / 5 + 32)

    return json.dumps({
        "location": location,
        "temperature": temp,
        "unit": unit,
        "condition": data["condition"],
    })


@tool(
    description="Perform mathematical calculations",
    parameters={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": 'Mathematical expression (e.g., "2 + 2", "15 * 3")',
            },
        },
        "required": ["expression"],
    },
)
def calculate(expression: str) -> str:
    """Evaluate a math expression."""
    try:
        # Use eval with caution - in production use a proper math parser
        result = eval(expression, {"__builtins__": {}}, {})
        return f"Result: {result}"
    except Exception:
        return "Error: Could not evaluate expression"


@tool(
    description="Search the web for information",
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query",
            },
        },
        "required": ["query"],
    },
)
def search_web(query: str) -> str:
    """Search the web."""
    # Simulated search results
    return json.dumps({
        "query": query,
        "results": [
            {"title": f"Information about {query}", "snippet": "Relevant result..."},
            {"title": f"{query} - Wikipedia", "snippet": "Encyclopedia entry..."},
        ],
    })


def main():
    client = TwoAPI()

    # ============================================================
    # Manual Tool Calling
    # ============================================================
    print("=== Manual Tool Calling ===\n")

    # Make request with tools
    response = client.chat(
        "What is the weather like in Tokyo?",
        tools=[get_weather.tool],
    )

    if response.tool_calls:
        print(f"Tool calls requested: {len(response.tool_calls)}")

        for tool_call in response.tool_calls:
            func = tool_call["function"]
            print(f"  - {func['name']}({func['arguments']})")

            # Execute the tool
            args = json.loads(func["arguments"])
            result = get_weather(**args)
            print(f"  Result: {result}")
    print()

    # ============================================================
    # Using ToolRunner
    # ============================================================
    print("=== Using ToolRunner ===\n")

    # Create a runner with all tools
    def on_tool_call(name: str, args: str, result: str):
        print(f"  [Tool] {name}({args}) => {result[:50]}...")

    runner = ToolRunner(
        [get_weather, calculate, search_web],
        max_iterations=5,
        on_tool_call=on_tool_call,
    )

    # Run a query that requires tool use
    print('Query: "What is 15 + 27, and what is the weather in London?"')
    result = runner.run(
        client,
        "What is 15 + 27, and what is the weather in London?"
    )

    print(f"\nFinal response: {result.response.content}")
    print(f"Tool calls made: {len(result.tool_calls_made)}")
    print(f"Total iterations: {result.total_iterations}")
    print()

    # ============================================================
    # Multi-Tool Conversation
    # ============================================================
    print("=== Multi-Tool Conversation ===\n")

    # Run with system prompt
    result = runner.run(
        client,
        "Compare the weather in Tokyo and New York, and tell me which is warmer.",
        system="You are a helpful weather assistant. Always check the weather before answering.",
        model="openai/gpt-4o",
    )

    print(f"Response: {result.response.content}")
    print(f"Tools used: {', '.join(tc.name for tc in result.tool_calls_made)}")
    print()

    # ============================================================
    # Conversation History with Tools
    # ============================================================
    print("=== Continuing Conversation ===\n")

    # Use the messages from the previous run to continue
    result2 = runner.run(
        client,
        "What if I need to know the weather in celsius for both?",
        messages=result.messages,
    )

    print(f"Follow-up response: {result2.response.content}")


if __name__ == "__main__":
    main()
