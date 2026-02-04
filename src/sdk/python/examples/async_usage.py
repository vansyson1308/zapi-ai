"""
2api.ai Python SDK - Async Usage Example

Demonstrates asynchronous usage of the SDK.
"""

import asyncio
import sys
from twoapi import AsyncTwoAPI, ToolRunner, tool

# ============================================================
# Define Tools
# ============================================================

@tool(description="Get weather")
async def get_weather(location: str) -> str:
    """Async weather tool."""
    # Simulate async API call
    await asyncio.sleep(0.1)
    return f"Weather in {location}: 22C, sunny"


async def main():
    client = AsyncTwoAPI()

    # ============================================================
    # Simple Async Chat
    # ============================================================
    print("=== Simple Async Chat ===\n")

    response = await client.chat("What is 2 + 2?")
    print(f"Response: {response.content}")
    print(f"Model: {response.model}")
    print()

    # ============================================================
    # Async Streaming
    # ============================================================
    print("=== Async Streaming ===\n")

    sys.stdout.write("Response: ")
    async for chunk in client.chat_stream("Tell me a joke"):
        sys.stdout.write(chunk.content)
        sys.stdout.flush()
    print("\n")

    # ============================================================
    # Concurrent Requests
    # ============================================================
    print("=== Concurrent Requests ===\n")

    # Make multiple requests in parallel
    questions = [
        "What is the capital of Japan?",
        "What is the capital of France?",
        "What is the capital of Brazil?",
    ]

    # Create tasks for all questions
    tasks = [client.chat(q) for q in questions]

    # Wait for all to complete
    responses = await asyncio.gather(*tasks)

    for question, response in zip(questions, responses):
        print(f"Q: {question}")
        print(f"A: {response.content}\n")

    # ============================================================
    # Async OpenAI-Compatible API
    # ============================================================
    print("=== Async OpenAI-Compatible API ===\n")

    completion = await client.chat.completions.create(
        model="openai/gpt-4o",
        messages=[
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello!"},
        ],
    )
    print(f"Response: {completion.choices[0].message.content}")
    print()

    # ============================================================
    # Async Streaming with OpenAI API
    # ============================================================
    print("=== Async OpenAI Streaming ===\n")

    stream = await client.chat.completions.create(
        model="anthropic/claude-3-5-sonnet",
        messages=[{"role": "user", "content": "Count to 5"}],
        stream=True,
    )

    sys.stdout.write("Counting: ")
    async for chunk in stream:
        content = chunk.choices[0].delta.content or ""
        sys.stdout.write(content)
        sys.stdout.flush()
    print("\n")

    # ============================================================
    # Async Tool Runner
    # ============================================================
    print("=== Async Tool Runner ===\n")

    runner = ToolRunner([get_weather], max_iterations=5)

    result = await runner.run_async(
        client,
        "What is the weather in Tokyo?",
    )

    print(f"Response: {result.response.content}")
    print(f"Tool calls: {len(result.tool_calls_made)}")
    print()

    # ============================================================
    # Async Embeddings
    # ============================================================
    print("=== Async Embeddings ===\n")

    embeddings = await client.embed(["Hello", "World"])
    print(f"Generated {len(embeddings.embeddings)} embeddings")
    print(f"Dimension: {len(embeddings.embeddings[0])}")
    print()

    # ============================================================
    # Cleanup
    # ============================================================
    await client.close()
    print("Client closed.")


if __name__ == "__main__":
    asyncio.run(main())
