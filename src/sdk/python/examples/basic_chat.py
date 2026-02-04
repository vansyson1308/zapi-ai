"""
2api.ai Python SDK - Basic Chat Example

Demonstrates the simplest way to use the SDK for chat completions.
"""

import os
from twoapi import TwoAPI

def main():
    # Create client (uses TWOAPI_API_KEY env var by default)
    client = TwoAPI(api_key=os.getenv("TWOAPI_API_KEY"))

    # ============================================================
    # Simple Chat
    # ============================================================
    print("=== Simple Chat ===\n")

    # Single message
    response = client.chat("What is the capital of France?")
    print(f"Response: {response.content}")
    print(f"Model used: {response.model}")
    print(f"Provider: {response.provider}")
    print()

    # ============================================================
    # Chat with Options
    # ============================================================
    print("=== Chat with Options ===\n")

    # Specify model and parameters
    response = client.chat(
        "Explain quantum computing in one sentence",
        model="anthropic/claude-3-5-sonnet",
        temperature=0.7,
        max_tokens=100,
    )
    print(f"Response: {response.content}")
    print()

    # ============================================================
    # Chat with System Prompt
    # ============================================================
    print("=== Chat with System Prompt ===\n")

    response = client.chat(
        "Write a haiku about programming",
        system="You are a creative poet who writes in traditional Japanese forms.",
        temperature=0.9,
    )
    print(f"Haiku:\n{response.content}")
    print()

    # ============================================================
    # Multi-turn Conversation
    # ============================================================
    print("=== Multi-turn Conversation ===\n")

    # Build conversation history
    messages = [
        {"role": "user", "content": "My name is Alice."},
        {"role": "assistant", "content": "Nice to meet you, Alice! How can I help you today?"},
        {"role": "user", "content": "What is my name?"},
    ]

    response = client.chat(messages)
    print(f"Response: {response.content}")
    print()

    # ============================================================
    # OpenAI-Compatible API
    # ============================================================
    print("=== OpenAI-Compatible API ===\n")

    # Use the familiar OpenAI API format
    completion = client.chat.completions.create(
        model="openai/gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ],
        temperature=0.7,
        max_tokens=100,
    )

    print(f"Response: {completion.choices[0].message.content}")
    print(f"Finish reason: {completion.choices[0].finish_reason}")
    print(f"Usage: {completion.usage}")


if __name__ == "__main__":
    main()
