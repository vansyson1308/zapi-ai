"""
2api.ai Python SDK - Streaming Example

Demonstrates streaming responses for real-time output.
"""

import sys
from twoapi import TwoAPI

def main():
    client = TwoAPI()

    # ============================================================
    # Simple Streaming
    # ============================================================
    print("=== Simple Streaming ===\n")

    sys.stdout.write("Response: ")
    for chunk in client.chat_stream("Tell me a short story about a robot"):
        sys.stdout.write(chunk.content)
        sys.stdout.flush()
    print("\n[Stream complete]\n")

    # ============================================================
    # OpenAI-Compatible Streaming
    # ============================================================
    print("=== OpenAI-Compatible Streaming ===\n")

    stream = client.chat.completions.create(
        model="openai/gpt-4o",
        messages=[{"role": "user", "content": "Count from 1 to 10, one number per line"}],
        stream=True,
    )

    sys.stdout.write("Counting: ")
    for chunk in stream:
        content = chunk.choices[0].delta.content or ""
        sys.stdout.write(content)
        sys.stdout.flush()

        if chunk.choices[0].finish_reason:
            print(f"\n[Finished: {chunk.choices[0].finish_reason}]")
    print()

    # ============================================================
    # Streaming with Options
    # ============================================================
    print("=== Streaming with Temperature ===\n")

    sys.stdout.write("Creative response: ")
    for chunk in client.chat_stream(
        "Give me a creative name for a coffee shop",
        model="anthropic/claude-3-5-sonnet",
        temperature=0.9,
        max_tokens=50,
    ):
        sys.stdout.write(chunk.content)
        sys.stdout.flush()
    print("\n")

    # ============================================================
    # Collecting Stream Results
    # ============================================================
    print("=== Collecting Stream Results ===\n")

    full_content = ""
    chunk_count = 0

    for chunk in client.chat_stream("What are the three primary colors?"):
        full_content += chunk.content
        chunk_count += 1

    print(f"Full response: {full_content}")
    print(f"Chunks received: {chunk_count}")


if __name__ == "__main__":
    main()
