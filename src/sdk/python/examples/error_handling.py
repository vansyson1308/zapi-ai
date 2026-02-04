"""
2api.ai Python SDK - Error Handling Example

Demonstrates proper error handling and retry logic.
"""

import os
from twoapi import TwoAPI
from twoapi.errors import (
    TwoAPIError,
    AuthenticationError,
    RateLimitError,
    InvalidRequestError,
    ProviderError,
    TimeoutError,
    is_retryable_error,
)
from twoapi.retry import RetryHandler, calculate_backoff

def main():
    # ============================================================
    # Basic Error Handling
    # ============================================================
    print("=== Basic Error Handling ===\n")

    try:
        client = TwoAPI(api_key="invalid_key")
        client.chat("Hello")
    except AuthenticationError as e:
        print(f"Authentication failed: {e.message}")
        print(f"Status code: {e.status_code}")
        print(f"Error code: {e.code}")
    print()

    # ============================================================
    # Handling Different Error Types
    # ============================================================
    print("=== Error Type Detection ===\n")

    def handle_api_call(client: TwoAPI, message: str):
        try:
            return client.chat(message)
        except AuthenticationError as e:
            # Don't retry - fix the API key
            print(f"Auth Error: Check your API key - {e.message}")
            raise

        except RateLimitError as e:
            # Wait and retry
            print(f"Rate limited. Retry after {e.retry_after} seconds")
            # In a real app, you might wait and retry here
            raise

        except InvalidRequestError as e:
            # Don't retry - fix the request
            print(f"Invalid Request: {e.message}")
            raise

        except ProviderError as e:
            # Provider is down - might retry or use fallback
            print(f"Provider {e.provider} error: {e.message}")
            raise

        except TimeoutError as e:
            # Request timed out - might retry
            print(f"Request timed out: {e.message}")
            raise

        except TwoAPIError as e:
            # Generic API error
            print(f"API Error: {e.message}")
            raise

    # ============================================================
    # Using is_retryable_error
    # ============================================================
    print("=== Checking Retryability ===\n")

    errors = [
        AuthenticationError("Invalid key"),
        RateLimitError("Too many requests", status_code=429),
        InvalidRequestError("Bad params", status_code=400),
        ProviderError("OpenAI down", status_code=503),
        TimeoutError("Timed out"),
    ]

    for error in errors:
        print(f"{error.__class__.__name__}: retryable = {is_retryable_error(error)}")
    print()

    # ============================================================
    # Custom Retry Configuration
    # ============================================================
    print("=== Custom Retry Configuration ===\n")

    def on_retry(attempt: int, error: Exception, delay: float):
        print(f"Retry attempt {attempt + 1} after {delay:.2f}s: {error}")

    client = TwoAPI(
        api_key=os.getenv("TWOAPI_API_KEY"),
        max_retries=5,  # Try up to 5 times
        on_retry=on_retry,
    )

    # The client will automatically retry retryable errors
    try:
        response = client.chat("Hello!")
        print(f"Success: {response.content}")
    except TwoAPIError as e:
        print(f"All retries failed: {e.message}")
    print()

    # ============================================================
    # Manual Retry Handler
    # ============================================================
    print("=== Manual Retry Handler ===\n")

    def manual_on_retry(attempt: int, error: Exception, delay: float):
        print(f"Manual retry {attempt + 1}, waiting {delay:.2f}s")

    retry_handler = RetryHandler(
        max_retries=3,
        initial_delay=1.0,  # 1 second
        max_delay=30.0,  # 30 seconds max
        exponential_base=2.0,
        on_retry=manual_on_retry,
    )

    def make_reliable_request():
        def _request():
            client = TwoAPI()
            return client.chat("Hello!")
        return retry_handler.execute(_request)

    try:
        response = make_reliable_request()
        print(f"Reliable request succeeded: {response.content}")
    except TwoAPIError as e:
        print(f"Reliable request failed: {e.message}")
    print()

    # ============================================================
    # Backoff Calculation
    # ============================================================
    print("=== Backoff Calculation ===\n")

    print("Exponential backoff delays:")
    for attempt in range(5):
        delay = calculate_backoff(attempt, initial_delay=1.0, jitter=False)
        delay_jitter = calculate_backoff(attempt, initial_delay=1.0, jitter=True)
        print(f"  Attempt {attempt}: {delay:.2f}s (with jitter: {delay_jitter:.2f}s)")
    print()

    # ============================================================
    # Error Details
    # ============================================================
    print("=== Error Details ===\n")

    try:
        client = TwoAPI(api_key="bad_key")
        client.chat("Hello")
    except TwoAPIError as e:
        print("Error Details:")
        print(f"  Message: {e.message}")
        print(f"  Status Code: {e.status_code}")
        print(f"  Error Code: {e.code}")
        print(f"  Is Retryable: {is_retryable_error(e)}")
        print(f"  Request ID: {e.request_id}")
        print(f"  Details: {e.details}")


if __name__ == "__main__":
    main()
