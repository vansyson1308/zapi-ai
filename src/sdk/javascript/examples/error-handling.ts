/**
 * 2api.ai JavaScript SDK - Error Handling Example
 *
 * Demonstrates proper error handling and retry logic.
 */

import {
  TwoAPI,
  TwoAPIError,
  AuthenticationError,
  RateLimitError,
  InvalidRequestError,
  ProviderError,
  TimeoutError,
  isRetryableError,
  RetryHandler,
} from 'twoapi';

async function main() {
  // ============================================================
  // Basic Error Handling
  // ============================================================
  console.log('=== Basic Error Handling ===\n');

  try {
    const client = new TwoAPI({ apiKey: 'invalid_key' });
    await client.chat('Hello');
  } catch (error) {
    if (error instanceof AuthenticationError) {
      console.log('Authentication failed:', error.message);
      console.log('Status code:', error.statusCode);
      console.log('Error code:', error.code);
    }
  }
  console.log();

  // ============================================================
  // Handling Different Error Types
  // ============================================================
  console.log('=== Error Type Detection ===\n');

  async function handleApiCall(client: TwoAPI, message: string) {
    try {
      return await client.chat(message);
    } catch (error) {
      if (error instanceof AuthenticationError) {
        // Don't retry - fix the API key
        console.error('Auth Error: Check your API key');
        throw error;
      }

      if (error instanceof RateLimitError) {
        // Wait and retry
        console.log(`Rate limited. Retry after ${error.retryAfter} seconds`);
        // In a real app, you might wait and retry here
        throw error;
      }

      if (error instanceof InvalidRequestError) {
        // Don't retry - fix the request
        console.error('Invalid Request:', error.message);
        throw error;
      }

      if (error instanceof ProviderError) {
        // Provider is down - might retry or use fallback
        console.error(`Provider ${error.provider} error:`, error.message);
        throw error;
      }

      if (error instanceof TimeoutError) {
        // Request timed out - might retry
        console.error('Request timed out');
        throw error;
      }

      if (error instanceof TwoAPIError) {
        // Generic API error
        console.error('API Error:', error.message);
        throw error;
      }

      // Unknown error
      throw error;
    }
  }

  // ============================================================
  // Using isRetryableError
  // ============================================================
  console.log('=== Checking Retryability ===\n');

  const errors = [
    new AuthenticationError('Invalid key'),
    new RateLimitError('Too many requests', 429),
    new InvalidRequestError('Bad params', 400),
    new ProviderError('OpenAI down', 503),
    new TimeoutError('Timed out'),
  ];

  for (const error of errors) {
    console.log(
      `${error.constructor.name}: retryable = ${isRetryableError(error)}`
    );
  }
  console.log();

  // ============================================================
  // Custom Retry Configuration
  // ============================================================
  console.log('=== Custom Retry Configuration ===\n');

  const client = new TwoAPI({
    apiKey: process.env.TWOAPI_API_KEY,
    maxRetries: 5, // Try up to 5 times
    onRetry: (attempt, error, delay) => {
      console.log(`Retry attempt ${attempt + 1} after ${delay}ms: ${error.message}`);
    },
  });

  // The client will automatically retry retryable errors
  try {
    const response = await client.chat('Hello!');
    console.log('Success:', response.content);
  } catch (error) {
    console.error('All retries failed:', (error as Error).message);
  }
  console.log();

  // ============================================================
  // Manual Retry Handler
  // ============================================================
  console.log('=== Manual Retry Handler ===\n');

  const retryHandler = new RetryHandler({
    maxRetries: 3,
    initialDelay: 1000, // 1 second
    maxDelay: 30000, // 30 seconds max
    exponentialBase: 2,
    onRetry: (attempt, error, delay) => {
      console.log(`Manual retry ${attempt + 1}, waiting ${delay}ms`);
    },
  });

  async function makeReliableRequest() {
    return retryHandler.execute(async () => {
      // Your API call here
      const client = new TwoAPI();
      return client.chat('Hello!');
    });
  }

  try {
    const response = await makeReliableRequest();
    console.log('Reliable request succeeded:', response.content);
  } catch (error) {
    console.error('Reliable request failed:', (error as Error).message);
  }
  console.log();

  // ============================================================
  // Error Details
  // ============================================================
  console.log('=== Error Details ===\n');

  try {
    const client = new TwoAPI({ apiKey: 'bad_key' });
    await client.chat('Hello');
  } catch (error) {
    if (error instanceof TwoAPIError) {
      console.log('Error Details:');
      console.log('  Message:', error.message);
      console.log('  Status Code:', error.statusCode);
      console.log('  Error Code:', error.code);
      console.log('  Is Retryable:', isRetryableError(error));
    }
  }
}

main().catch(console.error);
