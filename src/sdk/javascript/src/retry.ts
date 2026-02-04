/**
 * 2api.ai SDK - Retry Logic
 *
 * Exponential backoff with jitter for reliable API calls.
 */

import { TwoAPIError, RateLimitError, isRetryableError } from './errors';

// ============================================================
// Types
// ============================================================

export interface RetryConfig {
  maxRetries: number;
  initialDelay: number;
  maxDelay: number;
  exponentialBase: number;
  retryOnStatus: number[];
  onRetry?: (attempt: number, error: Error, delay: number) => void;
}

// ============================================================
// Utility Functions
// ============================================================

/**
 * Calculate delay for exponential backoff with jitter.
 */
export function calculateBackoff(
  attempt: number,
  options: {
    initialDelay?: number;
    maxDelay?: number;
    exponentialBase?: number;
    jitter?: boolean;
  } = {}
): number {
  const {
    initialDelay = 1000,
    maxDelay = 30000,
    exponentialBase = 2,
    jitter = true,
  } = options;

  // Calculate exponential delay
  let delay = initialDelay * Math.pow(exponentialBase, attempt);

  // Cap at max delay
  delay = Math.min(delay, maxDelay);

  // Add jitter (up to 25% variance)
  if (jitter) {
    const jitterRange = delay * 0.25;
    delay = delay + (Math.random() * 2 - 1) * jitterRange;
  }

  return Math.max(0, delay);
}

/**
 * Check if an error should be retried based on status codes.
 */
export function shouldRetry(
  error: unknown,
  retryOnStatus: number[] = [429, 500, 502, 503, 504]
): boolean {
  if (isRetryableError(error)) {
    return true;
  }

  if (error instanceof TwoAPIError) {
    return retryOnStatus.includes(error.statusCode);
  }

  return false;
}

/**
 * Sleep for a given number of milliseconds.
 */
export function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// ============================================================
// Retry Handler
// ============================================================

export class RetryHandler {
  private config: RetryConfig;

  constructor(config: Partial<RetryConfig> = {}) {
    this.config = {
      maxRetries: config.maxRetries ?? 3,
      initialDelay: config.initialDelay ?? 1000,
      maxDelay: config.maxDelay ?? 30000,
      exponentialBase: config.exponentialBase ?? 2,
      retryOnStatus: config.retryOnStatus ?? [429, 500, 502, 503, 504],
      onRetry: config.onRetry,
    };
  }

  /**
   * Execute a function with retry logic.
   */
  async execute<T>(fn: () => Promise<T>): Promise<T> {
    let lastError: Error | undefined;

    for (let attempt = 0; attempt <= this.config.maxRetries; attempt++) {
      try {
        return await fn();
      } catch (error) {
        lastError = error as Error;

        // Check if we should retry
        if (
          attempt >= this.config.maxRetries ||
          !shouldRetry(error, this.config.retryOnStatus)
        ) {
          throw error;
        }

        // Calculate delay
        let delay: number;
        if (error instanceof RateLimitError) {
          // Use server's suggested retry-after
          delay = error.retryAfter * 1000;
        } else {
          delay = calculateBackoff(attempt, {
            initialDelay: this.config.initialDelay,
            maxDelay: this.config.maxDelay,
            exponentialBase: this.config.exponentialBase,
          });
        }

        // Call retry callback if provided
        if (this.config.onRetry) {
          this.config.onRetry(attempt, lastError, delay);
        }

        // Wait before retry
        await sleep(delay);
      }
    }

    // Should never reach here, but just in case
    throw lastError || new Error('Retry logic error');
  }
}

// ============================================================
// Decorator-style Wrapper
// ============================================================

/**
 * Wrap a function with retry logic.
 */
export function withRetry<T extends (...args: unknown[]) => Promise<unknown>>(
  fn: T,
  config: Partial<RetryConfig> = {}
): T {
  const handler = new RetryHandler(config);

  return (async (...args: Parameters<T>) => {
    return handler.execute(() => fn(...args));
  }) as T;
}
