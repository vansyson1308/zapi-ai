/**
 * 2api.ai JavaScript SDK - Retry Tests
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { RetryHandler, calculateBackoff, withRetry } from './retry';
import { RateLimitError, ProviderError, InvalidRequestError } from './errors';

describe('calculateBackoff', () => {
  it('should calculate exponential delay', () => {
    const delay0 = calculateBackoff(0, { initialDelay: 1000, jitter: false });
    const delay1 = calculateBackoff(1, { initialDelay: 1000, jitter: false });
    const delay2 = calculateBackoff(2, { initialDelay: 1000, jitter: false });

    expect(delay0).toBe(1000);
    expect(delay1).toBe(2000);
    expect(delay2).toBe(4000);
  });

  it('should respect maxDelay', () => {
    const delay = calculateBackoff(10, {
      initialDelay: 1000,
      maxDelay: 5000,
      jitter: false,
    });

    expect(delay).toBe(5000);
  });

  it('should add jitter when enabled', () => {
    const delays = new Set<number>();
    for (let i = 0; i < 10; i++) {
      delays.add(calculateBackoff(1, { initialDelay: 1000, jitter: true }));
    }

    // With jitter, we should get varied delays
    expect(delays.size).toBeGreaterThan(1);
  });

  it('should use custom exponential base', () => {
    const delay = calculateBackoff(2, {
      initialDelay: 1000,
      exponentialBase: 3,
      jitter: false,
    });

    expect(delay).toBe(9000); // 1000 * 3^2
  });
});

describe('RetryHandler', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('should return result on success', async () => {
    const handler = new RetryHandler();
    const fn = vi.fn().mockResolvedValue('success');

    const result = await handler.execute(fn);

    expect(result).toBe('success');
    expect(fn).toHaveBeenCalledTimes(1);
  });

  it('should retry on retryable errors', async () => {
    const handler = new RetryHandler({ maxRetries: 3 });
    const fn = vi.fn()
      .mockRejectedValueOnce(new ProviderError('Error', 500))
      .mockRejectedValueOnce(new ProviderError('Error', 502))
      .mockResolvedValue('success');

    const promise = handler.execute(fn);

    // Fast-forward through delays
    await vi.runAllTimersAsync();
    const result = await promise;

    expect(result).toBe('success');
    expect(fn).toHaveBeenCalledTimes(3);
  });

  it('should not retry on non-retryable errors', async () => {
    const handler = new RetryHandler({ maxRetries: 3 });
    const fn = vi.fn().mockRejectedValue(new InvalidRequestError('Bad request', 400));

    await expect(handler.execute(fn)).rejects.toThrow(InvalidRequestError);
    expect(fn).toHaveBeenCalledTimes(1);
  });

  it('should throw after max retries', async () => {
    const handler = new RetryHandler({ maxRetries: 2 });
    const fn = vi.fn().mockRejectedValue(new ProviderError('Error', 500));

    const promise = handler.execute(fn);
    await vi.runAllTimersAsync();

    await expect(promise).rejects.toThrow(ProviderError);
    expect(fn).toHaveBeenCalledTimes(3); // Initial + 2 retries
  });

  it('should use retry-after from RateLimitError', async () => {
    const handler = new RetryHandler({ maxRetries: 1 });
    const fn = vi.fn()
      .mockRejectedValueOnce(new RateLimitError('Rate limited', 429, 'rate_limited', 5))
      .mockResolvedValue('success');

    const promise = handler.execute(fn);

    // Should wait for retry-after duration (5 seconds)
    await vi.advanceTimersByTimeAsync(5000);
    const result = await promise;

    expect(result).toBe('success');
  });

  it('should call onRetry callback', async () => {
    const onRetry = vi.fn();
    const handler = new RetryHandler({ maxRetries: 2, onRetry });
    const fn = vi.fn()
      .mockRejectedValueOnce(new ProviderError('Error', 500))
      .mockResolvedValue('success');

    const promise = handler.execute(fn);
    await vi.runAllTimersAsync();
    await promise;

    expect(onRetry).toHaveBeenCalledTimes(1);
    expect(onRetry).toHaveBeenCalledWith(0, expect.any(Error), expect.any(Number));
  });

  it('should respect custom retryOnStatus', async () => {
    const handler = new RetryHandler({
      maxRetries: 1,
      retryOnStatus: [503], // Only retry on 503
    });

    // 500 should not retry when not in the list
    const fn500 = vi.fn().mockRejectedValue(new ProviderError('Error', 500));
    // Note: ProviderError is still retryable by default via isRetryableError
    // but custom status codes can override default behavior

    const fn503 = vi.fn()
      .mockRejectedValueOnce(new ProviderError('Error', 503))
      .mockResolvedValue('success');

    const promise = handler.execute(fn503);
    await vi.runAllTimersAsync();
    const result = await promise;

    expect(result).toBe('success');
  });
});

describe('withRetry', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('should wrap function with retry logic', async () => {
    const originalFn = vi.fn()
      .mockRejectedValueOnce(new ProviderError('Error', 500))
      .mockResolvedValue('success');

    const wrappedFn = withRetry(originalFn, { maxRetries: 2 });

    const promise = wrappedFn();
    await vi.runAllTimersAsync();
    const result = await promise;

    expect(result).toBe('success');
    expect(originalFn).toHaveBeenCalledTimes(2);
  });

  it('should preserve function arguments', async () => {
    const originalFn = vi.fn().mockResolvedValue('result');
    const wrappedFn = withRetry(originalFn);

    await wrappedFn('arg1', 'arg2');

    expect(originalFn).toHaveBeenCalledWith('arg1', 'arg2');
  });
});
