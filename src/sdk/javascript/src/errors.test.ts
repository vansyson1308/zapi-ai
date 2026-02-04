/**
 * 2api.ai JavaScript SDK - Error Tests
 */

import { describe, it, expect } from 'vitest';
import {
  TwoAPIError,
  AuthenticationError,
  RateLimitError,
  InvalidRequestError,
  ProviderError,
  TimeoutError,
  ConnectionError,
  StreamError,
  isRetryableError,
} from './errors';

describe('TwoAPIError', () => {
  it('should create error with message', () => {
    const error = new TwoAPIError('Something went wrong');
    expect(error.message).toBe('Something went wrong');
    expect(error.name).toBe('TwoAPIError');
  });

  it('should store status code', () => {
    const error = new TwoAPIError('Error', 500);
    expect(error.statusCode).toBe(500);
  });

  it('should store error code', () => {
    const error = new TwoAPIError('Error', 400, 'invalid_request');
    expect(error.code).toBe('invalid_request');
  });

  it('should be instanceof Error', () => {
    const error = new TwoAPIError('Error');
    expect(error).toBeInstanceOf(Error);
  });
});

describe('AuthenticationError', () => {
  it('should create authentication error', () => {
    const error = new AuthenticationError('Invalid API key');
    expect(error.name).toBe('AuthenticationError');
    expect(error.message).toBe('Invalid API key');
  });

  it('should be instanceof TwoAPIError', () => {
    const error = new AuthenticationError('Invalid API key');
    expect(error).toBeInstanceOf(TwoAPIError);
  });
});

describe('RateLimitError', () => {
  it('should create rate limit error', () => {
    const error = new RateLimitError('Rate limit exceeded', 429, 'rate_limited');
    expect(error.name).toBe('RateLimitError');
    expect(error.statusCode).toBe(429);
  });

  it('should store retryAfter', () => {
    const error = new RateLimitError('Rate limit', 429, 'rate_limited', 60);
    expect(error.retryAfter).toBe(60);
  });

  it('should default retryAfter to 60', () => {
    const error = new RateLimitError('Rate limit', 429);
    expect(error.retryAfter).toBe(60);
  });
});

describe('InvalidRequestError', () => {
  it('should create invalid request error', () => {
    const error = new InvalidRequestError('Invalid parameters', 400, 'invalid_params');
    expect(error.name).toBe('InvalidRequestError');
    expect(error.statusCode).toBe(400);
    expect(error.code).toBe('invalid_params');
  });
});

describe('ProviderError', () => {
  it('should create provider error', () => {
    const error = new ProviderError('OpenAI unavailable', 503, 'provider_error', 'openai');
    expect(error.name).toBe('ProviderError');
    expect(error.provider).toBe('openai');
  });

  it('should default provider to unknown', () => {
    const error = new ProviderError('Provider error', 500);
    expect(error.provider).toBe('unknown');
  });
});

describe('TimeoutError', () => {
  it('should create timeout error', () => {
    const error = new TimeoutError('Request timed out', 0);
    expect(error.name).toBe('TimeoutError');
  });
});

describe('ConnectionError', () => {
  it('should create connection error', () => {
    const error = new ConnectionError('Failed to connect', 0);
    expect(error.name).toBe('ConnectionError');
  });
});

describe('StreamError', () => {
  it('should create stream error', () => {
    const error = new StreamError('Stream interrupted', 0);
    expect(error.name).toBe('StreamError');
  });
});

describe('isRetryableError', () => {
  it('should return true for RateLimitError', () => {
    const error = new RateLimitError('Rate limited', 429);
    expect(isRetryableError(error)).toBe(true);
  });

  it('should return true for ProviderError', () => {
    const error = new ProviderError('Provider down', 503);
    expect(isRetryableError(error)).toBe(true);
  });

  it('should return true for TimeoutError', () => {
    const error = new TimeoutError('Timeout', 0);
    expect(isRetryableError(error)).toBe(true);
  });

  it('should return true for ConnectionError', () => {
    const error = new ConnectionError('Connection failed', 0);
    expect(isRetryableError(error)).toBe(true);
  });

  it('should return false for AuthenticationError', () => {
    const error = new AuthenticationError('Invalid key');
    expect(isRetryableError(error)).toBe(false);
  });

  it('should return false for InvalidRequestError', () => {
    const error = new InvalidRequestError('Bad request', 400);
    expect(isRetryableError(error)).toBe(false);
  });

  it('should return false for StreamError', () => {
    const error = new StreamError('Stream error', 0);
    expect(isRetryableError(error)).toBe(false);
  });

  it('should return false for generic Error', () => {
    const error = new Error('Generic error');
    expect(isRetryableError(error)).toBe(false);
  });

  it('should return false for non-error values', () => {
    expect(isRetryableError('string')).toBe(false);
    expect(isRetryableError(123)).toBe(false);
    expect(isRetryableError(null)).toBe(false);
    expect(isRetryableError(undefined)).toBe(false);
  });
});
