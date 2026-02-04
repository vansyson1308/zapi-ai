/**
 * 2api.ai SDK - Error Classes
 *
 * Comprehensive error handling with taxonomy awareness.
 */

// ============================================================
// Base Error
// ============================================================

export class TwoAPIError extends Error {
  code: string;
  statusCode: number;
  requestId?: string;
  retryable: boolean;
  details?: Record<string, unknown>;

  constructor(
    message: string,
    options: {
      code?: string;
      statusCode?: number;
      requestId?: string;
      retryable?: boolean;
      details?: Record<string, unknown>;
    } = {}
  ) {
    super(message);
    this.name = 'TwoAPIError';
    this.code = options.code || 'unknown';
    this.statusCode = options.statusCode || 500;
    this.requestId = options.requestId;
    this.retryable = options.retryable ?? false;
    this.details = options.details;

    // Maintain proper stack trace
    if (Error.captureStackTrace) {
      Error.captureStackTrace(this, this.constructor);
    }
  }

  /**
   * Create an error from API response data.
   */
  static fromResponse(
    data: Record<string, unknown>,
    statusCode: number
  ): TwoAPIError {
    const error = (data.error as Record<string, unknown>) || {};
    const code = (error.code as string) || 'unknown';
    const message = (error.message as string) || 'Unknown error';
    const requestId = error.request_id as string | undefined;
    const retryable = (error.retryable as boolean) || false;
    const type = error.type as string | undefined;

    // Map to specific error classes
    const errorClasses: Record<string, typeof TwoAPIError> = {
      authentication_error: AuthenticationError,
      invalid_api_key: AuthenticationError,
      missing_api_key: AuthenticationError,
      rate_limit_exceeded: RateLimitError,
      invalid_request: InvalidRequestError,
      missing_required_field: InvalidRequestError,
      provider_error: ProviderError,
      provider_down: ProviderError,
      timeout: TimeoutError,
      connection_error: ConnectionError,
      stream_error: StreamError,
    };

    const ErrorClass = errorClasses[code] || TwoAPIError;

    return new ErrorClass(message, {
      code,
      statusCode,
      requestId,
      retryable,
      details: {
        type,
        provider: error.provider,
        param: error.param,
      },
    });
  }
}

// ============================================================
// Specific Errors
// ============================================================

/**
 * API key is invalid or missing.
 */
export class AuthenticationError extends TwoAPIError {
  constructor(message: string = 'Invalid or missing API key', options: Partial<ConstructorParameters<typeof TwoAPIError>[1]> = {}) {
    super(message, {
      code: 'authentication_error',
      statusCode: 401,
      retryable: false,
      ...options,
    });
    this.name = 'AuthenticationError';
  }
}

/**
 * Rate limit exceeded.
 */
export class RateLimitError extends TwoAPIError {
  retryAfter: number;

  constructor(
    message: string = 'Rate limit exceeded',
    options: Partial<ConstructorParameters<typeof TwoAPIError>[1]> & { retryAfter?: number } = {}
  ) {
    super(message, {
      code: 'rate_limit_exceeded',
      statusCode: 429,
      retryable: true,
      ...options,
    });
    this.name = 'RateLimitError';
    this.retryAfter = options.retryAfter || 60;
  }
}

/**
 * Request parameters are invalid.
 */
export class InvalidRequestError extends TwoAPIError {
  param?: string;

  constructor(
    message: string,
    options: Partial<ConstructorParameters<typeof TwoAPIError>[1]> & { param?: string } = {}
  ) {
    super(message, {
      code: 'invalid_request',
      statusCode: 400,
      retryable: false,
      ...options,
    });
    this.name = 'InvalidRequestError';
    this.param = options.param;
  }
}

/**
 * Error from the AI provider.
 */
export class ProviderError extends TwoAPIError {
  provider?: string;

  constructor(
    message: string,
    options: Partial<ConstructorParameters<typeof TwoAPIError>[1]> & { provider?: string } = {}
  ) {
    super(message, {
      code: 'provider_error',
      statusCode: 502,
      retryable: true,
      ...options,
    });
    this.name = 'ProviderError';
    this.provider = options.provider;
  }
}

/**
 * Request timed out.
 */
export class TimeoutError extends TwoAPIError {
  constructor(message: string = 'Request timed out', options: Partial<ConstructorParameters<typeof TwoAPIError>[1]> = {}) {
    super(message, {
      code: 'timeout',
      statusCode: 408,
      retryable: true,
      ...options,
    });
    this.name = 'TimeoutError';
  }
}

/**
 * Failed to connect to the API.
 */
export class ConnectionError extends TwoAPIError {
  constructor(message: string = 'Failed to connect to API', options: Partial<ConstructorParameters<typeof TwoAPIError>[1]> = {}) {
    super(message, {
      code: 'connection_error',
      statusCode: 503,
      retryable: true,
      ...options,
    });
    this.name = 'ConnectionError';
  }
}

/**
 * Error during streaming response.
 */
export class StreamError extends TwoAPIError {
  partialContent: string;

  constructor(
    message: string = 'Stream interrupted',
    options: Partial<ConstructorParameters<typeof TwoAPIError>[1]> & { partialContent?: string } = {}
  ) {
    super(message, {
      code: 'stream_error',
      statusCode: 500,
      retryable: false, // Can't retry mid-stream
      ...options,
    });
    this.name = 'StreamError';
    this.partialContent = options.partialContent || '';
  }
}

// ============================================================
// Utility Functions
// ============================================================

/**
 * Check if an error is retryable.
 */
export function isRetryableError(error: unknown): boolean {
  if (error instanceof TwoAPIError) {
    return error.retryable;
  }

  // Network errors are generally retryable
  if (error instanceof Error) {
    const name = error.name.toLowerCase();
    if (
      name.includes('timeout') ||
      name.includes('network') ||
      name.includes('fetch')
    ) {
      return true;
    }
  }

  return false;
}

/**
 * Check if an error is a semantic error (client's fault).
 */
export function isSemanticError(error: unknown): boolean {
  if (error instanceof TwoAPIError) {
    return (
      error instanceof AuthenticationError ||
      error instanceof InvalidRequestError ||
      (error.statusCode >= 400 && error.statusCode < 500)
    );
  }
  return false;
}

/**
 * Check if an error is an infra error (server/network issue).
 */
export function isInfraError(error: unknown): boolean {
  if (error instanceof TwoAPIError) {
    return (
      error instanceof RateLimitError ||
      error instanceof ProviderError ||
      error instanceof TimeoutError ||
      error instanceof ConnectionError ||
      error.statusCode >= 500
    );
  }
  return false;
}
