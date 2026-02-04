/**
 * 2api.ai JavaScript/TypeScript SDK
 *
 * A unified interface to access multiple AI providers.
 *
 * @example
 * // Simple usage
 * import { TwoAPI } from 'twoapi';
 *
 * const client = new TwoAPI({ apiKey: '2api_xxx' });
 * const response = await client.chat('Hello!');
 * console.log(response.content);
 *
 * // OpenAI-compatible usage
 * const response = await client.chat.completions.create({
 *   model: 'openai/gpt-4o',
 *   messages: [{ role: 'user', content: 'Hello!' }]
 * });
 * console.log(response.choices[0].message.content);
 *
 * // Streaming
 * const stream = await client.chat.completions.create({
 *   model: 'openai/gpt-4o',
 *   messages: [{ role: 'user', content: 'Tell me a story' }],
 *   stream: true
 * });
 * for await (const chunk of stream) {
 *   process.stdout.write(chunk.choices[0]?.delta?.content || '');
 * }
 *
 * // Tool calling
 * import { ToolRunner, tool } from 'twoapi';
 *
 * const getWeather = tool({
 *   name: 'get_weather',
 *   description: 'Get the weather',
 *   parameters: { location: { type: 'string' } },
 *   execute: (args) => `Weather in ${args.location}: sunny`
 * });
 *
 * const runner = new ToolRunner([getWeather]);
 * const result = await runner.run(client, 'What is the weather in Tokyo?');
 */

// Main client
export { TwoAPI, type TwoAPIConfig } from './client';

// Types
export type {
  Message,
  ContentPart,
  ToolCall,
  Tool,
  RoutingConfig,
  ChatOptions,
  Usage,
  RoutingInfo,
  ChatResponse,
  EmbeddingResponse,
  ImageResponse,
  ModelInfo,
  HealthStatus,
} from './types';

// OpenAI-compatible types
export type {
  ChatCompletion,
  ChatCompletionChunk,
  ChatCompletionMessage,
  Choice,
  StreamChoice,
  ChoiceDelta,
  CompletionUsage,
} from './openai-compat';

// Errors
export {
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

// Tool calling
export {
  ToolRunner,
  tool,
  createTool,
  type ToolFunction,
  type ToolResult,
  type RunResult,
} from './tools';

// Retry utilities
export {
  RetryConfig,
  RetryHandler,
  withRetry,
  calculateBackoff,
} from './retry';

// Default export
export { TwoAPI as default } from './client';
