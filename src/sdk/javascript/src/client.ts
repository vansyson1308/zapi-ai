/**
 * 2api.ai JavaScript/TypeScript SDK - Main Client
 *
 * A unified interface to access multiple AI providers.
 *
 * @example
 * // Simple usage
 * const client = new TwoAPI({ apiKey: '2api_xxx' });
 * const response = await client.chat('Hello!');
 * console.log(response.content);
 *
 * // OpenAI-compatible usage
 * const response = await client.chat.completions.create({
 *   model: 'openai/gpt-4o',
 *   messages: [{ role: 'user', content: 'Hello!' }]
 * });
 */

import type {
  Message,
  ChatOptions,
  ChatResponse,
  RoutingConfig,
  EmbeddingResponse,
  ImageResponse,
  ModelInfo,
  HealthStatus,
  Tool,
  ToolCall,
  Usage,
} from './types';

import {
  TwoAPIError,
  AuthenticationError,
  RateLimitError,
  InvalidRequestError,
  ProviderError,
  TimeoutError,
  ConnectionError,
  StreamError,
} from './errors';

import { RetryHandler, RetryConfig } from './retry';

import {
  ChatCompletion,
  ChatCompletionChunk,
  parseChatCompletion,
  parseStreamChunk,
} from './openai-compat';

// ============================================================
// Configuration
// ============================================================

export interface TwoAPIConfig {
  apiKey?: string;
  baseUrl?: string;
  timeout?: number;
  maxRetries?: number;
  defaultModel?: string;
  defaultRouting?: RoutingConfig;
  onRetry?: (attempt: number, error: Error, delay: number) => void;
}

const DEFAULT_BASE_URL = 'https://api.2api.ai/v1';
const DEFAULT_TIMEOUT = 60000;
const DEFAULT_MAX_RETRIES = 3;

// ============================================================
// Chat Completions Interface (OpenAI-compatible)
// ============================================================

interface CreateChatCompletionOptions {
  model: string;
  messages: Message[];
  stream?: false;
  temperature?: number;
  maxTokens?: number;
  topP?: number;
  frequencyPenalty?: number;
  presencePenalty?: number;
  stop?: string | string[];
  n?: number;
  tools?: Tool[];
  toolChoice?: string | { type: string; function?: { name: string } };
  responseFormat?: { type: string };
  seed?: number;
  user?: string;
  routing?: RoutingConfig;
}

interface CreateChatCompletionStreamOptions extends Omit<CreateChatCompletionOptions, 'stream'> {
  stream: true;
}

class ChatCompletions {
  constructor(private client: TwoAPI) {}

  /**
   * Create a chat completion (non-streaming).
   */
  async create(options: CreateChatCompletionOptions): Promise<ChatCompletion>;
  /**
   * Create a chat completion (streaming).
   */
  async create(options: CreateChatCompletionStreamOptions): Promise<AsyncIterable<ChatCompletionChunk>>;
  /**
   * Create a chat completion.
   */
  async create(
    options: CreateChatCompletionOptions | CreateChatCompletionStreamOptions
  ): Promise<ChatCompletion | AsyncIterable<ChatCompletionChunk>> {
    if (options.stream) {
      return this.createStream(options);
    }
    return this.createSync(options);
  }

  private async createSync(options: CreateChatCompletionOptions): Promise<ChatCompletion> {
    const payload = this.buildPayload(options);
    const response = await this.client._request('POST', '/chat/completions', payload);
    return parseChatCompletion(response);
  }

  private async createStream(
    options: CreateChatCompletionStreamOptions
  ): Promise<AsyncIterable<ChatCompletionChunk>> {
    const payload = this.buildPayload(options);
    payload.stream = true;
    return this.client._streamRequest('/chat/completions', payload);
  }

  private buildPayload(options: CreateChatCompletionOptions | CreateChatCompletionStreamOptions): Record<string, unknown> {
    const payload: Record<string, unknown> = {
      model: options.model,
      messages: options.messages,
    };

    if (options.temperature !== undefined) payload.temperature = options.temperature;
    if (options.maxTokens !== undefined) payload.max_tokens = options.maxTokens;
    if (options.topP !== undefined) payload.top_p = options.topP;
    if (options.frequencyPenalty !== undefined) payload.frequency_penalty = options.frequencyPenalty;
    if (options.presencePenalty !== undefined) payload.presence_penalty = options.presencePenalty;
    if (options.stop !== undefined) payload.stop = options.stop;
    if (options.n !== undefined) payload.n = options.n;
    if (options.tools !== undefined) payload.tools = options.tools;
    if (options.toolChoice !== undefined) payload.tool_choice = options.toolChoice;
    if (options.responseFormat !== undefined) payload.response_format = options.responseFormat;
    if (options.seed !== undefined) payload.seed = options.seed;
    if (options.user !== undefined) payload.user = options.user;
    if (options.routing !== undefined) payload.routing = options.routing;

    return payload;
  }
}

class Chat {
  public readonly completions: ChatCompletions;

  constructor(private client: TwoAPI) {
    this.completions = new ChatCompletions(client);
  }
}

// ============================================================
// Chat Proxy (Enables both simple and OpenAI-compatible API)
// ============================================================

type ChatFunction = (
  message: string | Message | Message[],
  options?: ChatOptions
) => Promise<ChatResponse>;

interface ChatProxy extends ChatFunction {
  completions: ChatCompletions;
}

function createChatProxy(client: TwoAPI): ChatProxy {
  // The callable function
  const chatFn = async (
    message: string | Message | Message[],
    options?: ChatOptions
  ): Promise<ChatResponse> => {
    return client._chat(message, options);
  };

  // Add completions property
  const proxy = chatFn as ChatProxy;
  proxy.completions = new ChatCompletions(client);

  return proxy;
}

// ============================================================
// Main Client
// ============================================================

export class TwoAPI {
  private apiKey: string;
  private baseUrl: string;
  private timeout: number;
  private defaultModel: string;
  private defaultRouting?: RoutingConfig;
  private retryHandler: RetryHandler;

  /** OpenAI-compatible chat interface AND simple chat function */
  public readonly chat: ChatProxy;

  constructor(config: TwoAPIConfig = {}) {
    // Get API key from config or environment
    this.apiKey = config.apiKey || this.getEnvApiKey();
    if (!this.apiKey) {
      throw new AuthenticationError('API key is required. Set TWOAPI_API_KEY environment variable or pass apiKey in config.');
    }

    this.baseUrl = (config.baseUrl || DEFAULT_BASE_URL).replace(/\/$/, '');
    this.timeout = config.timeout || DEFAULT_TIMEOUT;
    this.defaultModel = config.defaultModel || 'auto';
    this.defaultRouting = config.defaultRouting;

    // Setup retry handler
    this.retryHandler = new RetryHandler({
      maxRetries: config.maxRetries ?? DEFAULT_MAX_RETRIES,
      onRetry: config.onRetry,
    });

    // Create chat proxy
    this.chat = createChatProxy(this);
  }

  private getEnvApiKey(): string {
    // Node.js environment
    if (typeof process !== 'undefined' && process.env) {
      return process.env.TWOAPI_API_KEY || process.env.TWOAPI_KEY || '';
    }
    return '';
  }

  // ============================================================
  // Simple Chat API
  // ============================================================

  /**
   * Simple chat API - single message in, response out.
   *
   * @example
   * const response = await client.chat('Hello!');
   * console.log(response.content);
   */
  async _chat(
    message: string | Message | Message[],
    options: ChatOptions = {}
  ): Promise<ChatResponse> {
    // Build messages array
    let messages: Message[];
    if (typeof message === 'string') {
      messages = [{ role: 'user', content: message }];
    } else if (Array.isArray(message)) {
      messages = message;
    } else {
      messages = [message];
    }

    // Add existing messages if provided
    if (options.messages) {
      messages = [...options.messages, ...messages];
    }

    const payload: Record<string, unknown> = {
      model: options.model || this.defaultModel,
      messages,
    };

    // Add system prompt
    if (options.system) {
      messages.unshift({ role: 'system', content: options.system });
    }

    // Add optional parameters
    if (options.temperature !== undefined) payload.temperature = options.temperature;
    if (options.maxTokens !== undefined) payload.max_tokens = options.maxTokens;
    if (options.topP !== undefined) payload.top_p = options.topP;
    if (options.stop !== undefined) payload.stop = options.stop;
    if (options.tools !== undefined) payload.tools = options.tools;
    if (options.toolChoice !== undefined) payload.tool_choice = options.toolChoice;

    // Add routing config
    const routing = options.routing || this.defaultRouting;
    if (routing) {
      payload.routing = routing;
    }

    const response = await this._request('POST', '/chat/completions', payload);
    return this.parseSimpleResponse(response);
  }

  /**
   * Streaming chat API.
   *
   * @example
   * for await (const chunk of client.chatStream('Tell me a story')) {
   *   process.stdout.write(chunk.content || '');
   * }
   */
  async *chatStream(
    message: string | Message | Message[],
    options: ChatOptions = {}
  ): AsyncGenerator<{ content: string; done: boolean; toolCalls?: ToolCall[] }> {
    // Build messages array
    let messages: Message[];
    if (typeof message === 'string') {
      messages = [{ role: 'user', content: message }];
    } else if (Array.isArray(message)) {
      messages = message;
    } else {
      messages = [message];
    }

    // Add existing messages if provided
    if (options.messages) {
      messages = [...options.messages, ...messages];
    }

    const payload: Record<string, unknown> = {
      model: options.model || this.defaultModel,
      messages,
      stream: true,
    };

    // Add system prompt
    if (options.system) {
      messages.unshift({ role: 'system', content: options.system });
    }

    // Add optional parameters
    if (options.temperature !== undefined) payload.temperature = options.temperature;
    if (options.maxTokens !== undefined) payload.max_tokens = options.maxTokens;
    if (options.topP !== undefined) payload.top_p = options.topP;
    if (options.stop !== undefined) payload.stop = options.stop;
    if (options.tools !== undefined) payload.tools = options.tools;

    // Add routing config
    const routing = options.routing || this.defaultRouting;
    if (routing) {
      payload.routing = routing;
    }

    const stream = await this._streamRequest('/chat/completions', payload);

    for await (const chunk of stream) {
      const delta = chunk.choices[0]?.delta;
      yield {
        content: delta?.content || '',
        done: chunk.choices[0]?.finish_reason !== null,
        toolCalls: delta?.tool_calls?.map(tc => ({
          id: tc.id,
          type: tc.type,
          function: {
            name: tc.function.name,
            arguments: tc.function.arguments,
          },
        })),
      };
    }
  }

  // ============================================================
  // Embeddings API
  // ============================================================

  /**
   * Create embeddings for text.
   *
   * @example
   * const response = await client.embed('Hello world');
   * console.log(response.embeddings[0].length); // Vector dimension
   */
  async embed(
    input: string | string[],
    options: {
      model?: string;
      dimensions?: number;
      encodingFormat?: string;
    } = {}
  ): Promise<EmbeddingResponse> {
    const payload: Record<string, unknown> = {
      model: options.model || 'auto',
      input,
    };

    if (options.dimensions !== undefined) payload.dimensions = options.dimensions;
    if (options.encodingFormat !== undefined) payload.encoding_format = options.encodingFormat;

    const response = await this._request('POST', '/embeddings', payload);

    return {
      embeddings: response.data.map((d: { embedding: number[] }) => d.embedding),
      model: response.model,
      provider: response.provider || 'unknown',
      usage: {
        inputTokens: response.usage?.prompt_tokens || 0,
        outputTokens: 0,
        totalTokens: response.usage?.total_tokens || 0,
      },
    };
  }

  // ============================================================
  // Images API
  // ============================================================

  /**
   * Generate images from text.
   *
   * @example
   * const response = await client.generateImage('A sunset over mountains');
   * console.log(response.images[0].url);
   */
  async generateImage(
    prompt: string,
    options: {
      model?: string;
      n?: number;
      size?: string;
      quality?: string;
      style?: string;
      responseFormat?: string;
    } = {}
  ): Promise<ImageResponse> {
    const payload: Record<string, unknown> = {
      model: options.model || 'auto',
      prompt,
    };

    if (options.n !== undefined) payload.n = options.n;
    if (options.size !== undefined) payload.size = options.size;
    if (options.quality !== undefined) payload.quality = options.quality;
    if (options.style !== undefined) payload.style = options.style;
    if (options.responseFormat !== undefined) payload.response_format = options.responseFormat;

    const response = await this._request('POST', '/images/generations', payload);

    return {
      images: response.data.map((d: { url?: string; b64_json?: string; revised_prompt?: string }) => ({
        url: d.url,
        b64Json: d.b64_json,
        revisedPrompt: d.revised_prompt,
      })),
      model: response.model,
      provider: response.provider || 'unknown',
    };
  }

  // ============================================================
  // Models API
  // ============================================================

  /**
   * List available models.
   *
   * @example
   * const models = await client.listModels();
   * models.forEach(m => console.log(m.id));
   */
  async listModels(options: { provider?: string } = {}): Promise<ModelInfo[]> {
    let endpoint = '/models';
    if (options.provider) {
      endpoint += `?provider=${encodeURIComponent(options.provider)}`;
    }

    const response = await this._request('GET', endpoint);

    return response.data.map((m: Record<string, unknown>) => ({
      id: m.id,
      provider: m.provider || (m.id as string).split('/')[0],
      name: m.name || m.id,
      description: m.description,
      contextLength: m.context_length,
      inputPrice: m.input_price,
      outputPrice: m.output_price,
      capabilities: m.capabilities || [],
    }));
  }

  /**
   * Get model information.
   */
  async getModel(modelId: string): Promise<ModelInfo> {
    const response = await this._request('GET', `/models/${encodeURIComponent(modelId)}`);

    return {
      id: response.id,
      provider: response.provider || modelId.split('/')[0],
      name: response.name || response.id,
      description: response.description,
      contextLength: response.context_length,
      inputPrice: response.input_price,
      outputPrice: response.output_price,
      capabilities: response.capabilities || [],
    };
  }

  // ============================================================
  // Health API
  // ============================================================

  /**
   * Check API health status.
   */
  async health(): Promise<HealthStatus> {
    const response = await this._request('GET', '/health');

    return {
      status: response.status,
      version: response.version,
      providers: response.providers || {},
    };
  }

  // ============================================================
  // Internal Methods
  // ============================================================

  /**
   * Make an HTTP request to the API.
   * @internal
   */
  async _request(
    method: string,
    endpoint: string,
    body?: Record<string, unknown>
  ): Promise<Record<string, unknown>> {
    return this.retryHandler.execute(async () => {
      const url = `${this.baseUrl}${endpoint}`;

      const headers: Record<string, string> = {
        'Authorization': `Bearer ${this.apiKey}`,
        'Content-Type': 'application/json',
        'User-Agent': 'twoapi-js/1.0.0',
      };

      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), this.timeout);

      try {
        const response = await fetch(url, {
          method,
          headers,
          body: body ? JSON.stringify(body) : undefined,
          signal: controller.signal,
        });

        clearTimeout(timeoutId);

        if (!response.ok) {
          await this.handleErrorResponse(response);
        }

        return await response.json();
      } catch (error) {
        clearTimeout(timeoutId);

        if (error instanceof TwoAPIError) {
          throw error;
        }

        if ((error as Error).name === 'AbortError') {
          throw new TimeoutError(`Request timed out after ${this.timeout}ms`, 0);
        }

        throw new ConnectionError(`Failed to connect to API: ${(error as Error).message}`, 0);
      }
    });
  }

  /**
   * Make a streaming request to the API.
   * @internal
   */
  async *_streamRequest(
    endpoint: string,
    body: Record<string, unknown>
  ): AsyncGenerator<ChatCompletionChunk> {
    const url = `${this.baseUrl}${endpoint}`;

    const headers: Record<string, string> = {
      'Authorization': `Bearer ${this.apiKey}`,
      'Content-Type': 'application/json',
      'Accept': 'text/event-stream',
      'User-Agent': 'twoapi-js/1.0.0',
    };

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);

    try {
      const response = await fetch(url, {
        method: 'POST',
        headers,
        body: JSON.stringify(body),
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        await this.handleErrorResponse(response);
      }

      if (!response.body) {
        throw new StreamError('Response body is null', 0);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      try {
        while (true) {
          const { done, value } = await reader.read();

          if (done) {
            break;
          }

          buffer += decoder.decode(value, { stream: true });

          // Process complete lines
          const lines = buffer.split('\n');
          buffer = lines.pop() || '';

          for (const line of lines) {
            const trimmed = line.trim();
            if (!trimmed || trimmed === ':') continue;

            if (trimmed.startsWith('data: ')) {
              const data = trimmed.slice(6);

              if (data === '[DONE]') {
                return;
              }

              try {
                const parsed = JSON.parse(data);
                yield parseStreamChunk(parsed);
              } catch (e) {
                // Skip malformed chunks
                continue;
              }
            }
          }
        }

        // Process any remaining buffer
        if (buffer.trim()) {
          const trimmed = buffer.trim();
          if (trimmed.startsWith('data: ') && trimmed.slice(6) !== '[DONE]') {
            try {
              const parsed = JSON.parse(trimmed.slice(6));
              yield parseStreamChunk(parsed);
            } catch (e) {
              // Skip malformed chunks
            }
          }
        }
      } finally {
        reader.releaseLock();
      }
    } catch (error) {
      clearTimeout(timeoutId);

      if (error instanceof TwoAPIError) {
        throw error;
      }

      if ((error as Error).name === 'AbortError') {
        throw new TimeoutError(`Stream timed out after ${this.timeout}ms`, 0);
      }

      throw new StreamError(`Stream error: ${(error as Error).message}`, 0);
    }
  }

  private async handleErrorResponse(response: Response): Promise<never> {
    let errorData: Record<string, unknown>;

    try {
      errorData = await response.json();
    } catch {
      errorData = { error: { message: response.statusText } };
    }

    const error = (errorData.error as Record<string, unknown>) || errorData;
    const message = (error.message as string) || response.statusText;
    const code = (error.code as string) || `http_${response.status}`;
    const statusCode = response.status;

    // Determine error type based on status code
    switch (statusCode) {
      case 401:
      case 403:
        throw new AuthenticationError(message, statusCode, code);
      case 429:
        const retryAfter = parseInt(response.headers.get('retry-after') || '60', 10);
        throw new RateLimitError(message, statusCode, code, retryAfter);
      case 400:
      case 422:
        throw new InvalidRequestError(message, statusCode, code);
      case 500:
      case 502:
      case 503:
      case 504:
        throw new ProviderError(
          message,
          statusCode,
          code,
          (error.provider as string) || 'unknown'
        );
      default:
        throw new TwoAPIError(message, statusCode, code);
    }
  }

  private parseSimpleResponse(response: Record<string, unknown>): ChatResponse {
    const choice = (response.choices as Array<Record<string, unknown>>)?.[0];
    const message = choice?.message as Record<string, unknown>;
    const usage = response.usage as Record<string, number>;

    return {
      content: (message?.content as string) || '',
      role: (message?.role as string) || 'assistant',
      model: (response.model as string) || '',
      provider: (response.provider as string) || '',
      tool_calls: message?.tool_calls as ToolCall[] | undefined,
      finish_reason: (choice?.finish_reason as string) || 'stop',
      usage: usage
        ? {
            inputTokens: usage.prompt_tokens || 0,
            outputTokens: usage.completion_tokens || 0,
            totalTokens: usage.total_tokens || 0,
          }
        : undefined,
      routing: response.routing as ChatResponse['routing'],
      _2api: response._2api as Record<string, unknown>,
    };
  }
}
