/**
 * 2api.ai JavaScript/TypeScript SDK
 * 
 * A simple, unified interface to access multiple AI providers.
 * 
 * @example
 * // Quick start
 * import TwoAPI from 'twoapi';
 * 
 * const client = new TwoAPI({ apiKey: '2api_xxx' });
 * 
 * // Simple chat
 * const response = await client.chat('Hello!');
 * console.log(response.content);
 * 
 * // With specific model
 * const response = await client.chat('Explain quantum computing', {
 *   model: 'anthropic/claude-3-5-sonnet'
 * });
 * 
 * // With streaming
 * for await (const chunk of client.chatStream('Tell me a story')) {
 *   process.stdout.write(chunk);
 * }
 */

// ============================================================
// Types
// ============================================================

export interface TwoAPIConfig {
  apiKey?: string;
  baseUrl?: string;
  timeout?: number;
}

export interface Message {
  role: 'system' | 'user' | 'assistant' | 'tool';
  content: string | ContentPart[];
  name?: string;
  toolCallId?: string;
  toolCalls?: ToolCall[];
}

export interface ContentPart {
  type: 'text' | 'image_url';
  text?: string;
  imageUrl?: {
    url: string;
    detail?: 'low' | 'high' | 'auto';
  };
}

export interface ToolCall {
  id: string;
  type: 'function';
  function: {
    name: string;
    arguments: string;
  };
}

export interface Tool {
  type: 'function';
  function: {
    name: string;
    description?: string;
    parameters?: Record<string, unknown>;
  };
}

export interface RoutingConfig {
  strategy?: 'cost' | 'latency' | 'quality';
  fallback?: string[];
  maxLatencyMs?: number;
  maxCost?: number;
}

export interface ChatOptions {
  model?: string;
  system?: string;
  temperature?: number;
  maxTokens?: number;
  tools?: Tool[];
  toolChoice?: 'auto' | 'none' | 'required' | { type: 'function'; function: { name: string } };
  routing?: RoutingConfig;
  metadata?: Record<string, string>;
}

export interface Usage {
  promptTokens: number;
  completionTokens: number;
  totalTokens: number;
}

export interface RoutingInfo {
  strategyUsed: string;
  fallbackUsed: boolean;
  latencyMs: number;
  costUsd: number;
}

export interface ChatResponse {
  id: string;
  content: string | null;
  model: string;
  provider: string;
  usage: Usage;
  routing?: RoutingInfo;
  toolCalls?: ToolCall[];
  finishReason: string;
}

export interface EmbeddingResponse {
  embeddings: number[][];
  model: string;
  usage: Usage;
}

export interface ImageResponse {
  urls: string[];
  revisedPrompts: string[];
}

export interface ModelInfo {
  id: string;
  provider: string;
  name: string;
  capabilities: string[];
  contextWindow: number;
  maxOutputTokens: number;
  pricing: {
    inputPer1mTokens: number;
    outputPer1mTokens: number;
  };
}

export interface HealthStatus {
  status: 'healthy' | 'degraded' | 'unhealthy';
  version: string;
  providers: Record<string, {
    status: 'healthy' | 'unhealthy';
    latencyMs?: number;
  }>;
}

// ============================================================
// Errors
// ============================================================

export class TwoAPIError extends Error {
  code: string;
  statusCode: number;
  
  constructor(message: string, code: string = 'unknown', statusCode: number = 500) {
    super(message);
    this.name = 'TwoAPIError';
    this.code = code;
    this.statusCode = statusCode;
  }
}

export class AuthenticationError extends TwoAPIError {
  constructor(message: string = 'Invalid API key') {
    super(message, 'authentication_error', 401);
    this.name = 'AuthenticationError';
  }
}

export class RateLimitError extends TwoAPIError {
  retryAfter: number;
  
  constructor(message: string = 'Rate limit exceeded', retryAfter: number = 60) {
    super(message, 'rate_limit_exceeded', 429);
    this.name = 'RateLimitError';
    this.retryAfter = retryAfter;
  }
}

export class InvalidRequestError extends TwoAPIError {
  constructor(message: string) {
    super(message, 'invalid_request', 400);
    this.name = 'InvalidRequestError';
  }
}

// ============================================================
// Helper Functions
// ============================================================

function normalizeMessages(
  message: string | Message[],
  system?: string
): Message[] {
  const messages: Message[] = [];
  
  if (system) {
    messages.push({ role: 'system', content: system });
  }
  
  if (typeof message === 'string') {
    messages.push({ role: 'user', content: message });
  } else {
    messages.push(...message);
  }
  
  return messages;
}

function parseResponse(data: Record<string, unknown>): ChatResponse {
  const choices = (data.choices as Record<string, unknown>[]) || [];
  const choice = choices[0] || {};
  const message = (choice.message as Record<string, unknown>) || {};
  const usage = (data.usage as Record<string, unknown>) || {};
  const twoapi = (data._2api as Record<string, unknown>) || {};
  const routingDecision = (twoapi.routing_decision as Record<string, unknown>) || {};
  
  let routing: RoutingInfo | undefined;
  if (Object.keys(twoapi).length > 0) {
    routing = {
      strategyUsed: (routingDecision.strategy_used as string) || '',
      fallbackUsed: (routingDecision.fallback_used as boolean) || false,
      latencyMs: (twoapi.latency_ms as number) || 0,
      costUsd: (twoapi.cost_usd as number) || 0,
    };
  }
  
  return {
    id: (data.id as string) || '',
    content: (message.content as string) || null,
    model: (data.model as string) || '',
    provider: (data.provider as string) || '',
    usage: {
      promptTokens: (usage.prompt_tokens as number) || 0,
      completionTokens: (usage.completion_tokens as number) || 0,
      totalTokens: (usage.total_tokens as number) || 0,
    },
    routing,
    toolCalls: message.tool_calls as ToolCall[] | undefined,
    finishReason: (choice.finish_reason as string) || 'stop',
  };
}

// ============================================================
// Main Client Class
// ============================================================

/**
 * 2api.ai JavaScript Client
 * 
 * A unified interface to access multiple AI providers through 2api.ai.
 * 
 * @example
 * const client = new TwoAPI({ apiKey: '2api_xxx' });
 * const response = await client.chat('Hello!');
 * console.log(response.content);
 */
export class TwoAPI {
  private apiKey: string;
  private baseUrl: string;
  private timeout: number;
  
  constructor(config: TwoAPIConfig = {}) {
    this.apiKey = config.apiKey || process.env.TWOAPI_API_KEY || '';
    if (!this.apiKey) {
      throw new AuthenticationError(
        'API key required. Set TWOAPI_API_KEY environment variable or pass apiKey in config.'
      );
    }
    
    this.baseUrl = (config.baseUrl || process.env.TWOAPI_BASE_URL || 'https://api.2api.ai/v1').replace(/\/$/, '');
    this.timeout = config.timeout || 60000;
  }
  
  /**
   * Create a chat completion
   * 
   * @param message - The message(s) to send
   * @param options - Chat options
   * @returns Chat response
   * 
   * @example
   * // Simple chat
   * const response = await client.chat('What is 2+2?');
   * 
   * // With options
   * const response = await client.chat('Explain quantum computing', {
   *   model: 'anthropic/claude-3-5-sonnet',
   *   maxTokens: 500,
   *   routing: { strategy: 'cost' }
   * });
   */
  async chat(
    message: string | Message[],
    options: ChatOptions = {}
  ): Promise<ChatResponse> {
    const messages = normalizeMessages(message, options.system);
    
    const payload: Record<string, unknown> = {
      model: options.model || 'auto',
      messages: messages.map(m => ({
        role: m.role,
        content: m.content,
        ...(m.name && { name: m.name }),
        ...(m.toolCallId && { tool_call_id: m.toolCallId }),
        ...(m.toolCalls && { tool_calls: m.toolCalls }),
      })),
    };
    
    if (options.temperature !== undefined) payload.temperature = options.temperature;
    if (options.maxTokens !== undefined) payload.max_tokens = options.maxTokens;
    if (options.tools) payload.tools = options.tools;
    if (options.toolChoice) payload.tool_choice = options.toolChoice;
    if (options.metadata) payload.metadata = options.metadata;
    
    if (options.routing) {
      payload.routing = {
        ...(options.routing.strategy && { strategy: options.routing.strategy }),
        ...(options.routing.fallback && { fallback: options.routing.fallback }),
        ...(options.routing.maxLatencyMs && { max_latency_ms: options.routing.maxLatencyMs }),
        ...(options.routing.maxCost && { max_cost: options.routing.maxCost }),
      };
    }
    
    const data = await this.request('POST', '/chat/completions', payload);
    return parseResponse(data);
  }
  
  /**
   * Create a streaming chat completion
   * 
   * @param message - The message(s) to send
   * @param options - Chat options
   * @yields Content chunks as strings
   * 
   * @example
   * for await (const chunk of client.chatStream('Tell me a story')) {
   *   process.stdout.write(chunk);
   * }
   */
  async *chatStream(
    message: string | Message[],
    options: ChatOptions = {}
  ): AsyncGenerator<string> {
    const messages = normalizeMessages(message, options.system);
    
    const payload: Record<string, unknown> = {
      model: options.model || 'auto',
      messages: messages.map(m => ({
        role: m.role,
        content: m.content,
      })),
      stream: true,
    };
    
    if (options.temperature !== undefined) payload.temperature = options.temperature;
    if (options.maxTokens !== undefined) payload.max_tokens = options.maxTokens;
    
    const response = await fetch(`${this.baseUrl}/chat/completions`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${this.apiKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    });
    
    if (!response.ok) {
      throw new TwoAPIError(`HTTP ${response.status}`, 'http_error', response.status);
    }
    
    const reader = response.body?.getReader();
    if (!reader) throw new TwoAPIError('No response body');
    
    const decoder = new TextDecoder();
    let buffer = '';
    
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';
      
      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = line.slice(6);
          if (data === '[DONE]') return;
          
          try {
            const chunk = JSON.parse(data);
            const content = chunk?.choices?.[0]?.delta?.content;
            if (content) yield content;
          } catch {
            // Skip invalid JSON
          }
        }
      }
    }
  }
  
  /**
   * Create embeddings for text
   * 
   * @param text - Text or array of texts to embed
   * @param model - Embedding model to use
   * @returns Embedding response
   * 
   * @example
   * const response = await client.embed('Hello world');
   * console.log(response.embeddings[0].length); // 1536
   */
  async embed(
    text: string | string[],
    model: string = 'openai/text-embedding-3-small'
  ): Promise<EmbeddingResponse> {
    const data = await this.request('POST', '/embeddings', {
      model,
      input: text,
    });
    
    return {
      embeddings: ((data.data as Record<string, unknown>[]) || []).map(
        item => item.embedding as number[]
      ),
      model: (data.model as string) || '',
      usage: {
        promptTokens: ((data.usage as Record<string, unknown>)?.prompt_tokens as number) || 0,
        completionTokens: 0,
        totalTokens: ((data.usage as Record<string, unknown>)?.total_tokens as number) || 0,
      },
    };
  }
  
  /**
   * Generate images from a prompt
   * 
   * @param prompt - Text description of the image
   * @param options - Image generation options
   * @returns Image response with URLs
   * 
   * @example
   * const response = await client.generateImage('A cat wearing a top hat');
   * console.log(response.urls[0]);
   */
  async generateImage(
    prompt: string,
    options: {
      model?: string;
      size?: string;
      n?: number;
      quality?: 'standard' | 'hd';
      style?: 'vivid' | 'natural';
    } = {}
  ): Promise<ImageResponse> {
    const data = await this.request('POST', '/images/generations', {
      model: options.model || 'openai/dall-e-3',
      prompt,
      size: options.size || '1024x1024',
      n: options.n || 1,
      quality: options.quality || 'standard',
      style: options.style || 'vivid',
    });
    
    const items = (data.data as Record<string, unknown>[]) || [];
    
    return {
      urls: items.map(item => (item.url as string) || '').filter(Boolean),
      revisedPrompts: items.map(item => (item.revised_prompt as string) || ''),
    };
  }
  
  /**
   * List available models
   * 
   * @param options - Filter options
   * @returns Array of model information
   */
  async listModels(options: {
    provider?: string;
    capability?: string;
  } = {}): Promise<ModelInfo[]> {
    const params = new URLSearchParams();
    if (options.provider) params.append('provider', options.provider);
    if (options.capability) params.append('capability', options.capability);
    
    const query = params.toString();
    const data = await this.request('GET', `/models${query ? `?${query}` : ''}`);
    
    return ((data.data as Record<string, unknown>[]) || []).map(model => ({
      id: (model.id as string) || '',
      provider: (model.provider as string) || '',
      name: (model.name as string) || '',
      capabilities: (model.capabilities as string[]) || [],
      contextWindow: (model.context_window as number) || 0,
      maxOutputTokens: (model.max_output_tokens as number) || 0,
      pricing: {
        inputPer1mTokens: ((model.pricing as Record<string, unknown>)?.input_per_1m_tokens as number) || 0,
        outputPer1mTokens: ((model.pricing as Record<string, unknown>)?.output_per_1m_tokens as number) || 0,
      },
    }));
  }
  
  /**
   * Check API health status
   * 
   * @returns Health status
   */
  async health(): Promise<HealthStatus> {
    const response = await fetch(this.baseUrl.replace('/v1', '') + '/health');
    return response.json() as Promise<HealthStatus>;
  }
  
  /**
   * Make an HTTP request to the API
   */
  private async request(
    method: string,
    path: string,
    body?: unknown
  ): Promise<Record<string, unknown>> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);
    
    try {
      const response = await fetch(`${this.baseUrl}${path}`, {
        method,
        headers: {
          'Authorization': `Bearer ${this.apiKey}`,
          'Content-Type': 'application/json',
          'User-Agent': 'twoapi-js/1.0.0',
        },
        body: body ? JSON.stringify(body) : undefined,
        signal: controller.signal,
      });
      
      clearTimeout(timeoutId);
      
      if (response.status === 401) {
        throw new AuthenticationError();
      }
      
      if (response.status === 429) {
        const retryAfter = parseInt(response.headers.get('Retry-After') || '60');
        throw new RateLimitError('Rate limit exceeded', retryAfter);
      }
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({})) as Record<string, unknown>;
        const error = (errorData.error as Record<string, unknown>) || {};
        throw new TwoAPIError(
          (error.message as string) || `HTTP ${response.status}`,
          (error.code as string) || 'http_error',
          response.status
        );
      }
      
      return response.json() as Promise<Record<string, unknown>>;
      
    } catch (error) {
      clearTimeout(timeoutId);
      
      if (error instanceof TwoAPIError) throw error;
      
      if ((error as Error).name === 'AbortError') {
        throw new TwoAPIError('Request timed out', 'timeout', 408);
      }
      
      throw new TwoAPIError(
        `Request failed: ${(error as Error).message}`,
        'request_error',
        500
      );
    }
  }
}

// ============================================================
// Convenience Functions (for quick usage)
// ============================================================

let defaultClient: TwoAPI | null = null;

function getDefaultClient(): TwoAPI {
  if (!defaultClient) {
    defaultClient = new TwoAPI();
  }
  return defaultClient;
}

/**
 * Quick chat function using default client
 * 
 * @example
 * import { chat } from 'twoapi';
 * const response = await chat('Hello!');
 */
export async function chat(
  message: string | Message[],
  options?: ChatOptions
): Promise<ChatResponse> {
  return getDefaultClient().chat(message, options);
}

/**
 * Quick streaming chat function using default client
 */
export async function* chatStream(
  message: string | Message[],
  options?: ChatOptions
): AsyncGenerator<string> {
  yield* getDefaultClient().chatStream(message, options);
}

/**
 * Quick embed function using default client
 */
export async function embed(
  text: string | string[],
  model?: string
): Promise<EmbeddingResponse> {
  return getDefaultClient().embed(text, model);
}

/**
 * Quick image generation function using default client
 */
export async function generateImage(
  prompt: string,
  options?: Parameters<TwoAPI['generateImage']>[1]
): Promise<ImageResponse> {
  return getDefaultClient().generateImage(prompt, options);
}

// Default export
export default TwoAPI;
