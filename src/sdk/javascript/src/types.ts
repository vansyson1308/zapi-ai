/**
 * 2api.ai SDK - Type Definitions
 */

// ============================================================
// Message Types
// ============================================================

export interface ContentPart {
  type: 'text' | 'image_url';
  text?: string;
  image_url?: {
    url: string;
    detail?: 'low' | 'high' | 'auto';
  };
}

export interface ToolCallFunction {
  name: string;
  arguments: string;
}

export interface ToolCall {
  id: string;
  type: 'function';
  function: ToolCallFunction;
}

export interface Message {
  role: 'system' | 'user' | 'assistant' | 'tool';
  content: string | ContentPart[] | null;
  name?: string;
  tool_call_id?: string;
  tool_calls?: ToolCall[];
}

// ============================================================
// Tool Types
// ============================================================

export interface FunctionParameters {
  type: 'object';
  properties: Record<string, {
    type: string;
    description?: string;
    enum?: string[];
  }>;
  required?: string[];
}

export interface FunctionDefinition {
  name: string;
  description?: string;
  parameters?: FunctionParameters;
}

export interface Tool {
  type: 'function';
  function: FunctionDefinition;
}

// ============================================================
// Request Types
// ============================================================

export interface RoutingConfig {
  strategy?: 'cost' | 'latency' | 'quality';
  fallback?: string[];
  max_latency_ms?: number;
  max_cost?: number;
}

export interface ChatOptions {
  model?: string;
  system?: string;
  messages?: Message[];
  temperature?: number;
  maxTokens?: number;
  topP?: number;
  frequencyPenalty?: number;
  presencePenalty?: number;
  stop?: string | string[];
  tools?: Tool[];
  toolChoice?: 'auto' | 'none' | 'required' | { type: 'function'; function: { name: string } };
  routing?: RoutingConfig;
  metadata?: Record<string, string>;
  stream?: boolean;
}

// ============================================================
// Response Types
// ============================================================

export interface Usage {
  inputTokens: number;
  outputTokens: number;
  totalTokens: number;
}

export interface RoutingInfo {
  strategy_used: string;
  fallback_used: boolean;
  latency_ms: number;
  cost_usd: number;
  provider: string;
}

export interface ChatResponse {
  content: string | null;
  role: string;
  model: string;
  provider: string;
  tool_calls?: ToolCall[];
  finish_reason: string;
  usage?: Usage;
  routing?: RoutingInfo;
  _2api?: Record<string, unknown>;
}

export interface EmbeddingData {
  index: number;
  embedding: number[];
  object: string;
}

export interface EmbeddingResponse {
  embeddings: number[][];
  model: string;
  provider: string;
  usage: Usage;
}

export interface ImageData {
  url?: string;
  b64_json?: string;
  revised_prompt?: string;
}

export interface ImageResponse {
  images: {
    url?: string;
    b64Json?: string;
    revisedPrompt?: string;
  }[];
  model: string;
  provider: string;
}

export interface ModelPricing {
  input_per_1m_tokens: number;
  output_per_1m_tokens: number;
}

export interface ModelInfo {
  id: string;
  provider: string;
  name: string;
  description?: string;
  contextLength?: number;
  inputPrice?: number;
  outputPrice?: number;
  capabilities: string[];
}

export interface ProviderHealth {
  status: 'healthy' | 'unhealthy';
  latency_ms?: number;
  error?: string;
}

export interface HealthStatus {
  status: 'healthy' | 'degraded' | 'unhealthy';
  version?: string;
  providers: Record<string, ProviderHealth>;
}

