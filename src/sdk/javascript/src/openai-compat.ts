/**
 * 2api.ai SDK - OpenAI-Compatible Interface
 *
 * Provides an OpenAI-like API surface for easy migration.
 */

import type { Message, Tool, ToolCall, Usage } from './types';

// ============================================================
// Response Types
// ============================================================

export interface CompletionUsage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
}

export interface ChatCompletionMessage {
  role: 'assistant';
  content: string | null;
  tool_calls?: ToolCall[];
  refusal?: string | null;
}

export interface Choice {
  index: number;
  message: ChatCompletionMessage;
  finish_reason: string;
  logprobs?: unknown | null;
}

export interface ChatCompletion {
  id: string;
  object: 'chat.completion';
  created: number;
  model: string;
  choices: Choice[];
  usage: CompletionUsage;
  system_fingerprint?: string | null;
  service_tier?: string | null;
  // 2api extensions
  provider?: string;
  _2api?: Record<string, unknown>;
}

// ============================================================
// Streaming Types
// ============================================================

export interface ChoiceDelta {
  role?: 'assistant';
  content?: string | null;
  tool_calls?: ToolCall[];
  refusal?: string | null;
}

export interface StreamChoice {
  index: number;
  delta: ChoiceDelta;
  finish_reason: string | null;
  logprobs?: unknown | null;
}

export interface ChatCompletionChunk {
  id: string;
  object: 'chat.completion.chunk';
  created: number;
  model: string;
  choices: StreamChoice[];
  system_fingerprint?: string | null;
  service_tier?: string | null;
  usage?: CompletionUsage | null;
}

// ============================================================
// Request Types
// ============================================================

export interface ChatCompletionCreateParams {
  model: string;
  messages: Message[];
  temperature?: number;
  max_tokens?: number;
  top_p?: number;
  frequency_penalty?: number;
  presence_penalty?: number;
  stop?: string | string[];
  n?: number;
  stream?: boolean;
  tools?: Tool[];
  tool_choice?: 'auto' | 'none' | 'required' | { type: 'function'; function: { name: string } };
  response_format?: { type: 'text' | 'json_object' };
  seed?: number;
  user?: string;
  // 2api extensions
  routing?: {
    strategy?: 'cost' | 'latency' | 'quality';
    fallback?: string[];
    max_latency_ms?: number;
    max_cost?: number;
  };
}

// ============================================================
// Parsing Functions
// ============================================================

export function parseChatCompletion(data: Record<string, unknown>): ChatCompletion {
  const choices: Choice[] = [];

  for (const choiceData of (data.choices as Record<string, unknown>[]) || []) {
    const msgData = (choiceData.message as Record<string, unknown>) || {};

    const toolCalls = msgData.tool_calls
      ? (msgData.tool_calls as unknown[]).map((tc) => {
          const tcData = tc as Record<string, unknown>;
          const func = tcData.function as Record<string, unknown>;
          return {
            id: tcData.id as string,
            type: 'function' as const,
            function: {
              name: func.name as string,
              arguments: func.arguments as string,
            },
          };
        })
      : undefined;

    choices.push({
      index: (choiceData.index as number) || 0,
      message: {
        role: 'assistant',
        content: (msgData.content as string) || null,
        tool_calls: toolCalls,
      },
      finish_reason: (choiceData.finish_reason as string) || 'stop',
      logprobs: choiceData.logprobs,
    });
  }

  const usageData = (data.usage as Record<string, unknown>) || {};

  return {
    id: (data.id as string) || '',
    object: 'chat.completion',
    created: (data.created as number) || Math.floor(Date.now() / 1000),
    model: (data.model as string) || '',
    choices,
    usage: {
      prompt_tokens: (usageData.prompt_tokens as number) || 0,
      completion_tokens: (usageData.completion_tokens as number) || 0,
      total_tokens: (usageData.total_tokens as number) || 0,
    },
    system_fingerprint: data.system_fingerprint as string | undefined,
    provider: (data.provider as string) || (data._2api as Record<string, unknown>)?.provider as string,
    _2api: data._2api as Record<string, unknown>,
  };
}

export function parseChatCompletionChunk(data: Record<string, unknown>): ChatCompletionChunk {
  const choices: StreamChoice[] = [];

  for (const choiceData of (data.choices as Record<string, unknown>[]) || []) {
    const deltaData = (choiceData.delta as Record<string, unknown>) || {};

    const toolCalls = deltaData.tool_calls
      ? (deltaData.tool_calls as unknown[]).map((tc) => {
          const tcData = tc as Record<string, unknown>;
          const func = (tcData.function as Record<string, unknown>) || {};
          return {
            id: (tcData.id as string) || '',
            type: 'function' as const,
            function: {
              name: (func.name as string) || '',
              arguments: (func.arguments as string) || '',
            },
          };
        })
      : undefined;

    choices.push({
      index: (choiceData.index as number) || 0,
      delta: {
        role: deltaData.role as 'assistant' | undefined,
        content: deltaData.content as string | undefined,
        tool_calls: toolCalls,
      },
      finish_reason: (choiceData.finish_reason as string) || null,
      logprobs: choiceData.logprobs,
    });
  }

  let usage: CompletionUsage | undefined;
  if (data.usage) {
    const usageData = data.usage as Record<string, unknown>;
    usage = {
      prompt_tokens: (usageData.prompt_tokens as number) || 0,
      completion_tokens: (usageData.completion_tokens as number) || 0,
      total_tokens: (usageData.total_tokens as number) || 0,
    };
  }

  return {
    id: (data.id as string) || '',
    object: 'chat.completion.chunk',
    created: (data.created as number) || Math.floor(Date.now() / 1000),
    model: (data.model as string) || '',
    choices,
    system_fingerprint: data.system_fingerprint as string | undefined,
    usage,
  };
}
