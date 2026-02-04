/**
 * 2api.ai SDK - Tool Calling Helpers
 *
 * Utilities for implementing tool calling loops.
 *
 * @example
 * import { TwoAPI, ToolRunner, tool } from 'twoapi';
 *
 * // Define tools
 * const getWeather = tool({
 *   name: 'get_weather',
 *   description: 'Get the current weather',
 *   parameters: {
 *     type: 'object',
 *     properties: {
 *       location: { type: 'string', description: 'City name' }
 *     },
 *     required: ['location']
 *   },
 *   execute: async (args) => `Weather in ${args.location}: 22C, sunny`
 * });
 *
 * // Create runner and execute
 * const runner = new ToolRunner([getWeather]);
 * const client = new TwoAPI();
 * const result = await runner.run(client, 'What is the weather in Tokyo?');
 * console.log(result.response.content);
 */

import type { TwoAPI } from './client';
import type { Message, Tool, ToolCall, FunctionParameters } from './types';
import type { ChatResponse } from './types';

// ============================================================
// Types
// ============================================================

export interface ToolFunctionConfig<TArgs = Record<string, unknown>> {
  name: string;
  description?: string;
  parameters?: FunctionParameters;
  execute: (args: TArgs) => string | Promise<string>;
}

export interface ToolFunction<TArgs = Record<string, unknown>> {
  name: string;
  description: string;
  parameters: FunctionParameters;
  tool: Tool;
  execute: (args: TArgs) => Promise<string>;
}

export interface ToolResult {
  toolCallId: string;
  name: string;
  result: string;
  error?: string;
}

export interface RunResult {
  response: ChatResponse;
  toolCallsMade: ToolResult[];
  totalIterations: number;
  messages: Message[];
}

// ============================================================
// Tool Creation
// ============================================================

/**
 * Create a tool function from a configuration object.
 */
export function tool<TArgs = Record<string, unknown>>(
  config: ToolFunctionConfig<TArgs>
): ToolFunction<TArgs> {
  const parameters: FunctionParameters = config.parameters || {
    type: 'object',
    properties: {},
    required: [],
  };

  return {
    name: config.name,
    description: config.description || '',
    parameters,
    tool: {
      type: 'function',
      function: {
        name: config.name,
        description: config.description,
        parameters,
      },
    },
    execute: async (args: TArgs) => {
      const result = config.execute(args);
      return result instanceof Promise ? result : result;
    },
  };
}

/**
 * Create a tool from a simple function.
 */
export function createTool<TArgs = Record<string, unknown>>(
  name: string,
  description: string,
  parameters: FunctionParameters,
  execute: (args: TArgs) => string | Promise<string>
): ToolFunction<TArgs> {
  return tool({ name, description, parameters, execute });
}

// ============================================================
// Tool Runner
// ============================================================

export class ToolRunner {
  private toolsMap: Map<string, ToolFunction>;
  private toolsList: Tool[];
  private maxIterations: number;
  private onToolCall?: (name: string, args: string, result: string) => void;

  /**
   * Create a new ToolRunner.
   *
   * @param tools - List of tool functions
   * @param options - Configuration options
   */
  constructor(
    tools: ToolFunction[],
    options: {
      maxIterations?: number;
      onToolCall?: (name: string, args: string, result: string) => void;
    } = {}
  ) {
    this.toolsMap = new Map();
    this.toolsList = [];
    this.maxIterations = options.maxIterations ?? 10;
    this.onToolCall = options.onToolCall;

    for (const t of tools) {
      this.toolsMap.set(t.name, t);
      this.toolsList.push(t.tool);
    }
  }

  /**
   * Execute a single tool call.
   */
  async executeToolCall(toolCall: ToolCall): Promise<ToolResult> {
    const name = toolCall.function.name;
    const argsString = toolCall.function.arguments;

    const toolFunc = this.toolsMap.get(name);
    if (!toolFunc) {
      return {
        toolCallId: toolCall.id,
        name,
        result: '',
        error: `Unknown tool: ${name}`,
      };
    }

    try {
      let args: Record<string, unknown>;
      try {
        args = JSON.parse(argsString);
      } catch {
        return {
          toolCallId: toolCall.id,
          name,
          result: '',
          error: `Invalid JSON arguments: ${argsString}`,
        };
      }

      const result = await toolFunc.execute(args);

      if (this.onToolCall) {
        this.onToolCall(name, argsString, result);
      }

      return {
        toolCallId: toolCall.id,
        name,
        result,
      };
    } catch (error) {
      return {
        toolCallId: toolCall.id,
        name,
        result: '',
        error: error instanceof Error ? error.message : String(error),
      };
    }
  }

  /**
   * Run a complete tool calling conversation.
   *
   * @param client - TwoAPI client instance
   * @param message - User message or Message object
   * @param options - Additional options
   * @returns RunResult with final response and tool call history
   */
  async run(
    client: TwoAPI,
    message: string | Message,
    options: {
      messages?: Message[];
      model?: string;
      system?: string;
      [key: string]: unknown;
    } = {}
  ): Promise<RunResult> {
    const { messages: existingMessages, model = 'auto', system, ...kwargs } = options;

    // Build initial messages
    const conversation: Message[] = [];
    if (existingMessages) {
      conversation.push(...existingMessages);
    }

    if (typeof message === 'string') {
      conversation.push({ role: 'user', content: message });
    } else {
      conversation.push(message);
    }

    const toolCallsMade: ToolResult[] = [];
    let iterations = 0;
    let lastResponse: ChatResponse | undefined;

    while (iterations < this.maxIterations) {
      iterations++;

      // Make API call using the simple chat method
      const response = await client.chat(conversation, {
        model,
        system: iterations === 1 ? system : undefined,
        tools: this.toolsList,
        ...kwargs,
      });

      lastResponse = response;

      // Check if response has tool calls
      if (!response.tool_calls || response.tool_calls.length === 0) {
        // Final response
        return {
          response,
          toolCallsMade,
          totalIterations: iterations,
          messages: conversation,
        };
      }

      // Add assistant message with tool calls
      conversation.push({
        role: 'assistant',
        content: response.content,
        tool_calls: response.tool_calls,
      });

      // Execute tool calls
      for (const toolCall of response.tool_calls) {
        const result = await this.executeToolCall(toolCall);
        toolCallsMade.push(result);

        // Add tool result message
        const content = result.error ? `Error: ${result.error}` : result.result;
        conversation.push({
          role: 'tool',
          content,
          tool_call_id: toolCall.id,
        });
      }
    }

    // Max iterations reached
    return {
      response: lastResponse!,
      toolCallsMade,
      totalIterations: iterations,
      messages: conversation,
    };
  }
}
