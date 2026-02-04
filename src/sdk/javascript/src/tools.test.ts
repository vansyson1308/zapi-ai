/**
 * 2api.ai JavaScript SDK - Tool Calling Tests
 */

import { describe, it, expect, vi } from 'vitest';
import { tool, createTool, ToolRunner } from './tools';
import type { TwoAPI } from './client';

describe('Tool Helper Functions', () => {
  describe('tool()', () => {
    it('should create a tool function from config', () => {
      const getWeather = tool({
        name: 'get_weather',
        description: 'Get weather for a location',
        parameters: {
          type: 'object',
          properties: {
            location: { type: 'string', description: 'City name' },
          },
          required: ['location'],
        },
        execute: async (args) => `Weather in ${args.location}: sunny`,
      });

      expect(getWeather.name).toBe('get_weather');
      expect(getWeather.description).toBe('Get weather for a location');
      expect(getWeather.tool.type).toBe('function');
      expect(getWeather.tool.function.name).toBe('get_weather');
    });

    it('should execute tool function', async () => {
      const addNumbers = tool({
        name: 'add',
        description: 'Add two numbers',
        parameters: {
          type: 'object',
          properties: {
            a: { type: 'number' },
            b: { type: 'number' },
          },
        },
        execute: async (args: { a: number; b: number }) => String(args.a + args.b),
      });

      const result = await addNumbers.execute({ a: 5, b: 3 });
      expect(result).toBe('8');
    });

    it('should handle sync execute functions', async () => {
      const syncTool = tool({
        name: 'sync_tool',
        execute: () => 'sync result',
      });

      const result = await syncTool.execute({});
      expect(result).toBe('sync result');
    });

    it('should provide default parameters when not specified', () => {
      const simpleTool = tool({
        name: 'simple',
        execute: () => 'result',
      });

      expect(simpleTool.parameters).toEqual({
        type: 'object',
        properties: {},
        required: [],
      });
    });
  });

  describe('createTool()', () => {
    it('should create tool with positional arguments', () => {
      const myTool = createTool(
        'my_tool',
        'My tool description',
        {
          type: 'object',
          properties: {
            input: { type: 'string' },
          },
        },
        async (args) => `Processed: ${args.input}`
      );

      expect(myTool.name).toBe('my_tool');
      expect(myTool.description).toBe('My tool description');
    });
  });
});

describe('ToolRunner', () => {
  const mockClient = {
    chat: vi.fn(),
  } as unknown as TwoAPI;

  const weatherTool = tool({
    name: 'get_weather',
    description: 'Get weather',
    parameters: {
      type: 'object',
      properties: {
        location: { type: 'string' },
      },
    },
    execute: async (args: { location: string }) => `Weather in ${args.location}: 22C, sunny`,
  });

  const searchTool = tool({
    name: 'search',
    description: 'Search the web',
    parameters: {
      type: 'object',
      properties: {
        query: { type: 'string' },
      },
    },
    execute: async (args: { query: string }) => `Results for: ${args.query}`,
  });

  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('Constructor', () => {
    it('should create runner with tools', () => {
      const runner = new ToolRunner([weatherTool, searchTool]);
      expect(runner).toBeDefined();
    });

    it('should accept maxIterations option', () => {
      const runner = new ToolRunner([weatherTool], { maxIterations: 5 });
      expect(runner).toBeDefined();
    });

    it('should accept onToolCall callback', () => {
      const onToolCall = vi.fn();
      const runner = new ToolRunner([weatherTool], { onToolCall });
      expect(runner).toBeDefined();
    });
  });

  describe('executeToolCall()', () => {
    it('should execute a tool call', async () => {
      const runner = new ToolRunner([weatherTool]);

      const result = await runner.executeToolCall({
        id: 'call_123',
        type: 'function',
        function: {
          name: 'get_weather',
          arguments: '{"location":"Tokyo"}',
        },
      });

      expect(result.toolCallId).toBe('call_123');
      expect(result.name).toBe('get_weather');
      expect(result.result).toBe('Weather in Tokyo: 22C, sunny');
      expect(result.error).toBeUndefined();
    });

    it('should return error for unknown tool', async () => {
      const runner = new ToolRunner([weatherTool]);

      const result = await runner.executeToolCall({
        id: 'call_456',
        type: 'function',
        function: {
          name: 'unknown_tool',
          arguments: '{}',
        },
      });

      expect(result.error).toContain('Unknown tool');
    });

    it('should return error for invalid JSON arguments', async () => {
      const runner = new ToolRunner([weatherTool]);

      const result = await runner.executeToolCall({
        id: 'call_789',
        type: 'function',
        function: {
          name: 'get_weather',
          arguments: 'invalid json',
        },
      });

      expect(result.error).toContain('Invalid JSON');
    });

    it('should catch and return execution errors', async () => {
      const errorTool = tool({
        name: 'error_tool',
        execute: async () => {
          throw new Error('Execution failed');
        },
      });

      const runner = new ToolRunner([errorTool]);

      const result = await runner.executeToolCall({
        id: 'call_err',
        type: 'function',
        function: {
          name: 'error_tool',
          arguments: '{}',
        },
      });

      expect(result.error).toBe('Execution failed');
    });

    it('should call onToolCall callback', async () => {
      const onToolCall = vi.fn();
      const runner = new ToolRunner([weatherTool], { onToolCall });

      await runner.executeToolCall({
        id: 'call_123',
        type: 'function',
        function: {
          name: 'get_weather',
          arguments: '{"location":"Tokyo"}',
        },
      });

      expect(onToolCall).toHaveBeenCalledWith(
        'get_weather',
        '{"location":"Tokyo"}',
        'Weather in Tokyo: 22C, sunny'
      );
    });
  });

  describe('run()', () => {
    it('should handle simple message without tool calls', async () => {
      mockClient.chat = vi.fn().mockResolvedValue({
        content: 'Hello!',
        model: 'gpt-4o',
        provider: 'openai',
      });

      const runner = new ToolRunner([weatherTool]);
      const result = await runner.run(mockClient, 'Hello');

      expect(result.response.content).toBe('Hello!');
      expect(result.toolCallsMade).toHaveLength(0);
      expect(result.totalIterations).toBe(1);
    });

    it('should execute tool calls and continue conversation', async () => {
      // First call returns tool call
      mockClient.chat = vi.fn()
        .mockResolvedValueOnce({
          content: '',
          tool_calls: [{
            id: 'call_1',
            type: 'function',
            function: {
              name: 'get_weather',
              arguments: '{"location":"Tokyo"}',
            },
          }],
        })
        // Second call returns final response
        .mockResolvedValueOnce({
          content: 'The weather in Tokyo is 22C and sunny!',
          model: 'gpt-4o',
        });

      const runner = new ToolRunner([weatherTool]);
      const result = await runner.run(mockClient, 'What is the weather in Tokyo?');

      expect(result.response.content).toBe('The weather in Tokyo is 22C and sunny!');
      expect(result.toolCallsMade).toHaveLength(1);
      expect(result.toolCallsMade[0].name).toBe('get_weather');
      expect(result.totalIterations).toBe(2);
    });

    it('should handle multiple tool calls in one response', async () => {
      mockClient.chat = vi.fn()
        .mockResolvedValueOnce({
          content: '',
          tool_calls: [
            {
              id: 'call_1',
              type: 'function',
              function: {
                name: 'get_weather',
                arguments: '{"location":"Tokyo"}',
              },
            },
            {
              id: 'call_2',
              type: 'function',
              function: {
                name: 'search',
                arguments: '{"query":"Tokyo attractions"}',
              },
            },
          ],
        })
        .mockResolvedValueOnce({
          content: 'Here is the info about Tokyo...',
        });

      const runner = new ToolRunner([weatherTool, searchTool]);
      const result = await runner.run(mockClient, 'Tell me about Tokyo');

      expect(result.toolCallsMade).toHaveLength(2);
      expect(result.messages.length).toBeGreaterThan(1);
    });

    it('should respect maxIterations', async () => {
      // Always return tool calls (infinite loop scenario)
      mockClient.chat = vi.fn().mockResolvedValue({
        content: '',
        tool_calls: [{
          id: `call_${Math.random()}`,
          type: 'function',
          function: {
            name: 'get_weather',
            arguments: '{"location":"Tokyo"}',
          },
        }],
      });

      const runner = new ToolRunner([weatherTool], { maxIterations: 3 });
      const result = await runner.run(mockClient, 'Loop forever');

      expect(result.totalIterations).toBe(3);
    });

    it('should pass existing messages to client', async () => {
      mockClient.chat = vi.fn().mockResolvedValue({
        content: 'Response',
      });

      const runner = new ToolRunner([weatherTool]);
      await runner.run(mockClient, 'New message', {
        messages: [
          { role: 'user', content: 'Previous message' },
          { role: 'assistant', content: 'Previous response' },
        ],
      });

      expect(mockClient.chat).toHaveBeenCalledWith(
        expect.arrayContaining([
          expect.objectContaining({ content: 'Previous message' }),
          expect.objectContaining({ content: 'Previous response' }),
          expect.objectContaining({ content: 'New message' }),
        ]),
        expect.any(Object)
      );
    });

    it('should pass model and system options', async () => {
      mockClient.chat = vi.fn().mockResolvedValue({
        content: 'Response',
      });

      const runner = new ToolRunner([weatherTool]);
      await runner.run(mockClient, 'Hello', {
        model: 'anthropic/claude-3-opus',
        system: 'You are a helpful assistant',
      });

      expect(mockClient.chat).toHaveBeenCalledWith(
        expect.any(Array),
        expect.objectContaining({
          model: 'anthropic/claude-3-opus',
          system: 'You are a helpful assistant',
        })
      );
    });

    it('should include tool results in messages', async () => {
      mockClient.chat = vi.fn()
        .mockResolvedValueOnce({
          content: '',
          tool_calls: [{
            id: 'call_1',
            type: 'function',
            function: {
              name: 'get_weather',
              arguments: '{"location":"Tokyo"}',
            },
          }],
        })
        .mockResolvedValueOnce({
          content: 'Final response',
        });

      const runner = new ToolRunner([weatherTool]);
      const result = await runner.run(mockClient, 'Weather?');

      // Check that tool result message was added
      const toolMessage = result.messages.find(m => m.role === 'tool');
      expect(toolMessage).toBeDefined();
      expect(toolMessage?.content).toContain('22C');
      expect(toolMessage?.tool_call_id).toBe('call_1');
    });

    it('should include error messages for failed tool calls', async () => {
      const failingTool = tool({
        name: 'failing_tool',
        execute: async () => {
          throw new Error('Tool failed');
        },
      });

      mockClient.chat = vi.fn()
        .mockResolvedValueOnce({
          content: '',
          tool_calls: [{
            id: 'call_1',
            type: 'function',
            function: {
              name: 'failing_tool',
              arguments: '{}',
            },
          }],
        })
        .mockResolvedValueOnce({
          content: 'I encountered an error',
        });

      const runner = new ToolRunner([failingTool]);
      const result = await runner.run(mockClient, 'Use the tool');

      const toolMessage = result.messages.find(m => m.role === 'tool');
      expect(toolMessage?.content).toContain('Error');
    });
  });
});
