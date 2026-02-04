/**
 * 2api.ai JavaScript SDK - Client Tests
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { TwoAPI } from './client';
import {
  TwoAPIError,
  AuthenticationError,
  RateLimitError,
  InvalidRequestError,
  ProviderError,
  TimeoutError,
  ConnectionError,
} from './errors';

// Mock fetch globally
const mockFetch = vi.fn();
global.fetch = mockFetch;

describe('TwoAPI Client', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    // Set environment variable
    process.env.TWOAPI_API_KEY = 'test_api_key';
  });

  afterEach(() => {
    delete process.env.TWOAPI_API_KEY;
  });

  describe('Constructor', () => {
    it('should create client with API key from config', () => {
      delete process.env.TWOAPI_API_KEY;
      const client = new TwoAPI({ apiKey: 'config_key' });
      expect(client).toBeDefined();
    });

    it('should create client with API key from environment', () => {
      const client = new TwoAPI();
      expect(client).toBeDefined();
    });

    it('should throw error when no API key provided', () => {
      delete process.env.TWOAPI_API_KEY;
      expect(() => new TwoAPI()).toThrow(AuthenticationError);
    });

    it('should use custom base URL', () => {
      const client = new TwoAPI({
        apiKey: 'test',
        baseUrl: 'https://custom.api.com/v1',
      });
      expect(client).toBeDefined();
    });
  });

  describe('Simple Chat API', () => {
    it('should send simple chat message', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({
          choices: [{
            message: { role: 'assistant', content: 'Hello!' },
            finish_reason: 'stop',
          }],
          model: 'gpt-4o',
          provider: 'openai',
          usage: {
            prompt_tokens: 10,
            completion_tokens: 5,
            total_tokens: 15,
          },
        }),
      });

      const client = new TwoAPI({ apiKey: 'test' });
      const response = await client.chat('Hello');

      expect(response.content).toBe('Hello!');
      expect(response.role).toBe('assistant');
      expect(response.model).toBe('gpt-4o');
      expect(response.provider).toBe('openai');
    });

    it('should send chat with options', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({
          choices: [{
            message: { role: 'assistant', content: 'Response' },
            finish_reason: 'stop',
          }],
          model: 'claude-3-sonnet',
          provider: 'anthropic',
        }),
      });

      const client = new TwoAPI({ apiKey: 'test' });
      const response = await client.chat('Hello', {
        model: 'anthropic/claude-3-sonnet',
        temperature: 0.7,
        maxTokens: 100,
        system: 'You are helpful.',
      });

      expect(mockFetch).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({
          method: 'POST',
          body: expect.stringContaining('"temperature":0.7'),
        })
      );
    });

    it('should handle array of messages', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({
          choices: [{
            message: { role: 'assistant', content: 'Response' },
            finish_reason: 'stop',
          }],
        }),
      });

      const client = new TwoAPI({ apiKey: 'test' });
      await client.chat([
        { role: 'user', content: 'Hello' },
        { role: 'assistant', content: 'Hi!' },
        { role: 'user', content: 'How are you?' },
      ]);

      expect(mockFetch).toHaveBeenCalled();
    });

    it('should return tool calls when present', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({
          choices: [{
            message: {
              role: 'assistant',
              content: null,
              tool_calls: [{
                id: 'call_123',
                type: 'function',
                function: {
                  name: 'get_weather',
                  arguments: '{"location":"Tokyo"}',
                },
              }],
            },
            finish_reason: 'tool_calls',
          }],
        }),
      });

      const client = new TwoAPI({ apiKey: 'test' });
      const response = await client.chat('What is the weather in Tokyo?', {
        tools: [{
          type: 'function',
          function: {
            name: 'get_weather',
            description: 'Get weather',
            parameters: {
              type: 'object',
              properties: {
                location: { type: 'string', description: 'City' },
              },
            },
          },
        }],
      });

      expect(response.tool_calls).toHaveLength(1);
      expect(response.tool_calls![0].function.name).toBe('get_weather');
    });
  });

  describe('OpenAI-Compatible API', () => {
    it('should create completion via chat.completions.create', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({
          id: 'chatcmpl-123',
          object: 'chat.completion',
          created: 1234567890,
          model: 'gpt-4o',
          choices: [{
            index: 0,
            message: { role: 'assistant', content: 'Hello!' },
            finish_reason: 'stop',
          }],
          usage: {
            prompt_tokens: 10,
            completion_tokens: 5,
            total_tokens: 15,
          },
        }),
      });

      const client = new TwoAPI({ apiKey: 'test' });
      const response = await client.chat.completions.create({
        model: 'openai/gpt-4o',
        messages: [{ role: 'user', content: 'Hello' }],
      });

      expect(response.id).toBe('chatcmpl-123');
      expect(response.choices[0].message.content).toBe('Hello!');
      expect(response.usage.totalTokens).toBe(15);
    });

    it('should support all OpenAI parameters', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({
          choices: [{ message: { content: 'Response' }, finish_reason: 'stop' }],
          usage: { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 },
        }),
      });

      const client = new TwoAPI({ apiKey: 'test' });
      await client.chat.completions.create({
        model: 'gpt-4o',
        messages: [{ role: 'user', content: 'Hello' }],
        temperature: 0.8,
        maxTokens: 100,
        topP: 0.9,
        frequencyPenalty: 0.5,
        presencePenalty: 0.5,
        stop: ['END'],
        seed: 42,
      });

      const body = JSON.parse(mockFetch.mock.calls[0][1].body);
      expect(body.temperature).toBe(0.8);
      expect(body.max_tokens).toBe(100);
      expect(body.top_p).toBe(0.9);
      expect(body.frequency_penalty).toBe(0.5);
      expect(body.presence_penalty).toBe(0.5);
      expect(body.stop).toEqual(['END']);
      expect(body.seed).toBe(42);
    });
  });

  describe('Embeddings API', () => {
    it('should create embeddings', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({
          data: [{ embedding: [0.1, 0.2, 0.3] }],
          model: 'text-embedding-3-small',
          provider: 'openai',
          usage: { prompt_tokens: 5, total_tokens: 5 },
        }),
      });

      const client = new TwoAPI({ apiKey: 'test' });
      const response = await client.embed('Hello world');

      expect(response.embeddings).toHaveLength(1);
      expect(response.embeddings[0]).toEqual([0.1, 0.2, 0.3]);
      expect(response.model).toBe('text-embedding-3-small');
    });

    it('should create batch embeddings', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({
          data: [
            { embedding: [0.1, 0.2] },
            { embedding: [0.3, 0.4] },
          ],
          model: 'text-embedding-3-small',
          provider: 'openai',
        }),
      });

      const client = new TwoAPI({ apiKey: 'test' });
      const response = await client.embed(['Hello', 'World']);

      expect(response.embeddings).toHaveLength(2);
    });
  });

  describe('Images API', () => {
    it('should generate images', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({
          data: [{
            url: 'https://example.com/image.png',
            revised_prompt: 'A beautiful sunset',
          }],
          model: 'dall-e-3',
          provider: 'openai',
        }),
      });

      const client = new TwoAPI({ apiKey: 'test' });
      const response = await client.generateImage('A sunset');

      expect(response.images).toHaveLength(1);
      expect(response.images[0].url).toBe('https://example.com/image.png');
      expect(response.model).toBe('dall-e-3');
    });
  });

  describe('Models API', () => {
    it('should list models', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({
          data: [
            { id: 'openai/gpt-4o', provider: 'openai', name: 'GPT-4o' },
            { id: 'anthropic/claude-3-sonnet', provider: 'anthropic', name: 'Claude 3 Sonnet' },
          ],
        }),
      });

      const client = new TwoAPI({ apiKey: 'test' });
      const models = await client.listModels();

      expect(models).toHaveLength(2);
      expect(models[0].id).toBe('openai/gpt-4o');
    });

    it('should get single model', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({
          id: 'openai/gpt-4o',
          provider: 'openai',
          name: 'GPT-4o',
          context_length: 128000,
        }),
      });

      const client = new TwoAPI({ apiKey: 'test' });
      const model = await client.getModel('openai/gpt-4o');

      expect(model.id).toBe('openai/gpt-4o');
      expect(model.contextLength).toBe(128000);
    });
  });

  describe('Error Handling', () => {
    it('should throw AuthenticationError on 401', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 401,
        json: () => Promise.resolve({ error: { message: 'Invalid API key' } }),
      });

      const client = new TwoAPI({ apiKey: 'invalid' });
      await expect(client.chat('Hello')).rejects.toThrow(AuthenticationError);
    });

    it('should throw RateLimitError on 429', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 429,
        headers: { get: () => '60' },
        json: () => Promise.resolve({ error: { message: 'Rate limit exceeded' } }),
      });

      const client = new TwoAPI({ apiKey: 'test', maxRetries: 0 });
      await expect(client.chat('Hello')).rejects.toThrow(RateLimitError);
    });

    it('should throw InvalidRequestError on 400', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 400,
        json: () => Promise.resolve({ error: { message: 'Invalid request' } }),
      });

      const client = new TwoAPI({ apiKey: 'test' });
      await expect(client.chat('Hello')).rejects.toThrow(InvalidRequestError);
    });

    it('should throw ProviderError on 500', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 500,
        json: () => Promise.resolve({ error: { message: 'Provider error', provider: 'openai' } }),
      });

      const client = new TwoAPI({ apiKey: 'test', maxRetries: 0 });
      await expect(client.chat('Hello')).rejects.toThrow(ProviderError);
    });
  });

  describe('Retry Logic', () => {
    it('should retry on 500 errors', async () => {
      mockFetch
        .mockResolvedValueOnce({
          ok: false,
          status: 500,
          json: () => Promise.resolve({ error: { message: 'Server error' } }),
        })
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve({
            choices: [{ message: { content: 'Success!' }, finish_reason: 'stop' }],
          }),
        });

      const client = new TwoAPI({ apiKey: 'test', maxRetries: 3 });
      const response = await client.chat('Hello');

      expect(response.content).toBe('Success!');
      expect(mockFetch).toHaveBeenCalledTimes(2);
    });

    it('should not retry on 400 errors', async () => {
      mockFetch.mockResolvedValue({
        ok: false,
        status: 400,
        json: () => Promise.resolve({ error: { message: 'Bad request' } }),
      });

      const client = new TwoAPI({ apiKey: 'test', maxRetries: 3 });
      await expect(client.chat('Hello')).rejects.toThrow(InvalidRequestError);
      expect(mockFetch).toHaveBeenCalledTimes(1);
    });

    it('should call onRetry callback', async () => {
      const onRetry = vi.fn();

      mockFetch
        .mockResolvedValueOnce({
          ok: false,
          status: 500,
          json: () => Promise.resolve({ error: { message: 'Error' } }),
        })
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve({
            choices: [{ message: { content: 'OK' }, finish_reason: 'stop' }],
          }),
        });

      const client = new TwoAPI({ apiKey: 'test', maxRetries: 3, onRetry });
      await client.chat('Hello');

      expect(onRetry).toHaveBeenCalledTimes(1);
      expect(onRetry).toHaveBeenCalledWith(0, expect.any(Error), expect.any(Number));
    });
  });

  describe('Health API', () => {
    it('should return health status', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({
          status: 'healthy',
          version: '1.0.0',
          providers: {
            openai: { status: 'healthy', latency_ms: 50 },
            anthropic: { status: 'healthy', latency_ms: 60 },
          },
        }),
      });

      const client = new TwoAPI({ apiKey: 'test' });
      const health = await client.health();

      expect(health.status).toBe('healthy');
      expect(health.providers.openai.status).toBe('healthy');
    });
  });
});
