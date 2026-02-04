/**
 * 2api.ai JavaScript SDK - Basic Chat Example
 *
 * Demonstrates the simplest way to use the SDK for chat completions.
 */

import { TwoAPI } from 'twoapi';

async function main() {
  // Create client (uses TWOAPI_API_KEY env var by default)
  const client = new TwoAPI({
    apiKey: process.env.TWOAPI_API_KEY,
  });

  // ============================================================
  // Simple Chat
  // ============================================================
  console.log('=== Simple Chat ===\n');

  // Single message
  const response = await client.chat('What is the capital of France?');
  console.log('Response:', response.content);
  console.log('Model used:', response.model);
  console.log('Provider:', response.provider);
  console.log();

  // ============================================================
  // Chat with Options
  // ============================================================
  console.log('=== Chat with Options ===\n');

  // Specify model and parameters
  const response2 = await client.chat('Explain quantum computing in one sentence', {
    model: 'anthropic/claude-3-5-sonnet',
    temperature: 0.7,
    maxTokens: 100,
  });
  console.log('Response:', response2.content);
  console.log();

  // ============================================================
  // Chat with System Prompt
  // ============================================================
  console.log('=== Chat with System Prompt ===\n');

  const response3 = await client.chat('Write a haiku about programming', {
    system: 'You are a creative poet who writes in traditional Japanese forms.',
    temperature: 0.9,
  });
  console.log('Haiku:\n', response3.content);
  console.log();

  // ============================================================
  // Multi-turn Conversation
  // ============================================================
  console.log('=== Multi-turn Conversation ===\n');

  // Build conversation history
  const messages = [
    { role: 'user' as const, content: 'My name is Alice.' },
    { role: 'assistant' as const, content: 'Nice to meet you, Alice! How can I help you today?' },
    { role: 'user' as const, content: 'What is my name?' },
  ];

  const response4 = await client.chat(messages);
  console.log('Response:', response4.content);
  console.log();

  // ============================================================
  // OpenAI-Compatible API
  // ============================================================
  console.log('=== OpenAI-Compatible API ===\n');

  // Use the familiar OpenAI API format
  const completion = await client.chat.completions.create({
    model: 'openai/gpt-4o',
    messages: [
      { role: 'system', content: 'You are a helpful assistant.' },
      { role: 'user', content: 'Hello!' },
    ],
    temperature: 0.7,
    maxTokens: 100,
  });

  console.log('Response:', completion.choices[0].message.content);
  console.log('Finish reason:', completion.choices[0].finishReason);
  console.log('Usage:', completion.usage);
}

main().catch(console.error);
