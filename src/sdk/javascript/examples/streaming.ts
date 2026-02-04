/**
 * 2api.ai JavaScript SDK - Streaming Example
 *
 * Demonstrates streaming responses for real-time output.
 */

import { TwoAPI } from 'twoapi';

async function main() {
  const client = new TwoAPI();

  // ============================================================
  // Simple Streaming
  // ============================================================
  console.log('=== Simple Streaming ===\n');

  process.stdout.write('Response: ');
  for await (const chunk of client.chatStream('Tell me a short story about a robot')) {
    process.stdout.write(chunk.content);
    if (chunk.done) {
      console.log('\n[Stream complete]');
    }
  }
  console.log();

  // ============================================================
  // OpenAI-Compatible Streaming
  // ============================================================
  console.log('=== OpenAI-Compatible Streaming ===\n');

  const stream = await client.chat.completions.create({
    model: 'openai/gpt-4o',
    messages: [{ role: 'user', content: 'Count from 1 to 10, one number per line' }],
    stream: true,
  });

  process.stdout.write('Counting: ');
  for await (const chunk of stream) {
    const content = chunk.choices[0]?.delta?.content || '';
    process.stdout.write(content);

    if (chunk.choices[0]?.finishReason) {
      console.log(`\n[Finished: ${chunk.choices[0].finishReason}]`);
    }
  }
  console.log();

  // ============================================================
  // Streaming with Options
  // ============================================================
  console.log('=== Streaming with Temperature ===\n');

  process.stdout.write('Creative response: ');
  for await (const chunk of client.chatStream('Give me a creative name for a coffee shop', {
    model: 'anthropic/claude-3-5-sonnet',
    temperature: 0.9,
    maxTokens: 50,
  })) {
    process.stdout.write(chunk.content);
  }
  console.log('\n');

  // ============================================================
  // Collecting Stream Results
  // ============================================================
  console.log('=== Collecting Stream Results ===\n');

  let fullContent = '';
  let tokenCount = 0;

  for await (const chunk of client.chatStream('What are the three primary colors?')) {
    fullContent += chunk.content;
    tokenCount++;
  }

  console.log('Full response:', fullContent);
  console.log('Chunks received:', tokenCount);
}

main().catch(console.error);
