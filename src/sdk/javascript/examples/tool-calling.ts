/**
 * 2api.ai JavaScript SDK - Tool Calling Example
 *
 * Demonstrates how to use tools (function calling) with the SDK.
 */

import { TwoAPI, ToolRunner, tool } from 'twoapi';

// ============================================================
// Define Tools
// ============================================================

// Weather tool
const getWeather = tool({
  name: 'get_weather',
  description: 'Get the current weather for a location',
  parameters: {
    type: 'object',
    properties: {
      location: {
        type: 'string',
        description: 'City name (e.g., "Tokyo", "New York")',
      },
      unit: {
        type: 'string',
        enum: ['celsius', 'fahrenheit'],
        description: 'Temperature unit',
      },
    },
    required: ['location'],
  },
  execute: async (args: { location: string; unit?: string }) => {
    // Simulated weather data
    const weather = {
      Tokyo: { temp: 22, condition: 'sunny' },
      'New York': { temp: 15, condition: 'cloudy' },
      London: { temp: 12, condition: 'rainy' },
    };

    const data = weather[args.location as keyof typeof weather] || { temp: 20, condition: 'unknown' };
    const unit = args.unit || 'celsius';
    const temp = unit === 'fahrenheit' ? Math.round(data.temp * 9 / 5 + 32) : data.temp;

    return JSON.stringify({
      location: args.location,
      temperature: temp,
      unit,
      condition: data.condition,
    });
  },
});

// Calculator tool
const calculate = tool({
  name: 'calculate',
  description: 'Perform mathematical calculations',
  parameters: {
    type: 'object',
    properties: {
      expression: {
        type: 'string',
        description: 'Mathematical expression (e.g., "2 + 2", "sqrt(16)")',
      },
    },
    required: ['expression'],
  },
  execute: async (args: { expression: string }) => {
    try {
      // Simple eval for demo (use a proper math library in production)
      const result = Function(`'use strict'; return (${args.expression})`)();
      return `Result: ${result}`;
    } catch {
      return `Error: Could not evaluate expression`;
    }
  },
});

// Search tool
const searchWeb = tool({
  name: 'search_web',
  description: 'Search the web for information',
  parameters: {
    type: 'object',
    properties: {
      query: {
        type: 'string',
        description: 'Search query',
      },
    },
    required: ['query'],
  },
  execute: async (args: { query: string }) => {
    // Simulated search results
    return JSON.stringify({
      query: args.query,
      results: [
        { title: `Information about ${args.query}`, snippet: 'Relevant result...' },
        { title: `${args.query} - Wikipedia`, snippet: 'Encyclopedia entry...' },
      ],
    });
  },
});

async function main() {
  const client = new TwoAPI();

  // ============================================================
  // Manual Tool Calling
  // ============================================================
  console.log('=== Manual Tool Calling ===\n');

  // Make request with tools
  const response = await client.chat('What is the weather like in Tokyo?', {
    tools: [getWeather.tool],
  });

  if (response.tool_calls) {
    console.log('Tool calls requested:', response.tool_calls.length);

    for (const toolCall of response.tool_calls) {
      console.log(`  - ${toolCall.function.name}(${toolCall.function.arguments})`);

      // Execute the tool
      const args = JSON.parse(toolCall.function.arguments);
      const result = await getWeather.execute(args);
      console.log(`  Result: ${result}`);
    }
  }
  console.log();

  // ============================================================
  // Using ToolRunner
  // ============================================================
  console.log('=== Using ToolRunner ===\n');

  // Create a runner with all tools
  const runner = new ToolRunner([getWeather, calculate, searchWeb], {
    maxIterations: 5,
    onToolCall: (name, args, result) => {
      console.log(`  [Tool] ${name}(${args}) => ${result.substring(0, 50)}...`);
    },
  });

  // Run a query that requires tool use
  console.log('Query: "What is 15 + 27, and what is the weather in London?"');
  const result = await runner.run(
    client,
    'What is 15 + 27, and what is the weather in London?'
  );

  console.log('\nFinal response:', result.response.content);
  console.log('Tool calls made:', result.toolCallsMade.length);
  console.log('Total iterations:', result.totalIterations);
  console.log();

  // ============================================================
  // Multi-Tool Conversation
  // ============================================================
  console.log('=== Multi-Tool Conversation ===\n');

  // Run with system prompt
  const result2 = await runner.run(
    client,
    'Compare the weather in Tokyo and New York, and tell me which is warmer.',
    {
      system: 'You are a helpful weather assistant. Always check the weather before answering.',
      model: 'openai/gpt-4o',
    }
  );

  console.log('Response:', result2.response.content);
  console.log('Tools used:', result2.toolCallsMade.map(tc => tc.name).join(', '));
  console.log();

  // ============================================================
  // Conversation History with Tools
  // ============================================================
  console.log('=== Continuing Conversation ===\n');

  // Use the messages from the previous run to continue
  const result3 = await runner.run(
    client,
    'What if I need to know the weather in celsius for both?',
    {
      messages: result2.messages,
    }
  );

  console.log('Follow-up response:', result3.response.content);
}

main().catch(console.error);
