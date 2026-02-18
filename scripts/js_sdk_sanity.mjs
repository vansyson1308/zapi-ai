#!/usr/bin/env node
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, '..');
const sdkDir = path.join(repoRoot, 'src', 'sdk', 'javascript');
const pkgPath = path.join(sdkDir, 'package.json');
const sourceEntry = path.join(sdkDir, 'src', 'index.ts');
const streamFixture = path.join(repoRoot, 'tests', 'fixtures', 'js_sdk_stream.sse');

function assert(condition, message) {
  if (!condition) {
    throw new Error(message);
  }
}

function parseSSE(raw) {
  const payloads = [];
  let done = false;

  for (const line of raw.split(/\r?\n/)) {
    if (!line.startsWith('data: ')) {
      continue;
    }
    const payload = line.slice(6);
    payloads.push(payload);
    if (payload === '[DONE]') {
      done = true;
      break;
    }
  }

  return { payloads, done };
}

function main() {
  assert(fs.existsSync(pkgPath), 'package.json missing for JS SDK');
  assert(fs.existsSync(sourceEntry), 'JS SDK source entry missing: src/index.ts');

  const pkg = JSON.parse(fs.readFileSync(pkgPath, 'utf8'));
  assert(Boolean(pkg.name), 'package name is required');
  assert(Boolean(pkg.main), 'package main is required');
  assert(Boolean(pkg.module), 'package module is required');
  assert(Boolean(pkg.types), 'package types is required');

  // Offline-friendly packaging sanity.
  const packJson = execFileSync('npm', ['pack', '--dry-run', '--json'], {
    cwd: sdkDir,
    encoding: 'utf8',
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  const pack = JSON.parse(packJson);
  assert(Array.isArray(pack) && pack.length > 0, 'npm pack --dry-run returned empty result');

  // SSE parser sanity using a fixed fixture; verifies [DONE] termination behavior.
  const rawFixture = fs.readFileSync(streamFixture, 'utf8');
  const { payloads, done } = parseSSE(rawFixture);
  assert(payloads.length >= 2, 'expected at least one chunk and one terminal event');
  assert(done, 'expected [DONE] terminal event in SSE fixture');

  const firstChunk = JSON.parse(payloads[0]);
  assert(firstChunk.object === 'chat.completion.chunk', 'first SSE payload must be chat.completion.chunk');

  console.log('JS SDK sanity passed');
}

main();
