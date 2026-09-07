import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';
import ts from 'typescript';
import { streamConnection } from '../src/connection.mjs';

const source = readFileSync(new URL('../src/main.ts', import.meta.url), 'utf8');
const { outputText } = ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 },
});

function viewer() {
  const elements = new Map();
  const timers = new Map();
  let options;
  const element = (selector) => {
    if (!elements.has(selector)) elements.set(selector, {
      listeners: {},
      addEventListener(name, callback) { this.listeners[name] = callback; },
      focus() {},
    });
    return elements.get(selector);
  };
  const sdk = {
    AppStreamer: { connect: async (value) => { options = value; } },
    StreamType: { DIRECT: 'direct' },
    eStatus: { error: 'error' },
  };
  vm.runInNewContext(outputText, {
    exports: {}, console,
    require: (name) => {
      if (name === '@nvidia/omniverse-webrtc-streaming-library') return sdk;
      if (name === './connection.mjs') return { streamConnection };
      if (name === './style.css') return {};
      throw new Error(`Unexpected import: ${name}`);
    },
    document: { querySelector: element, addEventListener() {} },
    window: { addEventListener() {} },
    fetch: async () => ({ ok: true, json: async () => ({
      signalingServer: '127.0.0.1', signalingPort: 49100,
      mediaServer: '203.0.113.42', mediaPort: 31234,
    }) }),
    setTimeout: (callback) => { const id = Symbol(); timers.set(id, callback); return id; },
    clearTimeout: (id) => timers.delete(id),
  });
  return { element, timers, options: () => options };
}

test('reports connected only after video plays', async () => {
  const app = viewer();
  await app.element('#connect').listeners.click();
  assert.equal(app.element('#status').textContent, 'Connecting to Isaac Sim…');
  assert.equal(app.options().streamConfig.mediaPort, 31234);
  app.element('#remote-video').listeners.playing();
  assert.equal(app.element('#status').textContent, 'Connected');
  assert.equal(app.timers.size, 0);
});

for (const event of ['onStop', 'onTerminate']) {
  test(`${event} cancels the waiting message`, async () => {
    const app = viewer();
    await app.element('#connect').listeners.click();
    assert.equal(app.timers.size, 1);
    app.options().streamConfig[event]();
    assert.equal(app.timers.size, 0);
    assert.match(app.element('#status').textContent, /Stream (stopped|ended)/);
  });
}
