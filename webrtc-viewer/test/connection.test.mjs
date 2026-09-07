import { test } from 'node:test';
import assert from 'node:assert/strict';
import { streamConnection } from '../src/connection.mjs';

const connection = {
  signalingServer: '127.0.0.1', signalingPort: 49100,
  mediaServer: '203.0.113.42', mediaPort: 31234,
};

test('uses the mapped public media port while keeping signaling on SSH', () => {
  assert.deepEqual(streamConnection(connection), { ...connection, forceWSS: false });
});

test('rejects public signaling and malformed media endpoints', () => {
  for (const value of [null, {},
    { ...connection, signalingServer: '203.0.113.42' },
    { ...connection, mediaPort: '31234' },
    { ...connection, mediaPort: 70000 },
    { ...connection, mediaServer: '999.0.0.1' },
    { ...connection, mediaServer: 'example.com/path' },
  ]) assert.throws(() => streamConnection(value));
});
