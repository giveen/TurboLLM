// An engine error delivered inside a 200 SSE stream must reach the user, not end the turn as an empty message.
import assert from 'node:assert/strict'
import { test } from 'node:test'
import { litertLmTurnErrorMessage } from '../engines/litert-lm'
import { streamChunkError } from './chat-routes'

test('streamChunkError: reads both error shapes and ignores ordinary chunks', () => {
  assert.equal(streamChunkError({ error: 'RuntimeError: boom' }), 'RuntimeError: boom')
  assert.equal(streamChunkError({ error: { message: 'context full', code: 500 } }), 'context full')
  assert.equal(streamChunkError({ choices: [{ delta: { content: 'hi' } }] }), null)
  assert.equal(streamChunkError({ error: '' }), null)
})

test('litertLmTurnErrorMessage: explains a context-limit failure and leaves others alone', () => {
  assert.match(litertLmTurnErrorMessage('RuntimeError: Input token ids are too long. Exceeding the maximum number of tokens allowed: 15123 >= 4096'), /shorter system prompt/)
  assert.equal(litertLmTurnErrorMessage('RuntimeError: GPU lost'), 'RuntimeError: GPU lost')
})

test('engineErrorText: the model path, data dir and home dir never reach a client', async () => {
  const { engineErrorText } = await import('./chat-routes')
  const { homedir } = await import('node:os')
  const d = {
    manager: { currentOpts: () => ({ modelPath: `${homedir()}/m/model.litertlm` }) },
    store: { dir: () => `${homedir()}/.turbollm` },
  }
  const out = engineErrorText(d as never, 'llama-server', `failed ${homedir()}/m/model.litertlm and ${homedir()}/.turbollm/x and ${homedir()}/y`)
  assert.equal(out, 'failed <model> and <data dir>/x and ~/y')
})

test('engineErrorText: a path is replaced only as a whole segment, never as the prefix of a longer name', async () => {
  const { engineErrorText } = await import('./chat-routes')
  const d = { manager: { currentOpts: () => null }, store: { dir: () => '/home/al/.turbollm' } }
  assert.equal(
    engineErrorText(d as never, 'llama-server', 'open /home/al/.turbollm/x failed; see /home/al/.turbollmrc and /home/al/.turbollm.'),
    'open <data dir>/x failed; see /home/al/.turbollmrc and <data dir>.',
  )
})
