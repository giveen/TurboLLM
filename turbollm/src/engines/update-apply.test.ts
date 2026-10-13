// Dispatch of applyEngineUpdate for engines without an automatic update source. The release
// branch is covered in update-apply.release.test.ts.
import { test } from 'node:test'
import assert from 'node:assert/strict'
import type { Engine } from '../config/config'
import { applyEngineUpdate, type UpdateApplyDeps } from './update-apply'

test('a custom llama-server with no release provenance has no automatic update source', async () => {
  const deps = { store: { dir: () => 'unused' } } as unknown as UpdateApplyDeps
  const custom = { id: 'c1', name: 'Custom', kind: 'llama-server', binPath: '/opt/custom/llama-server' } as Engine
  await assert.rejects(applyEngineUpdate(deps, custom), /no automatic update source/)
})
