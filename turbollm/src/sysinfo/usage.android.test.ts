// What the usage monitor reads inside the Android app, where /proc/stat and most of sysfs are closed to apps.
import assert from 'node:assert/strict'
import { mkdirSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { test } from 'node:test'
import { tmpDir } from '../test-support/tmp'
import { countCpuList } from './sysinfo'
import { ownProcessTicks, parseGpuLoad, procStatTicks } from './usage'

test('procStatTicks: utime + stime, counted from the last parenthesis of the command name', () => {
  const line = '1234 (llama (server) x) S 1 1234 1234 0 -1 4194560 100 0 0 0 250 75 0 0 20 0 9 0 100 0 0'
  assert.equal(procStatTicks(line), 325)
})

test('ownProcessTicks: sums every numeric /proc entry and skips the rest', () => {
  const root = tmpDir('tllm-proc-')
  for (const [pid, u, s] of [['10', 5, 1], ['11', 7, 2]] as const) {
    mkdirSync(join(root, pid))
    writeFileSync(join(root, pid, 'stat'), `${pid} (node) S 1 1 1 0 -1 0 0 0 0 0 ${u} ${s} 0 0`)
  }
  mkdirSync(join(root, 'self'))
  assert.equal(ownProcessTicks(root), 15)
  assert.equal(ownProcessTicks(join(root, 'missing')), null)
})

test('parseGpuLoad: reads the formats Android kernels publish', () => {
  assert.equal(parseGpuLoad('37 %\n'), 37)
  assert.equal(parseGpuLoad('37@850000000Hz'), 37)
  assert.equal(parseGpuLoad('0'), 0)
  assert.equal(parseGpuLoad('busy'), null)
  assert.equal(parseGpuLoad('400'), null)
})

test('countCpuList: counts the kernel CPU list format', () => {
  assert.equal(countCpuList('0-7\n'), 8)
  assert.equal(countCpuList('0-3,6,7'), 6)
  assert.equal(countCpuList('0'), 1)
})
