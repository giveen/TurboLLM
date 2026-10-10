import assert from 'node:assert/strict'
import { before, test } from 'node:test'
import { existsSync, statSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { openSqlJsDatabaseSync, preload } from './sql-js-database.js'
import { tmpDir } from '../../test-support/tmp'

before(() => preload())

const FLUSH_INTERVAL_MS = 10_000

test('close() persists writes and a reopen loads them', () => {
  const path = join(tmpDir('turbollm-sqljs-'), 'chat.db')
  const db = openSqlJsDatabaseSync(path)
  db.exec('CREATE TABLE t (v TEXT)')
  db.prepare('INSERT INTO t (v) VALUES (?)').run('hello')
  db.close()

  assert.equal(existsSync(`${path}.tmp`), false, 'temp file is renamed away, not left behind')
  const again = openSqlJsDatabaseSync(path)
  assert.deepEqual(again.prepare('SELECT v FROM t').all(), [{ v: 'hello' }])
  again.close()
})

test('a leftover temp file from a killed flush does not affect the loaded snapshot', () => {
  const path = join(tmpDir('turbollm-sqljs-'), 'chat.db')
  const db = openSqlJsDatabaseSync(path)
  db.exec('CREATE TABLE t (v TEXT)')
  db.prepare('INSERT INTO t (v) VALUES (?)').run('kept')
  db.close()

  writeFileSync(`${path}.tmp`, 'truncated garbage')
  const again = openSqlJsDatabaseSync(path)
  assert.deepEqual(again.prepare('SELECT v FROM t').all(), [{ v: 'kept' }])
  again.close()
})

test('the periodic flush writes only when something changed', (t) => {
  t.mock.timers.enable({ apis: ['setInterval'] })
  const path = join(tmpDir('turbollm-sqljs-'), 'chat.db')
  const db = openSqlJsDatabaseSync(path)

  t.mock.timers.tick(FLUSH_INTERVAL_MS)
  assert.equal(existsSync(path), false, 'nothing written yet, so nothing flushed')

  db.exec('CREATE TABLE t (v TEXT)')
  db.prepare('INSERT INTO t (v) VALUES (?)').run('a')
  t.mock.timers.tick(FLUSH_INTERVAL_MS)
  assert.equal(existsSync(path), true)
  // Every flush renames a fresh file into place, so an unchanged inode means no rewrite.
  const ino = statSync(path).ino

  db.prepare('SELECT v FROM t').all()
  t.mock.timers.tick(FLUSH_INTERVAL_MS)
  assert.equal(statSync(path).ino, ino, 'reads alone do not trigger a rewrite')

  db.prepare('INSERT INTO t (v) VALUES (?)').run('b')
  t.mock.timers.tick(FLUSH_INTERVAL_MS)
  assert.notEqual(statSync(path).ino, ino, 'a new write is flushed on the next interval')
  db.close()
})
