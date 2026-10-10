// WASM SQLite (sql.js) backend for Android — see sqlite-adapter.ts's header for the full
// design (why this exists, why openSqlDb has to stay synchronous). Real differences from
// node:sqlite's DatabaseSync, all consciously accepted for this target:
//   • In-memory only; flushed to `path` periodically and on close() via db.export(). A
//     non-graceful kill (routine on Android — the OS can kill an app any time) loses writes
//     since the last flush, not since the last close — periodic flush (below) bounds that
//     window instead of only flushing at a clean shutdown, which chat data on Android can't
//     assume ever happens. Each flush replaces the file atomically (temp + rename), so a
//     kill mid-flush costs that interval's writes, never the whole file.
//   • No WAL journal mode / busy_timeout — meaningless for an in-process, single-connection,
//     single-threaded WASM database; both PRAGMAs are silently accepted as no-ops so
//     ConversationStore's migrate() doesn't need an Android-specific branch of its own.
import { closeSync, existsSync, fsyncSync, openSync, readFileSync, renameSync, writeFileSync } from 'node:fs'
import initSqlJs, { type Database as SqlJsDb, type SqlJsStatic } from 'sql.js'
import type { SqlDb, SqlStatement, SqlValue } from './sqlite-adapter.js'

const FLUSH_INTERVAL_MS = 10_000

let sqlJsModule: SqlJsStatic | null = null
let sqlJsInitPromise: Promise<SqlJsStatic> | null = null

/** Loads the sql.js WASM module exactly once per process. Must be awaited (cli.ts's Android
 *  startup path does this) before the first openSqlJsDatabaseSync() call — that function
 *  throws a clear error rather than hang or silently misbehave if called too early. */
export async function preload(): Promise<void> {
  if (sqlJsModule) return
  if (!sqlJsInitPromise) sqlJsInitPromise = initSqlJs()
  sqlJsModule = await sqlJsInitPromise
}

/** sql.js's C SQLite binding has no bigint support (its own SqlValue type is
 *  `number | string | Uint8Array | null`) — node:sqlite's does, so `SqlValue` (this
 *  module's shared type, matching node:sqlite's SQLInputValue) allows it for desktop parity.
 *  Converts to a decimal string, a safe lossless representation, rather than silently
 *  truncating to `number` (which loses precision above 2^53). None of this codebase's actual
 *  bound values are bigints today — this is a correctness guard against a future one, not a
 *  currently-exercised path. */
function toSqlJsValue(v: SqlValue): number | string | Uint8Array | null {
  return typeof v === 'bigint' ? v.toString() : v
}

/** Normalizes the same two call shapes `SqlStatement` supports — a single named-params
 *  object, or a spread of positional values — into whatever sql.js's own `bind()` expects
 *  (an object for named `$foo` placeholders, an array for positional `?` ones). `undefined`
 *  means "nothing to bind" (a parameterless statement). A lone Buffer/Uint8Array positional
 *  arg must stay positional, not get mistaken for a named-params object — both are
 *  `typeof === 'object'`. */
function bindArgs(params: (SqlValue | Record<string, SqlValue>)[]): Record<string, number | string | Uint8Array | null> | (number | string | Uint8Array | null)[] | undefined {
  if (params.length === 0) return undefined
  const [first] = params
  if (params.length === 1 && typeof first === 'object' && first !== null && !(first instanceof Uint8Array)) {
    const obj = first as Record<string, SqlValue>
    return Object.fromEntries(Object.entries(obj).map(([k, v]) => [k, toSqlJsValue(v)]))
  }
  return (params as SqlValue[]).map(toSqlJsValue)
}

class SqlJsAdapter implements SqlDb {
  private readonly flushTimer: ReturnType<typeof setInterval>
  private closed = false
  // Set by anything that may have written; cleared only by a successful flush, so an idle
  // daemon doesn't rewrite the whole file every interval and a failed flush is retried.
  private dirty = false

  constructor(private readonly db: SqlJsDb, private readonly path: string) {
    this.flushTimer = setInterval(() => this.flush(), FLUSH_INTERVAL_MS)
    this.flushTimer.unref?.()
  }

  /** Writes the snapshot to a sibling temp file, fsyncs it, then renames it over `path`, so
   *  a kill mid-write leaves the previous snapshot intact instead of a truncated file (which
   *  would fail to load and lose every conversation, not just the last interval's writes). */
  private flush(): void {
    if (this.closed || !this.dirty) return
    const tmp = `${this.path}.tmp`
    try {
      const fd = openSync(tmp, 'w')
      try {
        writeFileSync(fd, this.db.export()) // loops on short writes; a bare writeSync doesn't
        fsyncSync(fd)
      } finally {
        closeSync(fd)
      }
      renameSync(tmp, this.path)
      this.dirty = false
    } catch {
      // Best-effort — the in-memory DB stays usable even if a flush fails (e.g. disk full);
      // `dirty` stays set so the next interval retries.
    }
  }

  exec(sql: string): void {
    const trimmed = sql.trim().toLowerCase()
    if (trimmed.startsWith('pragma journal_mode') || trimmed.startsWith('pragma busy_timeout')) return
    this.db.exec(sql)
    this.dirty = true
  }

  prepare(sql: string): SqlStatement {
    const db = this.db
    const markDirty = () => { this.dirty = true }
    // get()/all() are reads everywhere today, but a statement that isn't a plain SELECT
    // (e.g. a future INSERT … RETURNING) must still be flushed.
    const mayWrite = !/^\s*select\b/i.test(sql)
    return {
      get(...params: SqlValue[] | [Record<string, SqlValue>]) {
        if (mayWrite) markDirty()
        const stmt = db.prepare(sql)
        try {
          const bound = bindArgs(params)
          if (bound !== undefined) stmt.bind(bound)
          // step() returning false means zero rows — matches DatabaseSync's `undefined`,
          // unlike sql.js's own getAsObject(), which returns an all-undefined-fields object
          // for a query with no rows instead of signaling "no row" at all.
          return stmt.step() ? (stmt.getAsObject() as Record<string, unknown>) : undefined
        } finally {
          stmt.free()
        }
      },
      all(...params: SqlValue[] | [Record<string, SqlValue>]) {
        if (mayWrite) markDirty()
        const stmt = db.prepare(sql)
        const rows: Record<string, unknown>[] = []
        try {
          const bound = bindArgs(params)
          if (bound !== undefined) stmt.bind(bound)
          while (stmt.step()) rows.push(stmt.getAsObject() as Record<string, unknown>)
        } finally {
          stmt.free()
        }
        return rows
      },
      run(...params: SqlValue[] | [Record<string, SqlValue>]) {
        markDirty()
        const stmt = db.prepare(sql)
        try {
          const bound = bindArgs(params)
          if (bound !== undefined) stmt.run(bound as never)
          else stmt.run()
        } finally {
          stmt.free()
        }
        return { changes: db.getRowsModified() }
      },
    }
  }

  close(): void {
    if (this.closed) return
    clearInterval(this.flushTimer)
    this.flush() // before `closed` is set — flush() is a no-op once it is
    this.closed = true
    this.db.close()
  }
}

/** Opens (or creates) a sql.js-backed database at `path`. Synchronous — requires preload()
 *  to have already resolved (cli.ts's Android startup sequence guarantees this before any
 *  ConversationStore gets constructed). Loads any existing bytes at `path` as the starting
 *  state; the file is otherwise just a periodic/close-time snapshot, not continuously
 *  written the way node:sqlite's file is. */
export function openSqlJsDatabaseSync(path: string): SqlDb {
  if (!sqlJsModule) {
    throw new Error(
      'sql.js was not preloaded — call preloadSqlJs() (sqlite-adapter.ts) and await it ' +
        'before constructing any ConversationStore on Android.',
    )
  }
  const db = existsSync(path) ? new sqlJsModule.Database(new Uint8Array(readFileSync(path))) : new sqlJsModule.Database()
  return new SqlJsAdapter(db, path)
}
