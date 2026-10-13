// LiteRT-LM models in the library. A `.litertlm` file is one self-contained bundle (weights, tokenizer, and for
// multimodal models the vision/audio encoders) that Google's LiteRT-LM runtime loads directly, so the library holds it
// the way it holds a GGUF: one file, one entry, deleted as a file. Unlike a GGUF it carries no header TurboLLM can read
// cheaply (the metadata is a FlatBuffer), so the entry is built from the file name and the 8-byte magic.
import { open } from 'node:fs/promises'
import { basename } from 'node:path'
import type { ModelEntry } from './scanner'

/** The first 8 bytes of every LiteRT-LM file (litert_lm_builder's HEADER_MAGIC_BYTES). */
const MAGIC = 'LITERTLM'

/** Unlike a GGUF a .litertlm can be tiny (small test bundles), so only the magic below gates it, not a size floor. */
export function isLitertlmFileName(name: string): boolean {
  return name.toLowerCase().endsWith('.litertlm')
}

/** True when the file really is a LiteRT-LM bundle — keeps a renamed HTML error page or a truncated download out of
 *  the library, where it would otherwise fail at load time with a native-runtime message. */
export async function hasLitertlmMagic(path: string): Promise<boolean> {
  let fh: Awaited<ReturnType<typeof open>> | null = null
  try {
    fh = await open(path, 'r')
    const head = Buffer.alloc(MAGIC.length)
    const { bytesRead } = await fh.read(head, 0, MAGIC.length, 0)
    return bytesRead === MAGIC.length && head.toString('latin1') === MAGIC
  } catch {
    return false
  } finally {
    await fh?.close().catch(() => {})
  }
}

// litert-community names its files by precision (`-int4`, `_q8`, `-fp16`). Anchored on separators, and the last match
// wins, for the same reason quantFromName's does (a model name can contain a number-letter run of its own).
const QUANT_RE = /(?:^|[-_. ])(int[248]|fp16|f16|fp32|bf16|q[48])(?=$|[-_. ])/gi

/** The precision label from the file name, '?' when it states none. */
export function litertlmQuantFromName(fileName: string): string {
  const matches = [...fileName.matchAll(QUANT_RE)]
  return matches.length > 0 ? matches[matches.length - 1][1].toUpperCase() : '?'
}

/** `ekv4096` in a litert-community name is the KV-cache length the bundle was exported with, which is the longest
 *  context the runtime can give it — the nearest thing to a native context the file name declares. 0 = not declared. */
export function litertlmNativeCtxFromName(fileName: string): number {
  const m = /(?:^|[-_. ])ekv(\d{3,6})(?=$|[-_. ])/i.exec(fileName)
  return m ? Number(m[1]) : 0
}

// ── Context limit from the bundle header ──────────────────────────────────────────────────
// A .litertlm starts with "LITERTLM", three uint32 versions, 4 bytes of padding, a uint64 header-end offset, then a
// FlatBuffer (schema/core/litertlm_header_schema.fbs) listing the sections with their byte ranges. The small
// ExecutorMetadata protobuf section describes each KV-cache state buffer, with its `maximum_sequence_length`
// (field 9). A static export sizes its global (full-attention) KV caches to exactly that, and the runtime clamps
// max_num_tokens down to it whatever the engine was asked for (ClampMaxNumTokens in
// llm_litert_compiled_model_executor.cc), so it is a hard limit.
// Only global caches count: a sliding-window (local) cache's length is its window, not the context, and LlmMetadata's
// own max_num_tokens is just the default for a dynamic export, which can grow past it. Neither is used here.
// litert-community/LFM2.5-230M, for one: no `ekv` in its file name, 4096-entry KV caches, and a 15k-token prompt failed
// with "Input token ids are too long ... 15753 >= 4096" while the slider offered 256k.

const SECTION_EXECUTOR_METADATA = 9
const MAX_HEADER = 1 << 20
const MAX_PROTO = 1 << 20

/** One FlatBuffer table over a buffer: just the accessors the header needs. */
class FbTable {
  constructor(private readonly b: Buffer, private readonly pos: number) {}
  private field(i: number): number {
    const vt = this.pos - this.b.readInt32LE(this.pos)
    const o = 4 + 2 * i
    return o < this.b.readUInt16LE(vt) ? this.b.readUInt16LE(vt + o) : 0
  }
  table(i: number): FbTable | null {
    const o = this.field(i)
    if (!o) return null
    const p = this.pos + o
    return new FbTable(this.b, p + this.b.readUInt32LE(p))
  }
  tables(i: number): FbTable[] {
    const o = this.field(i)
    if (!o) return []
    let p = this.pos + o
    p += this.b.readUInt32LE(p)
    const n = this.b.readUInt32LE(p)
    const out: FbTable[] = []
    for (let k = 0; k < n; k++) {
      const q = p + 4 + 4 * k
      out.push(new FbTable(this.b, q + this.b.readUInt32LE(q)))
    }
    return out
  }
  u64(i: number): number {
    const o = this.field(i)
    return o ? Number(this.b.readBigUInt64LE(this.pos + o)) : 0
  }
  u8(i: number): number {
    const o = this.field(i)
    return o ? this.b.readUInt8(this.pos + o) : 0
  }
}

type ProtoField = { field: number; varint?: number; bytes?: Buffer }

/** The top-level fields of a protobuf message (varint and length-delimited; fixed-width ones are skipped). */
export function protoFields(buf: Buffer): ProtoField[] {
  const out: ProtoField[] = []
  let i = 0
  const varint = (): number => {
    let v = 0
    let mul = 1
    for (;;) {
      if (i >= buf.length) throw new RangeError('truncated varint')
      const c = buf[i++]
      v += (c & 0x7f) * mul
      if (c < 0x80) return v
      mul *= 128
    }
  }
  while (i < buf.length) {
    const key = varint()
    const field = Math.floor(key / 8)
    const wire = key % 8
    if (wire === 0) out.push({ field, varint: varint() })
    else if (wire === 2) {
      const len = varint()
      if (i + len > buf.length) throw new RangeError('truncated field')
      out.push({ field, bytes: buf.subarray(i, i + len) })
      i += len
    } else if (wire === 5) i += 4
    else if (wire === 1) i += 8
    else throw new RangeError(`unsupported wire type ${wire}`)
  }
  return out
}

/** The longest sequence the bundle's global KV caches hold (ExecutorMetadata), or 0 when they declare none.
 *  StateBuffer.Type 1/2 are global key/value caches: every one must hold the sequence, so the smallest declared
 *  length is the limit. 3/4 (local, sliding-window: the length is the window) and 5 (linear-attention state) say
 *  nothing about context. 0 (TYPE_UNSPECIFIED, older exports) could be either kind, so it is used only when no cache
 *  is typed, and then the LARGEST length wins: a sliding window is shorter than the context, never longer, so the
 *  max is the global length, and taking the min could cap Context at a 512-token window. */
export function executorMetadataMaxSequence(proto: Buffer): number {
  let typed = 0
  let untyped = 0
  for (const top of protoFields(proto)) {
    if (top.field !== 1 || !top.bytes) continue // llm_executor_metadata
    for (const f of protoFields(top.bytes)) {
      if (f.field !== 2 || !f.bytes) continue // state_buffers
      const fields = protoFields(f.bytes)
      const type = fields.find((x) => x.field === 6)?.varint ?? 0
      const len = fields.find((x) => x.field === 9)?.varint ?? 0 // maximum_sequence_length
      if (len <= 0) continue
      if (type === 1 || type === 2) typed = typed === 0 ? len : Math.min(typed, len)
      else if (type === 0) untyped = Math.max(untyped, len)
    }
  }
  return typed || untyped
}

/** The hard context limit the bundle declares (see the section comment above), or 0 when the header says nothing or
 *  cannot be read. Reads only the header and the small ExecutorMetadata section, never the weights. */
export async function litertlmNativeCtxFromFile(path: string): Promise<number> {
  let fh: Awaited<ReturnType<typeof open>> | null = null
  try {
    fh = await open(path, 'r')
    const read = async (pos: number, len: number): Promise<Buffer> => {
      const buf = Buffer.alloc(len)
      const { bytesRead } = await fh!.read(buf, 0, len, pos)
      return buf.subarray(0, bytesRead)
    }
    const head = await read(0, 32)
    if (head.length < 32 || head.toString('latin1', 0, 8) !== MAGIC) return 0
    const headerEnd = Number(head.readBigUInt64LE(24))
    if (headerEnd <= 32 || headerEnd > MAX_HEADER) return 0
    const header = await read(32, headerEnd - 32)
    const root = new FbTable(header, header.readUInt32LE(0))
    const sections = root.table(1)?.tables(0) ?? [] // section_metadata.objects
    for (const s of sections) {
      if (s.u8(3) !== SECTION_EXECUTOR_METADATA) continue
      const begin = s.u64(1)
      const end = s.u64(2)
      if (end <= begin || end - begin > MAX_PROTO) continue
      return executorMetadataMaxSequence(await read(begin, end - begin))
    }
    return 0
  } catch {
    return 0 // a header this reader does not understand costs only the slider's precision
  } finally {
    await fh?.close().catch(() => {})
  }
}

function cleanName(fileName: string): string {
  return fileName
    .replace(/\.litertlm$/i, '')
    .replace(/[-_]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
}

/** The library entry for a `.litertlm` file. Its format is 'litertlm' (a single file, deleted as a file); compat.ts
 *  gives it to the LiteRT-LM engine and to no other. Vision/audio stay false: the bundle declares them only in the
 *  FlatBuffer header, and claiming a capability the file may lack would put an image button on a text-only model. */
export function litertlmEntryFor(path: string, dir: string, sizeBytes: number, mtimeMs: number, headerCtx = 0): ModelEntry {
  const fileName = basename(path)
  const name = cleanName(fileName)
  const quant = litertlmQuantFromName(fileName)
  return {
    key: `${name.toLowerCase()}|${quant}|${sizeBytes}`,
    name,
    path,
    dir,
    format: 'litertlm',
    sizeBytes,
    sizeLabel: '',
    arch: 'litertlm',
    quant,
    // The header is the bundle's own word; the `ekvNNNN` name tag is the fallback for headers that declare nothing.
    nativeCtx: headerCtx || litertlmNativeCtxFromName(fileName),
    blockCount: 0,
    headCountKv: 0,
    headDim: 0,
    moe: false,
    expertCount: 0,
    nextnLayers: 0,
    vision: false,
    audio: false,
    mmprojPath: null,
    mmprojSizeBytes: 0,
    hasChatTemplate: true, // the bundle embeds its own template; the runtime applies it
    reasoningEffort: false,
    embedding: false,
    incomplete: sizeBytes === 0,
    parseError: null,
    loaded: false,
    hasProfile: false,
    benchTps: null,
    mtime: new Date(mtimeMs || Date.now()).toISOString(),
  }
}
