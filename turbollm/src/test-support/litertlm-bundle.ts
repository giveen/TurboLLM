// A minimal .litertlm for tests: the real 32-byte prefix, a hand-built section FlatBuffer, and the LlmMetadata and
// ExecutorMetadata protos (models/litertlm.ts reads only these). `kvLens` is [StateBuffer.Type, maximum_sequence_length].

function varint(n: number): number[] {
  const out: number[] = []
  do {
    let b = n & 0x7f
    n = Math.floor(n / 128)
    if (n > 0) b |= 0x80
    out.push(b)
  } while (n > 0)
  return out
}
const pVarint = (field: number, v: number) => [...varint(field * 8), ...varint(v)]
const pBytes = (field: number, b: number[]) => [...varint(field * 8 + 2), ...varint(b.length), ...b]

export function fakeLitertlmBundle(llmMax: number, kvLens: Array<[type: number, len: number]>): Buffer {
  const llm = Buffer.from(pVarint(5, llmMax))
  const states = kvLens.flatMap(([type, n]) =>
    pBytes(2, [...pBytes(1, [...Buffer.from('kv_cache_k_0')]), ...pVarint(6, type), ...pVarint(9, n)]))
  const exec = Buffer.from(pBytes(1, states))
  const hdr = Buffer.alloc(108)
  hdr.writeUInt32LE(12, 0) // root table
  hdr.writeUInt16LE(8, 4); hdr.writeUInt16LE(8, 6); hdr.writeUInt16LE(0, 8); hdr.writeUInt16LE(4, 10) // root vtable
  hdr.writeInt32LE(8, 12); hdr.writeUInt32LE(28 - 16, 16) // root → section_metadata
  hdr.writeUInt16LE(6, 20); hdr.writeUInt16LE(8, 22); hdr.writeUInt16LE(4, 24) // section_metadata vtable
  hdr.writeInt32LE(8, 28); hdr.writeUInt32LE(36 - 32, 32) // → objects vector
  hdr.writeUInt32LE(2, 36); hdr.writeUInt32LE(60 - 40, 40); hdr.writeUInt32LE(84 - 44, 44)
  hdr.writeUInt16LE(12, 48); hdr.writeUInt16LE(24, 50); hdr.writeUInt16LE(0, 52); hdr.writeUInt16LE(4, 54); hdr.writeUInt16LE(12, 56); hdr.writeUInt16LE(20, 58) // object vtable
  hdr.writeInt32LE(60 - 48, 60); hdr.writeBigUInt64LE(256n, 64); hdr.writeBigUInt64LE(BigInt(256 + llm.length), 72); hdr.writeUInt8(5, 80)
  hdr.writeInt32LE(84 - 48, 84); hdr.writeBigUInt64LE(512n, 88); hdr.writeBigUInt64LE(BigInt(512 + exec.length), 96); hdr.writeUInt8(9, 104)
  const file = Buffer.alloc(512 + exec.length)
  file.write('LITERTLM', 0, 'latin1')
  file.writeUInt32LE(1, 8); file.writeUInt32LE(6, 12)
  file.writeBigUInt64LE(BigInt(32 + hdr.length), 24)
  hdr.copy(file, 32)
  llm.copy(file, 256)
  exec.copy(file, 512)
  return file
}

