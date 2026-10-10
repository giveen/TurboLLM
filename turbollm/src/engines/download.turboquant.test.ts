// TurboQuant release resolution: variant choice per machine, and release/asset matching
// against the fork's real release layout (one release per version tagged b<build>-<semver>
// carrying every platform, preceded by per-platform releases until mid-2026).
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { execFileSync } from 'node:child_process'
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import {
  provisionTurboquant,
  resolveTurboquantRelease,
  turboquantCandidates,
  type GithubRelease,
  type TurboquantHw,
} from './download'
import { tmpDir } from '../test-support/tmp'

// Default: an Ada-generation NVIDIA GPU (RTX 40xx, compute 8.9) on a CUDA 13 driver.
const hw = (o: Partial<TurboquantHw>): TurboquantHw => ({
  platform: 'linux', arch: 'x64', vendor: 'nvidia', hasGpu: true, driverCuda: '13.4', computeCaps: [8.9], ...o,
})
const names = (o: Partial<TurboquantHw>) => turboquantCandidates(hw(o)).map((v) => v.name)

// The asset list of b10298-2.0.0, the fork's latest release at the time of writing.
const CURRENT_ASSETS = [
  'linux-arm64-cuda-13.3', 'linux-x64-cpu', 'linux-x64-cuda-12.4', 'linux-x64-cuda-13.3',
  'linux-x64-rocm', 'linux-x64-vulkan', 'macos-arm64',
].flatMap((n) => [`llama-turboquant-${n}.tar.gz`, `llama-turboquant-${n}.zip`])
  .concat(['cpu', 'cuda-12.4', 'cuda-13.3', 'rocm', 'vulkan'].map((b) => `llama-turboquant-windows-x64-${b}.zip`))

const rel = (tag: string, assetNames: string[], extra: Partial<GithubRelease> = {}): GithubRelease => ({
  tag_name: tag,
  assets: assetNames.map((name) => ({ name, browser_download_url: `https://dl.example/${tag}/${name}` })),
  ...extra,
})

// Newest first, as the GitHub API lists them.
const RELEASES: GithubRelease[] = [
  rel('b10298-2.0.0', CURRENT_ASSETS),
  rel('dev-latest', CURRENT_ASSETS, { prerelease: true }),
  rel('b10269-1.7.0', CURRENT_ASSETS),
  rel('turboquant-windows-x64-cuda-13.3-066cc29', ['llama-turboquant-windows-x64-cuda-13.3.zip']),
  rel('turboquant-macos-arm64-066cc29', ['llama-turboquant-macos-arm64.tar.gz', 'llama-turboquant-macos-arm64.zip']),
  rel('turboquant-linux-x64-vulkan-519f0c5', ['llama-turboquant-linux-x64-vulkan.tar.gz', 'llama-turboquant-linux-x64-vulkan.zip']),
]

// ─── turboquantCandidates ────────────────────────────────────────────────────

test('turboquantCandidates: NVIDIA gets CUDA 13.3 then 12.4 when the driver supports CUDA 13', () => {
  assert.deepEqual(names({}), ['linux-x64-cuda-13.3', 'linux-x64-cuda-12.4', 'linux-x64-vulkan'])
  assert.deepEqual(names({ platform: 'win32' }), ['windows-x64-cuda-13.3', 'windows-x64-cuda-12.4', 'windows-x64-vulkan'])
})

test('turboquantCandidates: an older NVIDIA driver gets CUDA 12.4; too old or unreadable gets Vulkan', () => {
  assert.deepEqual(names({ driverCuda: '12.8' }), ['linux-x64-cuda-12.4', 'linux-x64-vulkan'])
  assert.deepEqual(names({ driverCuda: '12.2' }), ['linux-x64-vulkan'])
  assert.deepEqual(names({ driverCuda: null }), ['linux-x64-vulkan'])
})

test('turboquantCandidates: Blackwell never gets the 12.4 build (it has no sm_100/sm_120 kernels)', () => {
  assert.deepEqual(names({ computeCaps: [12.0] }), ['linux-x64-cuda-13.3', 'linux-x64-vulkan'])
  // A 570/575-series driver (CUDA 12.8/12.9) can't run the 13.3 build either → Vulkan.
  assert.deepEqual(names({ computeCaps: [12.0], driverCuda: '12.9' }), ['linux-x64-vulkan'])
  assert.deepEqual(names({ computeCaps: [10.0], platform: 'win32' })[0], 'windows-x64-cuda-13.3')
})

test('turboquantCandidates: pre-Turing GPUs never get the 13.3 build (CUDA 13 dropped them)', () => {
  assert.deepEqual(names({ computeCaps: [6.1] }), ['linux-x64-cuda-12.4', 'linux-x64-vulkan'])
  assert.deepEqual(names({ computeCaps: [7.0] }), ['linux-x64-cuda-12.4', 'linux-x64-vulkan'])
  assert.deepEqual(names({ computeCaps: [7.5] })[0], 'linux-x64-cuda-13.3')
})

test('turboquantCandidates: a mix of pre-Turing and Blackwell GPUs has no CUDA build that runs on both', () => {
  assert.deepEqual(names({ computeCaps: [6.1, 12.0] }), ['linux-x64-vulkan'])
})

test('turboquantCandidates: unknown compute capability gates on the driver alone', () => {
  assert.deepEqual(names({ computeCaps: [] }), ['linux-x64-cuda-13.3', 'linux-x64-cuda-12.4', 'linux-x64-vulkan'])
})

test('turboquantCandidates: only the Windows CUDA 13.3 build needs the separate runtime pack', () => {
  const win = turboquantCandidates(hw({ platform: 'win32' }))
  assert.equal(win[0].cudartPack, 'cudart-llama-bin-win-cuda-13.3-x64.zip')
  assert.equal(win[1].cudartPack, undefined, 'the 12.4 zip bundles its runtime')
  assert.ok(turboquantCandidates(hw({})).every((v) => !v.cudartPack), 'Linux CUDA archives bundle theirs')
})

test('turboquantCandidates: AMD/Intel get Vulkan, no GPU gets CPU, Apple Silicon gets the macOS build', () => {
  assert.deepEqual(names({ vendor: 'amd', driverCuda: null }), ['linux-x64-vulkan'])
  assert.deepEqual(names({ vendor: 'intel', platform: 'win32', driverCuda: null }), ['windows-x64-vulkan'])
  assert.deepEqual(names({ vendor: 'unknown', hasGpu: false, driverCuda: null }), ['linux-x64-cpu'])
  assert.deepEqual(names({ platform: 'darwin', arch: 'arm64', vendor: 'apple' }), ['macos-arm64'])
})

test('turboquantCandidates: no prebuilt for Intel Macs, non-NVIDIA arm64 Linux, or other platforms', () => {
  assert.deepEqual(names({ platform: 'darwin', arch: 'x64', vendor: 'amd' }), [])
  assert.deepEqual(names({ arch: 'arm64', vendor: 'arm', driverCuda: null }), [])
  assert.deepEqual(names({ arch: 'arm64' }), ['linux-arm64-cuda-13.3'])
  assert.deepEqual(names({ platform: 'freebsd' }), [])
})

// ─── resolveTurboquantRelease ────────────────────────────────────────────────

const resolve = (o: Partial<TurboquantHw>, releases = RELEASES) =>
  resolveTurboquantRelease(releases, turboquantCandidates(hw(o)), hw(o).platform)

test('resolveTurboquantRelease: picks the newest tagged release, not the old per-platform ones', () => {
  const r = resolve({})
  assert.equal(r?.tag, 'b10298-2.0.0')
  assert.equal(r?.assetName, 'llama-turboquant-linux-x64-cuda-13.3.tar.gz')
  assert.equal(resolve({ platform: 'darwin', arch: 'arm64', vendor: 'apple' })?.tag, 'b10298-2.0.0')
  assert.equal(resolve({ vendor: 'amd', driverCuda: null })?.assetName, 'llama-turboquant-linux-x64-vulkan.tar.gz')
})

test('resolveTurboquantRelease: Windows resolves a zip (the fork ships no Windows tarballs)', () => {
  const r = resolve({ platform: 'win32' })
  assert.equal(r?.assetName, 'llama-turboquant-windows-x64-cuda-13.3.zip')
  assert.equal(r?.variant.cudartPack, 'cudart-llama-bin-win-cuda-13.3-x64.zip')
})

test('resolveTurboquantRelease: skips the dev-latest prerelease and drafts', () => {
  const only = [rel('dev-latest', CURRENT_ASSETS, { prerelease: true }), rel('b1-0.0.1', CURRENT_ASSETS, { draft: true })]
  assert.equal(resolve({}, only), null)
})

test('resolveTurboquantRelease: falls back to the old per-platform releases when no tagged one fits', () => {
  const old = RELEASES.slice(3)
  assert.equal(resolve({ vendor: 'amd', driverCuda: null }, old)?.tag, 'turboquant-linux-x64-vulkan-519f0c5')
  // An NVIDIA box with no CUDA asset in reach falls through to Vulkan rather than nothing.
  assert.equal(resolve({}, old)?.assetName, 'llama-turboquant-linux-x64-vulkan.tar.gz')
})

test('resolveTurboquantRelease: within a release, the best candidate wins over a later one', () => {
  const no13 = [rel('b2-1.0.0', CURRENT_ASSETS.filter((n) => !n.includes('cuda-13.3')))]
  assert.equal(resolve({}, no13)?.assetName, 'llama-turboquant-linux-x64-cuda-12.4.tar.gz')
})

// ─── provisionTurboquant ─────────────────────────────────────────────────────

const unix = { skip: process.platform === 'win32' && 'tar-based extraction path' }

/** A tar.gz shaped like the fork's Linux archives: build/bin/<files>. */
function makeArchive(dir: string, files: string[]): Buffer {
  const src = join(dir, 'src')
  mkdirSync(join(src, 'build', 'bin'), { recursive: true })
  for (const f of files) writeFileSync(join(src, 'build', 'bin', f), f)
  const out = join(dir, 'a.tar.gz')
  execFileSync('tar', ['-czf', out, '-C', src, 'build'])
  return readFileSync(out)
}

function fakeGithub(t: { after: (fn: () => void) => void }, releases: GithubRelease[], archive: Buffer): string[] {
  const realFetch = globalThis.fetch
  t.after(() => { globalThis.fetch = realFetch })
  const calls: string[] = []
  globalThis.fetch = (async (url: string | URL) => {
    const u = String(url)
    calls.push(u)
    if (u.includes('/releases?')) return new Response(JSON.stringify(releases), { status: 200 })
    return new Response(new Uint8Array(archive), { status: 200 })
  }) as typeof fetch
  return calls
}

test('provisionTurboquant: installs the CUDA build for an NVIDIA box and records the tag', unix, async (t) => {
  const root = tmpDir('tllm-tq-')
  const archive = makeArchive(tmpDir('tllm-tq-src-'), ['llama-server', 'libggml-cuda.so', 'libcudart.so.13'])
  const calls = fakeGithub(t, RELEASES, archive)

  const bin = await provisionTurboquant(root, 'AtomicBot-ai/atomic-llama-cpp-turboquant', undefined, undefined, hw({}))
  assert.equal(bin, join(root, 'turboquant', 'build', 'bin', 'llama-server'))
  assert.ok(calls.includes('https://dl.example/b10298-2.0.0/llama-turboquant-linux-x64-cuda-13.3.tar.gz'))
  const marker = JSON.parse(readFileSync(join(root, 'turboquant', '.turbollm-provisioned'), 'utf8'))
  assert.equal(marker.tag, 'b10298-2.0.0')
  assert.equal(existsSync(join(root, 'llama-turboquant-linux-x64-cuda-13.3.tar.gz')), false, 'temp archive removed')

  // Second call: fast path, no network.
  calls.length = 0
  assert.equal(await provisionTurboquant(root, 'AtomicBot-ai/atomic-llama-cpp-turboquant', undefined, undefined, hw({})), bin)
  assert.deepEqual(calls, [])
})

test('provisionTurboquant: a marker-less CUDA install missing its runtime is reinstalled, not reused', unix, async (t) => {
  const root = tmpDir('tllm-tq-')
  const binDir = join(root, 'turboquant', 'build', 'bin')
  mkdirSync(binDir, { recursive: true })
  writeFileSync(join(binDir, 'llama-server'), '')
  writeFileSync(join(binDir, 'libggml-cuda.so'), '') // no libcudart → would silently run on CPU
  const archive = makeArchive(tmpDir('tllm-tq-src-'), ['llama-server', 'libggml-cuda.so', 'libcudart.so.13'])
  const calls = fakeGithub(t, RELEASES, archive)

  await provisionTurboquant(root, 'AtomicBot-ai/atomic-llama-cpp-turboquant', undefined, undefined, hw({}))
  assert.ok(calls.some((u) => u.includes('/releases?')), 'resolved and downloaded again')
  assert.ok(existsSync(join(binDir, 'libcudart.so.13')))
})

test('provisionTurboquant: a marker-less legacy Vulkan install is kept (and backfilled)', unix, async (t) => {
  const root = tmpDir('tllm-tq-')
  const binDir = join(root, 'turboquant', 'build', 'bin')
  mkdirSync(binDir, { recursive: true })
  writeFileSync(join(binDir, 'llama-server'), '')
  const calls = fakeGithub(t, RELEASES, Buffer.alloc(0))

  assert.equal(await provisionTurboquant(root, 'AtomicBot-ai/atomic-llama-cpp-turboquant', undefined, undefined, hw({})), join(binDir, 'llama-server'))
  assert.deepEqual(calls, [])
  assert.ok(existsSync(join(root, 'turboquant', '.turbollm-provisioned')))
})

test('provisionTurboquant: no matching build → no_release_asset', async (t) => {
  fakeGithub(t, RELEASES, Buffer.alloc(0))
  await assert.rejects(
    provisionTurboquant(tmpDir('tllm-tq-'), 'AtomicBot-ai/atomic-llama-cpp-turboquant', undefined, undefined,
      hw({ platform: 'darwin', arch: 'x64', vendor: 'amd' })),
    /no_release_asset/,
  )
})
