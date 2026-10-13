// Local paths out of text that leaves this machine's UI: engine error messages reach chat clients, API clients and
// Turbo Link peers, and engines routinely quote the model's absolute path, the data dir or the user's home.
import { homedir } from 'node:os'
import type { Deps } from '../deps'

/** `text` with the loaded model's path, the data dir and the home dir replaced by `<model>`, `<data dir>` and `~`.
 *  Longest first, so a model path inside the home dir is replaced whole. Only where the path is not followed by more
 *  of a name: a home of /home/al must not turn /home/alice/x into ~ice/x. */
export function redactLocalPaths(d: Pick<Deps, 'manager' | 'store'>, text: string): string {
  const paths: Array<[string | undefined, string]> = [
    [d.manager.currentOpts()?.modelPath, '<model>'],
    [d.store.dir(), '<data dir>'],
    [homedir(), '~'],
  ]
  let out = text
  for (const [p, label] of paths.filter(([p]) => !!p && p.length > 1).sort((a, b) => b[0]!.length - a[0]!.length)) {
    const escaped = p!.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
    out = out.replace(new RegExp(`${escaped}(?![\\w-]|\\.\\w)`, 'g'), label)
  }
  return out
}
