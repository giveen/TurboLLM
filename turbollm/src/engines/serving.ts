// Which engine is serving the loaded model. Not the same as `registry.active()` since `.litertlm` models load on the
// LiteRT-LM engine whichever engine is active (engineForModel): with llama-server active, a LiteRT-LM model is served
// by LiteRT-LM, and every request has to be shaped for LiteRT-LM (the model path in `model`, max_completion_tokens).
import type { Engine } from '../config/config'
import type { Deps } from '../deps'

/** The engine the running (or starting) model was launched on, else the active one. */
export function servingEngine(d: Pick<Deps, 'registry' | 'manager'>): Engine | null {
  return d.manager.currentOpts()?.engine ?? d.registry.active() ?? null
}

/** The kind of {@link servingEngine}, '' when there is none. */
export function servingEngineKind(d: Pick<Deps, 'registry' | 'manager'>): string {
  return servingEngine(d)?.kind ?? ''
}

type EngineUseDeps = {
  registry: Pick<Deps['registry'], 'active'>
  manager: Pick<Deps['manager'], 'currentOpts'>
  /** Optional so narrow callers and tests need not supply a router; without it, pool slots are not considered. */
  modelRouter?: Pick<Deps['modelRouter'], 'poolEngineIds'>
}

/** True when stopping or replacing `engineId` would pull the engine out from under a loaded model: it is the active
 *  engine, the one the primary model runs on (a .litertlm model on LiteRT-LM while llama.cpp is active), or the
 *  engine of an extra pool slot. */
export function engineInUse(d: EngineUseDeps, engineId: string): boolean {
  return d.registry.active()?.id === engineId || d.manager.currentOpts()?.engine.id === engineId ||
    (d.modelRouter?.poolEngineIds().has(engineId) ?? false)
}

/** Stop everything running on `engineId` before its files are replaced or it is removed: the primary when it is that
 *  engine (or the active one), and any extra pool slot on it. */
export async function stopEngineUsers(
  d: {
    registry: Pick<Deps['registry'], 'active'>
    manager: Pick<Deps['manager'], 'currentOpts' | 'stopAndWait'>
    modelRouter?: Pick<Deps['modelRouter'], 'stopPoolSlotsOnEngine'>
  },
  engineId: string,
): Promise<void> {
  if (d.registry.active()?.id === engineId || d.manager.currentOpts()?.engine.id === engineId) await d.manager.stopAndWait()
  await d.modelRouter?.stopPoolSlotsOnEngine(engineId)
}
