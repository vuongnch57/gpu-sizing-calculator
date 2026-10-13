import { estimatePerGpuMemory, type InferenceSettings, type MemoryBreakdown } from './estimate'
import type { ModelArchitecture } from './formulas'

/**
 * Whether a model can be split across `numGpus` with tensor parallelism.
 *
 * TP gives each GPU a slice of the attention heads, so the head count must
 * divide evenly. For KV heads, inference engines like vLLM also allow more GPUs
 * than KV heads as long as each KV head is copied to a whole number of GPUs
 * (numGpus is a multiple of numKvHeads).
 */
export function supportsTensorParallel(
  numGpus: number,
  numKvHeads: number,
  numAttentionHeads?: number,
): boolean {
  if (numAttentionHeads !== undefined && numAttentionHeads % numGpus !== 0) return false
  return numKvHeads % numGpus === 0 || numGpus % numKvHeads === 0
}

export interface GpuOption {
  id: string
  memoryGB: number
}

export interface GpuFit<G extends GpuOption> {
  gpu: G
  /** Smallest GPU count that fits, or null if none of the counts tried fit. */
  count: number | null
  /** Per-GPU memory at that count (or at the largest count tried, if none fit). */
  perGpu: MemoryBreakdown
  /** Unused memory per GPU in bytes; negative when it does not fit. */
  headroomBytes: number
}

/**
 * For each GPU, find the fewest GPUs (from `counts`) that hold the model.
 *
 * Uses the per-GPU estimate, so the per-GPU overhead floor and KV head
 * replication are included. Counts the model cannot be split across are skipped.
 */
export function recommendGpus<G extends GpuOption>(
  model: ModelArchitecture & { numAttentionHeads?: number },
  settings: InferenceSettings,
  gpus: readonly G[],
  counts: readonly number[],
): GpuFit<G>[] {
  const validCounts = counts.filter((n) =>
    supportsTensorParallel(n, model.numKvHeads, model.numAttentionHeads),
  )

  return gpus.map((gpu) => {
    const capacity = gpu.memoryGB * 1e9
    let last: MemoryBreakdown | undefined
    for (const n of validCounts) {
      const perGpu = estimatePerGpuMemory(model, settings, n)
      last = perGpu
      if (perGpu.total <= capacity) {
        return { gpu, count: n, perGpu, headroomBytes: capacity - perGpu.total }
      }
    }
    const perGpu = last ?? estimatePerGpuMemory(model, settings, 1)
    return { gpu, count: null, perGpu, headroomBytes: capacity - perGpu.total }
  })
}
