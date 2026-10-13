import {
  assertPositiveInteger,
  kvCacheBytes,
  weightsBytes,
  type ModelArchitecture,
} from './formulas'
import type { Precision } from './precision'

/** Everything the user picks besides the model itself. */
export interface InferenceSettings {
  weightPrecision: Precision
  kvPrecision: Precision
  /** Tokens per sequence: prompt plus generated output. */
  contextLength: number
  /** Concurrent sequences (users) served at once. */
  batchSize: number
  /** Extra margin on top of weights + KV cache, as a percentage (15 means 15%). */
  overheadPercent: number
}

/** A memory estimate, in bytes. */
export interface MemoryBreakdown {
  weights: number
  kvCache: number
  overhead: number
  total: number
}

/**
 * Smallest overhead a single GPU carries, whatever its share of the model.
 *
 * Each GPU runs its own CUDA context, kernels and framework buffers, which cost
 * roughly 0.5–1 GB no matter how small the model slice is. 1 GB is a round,
 * slightly cautious figure.
 */
export const DEFAULT_MIN_OVERHEAD_BYTES_PER_GPU = 1e9

/**
 * Total memory to serve a model on one GPU.
 *
 *   overhead = max((weights + kv_cache) × overhead%, 1 GB)
 *   total    = weights + kv_cache + overhead
 *
 * For most models this is simply (weights + kv_cache) × (1 + overhead%).
 *
 * The overhead margin stands in for things we don't model one by one: the CUDA
 * context, temporary activations during a forward pass, and memory fragmentation.
 * 10–20% is a common rule of thumb for inference. The 1 GB floor only matters for
 * small models, where a percentage would undercount the fixed CUDA context cost.
 */
export function estimateMemory(
  model: ModelArchitecture,
  settings: InferenceSettings,
): MemoryBreakdown {
  return estimatePerGpuMemory(model, settings, 1)
}

export interface TensorParallelOptions {
  /** Floor for each GPU's overhead. Defaults to {@link DEFAULT_MIN_OVERHEAD_BYTES_PER_GPU}. */
  minOverheadBytesPerGpu?: number
}

/**
 * Memory each GPU needs when the model is split across `numGpus` with tensor
 * parallelism (TP).
 *
 * Weights: TP slices every weight matrix across the GPUs, so each holds
 * weights / N.
 *
 * KV cache: TP gives each GPU a subset of the KV heads, so each holds the cache
 * for num_kv_heads / N heads. If there are fewer KV heads than GPUs (for example
 * an MQA model with 1 KV head on 4 GPUs), heads cannot be split further and
 * each GPU keeps a full copy of at least one head. So per-GPU KV heads are
 * max(1, ceil(num_kv_heads / N)), not num_kv_heads / N.
 *
 * Overhead: the % margin applies to each GPU's own share, but never drops below
 * a fixed per-GPU floor, because every GPU pays for its own CUDA context and
 * buffers. That is why 8 GPUs need more total memory than 1 GPU for the same model.
 */
export function estimatePerGpuMemory(
  model: ModelArchitecture,
  settings: InferenceSettings,
  numGpus: number,
  options: TensorParallelOptions = {},
): MemoryBreakdown {
  assertPositiveInteger('numGpus', numGpus)
  if (!Number.isFinite(settings.overheadPercent) || settings.overheadPercent < 0) {
    throw new RangeError(`overheadPercent must be zero or more, got ${settings.overheadPercent}`)
  }
  const minOverhead = options.minOverheadBytesPerGpu ?? DEFAULT_MIN_OVERHEAD_BYTES_PER_GPU

  const weights = weightsBytes(model.paramCount, settings.weightPrecision) / numGpus
  const kvHeadsPerGpu = Math.max(1, Math.ceil(model.numKvHeads / numGpus))
  const kvCache = kvCacheBytes(
    { ...model, numKvHeads: kvHeadsPerGpu },
    settings.kvPrecision,
    settings.contextLength,
    settings.batchSize,
  )
  const overhead = Math.max((weights + kvCache) * (settings.overheadPercent / 100), minOverhead)

  return { weights, kvCache, overhead, total: weights + kvCache + overhead }
}
