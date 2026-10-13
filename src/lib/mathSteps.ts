import {
  BYTES_PER_ELEMENT,
  DEFAULT_MIN_OVERHEAD_BYTES_PER_GPU,
  PRECISION_LABELS,
  estimateMemory,
  kvCacheBytesPerToken,
  type InferenceSettings,
  type ModelArchitecture,
} from './calc'
import { formatGB } from './units'

export interface MathStep {
  label: string
  /** The formula with symbolic names. */
  formula: string
  /** The same formula with the user's numbers plugged in. */
  substituted: string
  result: string
  why: string
}

const n = (value: number) => value.toLocaleString('en-US')
const bytes = (value: number) => `${n(value)} B`

/** The calculation behind the estimate, one step per formula, for the "Show the math" panel. */
export function buildMathSteps(model: ModelArchitecture, settings: InferenceSettings): MathStep[] {
  const breakdown = estimateMemory(model, settings)
  const wBytes = BYTES_PER_ELEMENT[settings.weightPrecision]
  const kvBytes = BYTES_PER_ELEMENT[settings.kvPrecision]
  const perToken = kvCacheBytesPerToken(model, settings.kvPrecision)
  const base = breakdown.weights + breakdown.kvCache
  const floorApplies =
    breakdown.overhead === DEFAULT_MIN_OVERHEAD_BYTES_PER_GPU &&
    base * (settings.overheadPercent / 100) < breakdown.overhead

  return [
    {
      label: 'Weights',
      formula: 'parameters × bytes per parameter',
      substituted: `${n(model.paramCount)} × ${wBytes} B (${PRECISION_LABELS[settings.weightPrecision]})`,
      result: formatGB(breakdown.weights),
      why: 'Every parameter must sit in GPU memory for every token generated, so this cost is fixed.',
    },
    {
      label: 'KV cache per token',
      formula: '2 × layers × KV heads × head dim × bytes per element',
      substituted: `2 × ${model.numLayers} × ${model.numKvHeads} × ${model.headDim} × ${kvBytes} B (${PRECISION_LABELS[settings.kvPrecision]})`,
      result: bytes(perToken),
      why: 'Each layer stores one key and one value vector (the 2) per KV head for every token, so later tokens can attend to it without recomputing. Models with grouped-query attention share KV heads, which is why KV heads are used here, not attention heads.',
    },
    {
      label: 'KV cache',
      formula: 'per token × context length × concurrent users',
      substituted: `${bytes(perToken)} × ${n(settings.contextLength)} × ${n(settings.batchSize)}`,
      result: formatGB(breakdown.kvCache),
      why: 'Every token of every active sequence keeps its own cache entries, so this grows linearly with context and users.',
    },
    {
      label: 'Overhead',
      formula: 'max((weights + KV cache) × overhead %, 1 GB)',
      substituted: `max(${formatGB(base)} × ${settings.overheadPercent}%, 1 GB)`,
      result: formatGB(breakdown.overhead),
      why: floorApplies
        ? 'The 1 GB floor applies here: every GPU needs its CUDA context and buffers, however small the model.'
        : 'Covers what is not modelled one by one: CUDA context, temporary activations and memory fragmentation.',
    },
    {
      label: 'Total',
      formula: 'weights + KV cache + overhead',
      substituted: `${formatGB(breakdown.weights)} + ${formatGB(breakdown.kvCache)} + ${formatGB(breakdown.overhead)}`,
      result: formatGB(breakdown.total),
      why: 'An estimate for one GPU. Splitting across GPUs divides weights and KV cache, but each GPU keeps its own overhead.',
    },
  ]
}
