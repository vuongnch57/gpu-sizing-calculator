import { CUSTOM_MODEL_ID, DEFAULT_INPUTS, findPreset, type CalculatorInputs } from './inputs'
import { PRECISION_LABELS, type Precision } from './lib/calc'

/*
 * The inputs live in the URL query string so a result can be shared as a link, e.g.
 *   ?model=llama-3.1-70b&weights=fp8&kv=fp8&ctx=32768&users=8&overhead=15
 * Custom models add params (in billions), layers, kvHeads and headDim.
 */

export function inputsToSearch(inputs: CalculatorInputs): string {
  const p = new URLSearchParams({
    model: inputs.modelId,
    weights: inputs.weightPrecision,
    kv: inputs.kvPrecision,
    ctx: String(inputs.contextLength),
    users: String(inputs.batchSize),
    overhead: String(inputs.overheadPercent),
  })
  if (inputs.modelId === CUSTOM_MODEL_ID) {
    p.set('params', String(inputs.custom.paramCount / 1e9))
    p.set('layers', String(inputs.custom.numLayers))
    p.set('kvHeads', String(inputs.custom.numKvHeads))
    p.set('headDim', String(inputs.custom.headDim))
  }
  return `?${p}`
}

/** Reads inputs from a query string. Missing or invalid values fall back to the defaults. */
export function searchToInputs(search: string): CalculatorInputs {
  const p = new URLSearchParams(search)
  const d = DEFAULT_INPUTS
  const model = p.get('model')
  const modelId = model && (model === CUSTOM_MODEL_ID || findPreset(model)) ? model : d.modelId
  const params = positive(p.get('params'))

  return {
    modelId,
    custom: {
      paramCount: params === undefined ? d.custom.paramCount : params * 1e9,
      numLayers: positiveInt(p.get('layers')) ?? d.custom.numLayers,
      numKvHeads: positiveInt(p.get('kvHeads')) ?? d.custom.numKvHeads,
      headDim: positiveInt(p.get('headDim')) ?? d.custom.headDim,
    },
    weightPrecision: precision(p.get('weights')) ?? d.weightPrecision,
    kvPrecision: precision(p.get('kv')) ?? d.kvPrecision,
    contextLength: positiveInt(p.get('ctx')) ?? d.contextLength,
    batchSize: positiveInt(p.get('users')) ?? d.batchSize,
    overheadPercent: nonNegative(p.get('overhead')) ?? d.overheadPercent,
  }
}

function precision(value: string | null): Precision | undefined {
  return value !== null && value in PRECISION_LABELS ? (value as Precision) : undefined
}

function positive(value: string | null): number | undefined {
  const n = Number(value)
  return value !== null && value !== '' && Number.isFinite(n) && n > 0 ? n : undefined
}

function positiveInt(value: string | null): number | undefined {
  const n = positive(value)
  return n !== undefined && Number.isInteger(n) ? n : undefined
}

function nonNegative(value: string | null): number | undefined {
  const n = Number(value)
  return value !== null && value !== '' && Number.isFinite(n) && n >= 0 ? n : undefined
}
