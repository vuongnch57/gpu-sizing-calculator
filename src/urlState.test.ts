import { describe, expect, it } from 'vitest'
import { CUSTOM_MODEL_ID, DEFAULT_INPUTS, type CalculatorInputs } from './inputs'
import { inputsToSearch, searchToInputs } from './urlState'

describe('URL state', () => {
  it('round-trips preset inputs', () => {
    const inputs: CalculatorInputs = {
      ...DEFAULT_INPUTS,
      modelId: 'llama-3.1-70b',
      weightPrecision: 'fp8',
      kvPrecision: 'fp8',
      contextLength: 32_768,
      batchSize: 8,
      overheadPercent: 20,
    }
    const search = inputsToSearch(inputs)
    expect(search).toBe('?model=llama-3.1-70b&weights=fp8&kv=fp8&ctx=32768&users=8&overhead=20')
    expect(searchToInputs(search)).toEqual(inputs)
  })

  it('round-trips a custom model', () => {
    const inputs: CalculatorInputs = {
      ...DEFAULT_INPUTS,
      modelId: CUSTOM_MODEL_ID,
      custom: { paramCount: 1.5e9, numLayers: 28, numKvHeads: 2, headDim: 128 },
    }
    expect(searchToInputs(inputsToSearch(inputs))).toEqual(inputs)
  })

  it('falls back to defaults for an empty query', () => {
    expect(searchToInputs('')).toEqual(DEFAULT_INPUTS)
  })

  it('ignores invalid values field by field', () => {
    const inputs = searchToInputs('?model=nope&weights=fp7&ctx=-1&users=2.5&overhead=abc&kv=fp8')
    expect(inputs).toEqual({ ...DEFAULT_INPUTS, kvPrecision: 'fp8' })
  })

  it('allows 0% overhead', () => {
    expect(searchToInputs('?overhead=0').overheadPercent).toBe(0)
  })
})
