import { describe, expect, it } from 'vitest'
import { MODEL_PRESETS } from './data/models'
import { CUSTOM_MODEL_ID, DEFAULT_INPUTS, resolveModel } from './inputs'

describe('resolveModel', () => {
  it('uses the preset architecture when a preset is selected', () => {
    const preset = MODEL_PRESETS[1]
    const model = resolveModel({ ...DEFAULT_INPUTS, modelId: preset.id })
    expect(model).toEqual({
      paramCount: preset.paramCount,
      numLayers: preset.numLayers,
      numKvHeads: preset.numKvHeads,
      headDim: preset.headDim,
    })
  })

  it('uses the custom architecture when Custom is selected', () => {
    const custom = { paramCount: 1e9, numLayers: 16, numKvHeads: 4, headDim: 64 }
    expect(resolveModel({ ...DEFAULT_INPUTS, modelId: CUSTOM_MODEL_ID, custom })).toEqual(custom)
  })
})
