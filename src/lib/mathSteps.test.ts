import { describe, expect, it } from 'vitest'
import type { InferenceSettings } from './calc'
import { buildMathSteps } from './mathSteps'

const llama8b = { paramCount: 8_030_261_248, numLayers: 32, numKvHeads: 8, headDim: 128 }
const settings: InferenceSettings = {
  weightPrecision: 'bf16',
  kvPrecision: 'bf16',
  contextLength: 8192,
  batchSize: 1,
  overheadPercent: 15,
}

describe('buildMathSteps', () => {
  it('plugs the inputs into each formula', () => {
    const steps = buildMathSteps(llama8b, settings)
    expect(steps.map((s) => s.label)).toEqual([
      'Weights',
      'KV cache per token',
      'KV cache',
      'Overhead',
      'Total',
    ])
    expect(steps[0].substituted).toBe('8,030,261,248 × 2 B (BF16)')
    expect(steps[0].result).toBe('16.06 GB')
    expect(steps[1].substituted).toBe('2 × 32 × 8 × 128 × 2 B (BF16)')
    expect(steps[1].result).toBe('131,072 B')
    expect(steps[2].substituted).toBe('131,072 B × 8,192 × 1')
    expect(steps[2].result).toBe('1.07 GB')
    expect(steps[4].result).toBe('19.70 GB')
  })

  it('explains when the 1 GB overhead floor applies', () => {
    const tiny = { paramCount: 0.5e9, numLayers: 24, numKvHeads: 2, headDim: 64 }
    const overhead = buildMathSteps(tiny, settings)[3]
    expect(overhead.result).toBe('1.00 GB')
    expect(overhead.why).toMatch(/floor applies/)
  })
})
