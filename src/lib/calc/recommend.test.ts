import { describe, expect, it } from 'vitest'
import type { InferenceSettings } from './estimate'
import { recommendGpus, supportsTensorParallel } from './recommend'

const llama70b = {
  paramCount: 70e9,
  numLayers: 80,
  numKvHeads: 8,
  headDim: 128,
  numAttentionHeads: 64,
}
const settings: InferenceSettings = {
  weightPrecision: 'fp16',
  kvPrecision: 'fp16',
  contextLength: 8192,
  batchSize: 1,
  overheadPercent: 15,
}
const gpus = [
  { id: 'l4', memoryGB: 24 },
  { id: 'h100', memoryGB: 80 },
  { id: 'h200', memoryGB: 141 },
]

describe('supportsTensorParallel', () => {
  it('requires attention heads to divide evenly', () => {
    // Qwen 2.5 7B: 28 attention heads cannot split across 8 GPUs
    expect(supportsTensorParallel(4, 4, 28)).toBe(true)
    expect(supportsTensorParallel(8, 4, 28)).toBe(false)
  })

  it('allows more GPUs than KV heads when each head copies to whole GPUs', () => {
    expect(supportsTensorParallel(8, 4)).toBe(true)
    expect(supportsTensorParallel(4, 1)).toBe(true)
    expect(supportsTensorParallel(4, 6)).toBe(false)
  })
})

describe('recommendGpus', () => {
  it('finds the fewest GPUs of each type that fit', () => {
    const fits = recommendGpus(llama70b, settings, gpus, [1, 2, 4, 8])
    // ≈ 82 GB per GPU on 2 (see estimate.test.ts), so H100 80GB needs 4
    expect(fits.map((f) => [f.gpu.id, f.count])).toEqual([
      ['l4', 8],
      ['h100', 4],
      ['h200', 2],
    ])
  })

  it('reports headroom per GPU', () => {
    const [h200] = recommendGpus(llama70b, settings, [gpus[2]], [1, 2, 4, 8])
    expect(h200.headroomBytes).toBeCloseTo(141e9 - h200.perGpu.total)
    expect(h200.headroomBytes).toBeGreaterThan(0)
  })

  it('returns count null with negative headroom when nothing fits', () => {
    const [l4] = recommendGpus(llama70b, settings, [gpus[0]], [1, 2])
    expect(l4.count).toBeNull()
    expect(l4.headroomBytes).toBeLessThan(0)
  })

  it('skips GPU counts the model cannot be split across', () => {
    const qwenLike = { ...llama70b, numAttentionHeads: 28, numKvHeads: 4 }
    const [l4] = recommendGpus(qwenLike, settings, [gpus[0]], [1, 2, 4, 8])
    // 4 GPUs is not enough for a 70B FP16 model on 24 GB cards, and 8 is not allowed
    expect(l4.count).toBeNull()
  })
})
