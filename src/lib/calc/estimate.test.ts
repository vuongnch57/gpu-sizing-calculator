import { describe, expect, it } from 'vitest'
import { estimateMemory, estimatePerGpuMemory, type InferenceSettings } from './estimate'
import type { ModelArchitecture } from './formulas'

// Param counts are rounded; exact values will come from the model presets in a later step.
const llama8b: ModelArchitecture = { paramCount: 8e9, numLayers: 32, numKvHeads: 8, headDim: 128 }
const llama70b: ModelArchitecture = {
  paramCount: 70e9,
  numLayers: 80,
  numKvHeads: 8,
  headDim: 128,
}

const fp16At8k: InferenceSettings = {
  weightPrecision: 'fp16',
  kvPrecision: 'fp16',
  contextLength: 8192,
  batchSize: 1,
  overheadPercent: 15,
}

describe('estimateMemory (single GPU)', () => {
  it('adds weights, KV cache and a percentage overhead', () => {
    const r = estimateMemory(llama8b, fp16At8k)
    const kv = 2 ** 30 // see formulas.test.ts
    expect(r.weights).toBe(16e9)
    expect(r.kvCache).toBe(kv)
    expect(r.overhead).toBeCloseTo((16e9 + kv) * 0.15)
    expect(r.total).toBeCloseTo((16e9 + kv) * 1.15)
    // ≈ 19.63 GB: does not fit a 16 GB card, fits a 24 GB L4 with room to spare
    expect(r.total / 1e9).toBeCloseTo(19.63, 2)
  })

  it('applies a 1 GB overhead floor for small models', () => {
    const tiny: ModelArchitecture = { paramCount: 1e9, numLayers: 16, numKvHeads: 8, headDim: 64 }
    const r = estimateMemory(tiny, { ...fp16At8k, contextLength: 1024 })
    // 15% of ~2.07 GB would be ~0.31 GB, below the floor
    expect(r.overhead).toBe(1e9)
  })

  it('has no overhead percentage when set to 0, only the floor', () => {
    const r = estimateMemory(llama70b, { ...fp16At8k, overheadPercent: 0 })
    expect(r.overhead).toBe(1e9)
  })

  it('rejects a negative overhead', () => {
    expect(() => estimateMemory(llama8b, { ...fp16At8k, overheadPercent: -5 })).toThrow(RangeError)
  })
})

describe('estimatePerGpuMemory (tensor parallel)', () => {
  it('matches estimateMemory on one GPU', () => {
    expect(estimatePerGpuMemory(llama70b, fp16At8k, 1)).toEqual(estimateMemory(llama70b, fp16At8k))
  })

  it('splits weights and KV cache evenly when KV heads divide across GPUs', () => {
    const one = estimateMemory(llama70b, fp16At8k)
    const four = estimatePerGpuMemory(llama70b, fp16At8k, 4)
    expect(four.weights).toBe(one.weights / 4)
    expect(four.kvCache).toBe(one.kvCache / 4)
  })

  it('Llama 70B FP16 with 15% overhead needs more than 2 × 80 GB', () => {
    // Per GPU on 2: 70 GB weights + 1.34 GB KV, × 1.15 ≈ 82 GB, just over an 80 GB card
    expect(estimatePerGpuMemory(llama70b, fp16At8k, 2).total / 1e9).toBeCloseTo(82.04, 2)
    expect(estimatePerGpuMemory(llama70b, fp16At8k, 4).total).toBeLessThan(80e9)
  })

  it('keeps a full KV head per GPU when there are fewer KV heads than GPUs', () => {
    // MQA-style model: 1 KV head cannot be split, so every GPU caches it.
    const mqa = { ...llama8b, numKvHeads: 1 }
    const one = estimateMemory(mqa, fp16At8k)
    const four = estimatePerGpuMemory(mqa, fp16At8k, 4)
    expect(four.weights).toBe(one.weights / 4)
    expect(four.kvCache).toBe(one.kvCache)
  })

  it('does not let per-GPU overhead shrink below the floor', () => {
    const eight = estimatePerGpuMemory(llama8b, fp16At8k, 8)
    // 15% of each GPU's ~2.13 GB share is ~0.32 GB, so the 1 GB floor applies
    expect(eight.overhead).toBe(1e9)
    // So 8 GPUs need more memory in total than 1 GPU for the same model
    expect(eight.total * 8).toBeGreaterThan(estimateMemory(llama8b, fp16At8k).total)
  })

  it('accepts a custom overhead floor', () => {
    const r = estimatePerGpuMemory(llama8b, fp16At8k, 8, { minOverheadBytesPerGpu: 0.5e9 })
    expect(r.overhead).toBe(0.5e9)
  })

  it('rejects a non-integer GPU count', () => {
    expect(() => estimatePerGpuMemory(llama8b, fp16At8k, 1.5)).toThrow(RangeError)
  })
})
