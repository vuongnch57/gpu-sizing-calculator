import { describe, expect, it } from 'vitest'
import { kvCacheBytes, kvCacheBytesPerToken, weightsBytes } from './formulas'

// Llama 3.1 8B, from https://huggingface.co/meta-llama/Llama-3.1-8B/blob/main/config.json
// 32 layers, 32 attention heads, 8 KV heads (GQA), head_dim = 4096 / 32 = 128.
const llama8b = { numLayers: 32, numKvHeads: 8, headDim: 128 }

describe('weightsBytes', () => {
  it('multiplies parameter count by bytes per parameter', () => {
    expect(weightsBytes(8e9, 'fp32')).toBe(32e9)
    expect(weightsBytes(8e9, 'bf16')).toBe(16e9)
    expect(weightsBytes(8e9, 'fp8')).toBe(8e9)
    expect(weightsBytes(8e9, 'int4')).toBe(4e9)
  })

  it('rejects non-positive parameter counts', () => {
    expect(() => weightsBytes(0, 'fp16')).toThrow(RangeError)
    expect(() => weightsBytes(Number.NaN, 'fp16')).toThrow(RangeError)
  })
})

describe('kvCacheBytesPerToken', () => {
  it('is 2 × layers × KV heads × head dim × bytes', () => {
    // 2 × 32 × 8 × 128 × 2 bytes = 131,072 bytes = 128 KiB per token
    expect(kvCacheBytesPerToken(llama8b, 'fp16')).toBe(131_072)
  })

  it('uses KV heads, so GQA is 4× smaller than the same model with MHA', () => {
    const mha = { ...llama8b, numKvHeads: 32 }
    expect(kvCacheBytesPerToken(mha, 'fp16')).toBe(4 * kvCacheBytesPerToken(llama8b, 'fp16'))
  })

  it('halves with an FP8 KV cache', () => {
    expect(kvCacheBytesPerToken(llama8b, 'fp8')).toBe(65_536)
  })
})

describe('kvCacheBytes', () => {
  it('Llama 3.1 8B at 8K context, 1 user, FP16 is exactly 1 GiB', () => {
    expect(kvCacheBytes(llama8b, 'fp16', 8192, 1)).toBe(2 ** 30)
  })

  it('grows linearly with context length and batch size', () => {
    const base = kvCacheBytes(llama8b, 'fp16', 8192, 1)
    expect(kvCacheBytes(llama8b, 'fp16', 16_384, 1)).toBe(2 * base)
    expect(kvCacheBytes(llama8b, 'fp16', 8192, 10)).toBe(10 * base)
  })

  it('rejects non-integer or non-positive sizes', () => {
    expect(() => kvCacheBytes(llama8b, 'fp16', 0, 1)).toThrow(RangeError)
    expect(() => kvCacheBytes(llama8b, 'fp16', 8192, 1.5)).toThrow(RangeError)
    expect(() => kvCacheBytes({ ...llama8b, numLayers: -1 }, 'fp16', 8192, 1)).toThrow(RangeError)
  })
})
