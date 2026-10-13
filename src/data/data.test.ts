import { describe, expect, it } from 'vitest'
import { kvCacheBytesPerToken, weightsBytes } from '../lib/calc'
import { GPU_COUNTS, GPUS } from './gpus'
import { MODEL_PRESETS } from './models'

describe('MODEL_PRESETS', () => {
  it('has unique ids', () => {
    const ids = MODEL_PRESETS.map((m) => m.id)
    expect(new Set(ids).size).toBe(ids.length)
  })

  it.each(MODEL_PRESETS)('$name has a valid architecture', (m) => {
    expect(() => weightsBytes(m.paramCount, 'fp16')).not.toThrow()
    expect(() => kvCacheBytesPerToken(m, 'fp16')).not.toThrow()
    // GQA: query heads are split evenly into groups, one group per KV head
    expect(m.numAttentionHeads % m.numKvHeads).toBe(0)
    expect(m.source).toMatch(/^https:\/\/huggingface\.co\/.+\/config\.json$/)
  })

  it('every GPU count splits every preset’s KV heads evenly', () => {
    // Keeps per-GPU KV estimates exact for the presets we ship.
    for (const m of MODEL_PRESETS) {
      for (const n of GPU_COUNTS) {
        if (n <= m.numKvHeads) expect(m.numKvHeads % n).toBe(0)
      }
    }
  })
})

describe('GPUS', () => {
  it('has unique ids and is sorted by memory', () => {
    expect(new Set(GPUS.map((g) => g.id)).size).toBe(GPUS.length)
    const mem = GPUS.map((g) => g.memoryGB)
    expect(mem).toEqual([...mem].sort((a, b) => a - b))
  })
})
