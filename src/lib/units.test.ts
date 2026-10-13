import { describe, expect, it } from 'vitest'
import { bytesToGB } from './units'

describe('bytesToGB', () => {
  it('uses decimal gigabytes (1e9 bytes)', () => {
    expect(bytesToGB(80e9)).toBe(80)
    expect(bytesToGB(2 ** 30)).toBeCloseTo(1.0737, 4)
  })
})
