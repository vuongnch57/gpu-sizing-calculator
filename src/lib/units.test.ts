import { describe, expect, it } from 'vitest'
import { bytesToGB, formatGB } from './units'

describe('bytesToGB', () => {
  it('uses decimal gigabytes (1e9 bytes)', () => {
    expect(bytesToGB(80e9)).toBe(80)
    expect(bytesToGB(2 ** 30)).toBeCloseTo(1.0737, 4)
  })
})

describe('formatGB', () => {
  it('shows two decimals and thousands separators', () => {
    expect(formatGB(19.6346e9)).toBe('19.63 GB')
    expect(formatGB(1234.5e9)).toBe('1,234.50 GB')
  })
})
