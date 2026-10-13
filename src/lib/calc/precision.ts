/**
 * Numeric formats a model's weights or KV cache can be stored in.
 *
 * FP16 and BF16 are both 16-bit; they differ in how bits are split between
 * exponent and mantissa, not in size, so they use the same memory.
 */
export type Precision = 'fp32' | 'fp16' | 'bf16' | 'fp8' | 'int8' | 'int4'

/**
 * Bytes needed to store one number (one weight, or one K/V element).
 *
 * INT4 packs two values into one byte, hence 0.5. Real INT4 checkpoints
 * (GPTQ, AWQ) also store small per-group scale factors, so actual INT4
 * weights are slightly larger than this; the overhead margin covers that.
 */
export const BYTES_PER_ELEMENT: Record<Precision, number> = {
  fp32: 4,
  fp16: 2,
  bf16: 2,
  fp8: 1,
  int8: 1,
  int4: 0.5,
}

export const PRECISION_LABELS: Record<Precision, string> = {
  fp32: 'FP32',
  fp16: 'FP16',
  bf16: 'BF16',
  fp8: 'FP8',
  int8: 'INT8',
  int4: 'INT4',
}
