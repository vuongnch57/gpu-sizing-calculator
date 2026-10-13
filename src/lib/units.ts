/** Bytes in one gigabyte. We use decimal GB (10^9), the same unit GPU vendors use for VRAM. */
export const BYTES_PER_GB = 1e9

export function bytesToGB(bytes: number): number {
  return bytes / BYTES_PER_GB
}

/** "19.63 GB". Two decimals keep small KV caches visible next to large weights. */
export function formatGB(bytes: number): string {
  return `${bytesToGB(bytes).toLocaleString('en-US', {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  })} GB`
}
