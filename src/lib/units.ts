/** Bytes in one gigabyte. We use decimal GB (10^9), the same unit GPU vendors use for VRAM. */
export const BYTES_PER_GB = 1e9

export function bytesToGB(bytes: number): number {
  return bytes / BYTES_PER_GB
}
