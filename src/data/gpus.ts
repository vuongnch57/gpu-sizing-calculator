export interface GpuSpec {
  id: string
  name: string
  /**
   * Memory per GPU in GB (10^9 bytes), as NVIDIA lists it. This is the full
   * capacity; the driver reserves a few hundred MB, which the overhead margin covers.
   */
  memoryGB: number
  /** Datacenter cards are the usual inference targets; consumer cards are listed for local setups. */
  segment: 'datacenter' | 'workstation' | 'consumer'
  source: string
}

/** Sorted by memory, smallest first. Values from NVIDIA's product pages. */
export const GPUS: GpuSpec[] = [
  {
    id: 'rtx-4090',
    name: 'RTX 4090 24GB',
    memoryGB: 24,
    segment: 'consumer',
    source: 'https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/',
  },
  {
    id: 'l4',
    name: 'L4 24GB',
    memoryGB: 24,
    segment: 'datacenter',
    source: 'https://www.nvidia.com/en-us/data-center/l4/',
  },
  {
    id: 'a10',
    name: 'A10 24GB',
    memoryGB: 24,
    segment: 'datacenter',
    source: 'https://www.nvidia.com/en-us/data-center/products/a10-gpu/',
  },
  {
    id: 'a100-40',
    name: 'A100 40GB',
    memoryGB: 40,
    segment: 'datacenter',
    source: 'https://www.nvidia.com/en-us/data-center/a100/',
  },
  {
    id: 'l40s',
    name: 'L40S 48GB',
    memoryGB: 48,
    segment: 'datacenter',
    source: 'https://www.nvidia.com/en-us/data-center/l40s/',
  },
  {
    id: 'a100-80',
    name: 'A100 80GB',
    memoryGB: 80,
    segment: 'datacenter',
    source: 'https://www.nvidia.com/en-us/data-center/a100/',
  },
  {
    id: 'h100-80',
    name: 'H100 80GB',
    memoryGB: 80,
    segment: 'datacenter',
    source: 'https://www.nvidia.com/en-us/data-center/h100/',
  },
  {
    id: 'h200',
    name: 'H200 141GB',
    memoryGB: 141,
    segment: 'datacenter',
    source: 'https://www.nvidia.com/en-us/data-center/h200/',
  },
]

/** Tensor-parallel sizes we consider. TP is almost always a power of two within one server. */
export const GPU_COUNTS = [1, 2, 4, 8] as const
