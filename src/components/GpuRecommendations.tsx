import type { GpuSpec } from '../data/gpus'
import type { GpuFit } from '../lib/calc'
import { formatGB } from '../lib/units'

interface GpuRecommendationsProps {
  fits: GpuFit<GpuSpec>[]
  maxCount: number
}

export function GpuRecommendations({ fits, maxCount }: GpuRecommendationsProps) {
  const sorted = [...fits].sort(compareFits)

  return (
    <section aria-labelledby="gpus-heading" className="flex flex-col gap-3">
      <div>
        <h2 id="gpus-heading" className="text-base font-semibold">
          GPUs that fit
        </h2>
        <p className="mt-1 text-xs text-slate-500">
          Fewest GPUs of each type, split with tensor parallelism. Memory shown per GPU.
        </p>
      </div>

      <ul className="flex flex-col divide-y divide-slate-100">
        {sorted.map((fit) => (
          <GpuRow key={fit.gpu.id} fit={fit} maxCount={maxCount} />
        ))}
      </ul>
    </section>
  )
}

function GpuRow({ fit, maxCount }: { fit: GpuFit<GpuSpec>; maxCount: number }) {
  const capacity = fit.gpu.memoryGB * 1e9
  const used = Math.min(fit.perGpu.total / capacity, 1)
  const fits = fit.count !== null

  return (
    <li className={`flex flex-col gap-1.5 py-3 ${fits ? '' : 'opacity-60'}`}>
      <div className="flex items-baseline justify-between gap-3 text-sm">
        <span className="font-medium">
          {fits ? `${fit.count} × ` : ''}
          {fit.gpu.name}
        </span>
        <span className="text-right whitespace-nowrap text-slate-600 tabular-nums">
          {fits
            ? `${formatGB(fit.headroomBytes)} free per GPU`
            : `Doesn’t fit on ${maxCount} or fewer`}
        </span>
      </div>
      <div
        className="h-2 w-full overflow-hidden rounded-full bg-slate-100"
        role="img"
        aria-label={`${formatGB(fit.perGpu.total)} of ${fit.gpu.memoryGB} GB per GPU`}
      >
        <div
          className={
            fits
              ? used > 0.9
                ? 'h-full bg-amber-400'
                : 'h-full bg-indigo-500'
              : 'h-full bg-red-400'
          }
          style={{ width: `${used * 100}%` }}
        />
      </div>
      <p className="text-xs text-slate-500 tabular-nums">
        {formatGB(fit.perGpu.total)} of {fit.gpu.memoryGB} GB per GPU
        {fits && used > 0.9 && ' · tight, little room for more users or longer context'}
      </p>
    </li>
  )
}

/** Fitting setups first: fewest GPUs, then least total memory. Non-fitting last. */
function compareFits(a: GpuFit<GpuSpec>, b: GpuFit<GpuSpec>): number {
  if (a.count === null || b.count === null) {
    if (a.count === b.count) return a.gpu.memoryGB - b.gpu.memoryGB
    return a.count === null ? 1 : -1
  }
  return a.count - b.count || a.gpu.memoryGB - b.gpu.memoryGB
}
