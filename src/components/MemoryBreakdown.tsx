import type { MemoryBreakdown as Breakdown } from '../lib/calc'
import { formatGB } from '../lib/units'

const SEGMENTS = [
  {
    key: 'weights',
    label: 'Weights',
    color: 'bg-indigo-500',
    hint: 'Fixed: every parameter stays in memory',
  },
  {
    key: 'kvCache',
    label: 'KV cache',
    color: 'bg-emerald-500',
    hint: 'Grows with context length × users',
  },
  {
    key: 'overhead',
    label: 'Overhead',
    color: 'bg-amber-400',
    hint: 'CUDA context, activations, fragmentation',
  },
] as const

export function MemoryBreakdown({ breakdown }: { breakdown: Breakdown }) {
  return (
    <section aria-labelledby="breakdown-heading" className="flex flex-col gap-4">
      <div className="flex items-baseline justify-between gap-4">
        <h2 id="breakdown-heading" className="text-base font-semibold">
          Estimated memory
        </h2>
        <p className="text-xs text-slate-500">Estimate · GB = 10⁹ bytes</p>
      </div>

      <p className="text-4xl font-semibold tracking-tight tabular-nums">
        {formatGB(breakdown.total)}
      </p>

      <div
        className="flex h-4 w-full overflow-hidden rounded-full bg-slate-100"
        role="img"
        aria-label={SEGMENTS.map((s) => `${s.label} ${formatGB(breakdown[s.key])}`).join(', ')}
      >
        {SEGMENTS.map((s) => (
          <div
            key={s.key}
            className={s.color}
            style={{ width: `${(breakdown[s.key] / breakdown.total) * 100}%` }}
          />
        ))}
      </div>

      <table className="w-full text-sm">
        <tbody>
          {SEGMENTS.map((s) => (
            <tr key={s.key} className="border-b border-slate-100">
              <td className="py-2 pr-2">
                <span className={`mr-2 inline-block size-2.5 rounded-full ${s.color}`} />
                {s.label}
                <span className="block pl-[1.125rem] text-xs text-slate-500">{s.hint}</span>
              </td>
              <td className="py-2 text-right whitespace-nowrap tabular-nums">
                {formatGB(breakdown[s.key])}
              </td>
              <td className="w-14 py-2 text-right text-slate-500 tabular-nums">
                {Math.round((breakdown[s.key] / breakdown.total) * 100)}%
              </td>
            </tr>
          ))}
          <tr className="font-semibold">
            <td className="py-2">Total</td>
            <td className="py-2 text-right whitespace-nowrap tabular-nums">
              {formatGB(breakdown.total)}
            </td>
            <td />
          </tr>
        </tbody>
      </table>
    </section>
  )
}
