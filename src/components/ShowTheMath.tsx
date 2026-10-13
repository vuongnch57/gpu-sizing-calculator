import type { MathStep } from '../lib/mathSteps'

export function ShowTheMath({ steps }: { steps: MathStep[] }) {
  return (
    <details className="group">
      <summary className="cursor-pointer text-base font-semibold select-none">
        Show the math
      </summary>
      <ol className="mt-4 flex flex-col gap-4">
        {steps.map((step) => (
          <li key={step.label} className="flex flex-col gap-1 text-sm">
            <div className="flex items-baseline justify-between gap-3">
              <span className="font-medium">{step.label}</span>
              <span className="font-semibold whitespace-nowrap tabular-nums">{step.result}</span>
            </div>
            <code className="block rounded bg-slate-50 px-2 py-1 font-mono text-xs text-slate-600">
              {step.formula}
            </code>
            <code className="block rounded bg-slate-50 px-2 py-1 font-mono text-xs break-words text-slate-900">
              = {step.substituted}
            </code>
            <p className="text-xs text-slate-500">{step.why}</p>
          </li>
        ))}
      </ol>
    </details>
  )
}
