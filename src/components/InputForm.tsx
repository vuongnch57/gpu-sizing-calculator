import { MODEL_PRESETS } from '../data/models'
import { CUSTOM_MODEL_ID, findPreset, pickArchitecture, type CalculatorInputs } from '../inputs'
import { PRECISION_LABELS, type Precision } from '../lib/calc'
import { Field, inputClass } from './Field'

interface InputFormProps {
  inputs: CalculatorInputs
  onChange: (patch: Partial<CalculatorInputs>) => void
}

const PRECISIONS = Object.keys(PRECISION_LABELS) as Precision[]
const CONTEXT_SUGGESTIONS = [2048, 4096, 8192, 16_384, 32_768, 65_536, 131_072]

export function InputForm({ inputs, onChange }: InputFormProps) {
  const preset = findPreset(inputs.modelId)
  const isCustom = !preset
  // Preset fields are shown read-only so users can see what drives the numbers.
  const arch = preset ?? inputs.custom

  function selectModel(modelId: string) {
    if (modelId === CUSTOM_MODEL_ID) {
      // Start custom from the preset that was showing, so the fields aren't blank.
      onChange({ modelId, custom: preset ? pickArchitecture(preset) : inputs.custom })
    } else {
      onChange({ modelId })
    }
  }

  function setCustom(field: keyof CalculatorInputs['custom'], value: number) {
    onChange({ custom: { ...inputs.custom, [field]: value } })
  }

  const overMaxContext = preset && inputs.contextLength > preset.maxContextLength

  return (
    <form className="flex flex-col gap-6" onSubmit={(e) => e.preventDefault()}>
      <fieldset className="flex flex-col gap-4">
        <legend className="mb-2 text-base font-semibold">Model</legend>
        <Field label="Preset" htmlFor="model">
          <select
            id="model"
            className={inputClass}
            value={inputs.modelId}
            onChange={(e) => selectModel(e.target.value)}
          >
            {MODEL_PRESETS.map((m) => (
              <option key={m.id} value={m.id}>
                {m.name}
              </option>
            ))}
            <option value={CUSTOM_MODEL_ID}>Custom…</option>
          </select>
        </Field>

        <div className="grid grid-cols-2 gap-4">
          <Field label="Parameters (billions)" htmlFor="params">
            <input
              id="params"
              type="number"
              min={0}
              step="any"
              className={inputClass}
              disabled={!isCustom}
              value={numberValue(round(arch.paramCount / 1e9, 3))}
              onChange={(e) => setCustom('paramCount', e.target.valueAsNumber * 1e9)}
            />
          </Field>
          <Field label="Layers" htmlFor="layers">
            <input
              id="layers"
              type="number"
              min={1}
              className={inputClass}
              disabled={!isCustom}
              value={numberValue(arch.numLayers)}
              onChange={(e) => setCustom('numLayers', e.target.valueAsNumber)}
            />
          </Field>
          <Field
            label="KV heads"
            htmlFor="kv-heads"
            hint={
              preset &&
              `${preset.numAttentionHeads} attention heads share ${preset.numKvHeads} KV heads`
            }
          >
            <input
              id="kv-heads"
              type="number"
              min={1}
              className={inputClass}
              disabled={!isCustom}
              value={numberValue(arch.numKvHeads)}
              onChange={(e) => setCustom('numKvHeads', e.target.valueAsNumber)}
            />
          </Field>
          <Field label="Head dim" htmlFor="head-dim">
            <input
              id="head-dim"
              type="number"
              min={1}
              className={inputClass}
              disabled={!isCustom}
              value={numberValue(arch.headDim)}
              onChange={(e) => setCustom('headDim', e.target.valueAsNumber)}
            />
          </Field>
        </div>

        {preset && (
          <div className="text-xs text-slate-500">
            <a
              href={preset.source}
              target="_blank"
              rel="noreferrer"
              className="underline hover:text-slate-700"
            >
              Source: config.json
            </a>
            {preset.notes?.map((note) => (
              <p key={note} className="mt-1 text-amber-700">
                {note}
              </p>
            ))}
          </div>
        )}
      </fieldset>

      <fieldset className="flex flex-col gap-4">
        <legend className="mb-2 text-base font-semibold">Workload</legend>
        <div className="grid grid-cols-2 gap-4">
          <Field label="Weight precision" htmlFor="weight-precision">
            <PrecisionSelect
              id="weight-precision"
              value={inputs.weightPrecision}
              onChange={(weightPrecision) => onChange({ weightPrecision })}
            />
          </Field>
          <Field label="KV cache precision" htmlFor="kv-precision">
            <PrecisionSelect
              id="kv-precision"
              value={inputs.kvPrecision}
              onChange={(kvPrecision) => onChange({ kvPrecision })}
            />
          </Field>
          <Field
            label="Context length (tokens)"
            htmlFor="context"
            hint={
              overMaxContext ? (
                <span className="text-amber-700">
                  Above this model’s native maximum of {preset.maxContextLength.toLocaleString()}.
                </span>
              ) : (
                'Prompt plus generated tokens, per user'
              )
            }
          >
            <input
              id="context"
              type="number"
              min={1}
              list="context-suggestions"
              className={inputClass}
              value={numberValue(inputs.contextLength)}
              onChange={(e) => onChange({ contextLength: e.target.valueAsNumber })}
            />
            <datalist id="context-suggestions">
              {CONTEXT_SUGGESTIONS.map((n) => (
                <option key={n} value={n} />
              ))}
            </datalist>
          </Field>
          <Field label="Concurrent users (batch)" htmlFor="batch">
            <input
              id="batch"
              type="number"
              min={1}
              className={inputClass}
              value={numberValue(inputs.batchSize)}
              onChange={(e) => onChange({ batchSize: e.target.valueAsNumber })}
            />
          </Field>
          <Field
            label="Overhead margin (%)"
            htmlFor="overhead"
            hint="CUDA context, activations, fragmentation. 10–20% is typical."
          >
            <input
              id="overhead"
              type="number"
              min={0}
              max={100}
              className={inputClass}
              value={numberValue(inputs.overheadPercent)}
              onChange={(e) => onChange({ overheadPercent: e.target.valueAsNumber })}
            />
          </Field>
        </div>
      </fieldset>
    </form>
  )
}

function PrecisionSelect(props: {
  id: string
  value: Precision
  onChange: (value: Precision) => void
}) {
  return (
    <select
      id={props.id}
      className={inputClass}
      value={props.value}
      onChange={(e) => props.onChange(e.target.value as Precision)}
    >
      {PRECISIONS.map((p) => (
        <option key={p} value={p}>
          {PRECISION_LABELS[p]}
        </option>
      ))}
    </select>
  )
}

/** Lets a number input be cleared while typing without React warning about NaN. */
function numberValue(value: number): number | '' {
  return Number.isFinite(value) ? value : ''
}

function round(value: number, digits: number): number {
  const f = 10 ** digits
  return Math.round(value * f) / f
}
