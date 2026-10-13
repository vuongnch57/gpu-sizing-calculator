import { MODEL_PRESETS, type ModelPreset } from './data/models'
import type { InferenceSettings, ModelArchitecture } from './lib/calc'

export const CUSTOM_MODEL_ID = 'custom'

/** Everything the form controls. Kept flat so it can map to URL params later. */
export interface CalculatorInputs extends InferenceSettings {
  /** A preset id, or CUSTOM_MODEL_ID to use `custom`. */
  modelId: string
  /** The architecture used when modelId is CUSTOM_MODEL_ID. */
  custom: ModelArchitecture
}

const defaultPreset = MODEL_PRESETS[0]

export const DEFAULT_INPUTS: CalculatorInputs = {
  modelId: defaultPreset.id,
  custom: pickArchitecture(defaultPreset),
  weightPrecision: 'bf16',
  kvPrecision: 'bf16',
  contextLength: 8192,
  batchSize: 1,
  overheadPercent: 15,
}

export function findPreset(modelId: string): ModelPreset | undefined {
  return MODEL_PRESETS.find((m) => m.id === modelId)
}

/** The architecture the calculation should use for the current inputs. */
export function resolveModel(inputs: CalculatorInputs): ModelArchitecture {
  const preset = findPreset(inputs.modelId)
  return preset ? pickArchitecture(preset) : inputs.custom
}

export function pickArchitecture(m: ModelArchitecture): ModelArchitecture {
  return {
    paramCount: m.paramCount,
    numLayers: m.numLayers,
    numKvHeads: m.numKvHeads,
    headDim: m.headDim,
  }
}
