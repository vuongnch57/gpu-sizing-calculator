import type { ModelArchitecture } from '../lib/calc'

export interface ModelPreset extends ModelArchitecture {
  id: string
  name: string
  /** `hidden_size` in config.json. Shown for reference; the math uses headDim. */
  hiddenSize: number
  /** `num_attention_heads`. Shown to contrast with numKvHeads (GQA). */
  numAttentionHeads: number
  /** `max_position_embeddings`: the longest context the model supports natively. */
  maxContextLength: number
  /** The config.json the architecture values were copied from. */
  source: string
  /** Caveats that affect the estimate, shown next to the preset. */
  notes?: string[]
}

/*
 * Architecture values are copied from each model's config.json on Hugging Face
 * (the `source` URL; for gated repos, an ungated mirror with an identical config).
 *
 * paramCount is not in config.json. It is computed from the config
 * (embeddings + per-layer attention, MLP and norms) and matches the totals the
 * model authors publish (e.g. 8.03B for Llama 3.1 8B).
 */
export const MODEL_PRESETS: ModelPreset[] = [
  {
    id: 'llama-3.1-8b',
    name: 'Llama 3.1 8B',
    paramCount: 8_030_261_248,
    numLayers: 32,
    hiddenSize: 4096,
    numAttentionHeads: 32,
    numKvHeads: 8,
    headDim: 128,
    maxContextLength: 131_072,
    source: 'https://huggingface.co/unsloth/Meta-Llama-3.1-8B-Instruct/blob/main/config.json',
  },
  {
    id: 'llama-3.1-70b',
    name: 'Llama 3.1 70B',
    paramCount: 70_553_706_496,
    numLayers: 80,
    hiddenSize: 8192,
    numAttentionHeads: 64,
    numKvHeads: 8,
    headDim: 128,
    maxContextLength: 131_072,
    source: 'https://huggingface.co/unsloth/Meta-Llama-3.1-70B-Instruct/blob/main/config.json',
  },
  {
    id: 'qwen-2.5-7b',
    name: 'Qwen 2.5 7B',
    paramCount: 7_615_616_512,
    numLayers: 28,
    hiddenSize: 3584,
    numAttentionHeads: 28,
    numKvHeads: 4,
    headDim: 128,
    maxContextLength: 32_768,
    source: 'https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/config.json',
    notes: ['Supports up to 131,072 tokens with YaRN rope scaling, which is off by default.'],
  },
  {
    id: 'qwen-2.5-72b',
    name: 'Qwen 2.5 72B',
    paramCount: 72_706_203_648,
    numLayers: 80,
    hiddenSize: 8192,
    numAttentionHeads: 64,
    numKvHeads: 8,
    headDim: 128,
    maxContextLength: 32_768,
    source: 'https://huggingface.co/Qwen/Qwen2.5-72B-Instruct/blob/main/config.json',
    notes: ['Supports up to 131,072 tokens with YaRN rope scaling, which is off by default.'],
  },
  {
    id: 'mistral-7b-v0.3',
    name: 'Mistral 7B v0.3',
    paramCount: 7_248_023_552,
    numLayers: 32,
    hiddenSize: 4096,
    numAttentionHeads: 32,
    numKvHeads: 8,
    headDim: 128,
    maxContextLength: 32_768,
    source: 'https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3/blob/main/config.json',
  },
  {
    id: 'gemma-2-9b',
    name: 'Gemma 2 9B',
    paramCount: 9_241_705_984,
    numLayers: 42,
    hiddenSize: 3584,
    numAttentionHeads: 16,
    numKvHeads: 8,
    // Not hidden_size / num_attention_heads (= 224): Gemma 2 sets head_dim explicitly.
    headDim: 256,
    maxContextLength: 8192,
    source: 'https://huggingface.co/unsloth/gemma-2-9b-it/blob/main/config.json',
    notes: [
      'Half the layers use a 4,096-token sliding window, so above 4K context the KV cache is overestimated by up to 25%.',
    ],
  },
  {
    id: 'gemma-2-27b',
    name: 'Gemma 2 27B',
    paramCount: 27_227_128_320,
    numLayers: 46,
    hiddenSize: 4608,
    numAttentionHeads: 32,
    numKvHeads: 16,
    headDim: 128,
    maxContextLength: 8192,
    source: 'https://huggingface.co/unsloth/gemma-2-27b-it/blob/main/config.json',
    notes: [
      'Half the layers use a 4,096-token sliding window, so above 4K context the KV cache is overestimated by up to 25%.',
    ],
  },
]
