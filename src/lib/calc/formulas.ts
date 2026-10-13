import { BYTES_PER_ELEMENT, type Precision } from './precision'

/** The parts of a model's architecture that drive inference memory. */
export interface ModelArchitecture {
  /** Total parameter count, e.g. 8.03e9 for Llama 3.1 8B. */
  paramCount: number
  /** Number of transformer layers (`num_hidden_layers` in a Hugging Face config.json). */
  numLayers: number
  /**
   * Number of key/value heads (`num_key_value_heads`).
   * Equal to the attention head count for classic multi-head attention (MHA),
   * smaller for grouped-query attention (GQA), and 1 for multi-query attention (MQA).
   */
  numKvHeads: number
  /** Size of each attention head (`head_dim`, or hidden_size / num_attention_heads). */
  headDim: number
}

/**
 * Memory for the model weights.
 *
 *   weights_bytes = param_count × bytes_per_param
 *
 * Every parameter has to sit in GPU memory for every forward pass, so this
 * part is fixed: it does not depend on context length or number of users.
 */
export function weightsBytes(paramCount: number, precision: Precision): number {
  assertPositive('paramCount', paramCount)
  return paramCount * BYTES_PER_ELEMENT[precision]
}

/**
 * KV cache memory for a single token.
 *
 *   per_token_bytes = 2 × num_layers × num_kv_heads × head_dim × bytes_per_element
 *
 * Why: when the model generates a token, each layer computes a key vector and a
 * value vector for every KV head. Those are cached so later tokens can attend to
 * them without recomputing. The 2 is one K plus one V.
 *
 * Why num_kv_heads and not num_attention_heads: with GQA several query heads
 * share one K/V head, so only the KV heads are stored. Using the attention head
 * count would overestimate Llama 3 8B's cache by 4× (32 heads vs 8 KV heads).
 */
export function kvCacheBytesPerToken(
  arch: Pick<ModelArchitecture, 'numLayers' | 'numKvHeads' | 'headDim'>,
  kvPrecision: Precision,
): number {
  assertPositiveInteger('numLayers', arch.numLayers)
  assertPositiveInteger('numKvHeads', arch.numKvHeads)
  assertPositiveInteger('headDim', arch.headDim)
  return 2 * arch.numLayers * arch.numKvHeads * arch.headDim * BYTES_PER_ELEMENT[kvPrecision]
}

/**
 * KV cache memory for a full workload.
 *
 *   kv_bytes = per_token_bytes × context_length × batch_size
 *
 * Every token of every concurrent sequence keeps its own K/V entries, so the
 * cache grows linearly with both context length and batch size. This is the
 * part that blows up for long contexts or many users.
 */
export function kvCacheBytes(
  arch: Pick<ModelArchitecture, 'numLayers' | 'numKvHeads' | 'headDim'>,
  kvPrecision: Precision,
  contextLength: number,
  batchSize: number,
): number {
  assertPositiveInteger('contextLength', contextLength)
  assertPositiveInteger('batchSize', batchSize)
  return kvCacheBytesPerToken(arch, kvPrecision) * contextLength * batchSize
}

export function assertPositive(name: string, value: number): void {
  if (!Number.isFinite(value) || value <= 0) {
    throw new RangeError(`${name} must be a positive number, got ${value}`)
  }
}

export function assertPositiveInteger(name: string, value: number): void {
  if (!Number.isInteger(value) || value <= 0) {
    throw new RangeError(`${name} must be a positive integer, got ${value}`)
  }
}
