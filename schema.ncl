let ReasoningEffort = [| 'none, 'minimal, 'low, 'medium, 'high, 'xhigh |] in

let FloatTransform = {
  disabled | Bool | default = false,
  fixed | Number | optional,
  default | Number | optional,
  scale | Number | default = 1.0,
  offset | Number | default = 0.0,
  min_value | Number | optional,
  max_value | Number | optional,
} in

let IntTransform = {
  disabled | Bool | default = false,
  fixed | Number | optional,
  default | Number | optional,
  min_value | Number | optional,
  max_value | Number | optional,
} in

let SamplingConfig = {
  temperature | FloatTransform | optional,
  top_p | FloatTransform | optional,
  top_logprobs | IntTransform | optional,
} in

let PublicUsageConfig = {
  uncached_input_to_output | Number | default = 0.25,
  cached_input_to_output | Number | default = 0.05,
  output_to_output | Number | default = 1.0,
} in

let PricingConfig = {
  input_per_token | Number,
  output_per_token | Number,
} in

let ModelInfoConfig = {
  display_name | String,
  description | String,
  mode | String,
  input_modalities | Array String,
  output_modalities | Array String,
  max_input_tokens | Number,
  max_output_tokens | Number,
  supported_parameters | Array String,
  pricing | PricingConfig,
  provider | String,
  deprecated | Bool | default = false,
} in

let FieldConfig = {
  model | String,
  max_completion_tokens | Number | optional,
  tokenizer_hf_repo | String | optional,
  tokenizer_revision | String | optional,
  tokenizer_trust_remote_code | Bool | default = false,
  reasoning_effort | ReasoningEffort | optional,
  service_tier | String | optional,
  sampling | SamplingConfig | default = {},
  public_usage | PublicUsageConfig | default = {},
} in

{
  display_name | String,
  model_info | ModelInfoConfig,
  default_reasoning_effort | ReasoningEffort | optional,
  main | FieldConfig,
  compactor | FieldConfig,
  defender | FieldConfig,
  reviewer | FieldConfig,
  arbitrator | FieldConfig,
  reasoning_summarizer | FieldConfig,
  reviewer_max_transcript_tokens | Number,
  arbitrator_max_transcript_tokens | Number,
  compact_threshold | Number | optional,
  compact_max_rounds | Number,
  debate_max_rounds | Number | default = 2,
  reasoning_to_output | Number | default = 1.0,
  by_model | { _ | { .. } } | optional,
  by_reasoning_effort | { _ | { .. } },
  by_service_tier | { _ | { .. } } | optional,
}
