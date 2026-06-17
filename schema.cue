package plap

#ReasoningEffort: "none" | "minimal" | "low" | "medium" | "high" | "xhigh"

#FloatTransform: {
  disabled?: bool | *false
  fixed?: number
  default?: number
  scale?: number | *1.0
  offset?: number | *0.0
  min_value?: number
  max_value?: number
}

#IntTransform: {
  disabled?: bool | *false
  fixed?: number
  default?: number
  min_value?: number
  max_value?: number
}

#SamplingConfig: {
  temperature?: #FloatTransform
  top_p?: #FloatTransform
  top_logprobs?: #IntTransform
}

#PublicUsageConfig: {
  uncached_input_to_output?: number | *0.25
  cached_input_to_output?: number | *0.05
  output_to_output?: number | *1.0
}

#PricingConfig: {
  input_per_token!: number
  output_per_token!: number
}

#ModelInfoConfig: {
  display_name!: string
  description!: string
  mode!: string
  input_modalities!: [...string]
  output_modalities!: [...string]
  max_input_tokens!: int
  max_output_tokens!: int
  supported_parameters!: [...string]
  pricing!: #PricingConfig
  provider!: string
  deprecated?: bool | *false
}

#FieldConfig: {
  model!: string
  max_completion_tokens?: int
  tokenizer_hf_repo?: string
  tokenizer_revision?: string
  tokenizer_trust_remote_code?: bool | *false
  reasoning_effort?: #ReasoningEffort
  service_tier?: string
  sampling?: #SamplingConfig | *{}
  public_usage?: #PublicUsageConfig | *{}
}

#Config: {
  display_name!: string
  model_info!: #ModelInfoConfig
  default_reasoning_effort?: #ReasoningEffort
  main!: #FieldConfig
  compactor!: #FieldConfig
  defender!: #FieldConfig
  reviewer!: #FieldConfig
  arbitrator!: #FieldConfig
  reasoning_summarizer!: #FieldConfig
  reviewer_max_transcript_tokens!: int
  arbitrator_max_transcript_tokens!: int
  compact_threshold?: int
  compact_max_rounds!: int
  debate_max_rounds?: int | *2
  reasoning_to_output?: number | *1.0
  overrides?: { [string]: { [string]: _ } }
}
