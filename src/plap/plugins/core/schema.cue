package plap

#ReasoningEffort: "none" | "minimal" | "low" | "medium" | "high" | "xhigh" | "max"

#FloatTransform: {
  disabled: *false | bool
  fixed: *null | number
  default: *null | number
  scale: *1.0 | number
  offset: *0.0 | number
  min_value: *null | number
  max_value: *null | number
}

#IntTransform: {
  disabled: *false | bool
  fixed: *null | number
  default: *null | number
  min_value: *null | number
  max_value: *null | number
}

#SamplingConfig: {
  temperature: *null | #FloatTransform
  top_p: *null | #FloatTransform
  min_p: *null | #FloatTransform
  top_k: *null | #IntTransform
  frequency_penalty: *null | #FloatTransform
  presence_penalty: *null | #FloatTransform
  repetition_penalty: *null | #FloatTransform
  seed: *null | #IntTransform
  top_logprobs: *null | #IntTransform
}

#OutputEquivalence: {
  uncached_input_to_output: *0.25 | >=0
  cached_input_to_output: *0.05 | >=0
  output_to_output: *1.0 | >0
}

#PricingConfig: {
  input_per_token!: number
  output_per_token!: number
}

#ModelInfoConfig: {
  display_name!: string
  description!: string
  created!: uint
  mode!: string
  input_modalities!: [...string]
  output_modalities!: [...string]
  max_input_tokens!: int
  max_output_tokens!: int
  supported_parameters!: [...string]
  pricing!: #PricingConfig
  provider!: string
  deprecated: *false | bool
}

#FieldConfig: {
  model!: string
  max_completion_tokens: *null | int
  tokenizer_hf_repo: *null | string
  tokenizer_revision: *null | string
  tokenizer_trust_remote_code: *false | bool
  reasoning_effort: *null | #ReasoningEffort
  service_tier: *null | string
  sampling: #SamplingConfig
  output_equivalence: #OutputEquivalence
}

#WellKnownSideCode: uint16 & <1024
#RegisteredSideCode: uint16 & >=1024 & <49152
#PrivateSideCode: uint16 & >=49152

#WellKnownSides: {
  main: 0
}

#RegisteredSides: {}

#PrivateSides: {}

#SideCodes: {
  for name, code in #WellKnownSides {
    "\(name)": code & #WellKnownSideCode
  }
  for name, code in #RegisteredSides {
    "\(name)": code & #RegisteredSideCode
  }
  for name, code in #PrivateSides {
    "\(name)": code & #PrivateSideCode
  }
}

#Config: {
  database_url!: string
  api_key_pepper!: string
  sealing_keys!: string
  log_level: *"INFO" | string
  foreign_log_level: *"WARNING" | string
  llm_api_keys: { [string]: string }

  display_name!: string
  model_info!: #ModelInfoConfig
  main?: #FieldConfig
  reasoning_to_output: *1.0 | >=0
  sides: #SideCodes
  _sidesByCode: {
    for name, code in sides {
      "\(code)": name
    }
  }
  overlays: *{} | { [string]: { [string]: _ } }
  overlays: {
    model?: {
      [string]: {
        main!: #FieldConfig
        [string]: _
      }
    }
  }
}
