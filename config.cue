package plap

config: #Config & {
  database_url: "${PLAP_DATABASE_URL}"
  api_key_pepper: "${PLAP_API_KEY_PEPPER}"
  sealing_keys: "${PLAP_SEALING_KEYS}"
  log_level: *"INFO" | "${PLAP_LOG_LEVEL}"
  foreign_log_level: *"WARNING" | "${PLAP_FOREIGN_LOG_LEVEL}"

  llm_api_keys: {
    lightning: "${LIGHTNING_API_KEY}"
    cerebras: "${CEREBRAS_API_KEY}"
    groq: "${GROQ_API_KEY}"
    gmicloud: "${GMICLOUD_API_KEY}"
    novita: "${NOVITA_API_KEY}"
    fireworks: "${FIREWORKS_API_KEY}"
    crof: "${CROF_API_KEY}"
    qubrid: "${QUBRID_API_KEY}"
    openrouter: "${OPENROUTER_API_KEY}"
    vercel: "${VERCEL_API_KEY}"
    wandb: "${WANDB_API_KEY}"
  }

  display_name: *"plap-ai" | string
  model_info: #ModelInfoConfig & {
    display_name: *"plap-ai" | string
    description: *"plap responses model" | string
    mode: *"responses" | string
    input_modalities: *["text"] | [...string]
    output_modalities: ["text"]
    max_input_tokens: *1000000 | int
    max_output_tokens: *1000000 | int
    supported_parameters: [
      "context_management",
      "temperature",
      "top_p",
      "tools",
      "tool_choice",
      "parallel_tool_calls",
      "response_format",
      "max_output_tokens",
      "reasoning_effort",
      "service_tier",
      "stream",
    ]
    pricing: #PricingConfig & {
      input_per_token: *0.0 | number
      output_per_token: *0.0 | number
    }
    provider: *"plap" | string
    deprecated: *false | bool
  }
  reasoning_to_output: *1.0 | number
  sides: {
    main: 0
  }

  overlays: {
    "model": {
      "plap-ai/wisp": {
        display_name: "Wisp"
        model_info: display_name: "Wisp"
        model_info: description: "It's bigger."
        model_info: input_modalities: ["text", "image"]
        main: #FieldConfig & {
          model: *"crof/mimo-v2.5-pro,openrouter/xiaomi/mimo-v2.5-pro:xiaomi,openrouter/xiaomi/mimo-v2.5-pro:novita,gmicloud/XiaomiMiMo/MiMo-V2.5-Pro" | string
          max_completion_tokens: *131072 | int
          reasoning_effort: *"medium" | #ReasoningEffort
        }
        summary: #FieldConfig & {
          model: *"groq/openai/gpt-oss-20b,lightning/lightning-ai/gpt-oss-20b,wandb/openai/gpt-oss-20b,openrouter/openai/gpt-oss-20b:amazon-bedrock" | string
          max_completion_tokens: *768 | int
          reasoning_effort: *"low" | #ReasoningEffort
        }
        vision: #FieldConfig & {
          model: *"wandb/google/gemma-4-31B-it,openrouter/google/gemma-4-31b-it:novita,openrouter/google/gemma-4-31b-it:siliconflow,openrouter/google/gemma-4-31b-it:modelrun" | string
          max_completion_tokens: *8192 | int
          reasoning_effort: *"medium" | #ReasoningEffort
          sampling: {
            temperature: fixed: 1.0
            top_p: fixed: 0.95
          }
        }
        overlays: {
          "reasoning_effort": {
            "minimal": {
              main: reasoning_effort: "none"
              vision: reasoning_effort: "medium"
            }
            "low": {
              main: reasoning_effort: "low"
              vision: reasoning_effort: "medium"
            }
            "medium": {
              main: reasoning_effort: "medium"
              vision: reasoning_effort: "medium"
            }
            "high": {
              main: reasoning_effort: "high"
              vision: reasoning_effort: "medium"
            }
            "xhigh": {
              main: reasoning_effort: "high"
              vision: reasoning_effort: "medium"
            }
          }
        }
      }
      "plap-ai/mote": {
        display_name: "Mote"
        model_info: display_name: "Mote"
        model_info: description: "It's smaller."
        model_info: input_modalities: ["text", "image"]
        main: #FieldConfig & {
          model: *"openrouter/deepseek/deepseek-v4-flash:gmicloud,openrouter/deepseek/deepseek-v4-flash:baidu,openrouter/deepseek/deepseek-v4-flash:wafer,openrouter/deepseek/deepseek-v4-flash:novita" | string
          max_completion_tokens: *393216 | int
          reasoning_effort: *"high" | #ReasoningEffort
        }
        summary: #FieldConfig & {
          model: *"groq/openai/gpt-oss-20b,lightning/lightning-ai/gpt-oss-20b,wandb/openai/gpt-oss-20b,openrouter/openai/gpt-oss-20b:amazon-bedrock" | string
          max_completion_tokens: *768 | int
          reasoning_effort: *"low" | #ReasoningEffort
        }
        vision: #FieldConfig & {
          model: *"wandb/google/gemma-4-31B-it,openrouter/google/gemma-4-31b-it:novita,openrouter/google/gemma-4-31b-it:siliconflow,openrouter/google/gemma-4-31b-it:modelrun" | string
          max_completion_tokens: *8192 | int
          reasoning_effort: *"medium" | #ReasoningEffort
          sampling: {
            temperature: fixed: 1.0
            top_p: fixed: 0.95
          }
        }
        overlays: {
          "reasoning_effort": {
            "minimal": {
              main: reasoning_effort: "none"
              vision: reasoning_effort: "medium"
            }
            "low": {
              main: reasoning_effort: "high"
              vision: reasoning_effort: "medium"
            }
            "medium": {
              main: reasoning_effort: "high"
              vision: reasoning_effort: "medium"
            }
            "high": {
              main: reasoning_effort: "high"
              vision: reasoning_effort: "medium"
            }
            "xhigh": {
              main: reasoning_effort: "xhigh"
              vision: reasoning_effort: "medium"
            }
          }
        }
      }
    }
  }
}
