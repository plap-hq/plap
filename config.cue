package plap

config: #Config & {
  database_url: "${PLAP_DATABASE_URL}"
  api_key_pepper: "${PLAP_API_KEY_PEPPER}"
  sealing_keys: "${PLAP_SEALING_KEYS}"
  plugins: ["core"]
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
  }

  display_name: *"plap-ai" | string
  model_info: #ModelInfoConfig & {
    display_name: *"plap-ai" | string
    description: *"plap responses model" | string
    mode: *"responses" | string
    input_modalities: ["text"]
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
  default_reasoning_effort: *"medium" | #ReasoningEffort
  main: #FieldConfig & {
    model: *"crof/mimo-v2.5-pro,openrouter/xiaomi/mimo-v2.5-pro:xiaomi,gmicloud/XiaomiMiMo/MiMo-V2.5-Pro" | string
    max_completion_tokens: *131072 | int
    reasoning_effort: *"medium" | #ReasoningEffort
  }
  reasoning_summarizer: #FieldConfig & {
    model: *"groq/openai/gpt-oss-20b,lightning/lightning-ai/gpt-oss-20b,openrouter/openai/gpt-oss-20b:wandb,openrouter/openai/gpt-oss-20b:amazon-bedrock" | string
  }
  reasoning_to_output: *1.0 | number
  sides: {
    main: 0
  }

  overlays: {
    "reasoning_effort": {
      "minimal": { main: reasoning_effort: "none" }
      "low": { main: reasoning_effort: "low" }
      "medium": { main: reasoning_effort: "medium" }
      "high": { main: reasoning_effort: "high" }
      "xhigh": { main: reasoning_effort: "xhigh" }
    }
    "model": {
      "plap-ai/wisp": {
        display_name: "Wisp"
        model_info: display_name: "Wisp"
        model_info: description: "General-purpose high-efficiency model optimized for balanced performance."
        main: model: "crof/mimo-v2.5-pro,openrouter/xiaomi/mimo-v2.5-pro:xiaomi,gmicloud/XiaomiMiMo/MiMo-V2.5-Pro"
        main: max_completion_tokens: 131072
      }
      "plap-ai/wisp-mini": {
        display_name: "Wisp Mini"
        model_info: display_name: "Wisp Mini"
        model_info: description: "General-purpose plap responses model for text and tool use."
        main: model: "qubrid/deepseek-ai/DeepSeek-V4-Flash,openrouter/deepseek/deepseek-v4-flash:novita,openrouter/deepseek/deepseek-v4-flash:atlas-cloud"
        main: max_completion_tokens: 393216
      }
    }
  }
}
