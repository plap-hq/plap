import { createOpenAI } from "@ai-sdk/openai"

export function remapPlapProviderOptions(options) {
  const plapOptions = options?.providerOptions?.plap
  if (!plapOptions) return options

  return {
    ...options,
    providerOptions: {
      ...options.providerOptions,
      openai: {
        ...(options.providerOptions?.openai ?? {}),
        ...plapOptions,
        forceReasoning: true,
      },
    },
  }
}

export function wrapLanguageModel(model) {
  return new Proxy(model, {
    get(target, prop, receiver) {
      const value = Reflect.get(target, prop, receiver)
      if (prop !== "doGenerate" && prop !== "doStream") return value
      if (typeof value !== "function") return value
      return (options) => value.call(target, remapPlapProviderOptions(options))
    },
  })
}

export function createPlap(options = {}) {
  const { name: _ignoredName, ...providerOptions } = options
  const provider = createOpenAI(providerOptions)
  return new Proxy(provider, {
    get(target, prop, receiver) {
      if (prop === "responses") {
        return (modelId) => wrapLanguageModel(target.responses(modelId))
      }
      if (prop === "languageModel") {
        return (modelId) => wrapLanguageModel(target.responses(modelId))
      }
      if (prop === "chat") {
        const chat = Reflect.get(target, prop, receiver)
        if (typeof chat !== "function") return chat
        return (modelId) => wrapLanguageModel(chat.call(target, modelId))
      }
      return Reflect.get(target, prop, receiver)
    },
  })
}

export default createPlap
