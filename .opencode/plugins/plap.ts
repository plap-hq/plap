import type { Plugin } from "@opencode-ai/plugin"

export default (async () => ({
  "chat.params": async (input, output) => {
    if (input.model.providerID !== "plap") return
    output.maxOutputTokens = undefined
  },
})) satisfies Plugin
