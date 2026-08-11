# Documentation

Plugins extend the Responses server.

## Use the server

- [Use the Responses API](../README.md#send-a-response) sends a response with the OpenAI client.
- [Use Chat Completions](chat-completions.md) calls plap from an existing Chat client and preserves conversations across turns.

## Add server functionality

- [Write your first plugin](first-plugin.md) adds a `server_time` tool and calls it through the Responses API.
- [Server tools](easy/server-tools.md) covers arguments, results, saved history, collisions, and failures.

## Modify execution

- [Built-in plugins](examples.md) covers the example plugins that ship with plap.
- [Event bus](bus.md) defines how plugin handlers compose.
- [Hooks](hooks.md) covers the response and bootstrap events that plugins can modify.
- [Reasoning summaries](summary.md) stream progress while the main answer remains private.
- [Response state](state.md) covers request data, configuration, memory, model context, and services.
- [Threads](threads.md) covers isolated model histories, active client-tool work, and parking.

## Add application components

- [Bootstrap helpers](easy/bootstrap.md) add config, routes, services, and shutdown hooks without writing bus listeners.

## Use the LLM library

- [Make your first completion](llms/getting-started.md) calls OpenRouter directly without starting the Responses server.
- [Completion retries and validators](llms/retries.md) explains how to reject an unusable model result and ask the model to
  correct it.
- The [`plap.llms` reference](llms/README.md) covers messages, providers, model whitelists, routing, streaming, retries, model
  JSON, completion budgets, and token measurement.
