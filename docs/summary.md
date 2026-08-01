# Reasoning Summaries

If you've used a coding agent, you've probably seen it post small progress updates while it works. Those messages make the work
easier to follow: they tell you what the agent is checking, what it has found, and what it plans to do next.

In plap, those progress messages are reasoning summaries. They can arrive as they are written, while the main response remains
available to plugins for review or revision.

## Why summaries are separate

plap lets plugins work with main-model output before the user sees it. A plugin can review the output, revise it, or keep it
private.

Once that text reaches the client, a later review cannot change what the user has already seen. Main output therefore stays in
`main`, with publication deferred to [`commit()`](state.md#persist-state-changes).

Reasoning summaries use a separate stream for progress updates. They can arrive while plugins continue working with the answer
in `main`.

## Raw reasoning text

Some models already emit reasoning that reads well as a progress update. Forwarding it unchanged preserves the model's own
wording. It also requires no additional model call.

This is the default behavior of the [`response.summary` hook](hooks.md#response-hooks). As the model emits reasoning, the core
handler passes the text through to the Responses client.

Other models produce reasoning that is repetitive, tied to provider conventions, or too detailed for a public update. An
application may still want to show progress, but in language written for the user rather than copied from the model's working
text.

## Built-in `summary` plugin

The built-in `summary` plugin is an example [`response.summary` listener](hooks.md#response-hooks) for that case. When the
client requests a summary mode, the plugin sends the reasoning to a separate summary model.

A public update needs enough context to be coherent. The plugin collects the incoming reasoning into fragments, then asks the
summary model to write one update for each fragment.

Each request includes the summary already shown to the user. The summary model writes only the next addition, without
repeating or rewriting earlier parts.

The prompt asks for an OpenAI-style reasoning summary: a first-person account of high-level checks, comparisons, revisions,
and conclusions, with raw private reasoning left out. The summary model's output streams to the client as it is generated.

## Summary modes

With the built-in plugin loaded, `reasoning.summary` controls how much detail appears in its OpenAI-style progress updates:

```json
{
  "reasoning": {
    "effort": "high",
    "summary": "concise"
  },
  "stream": true
}
```

| Mode | Intended result |
| --- | --- |
| `auto` | Let the summarizer choose an appropriate level of detail |
| `concise` | Short progress updates |
| `detailed` | Fuller explanations of the work in progress |

You can choose the mode for each Responses request. If you omit `reasoning.summary`, the plugin leaves the raw passthrough in
place. The deprecated `generate_summary` field selects the same modes.

Both approaches depend on the main model emitting reasoning text. If the model emits none, there is nothing to pass through or
summarize.

With `stream: true`, the client receives each summary update as it is generated. A non-streaming request returns the collected
updates on the completed reasoning item.

## Budget behavior

Raw passthrough adds no model call. The built-in plugin makes a separate call for each generated update, using the same
[completion budget](llms/budgeting.md) as the main model.

If the budget runs out during a summary request, that request stops. Text already emitted remains in the response, and the
main completion continues.

## Custom summary handlers

An application may need a different public voice or stricter rules about which details appear in progress updates. It can
provide another `response.summary` listener to translate the reasoning, remove application-specific details, or generate
updates in its own domain language.
