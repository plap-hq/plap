# Example Plugins

`vision`, `summary`, and `advisor` are loaded by default. To disable all three, keep only `core` and `easy` in
[`plugins.toml`](../plugins.toml), then restart plap:

```toml
plugins = ["core", "easy"]
```

## Vision: Q&A over images for text-only models

With this pattern, plap reached GPT-5-level performance on MMMU Pro.

Most vision proxies ask a vision model to describe an image once, then pass that description to the main model. This works
when the first description preserves everything the task needs. When it misses something, the main model has no way to ask
another question.

The `vision` plugin keeps the image available through a tool. The main model can ask about a specific detail, read the
answer, and follow up. Earlier questions and answers stay in the vision model's context, allowing the image to be worked
through one question at a time.

## Advisor: OMP-style review on the server

A basic proxy advisor sends the conversation to a second model and gets one comment back. That reviewer cannot ask the
client to run a tool, inspect the result, and continue investigating. plap's advisor can.

The advisor uses plap's [thread system](threads.md) to keep its own conversation separate from the main model. This split
context keeps its investigation out of the main conversation. It can call tools supplied by the client, receive their
results in later Responses requests, and continue where it left off. These client turns can repeat until the review is
complete.

This gives plap the same kind of advisor as [OMP's advisor](https://omp.sh/), but behind the Responses API. It can stop a
tool call before it runs or send advice back to the main model before the user sees it.

## Summary: summarized model progress

The `summary` plugin compresses reasoning into short progress updates while the model works. A separate model writes updates
for the user, giving any reasoning model the kind of live status feed people expect from a coding agent without exposing
raw chain-of-thought.

The summary can stream while any concrete actions remain private for review by `advisor` or any other plugin.

The [reasoning summaries guide](summary.md) covers configuration and custom summary handlers.
