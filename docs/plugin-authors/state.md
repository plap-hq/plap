# State, Sides, and Persistence

This page explains the main mutable state surfaces that plugin authors work with.

## `main`

`main` is the public spine of the conversation.

- `state.sides["main"]` is the mutable main transcript for the current response suffix.
- Visible assistant output is derived from `main`.
- If your plugin wants user-visible effect, the usual pattern is to append results back into `main`.

Design rule:

- Treat `main` as a user-facing append-only tape.

Why:

- Anything sent to the user through `main` can be persisted by the user's local agent.
- plap does not control those downstream copies.

Current code constraint:

- the persisted main prefix is immutable
- only the current response suffix may be replaced before commit

## Sides

Sides are named conversation branches stored in `State.sides`.

- `main` is the spine
- other sides are offshoots or facets of that spine
- side membership is validated against `config.sides`
- the side namespace is backed by 16-bit side codes, so there can be up to `65536` configured sides total

Current code reserves these code ranges:

- `0..1023`: well-known core sides
- `1024..49151`: registered plugin sides
- `49152..65535`: private or experimental sides

Unknown sides are rejected during staging and commit.

## Registering A Side

Plugins register side codes through CUE, not Python.

Example:

```cue
package plap

#RegisteredSides: {
  my_side: 1025
}

#Config: {
  my_plugin?: #FieldConfig
}
```

Then include that schema file through `config.collect`.

## What Sides Are For

Use a side when your plugin needs a separate private conversation branch.

Typical uses:

- a watchdog or reviewer transcript
- a planning branch
- plugin-owned hidden assistant or tool history outside `main`

Important limitation:

- the core runtime only knows how to stream the main model
- if your plugin wants side execution, it must implement that behavior itself

## `state.sides.active`

`state.sides.active` is the set of sides that are logically active.

Plugins can read or mutate it.

`main` publication depends on active membership, and active sides can affect which function-call items are emitted at commit time.

## Open Tool Calls

Use `state.open_calls(side)` to inspect unresolved tool calls for a side.

This matters because:

- the main runtime skips a main-model pass if `main` already has open tool calls
- commit and publication logic validate tool-call closure rules per side

## `durable`

`state.durable` is plugin-owned persisted JSON state.

- use it for data that must survive later client requests
- it is separate from local Python variables
- it must remain JSON-compatible

Persistence model:

- plap emits encrypted reasoning and compaction payloads
- later requests can replay those payloads back into state
- `durable` survives through that encrypted replay path

At the API layer, that persistence travels through encrypted payload fields such as `reasoning.encrypted_content`.

## Local Request State

Ordinary local Python variables are still useful.

Use them for:

- temporary parsed values
- current-iteration scratch state
- request-only helpers

They disappear when the request ends.

## Shared Services

Services registered through `svcs.collect` are different from `durable`.

- `durable` is per-conversation replayable state
- services are shared application infrastructure

Services are good for:

- database helpers
- provider clients
- caches
- shared resources reused across requests

They are not per-user hidden memory unless your plugin builds that scoping itself.

## Public Versus Hidden Work

Built-in plugins show different patterns:

- `advisor` keeps an internal `advisor` side and writes advisory results back into `main`
- `vision` does not create a persistent side; it executes an internal tool call and appends the result to `main`
- `summary` uses no side and only transforms summary streaming

That is the intended model: a plugin may register many sides, one side, or none.

## Read Next

- [Getting Started](getting-started.md)
- [Built-in Plugin Patterns](patterns.md)
