# Plugin Author Overview

`plap` plugins are Python modules that extend the responses runtime.

They are used to add features to LLMs without retraining them. A plugin can:

- rewrite the main model request
- validate model outputs and force retries
- run hidden internal tool or model work
- maintain private side conversations
- append visible results back into `main`
- register shared services or routes

## What A Plugin Is

A plugin is:

1. A Python module imported for side effects.
2. One or more `@bus.listen(...)` handlers.
3. Usually a `schema.cue` file contributed through `config.collect`.

There is no plugin base class.

## Where Plugins Live

- Built-in plugins live under `src/plap/plugins/`.
- Plugin discovery uses Python entry points in the `plap.plugin` group in `pyproject.toml`.
- Enabled plugins are listed in `plugins.toml`.

Current built-ins:

- `core`: the terminal main-model runtime
- `vision`: injects an internal vision tool and executes it
- `summary`: rewrites streamed reasoning summaries
- `advisor`: runs a separate watchdog model on its own side

## Loading And Order

Plugin loading starts in `src/plap/app.py`.

- `plugins.toml` provides the plugin list.
- Each name must exist as a `plap.plugin` entry point.
- Each module is imported in manifest order.

Dispatch uses the onion bus in `src/plap/bus.py`.

- later-registered listeners run first
- the terminal `@bus.emit(...)` body runs last
- a listener continues the stack by calling `await next(...)`
- after `next()` returns, the listener is back on the unwind path and can inspect or mutate the result

With:

```toml
plugins = ["core", "vision", "summary", "advisor"]
```

the `response.loop` entry path is effectively:

1. `advisor`
2. `vision`
3. `core`

Then it unwinds back out through `vision`, then `advisor`.

`core` must be present because it provides the terminal response runtime.

## Core Runtime Shape

The terminal runtime is in `src/plap/plugins/core/loop.py`.

At a high level:

1. Resolve config for the current request.
2. Build a mutable `State` from ingested response history.
3. Build the main `ChatCompletionRequest` from `state.sides["main"]`.
4. Stream the main model.
5. Let plugins react through `response.loop` unwind logic.
6. Commit state and publish visible output.
7. If `main` still needs another pass, loop again.

The response loop can run more than once. Plugins should assume each pass sees the current mutable state, not a frozen snapshot from the first pass.

## Read Next

- [Runtime Hooks](hooks.md)
- [State, Sides, and Persistence](state.md)
- [Getting Started](getting-started.md)
