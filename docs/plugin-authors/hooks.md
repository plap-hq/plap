# Runtime Hooks

This page explains what each plugin hook does and when it runs.

## Startup Hooks

### `config.collect`

Purpose:

- Add CUE files to the composed runtime config.

When it triggers:

- Startup: when plap is gathering all config files before loading config.
- During app startup, after plugin modules are imported.
- Before CUE config is loaded.
- It runs once per app creation.

What it receives:

- `paths`: the tuple of config/schema file paths collected so far

What it controls:

- which CUE files are loaded
- plugin config fields
- side registration through CUE side-code maps

Built-ins using it:

- `core`, `vision`, `summary`, `advisor`

### `routes.collect`

Purpose:

- Add Litestar route handlers to the app.

When it triggers:

- Startup: when plap is assembling the app's routes.
- During app startup, after config has been loaded.
- Before the `Litestar` app object is constructed.
- It runs once per app creation.

What it receives:

- `routes`: the route handlers collected so far
- `loaded`: the fully loaded `CueBox`

What it controls:

- HTTP routes
- websocket routes
- any plugin-owned API surface

Built-ins using it:

- `core`

### `svcs.collect`

Purpose:

- Register shared services in the application service registry.

When it triggers:

- Startup: when plap is building the shared service container.
- During app startup, after config has been loaded.
- Before request handling begins.
- It runs once per app creation against the shared `svcs.Registry`.

What it receives:

- `registry`: the shared `svcs.Registry`
- `loaded`: the fully loaded `CueBox`

What it controls:

- process-wide clients
- database-backed helpers
- shared caches or infrastructure objects

Built-ins using it:

- `core` registers the chat completion client

### `shutdown.collect`

Purpose:

- Add shutdown hooks that run when the app stops.

When it triggers:

- Startup registration: when plap is collecting hooks that should run later during shutdown.
- During app startup, after config has been loaded.
- The hook list it returns is installed onto the app shutdown sequence.
- The collection phase runs once per app creation; the returned hooks later run on app shutdown.

What it receives:

- `hooks`: the shutdown hooks collected so far
- `loaded`: the fully loaded `CueBox`

What it controls:

- plugin cleanup work at process shutdown

Built-ins using it:

- no built-in plugin currently uses it

## Response Lifecycle Hooks

### `response.start`

Purpose:

- Wrap the entire response execution.

When it triggers:

- Per response: right when plap starts executing a response.
- In both streaming and non-streaming create paths.
- After request preparation and ingest have built the mutable `State`.
- Before any `response.loop` iteration begins.

What it receives:

- `state`: the mutable `State` for the response

What it controls:

- end-to-end response orchestration
- behavior before any loop iteration begins
- behavior after the full response runtime returns

Built-ins using it:

- `core` starts the main runtime here

### `response.config`

Purpose:

- Resolve or rewrite the per-request config object used by the runtime.

When it triggers:

- Per response: near the start, when plap is deciding the runtime config for this request.
- Once per response execution inside `core.run_response(...)`.
- Before the usage ledger is created and before the main loop starts.

What it receives:

- `state`: the current response `State`
- `request`: a small config-resolution request dict built from the incoming response request

What it controls:

- the resolved `CueBox` config used for this response

Built-ins using it:

- `core` provides the terminal resolver

### `response.request`

Purpose:

- Build or rewrite the main `ChatCompletionRequest` before the main model runs.

When it triggers:

- Per main-model pass: right before plap calls the main model.
- Once per `response.loop` iteration that is about to call the main model.
- After `response.config` has resolved the per-request config.
- Before `response.stream` starts the main model stream.

What it receives:

- `state`: the current response `State`
- `config`: the resolved config for this response

What it controls:

- model choice
- main messages
- tools and tool choice
- sampling parameters
- response format

Built-ins using it:

- `core` builds the terminal request
- `vision` rewrites the request to replace inline images and inject the internal vision tool

### `response.validate`

Purpose:

- Add validators that can reject a model result and force a retry.

When it triggers:

- Per main-model pass: right before plap starts validating and retrying the main model stream.
- Once per `response.loop` iteration, just before `retry_stream(...)` is constructed.
- The validators it returns are then used across retry attempts within that iteration.

What it receives:

- `state`: the current response `State`
- `config`: the resolved config for this response
- `validators`: the retry validators collected so far

What it controls:

- which main-model outputs are accepted
- which outputs cause retry-stream to ask the model again

Built-ins using it:

- `core` installs the baseline validators
- `vision` adds validation for vision tool arguments and referenced image ids

### `response.summary`

Purpose:

- Consume and optionally rewrite the streamed reasoning-summary channel.

When it triggers:

- During streaming: while the current main-model pass is producing reasoning summary output.
- During a main-model streaming pass.
- It starts after `response.stream` creates the summary stream plumbing.
- It runs concurrently with the main model stream for that iteration.

What it receives:

- `state`: the current response `State`
- `config`: the resolved config for this response
- `source`: an async stream of `SummaryDelta` and `SummaryDone`

What it controls:

- what reasoning summary text is emitted to the client while the main model streams

Built-ins using it:

- `core` provides the terminal summary emitter
- `summary` wraps the stream and replaces raw reasoning deltas with summarized text

### `response.loop`

Purpose:

- Wrap one iteration of the main response runtime.

When it triggers:

- Per main-model pass: around one full main-runtime iteration.
- Once for each main-runtime iteration.
- It can run more than once per response if the previous pass leaves `main` active, with no open calls, ending in a non-assistant message.
- It is the per-iteration wrapper around the terminal main-model call.

What it receives:

- `state`: the current response `State`
- `config`: the resolved config for this response
- `ledger`: the shared `UsageLedger`

What it controls:

- what happens immediately before the main model call for that iteration
- what happens immediately after an accepted result is produced
- whether plugin work appends messages to `main` or plugin-owned sides before the next loop decision

Built-ins using it:

- `core` runs the terminal main-model iteration
- `vision` executes internal vision tool calls after the main result
- `advisor` runs advisor phases around the main iteration

### `response.commit`

Purpose:

- Run logic immediately before the runtime commits the final response state.

When it triggers:

- End of response: after plap has finished looping and is about to publish the final state.
- Once per response execution, after the loop has ended.
- After the final `response.loop` result has been chosen.
- Before `state.commit()` publishes persisted state and visible output.

What it receives:

- `state`: the current response `State`
- `config`: the resolved config for this response
- `result`: the final `StreamResult | None`

What it controls:

- last-moment state mutation before `state.commit()` publishes persisted updates and visible output

Built-ins using it:

- `core` provides the terminal commit behavior

### `response.finish`

Purpose:

- Decide how the coordinator finishes the response after commit.

When it triggers:

- End of response: immediately after commit, when plap decides the final response status.
- Once per response execution, immediately after `response.commit`.
- After state has been committed.
- Before the coordinator marks the response completed, incomplete, cancelled, or failed.

What it receives:

- `state`: the current response `State`
- `config`: the resolved config for this response
- `result`: the final `StreamResult | None`
- `ledger`: the shared `UsageLedger`

What it controls:

- whether the response is marked completed
- whether it is marked incomplete
- whether a plugin-raised error aborts the response
- which usage totals are exposed to the coordinator

Built-ins using it:

- `core` provides the terminal finish behavior

## Read Next

- [State, Sides, and Persistence](state.md)
- [Getting Started](getting-started.md)
