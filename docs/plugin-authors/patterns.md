# Built-in Plugin Patterns

Use the built-ins as templates.

## Recommended Reading Order

1. `src/plap/plugins/summary/__init__.py`
2. `src/plap/plugins/vision/__init__.py`
3. `src/plap/plugins/advisor/__init__.py`
4. `src/plap/plugins/core/loop.py`

Why this order:

- `summary` is the smallest wrapper
- `vision` shows request rewriting plus internal tool execution
- `advisor` shows side ownership plus `durable`
- `core` shows the terminal runtime everything else wraps

## `summary`

Files:

- `src/plap/plugins/summary/__init__.py`
- `src/plap/plugins/summary/summarizer.py`

Pattern:

- wraps `response.summary`
- consumes a summary stream and produces a rewritten summary stream
- does not need a side

Use it as the model for:

- stream transformation plugins
- plugins that do not need main-state mutation

## `vision`

Files:

- `src/plap/plugins/vision/__init__.py`

Pattern:

- rewrites the main request in `response.request`
- adds a retry validator in `response.validate`
- on `response.loop` unwind, executes internal tool calls and appends tool outputs to `main`
- charges hidden model usage to the shared `UsageLedger`

Use it as the model for:

- internal tool plugins
- request-rewriting plugins
- plugins that validate model-emitted tool arguments

## `advisor`

Files:

- `src/plap/plugins/advisor/__init__.py`

Pattern:

- registers its own side through CUE
- rebuilds that side from `main` history
- runs a separate model on that side
- appends blocking or non-blocking guidance back into `main`
- uses durable state for cross-phase notes

Use it as the model for:

- multi-phase plugins
- private side-conversation plugins
- plugins that need persisted hidden notes between phases

## `core`

Files:

- `src/plap/plugins/core/__init__.py`
- `src/plap/plugins/core/loop.py`
- `src/plap/plugins/core/request.py`
- `src/plap/plugins/core/ledger.py`

Pattern:

- provides the terminal runtime hooks
- builds the main request
- owns the loop and finish conditions
- commits state and publishes visible output
- performs normalized usage accounting

Read this when you need to understand exactly what every outer plugin is wrapping.

## Read Next

- [Plugin Author Overview](index.md)
- [Runtime Hooks](hooks.md)
- [State, Sides, and Persistence](state.md)
