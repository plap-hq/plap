# Getting Started

If you have never written a plap plugin before, start with a plugin that does one small thing to `main` and does not create its own side.

## Understanding The Runtime Model

Keep these rules in your head before you write code:

- `main` is the user-facing tape
- non-main sides are private plugin branches
- `durable` is per-conversation persisted plugin state
- local variables disappear after the request finishes
- shared services are global infrastructure, not per-user hidden memory
- the core plugin is the built-in runtime that directly streams the main model

## Adding A New Plugin To The App

The easiest first plugin shape is:

1. Add a module.
2. Add `schema.cue`.
3. Register the schema with `config.collect`.
4. Wrap `response.loop` or `response.request`.
5. Enable the plugin in `pyproject.toml` and `plugins.toml`.

## Creating A Small First Plugin

Build a plugin that appends one developer message to `main` when a certain condition is met.

Why this is a good starter:

- no custom side required
- no extra model call required
- no service registration required
- it teaches the bus, state mutation, and loop reruns

## Creating The Files

Model a new plugin after the existing directories:

```text
src/plap/plugins/my_plugin/
  __init__.py
  schema.cue
```

## Writing The Python Module

Start with this minimal module:

```py
from __future__ import annotations

from pathlib import Path

from plap.bus import bus
from plap.config import CueBox
from plap.plugins.core.ledger import UsageLedger
from plap.plugins.core.loop import StreamResult
from plap.responses.state import State


@bus.listen("config.collect")
async def collect(paths: tuple[str, ...], *, next):
    here = Path(__file__).resolve()
    return await next(paths=(*paths, str(here.parent / "schema.cue")))


@bus.listen("response.loop")
async def run_my_plugin(state: State, config: CueBox, ledger: UsageLedger, *, next) -> StreamResult | None:
    result = await next(state=state, config=config, ledger=ledger)
    if result is None:
        return None

    _ = config, ledger

    # Inspect state.sides["main"] or result.accepted here.
    return result
```

## Adding Plugin Configuration

Minimal `schema.cue` example:

```cue
package plap

#Config: {
  my_plugin?: {
    enabled: *true | bool
  }
}
```

If your plugin does not need config yet, a tiny schema file is still useful so the plugin structure is ready to grow.

## Enabling The Plugin

Add the entry point in `pyproject.toml`:

```toml
[project.entry-points."plap.plugin"]
my_plugin = "plap.plugins.my_plugin"
```

Then enable it in `plugins.toml`:

```toml
plugins = ["core", "vision", "summary", "advisor", "my_plugin"]
```

Ordering matters:

- later plugins wrap earlier ones
- placing `my_plugin` later makes it more outermost

## Changing The Main Request Before The Model Runs

Use `response.request` when your intent is to change what the main model sees.

Typical uses:

- add or remove tools
- rewrite messages
- inject developer instructions
- change sampling parameters

## Reacting To What The Main Model Just Did

Use `response.loop` when your intent is to inspect an accepted main-model result and react to it.

Typical uses:

- inspect tool calls
- execute internal tool work
- append tool outputs to `main`
- run a side-model pass after the main result
- add follow-up guidance into `main`

## Storing Variables For Later Requests

Use `state.durable` when your intent is to keep plugin-owned state across later client requests for the same conversation lineage.

Good examples:

- a private note for a later plugin phase
- a flag saying the plugin already performed a hidden check
- a compact plugin-owned record that should survive continuation

Example:

```py
state.durable["my_plugin"] = {"checked": True, "note": "Need follow-up if tool fails."}
```

## Keeping Temporary Variables Only For This Request

Use ordinary local Python variables when your intent is to keep state only during the current request.

Good examples:

- a temporary parsed result
- a list of calls found in the current iteration
- a value used only while building a request or response

## Creating Offshoot Conversations Using Sides

Use sides when your intent is to maintain a separate plugin-owned conversation branch.

Good examples:

- a watchdog or reviewer conversation
- a planning side with its own transcript
- a plugin that needs hidden assistant or tool history outside `main`

How to start:

1. Register the side in `schema.cue`.
2. Create or update `state.sides["your_side"]`.
3. Add or remove active membership in `state.sides.active` if needed.
4. Run your own model or tool logic for that side.
5. Append any final visible effect back into `state.sides["main"]`.

## Publishing Something Visible To The User

Use `state.sides["main"]` when your intent is to add visible conversation state.

Example:

```py
from plap.llms.completions.chat import ChatMessage

state.sides["main"].append(ChatMessage(role="developer", content="Double-check the selected file before answering."))
```

## Sharing Infrastructure Across Requests

Use `svcs.collect` when your intent is to provide shared infrastructure.

Typical uses:

- a database helper
- a provider client
- a cache client
- a shared resource created once and reused

This is for infrastructure, not conversation memory.

## Rejecting Bad Model Outputs And Forcing Retries

Use `response.validate` when your intent is to inspect the main model result and tell the retry layer that it is invalid.

Typical uses:

- a tool call is missing required arguments
- a tool references data that does not exist
- a model output violates plugin-specific structural rules

## Rewriting Reasoning Summary Output

Use `response.summary` when your intent is to transform reasoning-summary text before the client sees it.

## Charging Hidden Model Or Tool Work

If your plugin makes an internal model call, charge it through the shared `UsageLedger`.

- hidden internal work: call `ledger.hide(...)`
- visible accepted final output: the core runtime later calls `ledger.show(...)`

## Avoiding Common Mistakes

- creating a side before registering its code in CUE
- treating a shared service like per-user memory
- writing temporary scratch material into `main`
- forgetting that `response.loop` can run more than once
- forgetting to account for hidden model usage
- assuming a plugin needs its own side when `main` mutation is enough

## Debugging Checklist

1. Is the plugin listed in `pyproject.toml` entry points?
2. Is the plugin enabled in `plugins.toml`?
3. Is the plugin order correct for the wrapping behavior you want?
4. Did `config.collect` include the right `schema.cue` path?
5. If using a side, is its code registered in CUE?
6. If mutating `main`, are you only changing the current response suffix?
7. If doing hidden model work, are you charging the ledger?
8. If state must survive later requests, did you put it in `state.durable` instead of a local variable?

## Read Next

- [Runtime Hooks](hooks.md)
- [State, Sides, and Persistence](state.md)
- [Built-in Plugin Patterns](patterns.md)
