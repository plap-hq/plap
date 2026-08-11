# Threads

Your first plugin returned a tool result to `main`. That result belongs there: `main` is the history used for the next
main-model request, so the model should see it on its next turn.

Not every plugin call should share that history. A plugin may need private instructions, several turns with another model, or
intermediate tool results that do not belong in `main`'s context. A reviewer is one example:

```text
main
├── user request
├── assistant draft
└── revision after review

reviewer
├── review instruction
├── assistant tool call
└── tool result
```

A thread is only a message history. The plugin still decides when to call a model and what to do with its answer.

## Register the reviewer thread

Register the name before plap saves any state that refers to it. Add a code to the plugin's `schema.cue`:

```cue
package plap

#RegisteredThreads: {
  reviewer: 2048
}
```

Load that schema when the plugin starts:

```python
from plap.plugins.easy import bootstrap

bootstrap.config(__file__)
```

The code identifies the declaring thread in public function-call IDs.

## Build the reviewer history

`setdefault` creates the history on first use and returns the existing history thereafter:

```python
from plap.llms.completions import ChatMessage

reviewer = state.threads.setdefault("reviewer")
reviewer.append(
    ChatMessage(
        role="user",
        content="Review the proposed answer.",
    )
)
```

These messages become context for the review model without changing the next request built from `main`.

## Keep the draft private

The separate history protects the main model's context, but core could still publish the draft. Before review starts, enable
the reviewer thread and block `main`:

```python
state.threads.enable("reviewer")
state.threads.block("main", by="reviewer")

assert state.threads.active == {"reviewer"}
```

The reviewer is now the only active thread. `main` cannot take another model turn, and its draft and function calls remain
private. The plugin calls the review model itself; enabling a thread does not call one.

## Continue after a client-owned tool

The review involves requests moving in two directions:

| Term | Direction |
| --- | --- |
| Model request | plap calls an LLM provider |
| Responses request | An API client calls plap |

plap can make several model requests while handling one Responses request. A server tool stays inside that request: plap runs
the tool and calls the model again.

A client-owned tool must cross the other boundary. If the reviewer requests one, `commit()` returns the function call and ends
the current Responses request while the main draft remains private. The API client runs the tool and sends another Responses
request containing the output. That request is the **Responses continuation**.

The continuation restores the thread histories and activity state, routes the output back to `reviewer`, and lets the plugin
continue the same review. `state.open_calls()` reports calls that still need results:

```python
pending = state.open_calls("reviewer")
```

A call is declared when a model produces it and open after plap publishes it to the API client. The continuation closes it by
supplying the output. An inactive declaration can remain private, but a published call must receive its output before the
continuation advances beyond it.

## Finish without waking `main` too early

Suppose a policy plugin also blocks `main` before review finishes:

```python
state.threads.block("main", by="policy")
```

The reviewer removes only the state it created:

```python
state.threads.disable("reviewer")
state.threads.unblock("main", by="reviewer")

assert "main" not in state.threads.active
```

`main` stays blocked by `policy`. The reviewer does not need to know that policy was involved, and cannot accidentally undo
its decision. When policy is satisfied, it releases its own block:

```python
state.threads.unblock("main", by="policy")

assert "main" in state.threads.active
```

That is why each block has a `by` value: one plugin can finish without reactivating a thread another plugin still needs paused.

If the final block is released before core reaches `response.loop`, the main model can take another turn. A release just before
`response.commit` can publish the parked draft without another model call. Output already published or compacted is not sent
twice.

## Let a new user message take priority

A review can survive across Responses continuations, so its phase and block may still be present when the user sends a new
instruction. If the plugin simply resumes that phase, `main` remains paused and the reviewer keeps working on the old request.
To the user, the assistant appears to have ignored the change in direction.

`response.user_turn` runs before the response loop continues. The reviewer uses that boundary to abandon the old phase,
disable its thread, and release its block. `main` can then handle the new instruction unless another plugin still has a reason
to keep it paused.

Core cannot release every block on the reviewer's behalf. Another plugin may still have valid work, and only that plugin knows
whether the new message makes its work obsolete. Each plugin therefore removes its own block in `response.user_turn`.

## Thread code ranges

Codes must be unique. Core reserves the first range for well-known threads:

| Range | Use |
| --- | --- |
| `0..1023` | Core threads |
| `1024..49151` | Registered plugin threads |
| `49152..65535` | Private or experimental threads |

Core uses code `0` for `main`. The built-in `advisor` plugin reviews main-model work and uses code `1024` for its private
history.
