# Threads

Your first plugin returned a tool result to `main`. That result belongs there: core uses `main` to build the next main-model
request, so the model sees the result on its next turn.

Not every model call belongs in that history. A plugin may need private instructions, several turns with another model, or
intermediate tool results that the main model should never see. A reviewer is one example:

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

A thread is a message history, not a model or a task. The plugin still chooses when to call a model, builds the request from
the appropriate history, and appends the model's answer.

## Register the reviewer thread

Before the plugin can save a reviewer history, register its name in the plugin's `schema.cue`:

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

The code lets a later tool result find its way back to the history that requested it. It becomes part of public function-call
IDs, so each registered thread needs a unique value.

## Build the reviewer history

The reviewer needs a history before its first model call. `setdefault` creates one on first use and returns the existing
history when a continuation restores earlier review messages:

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

The review model can now receive this instruction without adding it to the next main-model request.

## Keep the draft private

The separate history keeps the review conversation out of the main model's context. It does not stop core from returning the
draft already waiting in `main`. Before the review begins, enable `reviewer` and block `main`:

```python
state.threads.enable("reviewer")
state.threads.block("main", by="reviewer")

assert state.threads.active == {"reviewer"}
```

The reviewer is now the only active thread. The draft stays private, and the main model cannot take another turn while the
review is in progress. Enabling `reviewer` does not call a model; the plugin still makes that request itself.

## Continue after a client-owned tool

The reviewer may ask for a tool that belongs to the API client. At that point, two request directions matter:

| Term | Direction |
| --- | --- |
| Model request | plap calls an LLM provider |
| Responses request | An API client calls plap |

plap can make several model requests while handling one Responses request. A server tool stays inside that request because
plap can run the tool and call the model again immediately.

A client-owned tool has to cross the API boundary. If the reviewer requests one, `commit()` returns the function call and
ends the current Responses request while the main draft remains private. The API client runs the tool and sends its result in
a new Responses request. That request is the **Responses continuation**.

The continuation restores both histories. The function-call ID identifies `reviewer` as the history that asked for the tool,
so plap appends the result there as a tool message. The review history now ends like this:

```text
reviewer
├── assistant tool call
└── tool result
```

The plugin can send that updated history to the review model and continue the same review. Nothing from this exchange is
added to `main`.

Before continuing, the plugin can check whether the reviewer is still waiting for a client result:

```python
pending = state.open_calls("reviewer")
```

`state.open_calls("reviewer")` contains tool calls in the reviewer's latest answer that do not yet have matching tool
messages. The review cannot move past such a call until its result arrives.

## Finish without waking `main` too early

The reviewer is not necessarily the only plugin asking `main` to wait. Suppose a policy plugin also blocks it:

```python
state.threads.block("main", by="policy")
```

When the review finishes, the reviewer undoes the two changes it made:

```python
state.threads.disable("reviewer")
state.threads.unblock("main", by="reviewer")

assert "main" not in state.threads.active
```

`main` stays paused because the policy plugin's block is still in effect. The reviewer does not need to know why policy is
waiting, and finishing the review cannot cancel that decision. When the policy plugin finishes, it releases its own block:

```python
state.threads.unblock("main", by="policy")

assert "main" in state.threads.active
```

This is why `block()` and `unblock()` require `by`: a plugin can release the block it added without reactivating a thread that
another plugin still needs paused.

What happens after the final block is released depends on where the response is. Before `response.loop`, the main model can
take another turn. After the loop but before `response.commit`, commit can return the waiting draft without another model
call. Output already returned or compacted is not returned twice.

## Let a new user message take priority

A review can span several Responses requests. If the user sends a new instruction before that review finishes, continuing the
old review would keep `main` paused and make the assistant appear to ignore the new direction.

`response.user_turn` runs before the response loop handles the new message. The reviewer can use it to abandon the unfinished
review, disable `reviewer`, and release its block on `main`. The main model can then handle the new instruction unless another
plugin still needs it paused.

Core cannot release every block on the reviewer's behalf. Another plugin may still have useful work in progress, and only
that plugin can decide whether the new message makes its work obsolete. Each plugin therefore releases its own block in
`response.user_turn`.

## Thread code ranges

Thread codes must be unique. Core reserves the first range for well-known threads:

| Range | Use |
| --- | --- |
| `0..1023` | Core threads |
| `1024..49151` | Registered plugin threads |
| `49152..65535` | Private or experimental threads |

Core uses code `0` for `main`. The built-in `advisor` plugin reviews main-model work and uses code `1024` for its private
history.
