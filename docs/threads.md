# Threads

The first plugin returned a tool message to `main`. `main` is the history used for the next main-model request, so the model
sees that tool result on its next turn.

The same history is not appropriate for every model call. A reviewer may need private instructions and several turns of its
own. Putting those messages in `main` would change the main model's context. A separate thread keeps each history independent:

```text
main
├── user request
├── assistant tool call
└── tool result

reviewer
├── review instruction
└── reviewer reply
```

A thread is a message history, not a task. The plugin still chooses a model, builds its request, and appends the resulting
messages.

## Which request crosses the boundary?

There are two request directions in this flow:

| Term | Meaning |
| --- | --- |
| Model request | plap calls an LLM provider |
| Responses request | An API client calls plap |

One Responses request can contain several model requests. The `server_time` example stays inside one Responses request: the
model calls the server tool, plap executes it, and plap calls the model again with the result.

A client-owned tool crosses the other boundary. plap returns a function-call item to the API client and finishes the current
Responses request. After executing the tool, the API client sends a new Responses request containing the function-call
output. This page calls that second API call a **Responses continuation**.

plap saves thread histories, thread activation, and `threads.blocking` with the first response. The Responses continuation restores all of it. A
plugin can therefore leave `main` parked while a reviewer takes one or more client-tool turns across several Responses
requests.

## Add a separate history

`setdefault` creates a thread the first time it is used and returns a thread wrapper in later Responses continuations:

```python
from plap.llms.completions import ChatMessage

reviewer = state.threads.setdefault("reviewer")
reviewer.messages.append(
    ChatMessage(
        role="user",
        content="Review the proposed answer.",
    )
)
```

The new message exists only in `reviewer`. A review completion can use that history without changing the next main-model
request. Custom thread names must be [registered](#register-a-thread-name) before response state is saved.

Each thread wrapper exposes three routed attributes:

- `thread.messages`: that thread's message history
- `thread.active`: whether that thread is active
- `thread.blocking`: whether that thread is currently blocking `main`

## Choose active threads

Thread histories control model context. Each thread wrapper controls whether that thread can exchange client-tool calls in
the current response.

An active non-main thread can publish a pending function call. When the API client sends the matching function-call output
in a Responses continuation, plap appends the result to that thread. The thread's messages stay private; plap does not return
non-main messages as response messages. Activating a thread does not call a model.

`main` starts active and has two additional rules:

- The main-model loop runs only while `main` is active and has no open client call.
- The latest main assistant message can be published only while `main` is active.

Setting any thread's `blocking` flag parks `main` while that blocker is present. A blocking thread often activates itself at
the same time:

```python
reviewer = state.threads.setdefault("reviewer")
reviewer.block()
reviewer.active = True
```

If the current response returns a reviewer function call, leave the thread in this state. Do not unblock `main` before the
response ends. The reviewer must still be active when the API client's Responses continuation supplies that call's output.

## What does a cross-request review look like?

Suppose the main model has produced an answer that must be reviewed before the API client sees it.

During the first Responses request:

1. A response hook blocks `main` from the reviewer thread.
2. The plugin builds a review request from the reviewer thread and calls the review model.
3. If the reviewer asks the API client to run a tool, the plugin leaves `reviewer` active and `main` inactive.
4. `commit()` saves both histories and the active set, then returns the reviewer function call. The main answer remains private.

During the API client's Responses continuation:

1. plap reads the call ID and appends the tool result to the reviewer thread.
2. The plugin continues the review from that history.
3. Another reviewer function call ends the response again with the same reviewer state.
4. Once the review is complete, the plugin can reactivate `main`:

```python
reviewer = state.threads["reviewer"]
reviewer.active = False
reviewer.unblock()
```

If the reviewer needs no client-owned tool, the entire review can finish during the first Responses request and `main` can
be unblocked before that request commits. The important condition is that review has finished, not that blocking and
unblocking happen as an immediate pair.

Reactivating `main` changes state; it does not itself call the main model. The hook that performs reactivation determines
what follows:

- Before the core `response.loop`, the main loop may take another model turn.
- After the loop but before `response.commit`, commit can publish an eligible parked main message or call without another
  main-model completion.

Already-public or compacted main output is not published again.

## Track client-tool calls

A call is **declared** when it appears in a private assistant message returned by an LLM provider. It becomes **open** after
plap publishes a function-call item to the API client. A Responses continuation from that client closes the call by
supplying its function-call output.

`state.open_calls(name)` returns calls in the final assistant message that do not yet have tool results:

```python
pending = state.open_calls("reviewer")
```

The public call ID records the declaring thread. plap uses that ID to route the output from a Responses continuation back to
the same thread. A public function call cannot claim a declaration from an inactive thread.

An inactive declared call may remain parked. An active declared call must be published. Once a call is open, the API client
must supply its output before the continuation can advance that thread past the call.

Each thread must remain a valid assistant/tool history. A tool message must match a pending call ID, and an unrelated
message cannot be inserted between an open call and its tool result.

## Register a thread name

Core registers `main`, which always exists. A plugin registers each additional thread in `schema.cue`:

```cue
package plap

#RegisteredThreads: {
  reviewer: 2048
}
```

Load that schema from the plugin module:

```python
from plap.plugins.easy import bootstrap

bootstrap.config(__file__)
```

The numeric code is stored in public call IDs so function-call outputs from Responses continuations can be routed to the
declaring thread. Codes must be unique:

| Range | Use |
| --- | --- |
| `0..1023` | Core threads |
| `1024..49151` | Registered plugin threads |
| `49152..65535` | Private or experimental threads |

Core uses code `0` for `main`. The built-in `advisor` plugin reviews main-model work and uses code `1024` for its private
history.
