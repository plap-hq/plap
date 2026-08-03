# Server Tools

Server tools give the model new capabilities implemented by your plugin. The model can choose a tool during a turn, receive
its result, and continue answering with that result in context.

The first plugin used this API to add `server_time`. A server tool can also accept arguments, use response state, call other
services, and return expected failures as tool results.

## Define and register a tool

A registered tool subclasses `ServerTool` and defines:

| Member | Purpose |
| --- | --- |
| `name` | Stable name used by plugin code and saved history |
| `description` | Tells the model when to call the tool |
| `parameters` | JSON Schema for the arguments, or `None` |
| `strict` | Requests exact schema conformance from providers that support it |
| `__call__(state, call)` | Executes the tool on the server |

Register the class with `server_tools.register`:

```python
from dataclasses import field
from typing import Any

import msgspec

from plap.llms.completions import ChatMessage, ChatToolCall
from plap.plugins.easy import ServerTool, server_tools
from plap.responses.state import State

_RECORDS = {
    "example": "record contents",
}


@server_tools.register
class Lookup(ServerTool):
    name: str = "lookup"
    description: str = "Look up a record by ID."
    parameters: dict[str, Any] = field(
        # ServerTool subclasses are dataclasses; mutable field defaults need a factory.
        default_factory=lambda: {
            "type": "object",
            "properties": {"id": {"type": "string"}},
            "required": ["id"],
            "additionalProperties": False,
        }
    )
    strict: bool = True

    async def __call__(
        self,
        state: State,
        call: ChatToolCall,
    ) -> ChatMessage:
        _ = state
        arguments = msgspec.json.decode(call.arguments)
        if not isinstance(arguments, dict):
            raise ValueError("lookup arguments must be an object")

        record_id = arguments.get("id")
        if not isinstance(record_id, str) or not record_id:
            raise ValueError("lookup id must be a non-empty string")

        return ChatMessage(
            role="tool",
            tool_call_id=call.id,
            content=_RECORDS.get(record_id, "record not found"),
        )
```

`ServerTool` subclasses are frozen dataclasses. Registration constructs one instance, so the class must be constructible
without arguments. Tool names must be unique among registered server tools.

## Read arguments

`call.arguments` contains JSON text. `Lookup` decodes the object, checks that `id` is a non-empty string, and then reads the
record.

The completion retry layer asks the model to correct malformed JSON and schema mismatches before the tool runs. The tool
still validates application rules such as permissions, missing records, and valid identifier values.

## Return a tool result

The implementation returns a `tool` message with `tool_call_id=call.id`. plap rejects a message with another role or call
ID. [Response state](../state.md) provides the request, configuration, memory, threads, and services available through
`state`.

## Stable and temporary names

Saved server-tool calls use the stable `name` declared by the plugin. If an API request already contains a client tool with
that name, plap gives the server tool a temporary wire name such as `lookup_2`. Plugin code and saved history continue to use
`lookup`.

`server_tools.rename_to_avoid_collisions(function, tools)` exposes the same naming rule for plugins that construct temporary
`ChatFunctionTool` values themselves.

## Failures and response budgets

An unhandled tool exception fails the response. Catch an exception when it represents an expected tool result, then return
that result as a tool message.

A server tool can make another model call through the budgeted completion client. If the call raises
`CompletionBudgetExhaustedError`, plap returns `server_tools.BUDGET_TOOL_OUTPUT` for that tool call so the response can stop
cleanly.

## Server tool or client tool?

Use a server tool when the plugin implements the function and can produce its result during the response.

Use a client tool when the API caller implements the function. plap returns a function-call item, and a later Responses
request supplies its output.

Server tools add a capability without changing the surrounding response behavior. A [response hook](../hooks.md#response-hooks)
can change or wrap behavior that is part of the normal response cycle.
