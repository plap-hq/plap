# Write Your First Plugin

In this tutorial, you will add a `server_time` tool. The model can call it when it needs the current UTC time.

## 1. Create the tool

Create `src/plap/plugins/server_time/__init__.py`:

```python
from __future__ import annotations

from dataclasses import field
from datetime import UTC, datetime
from typing import Any

from plap.llms.completions import ChatMessage, ChatToolCall
from plap.plugins.easy import ServerTool, server_tools
from plap.responses.state import State


@server_tools.register
class ServerTime(ServerTool):
    name: str = "server_time"
    description: str = "Return the current UTC time."
    parameters: dict[str, Any] = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {},
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
        return ChatMessage(
            role="tool",
            tool_call_id=call.id,
            content=datetime.now(UTC).isoformat(),
        )
```

The tool has no arguments, but it still declares an object schema. Its return value is a tool message tied to the call
with `tool_call_id=call.id`.

## 2. Register the plugin

Add an entry point to `[project.entry-points."plap.plugin"]` in `pyproject.toml`:

```toml
server_time = "plap.plugins.server_time"
```

Then add the entry-point name to `plugins.toml`:

```toml
plugins = ["core", "easy", "vision", "summary", "advisor", "server_time"]
```

The entry point makes the plugin discoverable. The manifest chooses which discovered plugins load and in what order.

Refresh the editable installation after changing entry points:

```sh
pixi install
```

## 3. Run it

Restart the development server:

```sh
pixi run dev
```

In another terminal:

```sh
source .dev/.env
```

Ask the model to use the tool:

```sh
pixi run python - <<'PY'
import os

from openai import OpenAI

client = OpenAI(
    base_url=os.environ["PLAP_DEV_BASE_URL"],
    api_key=os.environ["PLAP_DEV_API_KEY"],
)

response = client.responses.create(
    model=os.environ["PLAP_DEV_MODEL"],
    input="Use server_time to tell me the current UTC time.",
)

print(response.output_text)
PY
```

## What happened?

plap did four things for the plugin:

1. Added the tool description and schema to the model request.
2. Called `ServerTime` when the model selected it.
3. Added the returned tool message to the `main` thread.
4. Ran another model turn so the model could answer with the tool result.

The plugin did not call the model or manage the response loop itself.

## Continue learning

Congrats! You now have an additive plugin: it gives the model a new capability.

If your next plugin adds another server-owned capability, continue with [Server tools](easy/server-tools.md). It starts from
the `ServerTool` used here and develops it beyond the no-argument `server_time` example.

If your plugin needs to change the process around the model instead of adding another capability, continue with
[Hooks](hooks.md). A hook wraps the relevant response stage, allowing you to modify the corresponding behavior.
