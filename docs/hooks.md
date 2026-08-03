# Hooks

Server tools extend what the model can do. Hooks change how plap itself behaves.

A hook can change a model request before it is sent, inspect a turn before the loop continues, or alter how the application
starts. Underneath, the [event bus](bus.md) implements each hook as a core handler with an ordered listener chain.

## Response hooks

The response hooks follow the nested structure of one response:

```text
response.start
├── response.loop
│   └── repeat:
│       ├── response.request
│       └── response.turn
│           └── response.completion
│               ├── response.snapshot*
│               └── response.summary
└── response.commit
```

`response.turn` calls `response.completion` once. A completion can make several attempts, and each attempt can produce many
snapshots.

| Hook | Receives | Returns | What it can change |
| --- | --- | --- | --- |
| `response.start` | `state` | `None` | The complete response, including final status |
| `response.loop` | `state` | A result or `None` | Whether and how main-model turns repeat |
| `response.request` | `state` | `ChatCompletionRequest` | Messages, tools, and options for the next main-model call |
| `response.turn` | `state`, `request` | `ChatCompletionResult` | Work before or after one accepted main-model turn |
| `response.completion` | `state`, `request`, `validators` | `ChatCompletionResult` | Streaming and retry behavior for one turn |
| `response.snapshot` | `state`, `request`, `snapshot` | `Snapshot` | Each accumulated update from a completion attempt |
| `response.summary` | `state`, `source` | `None` | How [reasoning summaries](summary.md) stream during completion |
| `response.commit` | `state` | `None` | State finalization and published response output |

### Examples

#### Change the next model request

Suppose every main-model turn must follow the same house style. A `response.request` listener can add that instruction after
plap builds each request:

```python
from dataclasses import replace

from plap.bus import bus
from plap.llms.completions import ChatCompletionRequest, ChatMessage
from plap.responses.state import State


@bus.listen("response.request")
async def add_instruction(
    state: State,
    *,
    next,
) -> ChatCompletionRequest:
    request = await next(state=state)
    return replace(
        request,
        messages=[
            ChatMessage(role="developer", content="Prefer short answers."),
            *request.messages,
        ],
    )
```

The listener calls `next` first so the rest of the chain can build the request. It then prepends the instruction and returns
the replacement. `ChatCompletionRequest` is a frozen dataclass, so `dataclasses.replace` preserves the other request fields.

#### Run work around a turn

A model turn may need several completion attempts before one passes validation. If a plugin records one duration for that
whole process, `response.turn` provides the correct start and end points:

```python
import time

import structlog

from plap.bus import bus
from plap.llms.completions import ChatCompletionRequest, ChatCompletionResult
from plap.responses.state import State

logger = structlog.stdlib.get_logger(__name__)


@bus.listen("response.turn")
async def time_turn(
    state: State,
    request: ChatCompletionRequest,
    *,
    next,
) -> ChatCompletionResult:
    started = time.monotonic()
    result = await next(state=state, request=request)
    logger.info("my_plugin.turn", elapsed=time.monotonic() - started)
    return result
```

The timer starts before the first completion attempt and stops after one attempt passes the validators. Rejected attempts
remain inside the same measurement.

Every response hook has access to the current [State](state.md). A plugin that calls another model can keep its messages out
of the main request with a separate [thread](threads.md).

## Bootstrap hooks

Response hooks run after the server has accepted a request. Routes, services, configuration sources, and shutdown callbacks
must be added while the application is being built. Bootstrap hooks provide those earlier extension points.

| Hook | Receives | Returns | What it can change |
| --- | --- | --- | --- |
| `bootstrap.config` | `paths` | Config paths | CUE sources loaded by the application |
| `bootstrap.services` | `registry`, `loaded` | `None` | Services registered after configuration loads |
| `bootstrap.routes` | `routes`, `loaded` | Route handlers | Litestar routes included in the application |
| `bootstrap.shutdown_hooks` | `hooks`, `loaded` | Shutdown hooks | Work run during Litestar shutdown |

Configuration runs first. The other bootstrap hooks receive the loaded `CueBox`. Services are registered before routes and
shutdown hooks are collected.

`easy.bootstrap.routes` is enough for a route that is always enabled. A direct bootstrap listener can instead decide whether
to add a route from loaded configuration:

```python
from litestar import get

from plap.bus import bus
from plap.config import CueBox


@get("/plugin-status")
async def plugin_status() -> dict[str, str]:
    return {"status": "ok"}


@bus.listen("bootstrap.routes")
async def add_status_route(
    routes: tuple[object, ...],
    loaded: CueBox,
    *,
    next,
) -> tuple[object, ...]:
    if not loaded.plap.config.my_plugin.status_route_enabled:
        return await next(routes=routes, loaded=loaded)
    return await next(
        routes=(*routes, plugin_status),
        loaded=loaded,
    )
```

`status_route_enabled` is a boolean supplied by the plugin's registered CUE schema.

[`easy.bootstrap`](easy/bootstrap.md) handles static config paths, routes, service callbacks, and shutdown hooks. A direct
listener can inspect `loaded`, transform existing contributions, control ordering, or run code after the remaining handlers.
