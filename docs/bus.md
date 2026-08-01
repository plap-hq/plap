# Event Bus

Plugins need to extend the same application without replacing core code or depending on one another. plap composes those
extensions through `plap.bus.bus`. Application bootstrap and response execution are both implemented as ordered bus events.

The bus is not a broadcast system. An event has a core handler and a chain of plugin listeners. Each listener decides when
to continue the chain and can change the arguments or return value.

## How an event runs

Suppose core formats a value and a plugin needs to label the result without replacing the formatter. Core defines the final
handler with `@bus.emit`:

```python
from plap.bus import bus


@bus.emit("example.format")
async def format_text(text: str) -> str:
    return text.upper()
```

A plugin wraps that handler with `@bus.listen`:

```python
@bus.listen("example.format")
async def add_prefix(text: str, *, next) -> str:
    formatted = await next(text=text)
    return f"Result: {formatted}"
```

Calling the decorated emitter runs the complete chain:

```python
result = await format_text(text="hello")
assert result == "Result: HELLO"
```

The listener registered last runs first:

```text
later listener: before next
    earlier listener: before next
        core handler
    earlier listener: after next
later listener: after next
```

Plugin import order is therefore part of execution order. Keep `core` first in `plugins.toml`; later plugins then wrap the
core handlers.

## Call `next`

Every listener must accept `next` as a keyword-only argument. Calling it runs the remaining listeners and the core handler:

```python
result = await next(text=text)
```

Arguments passed to `next` replace the corresponding values for the rest of the chain. Arguments not passed again keep
their current values.

A listener can run code before `next`, after it, or both. It can also return without calling `next`, which skips every
remaining listener and the core handler. A listener that stops the chain must return the value expected by that event.

## Hook families

plap uses two main groups of bus events:

| Family | Runs while | Documented in |
| --- | --- | --- |
| `bootstrap.*` | Building the application | [Hooks](hooks.md#bootstrap-hooks) |
| `response.*` | Processing one Responses API request | [Hooks](hooks.md#response-hooks) |

Plugins can listen to either family directly. [`plap.plugins.easy`](easy/README.md) provides shorter APIs for common cases:

| Easy API | Bus hooks underneath |
| --- | --- |
| `easy.bootstrap` | `bootstrap.config`, `bootstrap.routes`, `bootstrap.services`, `bootstrap.shutdown_hooks` |
| `easy.server_tools` | `response.request`, `response.snapshot`, `response.completion` |

Use the easy API when it already expresses the contribution. Use `bus.listen` when the plugin needs to inspect existing
values, change ordering, wrap work before and after a stage, or stop a stage entirely.
