# Bootstrap Helpers

Application bootstrap is exposed through four [event-bus hooks](../hooks.md#bootstrap-hooks). Most plugins only need to add
one value to the config paths, routes, services, or shutdown hooks. `plap.plugins.easy.bootstrap` provides short forms for
those common listeners.

| Helper | Hook | Contribution |
| --- | --- | --- |
| `bootstrap.config(__file__)` | `bootstrap.config` | The plugin module's directory |
| `bootstrap.routes(*handlers)` | `bootstrap.routes` | Litestar route handlers |
| `@bootstrap.services` | `bootstrap.services` | A service-registration callback |
| `bootstrap.shutdown_hooks(*hooks)` | `bootstrap.shutdown_hooks` | Litestar shutdown hooks |

This helper:

```python
bootstrap.routes(plugin_status)
```

With `bus` and `plugin_status` already defined, it registers the same kind of listener as this fragment:

```python
@bus.listen("bootstrap.routes")
async def contribute_route(routes: tuple[object, ...], *, next):
    return await next(routes=(*routes, plugin_status))
```

The helper is for a static contribution. A direct listener can inspect or transform values already in the chain.

## Load CUE configuration

Place `schema.cue` beside the plugin module:

```cue
package plap

#Config: {
  greeting: {
    prefix: *"Hello" | string
  }
}
```

Register that directory from the plugin module:

```python
from plap.plugins.easy import bootstrap

bootstrap.config(__file__)
```

All production CUE files in the module's directory are loaded with the core configuration.

## Add HTTP routes

Pass Litestar route handlers or routers to `bootstrap.routes`:

```python
from litestar import get

from plap.plugins.easy import bootstrap


@get("/plugin-status")
async def plugin_status() -> dict[str, str]:
    return {"status": "ok"}


bootstrap.routes(plugin_status)
```

## Register services

The `bootstrap.services` decorator receives the application service registry and the loaded application configuration:

```python
import svcs

from plap.config import CueBox
from plap.plugins.easy import bootstrap


class GreetingService:
    def __init__(self, prefix: str) -> None:
        self.prefix = prefix

    def greet(self, name: str) -> str:
        return f"{self.prefix}, {name}"


@bootstrap.services
async def register_services(registry: svcs.Registry, config: CueBox) -> None:
    registry.register_value(
        GreetingService,
        GreetingService(prefix=str(config.greeting.prefix)),
    )
```

Hooks and server tools can retrieve the service from response state:

```python
greetings = await state.svcs.aget(GreetingService)
```

Use registry close callbacks for resources owned by a registered service.

## Add shutdown work

Pass application shutdown hooks to `bootstrap.shutdown_hooks`:

```python
import structlog

from plap.plugins.easy import bootstrap

logger = structlog.stdlib.get_logger(__name__)


async def flush_pending_events() -> None:
    logger.info("my_plugin.shutdown")


bootstrap.shutdown_hooks(flush_pending_events)
```

Plugin shutdown hooks run before plap closes its database, telemetry, and application service registry.

## Use a direct bootstrap hook

Use `bus.listen` instead of a helper when the contribution must inspect loaded configuration, reorder or replace existing
values, or run work after the rest of the bootstrap chain.
