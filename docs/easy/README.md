# Easy Plugin APIs

`plap.plugins.easy` is convenience syntax for common event-bus patterns. It does not provide a separate plugin runtime.

| Module | What it simplifies | Bus hooks used internally |
| --- | --- | --- |
| [`server_tools`](server-tools.md) | Registering and executing server-owned model tools | `response.request`, `response.snapshot`, `response.completion` |
| [`bootstrap`](bootstrap.md) | Contributing config, routes, services, and shutdown hooks | The four `bootstrap.*` hooks |

Server tools add model-callable functionality. Bootstrap helpers shorten application wiring.

Both modules register listeners on the [event bus](../bus.md). [Hooks](../hooks.md) lists the bootstrap and response event
signatures.
