from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import svcs

from plap.bus import EventBus
from plap.config import CueBox
from plap.plugins.easy import bootstrap


@pytest.fixture
def event_bus(monkeypatch: pytest.MonkeyPatch) -> EventBus:
    isolated = EventBus()
    monkeypatch.setattr(bootstrap, "bus", isolated)
    return isolated


async def test_config_contributes_the_plugin_directory(event_bus: EventBus, tmp_path: Path) -> None:
    @event_bus.emit("bootstrap.config")
    async def terminal(paths: tuple[str, ...]) -> tuple[str, ...]:
        return paths

    module_file = tmp_path / "example_plugin" / "__init__.py"
    bootstrap.config(str(module_file))

    paths = await terminal(paths=("existing",))

    assert paths == ("existing", str(module_file.parent.resolve()))


async def test_routes_and_shutdown_hooks_append_static_contributions(event_bus: EventBus) -> None:
    loaded = object()
    route = object()
    hook = object()

    @event_bus.emit("bootstrap.routes")
    async def terminal_routes(routes: tuple[object, ...], loaded: object) -> tuple[tuple[object, ...], object]:
        return routes, loaded

    @event_bus.emit("bootstrap.shutdown_hooks")
    async def terminal_hooks(hooks: tuple[object, ...], loaded: object) -> tuple[tuple[object, ...], object]:
        return hooks, loaded

    bootstrap.routes(route)
    bootstrap.shutdown_hooks(hook)

    routes, routes_loaded = await terminal_routes(routes=("existing",), loaded=loaded)
    hooks, hooks_loaded = await terminal_hooks(hooks=("existing",), loaded=loaded)

    assert routes == ("existing", route)
    assert hooks == ("existing", hook)
    assert routes_loaded is loaded
    assert hooks_loaded is loaded


async def test_services_receive_application_config_and_continue_downstream(event_bus: EventBus) -> None:
    registry = svcs.Registry()
    config = CueBox({"name": "test"}, frozen_box=True)
    loaded = SimpleNamespace(plap=SimpleNamespace(config=config))
    seen: list[str] = []

    @event_bus.emit("bootstrap.services")
    async def terminal(registry: svcs.Registry, loaded: object) -> None:
        _ = registry, loaded
        seen.append("terminal")

    @bootstrap.services
    async def services(contributed_registry: svcs.Registry, contributed_config: CueBox) -> None:
        assert contributed_registry is registry
        assert contributed_config is config
        seen.append("services")

    await terminal(registry=registry, loaded=loaded)

    assert seen == ["services", "terminal"]
