from __future__ import annotations

from plap.bus import EventBus


async def test_bus_next_override_updates_downstream_handlers() -> None:
    bus = EventBus()
    seen: list[str] = []

    @bus.emit("thing")
    async def body(request: str) -> None:
        seen.append(f"body:{request}")

    @bus.listen("thing")
    async def second(request: str, *, next) -> None:
        seen.append(request)
        await next()

    @bus.listen("thing")
    async def first(request: str, *, next) -> None:
        await next(request="mutated")

    await body(request="original")

    assert seen == ["mutated", "body:mutated"]


async def test_bus_next_override_can_introduce_new_key() -> None:
    bus = EventBus()
    seen: list[tuple[str, dict[str, int]]] = []

    @bus.emit("thing")
    async def body(request: str, tool_plan: dict[str, int]) -> None:
        seen.append((f"body:{request}", tool_plan))

    @bus.listen("thing")
    async def second(request: str, tool_plan: dict[str, int], *, next) -> None:
        seen.append((request, tool_plan))
        await next()

    @bus.listen("thing")
    async def first(request: str, *, next) -> None:
        await next(tool_plan={"steps": 1})

    await body(request="original")

    assert seen == [
        ("original", {"steps": 1}),
        ("body:original", {"steps": 1}),
    ]


async def test_bus_later_override_wins() -> None:
    bus = EventBus()
    seen: list[str] = []

    @bus.emit("thing")
    async def body(status: str) -> None:
        seen.append(status)

    @bus.listen("thing")
    async def second(status: str, *, next) -> None:
        await next(status="second")

    @bus.listen("thing")
    async def first(*, next) -> None:
        await next(status="first")

    await body()

    assert seen == ["second"]


async def test_bus_unused_override_key_is_ignored() -> None:
    bus = EventBus()
    seen: list[str] = []

    @bus.emit("thing")
    async def body(request: str) -> None:
        seen.append(request)

    @bus.listen("thing")
    async def first(request: str, *, next) -> None:
        await next(unused="value")

    await body(request="original")

    assert seen == ["original"]


async def test_bus_override_only_changes_downstream_args() -> None:
    bus = EventBus()
    seen: list[str] = []

    @bus.emit("thing")
    async def body(request: str) -> None:
        seen.append(f"body:{request}")

    @bus.listen("thing")
    async def first(request: str, *, next) -> None:
        seen.append(f"before:{request}")
        await next(request="mutated")
        seen.append(f"after:{request}")

    await body(request="original")

    assert seen == ["before:original", "body:mutated", "after:original"]


async def test_bus_emit_returns_terminal_result() -> None:
    bus = EventBus()

    @bus.emit("thing")
    async def body(request: str) -> str:
        return f"body:{request}"

    result = await body(request="original")

    assert result == "body:original"


async def test_bus_next_propagates_terminal_result() -> None:
    bus = EventBus()
    seen: list[str] = []

    @bus.emit("thing")
    async def body(request: str) -> str:
        return f"body:{request}"

    @bus.listen("thing")
    async def first(request: str, *, next) -> str:
        seen.append(f"before:{request}")
        result = await next(request="mutated")
        seen.append(result)
        return result

    result = await body(request="original")

    assert result == "body:mutated"
    assert seen == ["before:original", "body:mutated"]
