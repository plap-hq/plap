"""Declare simple plugin contributions during application bootstrap."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path

import svcs

from plap.bus import bus
from plap.config import CueBox


def config(module_file: str) -> None:
    """Include every production CUE file in the module's directory."""
    directory = str(Path(module_file).resolve().parent)

    @bus.listen("bootstrap.config")
    async def contribute(paths: tuple[str, ...], *, next):
        return await next(paths=(*paths, directory))


def routes(*handlers: object) -> None:
    contributed = tuple(handlers)

    @bus.listen("bootstrap.routes")
    async def contribute(routes: tuple[object, ...], *, next):
        return await next(routes=(*routes, *contributed))


def shutdown_hooks(*hooks: object) -> None:
    contributed = tuple(hooks)

    @bus.listen("bootstrap.shutdown_hooks")
    async def contribute(hooks: tuple[object, ...], *, next):
        return await next(hooks=(*hooks, *contributed))


def services[F: Callable[[svcs.Registry, CueBox], Awaitable[None]]](callback: F) -> F:
    @bus.listen("bootstrap.services")
    async def contribute(registry: svcs.Registry, loaded: CueBox, *, next) -> None:
        await callback(registry, loaded.plap.config)
        await next()

    return callback


__all__ = ["config", "routes", "services", "shutdown_hooks"]
