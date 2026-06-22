from __future__ import annotations

from pathlib import Path

import svcs

from plap.bus import bus
from plap.config import CueBox
from plap.llms.completions.chat import IChatCompletionClient
from plap.plugins.core.client import build_chat_completion_client
from plap.plugins.core.loop import run_response
from plap.responses.routes import RESPONSE_ROUTE_HANDLERS
from plap.responses.state import State


@bus.listen("config.collect")
async def collect(paths: tuple[str, ...], *, next):
    here = Path(__file__).resolve()
    return await next(paths=(*paths, str(here.parents[4] / "config.cue"), str(here.parent / "schema.cue")))


@bus.listen("routes.collect")
async def collect_routes(routes: tuple[object, ...], loaded: CueBox, *, next):
    _ = loaded
    return await next(routes=(*routes, *RESPONSE_ROUTE_HANDLERS))


@bus.listen("svcs.collect")
async def collect_svcs(registry: svcs.Registry, loaded: CueBox, *, next):
    client = build_chat_completion_client(loaded.plap.config)
    registry.register_value(IChatCompletionClient, client, on_registry_close=client.aclose)
    return await next()


@bus.listen("response.start")
async def start_response(state: State, *, next) -> None:
    await run_response(state=state)
    return await next()
