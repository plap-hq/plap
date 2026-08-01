from __future__ import annotations

from pathlib import Path

import svcs

from plap.bus import bus
from plap.config import CueBox
from plap.llms.completions.budget import BudgetedChatCompletionClient, CompletionBudget
from plap.llms.completions.chat import IChatCompletionClient
from plap.plugins.core.client import build_chat_completion_client
from plap.plugins.core.loop import run_response as run_response
from plap.responses.routes import RESPONSE_ROUTE_HANDLERS


@bus.listen("bootstrap.config")
async def bootstrap_config(paths: tuple[str, ...], *, next):
    here = Path(__file__).resolve()
    return await next(paths=(*paths, str(here.parents[4] / "config.cue"), str(here.parent)))


@bus.listen("bootstrap.routes")
async def bootstrap_routes(routes: tuple[object, ...], loaded: CueBox, *, next):
    _ = loaded
    return await next(routes=(*routes, *RESPONSE_ROUTE_HANDLERS))


@bus.listen("bootstrap.services")
async def bootstrap_services(registry: svcs.Registry, loaded: CueBox, *, next):
    base_client = build_chat_completion_client(loaded.plap.config)

    def budgeted_client_factory(svcs_container: svcs.Container) -> BudgetedChatCompletionClient:
        return BudgetedChatCompletionClient(
            svcs_container.get(IChatCompletionClient),
            svcs_container.get(CompletionBudget),
        )

    registry.register_value(IChatCompletionClient, base_client, on_registry_close=base_client.aclose)
    registry.register_factory(BudgetedChatCompletionClient, budgeted_client_factory)
    return await next()


__all__ = ["run_response"]
