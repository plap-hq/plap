from __future__ import annotations

from plap.config import CueBox
from plap.llms.completions import ChatCompletionClient, ModelRoute, RoutingChatCompletionClient, UnavailableChatCompletionClient
from plap.llms.completions.chat import IChatCompletionClient
from plap.llms.completions.providers import build_providers


def build_chat_completion_client(config: CueBox) -> IChatCompletionClient:
    providers = build_providers(config)
    if not providers:
        return UnavailableChatCompletionClient()
    routes = [ModelRoute(prefix=prefix, client=ChatCompletionClient(provider)) for prefix, provider in providers.items()]
    return RoutingChatCompletionClient(routes)
