from __future__ import annotations

from typing import Any

from plap.llms.openai_compatible import OpenAICompatibleChatCompletionClient


class LightningChatCompletionClient(OpenAICompatibleChatCompletionClient):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        client: Any | None = None,
    ) -> None:
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            client=client,
            developer_role="system",
        )
