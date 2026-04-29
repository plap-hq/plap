from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from typing import Any

import msgspec
import tiktoken

_DEFAULT_ENCODING = "o200k_base"


@lru_cache(maxsize=1)
def _encoding() -> tiktoken.Encoding:
    return tiktoken.get_encoding(_DEFAULT_ENCODING)


def estimate_text_tokens(text: str | None) -> int:
    if not text:
        return 1
    return max(1, len(_encoding().encode(text)))


def estimate_message_tokens(message: Mapping[str, Any]) -> int:
    encoded = msgspec.json.encode(message, order="deterministic").decode()
    return estimate_text_tokens(encoded)


def estimate_citation_tokens(citation: str) -> int:
    return estimate_text_tokens(f"{citation}\n")
