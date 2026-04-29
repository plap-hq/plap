from __future__ import annotations

from plap.responses.tokens import (
    estimate_citation_tokens,
    estimate_message_tokens,
    estimate_text_tokens,
)


def test_estimates_never_return_zero() -> None:
    assert estimate_text_tokens(None) >= 1
    assert estimate_text_tokens("") >= 1
    assert estimate_message_tokens({"role": "assistant", "content": ""}) >= 1
    assert estimate_citation_tokens("[~0_7]") >= 1


def test_message_estimate_is_deterministic_for_key_order() -> None:
    assert estimate_message_tokens(
        {"role": "user", "content": "hello", "name": "alice"}
    ) == estimate_message_tokens(
        {"name": "alice", "content": "hello", "role": "user"}
    )
