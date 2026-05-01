from __future__ import annotations

from plap.responses.models import ChatMessageSpan, StateMessage


def test_estimates_never_return_zero() -> None:
    empty_message = StateMessage(role="assistant", content="")
    cited_span = ChatMessageSpan(start=0, end=7, message=StateMessage(role="assistant", content="summary"), token_count=1)

    assert empty_message.estimated_token_count() >= 1
    assert cited_span.citation_token_count() >= 1


def test_message_estimate_is_deterministic_for_key_order() -> None:
    first = StateMessage(role="user", content="hello", name="alice")
    second = StateMessage.from_primitive({"name": "alice", "content": "hello", "role": "user"})

    assert first.estimated_token_count() == second.estimated_token_count()
