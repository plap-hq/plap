from pydantic import ValidationError

from plap.responses.contracts import ResponseCreateRequest


def test_rejects_unsupported_input_variant() -> None:
    try:
        ResponseCreateRequest.model_validate(
            {
                "input": [{"type": "computer_call"}],
                "model": "gpt-4.1",
            }
        )
    except ValidationError as exc:
        assert "Unsupported input item variant 'computer_call'" in str(exc)
    else:
        raise AssertionError("expected validation error")


def test_rejects_unsupported_tool_variant() -> None:
    try:
        ResponseCreateRequest.model_validate(
            {
                "model": "gpt-4.1",
                "tools": [{"type": "file_search"}],
            }
        )
    except ValidationError as exc:
        assert "Unsupported tool variant 'file_search'" in str(exc)
    else:
        raise AssertionError("expected validation error")


def test_rejects_unsupported_context_management_variant() -> None:
    try:
        ResponseCreateRequest.model_validate(
            {
                "context_management": [{"type": "retain_all"}],
                "model": "gpt-4.1",
            }
        )
    except ValidationError as exc:
        assert "compaction" in str(exc)
    else:
        raise AssertionError("expected validation error")
