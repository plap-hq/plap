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


def test_rejects_unknown_reasoning_fields() -> None:
    try:
        ResponseCreateRequest.model_validate(
            {
                "model": "gpt-4.1",
                "reasoning": {"effort": "medium", "unsupported": True},
            }
        )
    except ValidationError as exc:
        assert "unsupported" in str(exc)
    else:
        raise AssertionError("expected validation error")


def test_rejects_invalid_reasoning_effort() -> None:
    try:
        ResponseCreateRequest.model_validate(
            {"model": "gpt-4.1", "reasoning": {"effort": "maximum"}}
        )
    except ValidationError as exc:
        assert "effort" in str(exc)
    else:
        raise AssertionError("expected validation error")


def test_rejects_metadata_limits() -> None:
    cases = [
        {str(index): "value" for index in range(17)},
        {"k" * 65: "value"},
        {"key": "v" * 513},
    ]

    for metadata in cases:
        try:
            ResponseCreateRequest.model_validate(
                {"metadata": metadata, "model": "gpt-4.1"}
            )
        except ValidationError:
            pass
        else:
            raise AssertionError("expected validation error")
