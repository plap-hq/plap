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


def test_accepts_easy_input_message_shorthand() -> None:
    request = ResponseCreateRequest.model_validate(
        {
            "input": [{"role": "user", "content": "hello"}],
            "model": "gpt-4.1",
        }
    )

    assert request.input is not None
    assert isinstance(request.input, list)
    assert len(request.input) == 1
    assert request.input[0].type == "message"
    assert request.input[0].role == "user"
    assert request.input[0].content == "hello"


def test_rejects_typeless_input_item_without_message_role_shape() -> None:
    try:
        ResponseCreateRequest.model_validate(
            {
                "input": [{"content": "hello"}],
                "model": "gpt-4.1",
            }
        )
    except ValidationError as exc:
        assert "Missing input item type at index 0" in str(exc)
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


def test_accepts_function_tool_without_strict() -> None:
    request = ResponseCreateRequest.model_validate(
        {
            "model": "gpt-4.1",
            "tools": [
                {
                    "type": "function",
                    "name": "question",
                    "description": "Ask the user for clarification.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "question": {"type": "string"},
                        },
                        "required": ["question"],
                        "additionalProperties": False,
                    },
                }
            ],
        }
    )

    assert request.tools is not None
    assert len(request.tools) == 1
    assert request.tools[0].type == "function"
    assert request.tools[0].name == "question"
    assert request.tools[0].strict is None


def test_accepts_reasoning_input_without_id() -> None:
    request = ResponseCreateRequest.model_validate(
        {
            "input": [
                {
                    "type": "reasoning",
                    "encrypted_content": "sealed",
                    "summary": [],
                }
            ],
            "model": "gpt-4.1",
        }
    )

    assert request.input is not None
    assert isinstance(request.input, list)
    assert len(request.input) == 1
    assert request.input[0].type == "reasoning"
    assert request.input[0].id is None


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


def test_rejects_empty_compaction_context_management() -> None:
    try:
        ResponseCreateRequest.model_validate(
            {
                "context_management": [{"type": "compaction"}],
                "model": "gpt-4.1",
            }
        )
    except ValidationError as exc:
        assert "at least one budget override" in str(exc)
    else:
        raise AssertionError("expected validation error")


def test_rejects_invalid_compaction_budget_order() -> None:
    try:
        ResponseCreateRequest.model_validate(
            {
                "context_management": [{"type": "compaction", "soft_token_budget": 100, "hard_token_budget": 100}],
                "model": "gpt-4.1",
            }
        )
    except ValidationError as exc:
        assert "hard_token_budget" in str(exc)
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
        ResponseCreateRequest.model_validate({"model": "gpt-4.1", "reasoning": {"effort": "maximum"}})
    except ValidationError as exc:
        assert "effort" in str(exc)
    else:
        raise AssertionError("expected validation error")


def test_accepts_full_reasoning_effort_vocabulary() -> None:
    for effort in ("none", "minimal", "low", "medium", "high", "xhigh"):
        request = ResponseCreateRequest.model_validate({"model": "gpt-4.1", "reasoning": {"effort": effort}})

        assert request.reasoning is not None
        assert request.reasoning.effort == effort


def test_rejects_metadata_limits() -> None:
    cases = [
        {str(index): "value" for index in range(17)},
        {"k" * 65: "value"},
        {"key": "v" * 513},
    ]

    for metadata in cases:
        try:
            ResponseCreateRequest.model_validate({"metadata": metadata, "model": "gpt-4.1"})
        except ValidationError:
            pass
        else:
            raise AssertionError("expected validation error")
