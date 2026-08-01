from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from plap.llms.completions.budget import (
    BudgetedChatCompletionClient,
    CompletionBudget,
    CompletionBudgetExhaustedError,
)
from plap.llms.completions.chat import (
    ChatCompletionDelta,
    ChatCompletionRequest,
    ChatCompletionResult,
    ChatFinishReason,
    ChatMessage,
    ChatUsage,
    OutputEquivalence,
)


def _equivalence(**updates: float) -> OutputEquivalence:
    values = {
        "uncached_input_to_output": 0.25,
        "cached_input_to_output": 0.05,
        "output_to_output": 1.0,
        **updates,
    }
    return OutputEquivalence(**values)


def _usage(
    *,
    input_tokens: int,
    output_tokens: int,
    cached_tokens: int | None = None,
    reasoning_tokens: int | None = None,
) -> ChatUsage:
    return ChatUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        cached_tokens=cached_tokens,
        reasoning_tokens=reasoning_tokens,
    )


def _result(usage: ChatUsage) -> ChatCompletionResult:
    return ChatCompletionResult(
        id="cmpl_test",
        model="test-model",
        created_at=None,
        message=ChatMessage(role="assistant", content="done"),
        finish_reason=ChatFinishReason.STOP,
        usage=usage,
    )


def _delta(usage: ChatUsage) -> ChatCompletionDelta:
    return ChatCompletionDelta(
        id="cmpl_test",
        model="test-model",
        created_at=None,
        choice_index=0,
        finish_reason=ChatFinishReason.STOP,
        usage=usage,
    )


class _Client:
    def __init__(
        self,
        *,
        complete_results: list[ChatCompletionResult] | None = None,
        stream_results: list[ChatUsage] | None = None,
    ) -> None:
        self.complete_results = list(complete_results or [])
        self.stream_results = list(stream_results or [])
        self.requests: list[ChatCompletionRequest] = []

    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResult:
        self.requests.append(request)
        return self.complete_results.pop(0)

    def stream(self, request: ChatCompletionRequest) -> AsyncIterator[ChatCompletionDelta]:
        self.requests.append(request)
        usage = self.stream_results.pop(0)

        async def run() -> AsyncIterator[ChatCompletionDelta]:
            yield _delta(usage)

        return run()

    async def aclose(self) -> None:
        return None


def _request(
    equivalence: OutputEquivalence | None = None,
    *,
    max_completion_tokens: int | None = None,
) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="test-model",
        messages=[],
        max_completion_tokens=max_completion_tokens,
        output_equivalence=equivalence,
    )


async def _record(
    budget: CompletionBudget,
    usage: ChatUsage,
    *,
    equivalence: OutputEquivalence | None = None,
    max_completion_tokens: int | None = None,
) -> ChatCompletionRequest:
    raw_client = _Client(stream_results=[usage])
    client = BudgetedChatCompletionClient(raw_client, budget)
    async for _ in client.stream(_request(equivalence or _equivalence(), max_completion_tokens=max_completion_tokens)):
        pass
    return raw_client.requests[0]


def test_finish_returns_none_without_charges() -> None:
    budget = CompletionBudget(None, reasoning_to_output=1.0)

    assert budget.finish(output_usage=None) is None


async def test_completion_limit_uses_per_call_equivalence_and_configured_cap() -> None:
    uncapped = await _record(
        CompletionBudget(20, reasoning_to_output=1.0),
        _usage(input_tokens=1, output_tokens=1),
        equivalence=_equivalence(output_to_output=2.0),
    )
    capped = await _record(
        CompletionBudget(20, reasoning_to_output=1.0),
        _usage(input_tokens=1, output_tokens=1),
        equivalence=_equivalence(output_to_output=2.0),
        max_completion_tokens=7,
    )

    assert uncapped.max_completion_tokens == 10
    assert capped.max_completion_tokens == 7


async def test_finish_scales_reasoning_and_reclassifies_final_output() -> None:
    budget = CompletionBudget(20, reasoning_to_output=1.5)
    usage = _usage(input_tokens=10, output_tokens=12, cached_tokens=1, reasoning_tokens=5)

    await _record(budget, usage)
    aggregate = budget.finish(output_usage=usage)

    assert aggregate == ChatUsage(
        input_tokens=10,
        output_tokens=15,
        total_tokens=25,
        cached_tokens=1,
        reasoning_tokens=8,
    )
    assert budget.remaining == 5


async def test_anchor_keeps_first_main_input_while_other_usage_stays_internal() -> None:
    budget = CompletionBudget(100, reasoning_to_output=1.0)
    first_main = _usage(input_tokens=80, output_tokens=15)
    final_main = _usage(input_tokens=10, output_tokens=5, cached_tokens=2, reasoning_tokens=1)

    await _record(budget, first_main)
    budget.anchor(input_usage=first_main)
    await _record(budget, final_main)
    budget.anchor(input_usage=final_main)
    aggregate = budget.finish(output_usage=final_main)

    assert aggregate == ChatUsage(
        input_tokens=80,
        output_tokens=40,
        total_tokens=120,
        cached_tokens=0,
        reasoning_tokens=36,
    )


async def test_anchor_overrides_an_earlier_internal_charge() -> None:
    budget = CompletionBudget(100, reasoning_to_output=1.0)
    internal = _usage(input_tokens=80, output_tokens=2)
    main = _usage(input_tokens=10, output_tokens=5, cached_tokens=3)

    await _record(budget, internal)
    await _record(budget, main)
    budget.anchor(input_usage=main)
    aggregate = budget.finish(output_usage=main)

    assert aggregate is not None
    assert aggregate.input_tokens == 10
    assert aggregate.cached_tokens == 3


async def test_finish_without_anchor_uses_output_then_first_charge() -> None:
    output_budget = CompletionBudget(100, reasoning_to_output=1.0)
    internal = _usage(input_tokens=80, output_tokens=2)
    output = _usage(input_tokens=10, output_tokens=5)
    await _record(output_budget, internal)
    await _record(output_budget, output)

    output_aggregate = output_budget.finish(output_usage=output)

    internal_budget = CompletionBudget(100, reasoning_to_output=1.0)
    await _record(internal_budget, internal)
    internal_aggregate = internal_budget.finish(output_usage=None)

    assert output_aggregate is not None
    assert output_aggregate.input_tokens == 10
    assert internal_aggregate is not None
    assert internal_aggregate.input_tokens == 80


async def test_discounted_output_never_reports_less_than_visible_output() -> None:
    budget = CompletionBudget(100, reasoning_to_output=1.0)
    usage = _usage(input_tokens=10, output_tokens=20, reasoning_tokens=5)

    await _record(budget, usage, equivalence=_equivalence(output_to_output=0.5))
    aggregate = budget.finish(output_usage=usage)

    assert aggregate == ChatUsage(
        input_tokens=10,
        output_tokens=15,
        total_tokens=25,
        cached_tokens=0,
        reasoning_tokens=0,
    )
    assert budget.remaining == 90


async def test_exhaustion_stops_before_another_provider_call() -> None:
    usage = _usage(input_tokens=0, output_tokens=1)
    raw_client = _Client(stream_results=[usage])
    client = BudgetedChatCompletionClient(raw_client, CompletionBudget(1, reasoning_to_output=1.0))
    request = _request(_equivalence())

    async for _ in client.stream(request):
        pass
    with pytest.raises(CompletionBudgetExhaustedError):
        async for _ in client.stream(request):
            pass

    assert len(raw_client.requests) == 1


async def test_budgeted_client_requires_per_call_equivalence() -> None:
    raw_client = _Client(stream_results=[_usage(input_tokens=1, output_tokens=1)])
    client = BudgetedChatCompletionClient(raw_client, CompletionBudget(10, reasoning_to_output=1.0))

    with pytest.raises(ValueError, match="requires output equivalence"):
        async for _ in client.stream(_request()):
            pass

    assert raw_client.requests == []


async def test_complete_records_usage() -> None:
    usage = _usage(input_tokens=3, output_tokens=2)
    raw_client = _Client(complete_results=[_result(usage)])
    budget = CompletionBudget(10, reasoning_to_output=1.0)
    client = BudgetedChatCompletionClient(raw_client, budget)

    await client.complete(_request(_equivalence()))

    assert budget.finish(output_usage=usage) is not None


async def test_anchor_and_finish_reject_unrecorded_usage_and_duplicate_finish() -> None:
    budget = CompletionBudget(20, reasoning_to_output=1.0)
    usage = _usage(input_tokens=1, output_tokens=1)

    with pytest.raises(RuntimeError, match="input usage was not recorded"):
        budget.anchor(input_usage=usage)
    with pytest.raises(RuntimeError, match="output usage was not recorded"):
        budget.finish(output_usage=usage)

    budget = CompletionBudget(20, reasoning_to_output=1.0)
    await _record(budget, usage)
    budget.finish(output_usage=usage)

    with pytest.raises(RuntimeError, match="already been finished"):
        budget.finish(output_usage=usage)
