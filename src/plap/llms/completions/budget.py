"""Apply one output-equivalent token budget across chat completions."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import asdict, dataclass, replace
from math import ceil, floor

import structlog

from plap.llms.completions.chat import (
    ChatCompletionDelta,
    ChatCompletionRequest,
    ChatCompletionResult,
    ChatFinishReason,
    ChatUsage,
    IChatCompletionClient,
    OutputEquivalence,
)

logger = structlog.stdlib.get_logger(__name__)


def _cached_input_tokens(usage: ChatUsage) -> int:
    return min(usage.cached_tokens or 0, usage.input_tokens)


def _visible_output_tokens(usage: ChatUsage) -> int:
    return max(0, usage.output_tokens - (usage.reasoning_tokens or 0))


def _output_equivalent_tokens(usage: ChatUsage, *, reasoning_to_output: float) -> int:
    return _visible_output_tokens(usage) + ceil((usage.reasoning_tokens or 0) * reasoning_to_output)


def _output_cost(equivalence: OutputEquivalence, usage: ChatUsage, *, reasoning_to_output: float) -> int:
    return ceil(equivalence.output_to_output * _output_equivalent_tokens(usage, reasoning_to_output=reasoning_to_output))


def _internal_cost(equivalence: OutputEquivalence, usage: ChatUsage, *, reasoning_to_output: float) -> int:
    cached_input = _cached_input_tokens(usage)
    uncached_input = usage.input_tokens - cached_input
    debit = (
        uncached_input * equivalence.uncached_input_to_output
        + cached_input * equivalence.cached_input_to_output
        + equivalence.output_to_output * _output_equivalent_tokens(usage, reasoning_to_output=reasoning_to_output)
    )
    return ceil(debit)


@dataclass(frozen=True, slots=True)
class _Charge:
    equivalence: OutputEquivalence
    usage: ChatUsage


class CompletionBudgetExhaustedError(Exception):
    pass


class CompletionBudget:
    def __init__(self, max_output_tokens: int | None, *, reasoning_to_output: float) -> None:
        self._remaining = max_output_tokens
        self._reasoning_to_output = reasoning_to_output
        self._charges: list[_Charge] = []
        self._input: _Charge | None = None
        self._finished = False

    @property
    def remaining(self) -> int | None:
        return self._remaining

    def _completion_limit(self, equivalence: OutputEquivalence, configured_limit: int | None) -> int | None:
        if self._remaining is None:
            budget_cap = None
        elif self._remaining <= 0:
            budget_cap = 0
        else:
            budget_cap = floor(self._remaining / equivalence.output_to_output)
        if configured_limit is None:
            return budget_cap
        if budget_cap is None:
            return configured_limit
        return min(budget_cap, configured_limit)

    def _record(self, equivalence: OutputEquivalence, usage: ChatUsage | None) -> None:
        if usage is None:
            return
        self._charges.append(_Charge(equivalence=equivalence, usage=usage))
        if self._remaining is not None:
            self._remaining -= _internal_cost(equivalence, usage, reasoning_to_output=self._reasoning_to_output)

    def anchor(self, *, input_usage: ChatUsage | None) -> None:
        if input_usage is None or self._input is not None:
            return
        charge = next((charge for charge in self._charges if charge.usage is input_usage), None)
        if charge is None:
            raise RuntimeError("completion budget input usage was not recorded")
        self._input = charge

    def _build_usage(self, output: _Charge | None) -> ChatUsage | None:
        if not self._charges:
            return None

        anchor = self._input or output or self._charges[0]
        visible_tokens = 0 if output is None else _visible_output_tokens(output.usage)
        normalized_output_tokens = 0
        for charge in self._charges:
            if charge is output:
                normalized_output_tokens += _output_cost(
                    charge.equivalence,
                    charge.usage,
                    reasoning_to_output=self._reasoning_to_output,
                )
                continue
            normalized_output_tokens += _internal_cost(
                charge.equivalence,
                charge.usage,
                reasoning_to_output=self._reasoning_to_output,
            )

        output_tokens = max(visible_tokens, normalized_output_tokens)
        return ChatUsage(
            input_tokens=anchor.usage.input_tokens,
            cached_tokens=_cached_input_tokens(anchor.usage),
            output_tokens=output_tokens,
            reasoning_tokens=output_tokens - visible_tokens,
            total_tokens=anchor.usage.input_tokens + output_tokens,
        )

    def finish(self, *, output_usage: ChatUsage | None) -> ChatUsage | None:
        if self._finished:
            raise RuntimeError("completion budget has already been finished")

        output: _Charge | None = None
        if output_usage is not None:
            output = next((charge for charge in reversed(self._charges) if charge.usage is output_usage), None)
            if output is None:
                raise RuntimeError("completion budget output usage was not recorded")
            if self._remaining is not None:
                internal = _internal_cost(
                    output.equivalence,
                    output_usage,
                    reasoning_to_output=self._reasoning_to_output,
                )
                visible = _output_cost(
                    output.equivalence,
                    output_usage,
                    reasoning_to_output=self._reasoning_to_output,
                )
                self._remaining += internal - visible

        result = self._build_usage(output)
        self._finished = True
        return result


class BudgetedChatCompletionClient(IChatCompletionClient):
    def __init__(self, client: IChatCompletionClient, budget: CompletionBudget) -> None:
        self._client = client
        self._budget = budget

    def _limit_request(self, request: ChatCompletionRequest) -> tuple[ChatCompletionRequest, OutputEquivalence]:
        equivalence = request.output_equivalence
        if equivalence is None:
            raise ValueError("budgeted completion request requires output equivalence")
        limit = self._budget._completion_limit(equivalence, request.max_completion_tokens)
        if limit == 0:
            logger.info(
                "llm.completion.skipped",
                model=request.model,
                reason="budget_exhausted",
                remaining_budget=self._budget.remaining,
            )
            raise CompletionBudgetExhaustedError("completion budget exhausted")

        request = replace(request, max_completion_tokens=limit)
        logger.info(
            "llm.completion.started",
            max_completion_tokens=limit,
            model=request.model,
            remaining_budget=self._budget.remaining,
        )
        logger.bind(log_channel="payload").info(
            "llm.completion.request",
            model=request.model,
            request=asdict(request),
        )
        return request, equivalence

    def _record_completion(
        self,
        *,
        equivalence: OutputEquivalence,
        finish_reason: ChatFinishReason | None,
        request: ChatCompletionRequest,
        service_tier: str | None,
        usage: ChatUsage | None,
    ) -> None:
        self._budget._record(equivalence, usage)
        logger.info(
            "llm.completion.finished",
            finish_reason=finish_reason,
            model=request.model,
            remaining_budget=self._budget.remaining,
            service_tier=service_tier,
            usage=None if usage is None else asdict(usage),
        )

    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResult:
        request, equivalence = self._limit_request(request)
        result = await self._client.complete(request)
        self._record_completion(
            equivalence=equivalence,
            finish_reason=result.finish_reason,
            request=request,
            service_tier=result.service_tier,
            usage=result.usage,
        )
        return result

    def stream(self, request: ChatCompletionRequest) -> AsyncIterator[ChatCompletionDelta]:
        async def run() -> AsyncIterator[ChatCompletionDelta]:
            limited, equivalence = self._limit_request(request)
            finish_reason: ChatFinishReason | None = None
            service_tier: str | None = None
            usage: ChatUsage | None = None
            try:
                async for delta in self._client.stream(limited):
                    if delta.finish_reason is not None:
                        finish_reason = delta.finish_reason
                    if delta.service_tier is not None:
                        service_tier = delta.service_tier
                    if delta.usage is not None:
                        usage = delta.usage
                    yield delta
            finally:
                self._record_completion(
                    equivalence=equivalence,
                    finish_reason=finish_reason,
                    request=limited,
                    service_tier=service_tier,
                    usage=usage,
                )

        return run()

    async def aclose(self) -> None:
        return None


__all__ = [
    "BudgetedChatCompletionClient",
    "CompletionBudget",
    "CompletionBudgetExhaustedError",
]
