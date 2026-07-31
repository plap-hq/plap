"""Budget every completion that contributes to one response.

Each budgeted client receives the config field for the work it performs. Every
completion starts as internal work; ``finish`` reclassifies the final main usage
as visible response output.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import asdict, dataclass, replace
from math import ceil, floor

import structlog

from plap.config import CueBox
from plap.llms.completions.chat import (
    ChatCompletionDelta,
    ChatCompletionRequest,
    ChatCompletionResult,
    ChatFinishReason,
    ChatUsage,
    IChatCompletionClient,
)
from plap.responses.contracts import ResponseUsage, ResponseUsageInputTokensDetails, ResponseUsageOutputTokensDetails

logger = structlog.stdlib.get_logger(__name__)


def _cached_input_tokens(usage: ChatUsage) -> int:
    return min(usage.cached_tokens or 0, usage.input_tokens)


def _visible_output_tokens(usage: ChatUsage) -> int:
    return max(0, usage.output_tokens - (usage.reasoning_tokens or 0))


def _output_equivalent_tokens(usage: ChatUsage, *, reasoning_to_output: float) -> int:
    return _visible_output_tokens(usage) + ceil((usage.reasoning_tokens or 0) * reasoning_to_output)


def _response_output_cost(equivalence: object, usage: ChatUsage, *, reasoning_to_output: float) -> int:
    return ceil(equivalence.output_to_output * _output_equivalent_tokens(usage, reasoning_to_output=reasoning_to_output))


def _internal_cost(equivalence: object, usage: ChatUsage, *, reasoning_to_output: float) -> int:
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
    equivalence: object
    main: bool
    usage: ChatUsage


class ResponseBudgetExhaustedError(Exception):
    def __init__(self, *, last_service_tier: str | None) -> None:
        super().__init__("response output budget exhausted")
        self.last_service_tier = last_service_tier


class ResponseBudget:
    def __init__(self, config: CueBox, max_output_tokens: int | None) -> None:
        self._config = config
        self._main = config.main
        self._remaining = max_output_tokens
        self._reasoning_to_output = float(config.reasoning_to_output)
        self._charges: list[_Charge] = []
        self._finished = False
        self._service_tiers: dict[int, str] = {}

    @property
    def remaining(self) -> int | None:
        return self._remaining

    def _equivalence(self, field: object) -> object:
        equivalence = getattr(field, "output_equivalence", None)
        if equivalence is None:
            raise ValueError("completion config has no output equivalence")
        return equivalence

    def _completion_limit(self, field: object, configured_limit: int | None) -> int | None:
        equivalence = self._equivalence(field)
        if self._remaining is None:
            budget_cap = None
        elif self._remaining <= 0:
            budget_cap = 0
        else:
            budget_cap = floor(self._remaining / equivalence.output_to_output)
        if configured_limit is None:
            return budget_cap
        if budget_cap is None:
            return int(configured_limit)
        return min(budget_cap, int(configured_limit))

    def _record(self, field: object, usage: ChatUsage | None, *, service_tier: str | None) -> None:
        if service_tier is not None:
            self._service_tiers[id(field)] = service_tier
        if usage is None:
            return
        equivalence = self._equivalence(field)
        self._charges.append(_Charge(equivalence=equivalence, main=field is self._main, usage=usage))
        if self._remaining is not None:
            self._remaining -= _internal_cost(equivalence, usage, reasoning_to_output=self._reasoning_to_output)

    def _last_service_tier(self, field: object) -> str | None:
        return self._service_tiers.get(id(field))

    def _build_usage(self, output: _Charge | None) -> ResponseUsage | None:
        if not self._charges:
            return None

        anchor_charge = next(
            (charge for charge in self._charges if charge.main),
            self._charges[0],
        )
        anchor = anchor_charge.usage

        visible_tokens = 0 if output is None else _visible_output_tokens(output.usage)
        normalized_output_tokens = 0
        for charge in self._charges:
            if charge is output:
                normalized_output_tokens += _response_output_cost(
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
        reasoning_tokens = output_tokens - visible_tokens
        return ResponseUsage(
            input_tokens=anchor.input_tokens,
            input_tokens_details=ResponseUsageInputTokensDetails(cached_tokens=_cached_input_tokens(anchor)),
            output_tokens=output_tokens,
            output_tokens_details=ResponseUsageOutputTokensDetails(reasoning_tokens=reasoning_tokens),
            total_tokens=anchor.input_tokens + output_tokens,
        )

    def finish(self, usage: ChatUsage | None = None) -> ResponseUsage | None:
        if self._finished:
            raise RuntimeError("response budget has already been finished")

        output: _Charge | None = None
        if usage is not None:
            output = next((charge for charge in reversed(self._charges) if charge.usage is usage), None)
            if output is None:
                raise RuntimeError("response output usage was not recorded")
            if not output.main:
                raise RuntimeError("response output usage was not produced by the main completion")
            if self._remaining is not None:
                internal = _internal_cost(
                    output.equivalence,
                    usage,
                    reasoning_to_output=self._reasoning_to_output,
                )
                response_cost = _response_output_cost(
                    output.equivalence,
                    usage,
                    reasoning_to_output=self._reasoning_to_output,
                )
                self._remaining += internal - response_cost

        result = self._build_usage(output)
        self._finished = True
        return result


class _BudgetedChatCompletionClient(IChatCompletionClient):
    def __init__(self, client: IChatCompletionClient, budget: ResponseBudget, field: object) -> None:
        budget._equivalence(field)
        self._client = client
        self._budget = budget
        self._field = field

    def _limit_request(self, request: ChatCompletionRequest) -> ChatCompletionRequest:
        limit = self._budget._completion_limit(self._field, request.max_completion_tokens)
        if limit == 0:
            logger.info(
                "response.completion.skipped",
                model=request.model,
                reason="budget_exhausted",
                remaining_budget=self._budget.remaining,
            )
            raise ResponseBudgetExhaustedError(last_service_tier=self._budget._last_service_tier(self._field))

        request = replace(request, max_completion_tokens=limit)
        logger.info(
            "response.completion.started",
            max_completion_tokens=limit,
            model=request.model,
            remaining_budget=self._budget.remaining,
        )
        logger.bind(log_channel="payload").info(
            "response.completion.request",
            model=request.model,
            request=asdict(request),
        )
        return request

    def _record_completion(
        self,
        *,
        finish_reason: ChatFinishReason | None,
        request: ChatCompletionRequest,
        service_tier: str | None,
        usage: ChatUsage | None,
    ) -> None:
        self._budget._record(self._field, usage, service_tier=service_tier)
        logger.info(
            "response.completion.finished",
            finish_reason=finish_reason,
            model=request.model,
            remaining_budget=self._budget.remaining,
            service_tier=service_tier,
            usage=None if usage is None else asdict(usage),
        )

    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResult:
        request = self._limit_request(request)
        result = await self._client.complete(request)
        self._record_completion(
            finish_reason=result.finish_reason,
            request=request,
            service_tier=result.service_tier,
            usage=result.usage,
        )
        return result

    def stream(self, request: ChatCompletionRequest) -> AsyncIterator[ChatCompletionDelta]:
        async def run() -> AsyncIterator[ChatCompletionDelta]:
            limited = self._limit_request(request)
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
                    finish_reason=finish_reason,
                    request=limited,
                    service_tier=service_tier,
                    usage=usage,
                )

        return run()

    async def aclose(self) -> None:
        await self._client.aclose()


def budgeted(client: IChatCompletionClient, budget: ResponseBudget, field: object) -> IChatCompletionClient:
    return _BudgetedChatCompletionClient(client, budget, field)


__all__ = ["ResponseBudget", "ResponseBudgetExhaustedError", "budgeted"]
