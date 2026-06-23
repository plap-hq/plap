from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor

from plap.llms.completions.chat import ChatUsage
from plap.responses.contracts import ResponseUsage, ResponseUsageInputTokensDetails, ResponseUsageOutputTokensDetails


def _cached_input_tokens(usage: ChatUsage) -> int:
    return min(usage.cached_tokens or 0, usage.input_tokens)


def _visible_output_tokens(usage: ChatUsage) -> int:
    return max(0, usage.output_tokens - (usage.reasoning_tokens or 0))


def _output_equivalent_tokens(usage: ChatUsage, *, reasoning_to_output: float) -> int:
    return _visible_output_tokens(usage) + ceil((usage.reasoning_tokens or 0) * reasoning_to_output)


def _visible_debit(pricing: object, usage: ChatUsage, *, reasoning_to_output: float) -> int:
    return ceil(pricing.output_to_output * _output_equivalent_tokens(usage, reasoning_to_output=reasoning_to_output))


def _hidden_debit(pricing: object, usage: ChatUsage, *, reasoning_to_output: float) -> int:
    cached_input = _cached_input_tokens(usage)
    uncached_input = usage.input_tokens - cached_input
    debit = (
        uncached_input * pricing.uncached_input_to_output
        + cached_input * pricing.cached_input_to_output
        + pricing.output_to_output * _output_equivalent_tokens(usage, reasoning_to_output=reasoning_to_output)
    )
    return ceil(debit)


@dataclass(frozen=True, slots=True)
class _Charge:
    pricing: object
    usage: ChatUsage
    visible: bool


@dataclass(slots=True)
class UsageLedger:
    budget: int | None
    reasoning_to_output: float
    charges: list[_Charge]
    anchor: ChatUsage | None

    def __init__(self, budget: int | None, reasoning_to_output: float) -> None:
        self.budget = budget
        self.reasoning_to_output = reasoning_to_output
        self.charges = []
        self.anchor = None

    def remaining(self) -> int | None:
        return self.budget

    def cap(self, pricing: object | None, limit: int | None) -> int | None:
        if pricing is None or self.budget is None:
            budget_cap = self.budget
        elif self.budget <= 0:
            budget_cap = 0
        else:
            budget_cap = floor(self.budget / pricing.output_to_output)
        if limit is None:
            return budget_cap
        if budget_cap is None:
            return int(limit)
        return min(budget_cap, int(limit))

    def hide(self, pricing: object, usage: ChatUsage | None) -> None:
        self._record(pricing, usage, visible=False)

    def show(self, pricing: object, usage: ChatUsage | None) -> None:
        self._record(pricing, usage, visible=True)

    def usage(self) -> ResponseUsage | None:
        anchor = self.anchor
        if anchor is None:
            return None

        visible_tokens = sum(_visible_output_tokens(charge.usage) for charge in self.charges if charge.visible)
        normalized_output_tokens = 0
        for charge in self.charges:
            if charge.visible:
                normalized_output_tokens += _visible_debit(
                    charge.pricing,
                    charge.usage,
                    reasoning_to_output=self.reasoning_to_output,
                )
                continue
            normalized_output_tokens += _hidden_debit(
                charge.pricing,
                charge.usage,
                reasoning_to_output=self.reasoning_to_output,
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

    def _record(self, pricing: object, usage: ChatUsage | None, *, visible: bool) -> None:
        if usage is None:
            return
        if self.anchor is None:
            self.anchor = usage
        if self.budget is not None:
            debit = _visible_debit(pricing, usage, reasoning_to_output=self.reasoning_to_output)
            if not visible:
                debit = _hidden_debit(pricing, usage, reasoning_to_output=self.reasoning_to_output)
            self.budget -= debit
        self.charges.append(_Charge(pricing=pricing, usage=usage, visible=visible))
