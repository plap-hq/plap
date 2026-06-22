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


def _output_debit(config: object, usage: ChatUsage, *, reasoning_to_output: float) -> int:
    out_ratio = config.output_to_output
    return ceil(out_ratio * _output_equivalent_tokens(usage, reasoning_to_output=reasoning_to_output))


def _hidden_debit(config: object, usage: ChatUsage, *, reasoning_to_output: float) -> int:
    cached_input = _cached_input_tokens(usage)
    uncached_input = usage.input_tokens - cached_input
    debit = (
        uncached_input * config.uncached_input_to_output
        + cached_input * config.cached_input_to_output
        + config.output_to_output * _output_equivalent_tokens(usage, reasoning_to_output=reasoning_to_output)
    )
    return ceil(debit)


def _build_response_usage(
    *,
    input_tokens: int,
    cached_tokens: int,
    visible_tokens: int,
    normalized_output_tokens: int,
) -> ResponseUsage:
    output_tokens = max(visible_tokens, normalized_output_tokens)
    reasoning_tokens = output_tokens - visible_tokens
    return ResponseUsage(
        input_tokens=input_tokens,
        input_tokens_details=ResponseUsageInputTokensDetails(cached_tokens=cached_tokens),
        output_tokens=output_tokens,
        output_tokens_details=ResponseUsageOutputTokensDetails(reasoning_tokens=reasoning_tokens),
        total_tokens=input_tokens + output_tokens,
    )


@dataclass(slots=True)
class UsageLedger:
    budget: int | None
    reasoning_to_output: float
    hidden: list[tuple[object, ChatUsage]]
    output: list[tuple[object, ChatUsage]]
    hidden_output: list[int]
    input_anchor: ChatUsage | None

    def __init__(self, budget: int | None, reasoning_to_output: float) -> None:
        self.budget = budget
        self.reasoning_to_output = reasoning_to_output
        self.hidden = []
        self.output = []
        self.hidden_output = []
        self.input_anchor = None

    def remaining(self) -> int | None:
        return self.budget

    def budget_cap_for(self, config: object | None) -> int | None:
        if config is None or self.budget is None:
            return self.budget
        if self.budget <= 0:
            return 0
        return floor(self.budget / config.output_to_output)

    def completion_cap_for(self, budget_config: object | None, completion_config: object | None) -> int | None:
        value = self.budget_cap_for(budget_config)
        if completion_config is None:
            return value
        max_tokens = completion_config.max_completion_tokens
        if max_tokens is None:
            return value
        if value is None:
            return int(max_tokens)
        return min(value, int(max_tokens))

    def record_hidden(self, config: object, usage: ChatUsage | None) -> int | None:
        if usage is None:
            return None
        if self.budget is not None:
            self.budget -= _hidden_debit(config, usage, reasoning_to_output=self.reasoning_to_output)
        self.hidden.append((config, usage))
        return len(self.hidden) - 1

    def record_output(self, config: object, usage: ChatUsage | None) -> None:
        if usage is None:
            return
        if self.budget is not None:
            self.budget -= _output_debit(config, usage, reasoning_to_output=self.reasoning_to_output)
        self.output.append((config, usage))

    def promote_hidden_to_output(self, index: int) -> None:
        if index < 0 or index >= len(self.hidden):
            raise IndexError(index)
        if index in self.hidden_output:
            raise ValueError("hidden usage is already visible output")
        self.hidden_output.append(index)

    def set_input_anchor(self, usage: ChatUsage | None) -> None:
        if usage is None:
            return
        if self.input_anchor is not None:
            raise ValueError("input usage anchor is already set")
        self.input_anchor = usage

    def to_response_usage(self) -> ResponseUsage | None:
        if self.input_anchor is None:
            return None

        visible_tokens = sum(_visible_output_tokens(usage) for _, usage in self.output)
        visible_tokens += sum(_visible_output_tokens(self.hidden[index][1]) for index in self.hidden_output)
        normalized_output_tokens = sum(
            _output_debit(config, usage, reasoning_to_output=self.reasoning_to_output) for config, usage in self.output
        )
        normalized_output_tokens += sum(
            _hidden_debit(config, usage, reasoning_to_output=self.reasoning_to_output) for config, usage in self.hidden
        )
        return _build_response_usage(
            input_tokens=self.input_anchor.input_tokens,
            cached_tokens=_cached_input_tokens(self.input_anchor),
            visible_tokens=visible_tokens,
            normalized_output_tokens=normalized_output_tokens,
        )
