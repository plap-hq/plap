# Completion Budgeting

`max_completion_tokens` limits one model call. A response can make several model calls before producing its final answer, so
separate per-call limits do not cap the total work performed for that response.

`CompletionBudget` gives those calls one shared limit. It measures them in output-equivalent tokens, allowing input, cached
input, output, and reasoning tokens to consume the same budget at configured rates.

## Wrap a completion client

`BudgetedChatCompletionClient` applies a `CompletionBudget` to every call delegated to its base client:

```python
from plap.llms.completions import (
    BudgetedChatCompletionClient,
    CompletionBudget,
)

budget = CompletionBudget(
    max_output_tokens=4_000,
    reasoning_to_output=1.0,
)
client = BudgetedChatCompletionClient(base_client, budget)
```

The wrapper does not own `base_client`; `client.aclose()` is a no-op. The code that created the base client must close it.

plap registers a budget for each public response and makes a budgeted client available to its plugins. Use that client rather
than creating a separate budget:

```python
client = await state.svcs.aget(BudgetedChatCompletionClient)
```

## Define the cost of a call

Each request sent through the wrapper must include `OutputEquivalence`. Its rates convert the call's reported tokens into the
budget's output-token unit:

```python
from dataclasses import replace

from plap.llms.completions import OutputEquivalence

budgeted_request = replace(
    request,
    output_equivalence=OutputEquivalence(
        uncached_input_to_output=0.25,
        cached_input_to_output=0.05,
        output_to_output=1.0,
    ),
)
```

The conversion belongs to the call, because two workflows may account for the same provider model differently.
`reasoning_to_output` supplies the separate conversion for reasoning tokens reported inside output usage.

## Limit and charge calls

Before a call, the wrapper converts the remaining budget into that request's output-token scale. It sets
`max_completion_tokens` to the smaller of:

- The converted remaining budget.
- The request's existing `max_completion_tokens`, when set.

If no tokens remain, the wrapper raises `CompletionBudgetExhaustedError` without contacting the provider.

After the call, reported usage charges uncached input, cached input, visible output, and reasoning output at their configured
rates. Streaming records the latest reported usage even when iteration is cancelled or fails later.

## Build aggregate usage

The public response should report the final main completion as visible output while accounting for internal calls as reasoning
work. `anchor()` and `finish()` build that aggregate `ChatUsage`:

```python
budget.anchor(input_usage=first_main_usage)
aggregate = budget.finish(output_usage=final_main_usage)
```

`anchor()` selects the aggregate input usage; the first successful anchor is kept. `finish()` may be called once, and its
`output_usage` must be the same `ChatUsage` object previously recorded by the wrapper.

If `anchor()` has not selected an input call, `finish()` uses the final output call and then the first recorded call. It returns
`None` when no call reported usage.

The response loop performs these final steps for plugins. Standalone callers need them only when they want aggregate usage;
the wrapper enforces the budget while calls are running either way.
