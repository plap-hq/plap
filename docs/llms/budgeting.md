# Completion Budgeting

One response can make several model calls: a main completion, a private review, a summary, and correction attempts. Limiting
each call separately does not limit their combined cost. `CompletionBudget` accounts for all calls in one output-equivalent
token budget.

## Wrap a completion client

`BudgetedChatCompletionClient` implements the normal completion-client interface and delegates calls to another client:

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

The wrapper does not close `base_client`; its `aclose()` method is a no-op. The code that created the base client remains
responsible for closing it.

Inside a response plugin, plap has already created this wrapper. Get it from state instead of constructing another budget:

```python
client = await state.svcs.aget(BudgetedChatCompletionClient)
```

## Define the cost of a call

Different model calls may assign different relative costs to input, cached input, and output tokens. Every request sent
through the budgeted client must include `OutputEquivalence`:

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

`OutputEquivalence` converts provider token usage into the shared output-token unit. It belongs to the call's policy, not to
the provider model globally; two workflows can account for the same model differently.

`reasoning_to_output` performs a second conversion for reasoning tokens reported inside output usage.

## Limit and charge calls

Before each call, the wrapper converts the remaining budget into that request's output-token scale. The resulting
`max_completion_tokens` is the smaller of:

- The converted remaining budget.
- The request's existing `max_completion_tokens`, when set.

If no tokens remain, the wrapper raises `CompletionBudgetExhaustedError` without contacting the provider.

After the call, reported usage charges uncached input, cached input, visible output, and reasoning output at their configured
rates. Streaming records the latest usage even when iteration is cancelled or fails after a usage delta was received.

## Build aggregate usage

`finish()` converts recorded calls into one `ChatUsage` value. The final public completion can be identified so its visible
output remains visible while other charges are represented as reasoning output:

```python
budget.anchor(input_usage=first_main_usage)
aggregate = budget.finish(output_usage=final_main_usage)
```

`anchor()` selects the input usage reported for the aggregate. The first successful anchor is kept. `finish()` may be called
once and requires `output_usage` to be the same `ChatUsage` object recorded by the wrapper.

If there is no explicit input anchor, `finish()` uses the final output call, then the first recorded call. If no call
reported usage, it returns `None`.

The response loop performs the anchor and finish steps for plugins. Standalone library users need them only when they want
the aggregate `ChatUsage`; the wrapper enforces its remaining budget while calls are running.
