"""Replay ingest truth table.

Legend

- `R(h)`: reasoning item with a hidden assistant anchor
- `R(h-empty-main)`: main-side reasoning item whose hidden anchor exists but
  has empty public content
- `R(p->M)`: reasoning patch targeting explicit assistant `M`
- `M`: explicit assistant anchor message
- `FMa`: fabricated assistant message
- `FMu`: fabricated non-assistant message (`user` / `system` / `developer`)
- `FF / FFO`: fabricated `function_call` / `function_call_output`
- `F / FO`: sealed `function_call` / `function_call_output`
- `F1 / FO1`, `F2 / FO2`: multiple sealed calls in the same anchored bundle

Cluster model

Phase 1 normalization should operate on clusters around an anchored replay
bundle, not on raw items or raw messages one-by-one.

- Plain message cluster:
  - a `user` / `system` / `developer` message
  - or an assistant message with no attached fabricated calls
- Assistant mini-turn cluster:
  - an assistant message
  - zero or more fabricated calls attached to that assistant
  - zero or more fabricated tool outputs for those fabricated calls
- Anchored replay cluster:
  - the anchor assistant
  - zero or more sealed calls
  - zero or more fabricated calls that still belong to that same anchor
  - all corresponding tool outputs
  - hidden reasoning/tool rows if the anchor came from `R`

Fabricated calls attach to the nearest preceding assistant within the local
cluster parse. If there is no such assistant, they fall back to the anchor
assistant. Once a later fabricated call or output attaches to an assistant,
that assistant cluster is indivisible for chronology purposes.

The local normalized shape for one anchored replay region is:

- `pre_clusters`
- `anchor_cluster`
- `post_clusters`

`pre_clusters` are inserted immediately before the anchor assistant.
`post_clusters` are appended after the anchored bundle settles.

Normalized form

- `A(FMa)`: assistant message cluster with no attached fabricated calls
- `A(FMa; FF,FFO)`: assistant cluster with attached fabricated mini-turn
- `P(FMu)`: plain non-assistant message cluster
- `pre=[...]`: clusters hoisted immediately before the anchor assistant
- `anchor=...`
- `slots=[...]`: assistant `tool_calls` order
- `outputs=[...]`: `role: tool` message chronology
- `post=[...]`: clusters appended after the anchored bundle settles

Invariants

- If a later `FF/FFO` attach to an earlier assistant cluster, that assistant
  cluster becomes contiguous.
- `outputs` preserve real chronology. So `F1 F2 FO2 FO1` keeps
  `outputs=[FO2,FO1]` and is not reordered.

Slot rule

- Sealed calls occupy their declared indexed segment.
- Fabricated calls attached directly to the anchor append after the
  sealed-indexed segment, in first-appearance order.

Phase 1: Anchored cases

1. Baseline anchored bundles

- Raw: `M F FO`
  Normalized: `pre=[] anchor=M slots=[F] outputs=[FO] post=[]`
  Outcome: Accept
  Notes: Explicit anchor.

- Raw: `M F1 FO1 F2 FO2`
  Normalized: `pre=[] anchor=M slots=[F1,F2] outputs=[FO1,FO2] post=[]`
  Outcome: Accept
  Notes: Multiple sealed calls.

- Raw: `M F1 F2 FO2 FO1`
  Normalized: `pre=[] anchor=M slots=[F1,F2] outputs=[FO2,FO1] post=[]`
  Outcome: Accept
  Notes: Preserve output chronology.

- Raw: `R(h) F FO`
  Normalized: `pre=[] anchor=R.hidden slots=[F] outputs=[FO] post=[]`
  Outcome: Accept
  Notes: Hidden anchor.

- Raw: `R(h-empty-main) F FO`
  Normalized: `pre=[] anchor=R.hidden(empty-main) slots=[F] outputs=[FO] post=[]`
  Outcome: Accept
  Notes: Anchored, not synthetic.

- Raw: `R(p->M) M F FO`
  Normalized: `pre=[] anchor=M slots=[F] outputs=[FO] post=[]`
  Outcome: Accept
  Notes: Patch resolves to `M`.

- Raw: `M F FO` after stripping `R(p->M)`
  Normalized: `pre=[] anchor=M slots=[F] outputs=[FO] post=[]`
  Outcome: Accept
  Notes: Sealed targeting is based on `M`, not `R+M`.

2. Pre-anchor plain message hoists

- Raw: `R(h-empty-main) FMu F FO`
  Normalized:
  `pre=[P(FMu)] anchor=R.hidden(empty-main) slots=[F] outputs=[FO] post=[]`
  Outcome: Accept
  Notes: Effective visible order `FMu F FO`.

- Raw: `R(h-empty-main) FMu FMu2 F FO`
  Normalized:
  `pre=[P(FMu),P(FMu2)] anchor=R.hidden(empty-main) slots=[F] outputs=[FO] post=[]`
  Outcome: Accept
  Notes: Preserve order.

- Raw: `R(h-empty-main) FMa F FO`
  Normalized:
  `pre=[A(FMa)] anchor=R.hidden(empty-main) slots=[F] outputs=[FO] post=[]`
  Outcome: Accept
  Notes: Assistant plain message before hidden anchor.

- Raw: `R(p->M) FMu M F FO`
  Normalized: `pre=[P(FMu)] anchor=M slots=[F] outputs=[FO] post=[]`
  Outcome: Accept
  Notes: Hoist before `M`, not before whole `R`.

- Raw: `R(p->M) FMu FMu2 M F FO`
  Normalized: `pre=[P(FMu),P(FMu2)] anchor=M slots=[F] outputs=[FO] post=[]`
  Outcome: Accept
  Notes: Preserve order.

- Raw: `R(p->M) FMa M F FO`
  Normalized: `pre=[A(FMa)] anchor=M slots=[F] outputs=[FO] post=[]`
  Outcome: Accept
  Notes: Assistant plain message before `M`.

3. Pre-anchor assistant mini-bundles

- Raw: `R(h-empty-main) FMa FF FFO F FO`
  Normalized:
  `pre=[A(FMa;FF,FFO)] anchor=R.hidden(empty-main) slots=[F] outputs=[FO] post=[]`
  Outcome: Accept
  Notes: `FF/FFO` bind to `FMa`, sealed pair binds to hidden anchor.

- Raw: `R(p->M) FMa FF FFO M F FO`
  Normalized: `pre=[A(FMa;FF,FFO)] anchor=M slots=[F] outputs=[FO] post=[]`
  Outcome: Accept
  Notes: Same, explicit anchor.

4. Pre-anchor mixed `FMa` / `FMu`

- Raw: `R(h-empty-main) FMu FMa FF FFO F FO`
  Normalized:
  `pre=[P(FMu),A(FMa;FF,FFO)] anchor=R.hidden(empty-main) slots=[F] outputs=[FO] post=[]`
  Outcome: Accept
  Notes: Plain message stays before assistant cluster.

- Raw: `R(h-empty-main) FMa FMu FF FFO F FO`
  Normalized:
  `pre=[A(FMa;FF,FFO),P(FMu)] anchor=R.hidden(empty-main) slots=[F] outputs=[FO] post=[]`
  Outcome: Accept
  Notes: `FMu` cannot split `FMa` from `FF/FFO`.

- Raw: `R(p->M) FMu FMa FF FFO M F FO`
  Normalized:
  `pre=[P(FMu),A(FMa;FF,FFO)] anchor=M slots=[F] outputs=[FO] post=[]`
  Outcome: Accept
  Notes: Explicit anchor.

- Raw: `R(p->M) FMa FMu FF FFO M F FO`
  Normalized:
  `pre=[A(FMa;FF,FFO),P(FMu)] anchor=M slots=[F] outputs=[FO] post=[]`
  Outcome: Accept
  Notes: `FMu` pushed outside attached assistant cluster.

5. Pre-anchor non-assistant plus fabricated calls that fall back to anchor

- Raw: `R(h-empty-main) FMu FF FFO F FO`
  Normalized:
  `pre=[P(FMu)] anchor=R.hidden(empty-main) slots=[F,FF] outputs=[FFO,FO] post=[]`
  Outcome: Accept
  Notes: No assistant before `FF`, so `FF/FFO` bind to anchor.

- Raw: `R(h-empty-main) FMu FMu2 FF FFO F FO`
  Normalized:
  `pre=[P(FMu),P(FMu2)] anchor=R.hidden(empty-main) slots=[F,FF] outputs=[FFO,FO] post=[]`
  Outcome: Accept
  Notes: Same fallback.

- Raw: `R(p->M) FMu FF FFO M F FO`
  Normalized: N/A
  Outcome: Reject
  Notes: No assistant exists before `M` for `FF` to bind to.

- Raw: `R(p->M) FMu FMu2 FF FFO M F FO`
  Normalized: N/A
  Outcome: Reject
  Notes: Same reason.

6. Mixed fabricated + sealed calls on the same anchor

- Raw: `R(h) FF FFO F FO`
  Normalized: `pre=[] anchor=R.hidden slots=[F,FF] outputs=[FFO,FO] post=[]`
  Outcome: Accept
  Notes: `FF/FFO` fallback to hidden anchor.

- Raw: `R(h) FF FFO F FO FF2 FFO2 F2 FO2`
  Normalized:
  `pre=[] anchor=R.hidden slots=[F,F2,FF,FF2] outputs=[FFO,FO,FFO2,FO2] post=[]`
  Outcome: Accept
  Notes: Mixed fabricated and sealed.

- Raw: `R(h-empty-main) FF FFO F FO`
  Normalized:
  `pre=[] anchor=R.hidden(empty-main) slots=[F,FF] outputs=[FFO,FO] post=[]`
  Outcome: Accept
  Notes: Same on hidden-empty main anchor.

- Raw: `R(h-empty-main) FF FFO F FO FF2 FFO2 F2 FO2`
  Normalized:
  `pre=[] anchor=R.hidden(empty-main) slots=[F,F2,FF,FF2] outputs=[FFO,FO,FFO2,FO2] post=[]`
  Outcome: Accept
  Notes: Same generalized bundle.

- Raw: `R M FF FFO F FO`
  Normalized: `pre=[] anchor=M slots=[F,FF] outputs=[FFO,FO] post=[]`
  Outcome: Accept
  Notes: `FF/FFO` fallback to explicit `M`.

- Raw: `R M FF FFO F FO FF2 FFO2 F2 FO2`
  Normalized:
  `pre=[] anchor=M slots=[F,F2,FF,FF2] outputs=[FFO,FO,FFO2,FO2] post=[]`
  Outcome: Accept
  Notes: Explicit `M`, mixed fabricated and sealed.

7. Post-anchor plain messages

- Raw: `R M FMu F FO`
  Normalized: `pre=[] anchor=M slots=[F] outputs=[FO] post=[P(FMu)]`
  Outcome: Accept
  Notes: Append after sealed bundle.

- Raw: `R M FMu FMu2 F FO`
  Normalized: `pre=[] anchor=M slots=[F] outputs=[FO] post=[P(FMu),P(FMu2)]`
  Outcome: Accept
  Notes: Preserve order.

- Raw: `R M FMa F FO`
  Normalized: `pre=[] anchor=M slots=[F] outputs=[FO] post=[A(FMa)]`
  Outcome: Accept
  Notes: Assistant plain message after bundle.

8. Post-anchor assistant mini-bundles

- Raw: `R M FMa FF FFO F FO`
  Normalized: `pre=[] anchor=M slots=[F] outputs=[FO] post=[A(FMa;FF,FFO)]`
  Outcome: Accept
  Notes: Per chosen post-anchor policy.

- Raw: `R M FMa FF FFO F1 F2 FO2 FO1`
  Normalized:
  `pre=[] anchor=M slots=[F1,F2] outputs=[FO2,FO1] post=[A(FMa;FF,FFO)]`
  Outcome: Accept
  Notes: Post-anchor assistant mini-bundle with parallel sealed calls.

9. Post-anchor mixed `FMa` / `FMu`

- Raw: `R M FMu FMa FF FFO F FO`
  Normalized:
  `pre=[] anchor=M slots=[F] outputs=[FO] post=[P(FMu),A(FMa;FF,FFO)]`
  Outcome: Accept
  Notes: Preserve cluster-start order.

- Raw: `R M FMa FMu FF FFO F FO`
  Normalized:
  `pre=[] anchor=M slots=[F] outputs=[FO] post=[A(FMa;FF,FFO),P(FMu)]`
  Outcome: Accept
  Notes: `FMu` cannot split `FMa` from `FF/FFO`.

- Raw: `R M FMa FMu FF FFO F1 F2 FO2 FO1`
  Normalized:
  `pre=[] anchor=M slots=[F1,F2] outputs=[FO2,FO1] post=[A(FMa;FF,FFO),P(FMu)]`
  Outcome: Accept
  Notes: Same with parallel sealed calls.

10. Post-anchor non-assistant plus fabricated calls that fall back to anchor

- Raw: `R M FMu FF FFO F FO`
  Normalized: `pre=[] anchor=M slots=[F,FF] outputs=[FFO,FO] post=[P(FMu)]`
  Outcome: Accept
  Notes: `FF/FFO` bind to `M`, not `FMu`.

- Raw: `R M FMu FMu2 FF FFO F FO`
  Normalized:
  `pre=[] anchor=M slots=[F,FF] outputs=[FFO,FO] post=[P(FMu),P(FMu2)]`
  Outcome: Accept
  Notes: Same.

- Raw: `R M FMu FF FFO F1 F2 FO2 FO1`
  Normalized:
  `pre=[] anchor=M slots=[F1,F2,FF] outputs=[FFO,FO2,FO1] post=[P(FMu)]`
  Outcome: Accept
  Notes: Sealed outputs preserve chronology.

- Raw: `R M FMu FF FFO F FO FF2 FFO2 F2 FO2`
  Normalized:
  `pre=[] anchor=M slots=[F,F2,FF,FF2] outputs=[FFO,FO,FFO2,FO2] post=[P(FMu)]`
  Outcome: Accept
  Notes: All fabricated calls bind to `M`.

Phase 1: Rejections that stay rejected

- Raw: `R(p->M) FF FFO M F FO`
  Normalized: N/A
  Outcome: Reject
  Notes: `FF` has no assistant anchor before `M`.

- Raw: `R(p->M) FF FFO F FO`
  Normalized: N/A
  Outcome: Reject
  Notes: Same reason.

- Raw: `R(p->M) FMu FF FFO M F FO`
  Normalized: N/A
  Outcome: Reject
  Notes: `FMu` is not an assistant anchor.

- Raw: `R(p->M) FMu FMu2 FF FFO M F FO`
  Normalized: N/A
  Outcome: Reject
  Notes: Same reason.

- Raw: `Any anchored case requiring retroactive reopen of an older explicit`
  `anchor after a newer explicit anchored bundle is committed`
  Normalized: N/A
  Outcome: Reject
  Notes: Phase 1 has no retroactive reopen.

- Raw: `Any case where M never appears for R(p->M)`
  Normalized: N/A
  Outcome: Reject
  Notes: Patch target missing.

- Raw: `Sealed hash mismatch / slot mismatch / upstream-id mismatch`
  Normalized: N/A
  Outcome: Reject
  Notes: Replay integrity failure.

- Raw: `FO with no pending F`
  Normalized: N/A
  Outcome: Reject
  Notes: Output without call.

- Raw: `Pending F with no later FO`
  Normalized: N/A
  Outcome: Reject
  Notes: Missing output.

- Raw: `Duplicate pending call ids`
  Normalized: N/A
  Outcome: Reject
  Notes: Invalid replay state.

Phase 2: Synthetic-only cases

- Raw: `F FO on main`
  Normalized:
  `pre=[] anchor=synthetic-empty-main slots=[F] outputs=[FO] post=[]`
  Outcome: Accept
  Notes: Naked main sealed replay.

- Raw: `F1 FO1 F2 FO2 on main`
  Normalized:
  `pre=[] anchor=synthetic-empty-main slots=[F1,F2] outputs=[FO1,FO2] post=[]`
  Outcome: Accept
  Notes: Synthetic main bundle.

- Raw: `F1 F2 FO2 FO1 on main`
  Normalized:
  `pre=[] anchor=synthetic-empty-main slots=[F1,F2] outputs=[FO2,FO1] post=[]`
  Outcome: Accept
  Notes: Preserve chronology.

- Raw: `Stripped R(h-empty-main) F FO -> F FO`
  Normalized:
  `pre=[] anchor=synthetic-empty-main slots=[F] outputs=[FO] post=[]`
  Outcome: Accept
  Notes: Stripped-anchor recovery.

- Raw: `R M R(h-empty-main) F FO stripped into M F FO`
  Normalized:
  `anchor=M slots=[] outputs=[] and separate anchor=synthetic-empty-main`
  `slots=[F] outputs=[FO]`
  Outcome: Accept
  Notes: Must not bind naked `F/FO` to `M`.

- Raw: `Naked reviewer F FO`
  Normalized: N/A
  Outcome: Silent discard
  Notes: No synthetic recovery off main; drop the naked sealed pair.

- Raw: `Naked arbitrator F FO`
  Normalized: N/A
  Outcome: Silent discard
  Notes: No synthetic recovery off main; drop the naked sealed pair.

Three quick validation rules

- `R(p->M) M F FO -> M F FO`: yes
- `R FM M F FO`: hoist immediately before `M`, not before whole `R`
- `F1 F2 FO2 FO1`: keep `outputs=[FO2,FO1]`, do not reorder to slot order
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from plap.errors import PlapError
from plap.keyring import SealingKeyring
from plap.responses2.contracts import (
    RequestFunctionCallItem,
    RequestFunctionCallOutputItem,
    RequestInputItem,
    RequestMessageItem,
    RequestReasoningItem,
    ResponseCreateRequest,
    SummaryTextContent,
)
from plap.responses2.ingest.ingest import ingest_response_request
from plap.responses2.ingest.models import CallID, Message, MessagePatch, ReasoningPayload, SidesUpdate, ToolCall
from plap.responses2.ingest.sealing import content_hash, content_hash_prefix, seal_call_id, seal_reasoning_payload


@dataclass(frozen=True, slots=True)
class _AcceptCase:
    items: list[RequestInputItem]
    expected_main: list[dict[str, object]]


@dataclass(frozen=True, slots=True)
class _RejectCase:
    items: list[RequestInputItem]
    expected_reason: str


def _keyring() -> SealingKeyring:
    return SealingKeyring(roots=(b"i" * 32,))


def _request(items: list[RequestInputItem]) -> ResponseCreateRequest:
    return ResponseCreateRequest(model="plap/test", input=items)


def _tool_call_model(call_id: str) -> ToolCall:
    return ToolCall(id=call_id, name="read_file", arguments='{"path":"README.md"}')


def _tool_call_value(call_id: str) -> dict[str, object]:
    return {"id": call_id, "name": "read_file", "arguments": '{"path":"README.md"}'}


def _assistant_value(content: str, *tool_call_ids: str) -> dict[str, object]:
    value: dict[str, object] = {"role": "assistant", "content": content}
    if tool_call_ids:
        value["tool_calls"] = [_tool_call_value(call_id) for call_id in tool_call_ids]
    return value


def _plain_value(role: str, content: str) -> dict[str, object]:
    return {"role": role, "content": content}


def _tool_value(call_id: str, content: str) -> dict[str, object]:
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def _assistant_item(content: str) -> RequestMessageItem:
    return RequestMessageItem(content=content, role="assistant", type="message")


def _plain_item(role: str, content: str) -> RequestMessageItem:
    return RequestMessageItem(content=content, role=role, type="message")


def _message_hash(value: dict[str, object]) -> str:
    return content_hash(Message.from_primitive(value))


def _sealed_reasoning(payload: ReasoningPayload) -> RequestReasoningItem:
    return RequestReasoningItem(
        encrypted_content=seal_reasoning_payload(payload, keyring=_keyring()),
        id=None,
        summary=[SummaryTextContent(text="sealed", type="summary_text")],
        type="reasoning",
    )


def _carrier(label: str) -> RequestReasoningItem:
    return _sealed_reasoning(
        ReasoningPayload(
            machine=[{"op": "add", "path": f"/{label}", "value": True}],
            sides=SidesUpdate(),
        )
    )


def _reasoning_hidden_main(*messages: dict[str, object]) -> RequestReasoningItem:
    return _sealed_reasoning(
        ReasoningPayload(
            machine=[],
            sides=SidesUpdate(main=[Message.from_primitive(message) for message in messages]),
        )
    )


def _reasoning_patch_main(
    target: dict[str, object],
    *,
    tool_call_ids: tuple[str, ...],
    deferred_outputs: tuple[dict[str, object], ...] = (),
) -> RequestReasoningItem:
    patch = MessagePatch(
        content_hash=_message_hash(target),
        tool_calls=[_tool_call_model(call_id) for call_id in tool_call_ids],
    )
    return _sealed_reasoning(
        ReasoningPayload(
            machine=[],
            sides=SidesUpdate(
                main=[
                    patch,
                    *[Message.from_primitive(output) for output in deferred_outputs],
                ]
            ),
        )
    )


def _sealed_main_call(anchor: dict[str, object], upstream_id: str, *, tool_call_index: int = 0) -> RequestFunctionCallItem:
    call_id = seal_call_id(
        CallID(
            side="main",
            content_hash_prefix=content_hash_prefix(_message_hash(anchor)),
            tool_call_index=tool_call_index,
            upstream_tool_call_id=upstream_id,
        ),
        keyring=_keyring(),
    )
    return RequestFunctionCallItem(
        arguments='{"path":"README.md"}',
        call_id=call_id,
        name="read_file",
        type="function_call",
    )


def _sealed_main_output(
    anchor: dict[str, object], upstream_id: str, output: str, *, tool_call_index: int = 0
) -> RequestFunctionCallOutputItem:
    call_id = seal_call_id(
        CallID(
            side="main",
            content_hash_prefix=content_hash_prefix(_message_hash(anchor)),
            tool_call_index=tool_call_index,
            upstream_tool_call_id=upstream_id,
        ),
        keyring=_keyring(),
    )
    return RequestFunctionCallOutputItem(
        call_id=call_id,
        output=output,
        type="function_call_output",
    )


def _fabricated_call(call_id: str) -> RequestFunctionCallItem:
    return RequestFunctionCallItem(
        arguments='{"path":"README.md"}',
        call_id=call_id,
        name="read_file",
        type="function_call",
    )


def _fabricated_output(call_id: str, output: str) -> RequestFunctionCallOutputItem:
    return RequestFunctionCallOutputItem(call_id=call_id, output=output, type="function_call_output")


def _main_primitives(result) -> list[dict[str, object]]:
    return [message.to_primitive() for message in result.sides.main]


ACCEPT_CASES: list[pytest.ParamSpec] = []  # type: ignore[type-arg]

m = _assistant_value("m")
ACCEPT_CASES.append(
    pytest.param(
        _AcceptCase(
            items=[_assistant_item("m"), _sealed_main_call(m, "up_0"), _sealed_main_output(m, "up_0", "fo_0")],
            expected_main=[_assistant_value("m", "up_0"), _tool_value("up_0", "fo_0")],
        ),
        id="baseline_explicit_single",
    )
)

m = _assistant_value("m")
ACCEPT_CASES.append(
    pytest.param(
        _AcceptCase(
            items=[
                _assistant_item("m"),
                _sealed_main_call(m, "up_0", tool_call_index=0),
                _sealed_main_output(m, "up_0", "fo_0", tool_call_index=0),
                _sealed_main_call(m, "up_1", tool_call_index=1),
                _sealed_main_output(m, "up_1", "fo_1", tool_call_index=1),
            ],
            expected_main=[
                _assistant_value("m", "up_0", "up_1"),
                _tool_value("up_0", "fo_0"),
                _tool_value("up_1", "fo_1"),
            ],
        ),
        id="baseline_explicit_multiple_in_order",
    )
)

m = _assistant_value("m")
ACCEPT_CASES.append(
    pytest.param(
        _AcceptCase(
            items=[
                _assistant_item("m"),
                _sealed_main_call(m, "up_0", tool_call_index=0),
                _sealed_main_call(m, "up_1", tool_call_index=1),
                _sealed_main_output(m, "up_1", "fo_1", tool_call_index=1),
                _sealed_main_output(m, "up_0", "fo_0", tool_call_index=0),
            ],
            expected_main=[
                _assistant_value("m", "up_0", "up_1"),
                _tool_value("up_1", "fo_1"),
                _tool_value("up_0", "fo_0"),
            ],
        ),
        id="baseline_explicit_multiple_output_chronology",
    )
)

h = _assistant_value("hidden")
ACCEPT_CASES.append(
    pytest.param(
        _AcceptCase(
            items=[_reasoning_hidden_main(h), _sealed_main_call(h, "up_0"), _sealed_main_output(h, "up_0", "fo_0")],
            expected_main=[_assistant_value("hidden", "up_0"), _tool_value("up_0", "fo_0")],
        ),
        id="baseline_hidden_anchor",
    )
)

h = _assistant_value("")
ACCEPT_CASES.append(
    pytest.param(
        _AcceptCase(
            items=[_reasoning_hidden_main(h), _sealed_main_call(h, "up_0"), _sealed_main_output(h, "up_0", "fo_0")],
            expected_main=[_assistant_value("", "up_0"), _tool_value("up_0", "fo_0")],
        ),
        id="baseline_hidden_empty_anchor",
    )
)

m = _assistant_value("m")
ACCEPT_CASES.append(
    pytest.param(
        _AcceptCase(
            items=[
                _reasoning_patch_main(m, tool_call_ids=("up_0",)),
                _assistant_item("m"),
                _sealed_main_call(m, "up_0"),
                _sealed_main_output(m, "up_0", "fo_0"),
            ],
            expected_main=[_assistant_value("m", "up_0"), _tool_value("up_0", "fo_0")],
        ),
        id="baseline_patch_target_explicit_anchor",
    )
)

m = _assistant_value("m")
ACCEPT_CASES.append(
    pytest.param(
        _AcceptCase(
            items=[_assistant_item("m"), _sealed_main_call(m, "up_0"), _sealed_main_output(m, "up_0", "fo_0")],
            expected_main=[_assistant_value("m", "up_0"), _tool_value("up_0", "fo_0")],
        ),
        id="baseline_stripped_patch_same_as_explicit_anchor",
    )
)

h = _assistant_value("")
ACCEPT_CASES.append(
    pytest.param(
        _AcceptCase(
            items=[
                _reasoning_hidden_main(h),
                _plain_item("user", "u1"),
                _sealed_main_call(h, "up_0"),
                _sealed_main_output(h, "up_0", "fo_0"),
            ],
            expected_main=[
                _plain_value("user", "u1"),
                _assistant_value("", "up_0"),
                _tool_value("up_0", "fo_0"),
            ],
        ),
        id="pre_anchor_plain_hidden_single",
    )
)

h = _assistant_value("")
ACCEPT_CASES.append(
    pytest.param(
        _AcceptCase(
            items=[
                _reasoning_hidden_main(h),
                _plain_item("user", "u1"),
                _plain_item("user", "u2"),
                _sealed_main_call(h, "up_0"),
                _sealed_main_output(h, "up_0", "fo_0"),
            ],
            expected_main=[
                _plain_value("user", "u1"),
                _plain_value("user", "u2"),
                _assistant_value("", "up_0"),
                _tool_value("up_0", "fo_0"),
            ],
        ),
        id="pre_anchor_plain_hidden_multiple",
    )
)

h = _assistant_value("")
ACCEPT_CASES.append(
    pytest.param(
        _AcceptCase(
            items=[
                _reasoning_hidden_main(h),
                _assistant_item("pre"),
                _sealed_main_call(h, "up_0"),
                _sealed_main_output(h, "up_0", "fo_0"),
            ],
            expected_main=[
                _assistant_value("pre"),
                _assistant_value("", "up_0"),
                _tool_value("up_0", "fo_0"),
            ],
        ),
        id="pre_anchor_plain_assistant_hidden",
    )
)

m = _assistant_value("m")
ACCEPT_CASES.append(
    pytest.param(
        _AcceptCase(
            items=[
                _reasoning_patch_main(m, tool_call_ids=("up_0",)),
                _plain_item("user", "u1"),
                _assistant_item("m"),
                _sealed_main_call(m, "up_0"),
                _sealed_main_output(m, "up_0", "fo_0"),
            ],
            expected_main=[
                _plain_value("user", "u1"),
                _assistant_value("m", "up_0"),
                _tool_value("up_0", "fo_0"),
            ],
        ),
        id="pre_anchor_plain_patch_single",
    )
)

m = _assistant_value("m")
ACCEPT_CASES.append(
    pytest.param(
        _AcceptCase(
            items=[
                _reasoning_patch_main(m, tool_call_ids=("up_0",)),
                _plain_item("user", "u1"),
                _plain_item("user", "u2"),
                _assistant_item("m"),
                _sealed_main_call(m, "up_0"),
                _sealed_main_output(m, "up_0", "fo_0"),
            ],
            expected_main=[
                _plain_value("user", "u1"),
                _plain_value("user", "u2"),
                _assistant_value("m", "up_0"),
                _tool_value("up_0", "fo_0"),
            ],
        ),
        id="pre_anchor_plain_patch_multiple",
    )
)

m = _assistant_value("m")
ACCEPT_CASES.append(
    pytest.param(
        _AcceptCase(
            items=[
                _reasoning_patch_main(m, tool_call_ids=("up_0",)),
                _assistant_item("pre"),
                _assistant_item("m"),
                _sealed_main_call(m, "up_0"),
                _sealed_main_output(m, "up_0", "fo_0"),
            ],
            expected_main=[
                _assistant_value("pre"),
                _assistant_value("m", "up_0"),
                _tool_value("up_0", "fo_0"),
            ],
        ),
        id="pre_anchor_plain_patch_assistant",
    )
)

h = _assistant_value("")
ACCEPT_CASES.append(
    pytest.param(
        _AcceptCase(
            items=[
                _reasoning_hidden_main(h),
                _assistant_item("pre"),
                _fabricated_call("fab_0"),
                _fabricated_output("fab_0", "ffo_0"),
                _sealed_main_call(h, "up_0"),
                _sealed_main_output(h, "up_0", "fo_0"),
            ],
            expected_main=[
                _assistant_value("pre", "fab_0"),
                _tool_value("fab_0", "ffo_0"),
                _assistant_value("", "up_0"),
                _tool_value("up_0", "fo_0"),
            ],
        ),
        id="pre_anchor_assistant_minibundle_hidden",
    )
)

m = _assistant_value("m")
ACCEPT_CASES.append(
    pytest.param(
        _AcceptCase(
            items=[
                _reasoning_patch_main(m, tool_call_ids=("up_0",)),
                _assistant_item("pre"),
                _fabricated_call("fab_0"),
                _fabricated_output("fab_0", "ffo_0"),
                _assistant_item("m"),
                _sealed_main_call(m, "up_0"),
                _sealed_main_output(m, "up_0", "fo_0"),
            ],
            expected_main=[
                _assistant_value("pre", "fab_0"),
                _tool_value("fab_0", "ffo_0"),
                _assistant_value("m", "up_0"),
                _tool_value("up_0", "fo_0"),
            ],
        ),
        id="pre_anchor_assistant_minibundle_patch",
    )
)

h = _assistant_value("")
ACCEPT_CASES.append(
    pytest.param(
        _AcceptCase(
            items=[
                _reasoning_hidden_main(h),
                _plain_item("user", "u1"),
                _assistant_item("pre"),
                _fabricated_call("fab_0"),
                _fabricated_output("fab_0", "ffo_0"),
                _sealed_main_call(h, "up_0"),
                _sealed_main_output(h, "up_0", "fo_0"),
            ],
            expected_main=[
                _plain_value("user", "u1"),
                _assistant_value("pre", "fab_0"),
                _tool_value("fab_0", "ffo_0"),
                _assistant_value("", "up_0"),
                _tool_value("up_0", "fo_0"),
            ],
        ),
        id="pre_anchor_mixed_hidden_user_then_assistant_cluster",
    )
)

h = _assistant_value("")
ACCEPT_CASES.append(
    pytest.param(
        _AcceptCase(
            items=[
                _reasoning_hidden_main(h),
                _assistant_item("pre"),
                _plain_item("user", "u1"),
                _fabricated_call("fab_0"),
                _fabricated_output("fab_0", "ffo_0"),
                _sealed_main_call(h, "up_0"),
                _sealed_main_output(h, "up_0", "fo_0"),
            ],
            expected_main=[
                _assistant_value("pre", "fab_0"),
                _tool_value("fab_0", "ffo_0"),
                _plain_value("user", "u1"),
                _assistant_value("", "up_0"),
                _tool_value("up_0", "fo_0"),
            ],
        ),
        id="pre_anchor_mixed_hidden_assistant_cluster_then_user",
    )
)

m = _assistant_value("m")
ACCEPT_CASES.append(
    pytest.param(
        _AcceptCase(
            items=[
                _reasoning_patch_main(m, tool_call_ids=("up_0",)),
                _plain_item("user", "u1"),
                _assistant_item("pre"),
                _fabricated_call("fab_0"),
                _fabricated_output("fab_0", "ffo_0"),
                _assistant_item("m"),
                _sealed_main_call(m, "up_0"),
                _sealed_main_output(m, "up_0", "fo_0"),
            ],
            expected_main=[
                _plain_value("user", "u1"),
                _assistant_value("pre", "fab_0"),
                _tool_value("fab_0", "ffo_0"),
                _assistant_value("m", "up_0"),
                _tool_value("up_0", "fo_0"),
            ],
        ),
        id="pre_anchor_mixed_patch_user_then_assistant_cluster",
    )
)

m = _assistant_value("m")
ACCEPT_CASES.append(
    pytest.param(
        _AcceptCase(
            items=[
                _reasoning_patch_main(m, tool_call_ids=("up_0",)),
                _assistant_item("pre"),
                _plain_item("user", "u1"),
                _fabricated_call("fab_0"),
                _fabricated_output("fab_0", "ffo_0"),
                _assistant_item("m"),
                _sealed_main_call(m, "up_0"),
                _sealed_main_output(m, "up_0", "fo_0"),
            ],
            expected_main=[
                _assistant_value("pre", "fab_0"),
                _tool_value("fab_0", "ffo_0"),
                _plain_value("user", "u1"),
                _assistant_value("m", "up_0"),
                _tool_value("up_0", "fo_0"),
            ],
        ),
        id="pre_anchor_mixed_patch_assistant_cluster_then_user",
    )
)

h = _assistant_value("")
ACCEPT_CASES.append(
    pytest.param(
        _AcceptCase(
            items=[
                _reasoning_hidden_main(h),
                _plain_item("user", "u1"),
                _fabricated_call("fab_0"),
                _fabricated_output("fab_0", "ffo_0"),
                _sealed_main_call(h, "up_0"),
                _sealed_main_output(h, "up_0", "fo_0"),
            ],
            expected_main=[
                _plain_value("user", "u1"),
                _assistant_value("", "up_0", "fab_0"),
                _tool_value("fab_0", "ffo_0"),
                _tool_value("up_0", "fo_0"),
            ],
        ),
        id="pre_anchor_non_assistant_fallback_hidden_single",
    )
)

h = _assistant_value("")
ACCEPT_CASES.append(
    pytest.param(
        _AcceptCase(
            items=[
                _reasoning_hidden_main(h),
                _plain_item("user", "u1"),
                _plain_item("user", "u2"),
                _fabricated_call("fab_0"),
                _fabricated_output("fab_0", "ffo_0"),
                _sealed_main_call(h, "up_0"),
                _sealed_main_output(h, "up_0", "fo_0"),
            ],
            expected_main=[
                _plain_value("user", "u1"),
                _plain_value("user", "u2"),
                _assistant_value("", "up_0", "fab_0"),
                _tool_value("fab_0", "ffo_0"),
                _tool_value("up_0", "fo_0"),
            ],
        ),
        id="pre_anchor_non_assistant_fallback_hidden_multiple",
    )
)

h = _assistant_value("hidden")
ACCEPT_CASES.append(
    pytest.param(
        _AcceptCase(
            items=[
                _reasoning_hidden_main(h),
                _fabricated_call("fab_0"),
                _fabricated_output("fab_0", "ffo_0"),
                _sealed_main_call(h, "up_0"),
                _sealed_main_output(h, "up_0", "fo_0"),
            ],
            expected_main=[
                _assistant_value("hidden", "up_0", "fab_0"),
                _tool_value("fab_0", "ffo_0"),
                _tool_value("up_0", "fo_0"),
            ],
        ),
        id="mixed_same_hidden_anchor_single",
    )
)

h = _assistant_value("hidden")
ACCEPT_CASES.append(
    pytest.param(
        _AcceptCase(
            items=[
                _reasoning_hidden_main(h),
                _fabricated_call("fab_0"),
                _fabricated_output("fab_0", "ffo_0"),
                _sealed_main_call(h, "up_0", tool_call_index=0),
                _sealed_main_output(h, "up_0", "fo_0", tool_call_index=0),
                _fabricated_call("fab_1"),
                _fabricated_output("fab_1", "ffo_1"),
                _sealed_main_call(h, "up_1", tool_call_index=1),
                _sealed_main_output(h, "up_1", "fo_1", tool_call_index=1),
            ],
            expected_main=[
                _assistant_value("hidden", "up_0", "up_1", "fab_0", "fab_1"),
                _tool_value("fab_0", "ffo_0"),
                _tool_value("up_0", "fo_0"),
                _tool_value("fab_1", "ffo_1"),
                _tool_value("up_1", "fo_1"),
            ],
        ),
        id="mixed_same_hidden_anchor_multiple",
    )
)

h = _assistant_value("")
ACCEPT_CASES.append(
    pytest.param(
        _AcceptCase(
            items=[
                _reasoning_hidden_main(h),
                _fabricated_call("fab_0"),
                _fabricated_output("fab_0", "ffo_0"),
                _sealed_main_call(h, "up_0"),
                _sealed_main_output(h, "up_0", "fo_0"),
            ],
            expected_main=[
                _assistant_value("", "up_0", "fab_0"),
                _tool_value("fab_0", "ffo_0"),
                _tool_value("up_0", "fo_0"),
            ],
        ),
        id="mixed_same_hidden_empty_anchor_single",
    )
)

h = _assistant_value("")
ACCEPT_CASES.append(
    pytest.param(
        _AcceptCase(
            items=[
                _reasoning_hidden_main(h),
                _fabricated_call("fab_0"),
                _fabricated_output("fab_0", "ffo_0"),
                _sealed_main_call(h, "up_0", tool_call_index=0),
                _sealed_main_output(h, "up_0", "fo_0", tool_call_index=0),
                _fabricated_call("fab_1"),
                _fabricated_output("fab_1", "ffo_1"),
                _sealed_main_call(h, "up_1", tool_call_index=1),
                _sealed_main_output(h, "up_1", "fo_1", tool_call_index=1),
            ],
            expected_main=[
                _assistant_value("", "up_0", "up_1", "fab_0", "fab_1"),
                _tool_value("fab_0", "ffo_0"),
                _tool_value("up_0", "fo_0"),
                _tool_value("fab_1", "ffo_1"),
                _tool_value("up_1", "fo_1"),
            ],
        ),
        id="mixed_same_hidden_empty_anchor_multiple",
    )
)

m = _assistant_value("m")
ACCEPT_CASES.append(
    pytest.param(
        _AcceptCase(
            items=[
                _carrier("rm_single"),
                _assistant_item("m"),
                _fabricated_call("fab_0"),
                _fabricated_output("fab_0", "ffo_0"),
                _sealed_main_call(m, "up_0"),
                _sealed_main_output(m, "up_0", "fo_0"),
            ],
            expected_main=[
                _assistant_value("m", "up_0", "fab_0"),
                _tool_value("fab_0", "ffo_0"),
                _tool_value("up_0", "fo_0"),
            ],
        ),
        id="mixed_same_explicit_anchor_single",
    )
)

m = _assistant_value("m")
ACCEPT_CASES.append(
    pytest.param(
        _AcceptCase(
            items=[
                _carrier("rm_multi"),
                _assistant_item("m"),
                _fabricated_call("fab_0"),
                _fabricated_output("fab_0", "ffo_0"),
                _sealed_main_call(m, "up_0", tool_call_index=0),
                _sealed_main_output(m, "up_0", "fo_0", tool_call_index=0),
                _fabricated_call("fab_1"),
                _fabricated_output("fab_1", "ffo_1"),
                _sealed_main_call(m, "up_1", tool_call_index=1),
                _sealed_main_output(m, "up_1", "fo_1", tool_call_index=1),
            ],
            expected_main=[
                _assistant_value("m", "up_0", "up_1", "fab_0", "fab_1"),
                _tool_value("fab_0", "ffo_0"),
                _tool_value("up_0", "fo_0"),
                _tool_value("fab_1", "ffo_1"),
                _tool_value("up_1", "fo_1"),
            ],
        ),
        id="mixed_same_explicit_anchor_multiple",
    )
)

m = _assistant_value("m")
ACCEPT_CASES.append(
    pytest.param(
        _AcceptCase(
            items=[
                _carrier("post_plain_single"),
                _assistant_item("m"),
                _plain_item("user", "u1"),
                _sealed_main_call(m, "up_0"),
                _sealed_main_output(m, "up_0", "fo_0"),
            ],
            expected_main=[_assistant_value("m", "up_0"), _tool_value("up_0", "fo_0"), _plain_value("user", "u1")],
        ),
        id="post_anchor_plain_single",
    )
)

m = _assistant_value("m")
ACCEPT_CASES.append(
    pytest.param(
        _AcceptCase(
            items=[
                _carrier("post_plain_multi"),
                _assistant_item("m"),
                _plain_item("user", "u1"),
                _plain_item("user", "u2"),
                _sealed_main_call(m, "up_0"),
                _sealed_main_output(m, "up_0", "fo_0"),
            ],
            expected_main=[
                _assistant_value("m", "up_0"),
                _tool_value("up_0", "fo_0"),
                _plain_value("user", "u1"),
                _plain_value("user", "u2"),
            ],
        ),
        id="post_anchor_plain_multiple",
    )
)

m = _assistant_value("m")
ACCEPT_CASES.append(
    pytest.param(
        _AcceptCase(
            items=[
                _carrier("post_plain_assistant"),
                _assistant_item("m"),
                _assistant_item("post"),
                _sealed_main_call(m, "up_0"),
                _sealed_main_output(m, "up_0", "fo_0"),
            ],
            expected_main=[_assistant_value("m", "up_0"), _tool_value("up_0", "fo_0"), _assistant_value("post")],
        ),
        id="post_anchor_plain_assistant",
    )
)

m = _assistant_value("m")
ACCEPT_CASES.append(
    pytest.param(
        _AcceptCase(
            items=[
                _carrier("post_mini_single"),
                _assistant_item("m"),
                _assistant_item("post"),
                _fabricated_call("fab_0"),
                _fabricated_output("fab_0", "ffo_0"),
                _sealed_main_call(m, "up_0"),
                _sealed_main_output(m, "up_0", "fo_0"),
            ],
            expected_main=[
                _assistant_value("m", "up_0"),
                _tool_value("up_0", "fo_0"),
                _assistant_value("post", "fab_0"),
                _tool_value("fab_0", "ffo_0"),
            ],
        ),
        id="post_anchor_assistant_minibundle_single",
    )
)

m = _assistant_value("m")
ACCEPT_CASES.append(
    pytest.param(
        _AcceptCase(
            items=[
                _carrier("post_mini_multi"),
                _assistant_item("m"),
                _assistant_item("post"),
                _fabricated_call("fab_0"),
                _fabricated_output("fab_0", "ffo_0"),
                _sealed_main_call(m, "up_0", tool_call_index=0),
                _sealed_main_call(m, "up_1", tool_call_index=1),
                _sealed_main_output(m, "up_1", "fo_1", tool_call_index=1),
                _sealed_main_output(m, "up_0", "fo_0", tool_call_index=0),
            ],
            expected_main=[
                _assistant_value("m", "up_0", "up_1"),
                _tool_value("up_1", "fo_1"),
                _tool_value("up_0", "fo_0"),
                _assistant_value("post", "fab_0"),
                _tool_value("fab_0", "ffo_0"),
            ],
        ),
        id="post_anchor_assistant_minibundle_multiple_sealed",
    )
)

m = _assistant_value("m")
ACCEPT_CASES.append(
    pytest.param(
        _AcceptCase(
            items=[
                _carrier("post_mixed_1"),
                _assistant_item("m"),
                _plain_item("user", "u1"),
                _assistant_item("post"),
                _fabricated_call("fab_0"),
                _fabricated_output("fab_0", "ffo_0"),
                _sealed_main_call(m, "up_0"),
                _sealed_main_output(m, "up_0", "fo_0"),
            ],
            expected_main=[
                _assistant_value("m", "up_0"),
                _tool_value("up_0", "fo_0"),
                _plain_value("user", "u1"),
                _assistant_value("post", "fab_0"),
                _tool_value("fab_0", "ffo_0"),
            ],
        ),
        id="post_anchor_mixed_user_then_assistant_cluster",
    )
)

m = _assistant_value("m")
ACCEPT_CASES.append(
    pytest.param(
        _AcceptCase(
            items=[
                _carrier("post_mixed_2"),
                _assistant_item("m"),
                _assistant_item("post"),
                _plain_item("user", "u1"),
                _fabricated_call("fab_0"),
                _fabricated_output("fab_0", "ffo_0"),
                _sealed_main_call(m, "up_0"),
                _sealed_main_output(m, "up_0", "fo_0"),
            ],
            expected_main=[
                _assistant_value("m", "up_0"),
                _tool_value("up_0", "fo_0"),
                _assistant_value("post", "fab_0"),
                _tool_value("fab_0", "ffo_0"),
                _plain_value("user", "u1"),
            ],
        ),
        id="post_anchor_mixed_assistant_cluster_then_user",
    )
)

m = _assistant_value("m")
ACCEPT_CASES.append(
    pytest.param(
        _AcceptCase(
            items=[
                _carrier("post_mixed_3"),
                _assistant_item("m"),
                _assistant_item("post"),
                _plain_item("user", "u1"),
                _fabricated_call("fab_0"),
                _fabricated_output("fab_0", "ffo_0"),
                _sealed_main_call(m, "up_0", tool_call_index=0),
                _sealed_main_call(m, "up_1", tool_call_index=1),
                _sealed_main_output(m, "up_1", "fo_1", tool_call_index=1),
                _sealed_main_output(m, "up_0", "fo_0", tool_call_index=0),
            ],
            expected_main=[
                _assistant_value("m", "up_0", "up_1"),
                _tool_value("up_1", "fo_1"),
                _tool_value("up_0", "fo_0"),
                _assistant_value("post", "fab_0"),
                _tool_value("fab_0", "ffo_0"),
                _plain_value("user", "u1"),
            ],
        ),
        id="post_anchor_mixed_assistant_cluster_then_user_parallel_sealed",
    )
)

m = _assistant_value("m")
ACCEPT_CASES.append(
    pytest.param(
        _AcceptCase(
            items=[
                _carrier("post_fallback_1"),
                _assistant_item("m"),
                _plain_item("user", "u1"),
                _fabricated_call("fab_0"),
                _fabricated_output("fab_0", "ffo_0"),
                _sealed_main_call(m, "up_0"),
                _sealed_main_output(m, "up_0", "fo_0"),
            ],
            expected_main=[
                _assistant_value("m", "up_0", "fab_0"),
                _tool_value("fab_0", "ffo_0"),
                _tool_value("up_0", "fo_0"),
                _plain_value("user", "u1"),
            ],
        ),
        id="post_anchor_non_assistant_fallback_single",
    )
)

m = _assistant_value("m")
ACCEPT_CASES.append(
    pytest.param(
        _AcceptCase(
            items=[
                _carrier("post_fallback_2"),
                _assistant_item("m"),
                _plain_item("user", "u1"),
                _plain_item("user", "u2"),
                _fabricated_call("fab_0"),
                _fabricated_output("fab_0", "ffo_0"),
                _sealed_main_call(m, "up_0"),
                _sealed_main_output(m, "up_0", "fo_0"),
            ],
            expected_main=[
                _assistant_value("m", "up_0", "fab_0"),
                _tool_value("fab_0", "ffo_0"),
                _tool_value("up_0", "fo_0"),
                _plain_value("user", "u1"),
                _plain_value("user", "u2"),
            ],
        ),
        id="post_anchor_non_assistant_fallback_multiple",
    )
)

m = _assistant_value("m")
ACCEPT_CASES.append(
    pytest.param(
        _AcceptCase(
            items=[
                _carrier("post_fallback_3"),
                _assistant_item("m"),
                _plain_item("user", "u1"),
                _fabricated_call("fab_0"),
                _fabricated_output("fab_0", "ffo_0"),
                _sealed_main_call(m, "up_0", tool_call_index=0),
                _sealed_main_call(m, "up_1", tool_call_index=1),
                _sealed_main_output(m, "up_1", "fo_1", tool_call_index=1),
                _sealed_main_output(m, "up_0", "fo_0", tool_call_index=0),
            ],
            expected_main=[
                _assistant_value("m", "up_0", "up_1", "fab_0"),
                _tool_value("fab_0", "ffo_0"),
                _tool_value("up_1", "fo_1"),
                _tool_value("up_0", "fo_0"),
                _plain_value("user", "u1"),
            ],
        ),
        id="post_anchor_non_assistant_fallback_parallel_sealed",
    )
)

m = _assistant_value("m")
ACCEPT_CASES.append(
    pytest.param(
        _AcceptCase(
            items=[
                _carrier("post_fallback_4"),
                _assistant_item("m"),
                _plain_item("user", "u1"),
                _fabricated_call("fab_0"),
                _fabricated_output("fab_0", "ffo_0"),
                _sealed_main_call(m, "up_0", tool_call_index=0),
                _sealed_main_output(m, "up_0", "fo_0", tool_call_index=0),
                _fabricated_call("fab_1"),
                _fabricated_output("fab_1", "ffo_1"),
                _sealed_main_call(m, "up_1", tool_call_index=1),
                _sealed_main_output(m, "up_1", "fo_1", tool_call_index=1),
            ],
            expected_main=[
                _assistant_value("m", "up_0", "up_1", "fab_0", "fab_1"),
                _tool_value("fab_0", "ffo_0"),
                _tool_value("up_0", "fo_0"),
                _tool_value("fab_1", "ffo_1"),
                _tool_value("up_1", "fo_1"),
                _plain_value("user", "u1"),
            ],
        ),
        id="post_anchor_non_assistant_fallback_mixed_multiple",
    )
)

m = _assistant_value("m")
ACCEPT_CASES.append(
    pytest.param(
        _AcceptCase(
            items=[
                _sealed_reasoning(
                    ReasoningPayload(
                        machine=[],
                        sides=SidesUpdate(
                            main=[
                                Message(
                                    role="assistant",
                                    content="prefix turn",
                                    tool_calls=[_tool_call_model("pref_0")],
                                ),
                                Message(role="tool", tool_call_id="pref_0", content="prefix output"),
                                MessagePatch(content_hash=_message_hash(m), tool_calls=[_tool_call_model("up_0")]),
                            ]
                        ),
                    )
                ),
                _assistant_item("m"),
                _sealed_main_call(m, "up_0"),
                _sealed_main_output(m, "up_0", "fo_0"),
            ],
            expected_main=[
                _assistant_value("prefix turn", "pref_0"),
                _tool_value("pref_0", "prefix output"),
                _assistant_value("m", "up_0"),
                _tool_value("up_0", "fo_0"),
            ],
        ),
        id="extended_prefix_closed_tool_turn_before_patch_anchor",
    )
)

h = _assistant_value("hidden")
ACCEPT_CASES.append(
    pytest.param(
        _AcceptCase(
            items=[
                _sealed_reasoning(
                    ReasoningPayload(
                        machine=[],
                        sides=SidesUpdate(
                            main=[
                                Message(
                                    role="assistant",
                                    content="prefix turn",
                                    tool_calls=[_tool_call_model("pref_0")],
                                ),
                                Message(role="tool", tool_call_id="pref_0", content="prefix output"),
                                Message.from_primitive(_assistant_value("hidden", "up_0")),
                            ]
                        ),
                    )
                ),
                _sealed_main_call(h, "up_0"),
                _sealed_main_output(h, "up_0", "fo_0"),
            ],
            expected_main=[
                _assistant_value("prefix turn", "pref_0"),
                _tool_value("pref_0", "prefix output"),
                _assistant_value("hidden", "up_0"),
                _tool_value("up_0", "fo_0"),
            ],
        ),
        id="extended_prefix_closed_tool_turn_before_hidden_anchor",
    )
)

m = _assistant_value("m")
ACCEPT_CASES.append(
    pytest.param(
        _AcceptCase(
            items=[
                _sealed_reasoning(
                    ReasoningPayload(
                        machine=[],
                        sides=SidesUpdate(
                            main=[
                                Message(
                                    role="assistant",
                                    content="prefix turn",
                                    tool_calls=[_tool_call_model("pref_0")],
                                ),
                                Message(role="tool", tool_call_id="pref_0", content="prefix output"),
                                MessagePatch(
                                    content_hash=_message_hash(m),
                                    tool_calls=[_tool_call_model("up_0"), _tool_call_model("up_hidden_1")],
                                ),
                                Message(role="tool", tool_call_id="up_hidden_1", content="hidden output"),
                            ]
                        ),
                    )
                ),
                _assistant_item("m"),
                _sealed_main_call(m, "up_0"),
                _sealed_main_output(m, "up_0", "fo_0"),
            ],
            expected_main=[
                _assistant_value("prefix turn", "pref_0"),
                _tool_value("pref_0", "prefix output"),
                _assistant_value("m", "up_0", "up_hidden_1"),
                _tool_value("up_hidden_1", "hidden output"),
                _tool_value("up_0", "fo_0"),
            ],
        ),
        id="extended_patch_anchor_with_suffix_hidden_output_after_closed_prefix",
    )
)


REJECT_CASES: list[pytest.ParamSpec] = []  # type: ignore[type-arg]

m = _assistant_value("m")
REJECT_CASES.append(
    pytest.param(
        _RejectCase(
            items=[
                _reasoning_patch_main(m, tool_call_ids=("up_0",)),
                _fabricated_call("fab_0"),
                _fabricated_output("fab_0", "ffo_0"),
                _assistant_item("m"),
                _sealed_main_call(m, "up_0"),
                _sealed_main_output(m, "up_0", "fo_0"),
            ],
            expected_reason="fabricated_function_call_without_previous_assistant",
        ),
        id="reject_patch_anchor_fabricated_before_public_target",
    )
)

REJECT_CASES.append(
    pytest.param(
        _RejectCase(
            items=[
                _reasoning_patch_main(m, tool_call_ids=("up_0",)),
                _fabricated_call("fab_0"),
                _fabricated_output("fab_0", "ffo_0"),
                _sealed_main_call(m, "up_0"),
                _sealed_main_output(m, "up_0", "fo_0"),
            ],
            expected_reason="fabricated_function_call_without_previous_assistant",
        ),
        id="reject_patch_anchor_missing_public_target_before_sealed_pair",
    )
)

REJECT_CASES.append(
    pytest.param(
        _RejectCase(
            items=[
                _reasoning_patch_main(m, tool_call_ids=("up_0",)),
                _plain_item("user", "u1"),
                _fabricated_call("fab_0"),
                _fabricated_output("fab_0", "ffo_0"),
                _assistant_item("m"),
                _sealed_main_call(m, "up_0"),
                _sealed_main_output(m, "up_0", "fo_0"),
            ],
            expected_reason="fabricated_function_call_without_previous_assistant",
        ),
        id="reject_patch_anchor_non_assistant_and_fabricated_before_public_target",
    )
)

REJECT_CASES.append(
    pytest.param(
        _RejectCase(
            items=[
                _reasoning_patch_main(m, tool_call_ids=("up_0",)),
                _plain_item("user", "u1"),
                _plain_item("user", "u2"),
                _fabricated_call("fab_0"),
                _fabricated_output("fab_0", "ffo_0"),
                _assistant_item("m"),
                _sealed_main_call(m, "up_0"),
                _sealed_main_output(m, "up_0", "fo_0"),
            ],
            expected_reason="fabricated_function_call_without_previous_assistant",
        ),
        id="reject_patch_anchor_non_assistant_multiple_and_fabricated_before_public_target",
    )
)

m1 = _assistant_value("m1")
m2 = _assistant_value("m2")
REJECT_CASES.append(
    pytest.param(
        _RejectCase(
            items=[
                _carrier("retro_1"),
                _assistant_item("m1"),
                _sealed_main_call(m1, "up_0"),
                _sealed_main_output(m1, "up_0", "fo_0"),
                _carrier("retro_2"),
                _assistant_item("m2"),
                _sealed_main_call(m2, "up_1"),
                _sealed_main_output(m2, "up_1", "fo_1"),
                _sealed_main_call(m1, "up_old"),
            ],
            expected_reason="main_retroactive_reopen_unsupported",
        ),
        id="reject_retroactive_reopen_older_explicit_anchor",
    )
)

REJECT_CASES.append(
    pytest.param(
        _RejectCase(
            items=[_reasoning_patch_main(m, tool_call_ids=("up_0",))],
            expected_reason="main_message_patch_target_missing",
        ),
        id="reject_patch_target_missing_public_message",
    )
)

m = _assistant_value("m")
wrong_anchor = _assistant_value("wrong")
REJECT_CASES.append(
    pytest.param(
        _RejectCase(
            items=[_assistant_item("m"), _sealed_main_call(wrong_anchor, "up_0")],
            expected_reason="sealed_function_call_content_hash_target_missing",
        ),
        id="reject_sealed_hash_mismatch",
    )
)

m = _assistant_value("m")
REJECT_CASES.append(
    pytest.param(
        _RejectCase(
            items=[_assistant_item("m"), _sealed_main_call(m, "up_0", tool_call_index=1)],
            expected_reason="sealed_function_call_index_not_contiguous",
        ),
        id="reject_sealed_slot_mismatch",
    )
)

m = _assistant_value("m")
REJECT_CASES.append(
    pytest.param(
        _RejectCase(
            items=[_assistant_item("m"), _sealed_main_call(m, "wrong_up"), _sealed_main_output(m, "up_0", "fo_0")],
            expected_reason="sealed_function_call_output_upstream_id_mismatch",
        ),
        id="reject_sealed_upstream_id_mismatch",
    )
)

m = _assistant_value("m")
REJECT_CASES.append(
    pytest.param(
        _RejectCase(
            items=[_assistant_item("m"), _sealed_main_output(m, "up_0", "fo_0")],
            expected_reason="function_call_output_without_pending_function_call",
        ),
        id="reject_output_without_pending_call",
    )
)

m = _assistant_value("m")
REJECT_CASES.append(
    pytest.param(
        _RejectCase(
            items=[_assistant_item("m"), _sealed_main_call(m, "up_0")],
            expected_reason="function_call_missing_function_call_output",
        ),
        id="reject_pending_call_without_output",
    )
)

m = _assistant_value("m")
REJECT_CASES.append(
    pytest.param(
        _RejectCase(
            items=[_assistant_item("m"), _sealed_main_call(m, "up_0"), _sealed_main_call(m, "up_0")],
            expected_reason="duplicate_pending_function_call",
        ),
        id="reject_duplicate_pending_call_ids",
    )
)


@pytest.mark.parametrize("case", ACCEPT_CASES)
async def test_phase1_truth_table_accepts(case: _AcceptCase) -> None:
    result = await ingest_response_request(_request(case.items), keyring=_keyring())

    assert _main_primitives(result) == case.expected_main


@pytest.mark.parametrize("case", REJECT_CASES)
async def test_phase1_truth_table_rejects(case: _RejectCase) -> None:
    with pytest.raises(PlapError) as exc_info:
        await ingest_response_request(_request(case.items), keyring=_keyring())

    assert exc_info.value.private is not None
    assert exc_info.value.private.reason == case.expected_reason
