"""Half-chained response replay truth table.

Purpose
-------

This file is the readable, executable specification for request ingestion. It
describes the state lanes, the positional main reducer, and the interactions
between reasoning, public messages, and tool items. A row is useful only when a
reader can derive both the accepted history and the rejection reason without
reading the reducer implementation.

The table covers replay semantics from PLAN.md sections 28 through 31. Producer
selection and streamed lineage are not replay-table concerns; their required
rows live in test_responses_runtime.py and test_responses_streaming.py.

Notation
--------

- `C`: full compaction checkpoint.
- `U`: standalone public user message.
- `Rk(...)`: reasoning checkpoint. Its reasoning predecessor is always null.
- `Rp(...)`: reasoning patch. It references the preceding reasoning ID.
- `@C`: every reasoning item also references its current compaction anchor C.
- `A`: complete authenticated assistant stored inside reasoning main data.
- `P(A)`: terminal MessagePatch carrying complete authenticated assistant A.
- `R(A)`: reasoning appends A directly as a hidden assistant position.
- `M`: standalone public assistant message.
- `FMa`: fabricated public assistant message used to emphasize position.
- `FMu`: fabricated user, system, or developer message.
- `FC` / `FCO`: function_call and function_call_output request items.
- `SFC`: FC whose call ID opens a sealed thread declaration.
- `FFC`: FC whose call ID is unsealed and therefore fabricated.
- `D`: memory application state.
- `S[x]`: complete non-main history for thread x.
- `dD` / `dS[x]`: JSON patches for memory state or thread x.

There are three independent persistence lanes:

1. Memory state is replaced by Rk and JSON-patched by Rp.
2. Non-main threads are replaced as a complete map by Rk and individually
   JSON-patched by Rp.
3. Main is never in either reasoning state variant. It is reconstructed by one
   positional reducer from the common append-only `main` update and standalone
   input items.

Compaction is stronger than reasoning: C replaces all three lanes, including
main, establishes the authenticated main tail, clears reasoning lineage, and
becomes the required compaction anchor for every following reasoning item.

Boundary state machine
----------------------

Replay carries `(last_reasoning_id, last_compaction_id, checkpoint_required)`.

Raw transition                 Required result
-----------------------------  -----------------------------------------------
empty -> Rp(null, @null)       accept a root patch
C -> Rp(null, @C)              accept a post-compaction root patch
C -> Rp(null, @null)           reject: compaction predecessor mismatch
C2 -> Rp(..., @C1)             reject: stale compaction predecessor
U -> Rp                        reject: reasoning_checkpoint_required
U -> Rk(null, @current)        replace D, active, and every non-main thread
Rp/Rk -> Rp(previous=id)       accept and advance the chain
Rp/Rk -> Rp(wrong predecessor) reject: previous reasoning ID mismatch
no U -> Rk                     reject: reasoning_checkpoint_unexpected
U -> Rk(non-null predecessor)  reject: checkpoint has predecessor
... -> U -> Rk                 start a new chain root again

Only a standalone public U creates this boundary. A role=user message sealed
inside reasoning main is data in the append lane and does not request Rk.

An Rk transaction preserves reconstructed main, replaces D and active exactly,
removes omitted non-main threads, and preserves explicitly empty non-main threads.
Main is rejected structurally if placed in Rk.threads or Rp.threads.

Positional main model
---------------------

Main replay stores plain transcript nodes and assistant bundles. An assistant
bundle contains:

- one private authenticated assistant representation;
- an optional public assistant projection;
- calls currently owned by this assistant;
- outputs for those calls in observed output chronology.

The rendered bundle is always contiguous:

    assistant(tool_calls in declaration order)
    tool(output in observed chronology)
    tool(...)

A bundle does not own user, system, or developer nodes. Those nodes can be
pushed after a tool bundle by canonical rendering, but they are never moved as
part of source timewarp.

Authenticated tail
------------------

The authenticated tail is the latest assistant introduced by C, a full
assistant in reasoning main, or P(A). A fabricated public assistant never
becomes authenticated merely because it appears in input.

P(A) compares complete sealed A only with this one authenticated tail. It does
not compare A with public M, with arbitrary older assistants, or by public
content hash. Equal public messages remain distinct transcript occurrences.

MessagePatch modes
------------------

`content.assistant_output(A)` chooses one of two modes:

Patch form       State after reasoning       What resolves it
---------------  --------------------------  -------------------------------
public P(A)      awaiting public assistant   the next M or FMa
output-empty P(A) hidden immediately          nothing; it is already an owner

For public P(A), FMu is transparent and does not consume the patch. The first
assistant does consume it:

    R(P(A)) FMu M
      -> FMu, M(public fields from M; private reasoning from A)

The consuming assistant's content and refusal are authoritative. Sealed A's
content and refusal exist only for exact source selection. If no assistant
arrives, or an FC arrives before one, replay rejects with
`main_message_patch_target_missing`.

Output-empty P(A) resolves as a hidden position and may own calls immediately:

    R(P(A_empty_with_call)) SFC FCO
      -> hidden A(with call), tool output

Direct R(A) is also hidden immediately, even when A contains text. It never
requires a redundant public M.

Call lifecycle and owner selection
----------------------------------

Authenticated declarations begin DECLARED. FC changes a declaration to OPEN;
FCO changes it to CLOSED. Fabricated calls begin OPEN. Ordinary transcript
messages may appear while calls are OPEN; canonical rendering places each
eventual output inside its owner bundle. OPEN calls must be closed before a
reasoning boundary or replay completion. A DECLARED call may remain parked only
while its thread is inactive.

On main, FC chooses the nearest preceding eligible assistant position:

- FMu is skipped.
- a later FMa wins over an earlier hidden or patched assistant;
- hidden R(A) and output-empty P(A) are eligible;
- public P(A) awaiting an assistant is not eligible;
- claiming a sealed declaration may move that declaration to the positional
  owner, but its authenticated name and arguments are retained.

Representative rows:

Raw input                       Canonical main / result
------------------------------  ----------------------------------------------
R(P(A)) M SFC FCO               M(private from A, owns SFC), FCO
R(P(A)) FMa M SFC FCO           FMa(private from A), M(owns SFC), FCO
R(P(A)) FMu M                   FMu, M(private from A)
R(A) SFC FCO                    hidden A(owns SFC), FCO
R(A) FMa SFC FCO                hidden A, FMa(owns rehomed SFC), FCO
R(P(A_empty)) SFC FCO           hidden A(owns SFC), FCO
M FMu FFC FCO                   M(owns FFC), FCO, FMu
M FMa FFC FCO                   M, FMa(owns FFC), FCO
M FC1 FC2 FCO2 FCO1             M(FC1, FC2), FCO2, FCO1
M FC1 FMu FCO1                  M(FC1), FCO1, FMu
M FC1 FMa FCO1                  M(FC1), FCO1, FMa
M FC1 FCO1 FMa FC2 FCO2         M(FC1), FCO1, FMa(FC2), FCO2
M FC1 FMa FC2 FCO2 FCO1         M(FC1), FCO1, FMa(FC2), FCO2

Declared calls can cross FMu before they are opened. This is required for an
inactive source to be relocated later. Open calls may cross messages, but not
reasoning, compaction, or replay completion.

Timewarp
--------

When P(A) exactly matches the authenticated tail, the live source bundle is
removed from its old position and inserted at the patch position. The move
includes its assistant, hidden settlements, and fabricated calls/outputs that
were attached to that source. It excludes FMu and unrelated assistant bundles.

    R1(A) FMu RN(P(A))
      -> FMu, relocated A

    R1(A) FFC FCO FMu RN(P(A)) SFC FCO
      -> FMu, relocated A(with FFC and SFC), FCO(FFC), FCO(SFC)

If the authenticated tail does not equal A, that tail remains in place and a
new bundle is staged from P(A). This is also the fallback after compaction
slicing removes the original source.

A user boundary has one special parked-main interaction. U temporarily defers
interrupting inactive DECLARED main calls so the required Rk can relocate an
exact source with P(A). If the checkpoint does not match that source, replay
closes the parked declarations with the user-interruption output before
applying the checkpoint main update.

Non-main isolation
------------------

Non-main threads do not use positional MessagePatch semantics. Their complete
checkpoint histories or patched histories authenticate declarations. A sealed
non-main FC/FCO pair opens and closes only that thread's tracker and never enters
main nodes, owner selection, or timewarp.

- active non-main declaration without FC rejects;
- opened non-main call without FCO rejects;
- FC for an inactive known declaration rejects;
- inactive declarations may remain parked;
- a sealed pair with no surviving non-main declaration is stale and discarded;
- main timewarp cannot rehome a non-main declaration.

Coverage map
------------

The tests below retain named rows for state replacement and error contracts,
then exercise the complete positional grammar: projection, direct hidden
positions, parking/reactivation, hidden settlement, fabricated work, call
rehoming, open-call interleaving, timewarp, and non-main isolation. Runtime
producer tests separately prove Rk/Rp selection, R then optional M then FC
output order, delayed call-only activation, and checkpoint exclusion of main.
Streaming tests prove null checkpoint lineage, patch chaining, stable draft
variant, and cancellation behavior. Advisor and vision suites prove downstream
plugin behavior.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from plap.errors import PlapError
from plap.keyring import SealingKeyring
from plap.responses.contracts import (
    OutputRefusalContent,
    OutputTextContent,
    RequestCompactionItem,
    RequestFunctionCallItem,
    RequestFunctionCallOutputItem,
    RequestInputItem,
    RequestMessageItem,
    RequestReasoningItem,
    ResponseCreateRequest,
)
from plap.responses.ingest.ingest import ingest_response_request
from plap.responses.ingest.models import (
    CallID,
    CompactedMainTail,
    CompactionPayload,
    HiddenMainTail,
    Message,
    MessagePatch,
    PublicMainTail,
    ReasoningCheckpoint,
    ReasoningPatch,
    ReasoningPayload,
    Threads,
    ToolCall,
)
from plap.responses.ingest.sealing import seal_call_id, seal_compaction_payload, seal_reasoning_payload


def _keyring() -> SealingKeyring:
    return SealingKeyring(roots=(b"i" * 32,))


def _thread_codes() -> dict[str, int]:
    return {"main": 0, "reviewer": 1024, "arbitrator": 1025}


def _tool_call(call_id: str, *, name: str = "read_file") -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments='{"path":"README.md"}')


def _message(content: str, *, role: str = "assistant") -> RequestMessageItem:
    return RequestMessageItem(content=content, role=role, type="message")


def _function_call(call_id: str, *, name: str = "read_file") -> RequestFunctionCallItem:
    return RequestFunctionCallItem(
        arguments='{"path":"README.md"}',
        call_id=call_id,
        name=name,
        type="function_call",
    )


def _function_output(call_id: str, output: str = "result") -> RequestFunctionCallOutputItem:
    return RequestFunctionCallOutputItem(call_id=call_id, output=output, type="function_call_output")


def _sealed_call_id(thread: str, upstream_call_id: str) -> str:
    return seal_call_id(
        CallID(thread=thread, upstream_tool_call_id=upstream_call_id),
        keyring=_keyring(),
        thread_codes=_thread_codes(),
    )


def _checkpoint(
    payload_id: str,
    *,
    memory: dict[str, object] | None = None,
    active: set[str] | None = None,
    threads: dict[str, list[Message]] | None = None,
    main: list[Message | MessagePatch] | None = None,
    previous_reasoning_id: str | None = None,
    previous_compaction_id: str | None = None,
) -> ReasoningPayload:
    return ReasoningPayload(
        id=payload_id,
        previous_reasoning_id=previous_reasoning_id,
        previous_compaction_id=previous_compaction_id,
        state=ReasoningCheckpoint(
            memory={} if memory is None else memory,
            active={"main"} if active is None else active,
            threads={} if threads is None else threads,
        ),
        main=[] if main is None else main,
    )


def _patch(
    payload_id: str,
    *,
    previous_reasoning_id: str | None = None,
    previous_compaction_id: str | None = None,
    memory: list[dict[str, object]] | None = None,
    active: set[str] | None = None,
    threads: dict[str, list[dict[str, object]]] | None = None,
    main: list[Message | MessagePatch] | None = None,
) -> ReasoningPayload:
    return ReasoningPayload(
        id=payload_id,
        previous_reasoning_id=previous_reasoning_id,
        previous_compaction_id=previous_compaction_id,
        state=ReasoningPatch(
            memory=[] if memory is None else memory,
            active=active,
            threads={} if threads is None else threads,
        ),
        main=[] if main is None else main,
    )


def _sealed_reasoning(payload: ReasoningPayload) -> RequestReasoningItem:
    return RequestReasoningItem(
        encrypted_content=seal_reasoning_payload(payload, keyring=_keyring()),
        id=payload.id,
        summary=[],
        type="reasoning",
    )


def _sealed_compaction(payload: CompactionPayload) -> RequestCompactionItem:
    return RequestCompactionItem(
        encrypted_content=seal_compaction_payload(payload, keyring=_keyring()),
        id=payload.id,
        type="compaction",
    )


def _request(items: Sequence[RequestInputItem]) -> ResponseCreateRequest:
    return ResponseCreateRequest(model="plap/test", input=list(items))


async def _ingest(items: Sequence[RequestInputItem]):
    return await ingest_response_request(_request(items), keyring=_keyring(), thread_codes=_thread_codes())


def _private_reasoning(message: Message, public_content: str) -> Message:
    return Message(
        role="assistant",
        content=public_content,
        name=message.name,
        refusal=message.refusal,
        reasoning_content=message.reasoning_content,
        memory=message.memory,
    )


def _assert_reason(exc_info: pytest.ExceptionInfo[PlapError], reason: str) -> None:
    assert exc_info.value.private is not None
    assert exc_info.value.private.reason == reason


async def test_user_requires_checkpoint_and_checkpoint_starts_patch_chain() -> None:
    checkpoint = _checkpoint("rs_checkpoint", memory={"step": 1})
    patch = _patch(
        "rs_patch",
        previous_reasoning_id=checkpoint.id,
        memory=[{"op": "add", "path": "/tool_step", "value": 2}],
    )

    result = await _ingest([_message("u", role="user"), _sealed_reasoning(checkpoint), _sealed_reasoning(patch)])

    assert result.memory == {"step": 1, "tool_step": 2}
    assert result.last_reasoning_id == patch.id
    assert result.checkpoint_required is False


async def test_patch_after_user_is_rejected() -> None:
    with pytest.raises(PlapError) as exc_info:
        await _ingest([_message("u", role="user"), _sealed_reasoning(_patch("rs_patch"))])

    _assert_reason(exc_info, "reasoning_checkpoint_required")


async def test_checkpoint_without_user_is_rejected() -> None:
    with pytest.raises(PlapError) as exc_info:
        await _ingest([_sealed_reasoning(_checkpoint("rs_checkpoint"))])

    _assert_reason(exc_info, "reasoning_checkpoint_unexpected")


async def test_checkpoint_with_predecessor_is_rejected() -> None:
    checkpoint = _checkpoint("rs_checkpoint", previous_reasoning_id="rs_old")

    with pytest.raises(PlapError) as exc_info:
        await _ingest([_message("u", role="user"), _sealed_reasoning(checkpoint)])

    _assert_reason(exc_info, "reasoning_checkpoint_has_predecessor")


async def test_patch_after_compaction_uses_null_reasoning_predecessor() -> None:
    compaction = CompactionPayload(
        id="cmp_root",
        memory={"root": True},
        threads=Threads(messages={"main": [Message(role="assistant", content="old")]}),
    )
    patch = _patch(
        "rs_patch",
        previous_compaction_id=compaction.id,
        memory=[{"op": "add", "path": "/next", "value": True}],
    )

    result = await _ingest([_sealed_compaction(compaction), _sealed_reasoning(patch)])

    assert result.memory == {"root": True, "next": True}
    assert result.threads["main"] == [Message(role="assistant", content="old")]
    assert result.main_tail == CompactedMainTail(source=Message(role="assistant", content="old"))
    assert result.last_compaction_id == compaction.id


async def test_reasoning_with_compaction_predecessor_rejects_without_compaction() -> None:
    root = _patch("rs_root", previous_compaction_id="cmp_missing")

    with pytest.raises(PlapError) as exc_info:
        await _ingest([_sealed_reasoning(root)])

    _assert_reason(exc_info, "reasoning_previous_compaction_id_mismatch")


async def test_reasoning_without_compaction_predecessor_rejects_after_compaction() -> None:
    compaction = CompactionPayload(id="cmp_root", memory={}, threads=Threads())
    root = _patch("rs_root")

    with pytest.raises(PlapError) as exc_info:
        await _ingest([_sealed_compaction(compaction), _sealed_reasoning(root)])

    _assert_reason(exc_info, "reasoning_previous_compaction_id_mismatch")


async def test_checkpoint_and_patch_keep_compaction_anchor() -> None:
    compaction = CompactionPayload(id="cmp_root", memory={"root": True}, threads=Threads())
    checkpoint = _checkpoint(
        "rs_checkpoint",
        previous_compaction_id=compaction.id,
        memory={"turn": 1},
    )
    patch = _patch(
        "rs_patch",
        previous_reasoning_id=checkpoint.id,
        previous_compaction_id=compaction.id,
        memory=[{"op": "add", "path": "/tool", "value": True}],
    )

    result = await _ingest(
        [
            _sealed_compaction(compaction),
            _message("u", role="user"),
            _sealed_reasoning(checkpoint),
            _sealed_reasoning(patch),
        ]
    )

    assert result.memory == {"turn": 1, "tool": True}
    assert result.last_reasoning_id == patch.id
    assert result.last_compaction_id == compaction.id


async def test_checkpoint_without_compaction_anchor_rejects_after_compaction() -> None:
    compaction = CompactionPayload(id="cmp_root", memory={}, threads=Threads())
    checkpoint = _checkpoint("rs_checkpoint")

    with pytest.raises(PlapError) as exc_info:
        await _ingest(
            [
                _sealed_compaction(compaction),
                _message("u", role="user"),
                _sealed_reasoning(checkpoint),
            ]
        )

    _assert_reason(exc_info, "reasoning_previous_compaction_id_mismatch")


async def test_reasoning_anchored_to_old_compaction_rejects_after_new_compaction() -> None:
    current = CompactionPayload(id="cmp_current", memory={}, threads=Threads())
    stale = _patch("rs_stale", previous_compaction_id="cmp_old")

    with pytest.raises(PlapError) as exc_info:
        await _ingest([_sealed_compaction(current), _sealed_reasoning(stale)])

    _assert_reason(exc_info, "reasoning_previous_compaction_id_mismatch")


async def test_last_compaction_slice_uses_last_compaction_anchor() -> None:
    first_compaction = CompactionPayload(id="cmp_first", memory={"generation": 1}, threads=Threads())
    first_reasoning = _patch(
        "rs_first",
        previous_compaction_id=first_compaction.id,
        memory=[{"op": "add", "path": "/discarded", "value": True}],
    )
    last_compaction = CompactionPayload(id="cmp_last", memory={"generation": 2}, threads=Threads())
    last_reasoning = _patch(
        "rs_last",
        previous_compaction_id=last_compaction.id,
        memory=[{"op": "add", "path": "/kept", "value": True}],
    )

    result = await _ingest(
        [
            _sealed_compaction(first_compaction),
            _sealed_reasoning(first_reasoning),
            _sealed_compaction(last_compaction),
            _sealed_reasoning(last_reasoning),
        ]
    )

    assert result.memory == {"generation": 2, "kept": True}
    assert result.last_reasoning_id == last_reasoning.id
    assert result.last_compaction_id == last_compaction.id


async def test_checkpoint_replaces_non_main_state_but_preserves_main() -> None:
    old_reviewer = Message(role="assistant", content="old reviewer")
    root = _patch(
        "rs_root",
        active={"main", "reviewer"},
        threads={"reviewer": [{"op": "add", "path": "/0", "value": old_reviewer.to_primitive()}]},
        main=[Message(role="assistant", content="old main")],
    )
    checkpoint = _checkpoint(
        "rs_checkpoint",
        memory={"new": True},
        active={"main", "arbitrator"},
        threads={"arbitrator": [Message(role="assistant", content="new arbitrator")]},
        main=[Message(role="assistant", content="new main")],
    )

    result = await _ingest([_sealed_reasoning(root), _message("u", role="user"), _sealed_reasoning(checkpoint)])

    assert result.memory == {"new": True}
    assert result.threads.active == {"main", "arbitrator"}
    assert "reviewer" not in result.threads.messages
    assert result.threads["arbitrator"] == [Message(role="assistant", content="new arbitrator")]
    assert result.threads["main"] == [
        Message(role="assistant", content="old main"),
        Message(role="user", content="u"),
        Message(role="assistant", content="new main"),
    ]


async def test_later_user_requires_another_checkpoint_root() -> None:
    first = _checkpoint("rs_first", memory={"turn": 1})
    second = _checkpoint("rs_second", memory={"turn": 2})

    result = await _ingest(
        [
            _message("u1", role="user"),
            _sealed_reasoning(first),
            _message("u2", role="user"),
            _sealed_reasoning(second),
        ]
    )

    assert result.memory == {"turn": 2}
    assert result.last_reasoning_id == second.id
    assert result.checkpoint_required is False


async def test_checkpoint_preserves_explicit_empty_non_main_thread() -> None:
    old = Message(role="assistant", content="old reviewer")
    root = _patch(
        "rs_root",
        threads={"reviewer": [{"op": "add", "path": "/0", "value": old.to_primitive()}]},
    )
    checkpoint = _checkpoint("rs_checkpoint", threads={"reviewer": []})

    result = await _ingest([_sealed_reasoning(root), _message("u", role="user"), _sealed_reasoning(checkpoint)])

    assert "reviewer" in result.threads.messages
    assert result.threads["reviewer"] == []


async def test_user_message_inside_reasoning_main_does_not_require_checkpoint() -> None:
    root = _patch("rs_root", main=[Message(role="user", content="sealed user")])
    continuation = _patch("rs_continuation", previous_reasoning_id=root.id)

    result = await _ingest([_sealed_reasoning(root), _sealed_reasoning(continuation)])

    assert result.checkpoint_required is False
    assert result.last_reasoning_id == continuation.id


def test_reasoning_state_variants_reject_main_thread_state() -> None:
    with pytest.raises(ValueError, match="may not contain main"):
        ReasoningCheckpoint(memory={}, active={"main"}, threads={"main": []})
    with pytest.raises(ValueError, match="may not target main"):
        ReasoningPatch(memory=[], threads={"main": []})


@pytest.mark.parametrize(
    ("between", "expected"),
    [
        ([], [Message(role="assistant", content="edited", reasoning_content="private")]),
        (
            [_message("developer note", role="developer")],
            [
                Message(role="developer", content="developer note"),
                Message(role="assistant", content="edited", reasoning_content="private"),
            ],
        ),
    ],
)
async def test_public_patch_uses_next_assistant_without_content_validation(
    between: list[RequestInputItem],
    expected: list[Message],
) -> None:
    private = Message(role="assistant", content="sealed", reasoning_content="private")
    checkpoint = _checkpoint("rs_checkpoint", main=[private, MessagePatch(private)])

    result = await _ingest([_message("u", role="user"), _sealed_reasoning(checkpoint), *between, _message("edited")])

    assert result.threads["main"] == [Message(role="user", content="u"), *expected]
    assert result.main_tail == PublicMainTail(source=private)


async def test_first_fabricated_assistant_consumes_public_patch() -> None:
    private = Message(role="assistant", content="sealed", reasoning_content="private")
    checkpoint = _checkpoint("rs_checkpoint", main=[private, MessagePatch(private)])

    result = await _ingest(
        [
            _message("u", role="user"),
            _sealed_reasoning(checkpoint),
            _message("first"),
            _message("second"),
        ]
    )

    assert result.threads["main"] == [
        Message(role="user", content="u"),
        Message(role="assistant", content="first", reasoning_content="private"),
        Message(role="assistant", content="second"),
    ]


async def test_public_patch_takes_content_and_refusal_from_public_assistant() -> None:
    private = Message(
        role="assistant",
        content="sealed text",
        refusal="sealed refusal",
        reasoning_content="private",
    )
    checkpoint = _checkpoint("rs_checkpoint", main=[private, MessagePatch(private)])
    public = RequestMessageItem(
        content=[
            OutputTextContent(text="edited text", type="output_text"),
            OutputRefusalContent(refusal="edited refusal", type="refusal"),
        ],
        role="assistant",
        type="message",
    )

    result = await _ingest([_message("u", role="user"), _sealed_reasoning(checkpoint), public])

    assert result.threads["main"] == [
        Message(role="user", content="u"),
        Message(
            role="assistant",
            content="edited text",
            refusal="edited refusal",
            reasoning_content="private",
        ),
    ]


async def test_equal_public_assistants_remain_distinct_occurrences() -> None:
    private = Message(role="assistant", content="same", reasoning_content="private")
    checkpoint = _checkpoint("rs_checkpoint", main=[private, MessagePatch(private)])

    result = await _ingest(
        [
            _message("u", role="user"),
            _sealed_reasoning(checkpoint),
            _message("same"),
            _message("same"),
        ]
    )

    assert result.threads["main"] == [
        Message(role="user", content="u"),
        Message(role="assistant", content="same", reasoning_content="private"),
        Message(role="assistant", content="same"),
    ]


async def test_standalone_public_tail_has_no_authenticated_source() -> None:
    result = await _ingest([_message("public")])

    assert result.main_tail == PublicMainTail(source=None)


async def test_public_source_match_moves_live_source_past_plain_message() -> None:
    source = Message(role="assistant", content="sealed", reasoning_content="private")
    root = _patch("rs_root", main=[source])
    relocated = _patch("rs_relocated", previous_reasoning_id=root.id, main=[MessagePatch(source)])

    result = await _ingest(
        [
            _sealed_reasoning(root),
            _message("developer note", role="developer"),
            _sealed_reasoning(relocated),
            _message("edited"),
        ]
    )

    assert result.threads["main"] == [
        Message(role="developer", content="developer note"),
        Message(role="assistant", content="edited", reasoning_content="private"),
    ]
    assert result.main_tail == PublicMainTail(source=source)


async def test_repeated_public_timewarp_replaces_projection_on_one_bundle() -> None:
    source = Message(role="assistant", content="sealed", reasoning_content="private")
    first = _patch("rs_first", main=[source, MessagePatch(source)])
    second = _patch("rs_second", previous_reasoning_id=first.id, main=[MessagePatch(source)])
    third = _patch("rs_third", previous_reasoning_id=second.id, main=[MessagePatch(source)])

    result = await _ingest(
        [
            _sealed_reasoning(first),
            _message("first projection"),
            _sealed_reasoning(second),
            _message("second projection"),
            _sealed_reasoning(third),
            _message("final projection"),
        ]
    )

    assert result.threads["main"] == [Message(role="assistant", content="final projection", reasoning_content="private")]
    assert result.main_tail == PublicMainTail(source=source)


async def test_repeated_public_timewarp_with_same_projection_does_not_duplicate_bundle() -> None:
    source = Message(role="assistant", content="sealed", reasoning_content="private")
    first = _patch("rs_first", main=[source, MessagePatch(source)])
    second = _patch("rs_second", previous_reasoning_id=first.id, main=[MessagePatch(source)])

    result = await _ingest(
        [
            _sealed_reasoning(first),
            _message("projection"),
            _sealed_reasoning(second),
            _message("projection"),
        ]
    )

    assert result.threads["main"] == [Message(role="assistant", content="projection", reasoning_content="private")]
    assert result.main_tail == PublicMainTail(source=source)


async def test_sliced_public_source_stages_from_message_patch() -> None:
    source = Message(role="assistant", content="sealed", reasoning_content="private")
    root = _patch("rs_root", main=[MessagePatch(source)])

    result = await _ingest([_sealed_reasoning(root), _message("edited")])

    assert result.threads["main"] == [Message(role="assistant", content="edited", reasoning_content="private")]


async def test_public_bearing_patch_without_assistant_is_rejected() -> None:
    source = Message(role="assistant", content="sealed", reasoning_content="private")
    root = _patch("rs_root", main=[MessagePatch(source)])

    with pytest.raises(PlapError) as exc_info:
        await _ingest([_sealed_reasoning(root)])

    _assert_reason(exc_info, "main_message_patch_target_missing")


async def test_public_patch_rehomes_declared_call_to_later_assistant() -> None:
    private = Message(
        role="assistant",
        content="sealed",
        reasoning_content="private",
        tool_calls=[_tool_call("up_0")],
    )
    checkpoint = _checkpoint("rs_checkpoint", main=[private, MessagePatch(private)])
    call_id = _sealed_call_id("main", "up_0")

    result = await _ingest(
        [
            _message("u", role="user"),
            _sealed_reasoning(checkpoint),
            _message("patch recipient"),
            _message("call owner"),
            _function_call(call_id),
            _function_output(call_id),
        ]
    )

    assert result.threads["main"] == [
        Message(role="user", content="u"),
        Message(role="assistant", content="patch recipient", reasoning_content="private"),
        Message(role="assistant", content="call owner", tool_calls=[_tool_call("up_0")]),
        Message(role="tool", tool_call_id="up_0", content="result"),
    ]


async def test_public_patch_attaches_declared_call_to_consuming_assistant() -> None:
    private = Message(
        role="assistant",
        content="sealed",
        reasoning_content="private",
        tool_calls=[_tool_call("up_0")],
    )
    checkpoint = _checkpoint("rs_checkpoint", main=[private, MessagePatch(private)])
    call_id = _sealed_call_id("main", "up_0")

    result = await _ingest(
        [
            _message("u", role="user"),
            _sealed_reasoning(checkpoint),
            _message("public"),
            _function_call(call_id),
            _function_output(call_id),
        ]
    )

    assert result.threads["main"] == [
        Message(role="user", content="u"),
        Message(
            role="assistant",
            content="public",
            reasoning_content="private",
            tool_calls=[_tool_call("up_0")],
        ),
        Message(role="tool", tool_call_id="up_0", content="result"),
    ]


async def test_direct_hidden_main_rehomes_call_to_later_assistant() -> None:
    hidden = Message(role="assistant", tool_calls=[_tool_call("up_0")])
    checkpoint = _checkpoint("rs_checkpoint", main=[hidden])
    call_id = _sealed_call_id("main", "up_0")

    result = await _ingest(
        [
            _message("u", role="user"),
            _sealed_reasoning(checkpoint),
            _message("later"),
            _function_call(call_id),
            _function_output(call_id),
        ]
    )

    assert result.threads["main"] == [
        Message(role="user", content="u"),
        Message(role="assistant"),
        Message(role="assistant", content="later", tool_calls=[_tool_call("up_0")]),
        Message(role="tool", tool_call_id="up_0", content="result"),
    ]


async def test_direct_hidden_and_later_public_assistant_are_distinct_positions() -> None:
    hidden = Message(
        role="assistant",
        content="sealed hidden",
        reasoning_content="private",
        tool_calls=[_tool_call("up_0")],
    )
    checkpoint = _checkpoint("rs_checkpoint", main=[hidden])
    call_id = _sealed_call_id("main", "up_0")

    result = await _ingest(
        [
            _message("u", role="user"),
            _sealed_reasoning(checkpoint),
            _message("public"),
            _function_call(call_id),
            _function_output(call_id),
        ]
    )

    assert result.threads["main"] == [
        Message(role="user", content="u"),
        Message(role="assistant", content="sealed hidden", reasoning_content="private"),
        Message(role="assistant", content="public", tool_calls=[_tool_call("up_0")]),
        Message(role="tool", tool_call_id="up_0", content="result"),
    ]


async def test_direct_hidden_main_owns_call_without_public_assistant() -> None:
    hidden = Message(
        role="assistant",
        content="hidden text",
        reasoning_content="private",
        tool_calls=[_tool_call("up_0")],
    )
    checkpoint = _checkpoint("rs_checkpoint", main=[hidden])
    call_id = _sealed_call_id("main", "up_0")

    result = await _ingest(
        [
            _message("u", role="user"),
            _sealed_reasoning(checkpoint),
            _function_call(call_id),
            _function_output(call_id),
        ]
    )

    assert result.threads["main"] == [
        Message(role="user", content="u"),
        hidden,
        Message(role="tool", tool_call_id="up_0", content="result"),
    ]
    assert result.main_tail == HiddenMainTail(source=hidden)


async def test_output_empty_patch_is_immediate_hidden_owner() -> None:
    hidden = Message(role="assistant", tool_calls=[_tool_call("up_0")])
    checkpoint = _checkpoint("rs_checkpoint", main=[MessagePatch(hidden)])
    call_id = _sealed_call_id("main", "up_0")

    result = await _ingest(
        [
            _message("u", role="user"),
            _sealed_reasoning(checkpoint),
            _function_call(call_id),
            _function_output(call_id),
        ]
    )

    assert result.threads["main"] == [
        Message(role="user", content="u"),
        hidden,
        Message(role="tool", tool_call_id="up_0", content="result"),
    ]


async def test_output_empty_patch_rehomes_call_to_later_assistant() -> None:
    hidden = Message(role="assistant", tool_calls=[_tool_call("up_0")])
    checkpoint = _checkpoint("rs_checkpoint", main=[MessagePatch(hidden)])
    call_id = _sealed_call_id("main", "up_0")

    result = await _ingest(
        [
            _message("u", role="user"),
            _sealed_reasoning(checkpoint),
            _message("later"),
            _function_call(call_id),
            _function_output(call_id),
        ]
    )

    assert result.threads["main"] == [
        Message(role="user", content="u"),
        Message(role="assistant"),
        Message(role="assistant", content="later", tool_calls=[_tool_call("up_0")]),
        Message(role="tool", tool_call_id="up_0", content="result"),
    ]


async def test_inactive_call_only_main_reactivates_through_hidden_patch() -> None:
    hidden = Message(role="assistant", reasoning_content="private", tool_calls=[_tool_call("up_0")])
    parked = _patch("rs_parked", active=set(), main=[hidden])
    activated = _patch(
        "rs_activated",
        previous_reasoning_id=parked.id,
        active={"main"},
        main=[MessagePatch(hidden)],
    )
    call_id = _sealed_call_id("main", "up_0")

    result = await _ingest(
        [
            _sealed_reasoning(parked),
            _sealed_reasoning(activated),
            _function_call(call_id),
            _function_output(call_id),
        ]
    )

    assert result.threads.active == {"main"}
    assert result.threads["main"] == [hidden, Message(role="tool", tool_call_id="up_0", content="result")]


async def test_inactive_public_main_reactivates_through_public_patch() -> None:
    hidden = Message(
        role="assistant",
        content="sealed",
        reasoning_content="private",
        tool_calls=[_tool_call("up_0")],
    )
    parked = _patch("rs_parked", active=set(), main=[hidden])
    activated = _patch(
        "rs_activated",
        previous_reasoning_id=parked.id,
        active={"main"},
        main=[MessagePatch(hidden)],
    )
    call_id = _sealed_call_id("main", "up_0")

    result = await _ingest(
        [
            _sealed_reasoning(parked),
            _sealed_reasoning(activated),
            _message("public"),
            _function_call(call_id),
            _function_output(call_id),
        ]
    )

    assert result.threads.active == {"main"}
    assert result.threads["main"] == [
        Message(
            role="assistant",
            content="public",
            reasoning_content="private",
            tool_calls=[_tool_call("up_0")],
        ),
        Message(role="tool", tool_call_id="up_0", content="result"),
    ]


async def test_current_append_hidden_output_settles_local_assistant() -> None:
    hidden = Message(role="assistant", reasoning_content="private", tool_calls=[_tool_call("server_0")])
    hidden_output = Message(role="tool", tool_call_id="server_0", content="server result")
    root = _patch("rs_root", main=[hidden, hidden_output])

    result = await _ingest([_sealed_reasoning(root)])

    assert result.threads["main"] == [hidden, hidden_output]


async def test_current_append_hidden_output_survives_public_projection() -> None:
    hidden = Message(
        role="assistant",
        content="sealed",
        reasoning_content="private",
        tool_calls=[_tool_call("server_0")],
    )
    hidden_output = Message(role="tool", tool_call_id="server_0", content="server result")
    root = _patch("rs_root", main=[hidden, hidden_output, MessagePatch(hidden)])

    result = await _ingest([_sealed_reasoning(root), _message("public")])

    assert result.threads["main"] == [
        Message(
            role="assistant",
            content="public",
            reasoning_content="private",
            tool_calls=[_tool_call("server_0")],
        ),
        hidden_output,
    ]


async def test_persisted_hidden_output_settles_authenticated_tail() -> None:
    hidden = Message(role="assistant", reasoning_content="private", tool_calls=[_tool_call("server_0")])
    hidden_output = Message(role="tool", tool_call_id="server_0", content="server result")
    parked = _patch("rs_parked", active=set(), main=[hidden])
    settled = _patch("rs_settled", previous_reasoning_id=parked.id, main=[hidden_output])

    result = await _ingest([_sealed_reasoning(parked), _sealed_reasoning(settled)])

    assert result.threads["main"] == [hidden, hidden_output]


async def test_fully_settled_parked_turn_accepts_explicit_later_publication() -> None:
    hidden = Message(
        role="assistant",
        content="sealed",
        reasoning_content="private",
        tool_calls=[_tool_call("server_0")],
    )
    hidden_output = Message(role="tool", tool_call_id="server_0", content="server result")
    parked = _patch("rs_parked", active=set(), main=[hidden])
    settled = _patch("rs_settled", previous_reasoning_id=parked.id, main=[hidden_output])
    published = _patch(
        "rs_published",
        previous_reasoning_id=settled.id,
        active={"main"},
        main=[MessagePatch(hidden)],
    )

    result = await _ingest(
        [
            _sealed_reasoning(parked),
            _sealed_reasoning(settled),
            _sealed_reasoning(published),
            _message("public"),
        ]
    )

    assert result.threads["main"] == [
        Message(
            role="assistant",
            content="public",
            reasoning_content="private",
            tool_calls=[_tool_call("server_0")],
        ),
        hidden_output,
    ]


async def test_partially_settled_parked_turn_publishes_only_remaining_call() -> None:
    hidden = Message(
        role="assistant",
        content="sealed",
        reasoning_content="private",
        tool_calls=[_tool_call("server_0"), _tool_call("client_0")],
    )
    hidden_output = Message(role="tool", tool_call_id="server_0", content="server result")
    parked = _patch("rs_parked", active=set(), main=[hidden])
    settled = _patch("rs_settled", previous_reasoning_id=parked.id, main=[hidden_output])
    published = _patch(
        "rs_published",
        previous_reasoning_id=settled.id,
        active={"main"},
        main=[MessagePatch(hidden)],
    )
    call_id = _sealed_call_id("main", "client_0")

    result = await _ingest(
        [
            _sealed_reasoning(parked),
            _sealed_reasoning(settled),
            _sealed_reasoning(published),
            _message("public"),
            _function_call(call_id),
            _function_output(call_id, "client result"),
        ]
    )

    assert result.threads["main"] == [
        Message(
            role="assistant",
            content="public",
            reasoning_content="private",
            tool_calls=[_tool_call("server_0"), _tool_call("client_0")],
        ),
        hidden_output,
        Message(role="tool", tool_call_id="client_0", content="client result"),
    ]


async def test_partial_persisted_hidden_settlement_precedes_public_call() -> None:
    hidden = Message(
        role="assistant",
        content="sealed",
        reasoning_content="private",
        tool_calls=[_tool_call("server_0"), _tool_call("client_0")],
    )
    hidden_output = Message(role="tool", tool_call_id="server_0", content="server result")
    parked = _patch("rs_parked", active=set(), main=[hidden])
    activated = _patch(
        "rs_activated",
        previous_reasoning_id=parked.id,
        active={"main"},
        main=[hidden_output, MessagePatch(hidden)],
    )
    call_id = _sealed_call_id("main", "client_0")

    result = await _ingest(
        [
            _sealed_reasoning(parked),
            _sealed_reasoning(activated),
            _message("public"),
            _function_call(call_id),
            _function_output(call_id, "client result"),
        ]
    )

    assert result.threads["main"] == [
        Message(
            role="assistant",
            content="public",
            reasoning_content="private",
            tool_calls=[_tool_call("server_0"), _tool_call("client_0")],
        ),
        hidden_output,
        Message(role="tool", tool_call_id="client_0", content="client result"),
    ]


async def test_sliced_patch_accepts_leading_hidden_output() -> None:
    hidden = Message(
        role="assistant",
        content="sealed",
        reasoning_content="private",
        tool_calls=[_tool_call("server_0")],
    )
    hidden_output = Message(role="tool", tool_call_id="server_0", content="server result")
    root = _patch("rs_root", main=[hidden_output, MessagePatch(hidden)])

    result = await _ingest([_sealed_reasoning(root), _message("public")])

    assert result.threads["main"] == [
        Message(
            role="assistant",
            content="public",
            reasoning_content="private",
            tool_calls=[_tool_call("server_0")],
        ),
        hidden_output,
    ]


async def test_closed_hidden_prefix_precedes_current_assistant() -> None:
    prefix = Message(role="assistant", content="prefix", tool_calls=[_tool_call("server_0")])
    prefix_output = Message(role="tool", tool_call_id="server_0", content="server result")
    current = Message(role="assistant", content="current", reasoning_content="private")
    root = _patch("rs_root", main=[prefix, prefix_output, current])

    result = await _ingest([_sealed_reasoning(root)])

    assert result.threads["main"] == [prefix, prefix_output, current]


async def test_hidden_timewarp_moves_complete_source_bundle_and_leaves_user_in_place() -> None:
    hidden = Message(role="assistant", tool_calls=[_tool_call("up_0")])
    parked = _patch("rs_parked", active=set(), main=[hidden])
    fabricated_call = _function_call("fab_0", name="fabricated")
    checkpoint = _checkpoint("rs_checkpoint", main=[MessagePatch(hidden)])
    sealed_call = _sealed_call_id("main", "up_0")

    result = await _ingest(
        [
            _sealed_reasoning(parked),
            fabricated_call,
            _function_output("fab_0", "fabricated result"),
            _message("u", role="user"),
            _sealed_reasoning(checkpoint),
            _function_call(sealed_call),
            _function_output(sealed_call, "sealed result"),
        ]
    )

    assert result.threads["main"] == [
        Message(role="user", content="u"),
        Message(
            role="assistant",
            tool_calls=[_tool_call("up_0"), _tool_call("fab_0", name="fabricated")],
        ),
        Message(role="tool", tool_call_id="fab_0", content="fabricated result"),
        Message(role="tool", tool_call_id="up_0", content="sealed result"),
    ]


async def test_hidden_timewarp_crosses_plain_message_without_user_boundary() -> None:
    hidden = Message(role="assistant", reasoning_content="private", tool_calls=[_tool_call("up_0")])
    parked = _patch("rs_parked", active=set(), main=[hidden])
    relocated = _patch(
        "rs_relocated",
        previous_reasoning_id=parked.id,
        active={"main"},
        main=[MessagePatch(hidden)],
    )
    call_id = _sealed_call_id("main", "up_0")

    result = await _ingest(
        [
            _sealed_reasoning(parked),
            _message("developer note", role="developer"),
            _sealed_reasoning(relocated),
            _function_call(call_id),
            _function_output(call_id),
        ]
    )

    assert result.threads["main"] == [
        Message(role="developer", content="developer note"),
        hidden,
        Message(role="tool", tool_call_id="up_0", content="result"),
    ]


async def test_hidden_timewarp_moves_source_owned_fabricated_tool_work() -> None:
    hidden = Message(role="assistant", reasoning_content="private", tool_calls=[_tool_call("up_0")])
    parked = _patch("rs_parked", active=set(), main=[hidden])
    relocated = _patch(
        "rs_relocated",
        previous_reasoning_id=parked.id,
        active={"main"},
        main=[MessagePatch(hidden)],
    )
    call_id = _sealed_call_id("main", "up_0")

    result = await _ingest(
        [
            _sealed_reasoning(parked),
            _function_call("fab_0", name="fabricated"),
            _function_output("fab_0", "fabricated result"),
            _message("developer note", role="developer"),
            _sealed_reasoning(relocated),
            _function_call(call_id),
            _function_output(call_id, "sealed result"),
        ]
    )

    assert result.threads["main"] == [
        Message(role="developer", content="developer note"),
        Message(
            role="assistant",
            reasoning_content="private",
            tool_calls=[_tool_call("up_0"), _tool_call("fab_0", name="fabricated")],
        ),
        Message(role="tool", tool_call_id="fab_0", content="fabricated result"),
        Message(role="tool", tool_call_id="up_0", content="sealed result"),
    ]


async def test_hidden_source_mismatch_keeps_authenticated_tail() -> None:
    original = Message(role="assistant", reasoning_content="original private")
    other = Message(role="assistant", reasoning_content="other private")
    root = _patch("rs_root", main=[original])
    mismatch = _patch("rs_mismatch", previous_reasoning_id=root.id, main=[MessagePatch(other)])

    result = await _ingest([_sealed_reasoning(root), _sealed_reasoning(mismatch)])

    assert result.threads["main"] == [original, other]


async def test_source_mismatch_preserves_unrelated_authenticated_tail() -> None:
    original = Message(role="assistant", content="original", reasoning_content="original private")
    other = Message(role="assistant", content="other", reasoning_content="other private")
    checkpoint = _checkpoint("rs_checkpoint", main=[original, MessagePatch(other)])

    result = await _ingest([_message("u", role="user"), _sealed_reasoning(checkpoint), _message("edited")])

    assert result.threads["main"] == [
        Message(role="user", content="u"),
        original,
        _private_reasoning(other, "edited"),
    ]


async def test_plain_message_is_pushed_after_assistant_tool_bundle() -> None:
    result = await _ingest(
        [
            _message("owner"),
            _message("note", role="developer"),
            _function_call("fab_0", name="fabricated"),
            _function_output("fab_0"),
        ]
    )

    assert result.threads["main"] == [
        Message(role="assistant", content="owner", tool_calls=[_tool_call("fab_0", name="fabricated")]),
        Message(role="tool", tool_call_id="fab_0", content="result"),
        Message(role="developer", content="note"),
    ]


async def test_parallel_call_outputs_preserve_observed_chronology() -> None:
    result = await _ingest(
        [
            _message("owner"),
            _function_call("fab_1", name="first"),
            _function_call("fab_2", name="second"),
            _function_output("fab_2", "second result"),
            _function_output("fab_1", "first result"),
        ]
    )

    assert result.threads["main"] == [
        Message(
            role="assistant",
            content="owner",
            tool_calls=[_tool_call("fab_1", name="first"), _tool_call("fab_2", name="second")],
        ),
        Message(role="tool", tool_call_id="fab_2", content="second result"),
        Message(role="tool", tool_call_id="fab_1", content="first result"),
    ]


async def test_fabricated_pair_attaches_to_inactive_main() -> None:
    hidden = Message(role="assistant", content="hidden")
    parked = _patch("rs_parked", active=set(), main=[hidden])

    result = await _ingest(
        [
            _sealed_reasoning(parked),
            _function_call("fab_0", name="fabricated"),
            _function_output("fab_0", "fabricated result"),
        ]
    )

    assert result.threads.active == set()
    assert result.threads["main"] == [
        Message(
            role="assistant",
            content="hidden",
            tool_calls=[_tool_call("fab_0", name="fabricated")],
        ),
        Message(role="tool", tool_call_id="fab_0", content="fabricated result"),
    ]


async def test_fabricated_pair_preserves_other_parked_declaration() -> None:
    hidden = Message(role="assistant", tool_calls=[_tool_call("parked_0", name="parked")])
    parked = _patch("rs_parked", active=set(), main=[hidden])

    result = await _ingest(
        [
            _sealed_reasoning(parked),
            _function_call("fab_0", name="fabricated"),
            _function_output("fab_0", "fabricated result"),
        ]
    )

    assert result.threads["main"] == [
        Message(
            role="assistant",
            tool_calls=[_tool_call("parked_0", name="parked"), _tool_call("fab_0", name="fabricated")],
        ),
        Message(role="tool", tool_call_id="fab_0", content="fabricated result"),
    ]


async def test_fabricated_pair_can_settle_matching_parked_declaration() -> None:
    hidden = Message(role="assistant", tool_calls=[_tool_call("parked_0", name="parked")])
    parked = _patch("rs_parked", active=set(), main=[hidden])

    result = await _ingest(
        [
            _sealed_reasoning(parked),
            _function_call("parked_0", name="spoofed"),
            _function_output("parked_0", "fabricated result"),
        ]
    )

    assert result.threads["main"] == [
        hidden,
        Message(role="tool", tool_call_id="parked_0", content="fabricated result"),
    ]


async def test_rehomed_declaration_stays_before_later_fabricated_call() -> None:
    hidden = Message(role="assistant", tool_calls=[_tool_call("up_0", name="authenticated")])
    checkpoint = _checkpoint("rs_checkpoint", main=[hidden])
    call_id = _sealed_call_id("main", "up_0")

    result = await _ingest(
        [
            _message("u", role="user"),
            _sealed_reasoning(checkpoint),
            _message("later"),
            _function_call("fab_0", name="fabricated"),
            _function_output("fab_0", "fabricated result"),
            _function_call(call_id, name="spoofed"),
            _function_output(call_id, "authenticated result"),
        ]
    )

    assert result.threads["main"] == [
        Message(role="user", content="u"),
        Message(role="assistant"),
        Message(
            role="assistant",
            content="later",
            tool_calls=[_tool_call("up_0", name="authenticated"), _tool_call("fab_0", name="fabricated")],
        ),
        Message(role="tool", tool_call_id="fab_0", content="fabricated result"),
        Message(role="tool", tool_call_id="up_0", content="authenticated result"),
    ]


async def test_sealed_transplant_attaches_to_inactive_main() -> None:
    hidden = Message(role="assistant", content="hidden")
    parked = _patch("rs_parked", active=set(), main=[hidden])
    call_id = _sealed_call_id("main", "transplanted_0")

    result = await _ingest(
        [
            _sealed_reasoning(parked),
            _function_call(call_id, name="transplanted"),
            _function_output(call_id, "transplanted result"),
        ]
    )

    assert result.threads.active == set()
    assert result.threads["main"] == [
        Message(
            role="assistant",
            content="hidden",
            tool_calls=[_tool_call("transplanted_0", name="transplanted")],
        ),
        Message(role="tool", tool_call_id="transplanted_0", content="transplanted result"),
    ]


async def test_rehoming_preserves_authenticated_declaration_metadata() -> None:
    declared = ToolCall(id="up_0", name="authenticated", arguments='{"trusted":true}')
    hidden = Message(role="assistant", tool_calls=[declared])
    checkpoint = _checkpoint("rs_checkpoint", main=[hidden])
    call_id = _sealed_call_id("main", "up_0")
    spoofed = RequestFunctionCallItem(
        arguments='{"trusted":false}',
        call_id=call_id,
        name="spoofed",
        type="function_call",
    )

    result = await _ingest(
        [
            _message("u", role="user"),
            _sealed_reasoning(checkpoint),
            _message("later"),
            spoofed,
            _function_output(call_id),
        ]
    )

    assert result.threads["main"] == [
        Message(role="user", content="u"),
        Message(role="assistant"),
        Message(role="assistant", content="later", tool_calls=[declared]),
        Message(role="tool", tool_call_id="up_0", content="result"),
    ]


async def test_plain_message_may_cross_open_main_call() -> None:
    result = await _ingest(
        [
            _message("owner"),
            _function_call("fab_0", name="fabricated"),
            _message("developer note", role="developer"),
            _function_output("fab_0"),
        ]
    )

    assert result.threads["main"] == [
        Message(role="assistant", content="owner", tool_calls=[_tool_call("fab_0", name="fabricated")]),
        Message(role="tool", tool_call_id="fab_0", content="result"),
        Message(role="developer", content="developer note"),
    ]


async def test_assistant_may_cross_open_call_without_changing_its_owner() -> None:
    result = await _ingest(
        [
            _message("first"),
            _function_call("fab_1", name="first_call"),
            _message("second"),
            _function_output("fab_1", "first result"),
        ]
    )

    assert result.threads["main"] == [
        Message(role="assistant", content="first", tool_calls=[_tool_call("fab_1", name="first_call")]),
        Message(role="tool", tool_call_id="fab_1", content="first result"),
        Message(role="assistant", content="second"),
    ]


async def test_sequential_closed_assistant_tool_bundles() -> None:
    result = await _ingest(
        [
            _message("first"),
            _function_call("fab_1", name="first_call"),
            _function_output("fab_1", "first result"),
            _message("second"),
            _function_call("fab_2", name="second_call"),
            _function_output("fab_2", "second result"),
        ]
    )

    assert result.threads["main"] == [
        Message(role="assistant", content="first", tool_calls=[_tool_call("fab_1", name="first_call")]),
        Message(role="tool", tool_call_id="fab_1", content="first result"),
        Message(role="assistant", content="second", tool_calls=[_tool_call("fab_2", name="second_call")]),
        Message(role="tool", tool_call_id="fab_2", content="second result"),
    ]


async def test_concurrently_open_calls_render_by_assistant_position() -> None:
    result = await _ingest(
        [
            _message("first"),
            _function_call("fab_1", name="first_call"),
            _message("second"),
            _function_call("fab_2", name="second_call"),
            _function_output("fab_2", "second result"),
            _function_output("fab_1", "first result"),
        ]
    )

    assert result.threads["main"] == [
        Message(role="assistant", content="first", tool_calls=[_tool_call("fab_1", name="first_call")]),
        Message(role="tool", tool_call_id="fab_1", content="first result"),
        Message(role="assistant", content="second", tool_calls=[_tool_call("fab_2", name="second_call")]),
        Message(role="tool", tool_call_id="fab_2", content="second result"),
    ]


async def test_user_may_cross_open_call_when_output_precedes_checkpoint() -> None:
    checkpoint = _checkpoint("rs_checkpoint")

    result = await _ingest(
        [
            _message("owner"),
            _function_call("fab_0", name="fabricated"),
            _message("new request", role="user"),
            _function_output("fab_0"),
            _sealed_reasoning(checkpoint),
        ]
    )

    assert result.threads["main"] == [
        Message(role="assistant", content="owner", tool_calls=[_tool_call("fab_0", name="fabricated")]),
        Message(role="tool", tool_call_id="fab_0", content="result"),
        Message(role="user", content="new request"),
    ]
    assert result.last_reasoning_id == checkpoint.id


async def test_reasoning_boundary_rejects_interleaved_open_calls() -> None:
    root = _patch("rs_root")

    with pytest.raises(PlapError) as exc_info:
        await _ingest(
            [
                _message("first"),
                _function_call("fab_1", name="first_call"),
                _message("second"),
                _sealed_reasoning(root),
            ]
        )

    _assert_reason(exc_info, "pending_tool_outputs_block_message")


async def test_main_sealed_transplant_attaches_to_nearest_assistant() -> None:
    call_id = _sealed_call_id("main", "transplanted")

    result = await _ingest(
        [
            _message("owner"),
            _function_call(call_id, name="transplanted"),
            _function_output(call_id),
        ]
    )

    assert result.threads["main"] == [
        Message(role="assistant", content="owner", tool_calls=[_tool_call("transplanted", name="transplanted")]),
        Message(role="tool", tool_call_id="transplanted", content="result"),
    ]


async def test_stale_non_main_pair_is_discarded() -> None:
    call_id = _sealed_call_id("reviewer", "stale")

    result = await _ingest([_function_call(call_id), _function_output(call_id)])

    assert result.threads.get("reviewer") is None


@pytest.mark.parametrize("output_order", [("review_0", "review_1"), ("review_1", "review_0")])
async def test_parallel_non_main_call_outputs_preserve_observed_chronology(output_order: tuple[str, str]) -> None:
    reviewer = Message(
        role="assistant",
        tool_calls=[_tool_call("review_0", name="first"), _tool_call("review_1", name="second")],
    )
    root = _patch(
        "rs_root",
        active={"main", "reviewer"},
        threads={"reviewer": [{"op": "add", "path": "/0", "value": reviewer.to_primitive()}]},
    )
    call_ids = {call_id: _sealed_call_id("reviewer", call_id) for call_id in output_order}
    advanced = _patch(
        "rs_advanced",
        previous_reasoning_id=root.id,
        threads={
            "reviewer": [
                {
                    "op": "add",
                    "path": "/3",
                    "value": Message(role="developer", content="continue").to_primitive(),
                }
            ]
        },
    )

    result = await _ingest(
        [
            _sealed_reasoning(root),
            _function_call(call_ids["review_0"], name="first"),
            _function_call(call_ids["review_1"], name="second"),
            *[_function_output(call_ids[call_id], f"{call_id} result") for call_id in output_order],
            _sealed_reasoning(advanced),
        ]
    )

    assert result.threads["reviewer"] == [
        reviewer,
        *[Message(role="tool", tool_call_id=call_id, content=f"{call_id} result") for call_id in output_order],
        Message(role="developer", content="continue"),
    ]


async def test_main_timewarp_does_not_rehome_interleaved_non_main_pair() -> None:
    source = Message(role="assistant", reasoning_content="private")
    reviewer = Message(role="assistant", tool_calls=[_tool_call("review_0")])
    root = _patch(
        "rs_root",
        active={"reviewer"},
        threads={"reviewer": [{"op": "add", "path": "/0", "value": reviewer.to_primitive()}]},
        main=[source],
    )
    relocated = _patch(
        "rs_relocated",
        previous_reasoning_id=root.id,
        active={"main"},
        main=[MessagePatch(source)],
    )
    reviewer_call = _sealed_call_id("reviewer", "review_0")

    result = await _ingest(
        [
            _sealed_reasoning(root),
            _function_call(reviewer_call),
            _function_output(reviewer_call, "review result"),
            _message("developer note", role="developer"),
            _sealed_reasoning(relocated),
        ]
    )

    assert result.threads["main"] == [Message(role="developer", content="developer note"), source]
    assert result.threads["reviewer"] == [
        reviewer,
        Message(role="tool", tool_call_id="review_0", content="review result"),
    ]


async def test_non_main_call_attachment_is_isolated_from_main_position() -> None:
    reviewer = Message(role="assistant", content="review", tool_calls=[_tool_call("review_0")])
    checkpoint = _checkpoint(
        "rs_checkpoint",
        active={"main", "reviewer"},
        threads={"reviewer": [reviewer]},
    )
    reviewer_call = _sealed_call_id("reviewer", "review_0")

    result = await _ingest(
        [
            _message("u", role="user"),
            _sealed_reasoning(checkpoint),
            _message("main owner"),
            _function_call(reviewer_call),
            _function_output(reviewer_call, "review result"),
            _function_call("fab_0", name="fabricated"),
            _function_output("fab_0", "main result"),
        ]
    )

    assert result.threads["reviewer"] == [
        reviewer,
        Message(role="tool", tool_call_id="review_0", content="review result"),
    ]
    assert result.threads["main"] == [
        Message(role="user", content="u"),
        Message(role="assistant", content="main owner", tool_calls=[_tool_call("fab_0", name="fabricated")]),
        Message(role="tool", tool_call_id="fab_0", content="main result"),
    ]


async def test_inactive_non_main_declaration_remains_parked() -> None:
    reviewer = Message(role="assistant", tool_calls=[_tool_call("review_0")])
    root = _patch(
        "rs_root",
        active={"main"},
        threads={"reviewer": [{"op": "add", "path": "/0", "value": reviewer.to_primitive()}]},
    )

    result = await _ingest([_sealed_reasoning(root)])

    assert result.threads["reviewer"] == [reviewer]


async def test_inactive_non_main_declaration_activates_and_settles_publicly() -> None:
    reviewer = Message(role="assistant", tool_calls=[_tool_call("review_0")])
    parked = _patch(
        "rs_parked",
        active={"main"},
        threads={"reviewer": [{"op": "add", "path": "/0", "value": reviewer.to_primitive()}]},
    )
    activated = _patch(
        "rs_activated",
        previous_reasoning_id=parked.id,
        active={"main", "reviewer"},
    )
    call_id = _sealed_call_id("reviewer", "review_0")

    result = await _ingest(
        [
            _sealed_reasoning(parked),
            _sealed_reasoning(activated),
            _function_call(call_id),
            _function_output(call_id, "review result"),
        ]
    )

    assert result.threads.active == {"main", "reviewer"}
    assert result.threads["reviewer"] == [
        reviewer,
        Message(role="tool", tool_call_id="review_0", content="review result"),
    ]


async def test_inactive_non_main_declaration_rejects_public_call() -> None:
    reviewer = Message(role="assistant", tool_calls=[_tool_call("review_0")])
    root = _patch(
        "rs_root",
        active={"main"},
        threads={"reviewer": [{"op": "add", "path": "/0", "value": reviewer.to_primitive()}]},
    )
    call_id = _sealed_call_id("reviewer", "review_0")

    with pytest.raises(PlapError) as exc_info:
        await _ingest([_sealed_reasoning(root), _function_call(call_id)])

    _assert_reason(exc_info, "inactive_thread_function_call")


async def test_active_declaration_requires_function_call_item() -> None:
    hidden = Message(role="assistant", tool_calls=[_tool_call("up_0")])
    checkpoint = _checkpoint("rs_checkpoint", main=[hidden])

    with pytest.raises(PlapError) as exc_info:
        await _ingest([_message("u", role="user"), _sealed_reasoning(checkpoint)])

    _assert_reason(exc_info, "reasoning_tool_call_missing_function_call_item")


async def test_open_call_requires_function_call_output() -> None:
    hidden = Message(role="assistant", tool_calls=[_tool_call("up_0")])
    checkpoint = _checkpoint("rs_checkpoint", main=[hidden])
    call_id = _sealed_call_id("main", "up_0")

    with pytest.raises(PlapError) as exc_info:
        await _ingest([_message("u", role="user"), _sealed_reasoning(checkpoint), _function_call(call_id)])

    _assert_reason(exc_info, "function_call_missing_function_call_output")


async def test_function_call_output_requires_open_call() -> None:
    hidden = Message(role="assistant", tool_calls=[_tool_call("up_0")])
    checkpoint = _checkpoint("rs_checkpoint", main=[hidden])
    call_id = _sealed_call_id("main", "up_0")

    with pytest.raises(PlapError) as exc_info:
        await _ingest([_message("u", role="user"), _sealed_reasoning(checkpoint), _function_output(call_id)])

    _assert_reason(exc_info, "function_call_output_without_pending_function_call")


async def test_duplicate_function_call_item_is_rejected() -> None:
    hidden = Message(role="assistant", tool_calls=[_tool_call("up_0")])
    checkpoint = _checkpoint("rs_checkpoint", main=[hidden])
    call_id = _sealed_call_id("main", "up_0")

    with pytest.raises(PlapError) as exc_info:
        await _ingest(
            [
                _message("u", role="user"),
                _sealed_reasoning(checkpoint),
                _function_call(call_id),
                _function_call(call_id),
            ]
        )

    _assert_reason(exc_info, "duplicate_pending_function_call")


async def test_public_bearing_patch_rejects_function_call_before_public_assistant() -> None:
    private = Message(role="assistant", content="answer", tool_calls=[_tool_call("up_0")])
    checkpoint = _checkpoint("rs_checkpoint", main=[private, MessagePatch(private)])
    call_id = _sealed_call_id("main", "up_0")

    with pytest.raises(PlapError) as exc_info:
        await _ingest(
            [
                _message("u", role="user"),
                _sealed_reasoning(checkpoint),
                _function_call(call_id),
                _function_output(call_id),
            ]
        )

    _assert_reason(exc_info, "main_message_patch_target_missing")
