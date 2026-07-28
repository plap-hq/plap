Ingestion Rebuild Proposal
1. Objective
Replace the fully chained reasoning replay system with a half-chained system:
... U R_checkpoint A FC FCO R_patch A ...
The intended boundaries are:
- The first reasoning item after a standalone user message checkpoints durable state and every non-main side.
- Main is never checkpointed by reasoning. It always uses append and postfix publication semantics.
- Reasoning during subsequent tool-call turns remains patch-based and chained to the checkpoint.
- A later user message breaks the chain again and requires another checkpoint.
- Compaction remains a distinct full checkpoint containing durable state and every side, including main.
- Public assistant content becomes positional and editable rather than authenticated by equality with sealed content.
- Tool declarations remain authenticated and strict on every side.
- Main function calls attach to assistant positions rather than being permanently tied to the assistant that carried their sealed declaration.
- Timewarp moves complete assistant-owned bundles without rewriting sealed payloads.
2. Canonical Examples
Input	Meaning
U R	R must checkpoint durable state and non-main sides.
U R1 FC FCO R2	R1 is a checkpoint; R2 is a chained patch.
C R	R is a patch based on full compaction C.
R(P(A)) M FC FCO	M receives private reasoning from A; calls attach to M.
R(P(A)) FMa M FC FCO	Reasoning attaches to FMa; calls attach to positional M.
R(P(A)) FMu M	FMu does not consume the patch; M does.
M FMu FC FCO	Calls attach to M and render before FMu.
R(M) FC FCO	Hidden M owns the call without a public message item.
R1(M) RN(P(M)) FC FCO	Hidden M timewarps to RN and owns the call there.
Terminology:
- Durable: application-owned state persisted across turns, separate from message histories.
- U: standalone public user message.
- M: public assistant message.
- FMa: fabricated assistant message.
- FMu: fabricated user, system, or developer message.
- FC: function-call item.
- FCO: function-call-output item.
- R(M): reasoning appending hidden assistant M.
- P(A): postfix MessagePatch carrying full sealed assistant A.
- C: compaction item.
3. Reasoning Payload Version 6
Reasoning moves to payload version 6.
Compaction remains version 4.
Call IDs remain version 2.
ReasoningPayload {
    id: string
    previous_reasoning_id: string | null
    previous_compaction_id: string | null
    state: ReasoningCheckpoint | ReasoningPatch
    main: MainUpdate[]
}
Checkpoint state:
ReasoningCheckpoint {
    type: "checkpoint"
    durable: object
    active: Side[]
    sides: {
        <non-main side>: Message[]
    }
}
Patch state:
ReasoningPatch {
    type: "patch"
    durable: JSONPatch
    active: Side[] | null
    sides: {
        <non-main side>: JSONPatch
    }
}
Main updates remain:
MainUpdate = Message | MessagePatch
Structural rules:
- Reasoning checkpoints cannot contain main in state.sides.
- Reasoning patches cannot contain main in state.sides.
- Main updates exist only in the common append-only main field.
- A checkpoint replaces the complete non-main side map.
- A checkpoint side omitted from sides is removed.
- An explicitly empty checkpoint side remains explicitly present.
- A patch side omitted from sides is unchanged.
- Main cannot be checkpointed or JSON-patched by reasoning.
No reasoning-v5 compatibility decoder will be retained.
4. Reasoning Boundary State
Replay tracks:
last_reasoning_id: string | null
last_compaction_id: string | null
checkpoint_required: bool
Transitions:
Event	Result
Empty root	Patch allowed with both predecessors null
Compaction	Reasoning chain cleared; compaction anchor replaced
Standalone user	Checkpoint required
Checkpoint reasoning	New reasoning chain root
Patch reasoning	Chain advances
Later standalone user	New checkpoint required
Validation rules:
- A checkpoint is rejected unless checkpoint_required is true.
- A patch is rejected while checkpoint_required is true.
- A checkpoint must have previous_reasoning_id=null.
- A patch must reference last_reasoning_id.
- A root patch may have a null predecessor.
- A post-compaction patch may have a null predecessor.
- Every reasoning item must reference last_compaction_id.
- Reasoning without compaction must have previous_compaction_id=null.
- Reasoning after compaction C must have previous_compaction_id=C.id.
- A later compaction replaces the compaction anchor.
- A checkpoint clears checkpoint_required.
- Compaction clears checkpoint_required.
- Only standalone public role=user items create the boundary.
- User-role messages inside sealed main updates do not create the boundary.
A user message may appear in raw input while a call is open. Canonical main
rendering keeps the eventual output in its owner bundle before that user node.
The following reasoning boundary still requires all prior obligations to be
settled:
- Active declarations must have matching FC items before reasoning or replay completion.
- Open calls must have matching FCO items before reasoning or replay completion.
- A user message cannot substitute for an emitted call output.
- Inactive parked-main calls retain their existing interruption semantics.
5. Checkpoint Semantics
Applying a reasoning checkpoint performs one atomic transaction:
 1. Validate that a checkpoint is required.
 2. Validate that active/open tool obligations do not block reasoning.
 3. Preserve reconstructed main history.
 4. Replace durable state.
 5. Replace active membership.
 6. Remove all existing non-main sides.
 7. Install the checkpoint’s complete non-main side map.
 8. Validate configured side membership.
 9. Rebuild non-main call trackers.
10. Apply the append-only main update.
11. Validate all resulting histories.
12. Set the checkpoint ID as the reasoning chain head.
13. Clear checkpoint_required.
Main is deliberately preserved through steps 4 through 9.
A patch transaction performs the same validation and main transition, but applies durable and non-main JSON patches rather than replacing state.
6. Compaction Semantics
Compaction remains stronger than a reasoning checkpoint.
Compaction:
- Replaces durable state.
- Replaces active membership.
- Replaces main.
- Replaces all non-main sides.
- Establishes an authenticated main tail.
- Clears reasoning lineage.
- Establishes its ID as the compaction lineage anchor.
- Allows a following patch with previous_reasoning_id=null.
Reasoning checkpoint:
- Replaces durable state.
- Replaces active membership.
- Replaces only non-main sides.
- Appends to main.
- Establishes a new reasoning chain head.
Therefore:
C R_patch
is valid without an intervening user when R_patch.previous_compaction_id=C.id.
C U R_checkpoint
requires a checkpoint because the user appeared after compaction.
7. Main Replay Architecture
Remove the preprocessing normalization system.
Delete:
- _normalize_timewarped_patches()
- Payload replacement maps
- Replayed-prefix reconstruction
- Cross-reasoning payload rewriting
- Public assistant content matching
- Content-based public skipping
- Equal-content deduplication
- Source movement implemented through generated JSON patches
Replace it with one live positional main reducer.
The reducer tracks:
- Plain transcript positions.
- Assistant bundle objects.
- Authenticated assistant-tail identity.
- Call declarations.
- Call ownership.
- Tool output ownership.
- Pending public attachment state.
- Hidden attachment availability.
- Canonical assistant/tool ordering.
An assistant bundle owns:
- Its assistant message.
- Calls currently attached to that assistant.
- Tool outputs for those calls.
- Source-owned fabricated calls and outputs.
- Hidden settled calls and outputs.
An assistant bundle does not own unrelated user, system, or developer messages.
8. Authenticated Tail
The authenticated tail is the latest eligible main assistant introduced by:
- Compaction.
- A full assistant in reasoning main updates.
- A previously applied MessagePatch.
Fabricated public assistants do not replace authenticated source identity merely by appearing in the queue.
Source comparison uses the complete sealed assistant representation.
It does not compare against:
- Public assistant content.
- Public assistant hashes.
- Arbitrary older assistants.
- Fabricated message equality.
The current authenticated tail is the only source candidate.
Replay also exports transient structural metadata for the final main assistant
bundle. HiddenMainTail and CompactedMainTail retain their complete authenticated
source. PublicMainTail retains source A after P(A) B and has no source for an
unauthenticated standalone public assistant. This metadata is reconstructed
from request history on every ingestion and is never added to reasoning or
compaction wire formats.
9. MessagePatch Representation
MessagePatch continues carrying the full sealed assistant:
MessagePatch {
    message: A
}
The full assistant is required for:
- Authenticated source selection.
- Fallback reconstruction after slicing.
- Private reasoning recovery.
- Tool declaration authentication.
- Hidden call-only timewarp.
- Partial hidden settlement.
The patch’s sealed public content is not authoritative for the eventual public assistant.
A.content and A.refusal are retained for source selection only.
10. MessagePatch Resolution Modes
A patch has two resolution modes.
has_public_projection = bool(content.assistant_output(A))
Patch kind	Behavior
Public-bearing P(A)	Wait for the next public assistant
Output-empty P(M)	Resolve immediately as a hidden assistant position
The main reducer therefore tracks:
HIDDEN
AWAITING_PUBLIC_ASSISTANT
PUBLICLY_ATTACHED
COMPACTED_SNAPSHOT
Only AWAITING_PUBLIC_ASSISTANT is invalid at another reasoning boundary or replay completion.
Public-Bearing Patch
For:
R(P(A)) X
the reconstructed assistant receives:
content            <- X.content
refusal            <- X.refusal
reasoning_content  <- A.reasoning_content
private settled work <- A/source bundle
No equality check is performed between X and A.
A user, system, or developer message does not consume the patch.
A public-bearing patch without a later assistant is invalid.
Output-Empty Patch
For:
R(P(M)) FC FCO
where M has no public text or refusal:
- P(M) resolves immediately.
- Hidden M becomes an attachment position.
- No public assistant message is expected.
- FC/FCO may attach directly to hidden M.
- Replay completion does not report a missing patch target.
This allows call-only turns to remain hidden while their calls are public.
11. Direct Hidden Main
A full assistant appended without a MessagePatch is hidden immediately, regardless of whether its internal content is empty.
For:
R(M) FC FCO
processing is:
1. R(M) appends hidden assistant M.
2. M immediately becomes a valid attachment owner.
3. FC opens its authenticated declaration.
4. FCO settles the call on M.
5. No public assistant item is required.
Internal history:
M(
    content=<hidden or empty>,
    reasoning_content=<private>,
    tool_calls=[FC]
)
tool(FCO)
Public response output:
R
FC
A new call-only assistant does not need a redundant P(M) because it is already located at the current reasoning position.
A persisted call-only assistant does need P(M) when it must be moved to a later reasoning position.
Direct hidden and public positions remain distinct:
- R(M) M creates hidden M followed by a separate public assistant M.
- FC/FCO after that sequence attach to the later public M and may rehome a
  declaration from hidden M.
- R(M, P(M)) M creates one publicly projected assistant. The local full M
  authenticates private state and declarations, P(M) selects that source, and
  the standalone M supplies public content and refusal.
- A newly generated public-bearing assistant with private state or declarations
  therefore uses R(M, P(M)) M.
- A persisted public-bearing assistant uses R(P(M)) M because its full source is
  already present.
- A public-only assistant with no private state or declarations may be emitted
  as standalone M without reasoning data.
12. Source Timewarp
When processing P(A):
1. Inspect the current authenticated tail.
2. Compare that tail’s authenticated assistant exactly with sealed A.
3. If it matches, detach the live assistant bundle.
4. Move that bundle to the patch’s reasoning position.
5. If it does not match, leave the unrelated tail untouched.
6. If no source matches, stage private state from sealed A.
7. Apply hidden settlement data carried by the current main update.
8. Resolve according to whether A has a public projection.
The live bundle movement includes:
- The assistant.
- Hidden settled calls.
- Hidden outputs.
- Fabricated calls owned by that assistant.
- Fabricated outputs owned by that assistant.
The movement is confined to the main assistant bundle. Sealed non-main FC/FCO
items never become source-owned main material and keep their existing side
attachments when interleaved with a timewarp.
The movement excludes:
- Unrelated fabricated assistants.
- user messages.
- system messages.
- developer messages.
If prior fabricated history was sliced away, the patch does not invent it. Fallback staging contains only what is authenticated by P(A) and its current main update.
An exact repeated P(A) is an explicit revision of the same live bundle. If B
previously consumed P(A), another P(A) moves that bundle again and the next
public assistant C replaces B as its public projection. Canonical model history
contains one bundle rendered as C; B is not retained as a second assistant.
13. Hidden Timewarp
For:
R1(M) RN(P(M)) FC FCO
where M has no public projection:
1. R1(M) establishes hidden authenticated M.
2. RN(P(M)) matches the authenticated tail.
3. The complete live M bundle moves to RN.
4. P(M) resolves immediately because no public message is expected.
5. FC sees hidden M as the nearest eligible owner.
6. FCO settles the call there.
With a plain interposition:
R1(M) FMu RN(P(M)) FC FCO
canonical history becomes:
FMu
M(with FC)
FCO
FMu remains in place because it is not owned by M.
With source-owned fabricated work:
R1(M) FF FFO FMu RN(P(M)) FC FCO
canonical history becomes:
FMu
M(with FF and FC)
FFO
FCO
14. Public Positional Attachment
For:
R(P(A)) M FC FCO
- M consumes the public-bearing patch.
- M receives A.reasoning_content.
- M.content and M.refusal remain public-authoritative.
- FC/FCO attach to M.
For:
R(P(A)) FMa M FC FCO
- FMa consumes the patch.
- FMa receives A.reasoning_content.
- M creates a later assistant position.
- FC claims its authenticated declaration.
- The declaration is rehomed to positional owner M.
- FCO settles the call on M.
Result:
FMa(reasoning_content from A)
M(tool_calls=[FC])
tool(FCO)
For:
R(P(A)) FMu M
- FMu does not consume the patch.
- M consumes it.
Equal public assistant messages remain separate occurrences. No public-content deduplication remains.
15. Hidden Positional Attachment
For:
R(M) FMa FC FCO
- R(M) creates an immediately available hidden assistant M.
- FMa is a later assistant position.
- FC attaches to FMa.
- The authenticated declaration is rehomed from M to FMa.
For:
R(P(M)) FC FCO
- Hidden M resolves immediately.
- M is the nearest assistant owner.
- The call remains on M.
For:
R(P(M)) FMa FC FCO
- Hidden M resolves immediately.
- FMa is a later assistant position.
- FC attaches to FMa.
- The authenticated declaration is rehomed from M to FMa.
Position still wins when a later assistant exists.
16. Attachment Availability
Assistant state
Ordinary hidden R(M)
Output-empty relocated P(M)
Public-bearing P(A) awaiting assistant
Public-bearing patch consumed by assistant
Ordinary standalone assistant
This rejects:
R(P(A)) FC
when A requires a public assistant.
It accepts:
R(P(M)) FC
when no public assistant projection exists.
17. Strict Tool Declarations
Tool declarations remain strict on every side.
For main:
- Full hidden assistants carry declarations.
- MessagePatch carries declarations.
- Active declarations require matching FC items.
- Open calls require matching FCO items.
- A matching FC may rehome its declaration to a later positional assistant.
- Rehoming changes ownership, not declaration authenticity.
- Hidden settled calls remain attached to their hidden/source assistant.
- Missing FC is not interpreted as user deletion.
- Missing FCO is invalid.
For non-main sides:
- Checkpoints carry complete hidden histories and declarations.
- Patch reasoning JSON-patches those histories.
- FC opens the matching declaration.
- FCO closes it.
- Active declarations require public FC items.
- Open calls require public FCO items.
- Inactive declarations may remain parked.
- Sealed non-main FC/FCO never participate in main positional owner selection.
- Main MessagePatch attachment and timewarp never rehome a non-main call.
- Interleaving a non-main FC/FCO with main fabrication or timewarp preserves its
  attachment to the matching declaration on its sealed side.
This maintains symmetric call obligations without giving non-main sides MessagePatch semantics.
18. Main Call Ownership
Main calls choose their owner positionally.
Rules:
- A call attaches to the nearest preceding eligible assistant.
- FMu messages are skipped as owners.
- A later FMa becomes the nearer owner.
- An output-bearing patch awaiting public attachment is not eligible.
- A hidden assistant is eligible.
- A consumed public patch is eligible.
- Matching authenticated declarations retain authenticated metadata.
- Fabricated calls use item-derived metadata.
- Existing main transplant behavior remains.
- Existing stale non-main discard behavior remains.
Call outputs settle the assistant that actually owns the opened call.
19. Canonical Tool-Bundle Ordering
Assistant tool turns must remain contiguous.
For:
M FMu FC FCO
FC attaches backward through FMu to M.
Canonical history:
M(with FC)
FCO
FMu
Raw messages may also appear after FC and before FCO:
M FC FMu FCO
has the same canonical history:
M(with FC)
FCO
FMu
For:
M FMa FC FCO
FC/FCO belong to FMa.
For:
M FC FMa FCO
the already-open call remains owned by M and canonical history is:
M(with FC)
FCO
FMa
For sequential closed turns:
M FC1 FCO1 FMa FC2 FCO2
canonical history is:
M(with FC1)
FCO1
FMa(with FC2)
FCO2
For concurrently open positional turns:
M FC1 FMa FC2 FCO2 FCO1
canonical history is the same two contiguous bundles. FC1 remains owned by M;
FC2 belongs to FMa. Bundle position takes precedence over global output arrival
order when outputs belong to different assistants.
For parallel calls:
M FC1 FC2 FCO2 FCO1
the assistant declaration order remains:
FC1, FC2
The output chronology within that owner bundle remains:
FCO2, FCO1
Outputs are never reordered to declaration order.
20. Main Update Grammar
Reasoning main remains an ordered append transaction supporting:
- Full hidden assistant append.
- Hidden tool outputs settling a persisted assistant.
- Closed hidden prefixes.
- Terminal MessagePatch.
- Partial hidden settlement.
- Delayed text publication.
- Delayed call-only activation.
- Timewarp.
At most one MessagePatch may appear in a reasoning item.
The patch remains terminal in the update.
Leading hidden outputs may bind against:
- The persisted authenticated tail.
- A matching moved source.
- The full assistant carried by P(A) when the prior source was sliced.
Main parsing and main replay must use one canonical transition rather than separate context-free and normalized paths.
21. Producer Behavior
Ingested gains:
checkpoint_required: bool
last_compaction_id: string | null
State selects the reasoning variant once per draft.
Checkpoint generation:
- Serialize the complete current durable object.
- Serialize the complete active set.
- Serialize every current non-main side.
- Exclude main from checkpoint state.
- Generate ordinary append-only main updates.
Patch generation:
- Diff durable state against the ingested base.
- Diff each non-main side against the ingested base.
- Emit optional active replacement.
- Generate the same append-only main updates.
A streamed reasoning draft cannot change from checkpoint to patch or patch to checkpoint during replacement.
22. Main Producer Cases
Main text projection and call projection must be computed separately:
public_message: ResponseMessageItem | None
public_calls: ResponseFunctionCallItem[]
Publication eligibility is structural rather than content-based:
- A new logical assistant appended through State.main is eligible when main is active.
- A persisted HiddenMainTail is eligible when main is active.
- A persisted PublicMainTail is historical and is never automatically eligible.
- A persisted CompactedMainTail is historical and is never automatically eligible.
- An inactive main tail is never eligible.
- Active-to-active remains eligible when State.main contains a genuinely new logical assistant.
- Deactivation and reactivation do not make a public or compacted tail eligible again.
- The complete source retained by HiddenMainTail, not its rendered assistant, is carried by P(A).
Producer forms:
- Newly appended public-bearing assistant with private state or declarations:
  R(M, P(M)), then public M.
- Persisted hidden public-bearing assistant: R(P(M)), then public M.
- Newly appended call-only assistant: R(M), then FC.
- Persisted hidden call-only assistant: R(P(M)), then FC.
- Public-only assistant without private state or declarations: public M only.
- Persisted public and compacted assistants: no automatic patch or public item.
Public output order:
R
optional public message
FC...
Text-bearing turn:
R(P(A))
M
FC
Call-only turn:
R(P(M))
FC
Direct newly generated call-only turn:
R(M)
FC
For stored P(A) B, replay reconstructs PublicMainTail(source=A). A no-op
finalization therefore emits neither R(P(A)) nor B. A caller may still provide
another explicit P(A) C as an intentional revision; ingestion accepts it and
the canonical bundle is rendered with C.
23. Streaming Lineage
StreamCoordinator derives lineage from the reasoning state variant.
Checkpoint:
previous_reasoning_id = null
Patch:
previous_reasoning_id = chain.last_reasoning_id
Every reasoning variant:
previous_compaction_id = chain.last_compaction_id
After successful reasoning completion:
chain.last_reasoning_id = completed_reasoning_id
Draft replacement preserves:
- Reasoning ID.
- Checkpoint/patch variant.
- Previous reasoning ID.
- Previous compaction ID.
- Output index.
- Summary state.
Cancellation retains the current replayable-draft behavior and does not advance the completed chain head.
Compaction tracking is ingestion-owned. The runtime does not emit compaction
items. A route-created StreamCoordinator receives the latest inbound compaction
ID as an immutable seed and echoes it into every reasoning payload it emits.
24. Replay Validation
A reasoning boundary rejects when:
- An open call remains.
- An active declaration remains unclaimed.
- A public-bearing patch remains unresolved.
- A reasoning main update contains an unclosed hidden prefix before a later anchor.
- The reasoning variant is wrong for the current user boundary.
- Patch lineage does not match.
- Checkpoint state contains main.
- Non-main patches target main.
Replay completion permits:
- Inactive parked declarations.
- Hidden assistants without public content.
- Hidden settled tool turns.
- Output-empty P(M) that resolved immediately.
25. Error Contract
Condition	Reason
Patch after user	reasoning_checkpoint_required
Checkpoint without user	reasoning_checkpoint_unexpected
Checkpoint with predecessor	reasoning_checkpoint_has_predecessor
Patch predecessor mismatch	reasoning_previous_reasoning_id_mismatch
Compaction predecessor mismatch	reasoning_previous_compaction_id_mismatch
Public-bearing patch lacks assistant	main_message_patch_target_missing
Active declaration lacks FC	reasoning_tool_call_missing_function_call_item
Open call lacks FCO	function_call_missing_function_call_output
FC targets inactive declaration	inactive_side_function_call
FCO lacks opened owner	function_call_output_without_pending_function_call
Duplicate FC	duplicate_pending_function_call
Reasoning crosses unfinished calls	pending_tool_outputs_block_message
Existing invalid durable patch, invalid side patch, duplicate declaration, compaction, and sealed payload errors remain.
26. Code Structure
Build the replacement subsystem to completion before one coherent cutover.
Files:
- src/plap/responses/ingest/models.py
- src/plap/responses/ingest/main.py
- src/plap/responses/ingest/ingest.py
- src/plap/responses/ingest/sealing.py
- src/plap/responses/state.py
- src/plap/responses/streaming.py
- src/plap/responses/routes.py
main.py will own:
- Positional main nodes.
- Assistant bundles.
- Patch resolution.
- Authenticated source tracking.
- Timewarp.
- Call ownership and rehoming.
- Canonical rendering.
- Main transcript validation.
ingest.py will retain:
- Input normalization.
- Item decoding.
- Compaction slicing.
- Durable and non-main replay.
- Boundary state.
- Side routing.
- Top-level orchestration.
All called functions will remain above their callers per repository convention.
27. Implementation Order
 1. Add version-6 reasoning models.
 2. Add checkpoint and patch state variants.
 3. Move main append data to the common payload field.
 4. Update sealing documentation and exact wire serialization.
 5. Build the new positional main reducer independently.
 6. Add direct hidden-assistant handling.
 7. Add public-bearing and output-empty patch modes.
 8. Add authenticated-tail source matching.
 9. Add live-bundle timewarp.
10. Add declaration claiming and positional rehoming.
11. Add checkpoint-boundary replay.
12. Add durable and non-main checkpoint replacement.
13. Cut ingestion over atomically.
14. Remove the old _Main implementation and normalizer.
15. Update State checkpoint/patch generation.
16. Split public message and public call projection.
17. Update streaming lineage.
18. Update route/runtime construction.
19. Rewrite the main truth table.
20. Remove superseded content-matching tests.
21. Verify downstream plugins.
No compatibility shim or transitional replay path will remain.
28. Required Checkpoint Tests
- First reasoning after user is a checkpoint.
- Patch after user rejects.
- Checkpoint without user rejects.
- Checkpoint has a null predecessor.
- Next patch chains to checkpoint ID.
- Another user requires another checkpoint.
- Patch after compaction accepts a null predecessor.
- Root reasoning without compaction requires a null compaction predecessor.
- Every reasoning after compaction references that compaction ID.
- Reasoning created without compaction rejects after compaction.
- Reasoning anchored to an older compaction rejects after a newer compaction.
- Checkpoints preserve the current compaction anchor while clearing reasoning lineage.
- Checkpoint replaces durable state exactly.
- Checkpoint replaces active membership exactly.
- Checkpoint replaces every non-main side exactly.
- Omitted checkpoint side is removed.
- Explicit empty checkpoint side remains present.
- Checkpoint never snapshots main.
29. Required Public Patch Tests
- Public content may differ from sealed content.
- Public refusal may differ from sealed refusal.
- Public fields come from the public assistant.
- Private reasoning comes from the patch.
- Equal public assistants remain distinct.
- FMu does not consume a patch.
- First FMa consumes a public-bearing patch.
- Public-bearing patch without an assistant rejects.
- Source-matching patch moves its source.
- Source-mismatching patch preserves the unrelated tail.
- Sliced source stages from P(A).
- Staged A never overwrites public content.
- Repeated P(A) replaces the prior public projection on one live bundle.
- Repeated P(A) with equal public fields does not duplicate the bundle.
- Public tail provenance retains source A after A is rendered as B.
- A compaction snapshot re-roots source identity and is classified as historical.
30. Required Hidden Main Tests
- R(M) FC FCO accepts without a public message.
- R(M) FC rejects for missing output.
- Active R(M) without FC rejects.
- R1(M) RN(P(M)) FC FCO timewarps and accepts.
- Sliced-source P(M) FC FCO accepts.
- Mismatched authenticated tail is not removed.
- R1(M) FMu RN(P(M)) FC FCO leaves FMu in place.
- Source-owned fabricated tool work moves with hidden M.
- R(M) FMa FC FCO rehomes the call to FMa.
- R(P(M)) FMa FC FCO rehomes the call to FMa.
- R(P(A)) FC FCO rejects when public attachment is required.
- Persisted call-only activation emits reasoning followed directly by FC.
- Checkpoint reasoning may append a new hidden call-only assistant.
- Patch reasoning may relocate a persisted hidden call-only assistant.
31. Required Call Tests
- R(P(A)) M FC FCO attaches call to M.
- R(P(A)) FMa M FC FCO splits reasoning and call ownership.
- Missing FC rejects.
- Missing FCO rejects.
- Reversed parallel outputs preserve chronology.
- M FMu FC FCO pushes FMu after the tool bundle.
- M FC FMu FCO also pushes FMu after the tool bundle.
- M FMa FC FCO gives ownership to FMa.
- M FC FMa FCO keeps the open call on M and pushes FMa after its output.
- M FC1 FCO1 FMa FC2 FCO2 produces two sequential bundles.
- M FC1 FMa FC2 FCO2 FCO1 produces two positional bundles.
- Rehoming preserves authenticated declaration metadata.
- Non-main pairing remains strict.
- Main timewarp preserves interleaved non-main FC/FCO attachment to its sealed side.
- Inactive declarations remain parked.
- Stale non-main pairs remain discarded.
- Main transplant behavior remains covered.
32. End-to-End Tests
- User response emits checkpoint reasoning.
- Tool continuation emits a chained patch.
- Modified public assistant content reaches model history.
- Private reasoning survives public modification.
- Call-only hidden turns emit no public message.
- Delayed call-only activation emits R, then FC.
- Runtime finalization emits message before calls when both exist.
- An unchanged active public tail emits no output.
- Reactivating a public tail emits no public message or MessagePatch.
- Reactivating a compacted tail emits no public message or MessagePatch.
- Active-to-active with a new State.main assistant still publishes it.
- Cancellation preserves checkpoint/patch variant.
- Stored replay preserves chain boundaries.
- Advisor behavior remains unchanged.
- Vision behavior remains unchanged.
33. Verification
Run targeted tests:
tests/unit/test_responses_ingest.py
tests/unit/test_responses_ingest_truth_table.py
tests/unit/test_responses_runtime.py
tests/unit/test_responses_streaming.py
tests/unit/test_responses_routes.py
tests/unit/test_advisor.py
tests/unit/test_vision.py
Then run:
ruff check
ruff format --check
git diff --check
The full suite remains excluded unless explicitly requested.
34. Completion Criteria
The rebuild is complete only when:
- The first reasoning after user is a checkpoint.
- Tool-call reasoning remains chained.
- Reasoning never checkpoints main.
- Compaction still checkpoints main.
- Public assistant content is positional and editable.
- Sealed public content never validates the consuming public message.
- MessagePatch still authenticates source identity and declarations.
- Direct hidden assistants remain valid attachment owners.
- Output-empty patches resolve without public messages.
- Main and non-main call pairing remain strict.
- Main calls attach to positional assistants.
- Hidden calls fall back to hidden main owners.
- Declaration rehoming works when a later assistant exists.
- Timewarp moves complete assistant-owned tool bundles.
- Plain non-assistant messages are not dragged through timewarp.
- No payload-normalization pass remains.
- No content-hash publication system remains.
- Already-public and compacted tails are not automatically republished.
- Explicit repeated source timewarp revises one bundle rather than duplicating it.
- No old replay or compatibility path remains.
