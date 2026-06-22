"""Replay ingest truth table.

Legend

- `R(h)`: reasoning item whose hidden main anchor exists and declares no
  main-side `tool_calls`
- `R(h[up_0,...])`: reasoning item whose hidden main anchor declares those
  main-side `tool_calls` in that order
- `R(h-empty-main)`: same as `R(h)`, but the hidden anchor has empty public
  content
- `R(h-empty-main[up_0,...])`: same as `R(h[up_0,...])`, but the hidden anchor
  has empty public content
- `R(c)`: reasoning item whose main side is fully closed and contributes only
  carrier state before later standalone main replay
- `R(p->M)`: reasoning patch targeting explicit assistant `M` and declaring no
  main-side `tool_calls`
- `R(p->M[up_0,...])`: reasoning patch targeting explicit assistant `M` and
  declaring those main-side `tool_calls` in that order
- `M`: explicit assistant anchor message
- `FMa`: fabricated assistant message
- `FMu`: fabricated non-assistant message (`user` / `system` / `developer`)
- `FF / FFO`: fabricated `function_call` / `function_call_output`
- `F / FO`: sealed `function_call` / `function_call_output`
- `F1 / FO1`, `F2 / FO2`: multiple sealed calls in the same anchored bundle

Bundle model

Replay normalization operates on bundles, not on one region-wide anchor that
survives until the next reasoning item.

The normalized shape of one bundle is:

- `pre=[...]`: ordered clusters hoisted immediately before the anchor
- `anchor=...`: the anchor assistant
- `slots=[...]`: the anchor assistant's rendered `tool_calls`
- `outputs=[...]`: the anchor-owned `role: tool` messages in real chronology
- `post=[...]`: ordered clusters that remain in the same bundle after the
  anchor

`pre` exists only for pending-patch / hidden-prelude hoists.
`post` exists only while a later assistant has clustered into the current
bundle instead of superseding it.

Anchor call model

For any anchor `A`, define:

- `declared(A)`:
  - the ordered tool-call ids declared by the anchor itself
  - hidden anchors take them from the hidden assistant message in the reasoning
    payload
  - patch anchors take them from `R(p->M[up_0,...])` once `M` resolves
  - standalone public anchors start with `declared(A)=[]`
- `added(A)`:
  - the ordered direct calls later attached to `A` during standalone replay
    that are not already in `declared(A)`
  - includes sealed direct calls
  - includes fabricated direct calls
  - preserves first-arrival order across both kinds
- `slots(A) = declared(A) + added(A)`

A sealed direct call whose id is already in `declared(A)` replays that declared
slot; it does not create a second slot and it does not move the declared
segment.

- `free_anchor`: `declared(A)=[]`, so `slots(A)=added(A)`
- `strict_anchor`: `declared(A)!=[]`, so `slots(A)` begins with the declared
  segment and only then appends later additions

Release and settlement

- `pending_patch`
  - `R(p->M[...])` has appeared but `M` has not resolved yet.
- `released`
  - a pending patch is not released
  - otherwise, every id in `declared(A)` has a matching output
  - a free anchor is therefore released immediately once its anchor exists
- `settled`
  - no open call remains anywhere in the current bundle
  - this includes declared anchor calls, added anchor calls, and clustered
    assistant mini-turn calls

Turnover rules

- A later assistant starts a new anchor only when the current bundle is both
  released and settled.
- If the current bundle is not released or not settled, a later assistant stays
  inside that bundle as a cluster.
- A later reasoning item still forces finalization, but only if the current
  bundle is already settled and there is no pending patch.
- Non-assistant messages never supersede the current anchor.

Cluster kinds

- Plain message cluster:
  - a `user` / `system` / `developer` message
  - or an assistant message with no attached calls
- Assistant mini-turn cluster:
  - an assistant message
  - zero or more calls attached to that assistant
  - all corresponding tool outputs
- Anchor bundle:
  - the anchor assistant
  - its `declared(A)` segment
  - its `added(A)` segment
  - all corresponding tool outputs
  - hidden reasoning/tool rows if the anchor came from `R`

Call attachment

- A direct call first looks for the nearest preceding assistant within the local
  bundle parse.
- If such an assistant exists, the call belongs to that assistant mini-turn
  cluster.
- Otherwise the call belongs directly to the current anchor.
- A pending patch has no resolved anchor assistant yet, so before `M` appears a
  direct call without a preceding assistant has no owner and replay rejects.
- If a direct sealed call belongs to the anchor and its id is already in
  `declared(A)`, it replays that declared slot.
- Otherwise the direct call contributes the next entry in `added(A)`.
- Once a call or output attaches to an assistant mini-turn, that mini-turn is
  indivisible for chronology purposes.

Normalized form

- `A(FMa)`: assistant message cluster with no attached calls
- `A(FMa; FF,FFO)`: assistant cluster with attached mini-turn
- `P(FMu)`: plain non-assistant message cluster
- `pre=[...]`: clusters hoisted immediately before the anchor assistant
- `anchor=...`
- `slots=[...]`: assistant `tool_calls` order
- `outputs=[...]`: `role: tool` message chronology
- `post=[...]`: clusters appended after the anchored bundle settles

Invariants

- If `FF/FFO` attach to an assistant cluster, that assistant cluster becomes
  contiguous.
- `outputs` preserve real chronology. So `F1 F2 FO2 FO1` keeps
  `outputs=[FO2,FO1]` and is not reordered.
- Direct anchor slot order never reorders later additions by call kind. Only
  the declared segment may stay ahead of later additions.
- `free_anchor` later assistants supersede immediately once the current bundle
  is settled.
- `strict_anchor` later assistants cluster until the current bundle is both
  released and settled.

Replay cases

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
  Notes: Hidden free anchor. `F` becomes the first added slot.

- Raw: `R(h[up_0]) F(up_0) FO(up_0)`
  Normalized: `pre=[] anchor=R.hidden slots=[F(up_0)] outputs=[FO(up_0)] post=[]`
  Outcome: Accept
  Notes: Hidden strict anchor. The sealed call replays the declared slot.

- Raw: `R(h-empty-main) F FO`
  Normalized: `pre=[] anchor=R.hidden(empty-main) slots=[F] outputs=[FO] post=[]`
  Outcome: Accept
  Notes: Hidden free anchor with empty public content. Anchored, not synthetic.

- Raw: `R(h-empty-main[up_0]) F(up_0) FO(up_0)`
  Normalized:
  `pre=[] anchor=R.hidden(empty-main) slots=[F(up_0)] outputs=[FO(up_0)] post=[]`
  Outcome: Accept
  Notes: Hidden strict anchor with empty public content. The sealed call replays the declared slot.

- Raw: `R(p->M[up_0]) M F(up_0) FO(up_0)`
  Normalized: `pre=[] anchor=M slots=[F(up_0)] outputs=[FO(up_0)] post=[]`
  Outcome: Accept
  Notes: Patch resolves to `M`.

- Raw: `M F FO` after stripping `R(p->M[up_0])`
  Normalized: `pre=[] anchor=M slots=[F] outputs=[FO] post=[]`
  Outcome: Accept
  Notes: Sealed targeting is based on `M`, not `R+M`.

2. Pre-anchor plain message hoists

Hidden and patch-anchor rows in sections 2-5 use explicit declared ids when the
row is testing hoist or attachment rather than free-anchor direct-call order.

- Raw: `R(h-empty-main[up_0]) FMu F(up_0) FO(up_0)`
  Normalized:
  `pre=[P(FMu)] anchor=R.hidden(empty-main) slots=[F(up_0)] outputs=[FO(up_0)] post=[]`
  Outcome: Accept
  Notes: Effective visible order `FMu F FO`.

- Raw: `R(h-empty-main[up_0]) FMu FMu2 F(up_0) FO(up_0)`
  Normalized:
  `pre=[P(FMu),P(FMu2)] anchor=R.hidden(empty-main) slots=[F(up_0)] outputs=[FO(up_0)] post=[]`
  Outcome: Accept
  Notes: Preserve order.

- Raw: `R(h-empty-main[up_0]) FMa F(up_0) FO(up_0)`
  Normalized:
  `pre=[A(FMa)] anchor=R.hidden(empty-main) slots=[F(up_0)] outputs=[FO(up_0)] post=[]`
  Outcome: Accept
  Notes: Assistant plain message before hidden anchor.

- Raw: `R(p->M[up_0]) FMu M F(up_0) FO(up_0)`
  Normalized: `pre=[P(FMu)] anchor=M slots=[F(up_0)] outputs=[FO(up_0)] post=[]`
  Outcome: Accept
  Notes: Hoist before `M`, not before whole `R`.

- Raw: `R(p->M[up_0]) FMu FMu2 M F(up_0) FO(up_0)`
  Normalized: `pre=[P(FMu),P(FMu2)] anchor=M slots=[F(up_0)] outputs=[FO(up_0)] post=[]`
  Outcome: Accept
  Notes: Preserve order.

- Raw: `R(p->M[up_0]) FMa M F(up_0) FO(up_0)`
  Normalized: `pre=[A(FMa)] anchor=M slots=[F(up_0)] outputs=[FO(up_0)] post=[]`
  Outcome: Accept
  Notes: Assistant plain message before `M`.

3. Pre-anchor assistant mini-bundles

- Raw: `R(h-empty-main[up_0]) FMa FF FFO F(up_0) FO(up_0)`
  Normalized:
  `pre=[A(FMa;FF,FFO)] anchor=R.hidden(empty-main) slots=[F(up_0)] outputs=[FO(up_0)] post=[]`
  Outcome: Accept
  Notes: `FF/FFO` bind to `FMa`, sealed pair binds to hidden anchor.

- Raw: `R(h-empty-main[up_0]) FMa F(tx) FO(tx) F(up_0) FO(up_0)`
  Normalized:
  `pre=[A(FMa;F(tx),FO(tx))] anchor=R.hidden(empty-main) slots=[F(up_0)] outputs=[FO(up_0)] post=[]`
  Outcome: Accept
  Notes: Sealed transplant binds to `FMa`, declared sealed pair binds to the hidden anchor.

- Raw: `R(p->M[up_0]) FMa FF FFO M F(up_0) FO(up_0)`
  Normalized: `pre=[A(FMa;FF,FFO)] anchor=M slots=[F(up_0)] outputs=[FO(up_0)] post=[]`
  Outcome: Accept
  Notes: Same, explicit anchor.

- Raw: `R(p->M[up_0]) FMa F(tx) FO(tx) M F(up_0) FO(up_0)`
  Normalized: `pre=[A(FMa;F(tx),FO(tx))] anchor=M slots=[F(up_0)] outputs=[FO(up_0)] post=[]`
  Outcome: Accept
  Notes: Same, explicit patch target.

4. Pre-anchor mixed `FMa` / `FMu`

- Raw: `R(h-empty-main[up_0]) FMu FMa FF FFO F(up_0) FO(up_0)`
  Normalized:
  `pre=[P(FMu),A(FMa;FF,FFO)] anchor=R.hidden(empty-main) slots=[F(up_0)] outputs=[FO(up_0)] post=[]`
  Outcome: Accept
  Notes: Plain message stays before assistant cluster.

- Raw: `R(h-empty-main[up_0]) FMa FMu FF FFO F(up_0) FO(up_0)`
  Normalized:
  `pre=[A(FMa;FF,FFO),P(FMu)] anchor=R.hidden(empty-main) slots=[F(up_0)] outputs=[FO(up_0)] post=[]`
  Outcome: Accept
  Notes: `FMu` cannot split `FMa` from `FF/FFO`.

- Raw: `R(p->M[up_0]) FMu FMa FF FFO M F(up_0) FO(up_0)`
  Normalized:
  `pre=[P(FMu),A(FMa;FF,FFO)] anchor=M slots=[F(up_0)] outputs=[FO(up_0)] post=[]`
  Outcome: Accept
  Notes: Explicit anchor.

- Raw: `R(p->M[up_0]) FMa FMu FF FFO M F(up_0) FO(up_0)`
  Normalized:
  `pre=[A(FMa;FF,FFO),P(FMu)] anchor=M slots=[F(up_0)] outputs=[FO(up_0)] post=[]`
  Outcome: Accept
  Notes: `FMu` pushed outside attached assistant cluster.

5. Pre-anchor non-assistant plus fabricated fallback to the hidden anchor

These are attachment rows. The hidden anchor is declared explicitly so the row
tests which anchor owns `FF/FFO`, not free-anchor direct-call ordering.

- Raw: `R(h-empty-main[up_0]) FMu FF FFO F(up_0) FO(up_0)`
  Normalized:
  `pre=[P(FMu)] anchor=R.hidden(empty-main) slots=[F(up_0),FF] outputs=[FFO,FO(up_0)] post=[]`
  Outcome: Accept
  Notes: No assistant before `FF`, so `FF/FFO` fall back to the hidden anchor.

- Raw: `R(h-empty-main[up_0]) FMu FMu2 FF FFO F(up_0) FO(up_0)`
  Normalized:
  `pre=[P(FMu),P(FMu2)] anchor=R.hidden(empty-main) slots=[F(up_0),FF] outputs=[FFO,FO(up_0)] post=[]`
  Outcome: Accept
  Notes: Same fallback.

- Raw: `R(p->M[up_0]) FMu FF FFO M F(up_0) FO(up_0)`
  Normalized: N/A
  Outcome: Reject
  Notes: No assistant exists before `M` for `FF` to bind to.

- Raw: `R(p->M[up_0]) FMu FMu2 FF FFO M F(up_0) FO(up_0)`
  Normalized: N/A
  Outcome: Reject
  Notes: Same reason.

- Raw: `R(p->M[up_0]) F(tx) FO(tx) M F(up_0) FO(up_0)`
  Normalized: N/A
  Outcome: Reject
  Notes: No assistant exists before `M` for the sealed transplant to bind to.

- Raw: `R(p->M[up_0]) FMu F(tx) FO(tx) M F(up_0) FO(up_0)`
  Normalized: N/A
  Outcome: Reject
  Notes: `FMu` is not an assistant anchor for the sealed transplant.

6. Mixed direct calls on one anchor

6A. Hidden strict-anchor attachment

- Raw: `R(h[up_0]) FF FFO F(up_0) FO(up_0)`
  Normalized: `pre=[] anchor=R.hidden slots=[F(up_0),FF] outputs=[FFO,FO(up_0)] post=[]`
  Outcome: Accept
  Notes: Hidden strict anchor. `FF/FFO` fall back to the anchor while the sealed replay item replays the declared slot.

- Raw: `R(h[up_0,up_1]) FF FFO F(up_0) FO(up_0) FF2 FFO2 F(up_1) FO(up_1)`
  Normalized:
  `pre=[] anchor=R.hidden slots=[F(up_0),F(up_1),FF,FF2] outputs=[FFO,FO(up_0),FFO2,FO(up_1)] post=[]`
  Outcome: Accept
  Notes: The declared segment stays first; later direct additions keep arrival order.

- Raw: `R(h-empty-main[up_0]) FF FFO F(up_0) FO(up_0)`
  Normalized:
  `pre=[] anchor=R.hidden(empty-main) slots=[F(up_0),FF] outputs=[FFO,FO(up_0)] post=[]`
  Outcome: Accept
  Notes: Same on hidden-empty strict anchor.

- Raw: `R(h-empty-main[up_0,up_1]) FF FFO F(up_0) FO(up_0) FF2 FFO2 F(up_1) FO(up_1)`
  Normalized:
  `pre=[] anchor=R.hidden(empty-main) slots=[F(up_0),F(up_1),FF,FF2] outputs=[FFO,FO(up_0),FFO2,FO(up_1)] post=[]`
  Outcome: Accept
  Notes: Same generalized bundle with empty public content.

6B. Free explicit-anchor order

- Raw: `R(c) M FF FFO F FO`
  Normalized: `pre=[] anchor=M slots=[FF,F] outputs=[FFO,FO] post=[]`
  Outcome: Accept
  Notes: Free explicit anchor. With no declared segment, direct calls stay in arrival order.

- Raw: `R(c) M FF FFO F FO FF2 FFO2 F2 FO2`
  Normalized:
  `pre=[] anchor=M slots=[FF,F,FF2,F2] outputs=[FFO,FO,FFO2,FO2] post=[]`
  Outcome: Accept
  Notes: Same rule generalized across multiple direct additions.

7. Free-anchor bundles

A free anchor has `declared(A)=[]`.

- Therefore `slots(A)=added(A)`.
- Direct anchor-owned calls appear in first-arrival order across both sealed
  and fabricated calls.

- A later assistant starts a new anchor immediately once the current bundle is
  settled.
- A non-assistant message never supersedes the current anchor.
- Fabricated calls may attach to the current free anchor when there is no later
  assistant mini-turn to own them.

7A. Free-anchor plain continuations

- Raw: `R(c) M FMu F FO`
  Normalized: `pre=[] anchor=M slots=[F] outputs=[FO] post=[P(FMu)]`
  Outcome: Accept
  Notes: Plain non-assistant messages never supersede the current anchor.

- Raw: `R(c) M FMu FMu2 F FO`
  Normalized: `pre=[] anchor=M slots=[F] outputs=[FO] post=[P(FMu),P(FMu2)]`
  Outcome: Accept
  Notes: Preserve order.

- Raw: `R(c) M FMa F FO`
  Normalized:
  `bundle1 anchor=M slots=[] outputs=[]`
  `bundle2 anchor=FMa slots=[F] outputs=[FO]`
  Outcome: Accept
  Notes: The later assistant supersedes immediately because the free anchor is already settled.

7B. Free-anchor assistant mini-turns

- Raw: `R(c) M FMa FF FFO F FO`
  Normalized:
  `bundle1 anchor=M slots=[] outputs=[]`
  `bundle2 anchor=FMa slots=[FF,F] outputs=[FFO,FO]`
  Outcome: Accept
  Notes: The new free anchor has no declared segment, so its direct calls stay in arrival order while outputs keep chronology.

- Raw: `R(c) M FMa FF FFO F1 F2 FO2 FO1`
  Normalized:
  `bundle1 anchor=M slots=[] outputs=[]`
  `bundle2 anchor=FMa slots=[FF,F1,F2] outputs=[FFO,FO2,FO1]`
  Outcome: Accept
  Notes: The later assistant owns both fabricated and sealed calls, and free-anchor direct order follows arrival order across both kinds.

- Raw: `R(c) M FMa1 FF1 FFO1 FMa2 FF2 FFO2 F FO`
  Normalized:
  `bundle1 anchor=M slots=[] outputs=[]`
  `bundle2 anchor=FMa1 slots=[FF1] outputs=[FFO1]`
  `bundle3 anchor=FMa2 slots=[FF2,F] outputs=[FFO2,FO]`
  Outcome: Accept
  Notes: Each settled assistant mini-turn hands off to the next assistant anchor.

7C. Free-anchor mixed `FMa` / `FMu`

- Raw: `R(c) M FMu FMa FF FFO F FO`
  Normalized:
  `bundle1 anchor=M slots=[] outputs=[] post=[P(FMu)]`
  `bundle2 anchor=FMa slots=[FF,F] outputs=[FFO,FO]`
  Outcome: Accept
  Notes: Plain message stays between bundles; the later assistant becomes the new anchor.

- Raw: `R(c) M FMa FMu FF FFO F FO`
  Normalized:
  `bundle1 anchor=M slots=[] outputs=[]`
  `bundle2 anchor=FMa slots=[FF,F] outputs=[FFO,FO] post=[P(FMu)]`
  Outcome: Accept
  Notes: `FMu` cannot split the attached assistant mini-turn.

- Raw: `R(c) M FMa FMu FF FFO F1 F2 FO2 FO1`
  Normalized:
  `bundle1 anchor=M slots=[] outputs=[]`
  `bundle2 anchor=FMa slots=[FF,F1,F2] outputs=[FFO,FO2,FO1] post=[P(FMu)]`
  Outcome: Accept
  Notes: Same with parallel sealed calls.

- Raw: `R(c) M FMa1 FMu1 FF1 FFO1 FMa2 FMu2 FF2 FFO2 F FO`
  Normalized:
  `bundle1 anchor=M slots=[] outputs=[]`
  `bundle2 anchor=FMa1 slots=[FF1] outputs=[FFO1] post=[P(FMu1)]`
  `bundle3 anchor=FMa2 slots=[FF2,F] outputs=[FFO2,FO] post=[P(FMu2)]`
  Outcome: Accept
  Notes: Each interleaved plain message remains outside the assistant mini-turn that follows it.

7D. Free-anchor non-assistant fallback

- Raw: `R(c) M FMu FF FFO F FO`
  Normalized: `pre=[] anchor=M slots=[FF,F] outputs=[FFO,FO] post=[P(FMu)]`
  Outcome: Accept
  Notes: With no later assistant, `FF/FFO` fall back to `M`.

- Raw: `R(c) M FMu FMu2 FF FFO F FO`
  Normalized:
  `pre=[] anchor=M slots=[FF,F] outputs=[FFO,FO] post=[P(FMu),P(FMu2)]`
  Outcome: Accept
  Notes: Same fallback.

- Raw: `R(c) M FMu FF FFO F1 F2 FO2 FO1`
  Normalized:
  `pre=[] anchor=M slots=[FF,F1,F2] outputs=[FFO,FO2,FO1] post=[P(FMu)]`
  Outcome: Accept
  Notes: Sealed outputs preserve chronology.

- Raw: `R(c) M FMu FF FFO F FO FF2 FFO2 F2 FO2`
  Normalized:
  `pre=[] anchor=M slots=[FF,F,FF2,F2] outputs=[FFO,FO,FFO2,FO2] post=[P(FMu)]`
  Outcome: Accept
  Notes: All fabricated calls bind to the current free anchor because no later assistant supersedes it.

7E. Free-anchor assistant arrival before settlement

- Raw: `M F M2 FO FF FFO`
  Normalized:
  `pre=[] anchor=M slots=[F] outputs=[FO] post=[A(M2;FF,FFO)]`
  Outcome: Accept
  Notes: The later assistant arrives while the current free bundle is unsettled, so it clusters.

- Raw: `M F FO M2 FF FFO`
  Normalized:
  `bundle1 anchor=M slots=[F] outputs=[FO]`
  `bundle2 anchor=M2 slots=[FF] outputs=[FFO]`
  Outcome: Accept
  Notes: Once settled, the next assistant starts a new anchor immediately.

- Raw: `M F FO M2 F2 FO2 M3 F3 FO3`
  Normalized:
  `bundle1 anchor=M slots=[F] outputs=[FO]`
  `bundle2 anchor=M2 slots=[F2] outputs=[FO2]`
  `bundle3 anchor=M3 slots=[F3] outputs=[FO3]`
  Outcome: Accept
  Notes: Stripped replay forms repeated free bundles, not synthetic anchor splits.

8. Strict-anchor bundles

A strict anchor has `declared(A)!=[]`. In this truth table, strict anchors come
from hidden main anchors or resolved patches that declare main-side
`tool_calls`.

- Therefore `slots(A)=declared(A)+added(A)`.
- The declared segment keeps the order supplied by hidden reasoning state or by
  `R(p->M[up_0,...])`.
- Later direct anchor additions append after that declared segment in
  first-arrival order.

- While any declared hidden tool call is unresolved, later assistants cluster.
- Once all declared hidden tool calls are satisfied and the bundle is settled,
  the next assistant starts a new anchor immediately.
- Fabricated calls may still attach to the current anchor after release as long
  as no new assistant has superseded it yet.

8A. Strict-anchor plain clustered continuations

- Raw: `R(p->M[up_0]) M FMu F FO`
  Normalized: `pre=[] anchor=M slots=[F] outputs=[FO] post=[P(FMu)]`
  Outcome: Accept
  Notes: Plain non-assistant messages never supersede the strict anchor.

- Raw: `R(p->M[up_0]) M FMu FMu2 F FO`
  Normalized: `pre=[] anchor=M slots=[F] outputs=[FO] post=[P(FMu),P(FMu2)]`
  Outcome: Accept
  Notes: Preserve order.

- Raw: `R(p->M[up_0]) M post F FO`
  Normalized: `pre=[] anchor=M slots=[F] outputs=[FO] post=[A(post)]`
  Outcome: Accept
  Notes: The later assistant plain message clusters while the declared strict call is unresolved.

8B. Strict-anchor assistant mini-turn clusters

- Raw: `R(p->M[up_0]) M post FF FFO F FO`
  Normalized: `pre=[] anchor=M slots=[F] outputs=[FO] post=[A(post;FF,FFO)]`
  Outcome: Accept
  Notes: The clustered assistant keeps its fabricated mini-turn contiguous.

- Raw: `R(p->M[up_0,up_1]) M post FF FFO F1 F2 FO2 FO1`
  Normalized:
  `pre=[] anchor=M slots=[F1,F2] outputs=[FO2,FO1] post=[A(post;FF,FFO)]`
  Outcome: Accept
  Notes: The later assistant clusters and owns its fabricated mini-turn while the strict calls remain unresolved.

- Raw: `R(p->M[up_0]) M post1 FF1 FFO1 post2 FF2 FFO2 F FO`
  Normalized:
  `pre=[] anchor=M slots=[F] outputs=[FO] post=[A(post1;FF1,FFO1),A(post2;FF2,FFO2)]`
  Outcome: Accept
  Notes: Multiple later assistants remain separate clustered mini-turns while the strict call is unresolved.

8C. Strict-anchor mixed `FMa` / `FMu`

- Raw: `R(p->M[up_0]) M FMu post FF FFO F FO`
  Normalized:
  `pre=[] anchor=M slots=[F] outputs=[FO] post=[P(FMu),A(post;FF,FFO)]`
  Outcome: Accept
  Notes: Plain message stays before the clustered assistant mini-turn.

- Raw: `R(p->M[up_0,up_1]) M post FMu FF FFO F1 F2 FO2 FO1`
  Normalized:
  `pre=[] anchor=M slots=[F1,F2] outputs=[FO2,FO1] post=[A(post;FF,FFO),P(FMu)]`
  Outcome: Accept
  Notes: The later assistant still clusters because the declared sealed calls are unresolved.

- Raw: `R(p->M[up_0]) M post1 FMu1 FF1 FFO1 post2 FMu2 FF2 FFO2 F FO`
  Normalized:
  `pre=[] anchor=M slots=[F] outputs=[FO] post=[A(post1;FF1,FFO1),P(FMu1),A(post2;FF2,FFO2),P(FMu2)]`
  Outcome: Accept
  Notes: Each plain message stays outside the clustered assistant mini-turn that follows it.

8D. Strict-anchor non-assistant fallback

- Raw: `R(p->M[up_0]) M FMu FF FFO F FO`
  Normalized: `pre=[] anchor=M slots=[F,FF] outputs=[FFO,FO] post=[P(FMu)]`
  Outcome: Accept
  Notes: With no later assistant before `FF`, fabricated fallback still binds to the strict anchor.

- Raw: `R(p->M[up_0]) M FMu FMu2 FF FFO F FO`
  Normalized:
  `pre=[] anchor=M slots=[F,FF] outputs=[FFO,FO] post=[P(FMu),P(FMu2)]`
  Outcome: Accept
  Notes: Same fallback.

- Raw: `R(p->M[up_0,up_1]) M FMu FF FFO F1 F2 FO2 FO1`
  Normalized:
  `pre=[] anchor=M slots=[F1,F2,FF] outputs=[FFO,FO2,FO1] post=[P(FMu)]`
  Outcome: Accept
  Notes: Parallel strict calls preserve chronology while fabricated fallback still binds to the anchor.

- Raw: `R(p->M[up_0,up_1]) M FMu FF FFO F(up_0) FO(up_0) FF2 FFO2 F(up_1) FO(up_1)`
  Normalized:
  `pre=[] anchor=M slots=[F(up_0),F(up_1),FF,FF2] outputs=[FFO,FO(up_0),FFO2,FO(up_1)] post=[P(FMu)]`
  Outcome: Accept
  Notes: Fabricated fallback remains on the same strict anchor until a later assistant supersedes it.

8E. Strict-anchor release and turnover

- Raw: `R(p->M[up_0]) M F FO FF FFO`
  Normalized: `pre=[] anchor=M slots=[F,FF] outputs=[FO,FFO] post=[]`
  Outcome: Accept
  Notes: Satisfying the declared strict call releases the anchor, but fabricated fallback may still
  attach until a later assistant supersedes it.

- Raw: `R(p->M[up_0]) M F FO post FF FFO`
  Normalized:
  `bundle1 anchor=M slots=[F] outputs=[FO]`
  `bundle2 anchor=post slots=[FF] outputs=[FFO]`
  Outcome: Accept
  Notes: After release and settlement, the later assistant starts a new anchor.

- Raw: `R(p->M[up_0]) M F FO post F2 FO2`
  Normalized:
  `bundle1 anchor=M slots=[F] outputs=[FO]`
  `bundle2 anchor=post slots=[F2] outputs=[FO2]`
  Outcome: Accept
  Notes: The later assistant also owns later sealed calls once the strict anchor has settled.

- Raw: `R(p->M[up_0,up_1]) M F(up_0) FO(up_0) post FF FFO F(up_1) FO(up_1)`
  Normalized:
  `pre=[] anchor=M slots=[F(up_0),F(up_1)] outputs=[FO(up_0),FO(up_1)] post=[A(post;FF,FFO)]`
  Outcome: Accept
  Notes: Partial satisfaction keeps the later assistant clustered until all declared strict calls are satisfied.

9. Later reasoning finalization

- Raw: `R(p->M[up_0]) M F R2(...)`
  Normalized: N/A
  Outcome: Reject
  Notes: Later reasoning cannot finalize a bundle with unresolved declared outputs.

- Raw: `R(p->M[up_0]) M F FO FF R2(...)`
  Normalized: N/A
  Outcome: Reject
  Notes: Later reasoning cannot finalize a bundle with unresolved fabricated calls either.

- Raw: `R(p->M[up_0]) M F FO FF FFO R2(...)`
  Normalized:
  `bundle1 anchor=M slots=[F,FF] outputs=[FO,FFO]`
  `then next reasoning bundle`
  Outcome: Accept
  Notes: Later reasoning still forces closure, but only after the current bundle is settled.

Replay rejections that stay rejected

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

- Raw: `Any case requiring retroactive reopen of an older anchor after a newer`
  `bundle has started`
  Normalized: N/A
  Outcome: Reject
  Notes: Replay has no retroactive reopen.

- Raw: `Any case where M never appears for R(p->M)`
  Normalized: N/A
  Outcome: Reject
  Notes: Patch target missing.

- Raw: `Sealed output for a call id that is not pending in the current bundle`
  Normalized: N/A
  Outcome: Reject
  Notes: Output must match an actually pending call id.

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

Synthetic main recovery is removed.

- Raw: `F FO on main`
  Normalized: N/A
  Outcome: Reject
  Notes: No naked main sealed replay without a real anchor.

- Raw: `Stripped R(h-empty-main) F FO -> F FO`
  Normalized: N/A
  Outcome: Reject
  Notes: No synthetic stripped-anchor recovery when no current anchor exists.

- Raw: `R(c) M R(h-empty-main) F FO stripped into M F FO`
  Normalized: `pre=[] anchor=M slots=[F] outputs=[FO] post=[]`
  Outcome: Accept
  Notes: With an existing free anchor, stripped sealed replay attaches to that current anchor instead of synthesizing a new one.

- Raw: `Naked reviewer F FO`
  Normalized: N/A
  Outcome: Silent discard
  Notes: No main-side semantics here; non-main naked sealed pairs are still dropped.

- Raw: `Naked arbitrator F FO`
  Normalized: N/A
  Outcome: Silent discard
  Notes: Same.

- Raw: `Reviewer side already contains unrelated closed history, then reviewer F FO for a missing call id`
  Normalized: reviewer side unchanged
  Outcome: Silent discard
  Notes: Non-main sealed replay is discarded whenever the side has no matching call, not only when the side is empty.

- Raw: `Arbitrator side already contains unrelated closed history, then arbitrator F FO for a missing call id`
  Normalized: arbitrator side unchanged
  Outcome: Silent discard
  Notes: Same discard rule.

Quick validation rules

- `R(p->M[up_0]) M F(up_0) FO(up_0) -> M F FO`: yes
- `R(p->M[up_0]) FMu M F(up_0) FO(up_0)`: hoist `FMu` immediately before `M`, not before whole `R`
- `F1 F2 FO2 FO1`: keep `outputs=[FO2,FO1]`, do not reorder to slot order
- `R(h[up_0]) FF FFO F(up_0) FO(up_0)`: attachment row, so the hidden declared slot stays first
- `R(p->M[up_0]) FMa F(tx) FO(tx) M F(up_0) FO(up_0)`: sealed transplant binds to `FMa`; the declared slot stays on `M`
- `R(c) M FF FFO F FO`: free explicit anchor, so direct calls stay in arrival order as `slots=[FF,F]`
- `M F M2 FO FF FFO`: `M2` clusters because the first free bundle is unsettled
- `M F FO M2 FF FFO`: `M2` starts a new bundle because the first free bundle is settled
- `R(p->M[up_0]) M F(up_0) FO(up_0) FF FFO`: `FF/FFO` may still attach to `M` until a later assistant supersedes it
- `R(p->M[up_0,up_1]) M F(up_0) FO(up_0) M2 ...`: `M2` still clusters because the strict bundle is only partially satisfied
- later `R` forces finalization but does not define anchor lifetime by itself
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import count

import pytest

from plap.errors import PlapError
from plap.keyring import SealingKeyring
from plap.responses.contracts import (
    RequestFunctionCallItem,
    RequestFunctionCallOutputItem,
    RequestInputItem,
    RequestMessageItem,
    RequestReasoningItem,
    ResponseCreateRequest,
    SummaryTextContent,
)
from plap.responses.ingest.ingest import ingest_response_request as _ingest_response_request_impl
from plap.responses.ingest.models import (
    MAIN_SIDE,
    CallID,
    GuardedPatch,
    Message,
    MessagePatch,
    ReasoningPayload,
    Side,
    Sides,
    SidesUpdate,
    ToolCall,
)
from plap.responses.ingest.sealing import (
    open_reasoning_payload,
    seal_reasoning_payload,
)
from plap.responses.ingest.sealing import (
    seal_call_id as _seal_call_id_impl,
)


@dataclass(frozen=True, slots=True)
class _AcceptCase:
    items: list[RequestInputItem]
    expected_main: list[dict[str, object]]


@dataclass(frozen=True, slots=True)
class _RejectCase:
    items: list[RequestInputItem]
    expected_reason: str


@dataclass(frozen=True, slots=True)
class _DiscardCase:
    items: list[RequestInputItem]
    side: Side
    expected_side: list[dict[str, object]]


def _keyring() -> SealingKeyring:
    return SealingKeyring(roots=(b"i" * 32,))


def _side_codes() -> dict[str, int]:
    return {
        MAIN_SIDE: 0,
        "defender": 1,
        "reviewer": 2,
        "arbitrator": 3,
    }


async def ingest_response_request(request: ResponseCreateRequest, *, keyring: SealingKeyring):
    return await _ingest_response_request_impl(request, keyring=keyring, side_codes=_side_codes())


def seal_call_id(value: CallID, *, keyring: SealingKeyring) -> str:
    return _seal_call_id_impl(value, keyring=keyring, side_codes=_side_codes())


_REASONING_PAYLOAD_COUNTER = count()


def _next_reasoning_payload_id() -> str:
    return f"rs_truth_{next(_REASONING_PAYLOAD_COUNTER)}"


def _reasoning_payload(
    *,
    machine: list[dict[str, object]],
    sides: SidesUpdate,
    payload_id: str | None = None,
    previous_reasoning_id: str | None = None,
    previous_compaction_id: str | None = None,
) -> ReasoningPayload:
    return ReasoningPayload(
        id=payload_id or _next_reasoning_payload_id(),
        previous_reasoning_id=previous_reasoning_id,
        previous_compaction_id=previous_compaction_id,
        machine=machine,
        sides=sides,
    )


def _sides_update(
    *,
    main: list[Message | MessagePatch] | None = None,
    patches: dict[Side, list[dict[str, object]] | None] | None = None,
    current: Sides | None = None,
) -> SidesUpdate:
    current_sides = Sides() if current is None else current
    normalized_patches = {
        side: _guarded_patch(side, current_sides.get(side), patch) for side, patch in ({} if patches is None else patches).items()
    }
    return SidesUpdate(main=[] if main is None else list(main), patches=normalized_patches)


def _guarded_patch(side: Side, current: list[Message] | None, patch: list[dict[str, object]] | None) -> GuardedPatch:
    return GuardedPatch(
        shape=None if current is None else Sides(messages={side: list(current)}).shape(side),
        patch=patch,
    )


def _chain_reasoning_items(items: list[RequestInputItem]) -> list[RequestInputItem]:
    chained: list[RequestInputItem] = []
    last_reasoning_id: str | None = None
    for item in items:
        if not isinstance(item, RequestReasoningItem) or item.encrypted_content is None:
            chained.append(item)
            continue
        payload = open_reasoning_payload(item.encrypted_content, keyring=_keyring())
        chained_payload = _reasoning_payload(
            payload_id=payload.id,
            previous_reasoning_id=last_reasoning_id,
            previous_compaction_id=None,
            machine=payload.machine,
            sides=payload.sides,
        )
        last_reasoning_id = chained_payload.id
        chained.append(
            RequestReasoningItem(
                content=item.content,
                encrypted_content=seal_reasoning_payload(chained_payload, keyring=_keyring()),
                id=chained_payload.id,
                status=item.status,
                summary=item.summary,
                type="reasoning",
            )
        )
    return chained


def _request(items: list[RequestInputItem]) -> ResponseCreateRequest:
    return ResponseCreateRequest(model="plap/test", input=_chain_reasoning_items(items))


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
    return Message.from_primitive(value).content_hash()


def _sealed_reasoning(payload: ReasoningPayload) -> RequestReasoningItem:
    return RequestReasoningItem(
        encrypted_content=seal_reasoning_payload(payload, keyring=_keyring()),
        id=payload.id,
        summary=[SummaryTextContent(text="sealed", type="summary_text")],
        type="reasoning",
    )


def _reasoning_closed_main(label: str) -> RequestReasoningItem:
    return _sealed_reasoning(
        _reasoning_payload(
            machine=[{"op": "add", "path": f"/{label}", "value": True}],
            sides=_sides_update(),
        )
    )


def _reasoning_hidden_main(*messages: dict[str, object]) -> RequestReasoningItem:
    return _sealed_reasoning(
        _reasoning_payload(
            machine=[],
            sides=_sides_update(main=[Message.from_primitive(message) for message in messages]),
        )
    )


def _reasoning_closed_non_main(side: Side, assistant: dict[str, object], tool_output: dict[str, object]) -> RequestReasoningItem:
    return _sealed_reasoning(
        _reasoning_payload(
            machine=[],
            sides=_sides_update(
                patches={
                    side: [
                        {"op": "add", "path": "/0", "value": assistant},
                        {"op": "add", "path": "/1", "value": tool_output},
                    ]
                }
            ),
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
        _reasoning_payload(
            machine=[],
            sides=_sides_update(
                main=[
                    patch,
                    *[Message.from_primitive(output) for output in deferred_outputs],
                ]
            ),
        )
    )


def _sealed_main_call(anchor: dict[str, object], upstream_id: str, *, tool_call_index: int = 0) -> RequestFunctionCallItem:
    _ = anchor
    _ = tool_call_index
    call_id = seal_call_id(
        CallID(
            side="main",
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


def _sealed_call_id(side: str, upstream_tool_call_id: str) -> str:
    return seal_call_id(
        CallID(
            side=side,
            upstream_tool_call_id=upstream_tool_call_id,
        ),
        keyring=_keyring(),
    )


def _sealed_main_output(
    anchor: dict[str, object], upstream_id: str, output: str, *, tool_call_index: int = 0
) -> RequestFunctionCallOutputItem:
    _ = anchor
    _ = tool_call_index
    call_id = seal_call_id(
        CallID(
            side="main",
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
    return [message.to_primitive() for message in result.sides[MAIN_SIDE]]


def _side_primitives(result, side: Side) -> list[dict[str, object]]:
    return [message.to_primitive() for message in result.sides.get(side, []) or []]


def _append_patch_analogue_case(
    *,
    case_id: str,
    tool_call_ids: tuple[str, ...],
    tail_items: list[RequestInputItem],
    expected_main: list[dict[str, object]],
) -> None:
    m = _assistant_value("m")
    ACCEPT_CASES.append(
        pytest.param(
            _AcceptCase(
                items=[
                    _reasoning_patch_main(m, tool_call_ids=tool_call_ids),
                    _assistant_item("m"),
                    *tail_items,
                ],
                expected_main=expected_main,
            ),
            id=case_id,
        )
    )


ACCEPT_CASES: list[pytest.ParamSpec] = []  # type: ignore[type-arg]
DISCARD_CASES: list[pytest.ParamSpec] = []  # type: ignore[type-arg]

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

h = _assistant_value("", "up_0")
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

h = _assistant_value("", "up_0")
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

h = _assistant_value("", "up_0")
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

h = _assistant_value("", "up_0")
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

h = _assistant_value("", "up_0")
ACCEPT_CASES.append(
    pytest.param(
        _AcceptCase(
            items=[
                _reasoning_hidden_main(h),
                _assistant_item("pre"),
                _sealed_main_call(h, "up_tx"),
                _sealed_main_output(h, "up_tx", "fo_tx"),
                _sealed_main_call(h, "up_0"),
                _sealed_main_output(h, "up_0", "fo_0"),
            ],
            expected_main=[
                _assistant_value("pre", "up_tx"),
                _tool_value("up_tx", "fo_tx"),
                _assistant_value("", "up_0"),
                _tool_value("up_0", "fo_0"),
            ],
        ),
        id="pre_anchor_assistant_sealed_transplant_hidden",
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

m = _assistant_value("m")
ACCEPT_CASES.append(
    pytest.param(
        _AcceptCase(
            items=[
                _reasoning_patch_main(m, tool_call_ids=("up_0",)),
                _assistant_item("pre"),
                _sealed_main_call(m, "up_tx"),
                _sealed_main_output(m, "up_tx", "fo_tx"),
                _assistant_item("m"),
                _sealed_main_call(m, "up_0"),
                _sealed_main_output(m, "up_0", "fo_0"),
            ],
            expected_main=[
                _assistant_value("pre", "up_tx"),
                _tool_value("up_tx", "fo_tx"),
                _assistant_value("m", "up_0"),
                _tool_value("up_0", "fo_0"),
            ],
        ),
        id="pre_anchor_assistant_sealed_transplant_patch",
    )
)

h = _assistant_value("", "up_0")
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

h = _assistant_value("", "up_0")
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

h = _assistant_value("", "up_0")
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

h = _assistant_value("", "up_0")
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

h = _assistant_value("hidden", "up_0")
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

h = _assistant_value("hidden", "up_0", "up_1")
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

h = _assistant_value("", "up_0")
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

h = _assistant_value("", "up_0", "up_1")
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
                _reasoning_closed_main("rm_single"),
                _assistant_item("m"),
                _fabricated_call("fab_0"),
                _fabricated_output("fab_0", "ffo_0"),
                _sealed_main_call(m, "up_0"),
                _sealed_main_output(m, "up_0", "fo_0"),
            ],
            expected_main=[
                _assistant_value("m", "fab_0", "up_0"),
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
                _reasoning_closed_main("rm_multi"),
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
                _assistant_value("m", "fab_0", "up_0", "fab_1", "up_1"),
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
                _reasoning_closed_main("post_plain_single"),
                _assistant_item("m"),
                _plain_item("user", "u1"),
                _sealed_main_call(m, "up_0"),
                _sealed_main_output(m, "up_0", "fo_0"),
            ],
            expected_main=[_assistant_value("m", "up_0"), _tool_value("up_0", "fo_0"), _plain_value("user", "u1")],
        ),
        id="free_anchor_plain_nonassistant_single",
    )
)

m = _assistant_value("m")
ACCEPT_CASES.append(
    pytest.param(
        _AcceptCase(
            items=[
                _reasoning_closed_main("post_plain_multi"),
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
        id="free_anchor_plain_nonassistant_multiple",
    )
)

m = _assistant_value("m")
post = _assistant_value("post")
ACCEPT_CASES.append(
    pytest.param(
        _AcceptCase(
            items=[
                _reasoning_closed_main("post_plain_assistant"),
                _assistant_item("m"),
                _assistant_item("post"),
                _sealed_main_call(post, "up_0"),
                _sealed_main_output(post, "up_0", "fo_0"),
            ],
            expected_main=[_assistant_value("m"), _assistant_value("post", "up_0"), _tool_value("up_0", "fo_0")],
        ),
        id="free_anchor_turnover_plain_assistant_single",
    )
)

m = _assistant_value("m")
post = _assistant_value("post")
ACCEPT_CASES.append(
    pytest.param(
        _AcceptCase(
            items=[
                _reasoning_closed_main("post_mini_single"),
                _assistant_item("m"),
                _assistant_item("post"),
                _fabricated_call("fab_0"),
                _fabricated_output("fab_0", "ffo_0"),
                _sealed_main_call(post, "up_0"),
                _sealed_main_output(post, "up_0", "fo_0"),
            ],
            expected_main=[
                _assistant_value("m"),
                _assistant_value("post", "fab_0", "up_0"),
                _tool_value("fab_0", "ffo_0"),
                _tool_value("up_0", "fo_0"),
            ],
        ),
        id="free_anchor_turnover_assistant_minibundle_single",
    )
)

m = _assistant_value("m")
post = _assistant_value("post")
ACCEPT_CASES.append(
    pytest.param(
        _AcceptCase(
            items=[
                _reasoning_closed_main("post_mini_multi"),
                _assistant_item("m"),
                _assistant_item("post"),
                _fabricated_call("fab_0"),
                _fabricated_output("fab_0", "ffo_0"),
                _sealed_main_call(post, "up_0", tool_call_index=0),
                _sealed_main_call(post, "up_1", tool_call_index=1),
                _sealed_main_output(post, "up_1", "fo_1", tool_call_index=1),
                _sealed_main_output(post, "up_0", "fo_0", tool_call_index=0),
            ],
            expected_main=[
                _assistant_value("m"),
                _assistant_value("post", "fab_0", "up_0", "up_1"),
                _tool_value("fab_0", "ffo_0"),
                _tool_value("up_1", "fo_1"),
                _tool_value("up_0", "fo_0"),
            ],
        ),
        id="free_anchor_turnover_assistant_minibundle_multiple_sealed",
    )
)

m = _assistant_value("m")
post2 = _assistant_value("post2")
ACCEPT_CASES.append(
    pytest.param(
        _AcceptCase(
            items=[
                _reasoning_closed_main("post_multi_cluster_single"),
                _assistant_item("m"),
                _assistant_item("post1"),
                _fabricated_call("fab_0"),
                _fabricated_output("fab_0", "ffo_0"),
                _assistant_item("post2"),
                _fabricated_call("fab_1"),
                _fabricated_output("fab_1", "ffo_1"),
                _sealed_main_call(post2, "up_0"),
                _sealed_main_output(post2, "up_0", "fo_0"),
            ],
            expected_main=[
                _assistant_value("m"),
                _assistant_value("post1", "fab_0"),
                _tool_value("fab_0", "ffo_0"),
                _assistant_value("post2", "fab_1", "up_0"),
                _tool_value("fab_1", "ffo_1"),
                _tool_value("up_0", "fo_0"),
            ],
        ),
        id="free_anchor_turnover_multiple_assistant_anchors",
    )
)

m = _assistant_value("m")
post = _assistant_value("post")
ACCEPT_CASES.append(
    pytest.param(
        _AcceptCase(
            items=[
                _reasoning_closed_main("post_mixed_1"),
                _assistant_item("m"),
                _plain_item("user", "u1"),
                _assistant_item("post"),
                _fabricated_call("fab_0"),
                _fabricated_output("fab_0", "ffo_0"),
                _sealed_main_call(post, "up_0"),
                _sealed_main_output(post, "up_0", "fo_0"),
            ],
            expected_main=[
                _assistant_value("m"),
                _plain_value("user", "u1"),
                _assistant_value("post", "fab_0", "up_0"),
                _tool_value("fab_0", "ffo_0"),
                _tool_value("up_0", "fo_0"),
            ],
        ),
        id="free_anchor_user_then_new_assistant_anchor",
    )
)

m = _assistant_value("m")
post = _assistant_value("post")
ACCEPT_CASES.append(
    pytest.param(
        _AcceptCase(
            items=[
                _reasoning_closed_main("post_mixed_2"),
                _assistant_item("m"),
                _assistant_item("post"),
                _plain_item("user", "u1"),
                _fabricated_call("fab_0"),
                _fabricated_output("fab_0", "ffo_0"),
                _sealed_main_call(post, "up_0"),
                _sealed_main_output(post, "up_0", "fo_0"),
            ],
            expected_main=[
                _assistant_value("m"),
                _assistant_value("post", "fab_0", "up_0"),
                _tool_value("fab_0", "ffo_0"),
                _tool_value("up_0", "fo_0"),
                _plain_value("user", "u1"),
            ],
        ),
        id="free_anchor_new_assistant_then_user",
    )
)

m = _assistant_value("m")
ACCEPT_CASES.append(
    pytest.param(
        _AcceptCase(
            items=[
                _reasoning_patch_main(m, tool_call_ids=("up_0",)),
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
        id="strict_anchor_mixed_cluster_assistant_then_user_single",
    )
)

m = _assistant_value("m")
post = _assistant_value("post")
ACCEPT_CASES.append(
    pytest.param(
        _AcceptCase(
            items=[
                _reasoning_closed_main("post_mixed_3"),
                _assistant_item("m"),
                _assistant_item("post"),
                _plain_item("user", "u1"),
                _fabricated_call("fab_0"),
                _fabricated_output("fab_0", "ffo_0"),
                _sealed_main_call(post, "up_0", tool_call_index=0),
                _sealed_main_call(post, "up_1", tool_call_index=1),
                _sealed_main_output(post, "up_1", "fo_1", tool_call_index=1),
                _sealed_main_output(post, "up_0", "fo_0", tool_call_index=0),
            ],
            expected_main=[
                _assistant_value("m"),
                _assistant_value("post", "fab_0", "up_0", "up_1"),
                _tool_value("fab_0", "ffo_0"),
                _tool_value("up_1", "fo_1"),
                _tool_value("up_0", "fo_0"),
                _plain_value("user", "u1"),
            ],
        ),
        id="free_anchor_new_assistant_then_user_parallel_sealed",
    )
)

m = _assistant_value("m")
post2 = _assistant_value("post2")
ACCEPT_CASES.append(
    pytest.param(
        _AcceptCase(
            items=[
                _reasoning_closed_main("post_multi_cluster_interleaved"),
                _assistant_item("m"),
                _assistant_item("post1"),
                _plain_item("user", "u1"),
                _fabricated_call("fab_0"),
                _fabricated_output("fab_0", "ffo_0"),
                _assistant_item("post2"),
                _plain_item("user", "u2"),
                _fabricated_call("fab_1"),
                _fabricated_output("fab_1", "ffo_1"),
                _sealed_main_call(post2, "up_0"),
                _sealed_main_output(post2, "up_0", "fo_0"),
            ],
            expected_main=[
                _assistant_value("m"),
                _assistant_value("post1", "fab_0"),
                _tool_value("fab_0", "ffo_0"),
                _plain_value("user", "u1"),
                _assistant_value("post2", "fab_1", "up_0"),
                _tool_value("fab_1", "ffo_1"),
                _tool_value("up_0", "fo_0"),
                _plain_value("user", "u2"),
            ],
        ),
        id="free_anchor_multiple_assistant_anchors_with_interleaved_plain",
    )
)

m = _assistant_value("m")
ACCEPT_CASES.append(
    pytest.param(
        _AcceptCase(
            items=[
                _reasoning_closed_main("post_fallback_1"),
                _assistant_item("m"),
                _plain_item("user", "u1"),
                _fabricated_call("fab_0"),
                _fabricated_output("fab_0", "ffo_0"),
                _sealed_main_call(m, "up_0"),
                _sealed_main_output(m, "up_0", "fo_0"),
            ],
            expected_main=[
                _assistant_value("m", "fab_0", "up_0"),
                _tool_value("fab_0", "ffo_0"),
                _tool_value("up_0", "fo_0"),
                _plain_value("user", "u1"),
            ],
        ),
        id="free_anchor_nonassistant_fallback_single",
    )
)

m = _assistant_value("m")
ACCEPT_CASES.append(
    pytest.param(
        _AcceptCase(
            items=[
                _reasoning_closed_main("post_fallback_2"),
                _assistant_item("m"),
                _plain_item("user", "u1"),
                _plain_item("user", "u2"),
                _fabricated_call("fab_0"),
                _fabricated_output("fab_0", "ffo_0"),
                _sealed_main_call(m, "up_0"),
                _sealed_main_output(m, "up_0", "fo_0"),
            ],
            expected_main=[
                _assistant_value("m", "fab_0", "up_0"),
                _tool_value("fab_0", "ffo_0"),
                _tool_value("up_0", "fo_0"),
                _plain_value("user", "u1"),
                _plain_value("user", "u2"),
            ],
        ),
        id="free_anchor_nonassistant_fallback_multiple",
    )
)

m = _assistant_value("m")
ACCEPT_CASES.append(
    pytest.param(
        _AcceptCase(
            items=[
                _reasoning_closed_main("post_fallback_3"),
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
                _assistant_value("m", "fab_0", "up_0", "up_1"),
                _tool_value("fab_0", "ffo_0"),
                _tool_value("up_1", "fo_1"),
                _tool_value("up_0", "fo_0"),
                _plain_value("user", "u1"),
            ],
        ),
        id="free_anchor_nonassistant_fallback_parallel_sealed",
    )
)

m = _assistant_value("m")
ACCEPT_CASES.append(
    pytest.param(
        _AcceptCase(
            items=[
                _reasoning_closed_main("post_fallback_4"),
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
                _assistant_value("m", "fab_0", "up_0", "fab_1", "up_1"),
                _tool_value("fab_0", "ffo_0"),
                _tool_value("up_0", "fo_0"),
                _tool_value("fab_1", "ffo_1"),
                _tool_value("up_1", "fo_1"),
                _plain_value("user", "u1"),
            ],
        ),
        id="free_anchor_nonassistant_fallback_mixed_multiple",
    )
)

_append_patch_analogue_case(
    case_id="strict_anchor_mixed_same_anchor_single",
    tool_call_ids=("up_0",),
    tail_items=[
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
)

_append_patch_analogue_case(
    case_id="strict_anchor_mixed_same_anchor_multiple",
    tool_call_ids=("up_0", "up_1"),
    tail_items=[
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
)

_append_patch_analogue_case(
    case_id="strict_anchor_plain_cluster_nonassistant_single",
    tool_call_ids=("up_0",),
    tail_items=[
        _plain_item("user", "u1"),
        _sealed_main_call(m, "up_0"),
        _sealed_main_output(m, "up_0", "fo_0"),
    ],
    expected_main=[
        _assistant_value("m", "up_0"),
        _tool_value("up_0", "fo_0"),
        _plain_value("user", "u1"),
    ],
)

_append_patch_analogue_case(
    case_id="strict_anchor_plain_cluster_nonassistant_multiple",
    tool_call_ids=("up_0",),
    tail_items=[
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
)

_append_patch_analogue_case(
    case_id="strict_anchor_plain_cluster_assistant_single",
    tool_call_ids=("up_0",),
    tail_items=[
        _assistant_item("post"),
        _sealed_main_call(m, "up_0"),
        _sealed_main_output(m, "up_0", "fo_0"),
    ],
    expected_main=[
        _assistant_value("m", "up_0"),
        _tool_value("up_0", "fo_0"),
        _assistant_value("post"),
    ],
)

_append_patch_analogue_case(
    case_id="strict_anchor_assistant_cluster_miniturn_single",
    tool_call_ids=("up_0",),
    tail_items=[
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
)

_append_patch_analogue_case(
    case_id="strict_anchor_assistant_cluster_miniturn_parallel_sealed",
    tool_call_ids=("up_0", "up_1"),
    tail_items=[
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
)

_append_patch_analogue_case(
    case_id="strict_anchor_assistant_cluster_multiple_assistants",
    tool_call_ids=("up_0",),
    tail_items=[
        _assistant_item("post1"),
        _fabricated_call("fab_0"),
        _fabricated_output("fab_0", "ffo_0"),
        _assistant_item("post2"),
        _fabricated_call("fab_1"),
        _fabricated_output("fab_1", "ffo_1"),
        _sealed_main_call(m, "up_0"),
        _sealed_main_output(m, "up_0", "fo_0"),
    ],
    expected_main=[
        _assistant_value("m", "up_0"),
        _tool_value("up_0", "fo_0"),
        _assistant_value("post1", "fab_0"),
        _tool_value("fab_0", "ffo_0"),
        _assistant_value("post2", "fab_1"),
        _tool_value("fab_1", "ffo_1"),
    ],
)

_append_patch_analogue_case(
    case_id="strict_anchor_mixed_cluster_user_then_assistant",
    tool_call_ids=("up_0",),
    tail_items=[
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
)

_append_patch_analogue_case(
    case_id="strict_anchor_mixed_cluster_assistant_then_user_parallel_sealed",
    tool_call_ids=("up_0", "up_1"),
    tail_items=[
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
)

_append_patch_analogue_case(
    case_id="strict_anchor_mixed_cluster_multiple_assistants_interleaved_plain",
    tool_call_ids=("up_0",),
    tail_items=[
        _assistant_item("post1"),
        _plain_item("user", "u1"),
        _fabricated_call("fab_0"),
        _fabricated_output("fab_0", "ffo_0"),
        _assistant_item("post2"),
        _plain_item("user", "u2"),
        _fabricated_call("fab_1"),
        _fabricated_output("fab_1", "ffo_1"),
        _sealed_main_call(m, "up_0"),
        _sealed_main_output(m, "up_0", "fo_0"),
    ],
    expected_main=[
        _assistant_value("m", "up_0"),
        _tool_value("up_0", "fo_0"),
        _assistant_value("post1", "fab_0"),
        _tool_value("fab_0", "ffo_0"),
        _plain_value("user", "u1"),
        _assistant_value("post2", "fab_1"),
        _tool_value("fab_1", "ffo_1"),
        _plain_value("user", "u2"),
    ],
)

_append_patch_analogue_case(
    case_id="strict_anchor_nonassistant_fallback_single",
    tool_call_ids=("up_0",),
    tail_items=[
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
)

_append_patch_analogue_case(
    case_id="strict_anchor_nonassistant_fallback_multiple",
    tool_call_ids=("up_0",),
    tail_items=[
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
)

_append_patch_analogue_case(
    case_id="strict_anchor_nonassistant_fallback_parallel_sealed",
    tool_call_ids=("up_0", "up_1"),
    tail_items=[
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
)

_append_patch_analogue_case(
    case_id="strict_anchor_nonassistant_fallback_mixed_multiple",
    tool_call_ids=("up_0", "up_1"),
    tail_items=[
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
)

_append_patch_analogue_case(
    case_id="strict_anchor_release_same_anchor_fabricated_continues",
    tool_call_ids=("up_0",),
    tail_items=[
        _sealed_main_call(m, "up_0"),
        _sealed_main_output(m, "up_0", "fo_0"),
        _fabricated_call("fab_0"),
        _fabricated_output("fab_0", "ffo_0"),
    ],
    expected_main=[
        _assistant_value("m", "up_0", "fab_0"),
        _tool_value("up_0", "fo_0"),
        _tool_value("fab_0", "ffo_0"),
    ],
)

m = _assistant_value("m")
post = _assistant_value("post")
ACCEPT_CASES.append(
    pytest.param(
        _AcceptCase(
            items=[
                _reasoning_patch_main(m, tool_call_ids=("up_0",)),
                _assistant_item("m"),
                _sealed_main_call(m, "up_0"),
                _sealed_main_output(m, "up_0", "fo_0"),
                _assistant_item("post"),
                _fabricated_call("fab_0"),
                _fabricated_output("fab_0", "ffo_0"),
            ],
            expected_main=[
                _assistant_value("m", "up_0"),
                _tool_value("up_0", "fo_0"),
                _assistant_value("post", "fab_0"),
                _tool_value("fab_0", "ffo_0"),
            ],
        ),
        id="strict_anchor_turnover_after_release_fabricated",
    )
)

m = _assistant_value("m")
post = _assistant_value("post")
ACCEPT_CASES.append(
    pytest.param(
        _AcceptCase(
            items=[
                _reasoning_patch_main(m, tool_call_ids=("up_0",)),
                _assistant_item("m"),
                _sealed_main_call(m, "up_0"),
                _sealed_main_output(m, "up_0", "fo_0"),
                _assistant_item("post"),
                _sealed_main_call(post, "up_1"),
                _sealed_main_output(post, "up_1", "fo_1"),
            ],
            expected_main=[
                _assistant_value("m", "up_0"),
                _tool_value("up_0", "fo_0"),
                _assistant_value("post", "up_1"),
                _tool_value("up_1", "fo_1"),
            ],
        ),
        id="strict_anchor_turnover_after_release_sealed",
    )
)

m = _assistant_value("m")
ACCEPT_CASES.append(
    pytest.param(
        _AcceptCase(
            items=[
                _reasoning_patch_main(m, tool_call_ids=("up_0", "up_1")),
                _assistant_item("m"),
                _sealed_main_call(m, "up_0", tool_call_index=0),
                _sealed_main_output(m, "up_0", "fo_0", tool_call_index=0),
                _assistant_item("post"),
                _fabricated_call("fab_0"),
                _fabricated_output("fab_0", "ffo_0"),
                _sealed_main_call(m, "up_1", tool_call_index=1),
                _sealed_main_output(m, "up_1", "fo_1", tool_call_index=1),
            ],
            expected_main=[
                _assistant_value("m", "up_0", "up_1"),
                _tool_value("up_0", "fo_0"),
                _tool_value("up_1", "fo_1"),
                _assistant_value("post", "fab_0"),
                _tool_value("fab_0", "ffo_0"),
            ],
        ),
        id="strict_anchor_partial_satisfaction_clusters_later_assistant",
    )
)

m = _assistant_value("m")
ACCEPT_CASES.append(
    pytest.param(
        _AcceptCase(
            items=[
                _assistant_item("m"),
                _sealed_main_call(m, "up_0"),
                _assistant_item("post"),
                _sealed_main_output(m, "up_0", "fo_0"),
                _fabricated_call("fab_0"),
                _fabricated_output("fab_0", "ffo_0"),
            ],
            expected_main=[
                _assistant_value("m", "up_0"),
                _tool_value("up_0", "fo_0"),
                _assistant_value("post", "fab_0"),
                _tool_value("fab_0", "ffo_0"),
            ],
        ),
        id="free_anchor_unsettled_later_assistant_clusters",
    )
)

m = _assistant_value("m")
ACCEPT_CASES.append(
    pytest.param(
        _AcceptCase(
            items=[
                _assistant_item("m"),
                _sealed_main_call(m, "up_0"),
                _sealed_main_output(m, "up_0", "fo_0"),
                _assistant_item("post"),
                _fabricated_call("fab_0"),
                _fabricated_output("fab_0", "ffo_0"),
            ],
            expected_main=[
                _assistant_value("m", "up_0"),
                _tool_value("up_0", "fo_0"),
                _assistant_value("post", "fab_0"),
                _tool_value("fab_0", "ffo_0"),
            ],
        ),
        id="free_anchor_settled_later_assistant_new_anchor",
    )
)

m1 = _assistant_value("m1")
m2 = _assistant_value("m2")
ACCEPT_CASES.append(
    pytest.param(
        _AcceptCase(
            items=[
                _assistant_item("m1"),
                _sealed_main_call(m1, "up_0"),
                _sealed_main_output(m1, "up_0", "fo_0"),
                _assistant_item("m2"),
                _sealed_main_call(m2, "up_1"),
                _sealed_main_output(m2, "up_1", "fo_1"),
            ],
            expected_main=[
                _assistant_value("m1", "up_0"),
                _tool_value("up_0", "fo_0"),
                _assistant_value("m2", "up_1"),
                _tool_value("up_1", "fo_1"),
            ],
        ),
        id="stripped_free_bundle_two_turns",
    )
)

m1 = _assistant_value("m1")
m2 = _assistant_value("m2")
m3 = _assistant_value("m3")
ACCEPT_CASES.append(
    pytest.param(
        _AcceptCase(
            items=[
                _assistant_item("m1"),
                _sealed_main_call(m1, "up_0"),
                _sealed_main_output(m1, "up_0", "fo_0"),
                _assistant_item("m2"),
                _sealed_main_call(m2, "up_1"),
                _sealed_main_output(m2, "up_1", "fo_1"),
                _assistant_item("m3"),
                _sealed_main_call(m3, "up_2"),
                _sealed_main_output(m3, "up_2", "fo_2"),
            ],
            expected_main=[
                _assistant_value("m1", "up_0"),
                _tool_value("up_0", "fo_0"),
                _assistant_value("m2", "up_1"),
                _tool_value("up_1", "fo_1"),
                _assistant_value("m3", "up_2"),
                _tool_value("up_2", "fo_2"),
            ],
        ),
        id="stripped_free_bundle_three_turns",
    )
)

DISCARD_CASES.extend(
    [
        pytest.param(
            _DiscardCase(
                items=[
                    RequestFunctionCallItem(
                        arguments='{"path":"README.md"}',
                        call_id=_sealed_call_id("reviewer", "up_0"),
                        name="read_file",
                        type="function_call",
                    ),
                    RequestFunctionCallOutputItem(
                        call_id=_sealed_call_id("reviewer", "up_0"),
                        output="fo_0",
                        type="function_call_output",
                    ),
                ],
                side="reviewer",
                expected_side=[],
            ),
            id="discard_non_main_naked_reviewer_pair",
        ),
        pytest.param(
            _DiscardCase(
                items=[
                    RequestFunctionCallItem(
                        arguments='{"path":"README.md"}',
                        call_id=_sealed_call_id("arbitrator", "up_0"),
                        name="read_file",
                        type="function_call",
                    ),
                    RequestFunctionCallOutputItem(
                        call_id=_sealed_call_id("arbitrator", "up_0"),
                        output="fo_0",
                        type="function_call_output",
                    ),
                ],
                side="arbitrator",
                expected_side=[],
            ),
            id="discard_non_main_naked_arbitrator_pair",
        ),
        pytest.param(
            _DiscardCase(
                items=[
                    _reasoning_closed_non_main(
                        "reviewer",
                        _assistant_value("review hidden", "up_existing"),
                        _tool_value("up_existing", "existing result"),
                    ),
                    RequestFunctionCallItem(
                        arguments='{"path":"README.md"}',
                        call_id=_sealed_call_id("reviewer", "up_missing"),
                        name="read_file",
                        type="function_call",
                    ),
                    RequestFunctionCallOutputItem(
                        call_id=_sealed_call_id("reviewer", "up_missing"),
                        output="missing result",
                        type="function_call_output",
                    ),
                ],
                side="reviewer",
                expected_side=[
                    _assistant_value("review hidden", "up_existing"),
                    _tool_value("up_existing", "existing result"),
                ],
            ),
            id="discard_non_main_unmatched_reviewer_pair_with_existing_history",
        ),
        pytest.param(
            _DiscardCase(
                items=[
                    _reasoning_closed_non_main(
                        "arbitrator",
                        _assistant_value("arb hidden", "up_existing"),
                        _tool_value("up_existing", "existing result"),
                    ),
                    RequestFunctionCallItem(
                        arguments='{"path":"README.md"}',
                        call_id=_sealed_call_id("arbitrator", "up_missing"),
                        name="read_file",
                        type="function_call",
                    ),
                    RequestFunctionCallOutputItem(
                        call_id=_sealed_call_id("arbitrator", "up_missing"),
                        output="missing result",
                        type="function_call_output",
                    ),
                ],
                side="arbitrator",
                expected_side=[
                    _assistant_value("arb hidden", "up_existing"),
                    _tool_value("up_existing", "existing result"),
                ],
            ),
            id="discard_non_main_unmatched_arbitrator_pair_with_existing_history",
        ),
    ]
)

m = _assistant_value("m")
ACCEPT_CASES.append(
    pytest.param(
        _AcceptCase(
            items=[
                _sealed_reasoning(
                    _reasoning_payload(
                        machine=[],
                        sides=_sides_update(
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
hidden_with_call = _assistant_value("hidden", "up_0")
ACCEPT_CASES.append(
    pytest.param(
        _AcceptCase(
            items=[
                _sealed_reasoning(
                    _reasoning_payload(
                        machine=[],
                        sides=_sides_update(
                            main=[
                                Message(
                                    role="assistant",
                                    content="prefix turn",
                                    tool_calls=[_tool_call_model("pref_0")],
                                ),
                                Message(role="tool", tool_call_id="pref_0", content="prefix output"),
                                Message.from_primitive(hidden_with_call),
                            ]
                        ),
                    )
                ),
                _sealed_main_call(hidden_with_call, "up_0"),
                _sealed_main_output(hidden_with_call, "up_0", "fo_0"),
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
                    _reasoning_payload(
                        machine=[],
                        sides=_sides_update(
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
                _sealed_main_call(m, "up_tx"),
                _sealed_main_output(m, "up_tx", "fo_tx"),
                _assistant_item("m"),
                _sealed_main_call(m, "up_0"),
                _sealed_main_output(m, "up_0", "fo_0"),
            ],
            expected_reason="sealed_function_call_without_attachment_owner",
        ),
        id="reject_patch_anchor_sealed_transplant_without_previous_assistant",
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
                _reasoning_closed_main("retro_1"),
                _assistant_item("m1"),
                _sealed_main_call(m1, "up_0"),
                _sealed_main_output(m1, "up_0", "fo_0"),
                _sealed_reasoning(
                    _reasoning_payload(
                        machine=[{"op": "add", "path": "/retro_2", "value": True}],
                        sides=_sides_update(
                            current=Sides(
                                messages={
                                    MAIN_SIDE: [
                                        Message.from_primitive(_assistant_value("m1", "up_0")),
                                        Message.from_primitive(_tool_value("up_0", "fo_0")),
                                    ]
                                }
                            )
                        ),
                    )
                ),
                _assistant_item("m2"),
                _sealed_main_call(m2, "up_1"),
                _sealed_main_output(m2, "up_1", "fo_1"),
                _sealed_main_call(m1, "up_0"),
            ],
            expected_reason="duplicate_tool_call_id_in_history",
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

m = _assistant_value("")
REJECT_CASES.append(
    pytest.param(
        _RejectCase(
            items=[_sealed_main_call(m, "syn_0"), _sealed_main_output(m, "syn_0", "fo_0")],
            expected_reason="sealed_function_call_without_attachment_owner",
        ),
        id="reject_naked_main_sealed_pair_without_anchor",
    )
)

m = _assistant_value("m")
REJECT_CASES.append(
    pytest.param(
        _RejectCase(
            items=[
                _reasoning_patch_main(m, tool_call_ids=("up_0",)),
                _assistant_item("m"),
                _sealed_main_call(m, "up_0"),
                _reasoning_closed_main("next_reasoning_before_declared_output"),
            ],
            expected_reason="pending_tool_outputs_block_message",
        ),
        id="reject_later_reasoning_before_declared_output",
    )
)

m = _assistant_value("m")
REJECT_CASES.append(
    pytest.param(
        _RejectCase(
            items=[
                _reasoning_patch_main(m, tool_call_ids=("up_0",)),
                _assistant_item("m"),
                _sealed_main_call(m, "up_0"),
                _sealed_main_output(m, "up_0", "fo_0"),
                _fabricated_call("fab_0"),
                _reasoning_closed_main("next_reasoning_before_fabricated_output"),
            ],
            expected_reason="pending_tool_outputs_block_message",
        ),
        id="reject_later_reasoning_before_fabricated_output",
    )
)

m = _assistant_value("m")
wrong_anchor = _assistant_value("wrong")
REJECT_CASES.append(
    pytest.param(
        _RejectCase(
            items=[_assistant_item("m"), _sealed_main_call(wrong_anchor, "up_0")],
            expected_reason="function_call_missing_function_call_output",
        ),
        id="reject_sealed_hash_mismatch",
    )
)

m = _assistant_value("m")
REJECT_CASES.append(
    pytest.param(
        _RejectCase(
            items=[_assistant_item("m"), _sealed_main_call(m, "wrong_up"), _sealed_main_output(m, "up_0", "fo_0")],
            expected_reason="function_call_output_without_pending_function_call",
        ),
        id="reject_output_for_unseen_upstream_id",
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
async def test_truth_table_accepts(case: _AcceptCase) -> None:
    result = await ingest_response_request(_request(case.items), keyring=_keyring())

    assert _main_primitives(result) == case.expected_main


@pytest.mark.parametrize("case", REJECT_CASES)
async def test_truth_table_rejects(case: _RejectCase) -> None:
    with pytest.raises(PlapError) as exc_info:
        await ingest_response_request(_request(case.items), keyring=_keyring())

    assert exc_info.value.private is not None
    assert exc_info.value.private.reason == case.expected_reason


@pytest.mark.parametrize("case", DISCARD_CASES)
async def test_truth_table_discards_unmatched_non_main_pairs(case: _DiscardCase) -> None:
    result = await ingest_response_request(_request(case.items), keyring=_keyring())

    assert _side_primitives(result, case.side) == case.expected_side
