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
