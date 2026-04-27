Revised Stateless-First Ingestion Proposal
Goal
Build ingestion as a pure stateless transform:
Responses input items + declared tools + sealed artifacts -> reconstructed side queues + cursors + compaction state
No DB. No state tree. No response records. No repository.
---
1. Input Object Domains
There are two object shapes and they must not be confused.
Responses wire items:
- message
- reasoning
- function_call
- function_call_output
- compaction
Sealed payload objects:
- chat-completions messages
- message rows with optional namespace/ordinal
- compressed summarized/source context
- private reasoning/debate side queues
Responses message items do not contain tool_calls. Tool calls are separate Responses function_call items.
Chat-completions assistant messages may contain tool_calls, but only inside sealed compaction/reasoning payloads or runtime-generated hidden state.
---
2. Compaction Payload
Compaction is a root reset.
Wire item:
{
  "type": "compaction",
  "encrypted_content": "..."
}
Decrypted/decompressed payload:
{
  "version": 1,
  "type": "compaction",
  "active": [
    {
      "namespace": "m",
      "ordinal": 0,
      "message": {"role": "user", "content": "..."}
    },
    {
      "namespace": "s",
      "ordinal": 0,
      "message": {"role": "assistant", "content": "summary..."}
    }
  ],
  "source": [
    {
      "namespace": "m",
      "ordinal": 0,
      "message": {"role": "user", "content": "..."}
    }
  ],
  "cursors": {
    "m": 1,
    "s": 1
  }
}
Rules:
- Use the last compaction item in the input list.
- Discard everything before it.
- active becomes A/main summarized context.
- source is re-expansion material for reviewer/arbitrator packets.
- Only m/s need ordinals here.
- No r/f ordinals in stateless payloads.
---
3. Reasoning Payload
Wire item:
{
  "type": "reasoning",
  "encrypted_content": "..."
}
Decrypted payload:
{
  "version": 1,
  "type": "reasoning",
  "side": "main | reviewer | arbitrator",
  "temp": true,
  "messages": [
    {
      "role": "assistant",
      "content": null,
      "reasoning_content": "...",
      "tool_calls": [...]
    }
  ]
}
Message entries may also be references:
{
  "content_hash": "hash-of-chat-completions-message",
  "reasoning_content": "..."
}
Rules:
- Reasoning payloads carry private chat-completions-side state.
- Reviewer/arbitrator queues are ordered local message lists, not namespace-ordinal lanes.
- temp=true means provisional debate artifact.
- temp=false means committed replacement/final debate result.
---
4. Function Call ID
Public Responses function_call is separate from Responses message.
Sealed call_id plaintext:
{
  "side": "main | reviewer | arbitrator",
  "content_hash": "hash of chat-completions assistant message",
  "tool_call_index": 0,
  "upstream_tool_call_id": "provider-tool-call-id"
}
Rules:
- content_hash points to a chat-completions assistant message, not a Responses message item.
- tool_call_index selects the tool call inside that chat message if tool calls are present.
- If tool calls were stripped, association by content_hash + tool_call_index is allowed only if the target message/ref exists.
- No r ordinal.
- No f ordinal.
- No pending-call list.
---
5. Ingestion Flow
Input: ResponseCreateRequest.input list.
Order:
1. Classify declared tools.
2. Find last compaction item.
3. If found, discard every item before it.
4. Open compaction:
   - set base A context from active
   - retain source
   - load m/s cursors
5. Scan remaining Responses items in original order.
6. Decode minimal sealed metadata:
   - reasoning: side, temp
   - function_call/function_call_output: side, content_hash, tool_call_index, upstream_tool_call_id
7. Apply global temp pruning:
   - when a temp=false reasoning item appears, drop prior temp=true reasoning artifacts and associated tool call/output items
   - assume one active debate at a time for now
8. Route remaining items:
   - Responses message -> main/A queue
   - sealed reasoning -> queue by side
   - sealed function_call -> queue by side
   - sealed function_call_output -> queue by side
   - fabricated cohesive unsealed function_call + function_call_output -> main/A only
9. Per-side unpack/associate:
   - convert Responses message items to chat-completions messages
   - decrypt reasoning messages fully
   - build side-local index by content_hash
   - merge reasoning refs into target messages when applicable
   - associate sealed function calls/outputs via content_hash + tool_call_index
   - validate tool call index/upstream id if target message contains tool_calls
   - if tool calls were stripped, allow association only if target message/ref exists
   - fabricated cohesive A pair attaches to closest previous A message
10. Output ingestion result:
   - A/main chat queue
   - reviewer chat queue
   - arbitrator chat queue
   - active compaction context
   - source/re-expansion context
   - m/s cursors
   - tool policies
   - association diagnostics/errors
---
6. Fabricated Function Calls
Fabricated means client provides a coherent unsealed Responses function_call and function_call_output pair.
Rules:
- Allowed only for A/main.
- Associated with closest previous A message.
- Untrusted/client-authored.
- Cannot create reviewer/arbitrator private state.
- Cannot satisfy sealed pending debate calls.
---
7. Temp Pruning
Global, before side splitting.
Rule:
- A later temp=false reasoning payload prunes previous temp=true debate artifacts and their related function calls/outputs.
- This allows final A-side debate summary/result to replace the entire provisional A/B/C debate.
- One active debate at a time. If we later need concurrent debates, add debate_id.
---
8. Runtime After Ingestion
Once ingestion returns side queues:
- Normal main turn sends A queue upstream.
- Debate sends reviewer/arbitrator derived packets from A/source/debate queues.
- A/B/C may emit tool calls.
- Public output emits function_call items for client-owned tools.
- Private debate transcript stays sealed inside reasoning payloads.
- Arbitrator final result emits temp=false reasoning payload and/or visible final result.
- Future main context is built from compaction/source/current A state, not from DB assumptions.
---
9. Implementation Phases
First implementation should be no DB.
1. Define payload dataclasses:
   - ChatMessageRow
   - CompactionPayload
   - ReasoningPayload
   - SealedCallID
   - IngestedQueues
2. Implement AEAD + zstd pack/unpack for compaction/reasoning.
3. Implement call ID pack/unpack with content_hash.
4. Implement ingestion:
   - compaction slicing
   - minimal header decoding
   - global temp pruning
   - routing
   - per-side association
5. Add pure stateless tests:
   - compaction root reset
   - reasoning side routing
   - temp=false prunes entire temp debate
   - sealed function_call/output routes to B
   - sealed function_call/output routes to A
   - fabricated pair routes to A closest previous message
   - stripped tool_calls association by content_hash
   - missing content_hash target fails closed
   - no DB involved
Only after this passes should DB persistence be reintroduced as cache/storage.
