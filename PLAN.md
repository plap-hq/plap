Total Proposal
Shape
- Add exactly one new file: src/plap/responses/debate.py.
- Keep runtime.py as orchestration only.
- Do not add a state-machine library.
- Do not add phase fields or hidden metadata.
- Do not reintroduce a stale transcript snapshot.
Source Of Truth
- MutableQueues remains the only mutable state:
  - main_context
  - main_context_temp
  - reviewer
  - arbitrator
  - cursors
  - continuation_side
  - in_temp_debate
- MutableQueues stays responsible for:
  - current_actor()
  - effective_main_context()
  - main_transcript(...)
  - compact_transcript(...)
  - temp_main_parts()
  - append_main_stable(...)
  - append_main_temp(...)
  - append_side(...)
  - set_continuation(...)
  - clear_debate()
Actor Routing
- continuation_side == reviewer means run reviewer.
- continuation_side == arbitrator means run arbitrator.
- continuation_side == main and in_temp_debate means run main_debate.
- Otherwise run normal main.
That becomes the only debate dispatcher rule.
Main Temp Convention
Keep the positional convention exactly as-is:
- first temp assistant row = held candidate root
- immediately following temp tool rows = hidden held tool outputs/placeholders for that candidate
- everything after that = ongoing main_debate private conversation
state.temp_main_parts() remains the one parser for this.
Actions
Use the reduced action set:
- Reviewer: accept | reopen
- Arbitrator: accept | revise | reopen
No challenge.
No answer.
That means the only public success exits are:
- publish held candidate
- revise, then rerun normal main
This is much cleaner because it keeps final public answering authority on normal main.
Decision Shapes
Reviewer decision:
{
  "action": "accept | reopen",
  "rationale": "short private reason",
  "guidance": "required when action=reopen"
}
Arbitrator decision:
{
  "action": "accept | revise | reopen",
  "rationale": "short private reason",
  "guidance": "required when action=revise or action=reopen"
}
Use typed msgspec.Structs for these in debate.py, not loose dicts.
Context For Each Actor
- main_debate must always run on state.effective_main_context().
- Reviewer must always use state.compact_transcript(token_budget=profile.transcript_token_budget) as its stable conversation base.
- Arbitrator must also use state.compact_transcript(...) as its stable conversation base.
- Do not add effective_reviewer_context() or effective_arbitrator_context() methods. That would bake debate prompt semantics into MutableQueues.
Actor Inputs
Reviewer hidden user turn should contain:
- compact stable transcript
- held candidate
- optional latest main_debate result
- optional latest arbitrator guidance
main_debate hidden user turn should contain:
- held candidate
- latest reviewer reopen guidance
Arbitrator hidden user turn should contain:
- compact stable transcript
- held candidate
- latest reviewer decision
- latest main_debate result
These are just ordinary hidden user messages in the actor’s private conversation. No seed object, no phase object.
Actor Execution
Use two executors, not one fake-generic one:
- continue_thread_actor(...)
- Used for reviewer and arbitrator
- Runs on persisted side-thread messages plus one new hidden user turn
- continue_main_debate_actor(...)
- Used only for main_debate
- Runs on full state.effective_main_context() plus one new hidden user turn
That split is important. It prevents the old bug where main_debate lost stable main_context.
Both executors should:
- expose only safe tools
- loop safe server tools privately
- pause on safe client tool calls
- return typed outcomes:
  - ActorFinished(messages, assistant)
  - ActorAwaitingClientTool(messages, tool_calls)
Safe Tool Surface During Debate
Debate actors get only:
- safe client tools
- safe server/MCP tools
They do not get:
- visible
- mutation
- unknown
- unresolved contextual
- compact
Because the surface is already filtered, debate actor execution should not rerun the call-policy resolver.
Initial Risky Main Interception
When normal main returns any client tool call whose resolved policy is not safe or visible:
- If profile.debate_max_rounds == 0, skip debate and publish as today.
- Otherwise intercept the whole candidate batch.
- Execute any safe server calls privately right there.
- Create hidden placeholder tool outputs for every non-executed client call.
- Persist one temp main reasoning item containing:
  - held candidate assistant message
  - hidden held server outputs
  - hidden placeholders for non-executed client calls
- Set continuation_side = reviewer
- Set in_temp_debate = True
- Immediately enter continue_debate(...)
No public function calls are emitted from that held candidate yet.
Continuation On Safe Client Tools
If reviewer, main_debate, or arbitrator calls a safe client tool:
- Persist one temp reasoning item for that same side containing:
  - the new hidden user turn
  - any hidden safe server outputs from that turn
  - the assistant tool-call message
- Emit public function_call items with sealed call IDs for that side
- Complete the response
- On replay, ingestion routes function_call_output back to that same side
- When resuming, if the persisted private thread already ends in a role="tool" message, do not synthesize another hidden user turn first
This keeps continuation stateless and correct.
Debate Dispatcher
continue_debate(...) in debate.py should be one explicit loop, not recursion.
Flow:
1. Reviewer runs.
2. If reviewer accepts, publish held candidate and complete.
3. If reviewer reopens, run main_debate.
4. When main_debate finishes, run arbitrator.
5. Arbitrator:
   - accept -> publish held candidate and complete
   - revise -> emit stable hidden guidance, clear debate, continue normal main
   - reopen -> go back to reviewer
So the cycle stays:
- risky main
- reviewer
- main_debate
- arbitrator
- reviewer again if reopened
That gives A -> B -> A -> C -> B ... without inventing extra concepts.
Publishing Accepted Candidate
publish_accepted_candidate(...) must reconstruct from persisted temp-main state, not locals.
It should:
- parse state.temp_main_parts()
- rebuild the held candidate from the first temp assistant row
- map hidden held tool rows by tool_call_id
- emit fresh public assistant message
- emit fresh non-temp reasoning patch if the held candidate had provider reasoning
- emit fresh public function_call items for all original client calls
- emit public function_call_output only for the held safe server outputs that were actually executed
- append the public assistant message and public server outputs into stable main_context
- clear debate state
No re-execution.
No reuse of old sealed IDs.
No dependence on old locals.
Revise Path
resume_main_with_guidance(...) should:
- emit one stable non-temp main reasoning item
- hidden payload contains exactly one assistant guidance message
- append that guidance to stable main_context
- clear debate state
- return control to normal mainline runtime
This is the only path where debate writes stable hidden guidance.
Round Limits
Keep debate_max_rounds semantics simple:
- 0 means autoaccept risky candidate and skip debate entirely
- positive values cap reviewer rounds
- if arbitrator says reopen but the cap is already reached, convert that to revise
That avoids fail-open loops and avoids another special action.
Shared Completion Request Builder
Add one shared completion-request builder for both mainline and debate actors.
It should take:
- resolved actor config
- messages
- tools
- tool_choice
- response_format
- request
- prompt-cache base
- actor name
It should always forward the same standard knobs:
- max_output_tokens
- temperature
- top_p
- top_logprobs
- actor reasoning_effort
- actor service_tier
- actor-scoped prompt_cache_key
- user=None as today
This prevents mainline and debate from drifting again.
File Split
runtime.py should contain:
- prepare_tools
- resolve_tool_calls
- run_main_turn
- run_response
- stream_response_events
debate.py should contain:
- typed debate decision structs
- debate prompts / response formats
- actor input builders
- continue_thread_actor
- continue_main_debate_actor
- continue_debate
- publish_accepted_candidate
- resume_main_with_guidance
No more files than that.
Landing Order
1. Add typed debate structs and prompts in debate.py.
2. Add actor input builders.
3. Add continue_thread_actor(...).
4. Add continue_main_debate_actor(...).
5. Add publish_accepted_candidate(...).
6. Add resume_main_with_guidance(...).
7. Add continue_debate(...).
8. Add the shared completion-request builder.
9. Replace the two current “debate disabled during cleanup” branches in runtime.py.
10. Add replay tests for safe client tool continuation on reviewer, main_debate, and arbitrator.
11. Add reopen-cap behavior.
What I Would Not Do
- no state-machine framework
- no phase fields
- no seed objects
- no cached transcript snapshot in runtime state
- no main_debate_thread abstraction
- no extra queue state beyond MutableQueues
If you want, the next step can be the concrete landing plan by function and test order against the current files exactly as they exist now.
