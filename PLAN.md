The cleaner proposal is:
Corrections
1. Use positional convention, not hidden keys.
2. messages, not turn_messages.
3. Rename the decision verbs so they describe decisions, not implementation.
Better names
Reviewer actions:
- accept
- challenge
Arbitrator actions:
- accept
- answer
- revise
- reopen
These map cleanly:
- old publish_candidate -> accept
- old answer_directly -> answer
- old continue_with_guidance -> revise
- old recheck -> reopen
Why this is better:
- accept means “the original held candidate is good enough to publish”
- answer means “I am writing the public answer now”
- revise means “go back to normal A with hidden guidance”
- reopen means “send it back through B -> A -> C again”
So the model outputs decision semantics, not runtime mechanics.
---
The actual design
1. Debate is three ongoing private conversations
- A debate thread: main_context_temp
- B thread: reviewer
- C thread: arbitrator
They continue. They do not restart.
There are no standalone seed payloads.
Instead, when an actor takes a turn, that turn’s persisted reasoning payload contains:
- the new hidden user turn for that actor
- any hidden safe server tool outputs from that turn
- the actor’s assistant result or assistant tool-call message
So the new user turn is only persisted as part of the actor turn that actually consumed it.
That is proper continuation.
---
2. Positional convention for main_context_temp
Keep it simple and positional:
[
    held_candidate_assistant,
    *held_hidden_tool_messages,
    *main_debate_messages,
]
Meaning:
1. first temp assistant message = held candidate root
2. immediately following temp role="tool" messages = hidden held outputs/placeholders for that candidate
3. everything after that = ongoing main_debate conversation
No metadata keys.
No phase fields.
That gives you one parser:
def split_main_temp(rows) -> tuple[
    ChatMessage,        # held candidate
    list[ChatMessage],  # held hidden tool messages
    list[ChatMessage],  # main_debate thread
]
---
3. One actor runner
One chunky function in debate.py:
async def continue_actor(
    *,
    side: Side,
    model: str,
    persisted_thread: Sequence[ChatMessage],
    user_turn: ChatMessage,
    safe_tools: Sequence[FunctionTool],
    safe_policies: Mapping[str, ToolPolicy],
    safe_server_executors: Mapping[str, IMCPToolProvider],
    ...
) -> ActorResult
with:
@dataclass(slots=True)
class ActorFinished:
    messages: list[ChatMessage]
    assistant: ChatMessage
@dataclass(slots=True)
class ActorAwaitingTool:
    messages: list[ChatMessage]
    tool_calls: list[ChatToolCall]
No paused.
No turn_messages.
Behavior:
- append user_turn
- loop private safe server tools internally
- if safe client tool call:
  - messages includes the assistant tool-call message
  - emit public function_call
  - later public function_call_output will come back through ingestion
  - return ActorAwaitingTool
- if final assistant message:
  - return ActorFinished
---
4. Persisted shapes
Reviewer normal turn
ReasoningPayload(
    side="reviewer",
    temp=True,
    continuation_side="main",
    messages=(
        reviewer_user_turn,
        *reviewer_hidden_server_tool_messages,
        reviewer_assistant_result,
    ),
)
Reviewer client-tool continuation
ReasoningPayload(
    side="reviewer",
    temp=True,
    continuation_side="reviewer",
    messages=(
        reviewer_user_turn,
        *reviewer_hidden_server_tool_messages,
        reviewer_assistant_tool_call_message,
    ),
)
Same pattern for main in debate and arbitrator.
Important correction: those *...hidden_server_tool_messages are only hidden server tool outputs. Client tool outputs are not hidden there.
---
5. Debate flow
A risky normal-main candidate
Normal run_main_turn(...):
- capture held candidate
- execute any safe server calls privately and store hidden tool outputs
- store placeholder hidden tool messages for non-executed client calls
- append nothing to reviewer yet as a separate payload
- switch to reviewer
- immediately continue debate loop
Reviewer turn
Build reviewer user_turn from:
- compact stable transcript
- held candidate
- optional latest main_debate response
- optional latest arbitrator guidance
Then continue_actor(side="reviewer", ...).
If ActorAwaitingTool:
- persist reviewer turn with continuation_side="reviewer"
- end response awaiting client tool output
If ActorFinished:
- persist reviewer turn with continuation_side="main"
- parse last assistant JSON:
  - accept -> publish held candidate
  - challenge -> switch to main_debate
Main debate turn
Build user_turn from:
- held candidate
- latest reviewer result
Then continue_actor(side="main", ...).
If ActorAwaitingTool:
- persist temp main turn with continuation_side="main"
If ActorFinished:
- persist temp main turn with continuation_side="arbitrator"
- switch to arbitrator
Arbitrator turn
Build user_turn from:
- compact stable transcript
- held candidate
- latest reviewer result
- latest main_debate response
Then continue_actor(side="arbitrator", ...).
If ActorFinished, parse:
- accept -> publish held candidate
- answer -> emit adjudicator public message
- revise -> append stable hidden guidance, clear debate, go back to normal main
- reopen -> switch back to reviewer
That naturally gives ABACBAC.
---
6. No restart / no seeds
This is the key point.
Reviewer recheck is not:
- “rebuild reviewer from state blob”
It is:
- existing reviewer thread
- plus one new hidden user turn
- plus next reviewer assistant reply
Same for arbitrator.
Same for main_debate.
So there is no “seed concept” anywhere in the implementation.
---
7. File split
runtime.py
- prepare_tools
- resolve_tool_calls
- run_main_turn
- run_response
- stream_response_events
debate.py
- split_main_temp
- compact_transcript
- build_reviewer_user_turn
- build_main_debate_user_turn
- build_arbitrator_user_turn
- continue_actor
- publish_accepted_candidate
- resume_main_with_guidance
- continue_debate
That is one extra file, not a forest.
---
8. Hardcoding line
The only things hardcoded should be:
- actor graph: A -> B -> A -> C
- reviewer vocabulary: accept | challenge
- arbitrator vocabulary: accept | answer | revise | reopen
- debate tool surface = safe only
Everything else is derived from persisted queues.
---
9. The one part still worth watching
The only slightly sketchy part left is split_main_temp, but positional convention is still the right tradeoff. It is much less cursed than phase fields or metadata keys, and it stays local to one parser.
If you want, the next thing I can do is turn this into a concrete patch plan by function, meaning:
1. exactly what comes out of runtime.py
2. exactly what goes into debate.py
3. the exact order to land it without breaking mainline runtime
