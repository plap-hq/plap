from __future__ import annotations

import secrets
import time
from collections.abc import Sequence

from plap.responses.contracts import (
    ConversationReference,
    ResponseCreateRequest,
    ResponseObject,
    ResponseOutputItem,
    ResponseStatus,
    ResponseUsage,
)


def build_response_object(
    request: ResponseCreateRequest,
    *,
    response_id: str | None = None,
    output: Sequence[ResponseOutputItem] = (),
    status: ResponseStatus = "completed",
    usage: ResponseUsage | None = None,
) -> ResponseObject:
    created_at = time.time()
    return ResponseObject(
        completed_at=created_at if status in {"completed", "cancelled"} else None,
        conversation=_response_conversation(request),
        created_at=created_at,
        id=response_id or f"resp_{secrets.token_urlsafe(18)}",
        instructions=request.instructions,
        max_output_tokens=request.max_output_tokens,
        max_tool_calls=request.max_tool_calls,
        metadata=request.metadata,
        model=request.model,
        output=list(output),
        parallel_tool_calls=request.parallel_tool_calls,
        previous_response_id=request.previous_response_id,
        prompt=request.prompt,
        prompt_cache_key=request.prompt_cache_key,
        reasoning=request.reasoning,
        safety_identifier=request.safety_identifier,
        service_tier=request.service_tier,
        status=status,
        temperature=request.temperature,
        text=request.text,
        tool_choice=request.tool_choice,
        tools=request.tools,
        top_logprobs=request.top_logprobs,
        top_p=request.top_p,
        truncation=request.truncation,
        usage=usage,
        user=request.user,
    )


def _response_conversation(
    request: ResponseCreateRequest,
) -> ConversationReference | None:
    if request.conversation is None:
        return None
    if isinstance(request.conversation, str):
        return ConversationReference(id=request.conversation)
    return request.conversation
