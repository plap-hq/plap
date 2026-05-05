from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import Field

from plap.responses.contracts.base import StrictModel


class FunctionTool(StrictModel):
    defer_loading: bool | None = Field(
        default=None,
        description="Whether this function is deferred and loaded via tool search.",
    )
    description: str | None = Field(
        default=None,
        description="Model-facing description used to decide whether to call it.",
    )
    name: str = Field(description="Function name exposed to the model.")
    parameters: dict[str, Any] | None = Field(description="JSON Schema object describing function parameters.")
    strict: bool | None = Field(
        default=None,
        description="Whether strict parameter validation should be enforced.",
    )
    type: Literal["function"] = Field(description="Tool discriminator.")


class WebSearchFilters(StrictModel):
    allowed_domains: list[str] | None = Field(
        default=None,
        description="Allowed domains; subdomains are allowed when a domain is listed.",
    )


class WebSearchUserLocation(StrictModel):
    city: str | None = Field(default=None, description="Free-text city.")
    country: str | None = Field(
        default=None,
        description="Two-letter ISO country code, for example `US`.",
    )
    region: str | None = Field(default=None, description="Free-text region/state.")
    timezone: str | None = Field(
        default=None,
        description="IANA timezone, for example `America/Los_Angeles`.",
    )
    type: Literal["approximate"] | None = Field(
        default=None,
        description="Location discriminator. OpenAI currently uses `approximate`.",
    )


class WebSearchTool(StrictModel):
    filters: WebSearchFilters | None = Field(
        default=None,
        description="Domain filters for the search.",
    )
    search_context_size: Literal["low", "medium", "high"] | None = Field(
        default=None,
        description="Guidance for how much context to spend on search results.",
    )
    # OpenAI also defines web_search_2025_08_26; plap does not expose that literal.
    type: Literal["web_search"] = Field(description="Tool discriminator.")
    user_location: WebSearchUserLocation | None = Field(
        default=None,
        description="Approximate user location to localize results.",
    )


type SupportedTool = Annotated[
    FunctionTool | WebSearchTool,
    Field(discriminator="type"),
]


class ToolChoiceFunction(StrictModel):
    name: str = Field(description="Function tool name to force.")
    type: Literal["function"] = Field(description="Tool-choice discriminator.")


type ToolChoice = Literal["none", "auto", "required"] | ToolChoiceFunction

type ResponseIncludable = Literal[
    # "web_search_call.action.sources",
    "message.output_text.logprobs",
    "reasoning.encrypted_content",
]


class TextFormatText(StrictModel):
    type: Literal["text"] = Field(description="Text format discriminator.")


class TextFormatJSONObject(StrictModel):
    type: Literal["json_object"] = Field(description="Text format discriminator.")


class TextFormatJSONSchema(StrictModel):
    description: str | None = Field(
        default=None,
        description="Model-facing description of the schema.",
    )
    name: str = Field(description="Schema name.")
    schema_: dict[str, Any] = Field(
        alias="schema",
        description="JSON Schema object the output should conform to.",
    )
    strict: bool | None = Field(
        default=None,
        description="Whether structured output should follow the schema strictly.",
    )
    type: Literal["json_schema"] = Field(description="Text format discriminator.")


type TextFormat = Annotated[
    TextFormatText | TextFormatJSONObject | TextFormatJSONSchema,
    Field(discriminator="type"),
]


class ResponseTextConfig(StrictModel):
    format: TextFormat | None = Field(
        default=None,
        description="Plain text, JSON object, or JSON Schema output format.",
    )
    # verbosity: Literal["low", "medium", "high"] | None = Field(
    #     default=None,
    #     description="Verbosity constraint for supported models.",
    # )
