from __future__ import annotations

from typing import Literal

from pydantic import Field

from plap.responses.contracts.base import StrictModel


class ModelObject(StrictModel):
    id: str
    object: Literal["model"] = "model"
    created: int = Field(ge=0)
    owned_by: str


class ModelListObject(StrictModel):
    object: Literal["list"] = "list"
    data: list[ModelObject]


class ModelInfoPricingObject(StrictModel):
    input_per_token: float = Field(ge=0)
    output_per_token: float = Field(ge=0)


class ModelInfoObject(StrictModel):
    id: str
    object: Literal["model_info"] = "model_info"
    display_name: str
    description: str
    mode: str
    input_modalities: list[str]
    output_modalities: list[str]
    max_input_tokens: int = Field(ge=1)
    max_output_tokens: int = Field(ge=1)
    supported_parameters: list[str]
    pricing: ModelInfoPricingObject
    provider: str
    deprecated: bool = False


class ModelInfoListObject(StrictModel):
    object: Literal["list"] = "list"
    data: list[ModelInfoObject]
