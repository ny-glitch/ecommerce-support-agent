from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator


RequestText = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        strict=True,
        min_length=1,
        max_length=32_000,
    ),
]
CategoryText = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        strict=True,
        min_length=1,
        max_length=255,
    ),
]


class AfterSalesResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    order_id: str | None
    request_type: Literal[
        "refund", "return_refund", "exchange", "repair", "other", "unknown"
    ]
    expected_resolution: str | None

    @field_validator("order_id", "expected_resolution")
    @classmethod
    def reject_blank_optional_text(cls, value: str | None) -> str | None:
        if isinstance(value, str) and not value.strip():
            raise ValueError("value must be null or contain non-whitespace text")
        return value


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: RequestText
    session_id: UUID | None = None
    category: CategoryText | None = None


class ExtractRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: RequestText


class ActionConfirmRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class ActionConfirmResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    ticket_no: str = Field(min_length=1)
    conversation_id: str = Field(min_length=1)
    status: Literal["completed"]
    action_id: str = Field(min_length=1)
