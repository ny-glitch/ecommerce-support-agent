from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


OrderId = Annotated[
    str,
    Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$"),
]
ProductKeyword = Annotated[str, Field(min_length=1, max_length=100)]
FaqKeyword = Annotated[str, Field(min_length=2, max_length=32)]
IssueDescription = Annotated[str, Field(min_length=1, max_length=2000)]
TicketType = Literal[
    "refund",
    "return_refund",
    "exchange",
    "repair",
    "logistics",
    "complaint",
    "other",
]


class ToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class OrderInput(ToolInput):
    order_id: OrderId


class ProductInput(ToolInput):
    keyword: ProductKeyword


class LogisticsInput(ToolInput):
    order_id: OrderId


class FaqInput(ToolInput):
    keyword: FaqKeyword


class TicketInput(ToolInput):
    issue_description: IssueDescription
    ticket_type: TicketType

    @field_validator("ticket_type", mode="before")
    @classmethod
    def strip_ticket_type(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value
