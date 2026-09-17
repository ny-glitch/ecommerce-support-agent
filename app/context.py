import json
from dataclasses import dataclass
from typing import Sequence

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from app.config import Settings
from app.db.contracts import StoredTurn
from app.errors import ServiceError


@dataclass(frozen=True)
class Turn:
    user: str
    assistant: str


@dataclass(frozen=True)
class ContextWindow:
    messages: list[BaseMessage]
    retained_turns: list[Turn]
    estimated_input_tokens: int
    dropped_turns: int


@dataclass(frozen=True)
class ToolContextWindow:
    messages: list[BaseMessage]
    retained_turns: list[StoredTurn]
    estimated_input_tokens: int
    dropped_turns: int


def _json_bytes(value: object) -> int:
    return len(
        json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    )


def _tool_metadata_bytes(message: BaseMessage) -> int:
    if isinstance(message, AIMessage):
        tool_calls: object = message.tool_calls
        if not tool_calls:
            tool_calls = message.additional_kwargs.get("tool_calls", [])
        invalid_tool_calls = message.invalid_tool_calls
        metadata: dict[str, object] = {}
        if tool_calls:
            metadata["tool_calls"] = tool_calls
        if invalid_tool_calls:
            metadata["invalid_tool_calls"] = invalid_tool_calls
        return _json_bytes(metadata) if metadata else 0
    if isinstance(message, ToolMessage):
        metadata = {"tool_call_id": message.tool_call_id}
        if message.name is not None:
            metadata["name"] = message.name
        return _json_bytes(metadata)
    return 0


def estimate_tokens(messages: Sequence[BaseMessage]) -> int:
    return 3 + sum(
        len(str(message.content).encode("utf-8"))
        + _tool_metadata_bytes(message)
        + 12
        for message in messages
    )


def _messages_for(
    system_prompt: str, turns: Sequence[Turn], message: str
) -> list[BaseMessage]:
    messages: list[BaseMessage] = [SystemMessage(system_prompt)]
    for turn in turns:
        messages.extend([HumanMessage(turn.user), AIMessage(turn.assistant)])
    messages.append(HumanMessage(message))
    return messages


def build_context(
    system_prompt: str,
    turns: Sequence[Turn],
    message: str,
    settings: Settings,
) -> ContextWindow:
    required_messages = _messages_for(system_prompt, [], message)
    required_tokens = estimate_tokens(required_messages)
    reserved_tokens = settings.max_output_tokens + settings.token_safety_margin
    if required_tokens + reserved_tokens > settings.context_window_tokens:
        raise ServiceError(
            code="INPUT_TOO_LONG",
            message="输入内容超过可处理的上下文长度",
            status_code=413,
        )

    retained_turns = list(turns[-settings.max_history_turns :])
    messages = _messages_for(system_prompt, retained_turns, message)
    estimated_input_tokens = estimate_tokens(messages)

    while (
        retained_turns
        and estimated_input_tokens + reserved_tokens
        > settings.context_window_tokens
    ):
        retained_turns.pop(0)
        messages = _messages_for(system_prompt, retained_turns, message)
        estimated_input_tokens = estimate_tokens(messages)

    return ContextWindow(
        messages=messages,
        retained_turns=retained_turns,
        estimated_input_tokens=estimated_input_tokens,
        dropped_turns=len(turns) - len(retained_turns),
    )


def _tool_messages_for(
    system_prompt: str,
    turns: Sequence[StoredTurn],
    message: str,
    current_tool_messages: Sequence[BaseMessage],
) -> list[BaseMessage]:
    messages: list[BaseMessage] = [SystemMessage(system_prompt)]
    for turn in turns:
        messages.extend(turn.messages)
    messages.append(HumanMessage(message))
    messages.extend(current_tool_messages)
    return messages


def build_tool_context(
    system_prompt: str,
    turns: Sequence[StoredTurn],
    message: str,
    settings: Settings,
    *,
    tool_schemas: list[dict],
    current_tool_messages: Sequence[BaseMessage] = (),
) -> ToolContextWindow:
    schema_bytes = _json_bytes(tool_schemas)
    required_messages = _tool_messages_for(
        system_prompt, [], message, current_tool_messages
    )
    required_tokens = estimate_tokens(required_messages) + schema_bytes
    reserved_tokens = settings.max_output_tokens + settings.token_safety_margin
    if required_tokens + reserved_tokens > settings.context_window_tokens:
        raise ServiceError(
            code="INPUT_TOO_LONG",
            message="输入内容超过可处理的上下文长度",
            status_code=413,
        )

    retained_turns = list(turns[-settings.max_history_turns :])
    messages = _tool_messages_for(
        system_prompt, retained_turns, message, current_tool_messages
    )
    estimated_input_tokens = estimate_tokens(messages) + schema_bytes

    while (
        retained_turns
        and estimated_input_tokens + reserved_tokens
        > settings.context_window_tokens
    ):
        retained_turns.pop(0)
        messages = _tool_messages_for(
            system_prompt, retained_turns, message, current_tool_messages
        )
        estimated_input_tokens = estimate_tokens(messages) + schema_bytes

    return ToolContextWindow(
        messages=messages,
        retained_turns=retained_turns,
        estimated_input_tokens=estimated_input_tokens,
        dropped_turns=len(turns) - len(retained_turns),
    )
