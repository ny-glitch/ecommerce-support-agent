from dataclasses import dataclass
from typing import Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from app.config import Settings
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


def estimate_tokens(messages: Sequence[BaseMessage]) -> int:
    return 3 + sum(
        len(str(message.content).encode("utf-8")) + 12 for message in messages
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
