import json
from collections.abc import AsyncIterator
from contextlib import aclosing
from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.sse import EventSourceResponse, ServerSentEvent

from app.api.streaming import stream_events
from app.schemas import ChatRequest
from app.services.chat import PreparedTurn

router = APIRouter()


async def prepare_chat(body: ChatRequest, request: Request) -> AsyncIterator[PreparedTurn]:
    service = request.app.state.chat_service
    async with service.prepare(
        body.message, str(body.session_id) if body.session_id else None
    ) as prepared:
        yield prepared


@router.post("/api/chat", response_class=EventSourceResponse)
async def chat(
    request: Request,
    prepared: Annotated[PreparedTurn, Depends(prepare_chat, scope="request")],
) -> AsyncIterator[ServerSentEvent]:
    service = request.app.state.chat_service
    async with aclosing(stream_events(
        request, service.stream(prepared), deadline=prepared.deadline
    )) as events:
        async for item in events:
            yield ServerSentEvent(
                event=item.name, raw_data=json.dumps(item.data, ensure_ascii=False)
            )
