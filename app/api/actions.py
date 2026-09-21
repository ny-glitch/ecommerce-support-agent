from uuid import UUID

from fastapi import APIRouter, Request

from app.schemas import ActionConfirmRequest, ActionConfirmResult


router = APIRouter()


@router.post(
    "/api/conversations/{conversation_id}/actions/{action_id}/confirm",
    response_model=ActionConfirmResult,
)
async def confirm_action(
    conversation_id: UUID,
    action_id: UUID,
    body: ActionConfirmRequest,
    request: Request,
) -> ActionConfirmResult:
    result = await request.app.state.action_service.confirm(
        str(conversation_id), str(action_id)
    )
    return ActionConfirmResult.model_validate(result)
