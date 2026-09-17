from __future__ import annotations

from app.db.database import Database
from app.db.models import Conversation, FAQ, Message, Ticket


FAQ_RETURN_ID = 900_000_001
FAQ_SHIPPING_ID = 900_000_002
DEMO_CONVERSATION_ID = "00000000-0000-0000-0000-000000000001"
DEMO_MESSAGE_ID = 9_000_000_001
DEMO_TICKET_NO = "DEMO-TICKET-001"


async def seed_database(db: Database) -> None:
    async with db.sessions.begin() as session:
        if await session.get(FAQ, FAQ_RETURN_ID) is None:
            session.add(
                FAQ(
                    id=FAQ_RETURN_ID,
                    question="退货政策",
                    answer=(
                        "本演示店铺支持签收后7天内申请退货，商品需保持完好；"
                        "特殊商品以商品说明为准。"
                    ),
                    category="售后",
                )
            )
        if await session.get(FAQ, FAQ_SHIPPING_ID) is None:
            session.add(
                FAQ(
                    id=FAQ_SHIPPING_ID,
                    question="运费收取规则",
                    answer="本演示店铺普通配送每单10元，实付满99元免运费。",
                    category="配送",
                )
            )

        if await session.get(Conversation, DEMO_CONVERSATION_ID) is None:
            session.add(
                Conversation(
                    id=DEMO_CONVERSATION_ID,
                    user_id="demo-user",
                    status="human_pending",
                )
            )
            await session.flush()

        if await session.get(Message, DEMO_MESSAGE_ID) is None:
            session.add(
                Message(
                    id=DEMO_MESSAGE_ID,
                    conversation_id=DEMO_CONVERSATION_ID,
                    turn_id="demo-turn-001",
                    role="user",
                    content="收到的商品有破损，请帮我联系人工客服。",
                    tool_calls=None,
                    tool_call_id=None,
                    turn_status="completed",
                )
            )

        if await session.get(Ticket, DEMO_TICKET_NO) is None:
            session.add(
                Ticket(
                    ticket_no=DEMO_TICKET_NO,
                    conversation_id=DEMO_CONVERSATION_ID,
                    issue_description="演示商品破损售后申请",
                    ticket_type="售后",
                    status="pending",
                )
            )
