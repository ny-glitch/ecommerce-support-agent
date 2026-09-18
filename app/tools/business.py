from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import hashlib
import random

from langchain_core.tools import tool

from app.db.contracts import TurnRef
from app.db.faq import FaqRepository
from app.db.tickets import TicketRepository
from app.errors import ServiceError
from app.tools.registry import ToolPolicy, ToolRegistry
from app.tools.results import bounded_result
from app.tools.schemas import (
    KnowledgeInput,
    LogisticsInput,
    OrderInput,
    ProductInput,
    TicketInput,
)


@dataclass(frozen=True)
class ToolContext:
    ref: TurnRef
    user_id: str
    user_message: str
    ticket_no: str


def build_registry(
    context: ToolContext,
    faq: FaqRepository,
    tickets: TicketRepository,
    *,
    rng: random.Random | None = None,
    knowledge_call: Callable[[], Awaitable[str]] | None = None,
) -> ToolRegistry:
    random_source = rng if rng is not None else random.Random()

    @tool(args_schema=OrderInput)
    async def query_order(order_id: str) -> str:
        """查询指定订单的演示状态、商品摘要和金额。"""
        order_status = random_source.choice(
            ["待付款", "待发货", "运输中", "已完成", "已取消"]
        )
        return bounded_result(
            {
                "status": "ok",
                "simulated": True,
                "data": {
                    "order_id": order_id,
                    "order_status": order_status,
                    "items": [{"name": "演示商品", "quantity": 1}],
                    "amount": round(random_source.uniform(39, 799), 2),
                    "currency": "CNY",
                },
            }
        )

    @tool(args_schema=ProductInput)
    async def query_product(keyword: str) -> str:
        """根据商品名称或标识生成相关演示商品信息。"""
        product_id = hashlib.sha256(keyword.encode("utf-8")).hexdigest()[:12].upper()
        return bounded_result(
            {
                "status": "ok",
                "simulated": True,
                "data": [
                    {
                        "product_id": f"DEMO-{product_id}",
                        "keyword": keyword,
                        "name": f"{keyword}演示商品",
                        "price": round(random_source.uniform(19, 999), 2),
                        "currency": "CNY",
                        "stock": random_source.randint(0, 50),
                    }
                ],
            }
        )

    @tool(args_schema=LogisticsInput)
    async def query_logistics(order_id: str) -> str:
        """查询指定订单的演示物流状态和轨迹。"""
        states = ["已下单", "已揽收", "运输中", "派送中", "已签收"]
        last_index = random_source.randrange(len(states))
        start = datetime.now(UTC).replace(microsecond=0) - timedelta(
            hours=12 * last_index
        )
        tracking = [
            {
                "status": state,
                "time": (start + timedelta(hours=12 * index)).isoformat(),
            }
            for index, state in enumerate(states[: last_index + 1])
        ]
        return bounded_result(
            {
                "status": "ok",
                "simulated": True,
                "data": {
                    "order_id": order_id,
                    "logistics_status": states[last_index],
                    "tracking": tracking,
                },
            }
        )

    @tool(args_schema=KnowledgeInput)
    async def query_faq() -> str:
        """查询本店政策、商品型号规格和使用说明；自动使用本轮原始问题。"""
        if knowledge_call is None:
            raise ServiceError(
                "KNOWLEDGE_UNAVAILABLE",
                "知识服务暂时不可用",
                503,
            )
        return await knowledge_call()

    @tool(args_schema=TicketInput)
    async def create_ticket(issue_description: str, ticket_type: str) -> str:
        """创建关联当前会话的待处理客服工单。"""
        ticket = await tickets.create_once(
            context.ticket_no,
            context.ref.conversation_id,
            context.user_id,
            issue_description,
            ticket_type,
        )
        return bounded_result({"status": "ok", "data": ticket})

    return ToolRegistry(
        [query_order, query_product, query_logistics, query_faq, create_ticket],
        policies={
            "query_faq": ToolPolicy(
                max_bytes=48_000,
                max_attempts=1,
                shared_deadline=True,
            )
        },
    )
