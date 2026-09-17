import json
from collections.abc import AsyncIterator
from contextlib import aclosing
from typing import Any, Protocol

import openai
from httpx import AsyncClient
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from pydantic import ValidationError

from app.config import Settings
from app.context import build_context
from app.errors import ServiceError
from app.prompts import extraction_system_prompt
from app.schemas import AfterSalesResult


class ModelGateway(Protocol):
    async def select(
        self, messages: list[BaseMessage], tools: list[BaseTool]
    ) -> AIMessage: ...

    def stream(self, messages: list[BaseMessage]) -> AsyncIterator[str]: ...

    async def extract(self, description: str) -> AfterSalesResult: ...

    async def aclose(self) -> None: ...


class OpenAIModelGateway:
    def __init__(self, settings: Settings, *, model: ChatOpenAI | None = None) -> None:
        self._settings = settings
        self._chat_extra_body = {
            settings.llm_token_limit_param: settings.max_output_tokens,
            **settings.llm_chat_extra_body,
        }
        self._http_client: AsyncClient | None = None
        if model is None:
            self._http_client = AsyncClient(
                timeout=float(settings.request_timeout_seconds)
            )
            model = ChatOpenAI(
                model=settings.llm_model,
                api_key=settings.llm_api_key,
                base_url=settings.llm_base_url,
                timeout=float(settings.request_timeout_seconds),
                max_retries=0,
                stream_usage=False,
                use_responses_api=False,
                extra_body={
                    settings.llm_token_limit_param: settings.max_output_tokens
                },
                http_async_client=self._http_client,
            )
        self._model = model
        self._structured_model = model.with_structured_output(
            AfterSalesResult,
            method="json_mode",
            include_raw=True,
        )

    async def select(
        self, messages: list[BaseMessage], tools: list[BaseTool]
    ) -> AIMessage:
        selector = self._model.bind_tools(
            tools,
            tool_choice="auto",
            parallel_tool_calls=False,
            extra_body=self._chat_extra_body,
        )
        try:
            reply = await selector.ainvoke(messages)
        except openai.LengthFinishReasonError as exc:
            raise ServiceError(
                code="UPSTREAM_INCOMPLETE",
                message="模型回复未正常完成，请重试",
                status_code=502,
            ) from exc
        except Exception as exc:
            if isinstance(exc, ServiceError):
                raise
            raise ServiceError(
                code="UPSTREAM_ERROR",
                message="模型服务暂时不可用",
                status_code=502,
            ) from exc

        finish_reason = reply.response_metadata.get("finish_reason")
        incomplete_tool_call = finish_reason == "tool_calls" and not reply.tool_calls
        if (
            finish_reason not in {"stop", "tool_calls"}
            or reply.invalid_tool_calls
            or incomplete_tool_call
        ):
            raise ServiceError(
                code="UPSTREAM_INCOMPLETE",
                message="模型回复未正常完成，请重试",
                status_code=502,
            )
        return reply

    async def stream(self, messages: list[BaseMessage]) -> AsyncIterator[str]:
        emitted_text = False
        finish_reason: str | None = None
        try:
            async with aclosing(
                self._model.astream(messages, extra_body=self._chat_extra_body)
            ) as chunks:
                async for chunk in chunks:
                    current_finish = chunk.response_metadata.get("finish_reason")
                    if isinstance(current_finish, str):
                        finish_reason = current_finish
                    text = chunk.text
                    if text:
                        emitted_text = True
                        yield text
        except Exception as exc:
            if isinstance(exc, ServiceError):
                raise
            raise ServiceError(
                code="UPSTREAM_ERROR",
                message="模型服务暂时不可用",
                status_code=502,
            ) from exc

        if not emitted_text or finish_reason != "stop":
            raise ServiceError(
                code="UPSTREAM_INCOMPLETE",
                message="模型回复未正常完成，请重试",
                status_code=502,
            )

    async def extract(self, description: str) -> AfterSalesResult:
        system_prompt = extraction_system_prompt()
        messages = build_context(
            system_prompt,
            [],
            description,
            self._settings,
        ).messages
        try:
            result = await self._structured_model.ainvoke(messages)
        except (
            openai.ContentFilterFinishReasonError,
            openai.LengthFinishReasonError,
        ) as exc:
            raise ServiceError(
                code="STRUCTURED_OUTPUT_ERROR",
                message="售后信息提取失败，请重试",
                status_code=502,
            ) from exc
        except Exception as exc:
            raise ServiceError(
                code="UPSTREAM_ERROR",
                message="模型服务暂时不可用",
                status_code=502,
            ) from exc

        return self._validated_extraction(result)

    @staticmethod
    def _validated_extraction(result: Any) -> AfterSalesResult:
        try:
            raw = result["raw"]
            content = raw.content
            finish_reason = raw.response_metadata.get("finish_reason")
            if (
                finish_reason != "stop"
                or not isinstance(content, str)
                or not content.strip()
                or result["parsing_error"] is not None
                or not isinstance(result["parsed"], AfterSalesResult)
            ):
                raise ValueError("incomplete or unparseable structured response")

            decoded = json.loads(content)
            if not isinstance(decoded, dict):
                raise ValueError("structured response must be a JSON object")
            return AfterSalesResult.model_validate(decoded)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError, ValidationError) as exc:
            raise ServiceError(
                code="STRUCTURED_OUTPUT_ERROR",
                message="售后信息提取失败，请重试",
                status_code=502,
            ) from exc

    async def aclose(self) -> None:
        if self._http_client is not None:
            await self._http_client.aclose()
