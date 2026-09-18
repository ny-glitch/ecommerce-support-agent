from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool


@dataclass(frozen=True)
class ToolPolicy:
    max_bytes: int = 4096
    max_attempts: int | None = None
    shared_deadline: bool = False


class ToolRegistry:
    def __init__(
        self,
        tools: Iterable[BaseTool],
        *,
        policies: dict[str, ToolPolicy] | None = None,
    ) -> None:
        self.tools = list(tools)
        self._by_name = {business_tool.name: business_tool for business_tool in self.tools}
        if len(self._by_name) != len(self.tools):
            raise ValueError("tool names must be unique")
        self._policies = dict(policies or {})
        unknown = self._policies.keys() - self._by_name.keys()
        if unknown:
            raise ValueError(f"policies reference unknown tools: {sorted(unknown)}")

    def get(self, name: str) -> BaseTool | None:
        return self._by_name.get(name)

    def schemas(self) -> list[dict]:
        schemas = [convert_to_openai_tool(business_tool) for business_tool in self.tools]
        for schema in schemas:
            if schema["function"]["name"] == "query_faq":
                schema["function"]["parameters"]["additionalProperties"] = False
        return schemas

    def policy(self, name: str) -> ToolPolicy:
        return self._policies.get(name, ToolPolicy())
