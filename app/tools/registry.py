from __future__ import annotations

from collections.abc import Iterable

from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool


class ToolRegistry:
    def __init__(self, tools: Iterable[BaseTool]) -> None:
        self.tools = list(tools)
        self._by_name = {business_tool.name: business_tool for business_tool in self.tools}
        if len(self._by_name) != len(self.tools):
            raise ValueError("tool names must be unique")

    def get(self, name: str) -> BaseTool | None:
        return self._by_name.get(name)

    def schemas(self) -> list[dict]:
        return [convert_to_openai_tool(business_tool) for business_tool in self.tools]
