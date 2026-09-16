import json
from importlib import resources

from langchain_core.prompts import PromptTemplate

from app.schemas import AfterSalesResult


def _load_template(filename: str) -> str:
    return (
        resources.files("app")
        .joinpath("prompts")
        .joinpath(filename)
        .read_text(encoding="utf-8")
    )


def customer_system_prompt() -> str:
    template = PromptTemplate.from_template(_load_template("customer_service.txt"))
    return template.format()


def extraction_system_prompt() -> str:
    template = PromptTemplate.from_template(_load_template("after_sales.txt"))
    schema_json = json.dumps(
        AfterSalesResult.model_json_schema(), ensure_ascii=False
    )
    return template.format(schema_json=schema_json)
