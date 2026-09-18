from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import (
    Field,
    HttpUrl,
    SecretStr,
    TypeAdapter,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict


PositiveInt = Annotated[int, Field(gt=0)]
ToolAttempts = Literal[1, 2]
EmbeddingModel = Literal["BAAI/bge-m3"]
EmbeddingRevision = Literal["5617a9f61b028005a4858fdac845db406aefb181"]
RerankerModel = Literal["BAAI/bge-reranker-v2-m3"]
RerankerRevision = Literal["953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"]
_HTTP_URL_ADAPTER = TypeAdapter(HttpUrl)
_PROTECTED_CHAT_BODY_KEYS = frozenset(
    {
        "messages",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "stream",
        "stream_options",
        "model",
        "max_tokens",
        "max_completion_tokens",
        "response_format",
    }
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        hide_input_in_errors=True,
    )

    llm_base_url: str
    llm_model: str
    llm_api_key: SecretStr
    database_url: SecretStr | None = None
    llm_token_limit_param: Literal["max_tokens", "max_completion_tokens"] = (
        "max_completion_tokens"
    )
    llm_chat_extra_body: dict[str, Any] = Field(default_factory=dict)
    context_window_tokens: PositiveInt = 8192
    max_output_tokens: PositiveInt = 1024
    token_safety_margin: PositiveInt = 512
    max_history_turns: PositiveInt = 12
    session_ttl_seconds: PositiveInt = 3600
    max_sessions: PositiveInt = 100
    request_timeout_seconds: PositiveInt = 60
    tool_timeout_seconds: PositiveInt = 5
    tool_max_attempts: ToolAttempts = 2
    milvus_uri: str = "http://127.0.0.1:19530"
    milvus_collection: str = "knowledge"
    milvus_token: SecretStr | None = None
    knowledge_models_dir: Path = Path(".cache/ch04/models")
    knowledge_calibration_path: Path = Path(".cache/ch04/calibration.json")
    knowledge_request_timeout_seconds: PositiveInt = 240
    knowledge_batch_size: PositiveInt = 4
    knowledge_worker_queue_size: PositiveInt = 4
    knowledge_embedding_model: EmbeddingModel = "BAAI/bge-m3"
    knowledge_embedding_revision: EmbeddingRevision = (
        "5617a9f61b028005a4858fdac845db406aefb181"
    )
    knowledge_reranker_model: RerankerModel = "BAAI/bge-reranker-v2-m3"
    knowledge_reranker_revision: RerankerRevision = (
        "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
    )

    @field_validator("llm_base_url")
    @classmethod
    def validate_llm_base_url(cls, value: str) -> str:
        _HTTP_URL_ADAPTER.validate_python(value)
        return value

    @field_validator("milvus_uri")
    @classmethod
    def validate_milvus_uri(cls, value: str) -> str:
        _HTTP_URL_ADAPTER.validate_python(value)
        return value

    @field_validator("llm_chat_extra_body")
    @classmethod
    def validate_llm_chat_extra_body(cls, value: dict[str, Any]) -> dict[str, Any]:
        protected = _PROTECTED_CHAT_BODY_KEYS.intersection(value)
        if protected:
            names = ", ".join(sorted(protected))
            raise ValueError(f"LLM chat extra body cannot override: {names}")
        return value

    @model_validator(mode="after")
    def validate_token_budget(self) -> "Settings":
        reserved_tokens = self.max_output_tokens + self.token_safety_margin
        if self.context_window_tokens <= reserved_tokens:
            raise ValueError(
                "Context window must leave room beyond output and safety budgets"
            )
        return self


def load_settings() -> Settings:
    return Settings()
