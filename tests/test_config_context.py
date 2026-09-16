from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import ValidationError

from app.config import Settings, load_settings
from app.context import Turn, build_context, estimate_tokens
from app.errors import ServiceError


REQUIRED_ENV_NAMES = ("LLM_BASE_URL", "LLM_MODEL", "LLM_API_KEY")


@pytest.fixture
def settings() -> Settings:
    return Settings(
        llm_base_url="https://api.example.com/v1",
        llm_model="example-chat-model",
        llm_api_key="test-key",
        context_window_tokens=6_500,
        max_output_tokens=500,
        token_safety_margin=100,
    )


def clear_required_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in REQUIRED_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


def test_settings_requires_upstream_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_required_environment(monkeypatch)

    with pytest.raises(ValidationError) as exc_info:
        Settings(_env_file=None)

    missing_fields = {error["loc"] for error in exc_info.value.errors()}
    assert missing_fields == {("llm_base_url",), ("llm_model",), ("llm_api_key",)}


def test_load_settings_reads_dotenv_and_environment_wins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clear_required_environment(monkeypatch)
    (tmp_path / ".env").write_text(
        "LLM_BASE_URL=https://dotenv.example/v1\n"
        "LLM_MODEL=dotenv-model\n"
        "LLM_API_KEY=dotenv-key\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LLM_MODEL", "environment-model")

    result = load_settings()

    assert result.llm_base_url == "https://dotenv.example/v1"
    assert result.llm_model == "environment-model"
    assert result.llm_api_key.get_secret_value() == "dotenv-key"


def test_settings_defaults_include_cross_provider_token_field() -> None:
    result = Settings(
        llm_base_url="http://localhost:11434/v1",
        llm_model="local-model",
        llm_api_key="unused",
    )

    assert result.llm_token_limit_param == "max_completion_tokens"
    assert result.context_window_tokens == 8192
    assert result.max_output_tokens == 1024
    assert result.token_safety_margin == 512
    assert result.max_history_turns == 12
    assert result.session_ttl_seconds == 3600
    assert result.max_sessions == 100
    assert result.request_timeout_seconds == 60


@pytest.mark.parametrize(
    "overrides",
    [
        {"llm_base_url": "ftp://example.com/v1"},
        {"llm_base_url": "https://invalid host.example/v1"},
        {"llm_token_limit_param": "output_tokens"},
        {"max_sessions": 0},
        {
            "context_window_tokens": 1536,
            "max_output_tokens": 1024,
            "token_safety_margin": 512,
        },
    ],
)
def test_settings_rejects_invalid_urls_token_fields_and_budgets(
    overrides: dict[str, object],
) -> None:
    values: dict[str, object] = {
        "llm_base_url": "https://api.example.com/v1",
        "llm_model": "model",
        "llm_api_key": "secret-key",
    }
    values.update(overrides)

    with pytest.raises(ValidationError) as exc_info:
        Settings(**values)

    assert "input_value=" not in str(exc_info.value)


def test_environment_selects_max_tokens_parameter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clear_required_environment(monkeypatch)
    (tmp_path / ".env").write_text(
        "LLM_BASE_URL=https://api.example.com/v1\n"
        "LLM_MODEL=model\n"
        "LLM_API_KEY=key\n"
        "LLM_TOKEN_LIMIT_PARAM=max_tokens\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    assert load_settings().llm_token_limit_param == "max_tokens"


def test_validation_errors_hide_raw_input_values() -> None:
    marker = "do-not-leak-this-value"

    with pytest.raises(ValidationError) as exc_info:
        Settings(
            llm_base_url="https://api.example.com/v1",
            llm_model="model",
            llm_api_key="secret-key",
            max_sessions=marker,
        )

    assert marker not in str(exc_info.value)


def test_estimate_tokens_counts_utf8_bytes_with_message_overhead() -> None:
    messages = [SystemMessage("客服"), HumanMessage("退款")]

    assert estimate_tokens(messages) == 39


def test_drops_whole_old_turns(settings: Settings) -> None:
    turns = [Turn("旧问题" * 800, "旧回答" * 800), Turn("我叫小林", "记住了")]

    result = build_context("客服", turns, "我叫什么？", settings)

    assert result.retained_turns == [turns[-1]]
    assert [message.type for message in result.messages] == [
        "system",
        "human",
        "ai",
        "human",
    ]
    assert result.dropped_turns == 1
    assert turns[0].user.startswith("旧问题")


def test_history_limit_keeps_latest_turns_without_mutating_input(
    settings: Settings,
) -> None:
    settings = settings.model_copy(update={"max_history_turns": 2})
    turns = [Turn("one", "1"), Turn("two", "2"), Turn("three", "3")]
    original = list(turns)

    result = build_context("support", turns, "current", settings)

    assert result.retained_turns == original[-2:]
    assert result.retained_turns is not turns
    assert turns == original
    assert result.dropped_turns == 1


def test_context_never_drops_system_or_current_input(settings: Settings) -> None:
    result = build_context(
        "system instructions", [Turn("old", "answer")], "current question", settings
    )

    assert result.messages[0] == SystemMessage("system instructions")
    assert result.messages[-1] == HumanMessage("current question")
    assert result.estimated_input_tokens == estimate_tokens(result.messages)


def test_context_rejects_oversized_required_messages(settings: Settings) -> None:
    settings = settings.model_copy(
        update={
            "context_window_tokens": 650,
            "max_output_tokens": 500,
            "token_safety_margin": 100,
        }
    )

    with pytest.raises(ServiceError) as exc_info:
        build_context("system", [], "太长" * 20, settings)

    assert exc_info.value.code == "INPUT_TOO_LONG"
    assert exc_info.value.status_code == 413
    assert "太长" not in exc_info.value.message


def test_service_error_exposes_only_client_safe_fields() -> None:
    error = ServiceError("UPSTREAM_ERROR", "服务暂时不可用", 502)

    assert error.code == "UPSTREAM_ERROR"
    assert error.message == "服务暂时不可用"
    assert error.status_code == 502
    assert str(error) == "服务暂时不可用"


def test_context_emits_langchain_message_types(settings: Settings) -> None:
    result = build_context("support", [Turn("question", "answer")], "current", settings)

    assert isinstance(result.messages[0], SystemMessage)
    assert isinstance(result.messages[1], HumanMessage)
    assert isinstance(result.messages[2], AIMessage)
    assert isinstance(result.messages[3], HumanMessage)
