from __future__ import annotations

import os

import pytest

from app.config import Settings


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--require-mysql",
        action="store_true",
        default=False,
        help="fail instead of skipping when the isolated MySQL test database is absent",
    )
    parser.addoption(
        "--require-local-models",
        action="store_true",
        default=False,
        help="fail instead of skipping when the fixed local knowledge models are absent",
    )
    parser.addoption(
        "--require-milvus",
        action="store_true",
        default=False,
        help="fail instead of skipping when the isolated Milvus service is absent",
    )


@pytest.fixture(autouse=True)
def isolate_settings_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    field_names = {field_name.casefold() for field_name in Settings.model_fields}
    for env_name in tuple(os.environ):
        if env_name.casefold() in field_names:
            monkeypatch.delenv(env_name)
