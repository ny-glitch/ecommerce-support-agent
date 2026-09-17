from __future__ import annotations

import os

import pytest

from app.config import Settings


@pytest.fixture(autouse=True)
def isolate_settings_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    field_names = {field_name.casefold() for field_name in Settings.model_fields}
    for env_name in tuple(os.environ):
        if env_name.casefold() in field_names:
            monkeypatch.delenv(env_name)
