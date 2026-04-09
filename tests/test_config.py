"""Tests for environment-backed app configuration."""

from youtube_rag.config import get_config, get_missing_required_settings


def test_missing_required_settings_are_reported(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    get_config.cache_clear()

    assert get_missing_required_settings() == ["OPENAI_API_KEY", "DATABASE_URL"]


def test_config_loads_required_and_default_values(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/youtube_rag")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    get_config.cache_clear()

    config = get_config()

    assert config.openai_api_key == "test-key"
    assert config.database_url.startswith("postgresql://")
    assert config.log_level == "DEBUG"
    assert config.top_k_results == 5
    assert config.max_question_chars == 500
    assert config.min_question_interval_seconds == 2.0
    assert config.max_context_chars == 6000
