"""
Tests for Config.validate()

Covers all supported LLM_PROVIDER values plus missing-key scenarios.
Config class attributes are patched directly so tests are fully isolated
from any real .env file present on the system.
"""

from unittest.mock import patch

import pytest

from app.config import Config

# Provider name constants (mirrors valid values accepted by Config)
OPENAI = "openai"
OLLAMA = "ollama"
GEMINI = "gemini"


# ---------------------------------------------------------------------------
# Helper: patch multiple Config attributes at once
# ---------------------------------------------------------------------------

def _patch_config(provider, api_key, zep_key):
    """Return a context-manager that temporarily overrides LLM + ZEP config."""
    return patch.multiple(
        Config,
        LLM_PROVIDER=provider,
        LLM_API_KEY=api_key,
        ZEP_API_KEY=zep_key,
    )


# ---------------------------------------------------------------------------
# Valid configurations
# ---------------------------------------------------------------------------

class TestConfigValidateValid:
    def test_openai_with_all_keys_returns_no_errors(self):
        with _patch_config(OPENAI, 'sk-test', 'zep-test'):
            assert Config.validate() == []

    def test_ollama_without_api_key_returns_no_errors(self):
        """Ollama is a local service and requires no API key."""
        with _patch_config(OLLAMA, None, 'zep-test'):
            assert Config.validate() == []

    def test_gemini_with_all_keys_returns_no_errors(self):
        with _patch_config(GEMINI, 'AIza-test', 'zep-test'):
            assert Config.validate() == []


# ---------------------------------------------------------------------------
# OpenAI provider validation
# ---------------------------------------------------------------------------

class TestConfigValidateOpenAI:
    def test_missing_api_key_produces_error(self):
        with _patch_config(OPENAI, None, 'zep-test'):
            errors = Config.validate()
        assert any('LLM_API_KEY' in e for e in errors)

    def test_missing_api_key_no_false_provider_error(self):
        with _patch_config(OPENAI, None, 'zep-test'):
            errors = Config.validate()
        assert not any('LLM_PROVIDER' in e for e in errors)


# ---------------------------------------------------------------------------
# Gemini provider validation
# ---------------------------------------------------------------------------

class TestConfigValidateGemini:
    def test_missing_api_key_produces_error(self):
        with _patch_config(GEMINI, None, 'zep-test'):
            errors = Config.validate()
        assert any('LLM_API_KEY' in e for e in errors)


# ---------------------------------------------------------------------------
# Ollama provider validation
# ---------------------------------------------------------------------------

class TestConfigValidateOllama:
    def test_api_key_not_required_for_ollama(self):
        with _patch_config(OLLAMA, None, 'zep-test'):
            errors = Config.validate()
        assert not any('LLM_API_KEY' in e for e in errors)

    def test_ollama_with_api_key_also_valid(self):
        """Providing a key for Ollama is harmless."""
        with _patch_config(OLLAMA, 'some-key', 'zep-test'):
            assert Config.validate() == []


# ---------------------------------------------------------------------------
# Invalid / unknown provider
# ---------------------------------------------------------------------------

class TestConfigValidateInvalidProvider:
    def test_unknown_provider_produces_error(self):
        with _patch_config('anthropic', 'key', 'zep-test'):
            errors = Config.validate()
        assert any('LLM_PROVIDER' in e for e in errors)

    def test_empty_string_provider_produces_error(self):
        with _patch_config('', 'key', 'zep-test'):
            errors = Config.validate()
        assert any('LLM_PROVIDER' in e for e in errors)


# ---------------------------------------------------------------------------
# Zep key validation (applies to all providers)
# ---------------------------------------------------------------------------

class TestConfigValidateZep:
    def test_missing_zep_key_produces_error(self):
        with _patch_config(OPENAI, 'sk-test', None):
            errors = Config.validate()
        assert any('ZEP_API_KEY' in e for e in errors)

    def test_missing_zep_key_for_ollama_produces_error(self):
        with _patch_config(OLLAMA, None, None):
            errors = Config.validate()
        assert any('ZEP_API_KEY' in e for e in errors)


# ---------------------------------------------------------------------------
# Multiple simultaneous errors
# ---------------------------------------------------------------------------

class TestConfigValidateMultipleErrors:
    def test_openai_missing_both_keys_returns_two_errors(self):
        with _patch_config(OPENAI, None, None):
            errors = Config.validate()
        assert len(errors) == 2

    def test_ollama_missing_only_zep_returns_one_error(self):
        with _patch_config(OLLAMA, None, None):
            errors = Config.validate()
        assert len(errors) == 1
        assert any('ZEP_API_KEY' in e for e in errors)
