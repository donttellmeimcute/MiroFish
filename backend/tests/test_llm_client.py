"""
Tests for LLMClient

Covers:
- Constructor validation (invalid provider, missing API keys)
- Message format conversion (_build_langchain_messages)
- chat() routing to openai / ollama / gemini backends
- <think> tag removal
- chat_json() JSON parsing and markdown code-fence stripping
"""

import json
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from app.utils.llm_client import LLMClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MESSAGES = [{"role": "user", "content": "Hello"}]


def _mock_openai_response(content: str) -> MagicMock:
    """Build a minimal mock that mimics openai.ChatCompletion response."""
    choice = MagicMock()
    choice.message.content = content
    response = MagicMock()
    response.choices = [choice]
    return response


def _mock_lc_response(content: str) -> MagicMock:
    """Build a mock LangChain AI message."""
    msg = MagicMock()
    msg.content = content
    return msg


@contextmanager
def _patched_ollama(response_content: str = "ok"):
    """Patch ChatOllama and _build_langchain_messages together.

    Yields the ChatOllama class mock so callers can inspect call arguments.
    """
    with patch("langchain_ollama.ChatOllama") as mock_cls, \
         patch.object(LLMClient, "_build_langchain_messages", return_value=[]):
        mock_cls.return_value.invoke.return_value = _mock_lc_response(response_content)
        yield mock_cls


@contextmanager
def _patched_gemini(response_content: str = "ok"):
    """Patch ChatGoogleGenerativeAI and _build_langchain_messages together.

    Yields the ChatGoogleGenerativeAI class mock so callers can inspect call arguments.
    """
    with patch("langchain_google_genai.ChatGoogleGenerativeAI") as mock_cls, \
         patch.object(LLMClient, "_build_langchain_messages", return_value=[]):
        mock_cls.return_value.invoke.return_value = _mock_lc_response(response_content)
        yield mock_cls


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------

class TestLLMClientInit:
    def test_invalid_provider_raises_value_error(self):
        with pytest.raises(ValueError, match="LLM_PROVIDER 不支持"):
            LLMClient(provider="anthropic", api_key="key")

    def test_openai_without_api_key_raises_value_error(self):
        with pytest.raises(ValueError, match="LLM_API_KEY"):
            LLMClient(provider="openai", api_key=None)

    def test_gemini_without_api_key_raises_value_error(self):
        with pytest.raises(ValueError, match="LLM_API_KEY"):
            LLMClient(provider="gemini", api_key=None)

    def test_ollama_without_api_key_succeeds(self):
        """Ollama is a local service — no API key required."""
        client = LLMClient(provider="ollama", api_key=None, model="llama3.2")
        assert client.provider == "ollama"
        assert client.model == "llama3.2"

    def test_provider_is_normalised_to_lowercase(self):
        with patch("openai.OpenAI"):
            client = LLMClient(provider="OpenAI", api_key="sk-test")
        assert client.provider == "openai"

    def test_openai_stores_model_and_base_url(self):
        with patch("openai.OpenAI"):
            client = LLMClient(
                provider="openai",
                api_key="sk-test",
                model="gpt-4o",
                base_url="https://custom.example.com/v1",
            )
        assert client.model == "gpt-4o"
        assert client.base_url == "https://custom.example.com/v1"

    def test_gemini_stores_attributes(self):
        client = LLMClient(
            provider="gemini",
            api_key="AIza-test",
            model="gemini-2.0-flash",
        )
        assert client.provider == "gemini"
        assert client.model == "gemini-2.0-flash"


# ---------------------------------------------------------------------------
# Message format conversion
# ---------------------------------------------------------------------------

class TestBuildLangchainMessages:
    def test_system_message_mapped_correctly(self):
        from langchain_core.messages import SystemMessage

        result = LLMClient._build_langchain_messages(
            [{"role": "system", "content": "You are helpful"}]
        )
        assert len(result) == 1
        assert isinstance(result[0], SystemMessage)
        assert result[0].content == "You are helpful"

    def test_user_message_mapped_correctly(self):
        from langchain_core.messages import HumanMessage

        result = LLMClient._build_langchain_messages(
            [{"role": "user", "content": "Hello"}]
        )
        assert isinstance(result[0], HumanMessage)
        assert result[0].content == "Hello"

    def test_assistant_message_mapped_correctly(self):
        from langchain_core.messages import AIMessage

        result = LLMClient._build_langchain_messages(
            [{"role": "assistant", "content": "Hi there"}]
        )
        assert isinstance(result[0], AIMessage)
        assert result[0].content == "Hi there"

    def test_unknown_role_defaults_to_human_message(self):
        from langchain_core.messages import HumanMessage

        result = LLMClient._build_langchain_messages(
            [{"role": "tool", "content": "result"}]
        )
        assert isinstance(result[0], HumanMessage)

    def test_multi_turn_conversation_order_preserved(self):
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
        ]
        result = LLMClient._build_langchain_messages(messages)
        assert len(result) == 4
        assert isinstance(result[0], SystemMessage)
        assert isinstance(result[1], HumanMessage)
        assert isinstance(result[2], AIMessage)
        assert isinstance(result[3], HumanMessage)


# ---------------------------------------------------------------------------
# chat() — OpenAI backend
# ---------------------------------------------------------------------------

class TestChatOpenAI:
    def test_returns_model_content(self):
        with patch("openai.OpenAI") as mock_cls:
            mock_instance = MagicMock()
            mock_cls.return_value = mock_instance
            mock_instance.chat.completions.create.return_value = (
                _mock_openai_response("Hello from OpenAI")
            )

            client = LLMClient(provider="openai", api_key="sk-test", model="gpt-4o")
            result = client.chat(MESSAGES)

        assert result == "Hello from OpenAI"

    def test_passes_temperature_and_max_tokens(self):
        with patch("openai.OpenAI") as mock_cls:
            mock_instance = MagicMock()
            mock_cls.return_value = mock_instance
            mock_instance.chat.completions.create.return_value = (
                _mock_openai_response("OK")
            )

            client = LLMClient(provider="openai", api_key="sk-test")
            client.chat(MESSAGES, temperature=0.2, max_tokens=512)

        kwargs = mock_instance.chat.completions.create.call_args.kwargs
        assert kwargs["temperature"] == 0.2
        assert kwargs["max_tokens"] == 512

    def test_passes_response_format_when_given(self):
        fmt = {"type": "json_object"}
        with patch("openai.OpenAI") as mock_cls:
            mock_instance = MagicMock()
            mock_cls.return_value = mock_instance
            mock_instance.chat.completions.create.return_value = (
                _mock_openai_response("{}")
            )

            client = LLMClient(provider="openai", api_key="sk-test")
            client.chat(MESSAGES, response_format=fmt)

        kwargs = mock_instance.chat.completions.create.call_args.kwargs
        assert kwargs["response_format"] == fmt

    def test_omits_response_format_when_none(self):
        with patch("openai.OpenAI") as mock_cls:
            mock_instance = MagicMock()
            mock_cls.return_value = mock_instance
            mock_instance.chat.completions.create.return_value = (
                _mock_openai_response("OK")
            )

            client = LLMClient(provider="openai", api_key="sk-test")
            client.chat(MESSAGES, response_format=None)

        kwargs = mock_instance.chat.completions.create.call_args.kwargs
        assert "response_format" not in kwargs


# ---------------------------------------------------------------------------
# chat() — Ollama backend
# ---------------------------------------------------------------------------

class TestChatOllama:
    def _make_client(self, model="llama3.2"):
        return LLMClient(provider="ollama", api_key=None, model=model)

    def test_returns_model_content(self):
        with _patched_ollama("Hi Ollama"):
            client = self._make_client()
            result = client.chat(MESSAGES)
        assert result == "Hi Ollama"

    def test_passes_correct_kwargs_to_chat_ollama(self):
        with _patched_ollama() as mock_cls:
            client = self._make_client(model="mistral")
            client.chat(MESSAGES, temperature=0.5, max_tokens=200)

        init_kwargs = mock_cls.call_args.kwargs
        assert init_kwargs["model"] == "mistral"
        assert init_kwargs["temperature"] == 0.5
        assert init_kwargs["num_predict"] == 200

    def test_sets_json_format_when_response_format_provided(self):
        with _patched_ollama() as mock_cls:
            client = self._make_client()
            client.chat(MESSAGES, response_format={"type": "json_object"})

        init_kwargs = mock_cls.call_args.kwargs
        assert init_kwargs.get("format") == "json"

    def test_no_json_format_when_response_format_not_provided(self):
        with _patched_ollama() as mock_cls:
            client = self._make_client()
            client.chat(MESSAGES)

        init_kwargs = mock_cls.call_args.kwargs
        assert "format" not in init_kwargs

    def test_uses_default_base_url_when_not_set(self):
        with _patched_ollama() as mock_cls:
            client = LLMClient(
                provider="ollama", api_key=None, model="llama3.2", base_url=None
            )
            # Clear base_url to simulate missing env var
            client.base_url = None
            client.chat(MESSAGES)

        init_kwargs = mock_cls.call_args.kwargs
        assert init_kwargs["base_url"] == "http://localhost:11434"


# ---------------------------------------------------------------------------
# chat() — Gemini backend
# ---------------------------------------------------------------------------

class TestChatGemini:
    def _make_client(self, model="gemini-2.0-flash"):
        return LLMClient(provider="gemini", api_key="AIza-test", model=model)

    def test_returns_model_content(self):
        with _patched_gemini("Hi Gemini"):
            client = self._make_client()
            result = client.chat(MESSAGES)
        assert result == "Hi Gemini"

    def test_passes_correct_kwargs_to_chat_gemini(self):
        with _patched_gemini() as mock_cls:
            client = self._make_client(model="gemini-1.5-pro")
            client.chat(MESSAGES, temperature=0.3, max_tokens=1024)

        init_kwargs = mock_cls.call_args.kwargs
        assert init_kwargs["model"] == "gemini-1.5-pro"
        assert init_kwargs["temperature"] == 0.3
        assert init_kwargs["max_tokens"] == 1024
        assert init_kwargs["google_api_key"] == "AIza-test"


# ---------------------------------------------------------------------------
# <think> tag removal (applies to all providers)
# ---------------------------------------------------------------------------

class TestThinkTagRemoval:
    def _chat_with_openai_response(self, content: str) -> str:
        """Helper: get chat() result for a given raw model output."""
        with patch("openai.OpenAI") as mock_cls:
            mock_instance = MagicMock()
            mock_cls.return_value = mock_instance
            mock_instance.chat.completions.create.return_value = (
                _mock_openai_response(content)
            )
            client = LLMClient(provider="openai", api_key="sk-test")
            return client.chat(MESSAGES)

    def test_think_tag_single_line_removed(self):
        raw = "<think>internal reasoning</think>Final answer"
        assert self._chat_with_openai_response(raw) == "Final answer"

    def test_think_tag_multiline_removed(self):
        raw = "<think>\nline1\nline2\n</think>Answer"
        assert self._chat_with_openai_response(raw) == "Answer"

    def test_response_without_think_tags_unchanged(self):
        raw = "Plain response"
        assert self._chat_with_openai_response(raw) == "Plain response"

    def test_leading_and_trailing_whitespace_stripped(self):
        raw = "  \n  Result  \n  "
        assert self._chat_with_openai_response(raw) == "Result"


# ---------------------------------------------------------------------------
# chat_json() — JSON parsing and markdown fence stripping
# ---------------------------------------------------------------------------

class TestChatJson:
    def _chat_json_with(self, content: str):
        """Drive chat_json() by mocking the underlying chat() return value."""
        with patch.object(LLMClient, "chat", return_value=content):
            with patch("openai.OpenAI"):
                client = LLMClient(provider="openai", api_key="sk-test")
            return client.chat_json(MESSAGES)

    def test_plain_json_parsed_correctly(self):
        result = self._chat_json_with('{"key": "value", "num": 42}')
        assert result == {"key": "value", "num": 42}

    def test_markdown_json_fence_stripped_and_parsed(self):
        fenced = "```json\n{\"a\": 1}\n```"
        assert self._chat_json_with(fenced) == {"a": 1}

    def test_plain_code_fence_stripped_and_parsed(self):
        fenced = "```\n{\"b\": 2}\n```"
        assert self._chat_json_with(fenced) == {"b": 2}

    def test_invalid_json_raises_value_error(self):
        with patch.object(LLMClient, "chat", return_value="not valid json"):
            with patch("openai.OpenAI"):
                client = LLMClient(provider="openai", api_key="sk-test")
            with pytest.raises(ValueError, match="JSON格式无效"):
                client.chat_json(MESSAGES)

    def test_chat_json_passes_json_object_response_format(self):
        """chat_json must request json_object format from the underlying chat()."""
        with patch.object(LLMClient, "chat", return_value='{"x": 1}') as mock_chat:
            with patch("openai.OpenAI"):
                client = LLMClient(provider="openai", api_key="sk-test")
            client.chat_json(MESSAGES, temperature=0.1, max_tokens=128)

        _, kwargs = mock_chat.call_args
        assert kwargs.get("response_format") == {"type": "json_object"}
        assert kwargs.get("temperature") == 0.1
        assert kwargs.get("max_tokens") == 128
