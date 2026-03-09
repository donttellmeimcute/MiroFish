"""
LLM客户端封装
支持三种后端：
  - openai  : 任何兼容 OpenAI SDK 格式的 API（OpenAI、Qwen、MiniMax 等）
  - ollama  : 本地 Ollama 服务（通过 langchain-ollama）
  - gemini  : Google Gemini（通过 langchain-google-genai）

通过 .env 中的 LLM_PROVIDER 变量选择后端。
"""

import json
import re
from typing import Optional, Dict, Any, List

from ..config import Config


class LLMClient:
    """LLM客户端 - 支持多种后端（OpenAI兼容、Ollama、Gemini）"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
    ):
        self.api_key = api_key or Config.LLM_API_KEY
        self.base_url = base_url or Config.LLM_BASE_URL
        self.model = model or Config.LLM_MODEL_NAME
        self.provider = (provider or Config.LLM_PROVIDER).lower()

        if self.provider not in ('openai', 'ollama', 'gemini'):
            raise ValueError(
                f"LLM_PROVIDER 不支持: '{self.provider}'。"
                "请设置为 'openai'、'ollama' 或 'gemini'"
            )

        if self.provider in ('openai', 'gemini') and not self.api_key:
            label = 'Google' if self.provider == 'gemini' else 'OpenAI'
            raise ValueError(f"LLM_API_KEY ({label} API Key) 未配置")

        # 预先初始化 OpenAI 客户端（轻量，复用连接池）
        if self.provider == 'openai':
            from openai import OpenAI
            self._openai_client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )

    # ------------------------------------------------------------------
    # 内部辅助方法
    # ------------------------------------------------------------------

    @staticmethod
    def _build_langchain_messages(messages: List[Dict[str, str]]):
        """将 OpenAI 格式的消息列表转换为 LangChain 消息对象。"""
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        lc_messages = []
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            if role == 'system':
                lc_messages.append(SystemMessage(content=content))
            elif role == 'assistant':
                lc_messages.append(AIMessage(content=content))
            else:
                lc_messages.append(HumanMessage(content=content))
        return lc_messages

    def _invoke_ollama(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        json_mode: bool = False,
    ) -> str:
        """通过 LangChain 调用本地 Ollama 服务。"""
        from langchain_ollama import ChatOllama

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "base_url": self.base_url or 'http://localhost:11434',
            "temperature": temperature,
            "num_predict": max_tokens,
        }
        if json_mode:
            kwargs["format"] = "json"

        client = ChatOllama(**kwargs)
        lc_messages = self._build_langchain_messages(messages)
        response = client.invoke(lc_messages)
        return response.content

    def _invoke_gemini(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """通过 LangChain 调用 Google Gemini API。"""
        from langchain_google_genai import ChatGoogleGenerativeAI

        client = ChatGoogleGenerativeAI(
            model=self.model,
            google_api_key=self.api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        lc_messages = self._build_langchain_messages(messages)
        response = client.invoke(lc_messages)
        return response.content

    # ------------------------------------------------------------------
    # 公共接口（与原版保持一致）
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: Optional[Dict] = None,
    ) -> str:
        """
        发送聊天请求。

        Args:
            messages: 消息列表（OpenAI 格式）
            temperature: 温度参数
            max_tokens: 最大 token 数
            response_format: 响应格式（仅 provider=openai 时生效，
                             例如 {"type": "json_object"}）

        Returns:
            模型响应文本
        """
        if self.provider == 'ollama':
            json_mode = response_format is not None
            content = self._invoke_ollama(messages, temperature, max_tokens, json_mode)
        elif self.provider == 'gemini':
            content = self._invoke_gemini(messages, temperature, max_tokens)
        else:
            # OpenAI 兼容模式
            kwargs: Dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if response_format:
                kwargs["response_format"] = response_format
            response = self._openai_client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content

        # 部分模型（如 MiniMax M2.5）会在 content 中包含 <think> 思考内容，需要移除
        content = re.sub(r'<think>[\s\S]*?</think>', '', content).strip()
        return content

    def chat_json(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> Dict[str, Any]:
        """
        发送聊天请求并返回解析后的 JSON 对象。

        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大 token 数

        Returns:
            解析后的 JSON 对象
        """
        response = self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        # 清理 markdown 代码块标记
        cleaned = response.strip()
        cleaned = re.sub(r'^```(?:json)?\s*\n?', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\n?```\s*$', '', cleaned)
        cleaned = cleaned.strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            raise ValueError(f"LLM返回的JSON格式无效: {cleaned}")
