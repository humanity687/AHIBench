import os
from typing import Any, Dict, List, Optional, Union

from openai import OpenAI, AzureOpenAI


class LLMClient:
    """OpenAI-compatible LLM client. Supports Ollama, Azure, and any OpenAI-compatible API."""

    def __init__(self, config: Dict[str, Any] = None, **overrides: Any):
        config = config or {}
        config.update(overrides)

        self.api_key = config.get("api_key") or os.getenv("OPENAI_API_KEY") or "ollama"
        self.api_base = config.get("api_base", "http://localhost:11434/v1")
        self.api_type = config.get("api_type")
        self.api_version = config.get("api_version")
        self.timeout = int(config.get("timeout", 60))

        if self.api_type:
            os.environ["OPENAI_API_TYPE"] = self.api_type
        if self.api_version:
            os.environ["OPENAI_API_VERSION"] = self.api_version

        self.client = self._create_client()

        self.model = config.get("model", "llama2")
        self.temperature = float(config.get("temperature", 0.7))
        self.max_tokens = int(config.get("max_tokens", 1024))

        # 标准 OpenAI API 参数：可安全传递给 chat.completions.create()
        self.top_p = config.get("top_p")
        self.frequency_penalty = config.get("frequency_penalty")
        self.presence_penalty = config.get("presence_penalty")

        # 非标准参数（如 thinking 是 DeepSeek 专有），不自动传递给 API
        # 如果使用的 API 支持，请手动配置 extra_api_params
        self.thinking = config.get("thinking", False)

        # 额外 API 参数白名单：只有这些参数会传递给 create()
        # 如果要添加厂商特定参数，在这里加
        self.extra_api_params: Dict[str, Any] = {}
        extra_whitelist = {"stop", "n", "seed", "logprobs", "top_logprobs",
                           "response_format", "user", "extra_body"}
        for k in extra_whitelist:
            if k in config and config[k] is not None:
                self.extra_api_params[k] = config[k]

        self.system_prompt = config.get("system_prompt", "You are a helpful assistant.")
        self.messages: List[Dict[str, str]] = []

        # 原生 Ollama 通道（决策：OpenAI 兼容端点忽略 options.num_ctx，必须走 /api/chat
        # 才能控制上下文长度；native=true 时启用，见 config 的 ollama_options/keep_alive）
        self.native_ollama = bool(config.get("native", False))
        self.ollama_options = dict(config.get("ollama_options", {}) or {})
        self.keep_alive = config.get("keep_alive", "5m")

    def _create_client(self) -> Union[OpenAI, AzureOpenAI]:
        if self.api_type == "azure":
            if not self.api_version:
                raise ValueError("api_version is required for Azure OpenAI")
            return AzureOpenAI(
                api_key=self.api_key,
                azure_endpoint=self.api_base,
                api_version=self.api_version,
                timeout=self.timeout,
            )
        return OpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
            timeout=self.timeout,
        )

    def set_system_prompt(self, prompt: str) -> None:
        self.system_prompt = prompt
        if self.messages and self.messages[0]["role"] == "system":
            self.messages[0]["content"] = prompt
        else:
            self.messages.insert(0, {"role": "system", "content": prompt})

    def clear_context(self) -> None:
        self.messages = []
        self.set_system_prompt(self.system_prompt)

    def get_context(self) -> List[Dict[str, str]]:
        return list(self.messages)

    # 上下文压缩已移交 agent 子类自行管理（决策 A5）：本类不再自动压缩。
    # 需要压缩的 agent 类可自行调用 compress_middle() 或自定义策略。
    _MAX_MESSAGES = 120
    _KEEP_RECENT = 40

    def compress_middle(self, max_messages: int = None, keep_recent: int = None):
        """将中间消息替换为摘要（保留 system prompt + 最近 N 条）。由 agent 子类按需调用。"""
        max_messages = max_messages or self._MAX_MESSAGES
        keep_recent = keep_recent or self._KEEP_RECENT
        if len(self.messages) <= max_messages:
            return
        system_msg = self.messages[0] if self.messages and self.messages[0]["role"] == "system" else None
        recent = self.messages[-(keep_recent):]
        removed = self.messages[1:-(keep_recent)] if system_msg else self.messages[:-(keep_recent)]
        summary_lines = []
        for m in removed[-6:]:
            role = m.get("role", "?")
            snippet = m.get("content", "")[:80].replace("\n", " ")
            summary_lines.append(f"[{role}]: {snippet}")
        summary = "【上下文摘要】\n" + "\n".join(summary_lines) + "\n（以上是摘要，更早的上下文已被压缩）"
        if system_msg:
            self.messages = [system_msg] + [{"role": "assistant", "content": summary}] + recent
        else:
            self.messages = [{"role": "assistant", "content": summary}] + recent

    def send_message(self, content: str, role: str = "user") -> str:
        self.messages.append({"role": role, "content": content})
        if self.native_ollama:
            return self._send_native_ollama()

        # 组装标准参数
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": self.messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        # 标准 OpenAI 参数：逐个添加（只有设置了才传递）
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        if self.frequency_penalty is not None:
            kwargs["frequency_penalty"] = self.frequency_penalty
        if self.presence_penalty is not None:
            kwargs["presence_penalty"] = self.presence_penalty

        # 白名单中的额外参数
        kwargs.update(self.extra_api_params)

        # 注意: thinking 是 DeepSeek 专有参数，OpenAI/Ollama API 不支持
        # 如需使用，请通过 extra_body 传递（需 API 支持）
        # 示例: config 中加 "extra_body": {"thinking": {"type": "enabled"}}

        response = self.client.chat.completions.create(**kwargs)

        reply = response.choices[0].message.content or ""
        self.messages.append({"role": "assistant", "content": reply})
        return reply

    def _send_native_ollama(self) -> str:
        """走 Ollama 原生 /api/chat（支持 options.num_ctx / keep_alive）。"""
        import json
        import urllib.request
        base = self.api_base
        if base.endswith("/v1"):
            base = base[:-3]
        payload = {
            "model": self.model,
            "messages": self.messages,
            "stream": False,
            "keep_alive": self.keep_alive,
            # 统一关闭思考（与 realtest OllamaClient/DeepSeek 对齐）：
            # 顶层 think=False 才有效（options 里的 think 被忽略，实测）
            "think": False,
            "options": {
                "temperature": self.temperature,
                "top_p": self.top_p if self.top_p is not None else 0.9,
                "num_predict": self.max_tokens,
            },
        }
        payload["options"].update(self.ollama_options)
        req = urllib.request.Request(
            f"{base}/api/chat",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode())
        reply = (data.get("message", {}) or {}).get("content", "") or ""
        self.messages.append({"role": "assistant", "content": reply})
        return reply
