"""llmclient.py — DeepSeek / Ollama 统一客户端（每轮全新调用，无对话历史）。"""

import json
import os
import re
import time
import urllib.request


class LLMError(Exception):
    pass


class ToolCall:
    """function calling 结果：{id, name, arguments(dict)}。"""

    def __init__(self, tid, name, arguments):
        self.id = tid or f"tc_{name}_{abs(hash((name, str(arguments)))) % 100000}"
        self.name = name
        self.arguments = arguments or {}


def _parse_tool_args(raw) -> dict:
    """容错解析工具参数（JSON 字符串 → dict）。"""
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    s = str(raw).strip()
    if not s:
        return {}
    try:
        d = json.loads(s)
        return d if isinstance(d, dict) else {}
    except Exception:
        # 兜底：剥前后花括号暴力解析（三级容错精神）
        try:
            m = re.search(r"\{.*\}", s, re.S)
            if m:
                d = json.loads(m.group(0))
                return d if isinstance(d, dict) else {}
        except Exception:
            return {}
        return {}


def estimate_tokens(text: str) -> int:
    import re
    chinese = len(re.findall(r"[\u4e00-\u9fff]", text))
    other = len(text) - chinese
    return chinese + other // 4


class OllamaClient:
    def __init__(self, base_url="http://localhost:11434", api_base=None, api_key=None,
                 model="gemma4:12b", num_ctx=131072, temperature=0.8, timeout=600,
                 max_tokens=8192):
        # api_base 为别名（config 里沿用 sdk 命名）；api_key 本地服务不需要，接受但不使用
        # /v1 后缀归一化：原生 /api/chat 端点不带 /v1（OpenAI 兼容路径才带）
        self.base_url = (api_base or base_url).rstrip("/")
        if self.base_url.endswith("/v1"):
            self.base_url = self.base_url[:-3]
        self.model = model
        self.num_ctx = num_ctx
        self.temperature = temperature
        self.timeout = timeout
        self.max_tokens = int(max_tokens or 8192)

    # ── 消息构建（ollama 原生格式：tool_calls 无 id/type，arguments 为 dict）──
    def build_assistant_msg(self, text, calls):
        m = {"role": "assistant", "content": text or ""}
        if calls:
            m["tool_calls"] = [{"function": {"name": c.name, "arguments": c.arguments}}
                               for c in calls]
        return m

    def build_tool_msg(self, call, result):
        return {"role": "tool", "content": str(result)}

    def chat(self, system, user):
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            # 统一关闭思考（与 DeepSeek extra_body.thinking.disabled 对齐）：
            # 本地 glm 默认开 thinking（实测每轮烧 1800+ 推理 token、拖慢数倍）；
            # 开关必须在请求**顶层**（options 里的 think 无效，实测）
            "think": False,
            "options": {"num_ctx": self.num_ctx, "temperature": self.temperature,
                        "num_predict": self.max_tokens},
        }
        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as ex:
            raise LLMError(f"ollama 调用失败: {ex}") from ex
        text = (data.get("message") or {}).get("content", "") or ""
        usage = {
            "prompt_tokens": data.get("prompt_eval_count", estimate_tokens(system + user)),
            "completion_tokens": data.get("eval_count", estimate_tokens(text)),
        }
        return text, usage, time.time() - t0

    def chat_tools(self, system, user, tools):
        """function calling 版本（ollama /api/chat 原生 tools）。
        tools = OpenAI 风格 [{"type": "function", "function": {...}}]。
        返回 (text, tool_calls: list[ToolCall], usage, latency)。"""
        return self.chat_messages(
            [{"role": "system", "content": system},
             {"role": "user", "content": user}], tools)

    def chat_messages(self, messages, tools=None):
        """完整对话版：messages 含原生 tool_calls/tool 消息（v1 参考实现的循环结构）。
        返回 (text, tool_calls: list[ToolCall], usage, latency)。"""
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": False,
            "options": {"num_ctx": self.num_ctx, "temperature": self.temperature,
                        "num_predict": self.max_tokens},
        }
        if tools:
            payload["tools"] = tools
        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as ex:
            raise LLMError(f"ollama 调用失败: {ex}") from ex
        msg = data.get("message") or {}
        text = msg.get("content", "") or ""
        calls = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function") or {}
            calls.append(ToolCall(
                tc.get("id"), fn.get("name", ""),
                _parse_tool_args(fn.get("arguments"))))
        usage = {
            "prompt_tokens": data.get("prompt_eval_count", estimate_tokens(
                "".join(m.get("content") or "" for m in messages))),
            "completion_tokens": data.get("eval_count", estimate_tokens(text)),
        }
        return text, calls, usage, time.time() - t0


class DeepSeekClient:
    def __init__(self, base_url="https://api.deepseek.com/v1", model="deepseek-chat",
                 api_key=None, temperature=0.8, timeout=180, thinking_budget=0):
        api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise LLMError("缺少 DeepSeek API key（config 或环境变量 DEEPSEEK_API_KEY）")
        try:
            from openai import OpenAI
            import httpx
        except ImportError:
            raise LLMError("使用 DeepSeek 需要安装 openai 库") from None
        # timeout 分设（2026-08-28 修复：单 float 只设 connect/read 同值，服务器
        # trickle 心跳字节可绕过 read 间隔超时；显式 httpx.Timeout 逐字段控制）。
        # read=120 覆盖"无响应挂起"；future 层（realengine DISTILL_BATCH_TIMEOUT）
        # 再加总时长兜底，双保险。
        self.client = OpenAI(
            base_url=base_url, api_key=api_key,
            timeout=httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=10.0))
        self.model = model
        self.temperature = temperature
        # thinking_budget>0 → 启用限量思考（如 1024 token）；默认 0 = 关闭（既有标准）
        self.thinking_budget = int(thinking_budget or 0)

    # ── 消息构建（OpenAI SDK 格式：id/type + arguments JSON 字符串）──
    def build_assistant_msg(self, text, calls):
        m = {"role": "assistant", "content": text or ""}
        if calls:
            m["tool_calls"] = [{
                "id": c.id, "type": "function",
                "function": {"name": c.name,
                             "arguments": json.dumps(c.arguments, ensure_ascii=False)},
            } for c in calls]
        return m

    def build_tool_msg(self, call, result):
        return {"role": "tool", "tool_call_id": call.id, "content": str(result)}

    def _thinking_extra(self):
        """thinking 控制：budget>0 → 限量启用；默认 0 → 关闭（既有标准）。"""
        if self.thinking_budget > 0:
            return {"thinking": {"type": "enabled", "budget_tokens": self.thinking_budget}}
        return {"thinking": {"type": "disabled"}}

    def chat(self, system, user):
        t0 = time.time()
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=self.temperature,
                # 显式关闭思考：deepseek-v4-flash 默认开 reasoning（实测每轮烧
                # reasoning tokens 是延迟/成本大头），实验平台统一关
                extra_body=self._thinking_extra(),
            )
        except Exception as ex:
            raise LLMError(f"deepseek 调用失败: {ex}") from ex
        text = (resp.choices[0].message.content or "").strip()
        usage = resp.usage
        return text, {
            "prompt_tokens": usage.prompt_tokens if usage else estimate_tokens(system + user),
            "completion_tokens": usage.completion_tokens if usage else estimate_tokens(text),
        }, time.time() - t0

    def chat_tools(self, system, user, tools):
        """function calling 版本（OpenAI SDK tools）。返回同 OllamaClient.chat_tools。"""
        return self.chat_messages(
            [{"role": "system", "content": system},
             {"role": "user", "content": user}], tools)

    def chat_messages(self, messages, tools=None):
        """完整对话版：messages 含原生 tool_calls/tool 消息。"""
        t0 = time.time()
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                tools=tools or None,
                tool_choice="auto" if tools else None,
                extra_body=self._thinking_extra(),
            )
        except Exception as ex:
            raise LLMError(f"deepseek 调用失败: {ex}") from ex
        msg = resp.choices[0].message
        text = (msg.content or "").strip()
        calls = []
        for tc in msg.tool_calls or []:
            fn = tc.function
            calls.append(ToolCall(tc.id, fn.name, _parse_tool_args(fn.arguments)))
        usage = resp.usage
        return text, calls, {
            "prompt_tokens": usage.prompt_tokens if usage else estimate_tokens(
                "".join(m.get("content") or "" for m in messages)),
            "completion_tokens": usage.completion_tokens if usage else estimate_tokens(text),
        }, time.time() - t0


def make_client(cfg):
    provider = cfg.get("provider", "ollama")
    if provider == "ollama":
        o = cfg.get("ollama", {})
        return OllamaClient(**o)
    if provider == "deepseek":
        d = cfg.get("deepseek", {})
        return DeepSeekClient(**d)
    raise LLMError(f"未知 provider: {provider}")
