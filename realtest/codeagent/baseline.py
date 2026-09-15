#!/usr/bin/env python3
"""baseline.py — 代码智能体（基线：上下文窗口 agent，无记忆树）。

同壳同工具（read/write/command/finish），唯一变量 = 记忆层。
上下文 = 任务描述 + 最近 N 轮工具往返（窗口滑动，超预算截断）——模拟
"能读文件/执行代码/看报错，无跨会话记忆"的普通 coding agent。
无 save_context / recall / 蒸馏 / 检索 / 笔记本。
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llmclient import make_client, LLMError

# 与实验组共享的 read/write/command/finish 工具（无 save_context/recall）
from agent import TOOLS, SYSTEM, DANGEROUS, _detect_media, _py_syntax_check

BASELINE_TOOLS = [t for t in TOOLS
                  if t["function"]["name"] in ("read", "write", "edit", "command", "finish")]

BASELINE_SYSTEM = """你是一名资深软件工程师，正在修复一个旧版多智能体平台（AHI）的 bug。
你可以调用工具读写文件、执行命令。每次工具调用后你会看到结果，然后继续下一轮。

【工作纪律】
1. 定位 bug：先复现/观察现象 → read 相关文件 → 找到根因 → 最小改动修复（小改动用 edit
   定点替换，大改动用 write 整文件）→ 运行验收脚本验证（accept_tX.py，退出码 0 = 通过）→
   调用 finish。
2. 代码/配置/API 签名以 read 到的原文为准，禁止编造。
3. 注意：你的上下文有长度限制，过早的内容会被截断——重要信息（如已读过的文件
   结构、已确认的根因）请靠 read 重新定位，不要假设自己还记得。
4. 输出：直接给出工具调用即可。"""


class BaselineAgent:
    """基线：纯上下文窗口。"""

    def __init__(self, workdir, cfg, verbose=False):
        self.workdir = str(Path(workdir).resolve())
        self.cfg = cfg
        self.llm = make_client(cfg)
        self.verbose = verbose
        # 窗口预算（字符）：与实验组组装预算同量级（16k），保证可对比
        self.budget = int(cfg.get("assembly_budget", 16000))
        self.round_no = 0
        self.done = False
        self.history = []        # [(kind, text)] 工具往返记录
        self.tool_uses = {}
        self.trace = []
        self.last_usage = None
        self.total_tokens = {"prompt": 0, "completion": 0}
        self.llm_seconds = 0.0

    # ── 工具实现（与实验组共享逻辑，无记忆侧效应）──────

    def _abs(self, path):
        p = Path(path).expanduser()
        if not p.is_absolute():
            p = Path(self.workdir) / p
        return p.resolve()

    def tool_read(self, path, offset=1, limit=None):
        try:
            p = self._abs(path)
            if not p.exists():
                return f"错误：文件不存在 - {p}"
            if not p.is_file():
                return f"错误：路径不是文件 - {p}"
            content = p.read_text(encoding="utf-8", errors="replace")
        except PermissionError:
            return f"错误：没有权限读取文件 - {path}"
        except Exception as e:
            return f"读取文件失败：{e}"
        lines = content.splitlines()
        total = len(lines)
        offset = max(1, int(offset or 1))
        lines = lines[offset - 1:] if not limit else lines[offset - 1: offset - 1 + int(limit)]
        width = len(str(total))
        body = "\n".join(f"{i + offset:{width}}  {ln}" for i, ln in enumerate(lines))
        if len(body) > 12000:
            body = body[:12000] + f"\n...（共 {total} 行，输出截断）"
        return f"--- {p}（共 {total} 行）---\n{body}"

    def tool_write(self, path, content, mode="overwrite"):
        try:
            p = self._abs(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "a" if mode == "append" else "w", encoding="utf-8") as f:
                f.write(content)
            check = "" if mode == "append" else f"\n语法检查：{_py_syntax_check(content, p)}"
            return f"成功{'追加' if mode == 'append' else '写入'}文件：{p}{check}"
        except Exception as e:
            return f"写入失败：{e}"

    def tool_edit(self, path, old_string, new_string, replace_all=False):
        from agent import _locate_edit, _nearby_hint, _adopt_indent, _NUM_PREFIX_RE
        try:
            p = self._abs(path)
            if not p.exists():
                return f"错误：文件不存在 - {p}"
            content = p.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            return f"编辑失败：{e}"
        n = content.count(old_string)
        if n == 0:
            located, how = _locate_edit(content, old_string)
            if located is None:
                return (f"错误：old_string 在文件中未找到（文件共 {len(content)} 字符）。"
                        f"附近相似内容：\n{_nearby_hint(content, old_string)}")
            if how >= 2:
                model_old_clean = "\n".join(_NUM_PREFIX_RE.sub("", ln)
                                            for ln in old_string.rstrip("\n").split("\n"))
                new_string = _adopt_indent(model_old_clean, located, new_string)
            old_string = located
            n = content.count(old_string)
        if n > 1 and not replace_all:
            return f"错误：old_string 出现 {n} 次，请提供更长锚点或设 replace_all=true"
        new_content = content.replace(old_string, new_string) if replace_all \
            else content.replace(old_string, new_string, 1)
        try:
            with open(p, "w", encoding="utf-8") as f:
                f.write(new_content)
            return f"已替换 {'全部' if replace_all else ''}{n if replace_all else 1} 处：{p}\n语法检查：{_py_syntax_check(new_content, p)}"
        except Exception as e:
            return f"写入失败：{e}"

    def tool_command(self, command):
        cmd = command.strip()
        low = cmd.lower()
        for d in DANGEROUS:
            if d in low:
                return f"错误：禁止执行 {d.strip()} 类命令（删除类操作不允许）。"
        try:
            r = subprocess.run(cmd, shell=True, cwd=self.workdir,
                               capture_output=True, timeout=60)
            out = (r.stdout or b"").decode("utf-8", errors="replace")
            err = (r.stderr or b"").decode("utf-8", errors="replace")
            text = (out + err).strip()
            if r.returncode != 0:
                text = f"退出码 {r.returncode}\n{text}"
            if not text:
                text = "（命令执行成功，无输出）"
            if len(text) > 4000:
                text = text[:4000] + "\n...（输出过长已截断）"
        except subprocess.TimeoutExpired:
            text = "错误：命令执行超时（60s）"
        except Exception as e:
            text = f"执行失败：{e}"
        return text

    def tool_finish(self, summary):
        self.done = True
        return "任务已结束。"

    def _dispatch(self, name, args):
        fn = {
            "read": self.tool_read,
            "write": self.tool_write,
            "edit": self.tool_edit,
            "command": self.tool_command,
            "finish": self.tool_finish,
        }.get(name)
        if fn is None:
            return f"未知工具：{name}"
        try:
            return fn(**args)
        except TypeError:
            try:
                return fn()
            except Exception as e:
                return f"工具 {name} 参数错误：{e}"

    # ── 主循环（v1 参考结构：原生工具对话；无记忆层，超长裁剪）──

    MAX_CONV_CHARS = 40000   # 对话即"记忆"：上限（无独立记忆层）

    def _trim_messages(self):
        total = sum(len(m.get("content") or "") for m in self.messages)
        while total > self.MAX_CONV_CHARS and len(self.messages) > 2:
            i = 2
            removed = 0
            while i < len(self.messages):
                if self.messages[i].get("tool_calls"):
                    j = i + 1
                    while j < len(self.messages) and self.messages[j].get("role") == "tool":
                        j += 1
                    chunk = self.messages[i:j]
                    del self.messages[i:j]
                    removed = sum(len(m.get("content") or "") for m in chunk)
                    break
                i += 1
            if not removed:
                removed = len(self.messages[2].get("content") or "") + 1
                del self.messages[2]
            total -= removed

    def run_task(self, task_desc, task_id="T?", max_rounds=40, max_seconds=1800,
                 progress=None):
        # 原生对话结构（v1 参考实现）：system + task + 工具往返
        self.messages = [
            {"role": "system", "content": BASELINE_SYSTEM},
            {"role": "user", "content": task_desc},
        ]
        t0 = time.time()
        notes = []
        for rnd in range(1, max_rounds + 1):
            if time.time() - t0 > max_seconds:
                notes.append("超时")
                break
            self.round_no = rnd
            self._trim_messages()
            if self.verbose:
                print(f"\n---- [R{rnd}] 对话 {sum(len(m.get('content') or '') for m in self.messages)} 字符 ----",
                      flush=True)
            if progress:
                progress(rnd, f"LLM 调用（对话 {sum(len(m.get('content') or '') for m in self.messages)} 字符）")
            try:
                text, calls, usage, lat = self.llm.chat_messages(self.messages, BASELINE_TOOLS)
            except LLMError as e:
                notes.append(f"LLM 错误：{e}")
                if progress:
                    progress(rnd, f"LLM 错误：{e}")
                time.sleep(2)
                continue
            self.last_usage = usage
            self.total_tokens["prompt"] += usage.get("prompt_tokens") or 0
            self.total_tokens["completion"] += usage.get("completion_tokens") or 0
            self.llm_seconds += lat
            if self.verbose:
                print(f"[R{rnd}] LLM {lat:.1f}s in={usage.get('prompt_tokens')} "
                      f"out={usage.get('completion_tokens')} calls={len(calls)}", flush=True)
            self.messages.append(self.llm.build_assistant_msg(text, calls))
            if not calls:
                if getattr(self, "_idle_rounds", 0) >= 1:
                    notes.append("连续两轮无工具调用")
                    break
                self._idle_rounds = getattr(self, "_idle_rounds", 0) + 1
                continue
            self._idle_rounds = 0
            for tc in calls:
                name, args = tc.name, tc.arguments
                self.tool_uses[name] = self.tool_uses.get(name, 0) + 1
                if name == "finish":
                    result = self.tool_finish(args.get("summary", ""))
                else:
                    result = self._dispatch(name, args)
                rtext = str(result).replace("\n", " ")[:120]
                self.trace.append((rnd, name, dict(args), str(result)[:400]))
                self.messages.append(self.llm.build_tool_msg(tc, result))
                if progress:
                    progress(rnd, f"🔧 {name}({json.dumps(args, ensure_ascii=False)[:80]}) → {rtext[:80]}")
                if name == "finish":
                    return {"ok": True, "rounds": rnd, "summary": args.get("summary", ""),
                            "notes": notes}
        return {"ok": False, "rounds": self.round_no, "summary": "未完成", "notes": notes}

    def metrics(self):
        return {
            "rounds": self.round_no,
            "tool_uses": self.tool_uses,
            "conversation_chars": sum(len(m.get("content") or "") for m in getattr(self, "messages", [])),
            "total_tokens": self.total_tokens,
            "llm_seconds": round(self.llm_seconds, 1),
        }
