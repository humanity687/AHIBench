#!/usr/bin/env python3
"""agent.py — 代码智能体（实验组：记忆树 + 原生 function calling）。

架构（复用 v2 引擎）：
- RealEngine（core.py 机制：区间覆盖树 / 注意力窗口 / merge 条件式触发 / 编辑路径 / 混合检索 / 笔记本）
- 工具 = 原生 function calling（llmclient.chat_tools，ollama + deepseek 双后端）
- 每工具调用 = 一轮新组装（AGENTS.md §2 调度原则）；tick/merge 条件式触发
- read 结果按介质切分灌入记忆树（source=file:path，AST/配置项/md 标题 —— 卡片切分策略）
- write 走卡级 diff 编辑路径（§4.6 真实压力测试）
- 轨道B：save_context → state 条目；finish → worklog（调试经验跨任务复用）
- verbatim 兜底：原子层原文逐字可见 + [recall]/下钻展开

用法（由 harness 驱动）：
    agent = CodeAgent(workdir, cfg, verbose)
    result = agent.run_task(task_desc, task_id, max_rounds, max_seconds)
"""

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chattext import split_media, split_chat
from core import Atom
from llmclient import make_client, LLMError
from realengine import RealEngine

# ── 工具 schema（OpenAI 风格，ollama/deepseek 通用）────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read",
            "description": "读取文件内容（带行号）。代码/配置/文档文件读取后会自动进入你的记忆。"
                           "offset 为起始行号（1 起），limit 为最大行数，均可选。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文件路径（绝对或相对工作目录）"},
                    "offset": {"type": "integer", "description": "起始行号（默认 1）"},
                    "limit": {"type": "integer", "description": "最大读取行数"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write",
            "description": "写入/追加文件。overwrite 覆盖整文件，append 追加到末尾。"
                           "覆盖时系统会对旧内容做精确失效（编辑路径）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文件路径"},
                    "content": {"type": "string", "description": "文件内容"},
                    "mode": {"type": "string", "enum": ["overwrite", "append"], "description": "默认 overwrite"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit",
            "description": "精确替换文件中的一段文本（定点修复，推荐用于小改动，比整文件 write 稳妥）。"
                           "old_string 必须唯一匹配（含完整上下文锚点）；替换后系统对旧内容精确失效。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文件路径"},
                    "old_string": {"type": "string", "description": "被替换的原文（必须唯一）"},
                    "new_string": {"type": "string", "description": "替换后的新文本"},
                    "replace_all": {"type": "boolean", "description": "true=替换全部匹配（默认 false）"},
                },
                "required": ["path", "old_string", "new_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "command",
            "description": "执行 shell 命令（工作目录内）。禁止删除类命令。"
                           "用于运行测试/检查语法/查看目录等。",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "shell 命令"},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_context",
            "description": "把确认无误的关键事实/决策写入事实层（跨轮、跨任务永久保存，不衰减）："
                           "如项目架构结论、关键配置、已修复的 bug 与修复方式。"
                           "topic 用于去重（同 topic 覆盖更新）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "要保存的事实/决策内容"},
                    "topic": {"type": "string", "description": "主题关键词（同主题覆盖，建议 2-8 字）"},
                },
                "required": ["content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recall",
            "description": "主动回忆：检索记忆树中与查询相关的旧内容（摘要或原文），"
                           "命中的内容将在下一轮注入你的记忆流。定位问题时先看记忆流，"
                           "记忆流没有的细节再用 read 精确读取。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "检索关键词/描述"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "完成任务。调用前务必先运行验收脚本确认修复正确；"
                           "summary 写修复了什么、改了什么文件。",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string", "description": "修复总结（文件+改动+验证结果）"},
                },
                "required": ["summary"],
            },
        },
    },
]

TOOL_NAMES = [t["function"]["name"] for t in TOOLS]

SYSTEM = """你是一名资深软件工程师，正在修复一个旧版多智能体平台（AHI）的 bug。你的全部工作记忆来自【记忆流】（每轮全新组装，由记忆树管理，不依赖对话历史）。

【记忆流格式】每项带 id 与从属标注：
- N12·摘要 为摘要卡（旧内容已蒸馏）；#34·原文 为原文卡（代码按函数/类切分，配置按项，文档按标题）；
- ⊂Nx 表示父卡；缩进表示层级；〔源:file:路径〕标注内容来自哪个文件；
- "召回影子"为检索注入的旧内容；记忆流按文件位置排序，末尾是最近读入的内容；
- 记忆流之后是【笔记本·事实层】（去重发送：同 topic 只保留最新）。

【工作纪律】
1. 先看记忆流确认已知信息，记忆流中没有的细节再用 read 工具精确读取（行号定位）。
2. 代码/配置/API 签名一律以 read 到的原文为准（verbatim），禁止凭记忆编造。
3. read 文件后其内容会自动进入记忆（按函数/配置项切分），但 read 的返回值本身
   也会出现在工具结果里——两者一致。
4. 确认的关键事实（项目结构/根因/修复方案）用 save_context 写入事实层；
   每个 bug 修复完成后用 finish 总结。
4b. **改前必须读**：对已存在的文件做 edit/write 前，必须先 read 过该文件
   （系统会把读过的文件内容存进记忆树作为旧基线；未读过的文件 edit/write 会被拒绝，
   返回提示请先 read）。新文件直接用 write 创建。
4c. **docstring 强制**：写/改 Python 函数与类时**必须写 docstring**（一句话说明用途即可）
   ——记忆树对代码原子的检索以 docstring 为主索引，无 docstring 的代码跨轮无法召回。
5. 定位 bug：先复现/观察现象 → read 相关文件 → 找到根因 → 最小改动修复（小改动用 edit
   定点替换，大改动用 write 整文件）→ 运行验收脚本验证。验收脚本在任务目录下
   （accept_tX.py），`python3 accept_tX.py` 退出码 0 = 通过。
6. 每轮 LLM 调用后会执行你请求的工具，结果写回记忆；然后进入下一轮。
   不要假设记忆流不变——每次调用前都会刷新。
7. 输出：直接给出工具调用即可；需要向用户说明时用自然语言一并输出。
"""

# 命令黑名单（参考 v1 参考实现）
DANGEROUS = ["rm ", "rm -", "del ", "rmdir", "rd ", "erase", "shred", "unlink",
             "remove-item", " && rm", " | rm", "mv "]


def _detect_media(path: str) -> tuple:
    """(media, lang, fmt)：按后缀判断切分介质。"""
    p = path.lower()
    if p.endswith((".py", ".pyw", ".pyi")):
        return "code", "py", None
    if p.endswith((".json", ".yaml", ".yml", ".ini", ".toml", ".cfg", ".conf")):
        return "config", None, None
    if p.endswith((".md", ".rst", ".txt")):
        return "md" if p.endswith((".md",)) else "chat", None, None
    return "code", None, None


_NUM_PREFIX_RE = re.compile(r"^\s*\d{1,6}\s*[|:]?\s*")


def _locate_edit(content, old):
    """edit 容错定位：1) 精确  2) 剥行号前缀  3) 首行缩进不敏感
    4) 逐行去缩进正则匹配（任意缩进，顺序敏感）。
    返回 (匹配到的真实 old 文本, 定位方式)。失败返回 (None, 0)。"""
    old = old.rstrip("\n")
    if old in content:
        return old, 1
    cleaned = "\n".join(_NUM_PREFIX_RE.sub("", ln) for ln in old.split("\n"))
    if cleaned in content:
        return cleaned, 2
    first = cleaned.split("\n")[0].lstrip()
    if first:
        idx = content.find(first)
        if idx >= 0:
            line_start = content.rfind("\n", 0, idx) + 1
            actual_indent = content[line_start:idx]
            rebuilt = actual_indent + cleaned.lstrip()
            if rebuilt in content:
                return rebuilt, 3
    # 4) 逐行去缩进匹配（模型复制 read 输出时行号+缩进常整体漂移）
    try:
        pat_lines = [re.escape(ln.lstrip()) if ln.strip() else r"[ \t]*"
                     for ln in cleaned.split("\n")]
        if pat_lines and pat_lines[0] != r"[ \t]*":
            pattern = "[ \\t]*" + "\\n[ \\t]*".join(pat_lines)
            m = re.search(pattern, content)
            if m:
                return m.group(0), 4
    except re.error:
        pass
    return None, 0


def _nearby_hint(content, old):
    """失败时的邻近内容提示：显示首行匹配处的真实原文宽窗口（供模型精确复制）。
    模型常因自己的历史修改导致锚点过期——给出当前文件真实文本才能打破循环。"""
    first = old.split("\n")[0].strip()
    lines = content.split("\n")
    for i, ln in enumerate(lines):
        if first and (first in ln or ln.strip() == first):
            lo, hi = max(0, i - 2), min(len(lines), i + 16)
            return "\n".join(f"{j + 1}: {lines[j]}" for j in range(lo, hi))
    return "(未找到相似行)"


def _adopt_indent(model_old, real, model_new):
    """缩进收养：模糊匹配成功后，把 new_string 每行缩进按文件真实缩进修正。
    模型从带行号的 read 输出复制锚点时，缩进整体漂移（如 +2 空格）；
    new_string 常带着同样的漂移 → 不修正会生成语法错误代码。"""
    mo = model_old.split("\n")
    rl = real.split("\n")
    mn = model_new.split("\n")
    out = []
    last_shift = 0
    for i, ln in enumerate(mn):
        m_lead = len(ln) - len(ln.lstrip())
        if i < len(mo) and i < len(rl):
            mo_lead = len(mo[i]) - len(mo[i].lstrip())
            rl_lead = len(rl[i]) - len(rl[i].lstrip())
            shift = rl_lead - mo_lead
            last_shift = shift
        else:
            shift = last_shift
        if ln.strip():
            ln = " " * max(0, m_lead + shift) + ln.lstrip()
        out.append(ln)
    return "\n".join(out)


def _py_syntax_check(text, path=""):
    """.py 文件语法检查：返回 "OK" 或错误信息（行号）。"""
    if not str(path).lower().endswith((".py", ".pyw", ".pyi")):
        return "OK（非 Python 文件，跳过语法检查）"
    try:
        import ast
        ast.parse(text)
        return "OK"
    except SyntaxError as e:
        return f"语法错误 L{e.lineno}：{e.msg}"


class CodeAgent:
    """实验组：记忆树代码智能体。"""

    def __init__(self, workdir, cfg, verbose=False):
        self.workdir = str(Path(workdir).resolve())
        self.cfg = cfg
        self.llm = make_client(cfg)
        self.verbose = verbose
        self.budget = int(cfg.get("assembly_budget", 16000))
        self.slot_budget = int(cfg.get("slot_budget", 16000))   # 记忆流槽位上限（拍定 16k）
        self.retr_budget = int(cfg.get("retrieval_budget", 6))
        self.eng = RealEngine(
            self.llm,
            decay=float(cfg.get("decay", 5.0)),
            hp_merge_threshold=float(cfg.get("hp_merge_threshold", 30.0)),
            parent_init_hp=float(cfg.get("parent_init_hp", 70.0)),
            hot_threshold=float(cfg.get("hot_threshold", 60.0)),
            recall_boost=float(cfg.get("recall_boost", 30.0)),
            merge_depth_diff=int(cfg.get("merge_depth_diff", 1)),
            promote_threshold=int(cfg.get("promote_threshold", 3)),
            attention_atom_cap=int(cfg.get("attention_atom_cap", 40)),
            attention_node_cap=int(cfg.get("attention_node_cap", 20)),
            attention_out=float(cfg.get("attention_out", 45.0)),
            attention_in_decay=float(cfg.get("attention_in_decay", 0.5)),
            attention_out_decay=float(cfg.get("attention_out_decay", 3.0)),
            merge_pressure_threshold=int(cfg.get("merge_pressure_threshold", 20)),
            assembly_budget=self.budget,
            retrieval_budget=self.retr_budget,
            style_alpha=0.0,
        )
        self.last_shadows = []
        self.goal = ""
        self.receipt = "初始化：记忆树就绪。"
        self.round_no = 0
        self.done = False
        self.tool_log_hp = float(cfg.get("tool_log_hp", 50.0))  # 轨迹原子初始 HP（拍定 50）
        self.total_tokens = {"prompt": 0, "completion": 0}   # 跨轮累计 token（效率统计）
        self.llm_seconds = 0.0                                # 跨轮累计 LLM 墙钟
        self.trace = []          # 每轮 (工具名, 参数, 结果摘要)
        self.tool_uses = {}      # 工具名 → 次数
        self.distill_calls = 0

    # ── 工具实现 ──────────────────────────────────────

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
        except Exception as e:
            return f"读取文件失败：{e}"

        in_tree = self._file_units(path)
        if in_tree is not None:
            # 精确召回（拍定 2026-08-30）：文件已进树 → 从树按链序 verbatim 拼出
            # **当前版本**（含模型自己的历史修改）——树是单一真源，不读磁盘。
            # 这正是 v2 原子层 verbatim 通道 + 区间召回的应用。
            content = "\n".join(v for v, _ in in_tree)
            self.receipt = (f"read {Path(path).name} → 从记忆树精确召回 "
                            f"{len(in_tree)} 卡（当前版本，含此前修改）。")
        else:
            # 未入树：读磁盘 + 按介质切卡入树
            try:
                content = p.read_text(encoding="utf-8", errors="replace")
            except PermissionError:
                return f"错误：没有权限读取文件 - {path}"
            except Exception as e:
                return f"读取文件失败：{e}"
            src = f"file:{p}"
            media, lang, fmt = _detect_media(path)
            units = split_media(content, media, source=src, lang=lang, fmt=fmt)
            if units:
                ids = self.eng.insert_units(units, left_id=self.eng.tail)
                self.receipt = f"read {Path(path).name} → {len(ids)} 卡入树（{media} 切分）。"

        lines = content.splitlines()
        total = len(lines)
        offset = max(1, int(offset or 1))
        if limit:
            lines = lines[offset - 1: offset - 1 + int(limit)]
        else:
            lines = lines[offset - 1:]
        width = len(str(total))
        body = "\n".join(f"{i + offset:{width}}  {ln}" for i, ln in enumerate(lines))
        max_ret = 12000
        if len(body) > max_ret:
            body = body[:max_ret] + f"\n...（共 {total} 行，返回被截断）"
        return f"--- {p}（共 {total} 行）---\n{body}"

    def tool_edit(self, path, old_string, new_string, replace_all=False):
        try:
            p = self._abs(path)
            if not p.exists():
                return f"错误：文件不存在 - {p}"
            content = p.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            return f"编辑失败：{e}"
        # 改前必须读（拍定 2026-08-30）：树内无该文件旧基线 → 硬拒绝
        old_units = self._file_units(path)
        if old_units is None:
            return f"错误：请先 read {path}（改前必须读：树内没有该文件的旧内容基线）。"
        n = content.count(old_string)
        if n == 0:
            # 容错定位：剥行号前缀/首行缩进不敏感（模型常从带行号的 read 输出复制）
            located, how = _locate_edit(content, old_string)
            if located is None:
                return (f"错误：old_string 在文件中未找到（文件共 {len(content)} 字符）。"
                        f"附近相似内容：\n{_nearby_hint(content, old_string)}")
            # 模糊匹配成功：修正 new_string 缩进（收养文件真实缩进，防生成坏代码）
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
        # 记忆树：旧卡 vs 新卡卡级 diff（与 write 同一编辑路径，§4.6 精确失效）
        new_units = self._split_by_file(path, new_content)
        edits = self._replace_by_diff(path, old_units, new_units)
        try:
            with open(p, "w", encoding="utf-8") as f:
                f.write(new_content)
        except Exception as e:
            return f"写入失败：{e}"
        self.receipt = f"edit {Path(path).name}：替换 {n if replace_all else 1} 处，树 {edits} 处卡级变更。"
        return f"已替换 {'全部' if replace_all else ''}{n if replace_all else 1} 处：{p}\n语法检查：{_py_syntax_check(new_content, p)}"

    def tool_write(self, path, content, mode="overwrite"):
        try:
            p = self._abs(path)
            p.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            return f"写入失败：{e}"
        exists = p.exists() and p.is_file()
        # 改前必须读（拍定 2026-08-30）：已存在文件必须有树内旧基线（精确 diff 的锚），
        # 未读过 → 硬拒绝，提示先 read。新文件不受限。
        old_units = None
        if exists:
            old_units = self._file_units(path)
            if old_units is None:
                return f"错误：请先 read {path}（改前必须读：树内没有该文件的旧内容基线）。新文件请直接用 write。"
        if mode == "append":
            try:
                with open(p, "a", encoding="utf-8") as f:
                    f.write(content)
            except Exception as e:
                return f"追加失败：{e}"
            self._ingest_new(path, content)
            return f"成功追加文件：{p}（追加内容已入树）"
        # overwrite：卡级 diff 编辑路径（§4.6：旧内容精确失效）
        new_units = self._split_by_file(path, content)
        if old_units:
            edits = self._replace_by_diff(path, old_units, new_units)
        else:
            edits = self._ingest_new(path, content)
        try:
            with open(p, "w", encoding="utf-8") as f:
                f.write(content)
        except Exception as e:
            return f"写入失败：{e}"
        self.receipt = f"write {Path(path).name}：{edits} 处卡级变更（旧内容精确失效）。"
        return f"成功写入文件：{p}（记忆树 {edits} 处变更）\n语法检查：{_py_syntax_check(content, p)}"

    def _split_by_file(self, path, content):
        p = self._abs(path)
        media, lang, fmt = _detect_media(path)
        return split_media(content, media, source=f"file:{p}", lang=lang, fmt=fmt)

    def _file_units(self, path):
        """树中该文件的存活卡（按位置序）。"""
        p = self._abs(path)
        src = f"file:{p}"
        ids = [aid for aid, a in self.eng.atoms.items()
               if a.alive and (a.metadata or {}).get("source") == src]
        if not ids:
            return None
        chain = self.eng.chain_ids()
        ids.sort(key=lambda x: chain.index(x))
        return [(self.eng.atoms[aid].value, aid) for aid in ids]

    def _replace_by_diff(self, path, old_units, new_units):
        """卡级 diff（SequenceMatcher）→ 替换/插入，走编辑路径。零 LLM。"""
        import difflib
        old_texts = [v for v, _ in old_units]
        old_ids = [aid for _, aid in old_units]
        new_texts = [t for t, _ in new_units]
        sm = difflib.SequenceMatcher(None, old_texts, new_texts)
        edits = 0
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                continue
            dead = old_ids[i1:i2]
            units = new_units[j1:j2]
            if dead:
                self.eng.replace_units(dead, units)
                edits += 1
            elif units:
                anchor = old_ids[i1 - 1] if i1 > 0 else None
                self.eng.insert_units(units, left_id=anchor,
                                      right_id=old_ids[i1] if i1 < len(old_ids) else None)
                edits += 1
        return edits

    def _ingest_new(self, path, content):
        units = self._split_by_file(path, content)
        if not units:
            return 0
        self.eng.insert_units(units, left_id=self.eng.tail)
        return len(units)

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
        # 结果写回记忆树（source=tool，按句切分）——调试经验自然进记忆
        return text

    def _log_tool(self, name, args, result):
        """工具轨迹精简日志（轨道C 雏形）：单张低 HP 原子，只记录"干过啥"。
        - read 的文件已全量入树（source=file:path），这里不再存结果副本；
        - 其他工具结果不整段入树（实时可见在对话 tool 消息里）；
        - 初始 HP=TOOL_LOG_HP（拍定 50）：窗口让位文件卡、自然遗忘（~5 轮出窗），
          recall 可回血。"""
        r = str(result)
        first = r.splitlines()[0][:80] if r.splitlines() else ""
        if name == "read":
            line = f"read {Path(str(args.get('path', ''))).name} → 完成"
        elif name == "edit":
            line = f"edit {Path(str(args.get('path', ''))).name} → {first}"
        elif name == "write":
            line = f"write {Path(str(args.get('path', ''))).name} → {first}"
        elif name == "command":
            line = f"command {str(args.get('command', ''))[:80]} → {first}"
        else:
            line = f"{name} {json.dumps(args, ensure_ascii=False)[:80]} → {first or r[:80]}"
        a = self.eng._new_atom(line, weight=1, metadata={"source": "tool"})
        a.hp = self.tool_log_hp
        self.eng.insert(self.eng.tail, None, atoms=[a])

    def tool_save_context(self, content, topic=None):
        content = (content or "").strip()
        if not content:
            return "save_context 失败：content 不能为空"
        key = (topic or "").strip() or content[:10]
        self.eng.nb_set(key, "state", content)
        return f"已保存事实「{key}」"

    def tool_recall(self, query):
        q = (query or "").strip()
        if len(q) < 2:
            return "recall 查询过短"
        hits = self.eng.retrieve(q, budget=self.retr_budget)
        expanded = self.eng.expand_leaves(hits)
        self.last_shadows = list(dict.fromkeys(list(self.last_shadows or []) + expanded))
        brief = []
        for h in hits:
            if self.eng._is_atom(h):
                brief.append(f"#{h}·原文 {self.eng.atoms[h].value[:60]}")
            else:
                brief.append(f"N{h}·摘要 {str(self.eng.nodes[h].value)[:80]}")
        return "命中 {} 项（{} 原子原文下轮注入影子）：\n{}".format(
            len(hits), len(expanded), "\n".join(brief) if brief else "（无命中）")

    def tool_finish(self, summary):
        self.done = True
        self.eng.nb_set("current", "worklog", (summary or "").strip())
        return "任务已结束。"

    # ── 主循环（v1 参考结构：原生工具对话 + 记忆流槽位每轮刷新）────

    def _dispatch(self, name, args):
        fn = {
            "read": self.tool_read,
            "write": self.tool_write,
            "edit": self.tool_edit,
            "command": self.tool_command,
            "save_context": self.tool_save_context,
            "recall": self.tool_recall,
            "finish": self.tool_finish,
        }.get(name)
        if fn is None:
            return f"未知工具：{name}"
        try:
            return fn(**args)
        except TypeError as e:
            try:
                return fn()
            except Exception:
                return f"工具 {name} 参数错误：{e}"

    MAX_CONV_CHARS = 48000   # 对话消息总字符上限（槽位 16k + 对话 48k ≈ 32k token，64k ctx 内）

    def _trim_messages(self):
        """超限裁剪：从对话里删最早的一整轮（assistant tool_calls + 其 tool 结果），
        保留 system/槽位/task。整轮成对删，绝不拆散 tool_calls 与结果。"""
        total = sum(len(m.get("content") or "") for m in self.messages)
        while total > self.MAX_CONV_CHARS and len(self.messages) > 3:
            i = 3
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
                # 无 tool_calls 可删：删最早的非固定消息（已确保 len > 3）
                removed = len(self.messages[3].get("content") or "") + 1
                del self.messages[3]
            total -= removed

    def _slot_text(self, task_desc):
        """记忆流槽位（每轮刷新）：组装流 + 笔记本 + 回执 + 进度。"""
        eng = self.eng
        if self.goal:
            hits = eng.retrieve(self.goal, budget=4)
            expanded = eng.expand_leaves(hits)
            self.last_shadows = list(dict.fromkeys(list(self.last_shadows or []) + expanded))
        # 槽位预算 = slot_budget（独立于 assembly_budget；防槽位过大挤爆对话历史）
        emitted = eng.assemble(self.last_shadows, budget=self.slot_budget)
        # 文件原子折叠为索引行（拍定 2026-08-30）：正文只经 read 提供（View B 单一真源）。
        # 召回命中的文件原子展开逐字（召回 = 把原文拉回视野；tiling 中已存在故影子注入跳过）
        unfold = {aid for aid in (self.last_shadows or [])
                  if eng._is_atom(aid) and (eng.atoms[aid].metadata or {}).get("source", "").startswith("file:")}
        flow = eng.render_emission(emitted, fold_files=True, unfold_ids=unfold)
        nb_rows = eng.nb_render(self.goal or "")
        parts = [flow]
        if nb_rows:
            nb = "\n".join(f"[{k}·{key}] {c}" for k, key, c in nb_rows)
            parts.append("【笔记本·事实层】（去重发送）\n" + nb)
        if self.receipt:
            parts.append(f"【最近操作回执】\n{self.receipt}")
        parts.append(f"【进度】第 {self.round_no} 轮；已用工具："
                     f"{', '.join(f'{k}×{v}' for k, v in self.tool_uses.items()) or '无'}。"
                     f"定位到根因后做最小改动，改完运行验收脚本（python3 accept_tX.py）确认，"
                     f"通过后调用 finish。")
        return "\n\n".join(parts)

    def run_task(self, task_desc, task_id="T?", max_rounds=40, max_seconds=1800,
                 progress=None):
        """跑一个任务。progress: callable(round_no, msg) 供 harness 打日志。"""
        self.goal = (task_desc.split("】", 1)[0].replace("【", "").replace("任务", "")
                     if "】" in task_desc else task_id)
        # v1 参考结构：system + 记忆流槽位（每轮刷新）+ task + 原生工具对话
        slot = {"role": "system", "content": "【记忆流】（每轮刷新）\n（初始化中）"}
        self.messages = [
            {"role": "system", "content": SYSTEM},
            slot,
            {"role": "user", "content": task_desc},
        ]
        t0 = time.time()
        notes = []
        for rnd in range(1, max_rounds + 1):
            if time.time() - t0 > max_seconds:
                notes.append("超时")
                break
            self.round_no = rnd
            slot["content"] = self._slot_text(task_desc)
            self._trim_messages()
            if self.verbose:
                print(f"\n---- [R{rnd}] 对话 {sum(len(m.get('content') or '') for m in self.messages)} 字符 ----"
                      f"\n槽位 {len(slot['content'])} 字符", flush=True)
            if progress:
                progress(rnd, f"LLM 调用（对话 {sum(len(m.get('content') or '') for m in self.messages)} 字符）")
            try:
                text, calls, usage, lat = self.llm.chat_messages(self.messages, TOOLS)
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
            self.messages.append(self.llm.build_assistant_msg(text, calls))
            if not calls:
                if text.strip():
                    self.receipt = "模型说明已写回记忆。"
                if getattr(self, "_idle_rounds", 0) >= 1:
                    notes.append("连续两轮无工具调用")
                    break
                self._idle_rounds = getattr(self, "_idle_rounds", 0) + 1
                continue
            self._idle_rounds = 0
            for tc in calls:
                name, args = tc.name, tc.arguments
                self.tool_uses[name] = self.tool_uses.get(name, 0) + 1
                # 循环断路器（拍定 2026-08-30）：连续 3 次相同工具调用 → 系统警告。
                # 典型场景：模型用过期锚点 edit 自己已改过的文件（T4 实测 11 连败）。
                sig = f"{name}|{json.dumps(args, ensure_ascii=False, sort_keys=True)}"
                if sig == getattr(self, "_last_sig", None):
                    self._repeat = getattr(self, "_repeat", 0) + 1
                else:
                    self._repeat = 0
                self._last_sig = sig
                if self._repeat == 3:
                    hint = ("⚠ 系统警告：你已连续 3 次用相同参数调用同一工具且未成功。"
                            "很可能你的锚点已过期（文件被你自己或历史操作改过）。"
                            "请先 read 目标文件相关区域（用 offset/limit 定位），"
                            "基于**当前文件真实文本**构造新锚点，或改用 write 重写。"
                            "不要重复相同的失败调用。")
                    self.messages.append({"role": "system", "content": hint})
                    if progress:
                        progress(rnd, f"⚠ 循环断路器触发（{name} 连续 3 次相同调用）")
                if name == "finish":
                    result = self.tool_finish(args.get("summary", ""))
                else:
                    result = self._dispatch(name, args)
                rtext = str(result).replace("\n", " ")[:120]
                self.trace.append((rnd, name, dict(args), str(result)[:400]))
                self._log_tool(name, args, result)
                self.messages.append(self.llm.build_tool_msg(tc, result))
                if progress:
                    progress(rnd, f"🔧 {name}({json.dumps(args, ensure_ascii=False)[:80]}) → {rtext[:80]}")
                if name == "finish":
                    return {"ok": True, "rounds": rnd, "summary": args.get("summary", ""),
                            "notes": notes}
            # 每工具调用后：衰减 + 条件式 merge（2026-08-30 拍定机制）
            self.eng.tick(1)
            if self.eng.merge_pressure():
                self.eng.merge_pass()
                if progress:
                    progress(rnd, f"merge_pass（蒸馏 {self.eng.distill['calls'] - self.distill_calls} 次）")
                self.distill_calls = self.eng.distill["calls"]
        return {"ok": False, "rounds": self.round_no,
                "summary": "未完成", "notes": notes}

    def metrics(self):
        m = self.eng.stat()
        return {
            "rounds": self.round_no,
            "atoms_alive": m["atoms"]["alive"],
            "nodes_alive": m["nodes"]["alive"],
            "tiling": m["tiling_size"],
            "distill_calls": self.eng.distill["calls"],
            "distill_seconds": round(self.eng.distill["seconds"], 1),
            "tool_uses": self.tool_uses,
            "notebook": len(self.eng.notebook),
            "total_tokens": self.total_tokens,
            "llm_seconds": round(self.llm_seconds, 1),
        }
