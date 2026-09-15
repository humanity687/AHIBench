"""chattext.py — 聊天场景多介质切分器（确定性，零 LLM）。

按介质把输入切成原子卡单位（text + 元数据）：
- chat：按句切分（复用 splitter）；每句一张
- code：按 AST 切分（Python 用 ast 模块）
    - 函数（含 async）→ 一张；docstring 提取到 meta["doc"]
    - 类 → 整类一张（超限时：类头 + __init__/__del__ 一张，其余方法独立成卡）；
      docstring 提取到 meta["doc"]
    - 其余琐碎语句 → 按行块分组（空行/语句边界断开，受 ATOM_CHAR_CAP 约束）
- config：json/yaml/ini/toml 一个配置项一张（顶层键值对或 [section] 块）
- md：标题层级 + 列表项（*/-/数字.）为原子边界；项内超 ATOM_CHAR_CAP 再按句切

所有介质统一超限再切：单卡文本 > ATOM_CHAR_CAP → 按行块/句再切（防大卡截断蒸馏）。

返回值约定：split_*(text, source, **kw) -> list[(text, meta)]；
meta = {"source": str, "doc": str 可选}。meta 后续由引擎写入 Atom.metadata。
"""

import ast
import json
import re

from core import ATOM_CHAR_CAP
from splitter import split_sentences

# 行块上限（琐碎代码分组用）
CHUNK_LINES = 12
CHUNK_CHARS = 800


def _cap(text, chunk_lines=CHUNK_LINES, chunk_chars=CHUNK_CHARS):
    """把超限文本按行块/句再切，返回文本列表。"""
    lines = text.splitlines()
    if len(text) <= ATOM_CHAR_CAP:
        return [text]
    out, cur, chars = [], [], 0
    for ln in lines:
        if len(cur) >= chunk_lines or chars + len(ln) > chunk_chars:
            out.append("\n".join(cur))
            cur, chars = [], 0
        cur.append(ln)
        chars += len(ln) + 1
    if cur:
        out.append("\n".join(cur))
    return [c for c in out if c.strip()]


def _doc_of(node):
    d = ast.get_docstring(node)
    return (d or "").strip()


def split_chat(text, source="user"):
    """聊天信息按句切分。每句一张原子。"""
    out = []
    for s in split_sentences(text):
        out.append((s, {"source": source}))
    return out


def split_code(code, source="tool", lang=None):
    """代码按 AST 切分。lang=None 时按 .py 处理（当前仅支持 Python AST；
    其他语言退化为行块）。函数/类为原子；类 = 类头 + __init__/__del__ 一张
    （超限类拆开），其余方法独立成卡；未覆盖行聚合为琐碎行块。"""
    if lang not in (None, "py", "python"):
        return [(_cap_t(c), {"source": source}) for c in _cap(code)]
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return [(_cap_t(c), {"source": source}) for c in _cap(code)]

    lines = code.splitlines()
    cards = []  # (start_0idx, end_0idx, text, meta)

    def add_region(start, end, text, meta):
        cards.append((start, end, text, meta))

    def seg_of(node):
        """ast.get_source_segment 从 col_offset 开始切片 → 类内方法丢行首缩进。
        补回首行缩进（后续行自带缩进，不受影响）。"""
        seg = ast.get_source_segment(code, node)
        if seg is None:
            return None
        lo = node.lineno - 1
        if 0 <= lo < len(lines) and node.col_offset > 0:
            indent = lines[lo][:node.col_offset]
            seg = indent + seg
        return seg

    def add_pieces(start, seg, meta):
        """把 seg 切成 ≤ATOM_CHAR_CAP 的块**全部**加入 cards（行号连续推算，不丢行）。
        _cap 按 splitlines 切块、行序完整保留 → 拼接可还原原文（verbatim 树召回依赖）。"""
        pieces = _cap(seg)
        pos = start
        for pc in pieces:
            n = pc.count("\n") + 1
            add_region(pos, pos + n, pc, meta)
            pos += n

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            seg = seg_of(node)
            if seg:
                add_pieces(node.lineno - 1, seg,
                           {"source": source, "doc": _doc_of(node)})
        elif isinstance(node, ast.ClassDef):
            seg = seg_of(node)
            if not seg:
                continue
            if len(seg) <= ATOM_CHAR_CAP:
                add_region(node.lineno - 1, node.end_lineno, seg,
                           {"source": source, "doc": _doc_of(node)})
            else:
                # 超限类：类头→第一个普通方法的真实行切片为一张（含 docstring/空行/
                # 类级赋值与 __init__/__del__，原文缩进天然正确），其余方法独立成卡，
                # 剩余行（方法间空行/类级语句）交给 trivia（不丢行）。
                first_other = None
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                            and child.name not in ("__init__", "__del__"):
                        first_other = child
                        break
                if first_other is not None:
                    core_lines = lines[node.lineno - 1:first_other.lineno - 1]
                else:
                    core_lines = lines[node.lineno - 1:node.end_lineno]
                add_pieces(node.lineno - 1, "\n".join(core_lines),
                           {"source": source, "doc": _doc_of(node)})
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                            and child.name not in ("__init__", "__del__"):
                        cseg = seg_of(child)
                        if cseg:
                            add_pieces(child.lineno - 1, cseg,
                                       {"source": source, "doc": _doc_of(child)})

    # 行扫描：未覆盖行 → 琐碎行块（记录起始行号，按源码位置排序）
    covered = [False] * len(lines)
    for s, e, _, _ in cards:
        for i in range(s, min(e, len(lines))):
            covered[i] = True
    trivia = []
    cur = []
    cur_start = 0
    for i, ln in enumerate(lines):
        if not covered[i]:
            if not cur:
                cur_start = i
            cur.append(ln)
            if len(cur) >= CHUNK_LINES or sum(len(x) for x in cur) >= CHUNK_CHARS:
                trivia.append((cur_start, "\n".join(cur)))
                cur = []
        elif cur:
            trivia.append((cur_start, "\n".join(cur)))
            cur = []
    if cur:
        trivia.append((cur_start, "\n".join(cur)))
    for start, t in trivia:
        # 全保留（含纯空行块）：verbatim 树召回要求拼接 == 原文、行号不漂移。
        # 单个空行 → t=""，"\n".join 拼接时恰好还原为一个空行。
        cards.append((start, start, t, {"source": source}))

    # 按源码位置排序（trivia 用原位置近似：置于开头——用行号重新对齐）
    cards.sort(key=lambda c: (c[0], c[1]))
    return [(text, meta) for _, _, text, meta in cards]


def _cap_t(seg):
    """超限卡再切：返回单段（首段）；超限部分按行块返回多段。"""
    pieces = _cap(seg)
    return pieces[0] if pieces else ""


def _iter_lines_blocks(lines, start, end):
    cur = []
    for ln in lines[start:end]:
        cur.append(ln)
        if len(cur) >= CHUNK_LINES or sum(len(x) for x in cur) >= CHUNK_CHARS:
            yield "\n".join(cur)
            cur = []
    if cur:
        yield "\n".join(cur)


def split_config(content, source="tool", fmt="json"):
    """配置文件一个配置项一张。fmt ∈ json/yaml/ini/toml/auto。"""
    fmt = (fmt or "auto").lower()
    if fmt == "auto":
        fmt = _guess_config_fmt(content)
    out = []
    try:
        if fmt == "json":
            data = json.loads(content)
            for k, v in data.items():
                out.append((f"{k}: {json.dumps(v, ensure_ascii=False)}",
                            {"source": source, "config_key": k}))
        elif fmt == "ini":
            section = None
            for line in content.splitlines():
                s = line.strip()
                if not s or s.startswith(";") or s.startswith("#"):
                    continue
                if s.startswith("[") and s.endswith("]"):
                    section = s[1:-1]
                    continue
                if "=" in s:
                    k, v = s.split("=", 1)
                    key = f"{section}::{k.strip()}" if section else k.strip()
                    out.append((s, {"source": source, "config_key": key}))
        elif fmt == "toml":
            section = None
            for line in content.splitlines():
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                if s.startswith("[") and s.endswith("]"):
                    section = s
                    continue
                if "=" in s and not s.startswith("[["):
                    k = s.split("=", 1)[0].strip()
                    key = f"{section}::{k}" if section else k
                    out.append((s, {"source": source, "config_key": key}))
        elif fmt == "yaml":
            for line in content.splitlines():
                s = line.strip()
                if not s or s.startswith("#") or s.startswith("-"):
                    continue
                if ":" in s and not s.startswith(" "):
                    k = s.split(":", 1)[0].strip()
                    out.append((s, {"source": source, "config_key": k}))
    except Exception:
        pass
    if not out:
        out = [(t, {"source": source}) for t in _cap(content)]
    return out


def _guess_config_fmt(content):
    s = content.lstrip()
    if s.startswith("{"):
        return "json"
    if s.startswith("[") and s.startswith("[", 1):
        return "ini"
    return "yaml"


def split_md(text, source="user"):
    """Markdown：标题层级 + 列表项为原子边界；项内超限再按句切。保留格式标记。"""
    lines = text.splitlines()
    out = []
    buf = []
    in_list = False

    def flush():
        nonlocal buf
        if not buf:
            return
        raw = "\n".join(buf)
        joined = raw.strip()
        if not joined:
            # 纯空行块：保留原文（verbatim 树召回 + 行号不漂移）
            out.append((raw, {"source": source}))
            buf = []
            return
        if len(joined) > ATOM_CHAR_CAP:
            for s in split_sentences(joined):
                out.append((s, {"source": source}))
        else:
            out.append((joined, {"source": source}))
        buf = []

    for line in lines:
        s = line.strip()
        is_head = re.match(r"^#{1,6}\s", s) or re.match(r"^=+\s*$", s) or re.match(r"^-+\s*$", s)
        is_item = re.match(r"^[-*+]\s", s) or re.match(r"^\d+[.)]\s", s)
        if is_head:
            flush()
            in_list = False
            out.append((s, {"source": source}))
        elif is_item:
            if buf:
                flush()
            in_list = True
            buf.append(s)
        else:
            if in_list and s:
                flush()
                in_list = False
            buf.append(s)
    flush()
    if not out:
        out = [(t, {"source": source}) for t in split_sentences(text)]
    return out


def split_media(text, media, source="tool", lang=None, fmt=None):
    """统一入口：media ∈ chat / code / config / md。"""
    if media == "chat":
        return split_chat(text, source)
    if media == "code":
        return split_code(text, source, lang)
    if media == "config":
        return split_config(text, source, fmt)
    if media == "md":
        return split_md(text, source)
    return split_chat(text, source)
