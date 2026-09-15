"""protocol.py — 续写响应解析（三级容错）：[append]/[edit]/[recall]/[plan]/[span]/[note]/[dismiss] 块。"""

import re

APPEND_RE = re.compile(r"\[append\](.*?)\[/append\]", re.S)
EDIT_RE = re.compile(r"\[edit\s*([^\]]*)\](.*?)\[/edit\]", re.S)
RECALL_RE = re.compile(r"\[recall\](.*?)\[/recall\]", re.S)
PLAN_RE = re.compile(r"\[plan\](.*?)\[/plan\]", re.S)
SPAN_RE = re.compile(r"\[span\s*([^\]]*)\](.*?)\[/span\]", re.S)
NOTE_RE = re.compile(r"\[note\s*([^\]]*)\](.*?)\[/note\]", re.S)
DISMISS_RE = re.compile(r"\[dismiss\s*([^\]]*)\](.*?)\[/dismiss\]", re.S)
IDS_RE = re.compile(r"#?(\d+)")


def parse_ids(spec):
    ids = [int(m) for m in IDS_RE.findall(spec or "")]
    if not ids:
        return None
    return (min(ids), max(ids)) if len(ids) >= 2 else (ids[0], ids[0])


def parse_response(text):
    """返回 [(kind, payload), ...]；kind ∈ append / edit / recall / plan / span / note / dismiss / text。

    规则：工具块优先；块外的非空文本 → text（正文追加）。
    三级容错：标准闭合块 → 未闭合开标签截断（[plan] 截到双换行，其余截到行尾）→ 全部视为正文。
    """
    text = (text or "").strip()
    if not text:
        return []
    blocks = []
    for m in APPEND_RE.finditer(text):
        blocks.append(("append", m.group(1).strip()))
    for m in EDIT_RE.finditer(text):
        blocks.append(("edit", (m.group(1).strip(), m.group(2).strip())))
    for m in RECALL_RE.finditer(text):
        blocks.append(("recall", m.group(1).strip()))
    for m in PLAN_RE.finditer(text):
        blocks.append(("plan", m.group(1).strip()))
    for m in SPAN_RE.finditer(text):
        blocks.append(("span", (m.group(1).strip(), m.group(2).strip())))
    for m in NOTE_RE.finditer(text):
        blocks.append(("note", (m.group(1).strip(), m.group(2).strip())))
    for m in DISMISS_RE.finditer(text):
        blocks.append(("dismiss", (m.group(1).strip(), m.group(2).strip())))
    rest = APPEND_RE.sub("", text)
    rest = EDIT_RE.sub("", rest)
    rest = RECALL_RE.sub("", rest)
    rest = PLAN_RE.sub("", rest)
    rest = SPAN_RE.sub("", rest)
    rest = NOTE_RE.sub("", rest)
    rest = DISMISS_RE.sub("", rest)
    rest = rest.strip()
    if rest:
        # 容错 2：未闭合的开标签 → 截断解析（开头或句中，避免其内容被当正文）
        m = re.match(r"\[plan\]\s*(.+?)(?=\n\s*\n|$)", rest, re.S)
        if m:
            blocks.append(("plan", m.group(1).strip()))
            rest = re.sub(r"\[plan\]\s*.+?(?=\n\s*\n|$)", "", rest, count=1, flags=re.S).strip()
        changed = True
        while changed:
            changed = False
            for tag, kind in (("span", "span"), ("recall", "recall"), ("note", "note"),
                              ("dismiss", "dismiss")):
                pat = rf"\[{tag}\s*[^\]]*\]\s*([^\[]+?)(?=\n|$)"
                m = re.match(pat, rest, re.S) or re.search(pat, rest, re.S)
                if not m:
                    continue
                spec_m = re.match(rf"\[{tag}\s*([^\]]*)\]", rest[m.start():])
                spec = spec_m.group(1).strip() if spec_m else ""
                payload = m.group(1).strip()
                if kind == "span":
                    blocks.append(("span", (spec, payload)))
                elif kind == "note":
                    blocks.append(("note", (spec, payload)))
                elif kind == "dismiss":
                    blocks.append(("dismiss", (spec, payload)))
                else:
                    blocks.append(("recall", payload))
                rest = (rest[:m.start()] + rest[m.end():]).strip()
                changed = True
                break
        rest = rest.strip()
        if rest:
            blocks.append(("text", rest))
    if not blocks:
        blocks.append(("text", text))
    return blocks


def handle_edit(payload, eng):
    spec, new_text = payload
    rng = parse_ids(spec)
    if rng is None or not new_text.strip():
        return None, "edit 参数不完整（需要 #起止id 与 新文本）"
    lo, hi = rng
    if lo not in eng.atoms or hi not in eng.atoms:
        return None, f"edit 引用了不存在的原子 id（#{lo}..#{hi}）"
    if not eng.atoms[lo].alive or not eng.atoms[hi].alive:
        return None, f"edit 引用了已删除的原子（#{lo}..#{hi}）——请先 recall 查看当前内容"
    chain = eng.chain_ids()
    try:
        i1, i2 = chain.index(lo), chain.index(hi)
    except ValueError:
        return None, "edit 区间中的原子不在链上"
    if i1 > i2:
        i1, i2 = i2, i1
    old_ids = chain[i1:i2 + 1]
    new_ids = eng.replace(old_ids, [new_text.strip()])
    dirty = [nid for nid, nd in eng.nodes.items() if nd.alive and nd.dirty]
    return new_ids, f"已编辑 #{lo}..#{hi}（{len(old_ids)} 句替换为 1 句），覆盖摘要已置脏: {dirty}"
