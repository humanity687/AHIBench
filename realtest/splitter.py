"""splitter.py — 中文句级切分（确定性，不用 LLM）。"""

import re

_SENT_RE = re.compile(r"[^。！？!?…；;]+[。！？!?…；;]?")


def split_sentences(text):
    out = []
    for m in _SENT_RE.finditer(text or ""):
        s = m.group(0).strip()
        if len(s) >= 2:
            out.append(s)
    return out
