"""test_chattext.py — 聊天介质切分器测试（chat/code/config/md + 原子上限 + 来源）。"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "realtest"))

from chattext import (split_chat, split_code, split_config, split_md, split_media)


def test_chat_splits_sentences_with_source():
    out = split_chat("你好，我是小明。我喜欢写代码。", source="user:alice")
    assert [t for t, _ in out] == ["你好，我是小明。", "我喜欢写代码。"]
    assert all(m["source"] == "user:alice" for _, m in out)


def test_code_function_and_class_doc():
    code = '''import os

def greet(name, lang="zh"):
    """问候函数：按语言返回问候语。"""
    if lang == "zh":
        return f"你好，{name}"
    return f"Hello, {name}"

class Server:
    """服务器类：管理连接。"""
    def __init__(self, port=8080):
        self.port = port
    def start(self):
        """启动服务器。"""
        return f"listen :{self.port}"
    def __del__(self):
        self.port = None

PORT = 8080
'''
    out = split_code(code, source="file:server.py")
    metas = [m for _, m in out]
    docs = [m.get("doc", "") for m in metas]
    # 函数卡带 doc；类卡带 doc（类头+__init__/__del__ 一张）
    assert any("问候函数" in d for d in docs)
    assert any("服务器类" in d for d in docs)
    # 超限类拆分：__init__/__del__ 并入类卡，start 独立（doc 为"启动服务器"）
    texts = "".join(t for t, _ in out)
    assert "class Server:" in texts
    assert "def start(self)" in texts


def test_code_cap_rechunk():
    big = "def f():\n    return 1\n" * 400  # 远超 ATOM_CHAR_CAP
    out = split_code(big)
    assert len(out) >= 2
    assert all(len(t) <= 2100 for t, _ in out)


def test_config_one_item_per_atom():
    out = split_config('{"port": 8080, "name": "test"}', fmt="json")
    keys = [m["config_key"] for _, m in out]
    assert keys == ["port", "name"]
    assert len(out) == 2


def test_config_ini_sections():
    out = split_config("[server]\nport=8080\nhost=0.0.0.0", fmt="ini")
    keys = [m["config_key"] for _, m in out]
    assert keys == ["server::port", "server::host"]


def test_md_heading_and_list_boundaries():
    md = "# 标题一\n这是内容。\n\n* 项目A：做啥\n* 项目B：做啥\n\n## 标题二\n第二段内容。"
    out = [t for t, _ in split_md(md)]
    assert "# 标题一" in out
    assert "## 标题二" in out
    assert "* 项目A：做啥" in out
    assert "* 项目B：做啥" in out
    assert "这是内容。" in out


def test_split_media_dispatch():
    out = split_media("你好啊。再见。", "chat", source="user:u")
    assert len(out) == 2
    out = split_media("port=8080", "config", fmt="ini")
    assert len(out) == 1
