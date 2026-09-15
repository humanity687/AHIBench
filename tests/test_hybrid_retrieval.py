"""测试 J：混合检索（向量+关键词）+ 原子影子注入。"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "realtest"))
from realengine import RealEngine


class FakeLLM:
    def __init__(self):
        self.model = "fake"

    def chat(self, a, b):
        return "摘要。", {"prompt_tokens": 1, "completion_tokens": 1}, 0.01


class FakeEnc:
    def encode(self, text, normalize_embeddings=True):
        t = text if isinstance(text, str) else " ".join(text)
        v = np.zeros(64)
        for ch in t[:64]:
            v[ord(ch) % 64] += 1
        n = np.linalg.norm(v)
        return v / n if n else v


def make_engine():
    eng = RealEngine(FakeLLM(), encoder=FakeEnc(), assembly_budget=10000,
                     attention_in_decay=1.0, attention_out_decay=1.0)
    eng.write(["小明穿了一件红色棉袄，然后出了门。",
               "小红在院子里喂鸡。",
               "小明出门发现有点冷，回来找了一件红色棉袄穿。"])
    eng.tick(15)
    eng.merge_pass()
    return eng


def test_hybrid_retrieval_finds_both_style_sentences():
    eng = make_engine()
    hits = eng.retrieve("小明穿衣服出门", budget=6)
    atom_hits = [h for h in hits if eng._is_atom(h)]
    texts = {eng.atoms[h].value for h in atom_hits}
    assert "小明穿了一件红色棉袄，然后出了门。" in texts
    assert "小明出门发现有点冷，回来找了一件红色棉袄穿。" in texts


def test_retrieval_mixed_node_and_atom():
    eng = make_engine()
    hits = eng.retrieve("小明穿衣服出门", budget=6)
    assert any(eng._is_node(h) for h in hits)
    assert any(eng._is_atom(h) for h in hits)


def test_atom_shadow_injected_and_rendered():
    eng = make_engine()
    hits = eng.retrieve("小明穿衣服出门", budget=6)
    out = eng.assemble(shadows=hits, budget=10000)
    shadows = [(i, v) for k, i, v in out if k == "shadow"]
    assert shadows
    assert all(eng._is_atom(i) for i, _ in shadows)
    text = eng.render_emission(out)
    assert "召回原文" in text
    assert "⊂N" in text


def test_atom_shadow_dedup_when_emitted():
    eng = make_engine()
    eng.tick(15)
    eng.merge_pass()
    eng.write(["小明穿着棉袄回到了家。"])
    hits = eng.retrieve("小明棉袄回家", budget=6)
    out = eng.assemble(shadows=hits, budget=10000)
    for k, i, v in out:
        if k == "shadow":
            assert eng.atoms[i].owner is not None, "已发射的原子不应再入影子"


def test_retrieval_skips_dead_atoms():
    eng = make_engine()
    eng.delete([1])
    hits = eng.retrieve("小明穿衣服出门", budget=6)
    assert all(eng.atoms[h].alive for h in hits if eng._is_atom(h))


def test_atom_hp_boost_on_retrieve():
    eng = make_engine()
    before = eng.atoms[1].hp
    eng.retrieve("小明穿衣服出门", budget=6)
    assert eng.atoms[1].hp > before


def test_no_encoder_falls_back_keyword_only():
    eng = RealEngine(FakeLLM(), encoder=None, assembly_budget=10000,
                     attention_in_decay=1.0, attention_out_decay=1.0)
    eng.write(["小明穿了一件红色棉袄，然后出了门。", "小红在院子里喂鸡。"])
    eng.tick(15)
    eng.merge_pass()
    hits = eng.retrieve("小明红色棉袄", budget=6)
    assert any(eng._is_atom(h) for h in hits)
    assert any(eng._is_node(h) for h in hits)
