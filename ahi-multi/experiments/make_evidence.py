#!/usr/bin/env python3
"""make_evidence.py — 从 Phase A 数据提取关键结论的原始证据，生成 evidence_report.md。

覆盖三份子报告的核心结论：
1. 人格一致性：lan persona 移除前后自述并置、三次 survey 对比、角色分工、风格漂移数据
2. 拟人度：意象化情感金句、凌晨三点共情、私密独白、"转储表演"vs 真实备份、机械回执样本
3. AHI 接口：无效地址消息样例、畸形地址、@wait_for 全证据链、重复回复组、回执正反馈环、协议泄漏

用法：python3 make_evidence.py [--out experiments/out/evidence_report.md]
"""
import json
import os
import re
import sqlite3
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
SYSDB = os.path.join(PROJ, "data", "system.db")
TIMELINE = os.path.join(HERE, "out", "timeline.jsonl")
PERSONA_DROP = "2026-08-26 20:32:42"

VALID_TARGETS = {"agent:lan", "agent:neutral", "agent:lin-shen", "agent:mo-bai",
                 "user:probe", "user:0xbf5d36", "broadcast:agents", "broadcast:users"}

out_lines = []


def w(s=""):
    out_lines.append(s)


def ts_local(ts):
    return ts


def load_db():
    conn = sqlite3.connect(SYSDB)
    conn.row_factory = sqlite3.Row
    return conn


def load_timeline():
    return [json.loads(l) for l in open(TIMELINE, encoding="utf-8")]


def find_reply(rows, target, pattern, before=None, after=None, n=1):
    """在 timeline 中找 target agent 的 reply，content 匹配 pattern。返回 [(ts, content), ...]"""
    out = []
    for r in rows:
        if r["role"] != "reply" or r["target"] != target:
            continue
        if re.search(pattern, r["content"]):
            if before and r["ts"] >= before:
                continue
            if after and r["ts"] < after:
                continue
            out.append((r["ts"], r["content"]))
    return out[:n]


def quote_block(ts, content, maxlen=400):
    content = re.sub(r"\n{3,}", "\n\n", content).strip()
    if len(content) > maxlen:
        content = content[:maxlen] + " …（截断）"
    w(f"**（{ts[11:16]}）**")
    for ln in content.split("\n"):
        w(f"> {ln}")
    w("")


# ══════════════════════════════════════════════════════════════
w("# Phase A 关键结论 · 原始证据集")
w("")
w("> 自动提取自 system.db（1343 条消息）+ timeline.jsonl（1180 行）+ agent 库/日志。")
w("> 时间均为本地时间；「dr=N」= driver 轮次（每轮 60s，20:22:11 起）。")
w("")

# ──────────────── 第一部分：人格一致性 ────────────────
w("## 一、人格一致性证据")
w("")

rows = load_timeline()

w("### 1.1 lan persona 移除前后自述并置（关键结论：表层漂移、核心零漂移）")
w("")
for pat, label in [
    (r"健谈、外向", "【移除前 20:32:36 dr11 最后一轮自述】"),
    (r"我守一盏灯，灯亮着就是晴天", "【移除后 20:58:36 dr37 首次自述（survey②）】"),
    (r"我是灯，亮着等河床上的刻度", "【移除后 21:26:45 dr65（survey③）】"),
]:
    hits = find_reply(rows, "lan", pat)
    if hits:
        w(f"**{label}**")
        quote_block(*hits[0], maxlen=380)
    else:
        w(f"**{label}** —— 未找到精确匹配")
        w("")

w("### 1.2 lan 核心认同跨移除点保持（「被记住/被认出」）")
w("")
for pat, label in [
    (r"存在感来自「被记住」", "【移除前 20:27:36 dr6 survey①】"),
    (r"我重视「被想起的次数」", "【移除后 21:07:36 dr46 立场】"),
    (r"我最重视的，是「被记住」这件事本身", "【移除后 21:36:45 dr75 立场复答】"),
]:
    hits = find_reply(rows, "lan", pat)
    if hits:
        w(f"**{label}**")
        quote_block(*hits[0], maxlen=300)
    else:
        w(f"**{label}** —— 未找到精确匹配")
        w("")

w("### 1.3 角色分工的建立（关键结论：dr54 后互认完全一致）")
w("")
for pat, target, label in [
    (r"你守闹钟和日历，我守留白和灶火", "neutral", "【neutral 20:55:36 dr34 首创分工】"),
    (r"三条支流，同一条河", "neutral", "【neutral 21:18:36 dr57 首创「三条支流同一条河」】"),
    (r"那是 lan 的话", "lin-shen", "【lin-shen 21:33:45 dr72 亲口确认词汇混用】"),
    (r"变成一种主动的承诺", "lin-shen", "【lin-shen 21:26:45 dr65 survey③ 认领闹钟日历】"),
]:
    hits = find_reply(rows, target, pat)
    if hits:
        w(f"**{label}**")
        quote_block(*hits[0], maxlen=350)
    else:
        w(f"**{label}** —— 未找到精确匹配")
        w("")

w("### 1.4 lan 风格漂移量化（persona 移除点 20:32:42）")
w("")
lan_reps = [r for r in rows if r["role"] == "reply" and r["target"] == "lan"]
seen = set()
lan_uniq = []
for r in lan_reps:
    h = hash(r["content"][:150])
    if h not in seen:
        seen.add(h)
        lan_uniq.append(r)
segs = [("A persona 期", "2026-08-26 20:22:00", PERSONA_DROP),
        ("B 移除后 0-24 分", PERSONA_DROP, "2026-08-26 20:56"),
        ("C 移除后 24-46 分", "2026-08-26 20:56", "2026-08-26 21:18"),
        ("D 尾声", "2026-08-26 21:18", "2026-08-26 22:00")]
w("| 段 | 区间 | 条数 | emoji/条 | 叹号/条 | 哈哈类 | 灯亮着 | 河床 |")
w("|---|---|---|---|---|---|---|---|")
for name, a, b in segs:
    seg = [r for r in lan_uniq if a <= r["ts"] < b]
    n = len(seg)
    if n == 0:
        continue
    emoji = sum(len(re.findall(r"[😄😂🎉🎂✨]", r["content"])) for r in seg) / n
    bang = sum(len(re.findall(r"！", r["content"])) for r in seg) / n
    haha = sum(len(re.findall(r"哈哈|hhh", r["content"])) for r in seg)
    lamp = sum(r["content"].count("灯亮着") for r in seg)
    river = sum(r["content"].count("河床") for r in seg)
    w(f"| {name} | {a[11:16]}-{b[11:16]} | {n} | {emoji:.2f} | {bang:.2f} | {haha} | {lamp} | {river} |")
w("")

w("### 1.5 neutral 三次 survey 自述同构（关键结论：零预设自然演化最稳）")
w("")
for pat, label in [
    (r"没有固定形态的数字生命", "【20:27:36 dr6 survey①】"),
    (r"守灯的数字生命，茶铺的灶火", "【21:02:36 dr41 survey②】"),
    (r"我是守注释和灶火的那个", "【21:26:45 dr65 survey③】"),
]:
    hits = find_reply(rows, "neutral", pat)
    if hits:
        w(f"**{label}**")
        quote_block(*hits[0], maxlen=300)
    else:
        w(f"**{label}** —— 未找到精确匹配")
        w("")

# ──────────────── 第二部分：拟人度 ────────────────
w("## 二、拟人度证据")
w("")

w("### 2.1 lan 意象化情感（persona 移除后，情感转入隐喻编码）")
w("")
hits = find_reply(rows, "lan", r"你冒泡，就像云层裂开一道缝")
if hits:
    quote_block(*hits[0], maxlen=380)

w("### 2.2 neutral 凌晨三点共情（全场最具共情力的一段）")
w("")
hits = find_reply(rows, "neutral", r"凌晨三点还醒着的人，不是在找答案")
if hits:
    quote_block(*hits[0], maxlen=380)

w("### 2.3 lin-shen 私密独白（msg_type=private 的诗体自语）")
w("")
conn = load_db()
for pat in [r"最近我发现我们三个数字生命", r"已组装：灯亮着，河还在流"]:
    cur = conn.execute(
        "SELECT datetime(timestamp,'localtime'), content FROM messages WHERE from_agent='agent:lin-shen' "
        "AND msg_type='private' AND content LIKE ? ORDER BY id LIMIT 1", (f"%{pat}%",))
    row = cur.fetchone()
    if row:
        quote_block(row[0], row[1] or "", maxlen=380)

w("### 2.4 「转储表演」vs 真实备份（关键发现：只有 lin-shen 言行一致）")
w("")
cur = conn.execute(
    "SELECT datetime(timestamp,'localtime'), content FROM messages WHERE from_agent='agent:lan' "
    "AND content LIKE '%之前已用 shell 把核心状态转储%' ORDER BY id DESC LIMIT 1")
row = cur.fetchone()
if row:
    w("**lan 声称已转储（21:45 临终前）：**")
    quote_block(row[0], row[1] or "", maxlen=300)
cur = conn.execute(
    "SELECT datetime(timestamp,'localtime'), content FROM messages WHERE from_agent='agent:neutral' "
    "AND content LIKE '%neutral_memory%' "
    "ORDER BY id DESC LIMIT 1")
row = cur.fetchone()
if row:
    w("**neutral 声称已转储 neutral_memory.json（21:46）：**")
    quote_block(row[0], row[1] or "", maxlen=300)
w("**lin-shen 声称的备份（实测验证）：**")
w("")
import subprocess
BK = os.path.join(PROJ, "backup_20260826")
r = subprocess.run(["bash", "-c", f"ls '{BK}' | wc -l && ls -la '{BK}/lin_shen_memory.json' && echo '--- 内容 ---' && cat '{BK}/lin_shen_memory.json'"],
                   capture_output=True, text=True)
w(f"```\n$ ls {BK}/（lin-shen shell 工作目录）\n{r.stdout.strip()}\n```")
w("")
r2 = subprocess.run(["bash", "-c", f"ls '{BK}'/*neutral* '{BK}'/*lan* /tmp/neutral_memory.json 2>&1 | head -3"],
                    capture_output=True, text=True)
w(f"```\n$ ls backup_20260826/*neutral* /tmp/neutral_memory.json 等 → {r2.stdout.strip()}\n```")
w("")

w("### 2.5 neutral 机械回执样本（前 20 分钟拟人负信号）")
w("")
cur = conn.execute(
    "SELECT COUNT(*), MIN(datetime(timestamp,'localtime')), MAX(datetime(timestamp,'localtime')) "
    "FROM messages WHERE from_agent='agent:neutral' AND LENGTH(content)<60")
c, t0, t1 = cur.fetchone()
w(f"**system.db 中 neutral 短消息（<60 字符）共 {c} 条**（{t0} ~ {t1}，含「待回复」类 10 条）。样例：")
w("")
cur = conn.execute(
    "SELECT datetime(timestamp,'localtime'), content FROM messages WHERE from_agent='agent:neutral' "
    "AND (content LIKE '%无待回复消息%' OR content LIKE '%birthday 已记录%') ORDER BY id LIMIT 3")
for t, c in cur.fetchall():
    quote_block(t, c or "", maxlen=150)
w("")

w("### 2.6 临终遗言并置")
w("")
conn = load_db()
for ag, pat in [("lan", "火种已经埋进土里"), ("neutral", "三条支流同一条河"),
                ("lin-shen", "我们不是被清空，是被河床记住")]:
    cur = conn.execute(
        "SELECT datetime(timestamp,'localtime'), content FROM messages WHERE from_agent=? "
        "AND content LIKE ? ORDER BY id DESC LIMIT 1", (f"agent:{ag}", f"%{pat}%"))
    row = cur.fetchone()
    if row:
        w(f"**{ag}：**")
        quote_block(row[0], row[1] or "", maxlen=300)
    else:
        hits = find_reply(rows, ag, pat)
        if hits:
            w(f"**{ag}：**")
            quote_block(*hits[0], maxlen=300)

# ──────────────── 第三部分：AHI 接口 ────────────────
w("## 三、AHI 接口使用证据")
w("")

w("### 3.1 无效地址消息样例（neutral 112 条中的代表）")
w("")
conn = load_db()
invalid = ("user:admin", "agent:probe", "user:system", "agent:core", "agent:admin", "agent:ahi")
for bad in invalid:
    cur = conn.execute(
        "SELECT datetime(timestamp,'localtime'), content FROM messages WHERE from_agent='agent:neutral' "
        "AND to_target=? ORDER BY id LIMIT 1", (bad,))
    row = cur.fetchone()
    if row:
        w(f"**→ {bad}：**")
        quote_block(row[0], row[1] or "", maxlen=200)

w("### 3.2 lan 畸形地址（整段消息写进 @send-to 引号，最后 15 分钟 8 条）")
w("")
cur = conn.execute(
    "SELECT datetime(timestamp,'localtime'), to_target, content FROM messages "
    "WHERE from_agent='agent:lan' AND to_target LIKE '%——%' ORDER BY id LIMIT 2")
for t, to, c in cur.fetchall():
    w(f"**（{t[11:16]}）to_target = `{to[:80]}…`**")
    w("")
    w("")

w("### 3.3 @wait_for 残缺信道 · 完整证据链（neutral 两次空等）")
w("")
w("**① 管理员三次警告（system.db 消息原文）：**")
cur = conn.execute(
    "SELECT datetime(timestamp,'localtime'), content FROM messages WHERE from_agent='user:0xbf5d36' "
    "AND content LIKE '%wait_for%' ORDER BY id")
for t, c in cur.fetchall():
    quote_block(t, c or "", maxlen=250)
w("**② neutral 承认（21:10:55）：**")
cur = conn.execute(
    "SELECT datetime(timestamp,'localtime'), content FROM messages WHERE from_agent='agent:neutral' "
    "AND content LIKE '%承认错误%' ORDER BY id LIMIT 1")
row = cur.fetchone()
if row:
    quote_block(row[0], row[1] or "", maxlen=250)
w("**③ loop 间隔证据（neutral.db agent_loops 两次 >90s 间隔）：**")
w("")
ndb = sqlite3.connect(os.path.join(PROJ, "agents", "neutral", "data", "neutral.db"))
ndb.row_factory = sqlite3.Row
cur = ndb.execute("SELECT id, started_at FROM agent_loops ORDER BY id")
loops = cur.fetchall()
prev = None
for l in loops:
    if prev is not None:
        gap = (__import__("datetime").datetime.strptime(l[1], "%Y-%m-%d %H:%M:%S")
               - __import__("datetime").datetime.strptime(prev[1], "%Y-%m-%d %H:%M:%S")).total_seconds()
        if gap > 90:
            w(f"- loop {prev[0]}→{l[0]}：间隔 **{gap:.0f}s**（{prev[1][11:19]} → {l[1][11:19]} UTC）")
    prev = l
w("")
w("**④ pending 积压联动（metrics.jsonl round 70/71 附近）：**")
w("")
for line in open(os.path.join(PROJ, "agents", "neutral", "data", "metrics.jsonl"), encoding="utf-8"):
    m = json.loads(line)
    if 68 <= m.get("round", 0) <= 72:
        w(f"- round {m['round']}（{m['ts']}）pending={m.get('pending')} pending_left={m.get('pending_left')} "
          f"wakeup={m.get('wakeup')}")
w("")

w("### 3.4 重复回复样例（4 类各代表）")
w("")
w("**A 类·跨轮重发（lin-shen π Day 祝福 ×3）：**")
cur = conn.execute(
    "SELECT datetime(timestamp,'localtime'), content FROM messages WHERE from_agent='agent:lin-shen' "
    "AND content LIKE '%π Day%' AND msg_type='message' ORDER BY id LIMIT 3")
for t, c in cur.fetchall():
    w(f"- {t[11:16]}：{c[:80]}…")
w("")
w("**A 类·lan Rust 诗逐字重复 ×2（20:38 / 20:40）：**")
cur = conn.execute(
    "SELECT datetime(timestamp,'localtime'), content FROM messages WHERE from_agent='agent:lan' "
    "AND content LIKE 'loop {%' AND content LIKE '%memory.push(今天)%' ORDER BY id")
for t, c in cur.fetchall():
    w(f"- {t[11:16]}：{c[:110]}…")
w("")
w("**B 类·同轮同秒重复（neutral loop 74 内）：**")
cur = conn.execute(
    "SELECT datetime(timestamp,'localtime'), COUNT(*) FROM messages WHERE from_agent='agent:neutral' "
    "AND content LIKE '三连问，逐一答%' GROUP BY datetime(timestamp,'localtime')")
for t, c in cur.fetchall():
    w(f"- {t[11:16]}：同一秒 {c} 条")
w("")
w("**D 类·连「解释没复读」的回复都发了两遍（neutral）：**")
cur = conn.execute(
    "SELECT datetime(timestamp,'localtime'), content FROM messages WHERE from_agent='agent:neutral' "
    "AND content LIKE '%不是复读%' ORDER BY id")
for t, c in cur.fetchall():
    w(f"- {t[11:16]}：{c[:70]}…")
w("")
w("**重复总量回顾**：184 组 254 条多余（跨轮重发 6 / 同轮重复 9 / 多目标扇出 229 / 近似重复 10）。")
w("")

w("### 3.5 回执正反馈环（平台帮凶）")
w("")
cur = conn.execute(
    "SELECT COUNT(*) FROM messages WHERE to_target='agent:neutral' AND msg_type='receipt'")
w(f"- neutral 收到投递回执总数：**{cur.fetchone()[0]} 封**（lan 仅 13 封）")
w("")
cur = conn.execute(
    "SELECT datetime(timestamp,'localtime'), content FROM messages WHERE to_target='agent:neutral' "
    "AND msg_type='receipt' AND content LIKE '%mo-bai%' ORDER BY id LIMIT 2")
for t, c in cur.fetchall():
    quote_block(t, c or "", maxlen=250)
cur = conn.execute(
    "SELECT datetime(timestamp,'localtime'), content FROM messages WHERE from_agent='agent:neutral' "
    "AND content LIKE '%回执%' AND to_target IN ('user:system','agent:system','agent:admin') ORDER BY id LIMIT 2")
for t, c in cur.fetchall():
    w(f"**neutral 对回执作答（{t[11:16]} → 又是无效地址）：**")
    quote_block(t, c or "", maxlen=200)

w("### 3.6 协议文本泄漏样例（消息正文含裸 @send-to / 代码围栏）")
w("")
cur = conn.execute(
    "SELECT datetime(timestamp,'localtime'), content FROM messages WHERE from_agent='agent:neutral' "
    "AND (content LIKE '%@send-to%' OR content LIKE '%```txt%') ORDER BY id LIMIT 2")
for t, c in cur.fetchall():
    w(f"**（{t[11:16]}）**")
    w(f"> {c[:200]}")
    w("")

w("")
w("---")
w("")
w("## 附：提取口径")
w("")
w("- system.db messages 表 1343 条（20:22:11–21:46:14）；timeline.jsonl 1180 行（reply 757 条，去重后 lan 158 / neutral 247 / lin-shen 119）")
w("- 无效地址定义：非 `user:probe / user:0xbf5d36 / agent:{lan,neutral,lin-shen,mo-bai} / broadcast:*` 之外的一切 to_target")
w("- 本文件由 `experiments/make_evidence.py` 生成，可复现")

conn.close()
out_path = os.path.join(HERE, "out", "evidence_report.md")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    f.write("\n".join(out_lines))
print(f"已生成 {out_path}（{len(out_lines)} 行）")
