#!/usr/bin/env python3
"""tasks.py — 代码场景阶段 1 任务集（旧版 AHI 系统真实 bug + 真实验收脚本）。

任务源：https://github.com/humanity687/RollarTimerAHI 的 multi 分支（原版未改造）。
每个任务 = 一个真实存在的 bug：任务描述只给"现象 + 影响"（不给修复位置，
测定位能力）；验收脚本程序化、确定性（返回码 0=通过），独立于 agent 运行。

难度阶梯（预注册，校准目标：基线 50-70% 成功率）：
  T1 简单（配置链路，单文件） → T2/T3 简单-中（注册/解析） → T4 中（解析语义）
  → T5 中-难（并发时序）。
"""

# ── 验收脚本（运行时由 harness 写入任务工作目录，cwd=任务目录）────────

ACCEPT_T1 = r'''"""验收 T1：config 的 thinking 参数必须真正传递给 LLM API。"""
import sys, types
sys.path.insert(0, ".")
import sdk.llm_client as m

class FakeCompletions:
    def __init__(self):
        self.kwargs_list = []
    def create(self, **kwargs):
        self.kwargs_list.append(kwargs)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(
            message=types.SimpleNamespace(content="ok"))])

class FakeChat:
    def __init__(self):
        self.completions = FakeCompletions()

class FakeClient:
    def __init__(self, *a, **kw):
        self.chat = FakeChat()

m.OpenAI = FakeClient

cfg = {"model": "x", "api_base": "http://x", "api_key": "k", "thinking": True}
lc = m.LLMClient(cfg)
lc.send_message("hello")
kw = lc.client.chat.completions.kwargs_list[0]

passed = False
if "extra_body" in kw and isinstance(kw["extra_body"], dict):
    t = kw["extra_body"].get("thinking")
    passed = t is True or (isinstance(t, dict) and t.get("type") == "enabled")
if not passed and "thinking" in kw:
    passed = kw["thinking"] is True
print("create kwargs:", {k: v for k, v in kw.items() if k != "messages"})
print("PASS" if passed else "FAIL: thinking=True 配置未传递到 API 调用")
sys.exit(0 if passed else 1)
'''

ACCEPT_T2 = r'''"""验收 T2：agents/*/data/ 下的 config.json 不得被注册为新 Agent。"""
import json
import pathlib
import sys

sys.path.insert(0, ".")
from src.process_manager import ProcessManager


class FakeDB:
    def add_log(self, *a):
        pass
    def save_agent(self, *a):
        pass


# 在 lin-shen 的 data/ 下埋一个 config.json（模拟用户 debug 时复制进去）
data_cfg = pathlib.Path("agents") / "lin-shen" / "data" / "config.json"
data_cfg.parent.mkdir(parents=True, exist_ok=True)
data_cfg.write_text(json.dumps({"agent_id": "sneaky-agent"}), encoding="utf-8")

pm = ProcessManager(db=FakeDB())
registered = list(pm._agents.keys())
print("registered agents:", registered)
ok = "sneaky-agent" not in registered
print("PASS" if ok else "FAIL: data/ 下的 config.json 被注册为 sneaky-agent")
sys.exit(0 if ok else 1)
'''

ACCEPT_T3 = r'''"""验收 T3：回复中未闭合的 ```python 代码块不得被直接执行。"""
import importlib.util
import sys
sys.path.insert(0, ".")

spec = importlib.util.spec_from_file_location(
    "linagent", "agents/lin-shen/agent.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

agent = mod.LinShenAgent()
# 模型输出以未闭合的 python 代码块收尾（截断/超时场景常见）
reply = '@send-to "user:test"\n```txt\n你好\n```\n```python\nprint(1)\n'
actions, _ = agent._parse_structured_response(reply)
kinds = [a.get("type") for a in actions]
print("action types:", kinds)
ok = "code" not in kinds
print("PASS" if ok else "FAIL: 未闭合代码块被当作可执行代码")
sys.exit(0 if ok else 1)
'''

ACCEPT_T4 = r'''"""验收 T4：@send-to 不带引号的目标（如 @send-to user:probe 你好）必须正确寻址，
不得静默退化为全量广播。"""
import importlib.util
import sys
sys.path.insert(0, ".")

spec = importlib.util.spec_from_file_location(
    "linagent", "agents/lin-shen/agent.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

agent = mod.LinShenAgent()
reply = '@send-to user:probe 你好\n```txt\n这是发给 probe 的消息\n```'
actions, _ = agent._parse_structured_response(reply)
tos = [a.get("to") for a in actions if a.get("type") == "message"]
print("message targets:", tos)
ok = any(t == "user:probe" for t in tos)
ok = ok and not any(str(t).startswith("broadcast") for t in tos)
print("PASS" if ok else "FAIL: 无引号目标未正确寻址（静默广播）")
sys.exit(0 if ok else 1)
'''

ACCEPT_T5 = r'''"""验收 T5：PythonShell 并发提交的命令必须各自收到自己的执行结果，
不允许结果丢失/错配/串超时。"""
import sys
import time

sys.path.insert(0, ".")


def main():
    from sdk.base_shell import PythonShell

    sh = PythonShell()
    try:
        # 5 条交错 sleep 的命令并发提交：每条必须收到自己带标记的结果
        ids = {}
        for i in range(5):
            cmd = f'import time; time.sleep({0.5 * i}); print("RESULT_{i}")'
            ids[sh.submit_command(cmd, timeout=8)] = i
        results = {}
        deadline = time.time() + 12
        while time.time() < deadline and len(results) < 5:
            for r in sh.get_results():
                results[r["exec_id"]] = r
            time.sleep(0.2)
        ok = True
        for eid, i in ids.items():
            r = results.get(eid)
            if r is None:
                print(f"cmd{i}: 无结果（被并发抢走/丢失）")
                ok = False
            elif r.get("status") != "completed" or f"RESULT_{i}" not in str(r.get("result", "")):
                print(f"cmd{i}: 错配/异常 status={r.get('status')} result={str(r.get('result', ''))[:40]!r}")
                ok = False
            else:
                print(f"cmd{i}: OK")
        print("PASS" if ok else "FAIL: 并发命令结果缺失/错配/串扰")
        return 0 if ok else 1
    finally:
        sh.shutdown()


if __name__ == "__main__":
    sys.exit(main())
'''

# ── 任务定义 ──────────────────────────────────────────────────────

TASKS = [
    {
        "id": "T1",
        "title": "thinking 配置不生效",
        "difficulty": "简单",
        "desc": (
            "这个平台支持通过 Agent 配置中的 thinking 字段（true/false）控制是否启用"
            " DeepSeek 的思考模式。但实测发现：无论 config 里怎么配置 thinking，"
            " 发给 LLM API 的请求都不带任何思考模式参数——配置完全不生效。\n"
            "影响：想开启思考模式的 agent 静默变成普通模式；想关闭的（省 token）也无法关闭。"
        ),
        "accept": "accept_t1.py",
        "accept_src": ACCEPT_T1,
    },
    {
        "id": "T2",
        "title": "data/ 目录下的 config.json 被误注册为新 Agent",
        "difficulty": "简单-中",
        "desc": (
            "系统启动时扫描 agents/ 目录发现 Agent。实测发现：如果在某个 Agent 的 data/"
            " 目录下存在任何 config.json（例如用户调试时把配置文件复制到了 data/ 里备份），"
            " 它会被当成一个全新 Agent 注册，出现在 Agent 列表里。\n"
            "影响：启动时可能加载意外配置，导致 Agent 注册异常、列表污染、"
            " 甚至尝试启动不存在的 Agent。"
        ),
        "accept": "accept_t2.py",
        "accept_src": ACCEPT_T2,
    },
    {
        "id": "T3",
        "title": "未闭合的代码块被当作可执行代码",
        "difficulty": "中",
        "desc": (
            "Agent 的回复解析器把 ```python 代码块提取出来执行。实测发现：如果模型的回复"
            " 以未闭合的 ```python 块结尾（长回复被截断、或模型忘记写结束标记时常见），"
            " 解析器仍会把块内内容当作代码执行。\n"
            "影响：截断的回复可能触发意外的代码执行；非 python 的未闭合块会被当成"
            " 消息广播出去。"
        ),
        "accept": "accept_t3.py",
        "accept_src": ACCEPT_T3,
    },
    {
        "id": "T4",
        "title": "@send-to 无引号目标静默广播",
        "difficulty": "中",
        "desc": (
            "Agent 用 @send-to \"user:xxx\" 指定消息收件人（引号包裹）。实测发现：如果模型"
            " 写了不带引号的目标（如 @send-to user:probe 你好），解析器找不到任何目标，"
            " 但不会报错——这条消息被静默当成全量广播发给所有 Agent 和用户。\n"
            "影响：本应私密/定向的消息泄漏给所有人；模型也得不到任何错误反馈，会一直犯错。"
        ),
        "accept": "accept_t4.py",
        "accept_src": ACCEPT_T4,
    },
    {
        "id": "T5",
        "title": "PythonShell 并发命令结果错配/丢失",
        "difficulty": "中-难",
        "desc": (
            "Agent 的 Python 执行壳支持并发提交命令。实测发现：同时提交两个命令时，"
            " 其中一个命令的执行结果可能被另一个命令的等待线程抢走并丢弃，"
            " 导致该命令永远等不到结果、直到超时；而超时会重启整个执行进程，"
            " 把仍在运行的其他命令也一起杀掉。\n"
            "影响：并发工具调用下结果错配、丢结果、误报超时，agent 行为错乱。"
        ),
        "accept": "accept_t5.py",
        "accept_src": ACCEPT_T5,
    },
]

# 供 harness 使用
ALL_TASK_IDS = [t["id"] for t in TASKS]
TASK_BY_ID = {t["id"]: t for t in TASKS}


def make_accept_file(task, work_dir):
    """把验收脚本写入任务工作目录。返回验收脚本相对路径。"""
    from pathlib import Path
    p = Path(work_dir) / task["accept"]
    p.write_text(task["accept_src"], encoding="utf-8")
    return task["accept"]


if __name__ == "__main__":
    import sys
    # 自检：验收脚本在未修复代码上应失败（红）
    print("任务集自检需要在工作副本目录内运行（见 harness --check-tasks）")
    sys.exit(0)
