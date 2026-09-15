#!/usr/bin/env python3
"""ahi_shell_worker.py — 沙箱容器内的常驻 Python shell worker。

协议（JSON lines，走 stdin/stdout）：
  入：{"type":"exec","cmd_id":N,"command":"..."}
      {"type":"get_variables","cmd_id":N,"include_private":bool}
  出：{"type":"exec_result","cmd_id":N,"output":...,"error":...,"result_repr":...}
      {"type":"variables_result","cmd_id":N,"variables":{...}}

命名空间跨命令保持（与宿主 PythonShell 语义一致）。
fd 隔离：真实 stdout 被 dup 到私有 fd，fd1 重定向到 /dev/null——用户代码即使
os.write(1,...) 也无法污染协议通道。
"""
import builtins
import io
import json
import os
import sys
import traceback
from contextlib import redirect_stdout, redirect_stderr

# ── fd 隔离：协议走私有 fd，fd1 丢弃 ──
_proto_fd = os.dup(1)
_devnull = os.open(os.devnull, os.O_WRONLY)
os.dup2(_devnull, 1)


def _emit(obj):
    os.write(_proto_fd, (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8"))


def main():
    g = {
        "__name__": "__console__",
        "__doc__": None,
        "__builtins__": builtins,
    }
    g["sys"] = sys
    g["os"] = os

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except Exception:
            continue
        mtype = msg.get("type")
        cid = msg.get("cmd_id")

        if mtype == "exec":
            command = (msg.get("command") or "").lstrip("\n").lstrip()
            output, error, result_repr = "", "", None
            so, se = io.StringIO(), io.StringIO()
            try:
                with redirect_stdout(so), redirect_stderr(se):
                    try:
                        obj = compile(command, "<input>", "eval")
                        r = eval(obj, g)
                        if r is not None:
                            result_repr = repr(r)
                            print(repr(r))
                    except SyntaxError:
                        exec(compile(command, "<input>", "exec"), g)
            except SystemExit:
                error = "错误：代码尝试退出系统，操作被禁止"
            except KeyboardInterrupt:
                error = "KeyboardInterrupt"
            except Exception:
                error = traceback.format_exc()
            output = so.getvalue() + (se.getvalue() if se.getvalue() else "")
            _emit({"type": "exec_result", "cmd_id": cid, "output": output,
                   "error": error, "result_repr": result_repr})

        elif mtype == "get_variables":
            import pickle
            include_private = bool(msg.get("include_private"))
            variables = {}
            for name, value in list(g.items()):
                if not include_private and name.startswith("_"):
                    continue
                try:
                    pickle.dumps(value)
                    variables[name] = value
                except Exception:
                    try:
                        variables[name] = repr(value)
                    except Exception:
                        variables[name] = f"<{type(value).__name__} object>"
            _emit({"type": "variables_result", "cmd_id": cid, "variables": variables})

        elif mtype == "ping":
            _emit({"type": "pong", "cmd_id": cid})


if __name__ == "__main__":
    main()
