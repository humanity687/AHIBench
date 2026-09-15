from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Callable, List
from multiprocessing import Process, Queue
from concurrent.futures import ThreadPoolExecutor
import json
import subprocess
import threading
import queue
import time
import os


# ── Worker process (runs in subprocess for isolation + GUI support) ──

def _worker_main(input_queue: Queue, output_queue: Queue):
    """Persistent subprocess worker. Maintains namespace across commands."""
    import sys as _sys
    import os as _os
    import io as _io
    import traceback as _traceback
    import builtins as _builtins
    from contextlib import redirect_stdout as _redirect_stdout
    from contextlib import redirect_stderr as _redirect_stderr

    globals_dict: Dict[str, Any] = {
        '__name__': '__console__',
        '__doc__': None,
        '__builtins__': _builtins,
    }
    globals_dict['sys'] = _sys
    globals_dict['os'] = _os

    while True:
        try:
            msg = input_queue.get()
            if msg is None:
                break

            msg_type = msg[0]

            if msg_type == 'exec':
                command_id = msg[1]
                command = msg[2].lstrip('\\n').lstrip()

                output = ""
                error = ""
                result_repr = None

                stdout_buf = _io.StringIO()
                stderr_buf = _io.StringIO()

                try:
                    with _redirect_stdout(stdout_buf), _redirect_stderr(stderr_buf):
                        try:
                            code_obj = compile(command, "<input>", "eval")
                            result = eval(code_obj, globals_dict)
                            if result is not None:
                                result_repr = repr(result)
                                print(repr(result))
                        except SyntaxError:
                            code_obj = compile(command, "<input>", "exec")
                            exec(code_obj, globals_dict)
                except SystemExit:
                    error = "错误：代码尝试退出系统，操作被禁止"
                except KeyboardInterrupt:
                    error = "KeyboardInterrupt"
                except Exception:
                    error = _traceback.format_exc()

                stdout_content = stdout_buf.getvalue()
                stderr_content = stderr_buf.getvalue()

                output = stdout_content
                if stderr_content:
                    output += stderr_content

                try:
                    output_queue.put(('exec_result', command_id, output, error, result_repr))
                except Exception:
                    break

            elif msg_type == 'get_variables':
                command_id = msg[1]
                include_private = msg[2]

                import pickle as _pickle
                variables = {}
                for name, value in list(globals_dict.items()):
                    if not include_private and name.startswith('_'):
                        continue
                    try:
                        _pickle.dumps(value)
                        variables[name] = value
                    except Exception:
                        try:
                            variables[name] = repr(value)
                        except Exception:
                            variables[name] = f"<{type(value).__name__} object>"

                try:
                    output_queue.put(('variables_result', command_id, variables))
                except Exception:
                    break

        except EOFError:
            break
        except Exception:
            try:
                output_queue.put(('error', -1, "Worker process internal error"))
            except Exception:
                break


# ── Abstract Base ──

class BaseAHIShell(ABC):
    """Shell abstract base class. All custom shells inherit from this."""

    def __init__(self, on_result_callback: Optional[Callable] = None):
        self.command_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.running = True
        self.on_result_callback = on_result_callback
        self._executor = None
        self.execution_thread = threading.Thread(target=self._execution_loop, daemon=True)
        self.execution_thread.start()

    def submit_command(self, command: str, exec_id: str = None,
                       timeout: float = None) -> str:
        if exec_id is None:
            exec_id = f"cmd_{hash(command)}_{int(time.time() * 1000)}"
        self.command_queue.put((exec_id, command, timeout))
        return exec_id

    def get_results(self) -> List[Dict[str, Any]]:
        results = []
        while not self.result_queue.empty():
            results.append(self.result_queue.get())
        return results

    def _execution_loop(self):
        while self.running:
            try:
                exec_id, command, timeout = self.command_queue.get(timeout=1)
                try:
                    result = self._execute_command_impl(command)
                    result_data = {
                        "exec_id": exec_id,
                        "command": command,
                        "result": result,
                        "status": "completed",
                    }
                except Exception as e:
                    result_data = {
                        "exec_id": exec_id,
                        "command": command,
                        "error": str(e),
                        "status": "error",
                    }
                self.result_queue.put(result_data)
                if self.on_result_callback:
                    try:
                        self.on_result_callback(result_data)
                    except Exception:
                        pass
            except queue.Empty:
                continue
            except Exception:
                pass

    @abstractmethod
    def _execute_command_impl(self, command: str) -> Any:
        """Execute a command and return result."""

    def reset(self) -> None:
        """Reset shell state."""

    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """Return current shell state snapshot."""

    def shutdown(self):
        self.running = False


# ── PythonShell (subprocess-backed) ──

class PythonShell(BaseAHIShell):
    """Python shell using multiprocessing subprocess for GUI support and isolation.

    Maintains a persistent worker process that preserves namespace across commands.
    Uses ThreadPoolExecutor for timeout control.
    """

    def __init__(self, on_result_callback: Optional[Callable] = None,
                 config: Optional[Dict[str, Any]] = None):
        super().__init__(on_result_callback)
        self.config = config or {}
        self._input_queue: Optional[Queue] = None
        self._output_queue: Optional[Queue] = None
        self._process: Optional[Process] = None
        self._command_id: int = 0
        self._cid_lock = threading.Lock()
        self._pending: Dict[int, str] = {}  # cmd_id → exec_id

        self._executor = ThreadPoolExecutor(max_workers=2)

        self._start_worker()

    # ── Worker lifecycle ──

    def _start_worker(self):
        self._stop_worker()
        self._input_queue = Queue()
        self._output_queue = Queue()
        self._process = Process(
            target=_worker_main,
            args=(self._input_queue, self._output_queue),
        )
        self._process.daemon = True
        self._process.start()

    def _stop_worker(self):
        if self._input_queue:
            try:
                self._input_queue.put(None, timeout=1)
            except Exception:
                pass
        if self._process and self._process.is_alive():
            self._process.join(timeout=2)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=1)
        self._process = None
        self._input_queue = None
        self._output_queue = None

    def _ensure_worker(self):
        if self._process is None or not self._process.is_alive():
            self._start_worker()

    # ── Command submission (override with subprocess) ──

    def submit_command(self, command: str, exec_id: str = None,
                       timeout: float = 30) -> str:
        if exec_id is None:
            exec_id = f"cmd_{hash(command)}_{int(time.time() * 1000)}"

        self._ensure_worker()

        with self._cid_lock:
            self._command_id += 1
            cmd_id = self._command_id

        self._pending[cmd_id] = exec_id

        def _run():
            try:
                self._input_queue.put(('exec', cmd_id, command), timeout=5)
            except Exception as e:
                self.result_queue.put({
                    "exec_id": exec_id,
                    "command": command,
                    "error": f"无法发送命令到工作进程: {e}",
                    "status": "error",
                })
                self._start_worker()
                return

            deadline = time.time() + timeout
            while time.time() < deadline:
                try:
                    msg = self._output_queue.get(timeout=1)
                    msg_type = msg[0]
                    recv_id = msg[1]

                    if recv_id == cmd_id:
                        if msg_type == 'exec_result':
                            output = msg[2]
                            error = msg[3]
                            result_repr = msg[4]
                            result_text = output
                            if error:
                                result_text = result_text + ("\n" + error if result_text else error)

                            result_data = {
                                "exec_id": exec_id,
                                "command": command,
                                "result": result_text,
                                "output": output,
                                "error": error,
                                "result_repr": result_repr,
                                "status": "error" if error else "completed",
                            }
                            self.result_queue.put(result_data)
                            if self.on_result_callback:
                                try:
                                    self.on_result_callback(result_data)
                                except Exception:
                                    pass
                            self._pending.pop(cmd_id, None)
                            return
                        elif msg_type == 'error':
                            result_data = {
                                "exec_id": exec_id,
                                "command": command,
                                "error": str(msg[2]) if len(msg) > 2 else "Worker error",
                                "status": "error",
                            }
                            self.result_queue.put(result_data)
                            if self.on_result_callback:
                                try:
                                    self.on_result_callback(result_data)
                                except Exception:
                                    pass
                            self._pending.pop(cmd_id, None)
                            return
                    # Discard stale messages
                except queue.Empty:
                    if not self._process or not self._process.is_alive():
                        self.result_queue.put({
                            "exec_id": exec_id,
                            "command": command,
                            "error": "错误：工作进程已崩溃",
                            "status": "error",
                        })
                        self._start_worker()
                        self._pending.pop(cmd_id, None)
                        return

            # Timeout
            self.result_queue.put({
                "exec_id": exec_id,
                "command": command,
                "error": f"错误：命令执行超时（{timeout}秒）",
                "status": "timeout",
            })
            self._pending.pop(cmd_id, None)
            # Restart worker on timeout (stuck command)
            self._start_worker()

        self._executor.submit(_run)
        return exec_id

    # ── Base class overrides ──

    def _execute_command_impl(self, command: str) -> str:
        """Not used directly — submit_command handles execution via subprocess."""
        return ""

    def reset(self) -> None:
        self._start_worker()
        with self._cid_lock:
            self._command_id = 0
            self._pending.clear()

    def get_state(self) -> Dict[str, Any]:
        worker_alive = self._process is not None and self._process.is_alive()
        pending_count = len(self._pending) if self._pending else 0
        return {
            "status": "running" if worker_alive else "restarting",
            "worker_alive": worker_alive,
            "pending_commands": pending_count,
        }

    def shutdown(self):
        super().shutdown()
        self._stop_worker()
        if self._executor:
            self._executor.shutdown(wait=False)


# ── DockerShell（沙箱容器内执行，agent 进程留宿主） ──

class DockerShell(BaseAHIShell):
    """把代码执行放进 Docker 沙箱容器（持久 stdin/stdout worker）。

    与 PythonShell 接口/语义一致（命名空间跨命令保持、超时重启、结果格式），
    但代码在容器内运行——只能看到按配置挂载的沙箱目录，看不到宿主仓库。

    config（agent config 的 `sandbox` 块）：
      image    镜像名（默认 ahi-sandbox:latest）
      root     宿主沙箱根（挂到容器 /workspace:rw，必填）
      name     容器名（默认 ahi-shell）
      mounts   额外挂载 [{"host","container","mode"}]，mode 默认 ro
      network  Docker 网络（默认 none）
      memory / cpus  资源限额
    """

    def __init__(self, on_result_callback: Optional[Callable] = None,
                 config: Optional[Dict[str, Any]] = None):
        super().__init__(on_result_callback)
        self.config = config or {}
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._cid_lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._command_id = 0
        self._pending: Dict[int, str] = {}
        self._slots: Dict[int, "queue.Queue"] = {}
        self._slots_lock = threading.Lock()
        self._proc: Optional[subprocess.Popen] = None
        self._reader: Optional[threading.Thread] = None
        self._start_worker()

    # ── 容器/worker 生命周期 ──

    def _docker_args(self) -> List[str]:
        c = self.config
        root = c.get("root")
        if not root:
            raise RuntimeError("DockerShell 需要 sandbox.root")
        name = c.get("name") or "ahi-shell"
        args = [
            "docker", "run", "-i", "--rm", "--name", name,
            "--network", c.get("network", "none"),
            "--memory", str(c.get("memory", "512m")),
            "--cpus", str(c.get("cpus", "1")),
            "--read-only", "--tmpfs", "/tmp:size=64m",
            "-w", "/workspace",
            "-v", f"{root}:/workspace:rw",
        ]
        for m in c.get("mounts", []) or []:
            mode = m.get("mode", "ro")
            args += ["-v", f"{m['host']}:{m['container']}:{mode}"]
        args += [c.get("image", "ahi-sandbox:latest"),
                 "python3", "-u", "/opt/ahi_shell_worker.py"]
        return args

    def _start_worker(self):
        self._stop_worker()
        try:
            self._proc = subprocess.Popen(
                self._docker_args(),
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True, bufsize=1,
            )
        except Exception as e:
            self._proc = None
            raise RuntimeError(f"DockerShell 启动失败: {e}") from e
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

    def _read_loop(self):
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except Exception:
                continue
            cid = msg.get("cmd_id")
            with self._slots_lock:
                slot = self._slots.get(cid)
            if slot is not None:
                slot.put(msg)

    def _stop_worker(self):
        proc = self._proc
        name = self.config.get("name") or "ahi-shell"
        if proc is not None:
            try:
                if proc.stdin:
                    proc.stdin.close()
            except Exception:
                pass
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except Exception:
                    proc.kill()
        self._proc = None
        # 兜底清理同名容器
        try:
            subprocess.run(["docker", "rm", "-f", name],
                           capture_output=True, timeout=10)
        except Exception:
            pass

    def _ensure_worker(self):
        if self._proc is None or self._proc.poll() is not None:
            self._start_worker()

    # ── 命令提交 ──

    def submit_command(self, command: str, exec_id: str = None,
                       timeout: float = 30) -> str:
        if exec_id is None:
            exec_id = f"cmd_{hash(command)}_{int(time.time() * 1000)}"

        self._ensure_worker()

        with self._cid_lock:
            self._command_id += 1
            cmd_id = self._command_id

        slot: "queue.Queue" = queue.Queue()
        with self._slots_lock:
            self._slots[cmd_id] = slot
        self._pending[cmd_id] = exec_id

        def _run():
            try:
                with self._write_lock:
                    self._proc.stdin.write(json.dumps(
                        {"type": "exec", "cmd_id": cmd_id, "command": command},
                        ensure_ascii=False) + "\n")
                    self._proc.stdin.flush()
            except Exception as e:
                self.result_queue.put({
                    "exec_id": exec_id, "command": command,
                    "error": f"无法发送命令到沙箱容器: {e}", "status": "error",
                })
                self._start_worker()
                return

            deadline = time.time() + timeout
            while time.time() < deadline:
                try:
                    msg = slot.get(timeout=1)
                except queue.Empty:
                    if self._proc is None or self._proc.poll() is not None:
                        self.result_queue.put({
                            "exec_id": exec_id, "command": command,
                            "error": "错误：沙箱容器已崩溃", "status": "error",
                        })
                        self._start_worker()
                        self._pending.pop(cmd_id, None)
                        return
                    continue

                output = msg.get("output", "")
                error = msg.get("error", "")
                result_text = output
                if error:
                    result_text = result_text + ("\n" + error if result_text else error)
                result_data = {
                    "exec_id": exec_id, "command": command,
                    "result": result_text, "output": output, "error": error,
                    "result_repr": msg.get("result_repr"),
                    "status": "error" if error else "completed",
                }
                self.result_queue.put(result_data)
                if self.on_result_callback:
                    try:
                        self.on_result_callback(result_data)
                    except Exception:
                        pass
                self._pending.pop(cmd_id, None)
                with self._slots_lock:
                    self._slots.pop(cmd_id, None)
                return

            # 超时：重启容器（卡死命令）
            self.result_queue.put({
                "exec_id": exec_id, "command": command,
                "error": f"错误：命令执行超时（{timeout}秒）", "status": "timeout",
            })
            self._pending.pop(cmd_id, None)
            with self._slots_lock:
                self._slots.pop(cmd_id, None)
            self._start_worker()

        self._executor.submit(_run)
        return exec_id

    # ── Base 接口 ──

    def _execute_command_impl(self, command: str) -> str:
        return ""

    def reset(self) -> None:
        self._start_worker()
        with self._cid_lock:
            self._command_id = 0
            self._pending.clear()
        with self._slots_lock:
            self._slots.clear()

    def get_state(self) -> Dict[str, Any]:
        alive = self._proc is not None and self._proc.poll() is None
        return {
            "status": "running" if alive else "restarting",
            "worker_alive": alive,
            "pending_commands": len(self._pending),
            "sandbox": True,
            "container": self.config.get("name"),
        }

    def shutdown(self):
        super().shutdown()
        self._stop_worker()
        if self._executor:
            self._executor.shutdown(wait=False)
