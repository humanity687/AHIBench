import os
import json
import signal
import socket
import subprocess
import sys
import threading
import time
import glob
import asyncio
from typing import Dict, Any, Optional, List

import psutil

from src.system_db import SystemDB


class ProcessManager:
    """Agent lifecycle manager — scan, start, stop, monitor, crash-restart."""

    def __init__(self, db: SystemDB = None, ws_server=None, main_port: int = 8000):
        self.db = db or SystemDB()
        self.ws = ws_server
        self._base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self._agents_dir = os.path.join(self._base_dir, "agents")

        self._agents: Dict[str, Dict[str, Any]] = {}
        self._processes: Dict[str, subprocess.Popen] = {}
        self._ports: Dict[str, int] = {}
        self._retry_count: Dict[str, int] = {}
        self._retry_last_time: Dict[str, float] = {}
        self._heartbeat_misses: Dict[str, int] = {}
        self._lock = threading.Lock()
        self._running = threading.Event()
        self.on_drain_outputs = None  # 回调: fn(agent_id, outputs)
        self._main_port = main_port
        self._running.set()

        self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._heartbeat_thread.start()
        self._resource_thread = threading.Thread(target=self._resource_loop, daemon=True)
        self._resource_thread.start()

        self.discover_agents()

    # ── Auto-start ──

    def auto_start_agents(self):
        """Start all agents that have auto_start: true in their config."""
        for agent_id, config in self._agents.items():
            if config.get("auto_start", False):
                self.db.add_log("INFO", "ProcessManager",
                                f"Auto-starting agent {agent_id}")
                self.start_agent(agent_id)

    # ── Agent Discovery ──

    def discover_agents(self) -> List[Dict[str, Any]]:
        agents = []
        if not os.path.isdir(self._agents_dir):
            return agents
        pattern = os.path.join(self._agents_dir, "**", "config.json")
        for config_path in glob.glob(pattern, recursive=True):
            agent_dir = os.path.dirname(config_path)
            # 跳过 data/ 子目录下的 config.json，避免误匹配
            path_parts = agent_dir.replace(os.sep, "/").split("/")
            if "data" in path_parts:
                self.db.add_log("DEBUG", "ProcessManager",
                                f"Skipping config in data directory: {config_path}")
                continue
            try:
                with open(config_path, "r", encoding="utf-8-sig") as f:
                    config = json.load(f)
                agent_id = config.get("agent_id")
                if not agent_id:
                    continue
                config["_dir"] = agent_dir
                config["_config_path"] = config_path
                self._agents[agent_id] = config
                self.db.save_agent(config)
                self.db.add_log("INFO", "ProcessManager", f"Discovered agent: {agent_id}")
                agents.append(config)
            except Exception as e:
                self.db.add_log("ERROR", "ProcessManager",
                                f"Failed to parse config {config_path}: {e}")
        return agents

    # ── Port Allocation ──

    def _allocate_port(self, start: int = 50000) -> int:
        port = start
        while port < 60000:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(("127.0.0.1", port))
                    return port
            except OSError:
                port += 1
        raise RuntimeError("No available ports in range")

    # ── Agent Start ──

    def start_agent(self, agent_id: str) -> bool:
        config = self._agents.get(agent_id)
        if not config:
            self.db.add_log("ERROR", "ProcessManager", f"Agent {agent_id} not found in registry")
            return False

        with self._lock:
            if agent_id in self._processes and self._processes[agent_id].poll() is None:
                self.db.add_log("WARNING", "ProcessManager", f"Agent {agent_id} already running")
                return False

        # Allocate port
        port = self._ports.get(agent_id)
        if not port:
            port = self._allocate_port()
            self._ports[agent_id] = port

        # Find agent_runner.py
        runner_path = os.path.join(self._base_dir, "sdk", "agent_runner.py")
        if not os.path.isfile(runner_path):
            self.db.add_log("ERROR", "ProcessManager", f"agent_runner.py not found at {runner_path}")
            return False

        agent_dir = config.get("_dir", os.path.join(self._agents_dir, agent_id))
        data_dir = os.path.join(agent_dir, "data")
        os.makedirs(data_dir, exist_ok=True)

        env = os.environ.copy()
        env["PYTHONPATH"] = self._base_dir
        env["AHI_MAIN_PROCESS_URL"] = f"http://127.0.0.1:{self._main_port}"
        env["AHI_AGENT_ID"] = agent_id
        env["AHI_AGENT_PORT"] = str(port)
        env["AHI_AGENT_DATA_DIR"] = data_dir
        env["AGENT_CONFIG_PATH"] = config.get("_config_path",
                                               os.path.join(agent_dir, "config.json"))

        self.db.add_log("INFO", "ProcessManager", f"Starting agent {agent_id} on port {port}")

        # 诊断日志：stdout/stderr 写入 agent 数据目录
        stdout_log = open(os.path.join(data_dir, "stdout.log"), "a")
        stderr_log = open(os.path.join(data_dir, "stderr.log"), "a")
        stdout_log.write(f"\n=== Agent {agent_id} started at {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        stderr_log.write(f"\n=== Agent {agent_id} started at {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        stdout_log.flush(); stderr_log.flush()

        try:
            # 沙箱：agent 进程 cwd 设为沙箱根（相对路径文件操作自然落在沙箱内）
            sbx = config.get("sandbox", {}) or {}
            agent_cwd = self._base_dir
            if sbx.get("enabled") and sbx.get("root") and sbx.get("agent_cwd", True):
                os.makedirs(sbx["root"], exist_ok=True)
                agent_cwd = sbx["root"]

            proc = subprocess.Popen(
                [sys.executable, runner_path],
                env=env,
                cwd=agent_cwd,
                stdout=stdout_log,
                stderr=stderr_log,
                start_new_session=True,
            )

            with self._lock:
                self._processes[agent_id] = proc
                if agent_id not in self._retry_count:
                    self._retry_count[agent_id] = 0

            # Wait for registration (15s timeout)
            registered = self._wait_for_registration(agent_id, proc, port)
            if not registered:
                self.db.add_log("ERROR", "ProcessManager",
                                f"Agent {agent_id} registration timeout, killing")
                try:
                    proc.send_signal(signal.SIGKILL)
                except Exception:
                    pass
                with self._lock:
                    self._processes.pop(agent_id, None)
                    self._ports.pop(agent_id, None)
                return False

            self.db.add_log("INFO", "ProcessManager",
                            f"Agent {agent_id} registered (PID: {proc.pid})")
            self.db.add_event("agent_online", f"Agent {agent_id} 上线")
            if self.ws:
                self.ws.broadcast_agent_online(agent_id)
            return True
        except Exception as e:
            self.db.add_log("ERROR", "ProcessManager", f"Failed to start {agent_id}: {e}")
            with self._lock:
                self._processes.pop(agent_id, None)
                self._ports.pop(agent_id, None)
            return False

    def _wait_for_registration(self, agent_id: str, proc, port: int, timeout: float = 15.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if proc.poll() is not None:
                return False
            try:
                state = self._http_get(f"http://127.0.0.1:{port}/api/health")
                if state is not None:
                    return True
            except Exception:
                pass
            time.sleep(0.5)
        return False

    # ── Three-Level Stop ──

    def stop_agent(self, agent_id: str, level: int = 1) -> bool:
        with self._lock:
            proc = self._processes.get(agent_id)
            if not proc or proc.poll() is not None:
                self.db.set_agent_offline(agent_id)
                return True
            port = self._ports.get(agent_id)

        # Level 1: HTTP soft shutdown
        if level >= 1 and port:
            try:
                self._http_post(f"http://127.0.0.1:{port}/api/shutdown", timeout=3)
                time.sleep(1)
                if proc.poll() is not None:
                    self._cleanup_agent(agent_id)
                    return True
            except Exception:
                pass

        # Level 2: SIGTERM
        if level >= 2:
            try:
                proc.send_signal(signal.SIGTERM)
                self.db.add_log("INFO", "ProcessManager", f"SIGTERM sent to {agent_id}")
                try:
                    proc.wait(timeout=3)
                    self._cleanup_agent(agent_id)
                    return True
                except subprocess.TimeoutExpired:
                    pass
            except Exception as e:
                self.db.add_log("ERROR", "ProcessManager", f"SIGTERM failed for {agent_id}: {e}")

        # Level 3: SIGKILL
        if level >= 3:
            try:
                proc.send_signal(signal.SIGKILL)
                self.db.add_log("INFO", "ProcessManager", f"SIGKILL sent to {agent_id}")
                self._cleanup_agent(agent_id)
                return True
            except Exception as e:
                self.db.add_log("ERROR", "ProcessManager", f"SIGKILL failed for {agent_id}: {e}")

        return False

    def emergency_stop_all(self):
        with self._lock:
            agent_ids = list(self._processes.keys())
        self.db.add_log("INFO", "ProcessManager", f"Emergency stop: {len(agent_ids)} agents")
        for aid in agent_ids:
            if self.ws:
                self.ws.broadcast_agent_offline(aid)
            self.stop_agent(aid, level=3)

    def _cleanup_agent(self, agent_id: str):
        with self._lock:
            self._processes.pop(agent_id, None)
            self._ports.pop(agent_id, None)
        self.db.set_agent_offline(agent_id)
        self.db.add_event("agent_offline", f"Agent {agent_id} 下线")
        if self.ws:
            self.ws.broadcast_agent_offline(agent_id)

    # ── Heartbeat ──

    def _heartbeat_loop(self):
        while self._running.is_set():
            with self._lock:
                agents = list(self._processes.items())
            for agent_id, proc in agents:
                # 检查进程是否还活着（避免重复处理已被 _resource_loop 或 _handle_crash 清理的 agent）
                with self._lock:
                    current_proc = self._processes.get(agent_id)
                if current_proc is None or current_proc.poll() is not None:
                    # 进程已死，尝试 crash 处理（但只在 _heartbeat_misses 还跟踪着时触发）
                    if agent_id in self._heartbeat_misses:
                        self._handle_crash(agent_id)
                    continue
                port = self._ports.get(agent_id)
                if not port:
                    continue
                try:
                    state = self._http_get(f"http://127.0.0.1:{port}/api/state")
                    if state is not None:
                        data = state.get("data", state)
                        self.db.update_state(agent_id, {
                            "pid": proc.pid,
                            "status": "online",
                            "port": port,
                            "cpu_usage": data.get("cpu_usage", 0.0),
                            "memory_usage": data.get("memory_usage", 0.0),
                            "waiting": data.get("waiting", "") or "",
                            "muted": json.dumps(data.get("muted", []), ensure_ascii=False)
                                     if data.get("muted") else "",
                        })
                        self._heartbeat_misses[agent_id] = 0
                        if agent_id in self._retry_count:
                            last = self._retry_last_time.get(agent_id, 0)
                            if time.time() - last > 60:
                                self._retry_count[agent_id] = max(0, self._retry_count[agent_id] - 1)
                                self._retry_last_time[agent_id] = time.time()
                        # 兜底拉取孤儿输出（通知丢失的安全网）
                        self._drain_orphan_outputs(agent_id, port)
                    else:
                        self._handle_heartbeat_miss(agent_id)
                except Exception:
                    self._handle_heartbeat_miss(agent_id)
            time.sleep(2)

    def _handle_heartbeat_miss(self, agent_id: str):
        """连续 3 次心跳丢失才触发崩溃处理。"""
        with self._lock:
            misses = self._heartbeat_misses.get(agent_id, 0) + 1
            self._heartbeat_misses[agent_id] = misses
        if misses >= 3:
            self.db.add_log("WARNING", "ProcessManager",
                            f"Agent {agent_id} heartbeat lost {misses} times")
            self._handle_crash(agent_id)

    def _drain_orphan_outputs(self, agent_id: str, port: int):
        """兜底：心跳时拉取可能因通知丢失而滞留的输出。"""
        try:
            result = self._http_get(f"http://127.0.0.1:{port}/api/outputs")
            if result and result.get("code") == 200:
                outputs = result.get("data", [])
                if outputs and self.on_drain_outputs:
                    self.on_drain_outputs(agent_id, outputs)
        except Exception:
            pass

    # ── Resource Monitor ──

    def _resource_loop(self):
        while self._running.is_set():
            with self._lock:
                agents = list(self._processes.items())
            for agent_id, proc in agents:
                # 双重检查：确保进程仍然存在且未被其他线程处理
                with self._lock:
                    current_proc = self._processes.get(agent_id)
                if current_proc is None or current_proc.poll() is not None:
                    continue
                config = self._agents.get(agent_id, {})
                max_memory_mb = config.get("max_memory_mb", 512)
                try:
                    p = psutil.Process(proc.pid)
                    memory_mb = p.memory_info().rss / (1024 * 1024)
                    if memory_mb > max_memory_mb:
                        self.db.add_log("WARNING", "ProcessManager",
                                        f"Agent {agent_id} memory {memory_mb:.0f}MB > {max_memory_mb}MB, restarting")
                        # 先标记为离线，再 stop → start 原子化操作
                        with self._lock:
                            # 确保进程还在（可能在 stop 过程中被其他线程修改）
                            if agent_id not in self._processes:
                                continue
                            if self.ws:
                                self.ws.broadcast_agent_offline(agent_id)
                            # 释放锁后再 stop/start 避免死锁
                        self.stop_agent(agent_id, level=2)
                        if config.get("auto_restart", True):
                            time.sleep(1)
                            self.start_agent(agent_id)
                            if self.ws:
                                self.ws.broadcast_agent_online(agent_id)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            time.sleep(5)

    # ── Crash Handling ──

    def _handle_crash(self, agent_id: str):
        with self._lock:
            proc = self._processes.pop(agent_id, None)
            if proc is None:
                # 已被其他线程处理（如 _resource_loop）
                self._heartbeat_misses.pop(agent_id, None)
                return
            config = self._agents.get(agent_id, {})
            retries = self._retry_count.get(agent_id, 0)
            # 清理 heartbeat misses
            self._heartbeat_misses.pop(agent_id, None)

        self.db.set_agent_offline(agent_id)
        self.db.add_event("agent_offline", f"Agent {agent_id} 离线（崩溃/停止）")
        if self.ws:
            self.ws.broadcast_agent_offline(agent_id)

        if config.get("auto_restart", True):
            if retries < 3:
                with self._lock:
                    self._retry_count[agent_id] = retries + 1
                self.db.add_log("INFO", "ProcessManager",
                                f"Agent {agent_id} crashed, restarting ({retries + 1}/3)")
                time.sleep(1)
                self.start_agent(agent_id)
            else:
                self.db.add_log("ERROR", "ProcessManager",
                                f"Agent {agent_id} max retries reached, giving up")
                with self._lock:
                    self._retry_count.pop(agent_id, None)
        else:
            self.db.add_log("INFO", "ProcessManager",
                            f"Agent {agent_id} crashed, auto_restart disabled")

    # ── Query Methods ──

    def get_online_agents(self) -> List[Dict[str, Any]]:
        return self.db.get_online_agents()

    def get_agent(self, agent_id: str) -> Optional[Dict[str, Any]]:
        return self._agents.get(agent_id)

    def get_agent_info(self, agent_id: str) -> Optional[Dict[str, Any]]:
        state = self.db.get_agent_state(agent_id)
        if state:
            with self._lock:
                proc = self._processes.get(agent_id)
                state["running"] = proc is not None and proc.poll() is None if proc else False
                # 用内存中的端口覆盖 DB 中的旧端口（避免 crash-restart 窗口期读到旧值）
                if agent_id in self._ports:
                    state["port"] = self._ports[agent_id]
        return state

    def get_agent_count(self) -> int:
        with self._lock:
            return len([p for p in self._processes.values() if p.poll() is None])

    # ── HTTP Helpers (sync wrappers) ──

    @staticmethod
    def _http_get(url: str, timeout: float = 2.0) -> Optional[dict]:
        try:
            import urllib.request
            import json
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if resp.status in (200, 202):
                    return json.loads(resp.read().decode())
                return None
        except Exception:
            return None

    @staticmethod
    def _http_post(url: str, timeout: float = 2.0,
                   json_data: dict = None) -> Optional[dict]:
        try:
            import urllib.request
            import json
            data = json.dumps(json_data).encode() if json_data else None
            req = urllib.request.Request(url, data=data, method="POST")
            if data:
                req.add_header("Content-Type", "application/json")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if resp.status == 200:
                    return json.loads(resp.read().decode())
                return None
        except Exception:
            return None

    @staticmethod
    def _http_put(url: str, timeout: float = 2.0,
                  json_data: dict = None) -> Optional[dict]:
        try:
            import urllib.request
            import json
            data = json.dumps(json_data).encode() if json_data else None
            req = urllib.request.Request(url, data=data, method="PUT")
            if data:
                req.add_header("Content-Type", "application/json")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if resp.status == 200:
                    return json.loads(resp.read().decode())
                return None
        except Exception:
            return None

    def cleanup(self):
        self._running.clear()
        self.db.add_log("INFO", "ProcessManager", "Shutting down all agents...")
        with self._lock:
            agent_ids = list(self._processes.keys())
        for aid in agent_ids:
            try:
                proc = self._processes.get(aid)
                if proc and proc.poll() is None:
                    proc.send_signal(signal.SIGTERM)
                    try:
                        proc.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        proc.kill()
            except Exception:
                pass
        self.db.add_log("INFO", "ProcessManager", "All agents stopped")
