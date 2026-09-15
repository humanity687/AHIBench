import os
import sys
import json
import time
import asyncio
import importlib
import logging
import threading
from logging.handlers import RotatingFileHandler
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from sdk.base_agent import BaseAHIAgent
from sdk.base_shell import BaseAHIShell, PythonShell, DockerShell
from sdk.ahi_bus import AHIBus
from sdk.database import AgentDB
from sdk.llm_client import LLMClient

# 默认控制台日志（文件 handler 在初始化 data_dir 之后才添加）
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("agent_runner")


class AgentRunner:
    """Standardized Agent Process container.

    Loads an agent class from a Python script, injects dependencies,
    manages lifecycle, and exposes standard FastAPI endpoints.
    """

    def __init__(self):
        self.agent_id = os.environ["AHI_AGENT_ID"]
        self.port = int(os.environ.get("AHI_AGENT_PORT", "8001"))
        self.data_dir = os.environ.get("AHI_AGENT_DATA_DIR", f"agents/{self.agent_id}/data")
        self.main_url = os.environ.get("AHI_MAIN_PROCESS_URL", "http://127.0.0.1:8000")

        self.config = self._load_config()
        self.agent: Optional[BaseAHIAgent] = None
        self.shell: Optional[BaseAHIShell] = None
        self.db: Optional[AgentDB] = None
        self.ahi_bus: Optional[AHIBus] = None

        self._scheduler = None
        self._loop_running = False
        self._loop_paused = False
        self._wakeup_interval = self.config.get("wakeup_interval", 10)
        self._scheduler_lock = threading.Lock()
        self._startup_ok = True  # 启动状态标记

    # ── Config Loading ──

    def _load_config(self) -> dict:
        agent_dir = os.path.dirname(os.environ.get("AGENT_CONFIG_PATH", ""))
        if not agent_dir or not os.path.isdir(agent_dir):
            for base in [os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         os.path.dirname(os.path.abspath(__file__))]:
                candidate = os.path.join(base, "agents", self.agent_id)
                if os.path.isdir(candidate):
                    agent_dir = candidate
                    break

        config_path = os.path.join(agent_dir, "config.json") if agent_dir else None
        if not config_path or not os.path.isfile(config_path):
            logger.warning("No config.json found for %s, using defaults", self.agent_id)
            return {}

        with open(config_path, "r", encoding="utf-8-sig") as f:
            config = json.load(f)
        config["_agent_dir"] = agent_dir
        logger.info("Loaded config for %s from %s", self.agent_id, config_path)
        return config

    # ── Agent Loading ──

    def _load_agent(self) -> BaseAHIAgent:
        entry_point = self.config.get("entry_point", "agent:Agent")
        parts = entry_point.split(":")
        if len(parts) != 2:
            raise ValueError(f"Invalid entry_point format: {entry_point}. Expected 'module:ClassName'")

        module_name, class_name = parts

        agent_dir = self.config.get("_agent_dir", "")
        if agent_dir:
            sys.path.insert(0, os.path.dirname(agent_dir))
            sys.path.insert(0, agent_dir)

        try:
            module = importlib.import_module(module_name)
        except ImportError:
            agent_file = os.path.join(agent_dir, f"{module_name}.py")
            if not os.path.isfile(agent_file):
                raise ImportError(f"Cannot find {module_name}.py in {agent_dir}")
            spec = importlib.util.spec_from_file_location(module_name, agent_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

        agent_class = getattr(module, class_name)
        agent = agent_class()

        agent.agent_id = self.agent_id
        agent.agent_name = self.config.get("agent_name", self.agent_id)
        agent._agent_dir = agent_dir
        return agent

    # ── Dependency Injection ──

    def _inject_dependencies(self):
        os.makedirs(self.data_dir, exist_ok=True)
        db_path = os.path.join(self.data_dir, f"{self.agent_id}.db")
        self.db = AgentDB(db_path)

        self.ahi_bus = AHIBus(main_process_url=self.main_url)
        self.ahi_bus.agent_id = self.agent_id
        self.ahi_bus.agent_port = self.port

        shell_type = self.config.get("shell_type", "python")
        if shell_type == "python":
            self.shell = PythonShell(on_result_callback=self._on_shell_result)
        elif shell_type == "docker":
            self.shell = DockerShell(on_result_callback=self._on_shell_result,
                                     config=self.config.get("sandbox", {}))
        else:
            raise ValueError(f"Unknown shell_type: {shell_type}")

        # LLM Client
        model_cfg = self.config.get("model_config", {})
        model_cfg["system_prompt"] = self.config.get("system_prompt",
                                                     "You are a helpful assistant.")
        self.agent.llm_client = LLMClient(model_cfg)
        self.agent.llm_client.set_system_prompt(model_cfg["system_prompt"])

        self.agent.shell = self.shell
        self.agent.db = self.db
        self.agent.ahi_bus = self.ahi_bus

        logger.info("Dependencies injected: shell=%s, db=%s, model=%s",
                    shell_type, db_path, model_cfg.get("model", "unknown"))

    # ── Shell Result Callback ──

    def _on_shell_result(self, result: dict):
        self.agent._put_action({
            "type": "shell_result",
            "content": result.get("result", ""),
            "to": "broadcast:users",
            "source": "shell",
            "exec_id": result.get("exec_id", ""),
            "status": result.get("status", "completed"),
            "command": result.get("command", ""),
            "error": result.get("error", ""),
        })

    # ── Lifecycle ──

    def start(self):
        # 添加 per-agent 日志文件（agent-{id}.log）
        os.makedirs(self.data_dir, exist_ok=True)
        agent_log_path = os.path.join(self.data_dir, f"agent-{self.agent_id}.log")
        _fh = RotatingFileHandler(agent_log_path, maxBytes=10*1024*1024, backupCount=3, encoding="utf-8")
        _fh.setLevel(logging.DEBUG)
        _fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        logging.getLogger().addHandler(_fh)
        logger.info("Agent log file: %s", agent_log_path)

        logger.info("Starting agent %s on port %d", self.agent_id, self.port)
        # 捕获事件循环引用，供 _put_action 跨线程安全调度异步通知
        try:
            self.agent._event_loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.warning("Cannot capture event loop, async notifications may be unreliable")
        try:
            self.agent.on_start()
        except Exception as e:
            logger.error("on_start failed for %s: %s", self.agent_id, e)
            self._startup_ok = False
        self._start_autonomous_loop()
        self._register()
        logger.info("Agent %s started (startup_ok=%s)", self.agent_id, self._startup_ok)

    def stop(self):
        logger.info("Stopping agent %s", self.agent_id)
        self._stop_autonomous_loop()
        try:
            self.agent.on_stop()
        except Exception as e:
            logger.error("on_stop failed: %s", e)
        if self.shell:
            try:
                self.shell.shutdown()
            except Exception:
                pass
        logger.info("Agent %s stopped", self.agent_id)

    def _register(self):
        if self.ahi_bus:
            try:
                self._run_async(self.ahi_bus.register_agent())
            except Exception as e:
                logger.warning("Registration failed: %s", e)

    # ── Autonomous Loop ──

    def _start_autonomous_loop(self):
        with self._scheduler_lock:
            if self._loop_running:
                return
            try:
                from apscheduler.schedulers.background import BackgroundScheduler
                from apscheduler.triggers.interval import IntervalTrigger
            except ImportError:
                logger.warning("APScheduler not available, autonomous loop disabled")
                return

            try:
                self._scheduler = BackgroundScheduler(daemon=True)
                self._scheduler.add_job(
                    self._loop_tick,
                    IntervalTrigger(seconds=self._wakeup_interval),
                    id=f"loop_{self.agent_id}",
                    replace_existing=True,
                )
                self._scheduler.start()
                self._loop_running = True
                logger.info("Autonomous loop started (interval=%ds)", self._wakeup_interval)
            except Exception as e:
                logger.error("Failed to start loop: %s", e)

    def _stop_autonomous_loop(self):
        with self._scheduler_lock:
            if not self._loop_running or not self._scheduler:
                return
            try:
                self._scheduler.shutdown(wait=False)
            except Exception:
                pass
            self._scheduler = None
            self._loop_running = False
            self._loop_paused = False

    def _pause_loop(self):
        with self._scheduler_lock:
            if not self._loop_running or not self._scheduler:
                return False
            try:
                self._scheduler.pause_job(f"loop_{self.agent_id}")
                self._loop_paused = True
                logger.info("Autonomous loop paused")
                return True
            except Exception as e:
                logger.error("Failed to pause loop: %s", e)
                return False

    def _resume_loop(self):
        with self._scheduler_lock:
            if not self._loop_running or not self._scheduler:
                return False
            try:
                self._scheduler.resume_job(f"loop_{self.agent_id}")
                self._loop_paused = False
                logger.info("Autonomous loop resumed")
                return True
            except Exception as e:
                logger.error("Failed to resume loop: %s", e)
                return False

    def _loop_tick(self):
        try:
            self.agent.on_wakeup()
        except Exception as e:
            logger.error("Loop tick error: %s", e)

    def set_wakeup_interval(self, seconds: int):
        if seconds < 5:
            seconds = 5
        self._wakeup_interval = seconds
        with self._scheduler_lock:
            if self._loop_running and self._scheduler:
                try:
                    from apscheduler.triggers.interval import IntervalTrigger
                    self._scheduler.reschedule_job(
                        f"loop_{self.agent_id}",
                        trigger=IntervalTrigger(seconds=seconds),
                    )
                except Exception:
                    pass

    # ── Async Helper ──

    @staticmethod
    def _run_async(coro):
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(coro)
            else:
                asyncio.run(coro)
        except RuntimeError:
            try:
                asyncio.run(coro)
            except Exception:
                pass


# ── Entry Point ──

if __name__ == "__main__":
    import uvicorn

    runner = AgentRunner()
    runner.agent = runner._load_agent()
    runner._inject_dependencies()

    app = FastAPI(
        title=f"AHI Agent: {runner.agent_id}",
        version="3.0.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


    @app.on_event("startup")
    async def startup():
        runner.start()


    @app.on_event("shutdown")
    async def shutdown():
        runner.stop()


    # ── Standard Agent API Endpoints (interfaces-checklist §3.2) ──

    @app.post("/api/input")
    def api_input(input_data: dict):
        try:
            threading.Thread(target=runner.agent.process_input,
                             args=(input_data,), daemon=True).start()
        except Exception as e:
            logger.error("process_input failed: %s", e)
            return JSONResponse(status_code=202, content={"code": 202, "message": "accepted_with_errors",
                                                           "error": str(e)})
        return JSONResponse(status_code=202, content={"code": 202, "message": "accepted"})


    @app.get("/api/outputs")
    def api_outputs():
        outputs = runner.agent.get_outputs()
        return {"code": 200, "data": outputs}


    @app.get("/api/state")
    def api_state():
        state = runner.agent.get_state()
        state["agent_id"] = runner.agent_id
        state["loop_running"] = runner._loop_running
        state["loop_paused"] = runner._loop_paused
        state["wakeup_interval"] = runner._wakeup_interval
        state["startup_ok"] = runner._startup_ok
        return {"code": 200, "data": state}


    @app.post("/api/shutdown")
    def api_shutdown():
        runner.stop()
        return {"code": 200, "message": "Agent shutting down"}


    @app.post("/api/shell/reset")
    def api_shell_reset():
        if runner.shell:
            runner.shell.reset()
            return {"code": 200, "message": "Shell reset"}
        return {"code": 400, "message": "No shell available"}


    @app.put("/api/config")
    def api_config_update(data: dict):
        if "wakeup_interval" in data:
            runner.set_wakeup_interval(int(data["wakeup_interval"]))
        return {"code": 200, "message": "Config updated"}


    @app.post("/api/loop/start")
    def api_loop_start():
        if runner._loop_running:
            return {"code": 200, "message": "Loop already running"}
        runner._start_autonomous_loop()
        return {"code": 200, "message": "Loop started"}


    @app.post("/api/loop/pause")
    def api_loop_pause():
        ok = runner._pause_loop()
        if ok:
            return {"code": 200, "message": "Loop paused"}
        return {"code": 400, "message": "Loop not running"}


    @app.post("/api/loop/resume")
    def api_loop_resume():
        ok = runner._resume_loop()
        if ok:
            return {"code": 200, "message": "Loop resumed"}
        return {"code": 400, "message": "Loop not running"}


    @app.get("/api/health")
    def api_health():
        return {"status": "ok", "agent_id": runner.agent_id}


    logger.info("Booting AgentRunner for %s on port %d", runner.agent_id, runner.port)
    uvicorn.run(app, host="127.0.0.1", port=runner.port, log_level="info")
