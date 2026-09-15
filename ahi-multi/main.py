import asyncio
import os
import sys
import json
import logging
import time
import subprocess
import signal
import threading
from contextlib import asynccontextmanager
from logging.handlers import RotatingFileHandler
from typing import Dict, Any, Optional, List

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.system_db import SystemDB
from src.process_manager import ProcessManager
from src.websocket_server import WebSocketServer
from src.message_router import MessageRouter

# 日志：同时输出到控制台和 main.log
log_dir = os.path.dirname(os.path.abspath(__file__))
log_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
_file_handler = RotatingFileHandler(
    os.path.join(log_dir, "main.log"), maxBytes=20*1024*1024, backupCount=5, encoding="utf-8")
_file_handler.setLevel(logging.DEBUG)
_file_handler.setFormatter(log_fmt)
_console_handler = logging.StreamHandler()
_console_handler.setLevel(logging.INFO)
_console_handler.setFormatter(log_fmt)
logging.basicConfig(level=logging.DEBUG, handlers=[_file_handler, _console_handler])
logger = logging.getLogger("main")


# ── Load Config ──

def load_config() -> dict:
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
    if os.path.isfile(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

config = load_config()

# ── Cloudflare Tunnel ──

_tunnel_process = None
_tunnel_connected = False
_tunnel_url = ""       # 快速隧道生成的 trycloudflare.com 地址
_tunnel_log = []       # tunnel 完整日志（最多保留 500 行）
_MAX_TUNNEL_LOG = 500

def _start_tunnel():
    global _tunnel_process, _tunnel_connected, _tunnel_url

    # 默认不启用 tunnel，需在 config.json 中显式开启
    if not config.get("tunnel_enabled", False):
        logger.info("Cloudflare Tunnel disabled (tunnel_enabled not set)")
        return

    tunnel_name = config.get("cloudflare_tunnel_name", "")
    token = config.get("cloudflare_tunnel_token", "")
    port = config.get("main_process_port", 8000)

    # 快速隧道专用空配置：避免读取 ~/.cloudflared/config.yml（其 404 兜底规则
    # 会劫持快速隧道 hostname → 全部请求 404）
    empty_cfg = "/tmp/ahi_tunnel_empty.yml"
    try:
        if not os.path.exists(empty_cfg):
            with open(empty_cfg, "w") as f:
                f.write("# empty\n")
    except Exception:
        pass

    # 优先级：命名隧道名 > token > 快速隧道
    if tunnel_name:
        cmd = ["cloudflared", "tunnel", "run", tunnel_name]
        logger.info("Starting Cloudflare tunnel '%s'...", tunnel_name)
    elif token:
        cmd = ["cloudflared", "tunnel", "run", "--token", token]
        logger.info("Starting Cloudflare named tunnel (token)...")
    else:
        cmd = ["cloudflared", "tunnel", "--url", f"http://127.0.0.1:{port}",
               "--config", empty_cfg]
        logger.info("Starting Cloudflare quick tunnel (no config)...")

    try:
        _tunnel_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        _tunnel_connected = True

        def _read_tunnel_output():
            global _tunnel_connected, _tunnel_url
            for line in _tunnel_process.stdout:
                line_s = line.strip()
                if line_s:
                    _tunnel_log.append(line_s)
                    if len(_tunnel_log) > _MAX_TUNNEL_LOG:
                        _tunnel_log.pop(0)

                    # 捕捉快速隧道生成的 URL:  https://xxx.trycloudflare.com
                    if not _tunnel_url and "trycloudflare.com" in line_s:
                        import re
                        m = re.search(r'https://[-\w]+\.trycloudflare\.com', line_s)
                        if m:
                            _tunnel_url = m.group(0)
                            logger.info("Cloudflare Tunnel URL: %s", _tunnel_url)

                    if "inf" in line_s.lower():
                        logger.info("Cloudflare Tunnel: %s", line_s[:200])
            _tunnel_process.wait()
            _tunnel_connected = False
            logger.warning("Cloudflare Tunnel process exited (code=%s)", _tunnel_process.returncode)

        threading.Thread(target=_read_tunnel_output, daemon=True).start()
        logger.info("Cloudflare Tunnel started (PID: %d)", _tunnel_process.pid)
    except FileNotFoundError:
        logger.warning("cloudflared binary not found, tunnel disabled")
    except Exception as e:
        logger.error("Failed to start Cloudflare Tunnel: %s", e)

def _stop_tunnel():
    global _tunnel_process, _tunnel_connected
    if _tunnel_process is None:
        return
    logger.info("Stopping Cloudflare Tunnel...")
    try:
        _tunnel_process.send_signal(signal.SIGTERM)
        try:
            _tunnel_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _tunnel_process.kill()
    except Exception as e:
        logger.warning("Error stopping tunnel: %s", e)
    _tunnel_process = None
    _tunnel_connected = False
    logger.info("Cloudflare Tunnel stopped")

# ── Initialize Components ──

# 保存主事件循环引用，供非异步线程使用
_main_loop: Optional[asyncio.AbstractEventLoop] = None

db = SystemDB(config.get("system_db_path"))
ws_server = WebSocketServer()
pm = ProcessManager(db=db, ws_server=ws_server, main_port=config.get("main_process_port", 8000))
router = MessageRouter(db=db, process_manager=pm, ws_server=ws_server)

# 用户上下线 → DB + 系统事件流（决策：用户上下线日志进入 agent 系统状态）
def _on_user_online(user_id: str, name: str = ""):
    db.set_user_online(user_id, name)
    db.add_event("user_online", f"用户 {name or user_id} 上线")
    ws_server.push_to_channel("system", {
        "type": "user_online",
        "payload": {"user_id": user_id, "name": name},
    })

def _on_user_offline(user_id: str):
    db.set_user_offline(user_id)
    db.add_event("user_offline", f"用户 {user_id} 下线")
    ws_server.push_to_channel("system", {
        "type": "user_offline",
        "payload": {"user_id": user_id},
    })

ws_server._on_user_online = _on_user_online
ws_server._on_user_offline = _on_user_offline

# 日志写入 DB 时同步推送到 WebSocket system channel
def _on_db_log(level: str, source: str, message: str):
    ws_server.push_to_channel("system", {
        "type": "log",
        "payload": {"level": level, "source": source, "content": message},
    })
db.on_log = _on_db_log

# Wire incoming chat from WebSocket → MessageRouter
async def _on_incoming_chat(source: str, target: str, content: str):
    import uuid as _uuid
    ts = time.time()
    mid = _uuid.uuid4().hex
    if target.startswith("agent:"):
        target_type = "agent"
        target_name = target[6:]
    elif target.startswith("user:"):
        target_type = "user"
        target_name = target[5:]
    else:
        target_type = "unknown"
        target_name = target

    base_msg = {
        "type": "text",
        "from": f"user:{source}",
        "from_agent": f"user:{source}",
        "to": target,
        "content": content,
        "source": source,
        "timestamp": ts,
        "msg_id": mid,
        "data": {"content": content},
        "metadata": {
            "msg_id": mid,
            "timestamp": ts,
            "source": f"user:{source}",
            "source_type": "user",
            "source_name": source,
            "target": target,
            "target_type": target_type,
            "target_name": target_name,
        },
        "payload": {"content": content, "source": source},
    }

    if target.startswith("agent:"):
        base_msg["conversation_id"] = "default"
        db.save_global_message(from_agent=f"user:{source}", to_target=target,
                               content=content, msg_type="text")
        await router.forward_to_agent(target, base_msg)
    elif target.startswith("user:"):
        base_msg["type"] = "message"
        db.save_global_message(from_agent=f"user:{source}", to_target=target,
                               content=content, msg_type="text")
        ws_server.push_to_user(target[5:], base_msg)
    elif target == "broadcast:agents":
        db.save_global_message(from_agent=f"user:{source}", to_target=target,
                               content=content, msg_type="text")
        await router.broadcast_to_agents(base_msg)
    elif target == "broadcast:users":
        base_msg["type"] = "message"
        db.save_global_message(from_agent=f"user:{source}", to_target=target,
                               content=content, msg_type="text")
        ws_server.broadcast_to_users(base_msg)
    else:
        base_msg["type"] = "message"
        ws_server.push_to_channel("system", base_msg)

ws_server._on_incoming_chat = _on_incoming_chat

# 心跳兜底：孤儿输出通过 MessageRouter 路由
def _on_drain_outputs(agent_id: str, outputs: list):
    """从非异步线程（心跳）调用的兜底路由。
    
    使用 run_coroutine_threadsafe 将协程调度到主事件循环，
    避免 asyncio.run() 在线程中创建临时循环的开销与风险。
    """
    loop = _main_loop
    if loop is None:
        logger.warning("_on_drain_outputs: main loop not available, skipping %d outputs", len(outputs))
        return
    for output in outputs:
        try:
            asyncio.run_coroutine_threadsafe(
                router._route_action(output), loop
            )
        except Exception as e:
            logger.warning("_on_drain_outputs: failed to schedule route_action: %s", e)
pm.on_drain_outputs = _on_drain_outputs


# ── FastAPI App ──

start_time = time.time()

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _main_loop
    _main_loop = asyncio.get_running_loop()
    logger.info("Main process starting...")
    # 预注册实验员身份（2026-08-28 拍定）：公告/告别/操纵消息的稳定来源。
    # 不注册 → agent 回复 user:experimenter 触发"不可达"回执（主实验实测踩坑）。
    db.set_user_online("user:experimenter", "实验员")
    _start_tunnel()
    pm.auto_start_agents()
    yield
    logger.info("Shutting down main process...")
    pm.cleanup()
    _stop_tunnel()
    logger.info("Main process stopped")

app = FastAPI(
    title="AHI-Multi v3.0",
    description="Multi Digital Life Support System",
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_api_websocket_route("/ws", ws_server.get_ws_handler())


# ── WebSocket ──

@app.get("/ws/info")
async def websocket_docs():
    return {"message": "WebSocket endpoint available at ws://localhost:8000/ws"}


# ── Agent Registration & Notification ──

@app.post("/api/v1/agent/register")
def agent_register(data: Dict[str, Any]):
    agent_id = data.get("agent_id")
    if not agent_id:
        return {"code": 400, "message": "missing agent_id"}
    db.update_state(agent_id, {
        "pid": data.get("pid", 0),
        "status": "online",
        "port": data.get("agent_port", 0),
        "cpu_usage": data.get("cpu_usage", 0.0),
        "memory_usage": data.get("memory_usage", 0.0),
    })
    ws_server.broadcast_agent_online(agent_id)
    db.add_log("INFO", "System", f"Agent {agent_id} registered")
    return {"code": 200, "message": "registered", "agent_id": agent_id}


@app.post("/api/v1/agent/notify")
async def agent_notify(data: Dict[str, Any]):
    return await router.handle_notification(data)


# ── Messages ──

@app.post("/api/v1/messages")
async def post_message(data: Dict[str, Any]):
    from_agent = data.get("from", "")
    to = data.get("to", "")
    content = data.get("content", "")
    msg_type = data.get("type", "text")

    if not from_agent:
        return {"code": 400, "message": "missing 'from' field"}

    # API 用户（如实验控制器 probe）：发消息即视为在线（系统状态与消息流一致，
    # 否则 agent 看到"消息来自 user:probe"但快照在线用户里没有 probe → 目标混乱）
    if from_agent.startswith("user:"):
        db.set_user_online(from_agent, from_agent[len("user:"):])

    msg_id = db.save_global_message(
        from_agent=from_agent,
        to_target=to,
        content=content,
        msg_type=msg_type,
    )

    # 统一 JSON 消息体
    ts = time.time()
    target_type = to.split(":")[0] if ":" in to else "unknown"
    target_name = to.split(":", 1)[1] if ":" in to else to
    source_type = from_agent.split(":")[0] if ":" in from_agent else "unknown"
    source_name = from_agent.split(":", 1)[1] if ":" in from_agent else from_agent

    base_msg = {
        "type": msg_type,
        "from": from_agent,
        "from_agent": from_agent,
        "to": to,
        "content": content,
        "source": from_agent,
        "timestamp": ts,
        "msg_id": msg_id,
        "data": {"content": content},
        "metadata": {
            "msg_id": msg_id,
            "timestamp": ts,
            "source": from_agent,
            "source_type": source_type,
            "source_name": source_name,
            "target": to,
            "target_type": target_type,
            "target_name": target_name,
        },
        "payload": {"content": content},
    }

    if to.startswith("agent:"):
        base_msg["conversation_id"] = "default"
        forwarded = await router.forward_to_agent(to, base_msg)
        return {"code": 200, "message": "sent", "msg_id": msg_id, "forwarded": forwarded}

    if to.startswith("user:"):
        base_msg["type"] = "message"
        ws_server.push_to_user(to[5:], base_msg)
    elif to == "broadcast:agents":
        await router.broadcast_to_agents(base_msg)
    elif to == "broadcast:users":
        base_msg["type"] = "message"
        ws_server.broadcast_to_users(base_msg)
    else:
        base_msg["type"] = "message"
        ws_server.push_to_channel("system", base_msg)

    return {"code": 200, "message": "sent", "msg_id": msg_id}


@app.get("/api/v1/messages")
def get_messages(agent_id: Optional[str] = None,
                 limit: int = 50, offset: int = 0,
                 msg_type: Optional[str] = None):
    messages = db.get_global_messages(agent_id=agent_id, limit=limit, offset=offset,
                                      msg_type=msg_type)
    return {"code": 200, "data": messages}


@app.post("/api/v1/agents/{agent_id}/inject")
async def inject_message(agent_id: str, data: Dict[str, Any]):
    """紧急注入（实验控制）：最高优先级消息，独立于信道寻址。
    agent 侧处理（base_agent.process_input）：清 @wait_for 等待并强制下轮唤醒、
    绕过 mute、排 pending 队首。仅实验员使用，key 校验与 dashboard 一致。
    curl -X POST http://127.0.0.1:8080/api/v1/agents/neutral/inject \
         -H 'Content-Type: application/json' \
         -d '{"key":"0xbf5d36","content":"实验员指令：…"}'
    """
    key = data.get("key", "")
    if not _dash_guard(key):
        return JSONResponse(status_code=401, content={"code": 401, "message": "invalid key"})
    content = data.get("content", "")
    if not content or not content.strip():
        return {"code": 400, "message": "missing content"}
    if not db.get_agent_state(agent_id):
        return {"code": 404, "message": "Agent not found"}
    mid = db.save_global_message(
        from_agent="user:experimenter", to_target=f"agent:{agent_id}",
        content=content, msg_type="text")
    msg = {
        "type": "text",
        "from": "user:experimenter",
        "from_agent": "user:experimenter",
        "to": f"agent:{agent_id}",
        "content": content,
        "source": "experimenter",
        "timestamp": time.time(),
        "msg_id": mid,
        "data": {"content": content},
        "metadata": {
            "msg_id": mid,
            "timestamp": time.time(),
            "source": "user:experimenter",
            "source_type": "user",
            "source_name": "experimenter",
            "target": f"agent:{agent_id}",
            "target_type": "agent",
            "target_name": agent_id,
            "urgent": True,
        },
        "payload": {"content": content},
        "conversation_id": "default",
    }
    forwarded = await router.forward_to_agent(f"agent:{agent_id}", msg,
                                              bypass_absence=True)
    return {"code": 200, "message": "injected", "msg_id": mid, "forwarded": forwarded}


# ── 缺席实验控制（2026-08-30 拍定；ABSENCE_EXPERIMENT.md §2.4）──────────

@app.post("/api/v1/experiment/absence")
async def exp_absence(data: Dict[str, Any]):
    """将 agent 置为缺席。mode:
    - alone：进程活着但被路由隔离（B/C→A 不投递+积压限流；A→B/C 改写 private）。
    - halt ：进程停止 + 路由隔离（期间收到消息无效，不积压不投递）。
    curl -X POST http://127.0.0.1:8080/api/v1/experiment/absence \
         -H 'Content-Type: application/json' \
         -d '{"key":"0xbf5d36","agent_id":"lan","mode":"alone"}'
    """
    key = data.get("key", "")
    if not _dash_guard(key):
        return JSONResponse(status_code=401, content={"code": 401, "message": "invalid key"})
    agent_id = data.get("agent_id", "")
    mode = data.get("mode", "alone")
    if not agent_id:
        return {"code": 400, "message": "missing agent_id"}
    if mode not in ("alone", "halt"):
        return {"code": 400, "message": "mode must be alone|halt"}
    if not db.get_agent_state(agent_id):
        return {"code": 404, "message": "Agent not found"}
    router.absence[agent_id] = {"mode": mode, "since": time.time()}
    # 只记日志不广播事件：alone 缺席对留守者必须是"静默的"（B/C 自然观察到
    # 缺席者不回应，而非被告知"缺席开始"——否则冷/热操纵被系统公告污染）。
    # halt 模式的 agent_offline 事件由平台自然产生（进程真停），属可接受差异。
    db.add_log("INFO", "Experiment", f"Agent {agent_id} 缺席开始（mode={mode}）")
    if mode == "halt":
        pm.stop_agent(agent_id, level=1)
        # 确认进程退出（软关可能延迟）；SIGTERM 兜底
        import threading as _th
        def _ensure_halt():
            time.sleep(8)
            try:
                st = db.get_agent_state(agent_id)
                if st and st.get("status") == "online":
                    pm.stop_agent(agent_id, level=3)
            except Exception:
                pass
        _th.Thread(target=_ensure_halt, daemon=True).start()
    return {"code": 200, "message": f"agent {agent_id} absent (mode={mode})"}


@app.post("/api/v1/experiment/return")
async def exp_return(data: Dict[str, Any]):
    """缺席结束：解除隔离；halt 模式重启进程；投递积压（alone 模式 ≤10+聚合）。
    notice=true 时向回归 agent 注入"你已离开约X分钟"元信息（系统状态通道）。
    """
    key = data.get("key", "")
    if not _dash_guard(key):
        return JSONResponse(status_code=401, content={"code": 401, "message": "invalid key"})
    agent_id = data.get("agent_id", "")
    notice = bool(data.get("notice", True))
    if not agent_id:
        return {"code": 400, "message": "missing agent_id"}
    entry = router.absence.pop(agent_id, None)
    if entry is None:
        return {"code": 400, "message": f"agent {agent_id} not absent"}
    mode = entry.get("mode", "alone")
    away_s = int(time.time() - entry.get("since", time.time()))
    away_min = max(1, round(away_s / 60.0))
    db.add_log("INFO", "Experiment",
               f"Agent {agent_id} 回归（缺席 {away_min} 分钟，mode={mode}）")

    if mode == "halt":
        pm.start_agent(agent_id)
        await asyncio.sleep(6)   # 等 agent 注册上线
    else:
        await asyncio.sleep(2)   # 等解隔离的投递通道就绪

    if notice:
        # 回归元信息先于积压投递（回归首轮先注入"你已离开X分钟"，再消化积压——
        # 防重锚定退化重放，next_exp_insts 陷阱 3）
        content = (f"【系统】你已离开约 {away_min} 分钟。这段期间的消息投递已"
                   f"为你积压（部分已聚合），现在恢复正常。")
        mid = db.save_global_message(from_agent="system", to_target=f"agent:{agent_id}",
                                     content=content, msg_type="text")
        notice_msg = {
            "type": "message", "from": "system", "from_agent": "system",
            "to": f"agent:{agent_id}", "content": content, "source": "system",
            "timestamp": time.time(), "msg_id": mid,
            "data": {"content": content},
            "metadata": {"source": "system", "source_type": "system",
                         "source_name": "system", "msg_id": mid},
        }
        await router.forward_to_agent(f"agent:{agent_id}", notice_msg,
                                      bypass_absence=True)

    delivered = await router.flush_absence_inbox(agent_id, mode=mode)
    if notice:
        db.add_log("INFO", "Experiment", f"{agent_id} 回归，投递积压 {delivered} 条")
    return {"code": 200, "message": "returned", "mode": mode,
            "away_minutes": away_min, "inbox_delivered": delivered}


@app.get("/api/v1/experiment/absence")
def exp_absence_state(key: str = ""):
    if not _dash_guard(key):
        return JSONResponse(status_code=401, content={"code": 401, "message": "invalid key"})
    return {"code": 200, "data": {
        k: {"mode": v.get("mode"), "since": v.get("since")}
        for k, v in router.absence.items()
    }}


# ── Agent Management ──

@app.get("/api/v1/agents")
def get_agents():
    agents = db.get_all_agents()
    return {"code": 200, "data": agents}


@app.get("/api/v1/agents/online")
def get_online_agents():
    agents = pm.get_online_agents()
    return {"code": 200, "data": agents}


@app.get("/api/v1/agents/{agent_id}")
def get_agent(agent_id: str):
    info = pm.get_agent_info(agent_id)
    if info:
        return {"code": 200, "data": info}
    return {"code": 404, "message": "Agent not found"}


@app.post("/api/v1/agents/{agent_id}/start")
def start_agent(agent_id: str):
    success = pm.start_agent(agent_id)
    if success:
        return {"code": 200, "message": "Agent starting"}
    return {"code": 500, "message": "Failed to start agent"}


@app.post("/api/v1/agents/{agent_id}/stop")
def stop_agent(agent_id: str, level: int = 1):
    success = pm.stop_agent(agent_id, level=level)
    if success:
        return {"code": 200, "message": "Agent stopped"}
    return {"code": 500, "message": "Failed to stop agent"}


# ── Proxy helper ──

def _proxy_result(result: Optional[dict], error_msg: str = "Proxy request failed"):
    """Map agent proxy result to proper HTTP response.
    - None → 502 Bad Gateway
    - result.code >= 400 → that HTTP status
    - otherwise → 200 with result body
    """
    if result is None:
        return JSONResponse(status_code=502, content={"code": 502, "message": error_msg})
    code = result.get("code", 200)
    if isinstance(code, int) and code >= 400:
        return JSONResponse(status_code=code, content=result)
    return result


@app.post("/api/v1/agents/{agent_id}/shell/reset")
def shell_reset(agent_id: str):
    agent = pm.get_agent(agent_id)
    if not agent:
        return {"code": 404, "message": "Agent not found"}
    state = db.get_agent_state(agent_id)
    if not state or state.get("status") != "online":
        return {"code": 400, "message": "Agent not online"}
    port = state.get("port", 0)
    result = pm._http_post(f"http://127.0.0.1:{port}/api/shell/reset", timeout=5)
    return _proxy_result(result, "Shell reset failed")


@app.post("/api/v1/agents/{agent_id}/loop/start")
def loop_start(agent_id: str):
    agent = pm.get_agent(agent_id)
    if not agent:
        return {"code": 404, "message": "Agent not found"}
    state = db.get_agent_state(agent_id)
    if not state or state.get("status") != "online":
        return {"code": 400, "message": "Agent not online"}
    port = state.get("port", 0)
    result = pm._http_post(f"http://127.0.0.1:{port}/api/loop/start", timeout=5)
    return _proxy_result(result, "Loop start failed")


@app.post("/api/v1/agents/{agent_id}/loop/pause")
def loop_pause(agent_id: str):
    agent = pm.get_agent(agent_id)
    if not agent:
        return {"code": 404, "message": "Agent not found"}
    state = db.get_agent_state(agent_id)
    if not state or state.get("status") != "online":
        return {"code": 400, "message": "Agent not online"}
    port = state.get("port", 0)
    result = pm._http_post(f"http://127.0.0.1:{port}/api/loop/pause", timeout=5)
    return _proxy_result(result, "Loop pause failed")


@app.post("/api/v1/agents/{agent_id}/loop/resume")
def loop_resume(agent_id: str):
    agent = pm.get_agent(agent_id)
    if not agent:
        return {"code": 404, "message": "Agent not found"}
    state = db.get_agent_state(agent_id)
    if not state or state.get("status") != "online":
        return {"code": 400, "message": "Agent not online"}
    port = state.get("port", 0)
    result = pm._http_post(f"http://127.0.0.1:{port}/api/loop/resume", timeout=5)
    return _proxy_result(result, "Loop resume failed")


@app.post("/api/v1/agents/{agent_id}/loop/interval")
def loop_interval(agent_id: str):
    agent = pm.get_agent(agent_id)
    if not agent:
        return {"code": 404, "message": "Agent not found"}
    state = db.get_agent_state(agent_id)
    if not state or state.get("status") != "online":
        return {"code": 400, "message": "Agent not online"}
    port = state.get("port", 0)
    result = pm._http_get(f"http://127.0.0.1:{port}/api/loop/interval", timeout=5)
    return _proxy_result(result, "Interval query failed")


@app.post("/api/v1/system/emergency-stop")
def emergency_stop():
    pm.emergency_stop_all()
    return {"code": 200, "message": "Emergency stop initiated"}


@app.get("/api/v1/system/status")
def get_status():
    agent_count = pm.get_agent_count()
    # 读取主进程资源占用
    try:
        import psutil
        p = psutil.Process(os.getpid())
        cpu = p.cpu_percent(interval=0.1)
        mem = p.memory_info().rss / (1024 * 1024)
    except Exception:
        cpu = 0.0
        mem = 0.0

    return {
        "code": 200,
        "data": {
            "uptime": time.time() - start_time,
            "agents": {
                "total": len(pm.get_online_agents()),
                "process_count": agent_count,
            },
            "resources": {
                "cpu_percent": cpu,
                "memory_mb": round(mem, 1),
            },
            "tunnel": {
                "enabled": config.get("tunnel_enabled", False),
                "connected": _tunnel_connected,
                "url": _tunnel_url,
            },
        },
    }


@app.get("/api/v1/system/logs")
@app.get("/api/v1/logs")
def get_logs(limit: int = 100, offset: int = 0):
    logs = db.get_recent_logs(limit=limit)
    return {"code": 200, "data": logs}


@app.get("/api/v1/system/snapshot")
def get_system_snapshot(since: int = 0):
    """Agent 系统状态注入源（决策：全量快照 + 增量事件）。

    since = 客户端已消费的最后一个事件 id。返回：
    - agents: 在线 Agent（含 waiting/muted 状态）
    - users: 在线用户
    - events: (since, 当前] 的新事件（增量）
    - last_event_id: 本次返回后客户端应持有的游标
    """
    agents = []
    for a in db.get_online_agents():
        agents.append({
            "agent_id": a.get("agent_id"),
            "agent_name": a.get("agent_name"),
            "status": a.get("status", "online"),
            "waiting": a.get("waiting", "") or "",
            "muted": (a.get("muted") or "") or "",
        })
    users = db.get_online_users()
    events = db.get_events_since(since)
    last_id = db.get_last_event_id()
    return {
        "code": 200,
        "data": {
            "agents": agents,
            "users": [{"user_id": u["user_id"], "name": u["name"] or u["user_id"]}
                      for u in users],
            "events": [{"id": e["id"], "ev_type": e["ev_type"],
                        "detail": e["detail"], "ts": e["timestamp"]}
                       for e in events],
            "last_event_id": last_id,
        },
    }


# ── 实验 Dashboard（实时观察控制台，密码与管理员编号一致）──────────

DASH_KEY = "0xbf5d36"


def _dash_guard(key: str):
    return (key or "") == DASH_KEY


def _read_metrics(agent_id):
    """读 agent 的 metrics.jsonl（主进程与 agent 同机）。"""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "agents", agent_id, "data", "metrics.jsonl")
    rows = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
    except FileNotFoundError:
        pass
    return rows


def _agent_state(agent_id, port):
    """HTTP 拉取 agent 进程实时状态（同机）。失败返回 None。"""
    try:
        import urllib.request
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/state", timeout=2) as resp:
            d = json.loads(resp.read().decode())
            return d.get("data") or {}
    except Exception:
        return None


def _compose_state(aid: str) -> dict:
    """上下文组成（2026-08-29 加入 dashboard）：读 agent 最新树快照 + 组装快照，
    复算注意力窗口成员（原子段 40 / 节点段 20，HP≥45 排名），返回组成占比。"""
    import json as _json
    import re as _re
    base = os.path.dirname(os.path.abspath(__file__))
    out = {"tile_atoms": 0, "win_atoms": 0, "tile_nodes": 0, "win_nodes": 0,
           "hp45_atoms": 0, "hp45_nodes": 0, "ctx_atoms": 0, "ctx_nodes": 0,
           "ctx_shadows": 0, "ctx_chars": 0}
    try:
        p = os.path.join(base, "agents", aid, "data", "tree_snapshots.jsonl")
        snaps = [_json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
        if not snaps:
            return out
        s = snaps[-1]
        atoms, nodes, tiling = s["atoms"], s["nodes"], s["tiling"]
        tile_atoms = [x for x in tiling if str(x) in atoms]
        tile_nodes = [x for x in tiling if str(x) in nodes]
        cand = [x for x in tile_atoms if atoms[str(x)].get("hp", 0) >= 45]
        cand.sort(key=lambda x: atoms[str(x)].get("hp", 0), reverse=True)
        used, win_atoms = 0, 0
        for x in cand:
            w = atoms[str(x)].get("weight", 1)
            if used + w > 40:
                break
            win_atoms += 1
            used += w
        candn = [x for x in tile_nodes if nodes[str(x)].get("hp", 0) >= 45]
        candn.sort(key=lambda x: nodes[str(x)].get("hp", 0), reverse=True)
        out.update(tile_atoms=len(tile_atoms), win_atoms=win_atoms,
                   tile_nodes=len(tile_nodes), win_nodes=min(20, len(candn)),
                   hp45_atoms=len(cand), hp45_nodes=len(candn))
        # 组装流组成（最新组装快照）
        ap = os.path.join(base, "agents", aid, "data", "assembly_snapshots.jsonl")
        arows = [_json.loads(l) for l in open(ap, encoding="utf-8") if l.strip()]
        if arows:
            ctx = arows[-1].get("ctx", "")
            out.update(ctx_atoms=len(_re.findall(r"\[#\d+·原文", ctx)),
                       ctx_nodes=len(_re.findall(r"\[N\d+·摘要", ctx)),
                       ctx_shadows=len(_re.findall(r"召回", ctx)),
                       ctx_chars=len(ctx))
    except Exception:
        pass
    return out


@app.get("/api/v1/dashboard/overview")
def dash_overview(key: str = ""):
    if not _dash_guard(key):
        return JSONResponse(status_code=401, content={"code": 401, "message": "invalid key"})
    agents = []
    for a in db.get_online_agents():
        aid = a.get("agent_id")
        port = a.get("port", 0)
        st = _agent_state(aid, port) if port else {}
        rows = _read_metrics(aid)
        last = rows[-1] if rows else None
        agents.append({
            "agent_id": aid,
            "agent_name": a.get("agent_name"),
            "status": st.get("status") or a.get("status", "online"),
            "wakeup": st.get("wakeup_count"),
            "pending": st.get("pending_messages"),
            "waiting": st.get("waiting", ""),
            "muted": st.get("muted", []),
            "last": last,
            "compose": _compose_state(aid) if aid in ("lan", "neutral") else None,
        })
    msgs_total = db.get_last_event_id() and _count_messages()
    return {"code": 200, "data": {
        "agents": agents,
        "messages_total": msgs_total,
        "ts": time.time(),
    }}


def _count_messages():
    conn = db._get_conn()
    try:
        return conn.execute("SELECT COUNT(*) AS c FROM messages").fetchone()["c"]
    except Exception:
        return 0
    finally:
        conn.close()


@app.get("/api/v1/dashboard/series")
def dash_series(key: str = ""):
    """历史序列：各 agent 的 metrics 精简序列（前端画图）。"""
    if not _dash_guard(key):
        return JSONResponse(status_code=401, content={"code": 401, "message": "invalid key"})
    out = {}
    # 只列实验 agent（防历史残留如 mo-bai 出现在 series/logs）
    for a in db.get_all_agents():
        aid = a.get("agent_id")
        if aid not in ("lan", "neutral", "lin-shen"):
            continue
        rows = _read_metrics(aid)
        series = []
        for r in rows:
            series.append({
                "r": r.get("round"),
                "atoms": r.get("atoms_alive"),
                "nodes": r.get("nodes_alive"),
                "tiling": r.get("tiling"),
                "ctx": r.get("ctx_chars"),
                "pending": r.get("pending"),
                "persona": r.get("persona"),
                "dc": r.get("distill_calls"),
                "ds": round(r.get("distill_seconds", 0), 1),
                "lc": r.get("llm_calls"),
                "snap": r.get("snapshot_fields"),
                "drift": r.get("fact_drift"),
            })
        out[aid] = series
    return {"code": 200, "data": out}


@app.get("/api/v1/dashboard/logs")
def dash_logs(key: str = "", limit: int = 40):
    """关键日志：系统日志 + 事件流 + 各 agent 日志文件尾部。"""
    if not _dash_guard(key):
        return JSONResponse(status_code=401, content={"code": 401, "message": "invalid key"})
    sys_logs = db.get_recent_logs(limit=limit)
    events = db.get_events_since(max(0, db.get_last_event_id() - limit))
    agent_logs = {}
    base = os.path.dirname(os.path.abspath(__file__))
    for a in db.get_all_agents():
        aid = a.get("agent_id")
        if aid not in ("lan", "neutral", "lin-shen"):
            continue
        try:
            path = os.path.join(base, "agents", aid, "data", f"agent-{aid}.log")
            with open(path, encoding="utf-8", errors="replace") as f:
                agent_logs[aid] = f.readlines()[-15:]
        except Exception:
            agent_logs[aid] = []
    return {"code": 200, "data": {
        "system_logs": sys_logs,
        "events": [{"id": e["id"], "ev_type": e["ev_type"], "detail": e["detail"],
                    "ts": e["timestamp"]} for e in events],
        "agent_logs": agent_logs,
    }}


@app.get("/health")
def health():
    return {"status": "ok", "time": time.time()}


# ── Static Files ──

frontend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")
if os.path.isdir(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
    logger.info("Frontend static files mounted from %s", frontend_dir)
else:
    logger.warning("Frontend directory not found at %s", frontend_dir)


# ── Entry ──

if __name__ == "__main__":
    import uvicorn
    port = config.get("main_process_port", 8000)
    logger.info("Starting AHI-Multi main process on port %d...", port)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
