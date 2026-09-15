# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AHI-Multi v3.0 is a multi-agent "digital life" platform. Each AI agent runs as an **independent OS process** with its own SQLite database, persistent Python shell, and autonomous wake-up loop. The main process does not control agent behavior — it provides the "living environment" (lifecycle management, message routing, WebSocket push).

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Start the main process (FastAPI on :8000 + WebSocket /ws)
python main.py

# Start a specific agent via API
curl -X POST http://localhost:8000/api/v1/agents/<agent_id>/start

# Stop an agent (level: 1=graceful, 2=SIGTERM, 3=SIGKILL)
curl -X POST "http://localhost:8000/api/v1/agents/<agent_id>/stop?level=1"
```

There is no test suite or lint configuration in this project.

## Architecture

### Process Model

```
main.py (FastAPI :8000)
  ├── ProcessManager   — spawn/monitor/restart agent processes, heartbeat at 2s interval
  ├── MessageRouter    — routes messages between agents/users via three-level addressing
  ├── WebSocketServer  — channel subscription + per-user push
  └── SystemDB         — global SQLite (agent registry, online states, message archive, logs)

Agent Process (independent, port 50000+)
  └── sdk/agent_runner.py  — entry point, loads agent class, injects dependencies
       ├── BaseAHIAgent subclass  — LLM-driven autonomous loop
       ├── PythonShell           — subprocess sandbox, namespace persists across wakeups
       ├── AgentDB               — per-agent SQLite (messages, code_executions, loops)
       ├── LLMClient             — OpenAI-compatible client (Ollama/DeepSeek/Groq/OpenAI)
       └── AHIBus                — HTTP client to main process (register, notify, send)
```

### Message Flow

1. Agent produces output via `_put_action()` → pushes to internal queue → notifies main process via AHIBus
2. Main process (MessageRouter) fetches outputs from agent's `/api/outputs`, routes each action by type and target
3. Targets use three-level addressing: `user:<name>`, `agent:<id>`, `broadcast:agents`, `broadcast:users`
4. WebSocket channels: `system`, `chat:agent:<id>`, per-user push

### Key Abstractions

- **`BaseAHIAgent`** (`sdk/base_agent.py:8`): Abstract base with 3 required methods — `process_input()`, `get_outputs()`, `get_state()`. Optional hooks: `on_start()`, `on_stop()`, `on_wakeup()`. Built-in `_put_action()` enqueues output and notifies main process.
- **`AgentRunner`** (`sdk/agent_runner.py:28`): Standardized container. Loads agent class from `entry_point` (e.g. `agent:LinShenAgent`), injects shell/db/bus/llm_client, starts APScheduler-based autonomous loop.
- **`PythonShell`** (`sdk/base_shell.py:186`): Executes code in a `multiprocessing.Process` worker. Variables persist across commands. Uses `ThreadPoolExecutor` for timeout control.
- **`LLMClient`** (`sdk/llm_client.py:7`): Wraps OpenAI SDK. Supports any OpenAI-compatible endpoint. Auto-compresses context when messages exceed 120 (keeps system prompt + last 40, summarizes removed middle).

### Agent Creation Pattern

1. Create `agents/<id>/config.json` with `agent_id`, `entry_point` (`agent:ClassName`), `model_config`, `system_prompt`, `wakeup_interval`
2. Create `agents/<id>/agent.py` with a class inheriting `BaseAHIAgent`
3. The built-in agents (LinShen, MoBai) implement a full autonomous loop in `on_wakeup()`: drain pending messages → build wakeup prompt → LLM-driven loop (up to 8 iterations) with structured response parsing (`@send-to`, code blocks, `@exit`)

### Agent Structured Response Format

Agents communicate with the LLM using a structured format parsed by `_parse_structured_response()`:
```
@send-to "user:human" "agent:mo-bai"
```txt
Message content (Markdown)
```
```python
print("code execution")
```
@exit
```

Without `@send-to`, content is broadcast to all agents and users.

### Configuration

- **Global** (`config.json`): ports, heartbeat intervals, Cloudflare Tunnel settings
- **Per-Agent** (`agents/*/config.json`): LLM endpoint/model, system prompt (personality definition), wakeup interval (5-3600s), auto_start, auto_restart (max 3 retries), max_memory_mb
- Agent configs contain API keys and are gitignored (`agents/*/config.json`). Templates in `config.example.json`.
