import sqlite3
import os
import uuid
import json
from datetime import datetime
from typing import Dict, Any, Optional, List


class SystemDB:
    def __init__(self, db_path: str = None):
        if db_path is None:
            db_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
            db_path = os.path.join(db_dir, "system.db")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self.on_log = None  # 可选回调: fn(level, source, message)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self):
        conn = self._get_conn()
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS agents (
                    agent_id TEXT PRIMARY KEY,
                    agent_name TEXT NOT NULL,
                    agent_type TEXT DEFAULT 'llm',
                    description TEXT DEFAULT '',
                    entry_point TEXT NOT NULL,
                    shell_type TEXT DEFAULT 'python',
                    config_json TEXT DEFAULT '{}',
                    auto_start INTEGER DEFAULT 1,
                    auto_restart INTEGER DEFAULT 1,
                    max_memory_mb INTEGER DEFAULT 512,
                    wakeup_interval INTEGER DEFAULT 10,
                    created_at TEXT DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS online_states (
                    agent_id TEXT PRIMARY KEY,
                    pid INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'offline',
                    port INTEGER DEFAULT 0,
                    cpu_usage REAL DEFAULT 0.0,
                    memory_usage REAL DEFAULT 0.0,
                    last_heartbeat TEXT DEFAULT (datetime('now')),
                    FOREIGN KEY (agent_id) REFERENCES agents(agent_id)
                );

                CREATE TABLE IF NOT EXISTS system_logs (
                    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    level TEXT NOT NULL DEFAULT 'INFO',
                    source TEXT NOT NULL,
                    message TEXT NOT NULL,
                    timestamp TEXT DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    msg_id TEXT UNIQUE NOT NULL,
                    from_agent TEXT NOT NULL,
                    to_target TEXT NOT NULL DEFAULT '',
                    content TEXT NOT NULL DEFAULT '',
                    msg_type TEXT NOT NULL DEFAULT 'text',
                    timestamp TEXT DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    name TEXT DEFAULT '',
                    online INTEGER DEFAULT 0,
                    last_seen TEXT DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS system_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ev_type TEXT NOT NULL,
                    detail TEXT NOT NULL DEFAULT '',
                    timestamp TEXT DEFAULT (datetime('now'))
                );
            """)
            conn.commit()
            # 迁移：online_states 增加 waiting/muted 列（老库兼容）
            try:
                cols = [r["name"] for r in conn.execute("PRAGMA table_info(online_states)").fetchall()]
                if "waiting" not in cols:
                    conn.execute("ALTER TABLE online_states ADD COLUMN waiting TEXT DEFAULT ''")
                if "muted" not in cols:
                    conn.execute("ALTER TABLE online_states ADD COLUMN muted TEXT DEFAULT ''")
                conn.commit()
            except Exception:
                pass
        finally:
            conn.close()

    # ── Agent 配置管理 ──

    def save_agent(self, config: Dict[str, Any]) -> bool:
        conn = self._get_conn()
        try:
            conn.execute("""
                INSERT OR REPLACE INTO agents
                    (agent_id, agent_name, agent_type, description, entry_point, shell_type,
                     config_json, auto_start, auto_restart, max_memory_mb, wakeup_interval)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                config.get("agent_id"),
                config.get("agent_name"),
                config.get("agent_type", "llm"),
                config.get("description", ""),
                config.get("entry_point", "agent:Agent"),
                config.get("shell_type", "python"),
                json.dumps(config, ensure_ascii=False),
                1 if config.get("auto_start", True) else 0,
                1 if config.get("auto_restart", True) else 0,
                config.get("max_memory_mb", 512),
                config.get("wakeup_interval", 10),
            ))
            conn.commit()
            return True
        except Exception as e:
            self.add_log("ERROR", "SystemDB", f"save_agent failed: {e}")
            return False
        finally:
            conn.close()

    def get_agent(self, agent_id: str) -> Optional[Dict[str, Any]]:
        conn = self._get_conn()
        try:
            cursor = conn.execute("SELECT * FROM agents WHERE agent_id = ?", (agent_id,))
            row = cursor.fetchone()
            if row:
                d = dict(row)
                d["config"] = json.loads(d.pop("config_json", "{}"))
                return d
            return None
        finally:
            conn.close()

    def get_all_agents(self) -> List[Dict[str, Any]]:
        conn = self._get_conn()
        try:
            cursor = conn.execute("SELECT * FROM agents ORDER BY created_at")
            results = []
            for row in cursor.fetchall():
                d = dict(row)
                d["config"] = json.loads(d.pop("config_json", "{}"))
                results.append(d)
            return results
        finally:
            conn.close()

    def delete_agent(self, agent_id: str) -> bool:
        conn = self._get_conn()
        try:
            conn.execute("DELETE FROM agents WHERE agent_id = ?", (agent_id,))
            conn.execute("DELETE FROM online_states WHERE agent_id = ?", (agent_id,))
            conn.commit()
            return True
        finally:
            conn.close()

    # ── 在线状态管理 ──

    def update_state(self, agent_id: str, state: Dict[str, Any]) -> bool:
        conn = self._get_conn()
        try:
            conn.execute("""
                INSERT OR REPLACE INTO online_states
                    (agent_id, pid, status, port, cpu_usage, memory_usage, waiting, muted, last_heartbeat)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
            """, (
                agent_id,
                state.get("pid", 0),
                state.get("status", "online"),
                state.get("port", 0),
                state.get("cpu_usage", 0.0),
                state.get("memory_usage", 0.0),
                state.get("waiting", ""),
                state.get("muted", ""),
            ))
            conn.commit()
            return True
        except Exception as e:
            self.add_log("ERROR", "SystemDB", f"update_state failed: {e}")
            return False
        finally:
            conn.close()

    def set_agent_offline(self, agent_id: str):
        conn = self._get_conn()
        try:
            conn.execute(
                "UPDATE online_states SET status='offline', last_heartbeat=datetime('now') WHERE agent_id=?",
                (agent_id,)
            )
            conn.commit()
        finally:
            conn.close()

    def get_online_agents(self) -> List[Dict[str, Any]]:
        conn = self._get_conn()
        try:
            cursor = conn.execute("""
                SELECT a.*, s.pid, s.status, s.port, s.cpu_usage, s.memory_usage,
                       s.waiting, s.muted, s.last_heartbeat
                FROM agents a
                INNER JOIN online_states s ON a.agent_id = s.agent_id
                WHERE s.status = 'online'
                ORDER BY a.created_at
            """)
            results = []
            for row in cursor.fetchall():
                d = dict(row)
                d["config"] = json.loads(d.pop("config_json", "{}"))
                results.append(d)
            return results
        finally:
            conn.close()

    def get_agent_state(self, agent_id: str) -> Optional[Dict[str, Any]]:
        conn = self._get_conn()
        try:
            cursor = conn.execute("""
                SELECT a.*, s.pid, s.status, s.port, s.cpu_usage, s.memory_usage,
                       s.waiting, s.muted, s.last_heartbeat
                FROM agents a
                LEFT JOIN online_states s ON a.agent_id = s.agent_id
                WHERE a.agent_id = ?
            """, (agent_id,))
            row = cursor.fetchone()
            if row:
                d = dict(row)
                d["config"] = json.loads(d.pop("config_json", "{}"))
                return d
            return None
        finally:
            conn.close()

    # ── 用户在线状态（决策：用户上下线事件）──────────

    def set_user_online(self, user_id: str, name: str = ""):
        conn = self._get_conn()
        try:
            conn.execute("""
                INSERT OR REPLACE INTO users (user_id, name, online, last_seen)
                VALUES (?, ?, 1, datetime('now'))
            """, (user_id, name or user_id))
            conn.commit()
        finally:
            conn.close()

    def set_user_offline(self, user_id: str):
        conn = self._get_conn()
        try:
            conn.execute(
                "UPDATE users SET online=0, last_seen=datetime('now') WHERE user_id=?",
                (user_id,)
            )
            conn.commit()
        finally:
            conn.close()

    def get_online_users(self) -> List[Dict[str, Any]]:
        conn = self._get_conn()
        try:
            cursor = conn.execute(
                "SELECT * FROM users WHERE online=1 ORDER BY last_seen DESC")
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    # ── 系统事件流（决策：增量事件，供 agent 系统状态注入）──

    def add_event(self, ev_type: str, detail: str) -> int:
        conn = self._get_conn()
        try:
            cursor = conn.execute(
                "INSERT INTO system_events (ev_type, detail) VALUES (?, ?)",
                (ev_type, detail)
            )
            conn.commit()
            return cursor.lastrowid
        except Exception:
            return 0
        finally:
            conn.close()

    def get_events_since(self, last_id: int = 0, limit: int = 30) -> List[Dict[str, Any]]:
        conn = self._get_conn()
        try:
            cursor = conn.execute(
                "SELECT * FROM system_events WHERE id > ? ORDER BY id ASC LIMIT ?",
                (last_id, limit)
            )
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def get_last_event_id(self) -> int:
        conn = self._get_conn()
        try:
            cursor = conn.execute("SELECT COALESCE(MAX(id), 0) AS m FROM system_events")
            return cursor.fetchone()["m"]
        finally:
            conn.close()

    # ── 系统日志 ──

    def add_log(self, level: str, source: str, message: str) -> bool:
        conn = self._get_conn()
        try:
            conn.execute(
                "INSERT INTO system_logs (level, source, message) VALUES (?, ?, ?)",
                (level.upper(), source, message)
            )
            conn.commit()
            if self.on_log:
                try:
                    self.on_log(level, source, message)
                except Exception:
                    pass
            return True
        except Exception:
            return False
        finally:
            conn.close()

    def get_recent_logs(self, limit: int = 100) -> List[Dict[str, Any]]:
        conn = self._get_conn()
        try:
            cursor = conn.execute(
                "SELECT * FROM system_logs ORDER BY timestamp DESC LIMIT ?", (limit,)
            )
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    # ── 全局消息存档 ──

    def save_global_message(self, from_agent: str, to_target: str = "",
                            content: str = "", msg_type: str = "text") -> Optional[str]:
        conn = self._get_conn()
        try:
            msg_id = uuid.uuid4().hex
            conn.execute(
                "INSERT INTO messages (msg_id, from_agent, to_target, content, msg_type) VALUES (?, ?, ?, ?, ?)",
                (msg_id, from_agent, to_target, content, msg_type)
            )
            conn.commit()
            return msg_id
        except Exception as e:
            self.add_log("ERROR", "SystemDB", f"save_global_message failed: {e}")
            return None
        finally:
            conn.close()

    def get_global_messages(self, agent_id: Optional[str] = None,
                            limit: int = 50, offset: int = 0,
                            msg_type: Optional[str] = None) -> List[Dict[str, Any]]:
        conn = self._get_conn()
        try:
            type_filter = ""
            params = []
            if agent_id:
                agent_target = f"agent:{agent_id}"
                type_filter = "AND msg_type = ?" if msg_type else ""
                query = f"""SELECT * FROM messages
                       WHERE (from_agent IN (?, ?) OR to_target IN (?, ?))
                       {type_filter}
                       ORDER BY timestamp DESC LIMIT ? OFFSET ?"""
                params = [agent_id, agent_target, agent_id, agent_target]
            else:
                type_filter = "AND msg_type = ?" if msg_type else ""
                query = f"""SELECT * FROM messages
                       WHERE 1=1 {type_filter}
                       ORDER BY timestamp DESC LIMIT ? OFFSET ?"""
                params = []
            if msg_type:
                params.append(msg_type)
            params.extend([limit, offset])
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()
