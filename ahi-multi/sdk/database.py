import sqlite3
import os
import time
import uuid
from typing import Dict, Any, Optional, List


class AgentDB:
    """Per-agent independent SQLite database."""

    def __init__(self, db_path: str):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self):
        conn = self._get_conn()
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    message_id TEXT UNIQUE,
                    conversation_id TEXT DEFAULT 'default',
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT DEFAULT '{}',
                    timestamp TEXT DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS conversations (
                    conversation_id TEXT PRIMARY KEY,
                    name TEXT DEFAULT '',
                    agent_id TEXT DEFAULT '',
                    created_at TEXT DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS code_executions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    exec_id TEXT UNIQUE,
                    command TEXT NOT NULL,
                    result TEXT,
                    status TEXT DEFAULT 'pending',
                    error TEXT,
                    started_at TEXT DEFAULT (datetime('now')),
                    completed_at TEXT
                );

                CREATE TABLE IF NOT EXISTS command_executions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    exec_id TEXT UNIQUE,
                    command TEXT NOT NULL,
                    result TEXT,
                    status TEXT DEFAULT 'pending',
                    error TEXT,
                    started_at TEXT DEFAULT (datetime('now')),
                    completed_at TEXT
                );

                CREATE TABLE IF NOT EXISTS agent_loops (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_id TEXT NOT NULL,
                    trigger TEXT DEFAULT 'scheduled',
                    result TEXT,
                    error TEXT,
                    started_at TEXT DEFAULT (datetime('now')),
                    completed_at TEXT
                );

                CREATE TABLE IF NOT EXISTS shell_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_id TEXT NOT NULL,
                    variables TEXT,
                    state_json TEXT,
                    created_at TEXT DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL DEFAULT '',
                    updated_at TEXT DEFAULT (datetime('now'))
                );
            """)
            conn.commit()
        finally:
            conn.close()

    # ── 设置持久化（mute 列表、执行超时等）──────────

    def get_setting(self, key: str) -> str:
        conn = self._get_conn()
        try:
            cursor = conn.execute("SELECT value FROM settings WHERE key = ?", (key,))
            row = cursor.fetchone()
            return row["value"] if row else ""
        finally:
            conn.close()

    def set_setting(self, key: str, value: str):
        conn = self._get_conn()
        try:
            conn.execute("""
                INSERT OR REPLACE INTO settings (key, value, updated_at)
                VALUES (?, ?, datetime('now'))
            """, (key, value))
            conn.commit()
        finally:
            conn.close()

    def create_conversation(self, conversation_id: str, name: str = "",
                            agent_id: str = ""):
        conn = self._get_conn()
        try:
            conn.execute(
                "INSERT OR IGNORE INTO conversations (conversation_id, name, agent_id) VALUES (?, ?, ?)",
                (conversation_id, name, agent_id),
            )
            conn.commit()
        finally:
            conn.close()

    def add_message(self, conversation_id: str, role: str, content: str,
                    metadata: dict = None) -> Optional[str]:
        conn = self._get_conn()
        try:
            import json
            message_id = uuid.uuid4().hex[:12]
            conn.execute(
                "INSERT INTO messages (message_id, conversation_id, role, content, metadata) VALUES (?, ?, ?, ?, ?)",
                (message_id, conversation_id, role, content,
                 json.dumps(metadata or {}, ensure_ascii=False)),
            )
            conn.commit()
            return message_id
        finally:
            conn.close()

    def get_conversation_messages(self, conversation_id: str,
                                  limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        conn = self._get_conn()
        try:
            cursor = conn.execute(
                "SELECT * FROM messages WHERE conversation_id = ? ORDER BY id ASC LIMIT ? OFFSET ?",
                (conversation_id, limit, offset),
            )
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def get_messages(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        conn = self._get_conn()
        try:
            cursor = conn.execute(
                "SELECT * FROM messages ORDER BY id DESC LIMIT ? OFFSET ?",
                (limit, offset),
            )
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def add_code_execution(self, command: str) -> Optional[int]:
        conn = self._get_conn()
        try:
            exec_id = uuid.uuid4().hex[:12]
            cursor = conn.execute(
                "INSERT INTO code_executions (exec_id, command, status) VALUES (?, ?, 'pending')",
                (exec_id, command),
            )
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()

    def update_code_execution(self, exec_id: int, result: str = None,
                              status: str = "completed", error: str = None):
        conn = self._get_conn()
        try:
            conn.execute(
                """UPDATE code_executions
                   SET result=?, status=?, error=?, completed_at=datetime('now')
                   WHERE id=?""",
                (result, status, error, exec_id),
            )
            conn.commit()
        finally:
            conn.close()

    def add_command_execution(self, command: str) -> Optional[int]:
        conn = self._get_conn()
        try:
            exec_id = uuid.uuid4().hex[:12]
            cursor = conn.execute(
                "INSERT INTO command_executions (exec_id, command, status) VALUES (?, ?, 'pending')",
                (exec_id, command),
            )
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()

    def update_command_execution(self, exec_id: int, result: str = None,
                                 status: str = "completed", error: str = None):
        conn = self._get_conn()
        try:
            conn.execute(
                """UPDATE command_executions
                   SET result=?, status=?, error=?, completed_at=datetime('now')
                   WHERE id=?""",
                (result, status, error, exec_id),
            )
            conn.commit()
        finally:
            conn.close()

    def add_loop_record(self, agent_id: str, trigger: str = "scheduled") -> Optional[int]:
        conn = self._get_conn()
        try:
            cursor = conn.execute(
                "INSERT INTO agent_loops (agent_id, trigger) VALUES (?, ?)",
                (agent_id, trigger),
            )
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()

    def complete_loop_record(self, loop_id: int, result: str = None,
                             error: str = None):
        conn = self._get_conn()
        try:
            conn.execute(
                """UPDATE agent_loops
                   SET result=?, error=?, completed_at=datetime('now') WHERE id=?""",
                (result, error, loop_id),
            )
            conn.commit()
        finally:
            conn.close()

    def save_shell_snapshot(self, agent_id: str, variables: list,
                            state_json: str = ""):
        conn = self._get_conn()
        try:
            import json
            conn.execute(
                "INSERT INTO shell_snapshots (agent_id, variables, state_json) VALUES (?, ?, ?)",
                (agent_id, json.dumps(variables), state_json),
            )
            conn.commit()
        finally:
            conn.close()

    def save_message(self, msg: dict) -> None:
        self.add_message(
            conversation_id=msg.get("conversation_id", "default"),
            role=msg.get("role", "user"),
            content=msg.get("content", ""),
            metadata=msg.get("metadata"),
        )

    def save_code_execution(self, exec_data: dict) -> None:
        exec_id = self.add_code_execution(exec_data.get("command", ""))
        if exec_id:
            self.update_code_execution(
                exec_id,
                result=exec_data.get("result"),
                status=exec_data.get("status", "completed"),
                error=exec_data.get("error"),
            )

    def save_command_execution(self, exec_data: dict) -> None:
        exec_id = self.add_command_execution(exec_data.get("command", ""))
        if exec_id:
            self.update_command_execution(
                exec_id,
                result=exec_data.get("result"),
                status=exec_data.get("status", "completed"),
                error=exec_data.get("error"),
            )

    def add_log(self, level: str, source: str, message: str):
        pass
