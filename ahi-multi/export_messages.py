#!/usr/bin/env python3
"""导出 AHI-Multi 平台完整消息日志为 JSON 格式。

从 system.db（全局消息存档）和每个 agent 的独立数据库
中导出所有消息，合并输出为一个 JSON 文件。

用法:
    python export_messages.py                      # 输出到 stdout
    python export_messages.py -o messages.json     # 输出到文件
    python export_messages.py --global-only        # 仅导出全局消息
    python export_messages.py --agents-only        # 仅导出 agent 消息
    python export_messages.py --pretty             # 格式化 JSON 输出
"""

import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime, timezone


def get_project_root():
    return os.path.dirname(os.path.abspath(__file__))


def query_all(conn, table, order_by="id ASC"):
    """查询表中全部记录，返回 dict 列表。"""
    cursor = conn.execute(f"SELECT * FROM {table} ORDER BY {order_by}")
    return [dict(row) for row in cursor.fetchall()]


def export_system_db(db_path):
    """导出系统全局数据库中的所有消息。"""
    if not os.path.exists(db_path):
        return {"error": f"文件不存在: {db_path}", "messages": [], "count": 0}

    conn = sqlite3.connect(db_path, timeout=10.0)
    conn.row_factory = sqlite3.Row
    try:
        messages = query_all(conn, "messages", "id ASC")
        return {"path": db_path, "messages": messages, "count": len(messages)}
    finally:
        conn.close()


def export_agent_db(db_path, agent_id):
    """导出单个 agent 数据库中的所有消息。"""
    if not os.path.exists(db_path):
        return {"agent_id": agent_id, "error": f"文件不存在: {db_path}", "messages": [], "count": 0}

    conn = sqlite3.connect(db_path, timeout=10.0)
    conn.row_factory = sqlite3.Row
    try:
        messages = query_all(conn, "messages", "id ASC")
        return {"agent_id": agent_id, "path": db_path, "messages": messages, "count": len(messages)}
    finally:
        conn.close()


def discover_agent_dbs(project_root):
    """扫描 agents/ 目录，返回所有 agent 数据库路径。"""
    agents_dir = os.path.join(project_root, "agents")
    if not os.path.isdir(agents_dir):
        return []

    result = []
    for agent_id in sorted(os.listdir(agents_dir)):
        agent_path = os.path.join(agents_dir, agent_id)
        if not os.path.isdir(agent_path):
            continue
        db_path = os.path.join(agent_path, "data", f"{agent_id}.db")
        result.append((agent_id, db_path))
    return result


def main():
    parser = argparse.ArgumentParser(description="导出 AHI-Multi 完整消息日志")
    parser.add_argument("-o", "--output", help="输出 JSON 文件路径（默认 stdout）")
    parser.add_argument("--global-only", action="store_true", help="仅导出全局消息")
    parser.add_argument("--agents-only", action="store_true", help="仅导出 agent 消息")
    parser.add_argument("--pretty", action="store_true", help="格式化 JSON 输出（默认压缩）")
    args = parser.parse_args()

    project_root = get_project_root()
    export_global = not args.agents_only
    export_agents = not args.global_only

    result = {
        "export_info": {
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "project_root": project_root,
        },
    }

    total = 0

    # 导出全局消息
    if export_global:
        system_db_path = os.path.join(project_root, "data", "system.db")
        global_data = export_system_db(system_db_path)
        result["global_messages"] = global_data.pop("messages")
        result["export_info"]["system_db"] = global_data
        total += global_data["count"]

    # 导出各 agent 消息
    if export_agents:
        agent_dbs = discover_agent_dbs(project_root)
        agent_exports = []
        for agent_id, db_path in agent_dbs:
            agent_data = export_agent_db(db_path, agent_id)
            result[f"agent_messages_{agent_id}"] = agent_data.pop("messages")
            agent_exports.append(agent_data)
            total += agent_data["count"]
        result["export_info"]["agent_dbs"] = agent_exports

    result["export_info"]["total_messages"] = total

    indent = 2 if args.pretty else None
    json_str = json.dumps(result, ensure_ascii=False, indent=indent)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(json_str)
            f.write("\n")
        print(f"已导出 {total} 条消息到 {args.output}")
    else:
        sys.stdout.write(json_str)
        sys.stdout.write("\n")


if __name__ == "__main__":
    main()
