"""sandbox.py — agent 文件访问的沙箱路径解析（宿主侧）。

agent 的"文件世界" = 沙箱根（宿主机某目录），在容器内挂载为 /workspace。
@ingest 收到的路径按以下规则映射，越界一律拒绝：

- `/workspace/xxx` 或 `/workspace`        → <root>/xxx（容器绝对路径）
- 相对路径 `xxx`                          → <root>/xxx
- 其它绝对路径（/etc/...、/Users/...）    → None（拒绝）
- 解析后 realpath 逃出 root（../ 穿越）    → None（拒绝）

沙箱未启用（root 为空）时按原样返回，保持旧行为。
"""
import os

CONTAINER_ROOT = "/workspace"


def resolve_path(path: str, root: str):
    """把 agent 可见路径解析为宿主沙箱内路径；越界返回 None。"""
    if not root:
        return path
    root = os.path.realpath(root)
    p = (path or "").strip()
    if not p:
        return None

    if p == CONTAINER_ROOT:
        rel = ""
    elif p.startswith(CONTAINER_ROOT + "/"):
        rel = p[len(CONTAINER_ROOT) + 1:]
    elif os.path.isabs(p):
        return None
    else:
        rel = p

    host = os.path.realpath(os.path.join(root, rel))
    if host != root and not host.startswith(root + os.sep):
        return None
    return host


def is_inside(path: str, root: str) -> bool:
    return resolve_path(path, root) is not None
