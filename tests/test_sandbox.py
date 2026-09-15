"""测试：沙箱（@ingest 路径解析 + DockerShell 容器执行）。

1. resolve_path 单元：相对/容器绝对/越界/穿越/未启用
2. DockerShell 集成（需 docker，不可用则跳过）：沙箱读写、命名空间保持、宿主不可见
"""

import os
import subprocess
import sys
import time

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ahi-multi"))

from sdk.sandbox import resolve_path          # noqa: E402
from sdk.base_shell import DockerShell        # noqa: E402


# ── resolve_path 单元 ──

def test_resolve_relative_inside(tmp_path):
    got = resolve_path("a/b.txt", str(tmp_path))
    assert got == os.path.realpath(str(tmp_path / "a" / "b.txt"))


def test_resolve_container_abs(tmp_path):
    got = resolve_path("/workspace/a.txt", str(tmp_path))
    assert got == os.path.realpath(str(tmp_path / "a.txt"))


def test_resolve_workspace_root(tmp_path):
    got = resolve_path("/workspace", str(tmp_path))
    assert got == os.path.realpath(str(tmp_path))


def test_resolve_reject_other_abs(tmp_path):
    assert resolve_path("/etc/passwd", str(tmp_path)) is None


def test_resolve_reject_traversal(tmp_path):
    assert resolve_path("../secret.txt", str(tmp_path)) is None
    assert resolve_path("/workspace/../secret", str(tmp_path)) is None


def test_resolve_disabled_passthrough():
    assert resolve_path("a.txt", "") == "a.txt"


# ── DockerShell 集成 ──

def _docker_ok():
    try:
        r = subprocess.run(["docker", "info"], capture_output=True, timeout=15)
        return r.returncode == 0
    except Exception:
        return False


docker_required = pytest.mark.skipif(not _docker_ok(), reason="docker 不可用")


def _wait(sh, timeout=40):
    deadline = time.time() + timeout
    while time.time() < deadline:
        rs = sh.get_results()
        if rs:
            return rs[-1]
        time.sleep(0.2)
    raise AssertionError("等待 shell 结果超时")


@docker_required
def test_dockershell_isolation_and_namespace(tmp_path):
    root = tmp_path / "sandbox"
    root.mkdir()
    (root / "seed.txt").write_text("受控种子")
    name = f"ahi-test-{os.getpid()}"
    sh = DockerShell(config={"root": str(root), "name": name,
                             "image": "ahi-sandbox:latest"})
    try:
        # 1) 读沙箱内文件
        sh.submit_command("open('seed.txt', encoding='utf-8').read()", timeout=40)
        assert "受控种子" in (_wait(sh).get("result") or "")

        # 2) 写沙箱 → 宿主目录可见
        sh.submit_command("open('out.txt', 'w').write('written')", timeout=40)
        _wait(sh)
        assert (root / "out.txt").read_text() == "written"

        # 3) 命名空间跨命令保持
        sh.submit_command("v = 7", timeout=40)
        _wait(sh)
        sh.submit_command("v * 2", timeout=40)
        assert "14" in (_wait(sh).get("result") or "")

        # 4) 宿主仓库不可见
        sh.submit_command(
            "__import__('os').path.exists('/Users/y/Downloads')", timeout=40)
        assert "False" in (_wait(sh).get("result") or "")
    finally:
        sh.shutdown()
