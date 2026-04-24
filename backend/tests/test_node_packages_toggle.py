"""m12 节点包 enable / disable — 验证 .disabled marker 控制 scan 注册行为。"""

from __future__ import annotations

from pathlib import Path

import pytest

from nodes import _PACKAGE_DIR, scan_packages


@pytest.fixture
def temp_pkg(monkeypatch, tmp_path):
    """在临时目录里造一个最小节点包，把 _PACKAGE_DIR 指过去。"""
    pkg_dir = tmp_path / "fake_pkg"
    pkg_dir.mkdir()
    (pkg_dir / "node.yaml").write_text(
        "name: fake_pkg\nversion: 0.1.0\nnodes:\n  fake_node:\n    label: Fake\n"
    )

    import nodes
    monkeypatch.setattr(nodes, "_PACKAGE_DIR", tmp_path)
    yield pkg_dir
    # 把全局 registry 清掉避免污染下游测试
    nodes._packages.clear()
    nodes._node_definitions.clear()
    nodes._node_executors.clear()
    monkeypatch.setattr(nodes, "_PACKAGE_DIR", _PACKAGE_DIR)


def test_default_package_is_enabled(temp_pkg: Path):
    pkgs = scan_packages()
    assert "fake_pkg" in pkgs
    assert pkgs["fake_pkg"]["enabled"] is True
    # 节点定义被注册（启用态）
    from nodes import get_all_definitions
    assert "fake_node" in get_all_definitions()


def test_disabled_marker_skips_definitions(temp_pkg: Path):
    (temp_pkg / ".disabled").touch()
    pkgs = scan_packages()
    assert pkgs["fake_pkg"]["enabled"] is False
    # 节点 definition 不该被注册（禁用态）
    from nodes import get_all_definitions
    assert "fake_node" not in get_all_definitions()


def test_re_enable_by_removing_marker(temp_pkg: Path):
    marker = temp_pkg / ".disabled"
    marker.touch()
    pkgs = scan_packages()
    assert pkgs["fake_pkg"]["enabled"] is False

    marker.unlink()
    pkgs = scan_packages()
    assert pkgs["fake_pkg"]["enabled"] is True
    from nodes import get_all_definitions
    assert "fake_node" in get_all_definitions()
