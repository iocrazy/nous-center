"""服务层 API spec PR-4:输出交付 TTL 归服务层配置(不再是出图节点 widget)。

用户:URL 有效期不该是节点的事,该是工作流 API 的功能。url_ttl_seconds widget 从 3 个出图节点
(flux2 VAE decode / seedvr2 增强 / image-io)删掉,改读 get_settings().IMAGE_URL_TTL_SECONDS。
"""
from __future__ import annotations

import pathlib

import yaml

_ROOT = pathlib.Path(__file__).parent.parent
_SRC = _ROOT / "src"
_NODES = _ROOT / "nodes"


def test_config_has_image_url_ttl():
    from src.config import get_settings
    assert get_settings().IMAGE_URL_TTL_SECONDS == 3600


def test_node_yaml_no_url_ttl_widget():
    """3 个出图节点的 node.yaml 不再有 url_ttl_seconds widget(仍是合法 yaml)。"""
    for rel in ("flux2-components/node.yaml", "seedvr2/node.yaml", "image-io/node.yaml"):
        text = (_NODES / rel).read_text()
        assert "url_ttl_seconds" not in text, f"{rel} 仍有 url_ttl_seconds widget"
        d = yaml.safe_load(text)
        assert d.get("nodes"), f"{rel} 不是合法 node yaml"


def test_code_reads_ttl_from_config_not_node():
    """3 个落盘/签 URL 处都读 config,不再 node.inputs.get('url_ttl_seconds')。"""
    runner = (_SRC / "runner/runner_process.py").read_text()
    assert "get_settings().IMAGE_URL_TTL_SECONDS" in runner
    assert 'node.inputs.get("url_ttl_seconds")' not in runner
    io_exec = (_NODES / "image-io/executor.py").read_text()
    assert "get_settings().IMAGE_URL_TTL_SECONDS" in io_exec
    assert 'data.get("url_ttl_seconds")' not in io_exec
