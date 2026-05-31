"""round7:persona 缓存 key 覆盖全部 4 文件 —— 改 .md(不动 config.json)也失效。"""
import os
import time

import pytest


@pytest.fixture
def agents_root(monkeypatch, tmp_path):
    from src.services.prompt_composer import _persona
    root = tmp_path / "agents"
    monkeypatch.setattr(_persona, "_agents_root", lambda: root)
    _persona._load_cached.cache_clear()
    return root


def _make_agent(root, name, agent_md=""):
    d = root / name
    d.mkdir(parents=True)
    (d / "config.json").write_text('{"skills": []}')
    (d / "AGENT.md").write_text(agent_md)
    return d


def test_editing_md_invalidates_cache(agents_root):
    from src.services.prompt_composer import _persona
    d = _make_agent(agents_root, "a", agent_md="v1")
    b1 = _persona.load_persona("a")
    assert b1.agent == "v1"
    # 改 AGENT.md(不动 config.json),把 mtime 推后
    time.sleep(0.01)
    (d / "AGENT.md").write_text("v2")
    new = time.time() + 5
    os.utime(d / "AGENT.md", (new, new))
    b2 = _persona.load_persona("a")
    assert b2.agent == "v2"  # 老 bug:只看 config.json mtime → 仍读到 v1
