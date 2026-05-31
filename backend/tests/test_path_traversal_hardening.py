"""round7:path traversal 加固 —— validate_path is_relative_to + load_persona 拒穿越。"""
import pytest

from src.utils.path_security import validate_path


def test_validate_path_blocks_same_prefix_sibling(tmp_path):
    """前缀兄弟目录(agents → agents-evil)必须被拒(老 startswith 缺陷)。"""
    base = tmp_path / "agents"
    base.mkdir()
    (tmp_path / "agents-evil").mkdir()
    with pytest.raises(ValueError):
        validate_path(tmp_path / "agents-evil" / "secret", base)


def test_validate_path_blocks_dotdot(tmp_path):
    base = tmp_path / "agents"
    base.mkdir()
    with pytest.raises(ValueError):
        validate_path(base / ".." / ".." / "etc" / "passwd", base)


def test_validate_path_allows_normal_subpath(tmp_path):
    base = tmp_path / "agents"
    base.mkdir()
    assert validate_path(base / "my-agent" / "config.json", base)


def test_load_persona_rejects_traversal(monkeypatch, tmp_path):
    """load_persona 的 agent_id 含 ../ → AgentNotFound,不读越界文件。"""
    from src.services.prompt_composer import _persona
    monkeypatch.setattr(_persona, "_agents_root", lambda: tmp_path / "agents")
    (tmp_path / "agents").mkdir()
    # 在越界处放个 config.json 诱饵
    (tmp_path / "evil").mkdir()
    (tmp_path / "evil" / "config.json").write_text("{}")
    with pytest.raises(_persona.AgentNotFound):
        _persona.load_persona("../evil")
