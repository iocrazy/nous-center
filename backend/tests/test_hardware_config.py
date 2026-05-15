"""Lane A: load_hardware_config 解析覆盖。"""
from src.config import load_hardware_config


def test_load_default_hardware_yaml():
    """configs/hardware.yaml 解析出 groups 列表。"""
    cfg = load_hardware_config()
    assert "groups" in cfg
    groups = cfg["groups"]
    assert isinstance(groups, list)
    assert len(groups) >= 1
    # 当前 2 卡布局：单个 llm-tp group
    llm = next(g for g in groups if g["id"] == "llm-tp")
    assert llm["gpus"] == [0, 1]
    assert llm["nvlink"] is True
    assert llm["role"] == "llm"
    assert llm["vram_gb"] == 48


def test_load_3gpu_template(tmp_path):
    """hardware.3gpu.yaml 样板解析出 3 个 group。"""
    cfg = load_hardware_config(path="configs/hardware.3gpu.yaml")
    ids = {g["id"] for g in cfg["groups"]}
    assert ids == {"image", "llm-tp", "tts"}
    image = next(g for g in cfg["groups"] if g["id"] == "image")
    assert image["gpus"] == [2]
    assert image["nvlink"] is False
    assert image["vram_gb"] == 96


def test_load_missing_file_returns_empty(tmp_path):
    """文件缺失 → fail-soft 返回 {'groups': []}，不抛异常。"""
    missing = tmp_path / "nope.yaml"
    cfg = load_hardware_config(path=str(missing))
    assert cfg == {"groups": []}


def test_load_corrupt_yaml_returns_empty(tmp_path):
    """yaml 损坏 → fail-soft 返回 {'groups': []}。"""
    bad = tmp_path / "bad.yaml"
    bad.write_text("groups: [ this is not: valid: yaml")
    cfg = load_hardware_config(path=str(bad))
    assert cfg == {"groups": []}


def test_load_missing_groups_key_returns_empty(tmp_path):
    """yaml 合法但无 groups 键 → 返回 {'groups': []}。"""
    nogroups = tmp_path / "nogroups.yaml"
    nogroups.write_text("detection:\n  mode: auto\n")
    cfg = load_hardware_config(path=str(nogroups))
    assert cfg == {"groups": []}
