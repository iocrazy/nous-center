"""Lane J: hardware_topo fixture self-test — two hardware.yaml contents + tmp file write."""
from tests.fixtures.hardware_topo import (
    HARDWARE_2GPU,
    HARDWARE_3GPU,
    write_hardware_yaml,
)


def test_2gpu_topo_has_single_llm_group():
    """2gpu layout (spec §1.4 plan A): single group llm-tp, GPU [0,1] NVLink."""
    groups = HARDWARE_2GPU["groups"]
    assert len(groups) == 1
    g = groups[0]
    assert g["id"] == "llm-tp"
    assert g["gpus"] == [0, 1]
    assert g["nvlink"] is True
    assert g["role"] == "llm"


def test_3gpu_topo_has_three_independent_groups():
    """3gpu layout (spec §3.2): image / llm-tp / tts."""
    groups = HARDWARE_3GPU["groups"]
    ids = {g["id"] for g in groups}
    assert ids == {"image", "llm-tp", "tts"}
    llm = next(g for g in groups if g["id"] == "llm-tp")
    assert llm["nvlink"] is True and llm["gpus"] == [0, 1]
    image = next(g for g in groups if g["id"] == "image")
    assert image["nvlink"] is False and image["gpus"] == [2]


def test_write_hardware_yaml_roundtrips(tmp_path):
    """write_hardware_yaml writes a tmp file; yaml.safe_load reads back equivalent."""
    import yaml

    path = write_hardware_yaml(tmp_path, HARDWARE_2GPU)
    assert path.exists()
    loaded = yaml.safe_load(path.read_text())
    assert loaded == HARDWARE_2GPU


def test_hardware_2gpu_fixture_resolves(hardware_2gpu):
    """hardware_2gpu fixture returns a Path to a written tmp file."""
    import yaml

    assert hardware_2gpu.exists()
    loaded = yaml.safe_load(hardware_2gpu.read_text())
    assert loaded == HARDWARE_2GPU


def test_hardware_3gpu_fixture_resolves(hardware_3gpu):
    """hardware_3gpu fixture returns a Path to a written tmp file."""
    import yaml

    assert hardware_3gpu.exists()
    loaded = yaml.safe_load(hardware_3gpu.read_text())
    assert loaded == HARDWARE_3GPU


def test_loader_reads_fixture_file(hardware_2gpu):
    """src.config.load_hardware_config(path=hardware_2gpu) reads fixture file."""
    from src.config import load_hardware_config

    cfg = load_hardware_config(str(hardware_2gpu))
    assert cfg == HARDWARE_2GPU
