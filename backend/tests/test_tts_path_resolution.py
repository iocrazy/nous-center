"""TTS 模型路径解析(修 2026-06-10 体检逮到的 TTS 链回归)。

models.yaml 的 main 是相对 LOCAL_MODELS_PATH 的(speech/xxx);TTSEngine 基类
此前不拼基路径 → 相对串直通引擎,CosyVoice 当 modelscope repo id 下载 → 404。
"""
from __future__ import annotations

import pathlib


def test_tts_base_resolves_model_path_against_local_models(tmp_path, monkeypatch):
    """存在 LOCAL_MODELS_PATH/<main> → 用绝对;不存在 → 原样(绝对路径/HF id 直通)。"""
    from src.workers.tts_engines.base import TTSEngine

    class _Probe(TTSEngine):
        ENGINE_NAME = "_probe_path"

        @property
        def engine_name(self) -> str:  # pragma: no cover
            return "_probe_path"

        def load_sync(self):  # pragma: no cover
            pass

        def synthesize(self, *a, **k):  # pragma: no cover
            pass

    (tmp_path / "speech" / "fake-model").mkdir(parents=True)
    monkeypatch.setenv("LOCAL_MODELS_PATH", str(tmp_path))
    from src.config import get_settings
    get_settings.cache_clear()
    try:
        e = _Probe(paths={"main": "speech/fake-model"})
        assert e.model_path == tmp_path / "speech" / "fake-model", "相对路径必须拼 LOCAL_MODELS_PATH"
        e2 = _Probe(paths={"main": "speech/not-on-disk"})
        assert str(e2.model_path) == "speech/not-on-disk", "盘上没有 → 原样直通(HF id 语义)"
    finally:
        get_settings.cache_clear()


def test_models_yaml_tts_paths_not_stale_tts_dir():
    """models.yaml 的 main 不得再指旧 tts/ 目录(盘上真实目录是 speech/)。"""
    import yaml

    cfg = yaml.safe_load(
        (pathlib.Path(__file__).parent.parent / "configs/models.yaml").read_text())
    for m in cfg["models"]:
        main = (m.get("paths") or {}).get("main", "")
        assert not str(main).startswith("tts/"), \
            f"{m.get('id')} 仍指旧 tts/ 目录(盘上是 speech/)"
