"""TTS engine dependency installer.

Maps each TTS engine to its required Python package(s). Provides probe
(installed?) + install (one-shot pip into the running .venv).

Intentionally async-friendly: install runs in a subprocess, never blocks
the event loop.
"""

from __future__ import annotations

import asyncio
import importlib.util
import logging
import shutil
import sys
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EngineDeps:
    engine: str
    # Top-level python module to import-test ("cosyvoice", "indextts", ...)
    probe_module: str
    # Pip install spec(s) — supports git+https URLs or PyPI names
    pip_specs: tuple[str, ...]
    # Friendly note shown in /api/v1/engines/{name}/deps responses
    note: str = ""


# ---- Manifest -----------------------------------------------------------------
# Source URLs verified from upstream READMEs as of 2026-04-15.
# When a package isn't on PyPI, we install from git HEAD with --no-deps when
# possible to avoid pulling massive transitive trees we already have (torch).

_MANIFEST: dict[str, EngineDeps] = {
    "cosyvoice2": EngineDeps(
        engine="cosyvoice2",
        probe_module="cosyvoice",
        pip_specs=("git+https://github.com/FunAudioLLM/CosyVoice.git",),
        note="重型依赖（含 conformer / matcha-tts），首次安装可能 5-10 分钟",
    ),
    "indextts2": EngineDeps(
        engine="indextts2",
        probe_module="indextts",
        pip_specs=("git+https://github.com/index-tts/index-tts.git",),
        note="官方 README 推荐用 uv pip install -e；这里走 git 直装",
    ),
    "qwen3_tts_base": EngineDeps(
        engine="qwen3_tts_base",
        probe_module="qwen_tts",
        pip_specs=("qwen-tts",),
        note="PyPI 官方包（共用 qwen-tts，与 customvoice/voicedesign 同一依赖）",
    ),
    "qwen3_tts_customvoice": EngineDeps(
        engine="qwen3_tts_customvoice",
        probe_module="qwen_tts",
        pip_specs=("qwen-tts",),
        note="共用 qwen-tts 包；9 个内置 speaker + instruction 控制",
    ),
    "qwen3_tts_voicedesign": EngineDeps(
        engine="qwen3_tts_voicedesign",
        probe_module="qwen_tts",
        pip_specs=("qwen-tts",),
        note="共用 qwen-tts 包；文字描述生成 voice",
    ),
    "moss_tts": EngineDeps(
        engine="moss_tts",
        probe_module="transformers",
        pip_specs=(),  # 走 transformers + AutoModel，无独立包
        note="无独立 pip 依赖，仅需 transformers + 模型权重",
    ),
    "voxcpm2": EngineDeps(
        engine="voxcpm2",
        probe_module="voxcpm",
        pip_specs=("voxcpm",),
        note="PyPI 包，依赖少",
    ),
}


def list_manifest() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for name, dep in _MANIFEST.items():
        out[name] = {
            "engine": dep.engine,
            "probe_module": dep.probe_module,
            "pip_specs": list(dep.pip_specs),
            "installed": is_installed(name),
            "note": dep.note,
        }
    return out


def get(engine: str) -> EngineDeps | None:
    return _MANIFEST.get(engine)


def is_installed(engine: str) -> bool:
    dep = _MANIFEST.get(engine)
    if dep is None:
        return False
    if not dep.pip_specs and dep.probe_module == "transformers":
        # transformers is a hard core dep — assume present
        return importlib.util.find_spec("transformers") is not None
    spec = importlib.util.find_spec(dep.probe_module)
    return spec is not None


def _python_executable() -> str:
    return sys.executable or shutil.which("python3") or "python3"


async def install(engine: str, *, on_log=None) -> tuple[bool, str]:
    """Install pip specs for engine. Returns (success, combined_log).

    on_log: optional async callable(line: str) — invoked per stdout line for
    live UI streaming.
    """
    dep = _MANIFEST.get(engine)
    if dep is None:
        return False, f"unknown engine: {engine}"
    if not dep.pip_specs:
        return True, "no extra pip deps required"

    py = _python_executable()
    # Prefer uv when available — much faster, respects the venv we're in.
    uv = shutil.which("uv")
    cmd: list[str]
    if uv:
        cmd = [uv, "pip", "install", "--python", py, *dep.pip_specs]
    else:
        cmd = [py, "-m", "pip", "install", *dep.pip_specs]

    logger.info("installing TTS deps for %s: %s", engine, " ".join(cmd))
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    out_lines: list[str] = []
    assert proc.stdout is not None
    async for raw in proc.stdout:
        line = raw.decode(errors="replace").rstrip()
        out_lines.append(line)
        if on_log is not None:
            try:
                await on_log(line)
            except Exception:
                pass
    rc = await proc.wait()
    log = "\n".join(out_lines)
    return rc == 0, log
