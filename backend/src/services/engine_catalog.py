"""统一引擎库目录扩展 —— 把 by-key 超分(SeedVR2)+ 单文件组件(diffusion_models/clip/vae/loras)
也作为引擎库条目,带 VRAM 残留状态(loaded/gpu)。spec 2026-06-02-unified-engine-library。

引擎库原 `list_all_engines` 只列 registry + 自动发现的整模型(model_scanner 显式 skip 组件、
SeedVR2 by-key 不在 registry)→ 用户无法在引擎库统一看/管它们。本模块补这些条目:
- loaded 状态从 `aggregate_runner_loaded`(跨进程单一真相,runner 经 Pong 上报)**多键匹配**:
  SeedVR2 按 model_id 前缀 `image:SeedVR2:` + source_files 含 DiT 文件名;组件按 source_files basename。
- `has_adapter`:SeedVR2/超分可独立加载(True);单文件组件随 pipeline 加载、不独立可加载(False →
  UI 禁用加载按钮,不给假操作)。
- `kind`:upscale / component / lora,供前端分组(PR-2)。

CI 安全:顶层只 import os/typing + schemas;image_seedvr2/component_scanner/runner_models 顶层无 torch。
磁盘空(CI)时 scan 返空 → 目录为空,不抛。
"""
from __future__ import annotations

import os
from typing import Any

from src.models.schemas import EngineInfo


def _loaded_index(app_state: Any) -> tuple[dict, list]:
    """aggregate_runner_loaded → ({源文件 basename -> entry}, [SeedVR2 loaded entries])。"""
    from src.services.runner_models import aggregate_runner_loaded  # noqa: PLC0415

    by_src: dict[str, dict] = {}
    seedvr2_loaded: list[dict] = []
    try:
        entries = aggregate_runner_loaded(app_state)
    except Exception:  # noqa: BLE001 — best-effort 镜像,拿不到就当无加载
        entries = []
    for e in entries:
        if str(e.get("model_id", "")).startswith("image:SeedVR2:"):
            seedvr2_loaded.append(e)
        for s in (e.get("source_files") or []):
            by_src[os.path.basename(str(s))] = e
    return by_src, seedvr2_loaded


def _component_loaded_index(app_state: Any) -> dict:
    """aggregate_runner_components → {组件文件 basename -> comp 快照 entry}(组件 L1 PR-3a)。

    含预加载的孤组件(不属任何 combo)+ 它们的 resident 态。引擎库据此标单组件 loaded@卡 + 常驻。
    """
    from src.services.runner_models import aggregate_runner_components  # noqa: PLC0415

    idx: dict[str, dict] = {}
    try:
        entries = aggregate_runner_components(app_state)
    except Exception:  # noqa: BLE001 — best-effort,拿不到就当无加载
        entries = []
    for e in entries:
        f = e.get("file")
        if f:
            idx[os.path.basename(str(f))] = e
    return idx


def _dev_to_gpu(dev: Any) -> int:
    if isinstance(dev, str) and dev.startswith("cuda:"):
        try:
            return int(dev.split(":")[1])
        except (ValueError, IndexError):
            return 0
    return 0


def seedvr2_catalog_entries(app_state: Any) -> list[EngineInfo]:
    """SeedVR2 DiT(磁盘已有的白名单)→ 引擎库条目(kind=upscale,可独立加载)。
    loaded:某 SeedVR2 adapter 的 source_files 含这个 DiT 文件名。"""
    from src.services.inference.image_seedvr2 import (  # noqa: PLC0415
        seedvr2_dit_models_with_disk_status,
    )

    _by_src, sv_loaded = _loaded_index(app_state)
    out: list[EngineInfo] = []
    for m in seedvr2_dit_models_with_disk_status():
        if not m.get("present"):
            continue  # 引擎库只列磁盘已有的(可下载的留节点下拉)
        match = next(
            (e for e in sv_loaded
             if any(os.path.basename(str(s)) == m["filename"] for s in (e.get("source_files") or []))),
            None,
        )
        out.append(EngineInfo(
            name=f"seedvr2:{m['filename']}",
            display_name=f"SeedVR2 {m['label']}",
            type="image",
            kind="upscale",
            status="loaded" if match else "unloaded",
            gpu=int(match.get("gpu_index", 0)) if match else 0,
            vram_gb=round((m.get("size_mb") or 0) / 1024, 1),
            resident=False,
            has_adapter=True,  # by-key 可独立加载
            local_path=f"image/SEEDVR2/{m['filename']}",
            local_exists=True,
            auto_detected=True,
            loaded_gpu=int(match["gpu_index"]) if match and match.get("gpu_index") is not None else None,
        ))
    return out


_COMPONENT_ROLES = [
    ("diffusion_models", "component"),
    ("clip", "component"),
    ("vae", "component"),
    ("loras", "lora"),
]


def component_catalog_entries(app_state: Any) -> list[EngineInfo]:
    """单文件组件 → 引擎库条目(kind=component/lora,不独立可加载)。
    loaded:该文件出现在某 loaded adapter 的 source_files(随 pipeline 加载)。"""
    from src.services.component_scanner import scan_components  # noqa: PLC0415

    by_src, _sv = _loaded_index(app_state)
    comp_idx = _component_loaded_index(app_state)  # PR-3a:预加载的孤组件 + resident 态
    out: list[EngineInfo] = []
    for role, kind in _COMPONENT_ROLES:
        try:
            files = scan_components(role)
        except Exception:  # noqa: BLE001
            files = []
        for c in files:
            fn = c.get("filename", "")
            combo_match = by_src.get(fn)        # 属于某 loaded combo(随 pipeline 加载)
            comp_match = comp_idx.get(fn)       # 单组件 L1 池(含预加载孤组件)
            match = comp_match or combo_match   # 任一即视为已加载
            # gpu:组件快照带 device(cuda:N);combo entry 带 gpu_index。
            if comp_match is not None:
                gpu_idx = _dev_to_gpu(comp_match.get("device"))
            elif combo_match is not None and combo_match.get("gpu_index") is not None:
                gpu_idx = int(combo_match["gpu_index"])
            else:
                gpu_idx = None
            out.append(EngineInfo(
                name=f"component:{role}:{c.get('abs_path')}",
                display_name=fn,
                type="image",
                kind=kind,
                status="loaded" if match else "unloaded",
                gpu=gpu_idx if gpu_idx is not None else 0,
                vram_gb=round((c.get("size_mb") or 0) / 1024, 1),
                # resident 只来自单组件 L1 池(combo 随 pipeline 的不算组件级常驻)。组件 L1 PR-3a。
                resident=bool(comp_match and comp_match.get("resident")),
                has_adapter=False,  # 不作为独立 adapter 加载;前端按 kind 开「预加载/常驻」动作(PR-3b)
                local_path=c.get("abs_path"),
                local_exists=True,
                auto_detected=True,
                loaded_gpu=gpu_idx,
                # 已加载组件的 L1 身份串 → 前端常驻 toggle 按它精确匹配(含真实 device)。组件 L1 PR-3a。
                state_key=(comp_match.get("state_key") if comp_match else None),
            ))
    return out


def catalog_extra_engines(app_state: Any, type_filter: str | None) -> list[EngineInfo]:
    """引擎库补充条目(SeedVR2 + 组件)。只在无过滤或 type=image 时出(它们都是图像类)。"""
    if type_filter and type_filter != "image":
        return []
    return seedvr2_catalog_entries(app_state) + component_catalog_entries(app_state)
