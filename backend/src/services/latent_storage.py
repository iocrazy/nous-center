"""Latent tensor storage(spec 2026-06-08 路 B,PR-B1)。

同空间真 latent 接力:终端 VAE Decode 在 output_mode="latent" 时把真 latent 张量落盘,节点间传
**路径引用**(latent_ref.path),不进 msgpack(torch.Tensor 进 msgpack 直接炸,见 protocol.py)。
下游 sample_from_latent(PR-B2)从 path 读回张量注入 latents=。

与 image_output_storage 的区别:latent 不经 HTTP 服务(runner 同机直接读盘),**无需签名 URL**——
latent_ref 只带本地路径字符串。落盘格式 = safetensors(diffusers 钉的依赖已带,跨进程/版本稳)。
落盘根 = NAS_OUTPUTS_PATH/latents(与 images 同盘,统一 reap)。
"""
from __future__ import annotations

import logging
import os
import time
import uuid as _uuid
from datetime import datetime, timezone
from pathlib import Path

from src.config import get_settings

logger = logging.getLogger(__name__)


def _latents_root() -> Path:
    """Latent 落盘根。优先级同 image_output_storage:NOUS_IMAGE_OUTPUTS override(末段换 latents)
    → NAS_OUTPUTS_PATH/latents → ~/.gstack/outputs/latents。"""
    override = os.environ.get("NOUS_IMAGE_OUTPUTS")
    if override:
        # override 指向 .../images 时挪到同级 latents;否则在其下建 latents。
        p = Path(override).expanduser()
        return (p.parent / "latents") if p.name == "images" else (p / "latents")
    nas_root = (get_settings().NAS_OUTPUTS_PATH or "").strip()
    if nas_root:
        return Path(nas_root).expanduser() / "latents"
    return Path.home() / ".gstack" / "outputs" / "latents"


def write_latent(latent_bytes: bytes, *, ext: str = "safetensors") -> dict:
    """把序列化好的 latent 字节落盘到 latents/<date>/<uuid>.<ext>。

    Returns {uuid, path(str), date}。path 是绝对本地路径,直接进 latent_ref 描述符(runner 同机读)。
    原子写(.tmp→rename)防半截读。
    """
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    root = _latents_root() / today
    root.mkdir(parents=True, exist_ok=True)

    file_uuid = _uuid.uuid4().hex
    path = root / f"{file_uuid}.{ext}"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(latent_bytes)
    os.replace(tmp, path)

    return {"uuid": file_uuid, "path": str(path), "date": today}


def reap_orphans(*, older_than_seconds: int) -> dict:
    """删 latents 根下过期 .safetensors(latent 是中间产物,比图更该短 TTL 清)。
    由 backend lifespan 任务调,与 image_output_storage.reap_orphans 同形。"""
    root = _latents_root()
    summary = {"scanned": 0, "deleted": 0, "dirs_pruned": 0, "errors": 0}
    if not root.exists():
        return summary
    cutoff = time.time() - max(60, int(older_than_seconds))
    for date_dir in sorted(root.iterdir()):
        if not date_dir.is_dir():
            continue
        for f in date_dir.iterdir():
            if not f.is_file() or f.suffix.lower() not in (".safetensors", ".pt"):
                continue
            summary["scanned"] += 1
            try:
                if f.stat().st_mtime < cutoff:
                    f.unlink()
                    summary["deleted"] += 1
            except OSError as e:
                logger.warning("latent_storage.reap_orphans: unlink %s: %s", f, e)
                summary["errors"] += 1
        try:
            if not any(date_dir.iterdir()):
                date_dir.rmdir()
                summary["dirs_pruned"] += 1
        except OSError:
            pass
    if summary["deleted"] or summary["dirs_pruned"]:
        logger.info("latent_storage.reap_orphans: scanned=%d deleted=%d dirs_pruned=%d errors=%d",
                    summary["scanned"], summary["deleted"], summary["dirs_pruned"], summary["errors"])
    return summary
