"""Fetch and cache model metadata from ModelScope / HuggingFace."""

import logging
from pathlib import Path

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import get_settings, load_model_configs
from src.models.model_metadata import ModelMetadata

logger = logging.getLogger(__name__)

MODELSCOPE_API = "https://modelscope.cn/api/v1/models"
HF_API = "https://huggingface.co/api/models"


def _format_size(size_bytes: int | None) -> str | None:
    if size_bytes is None:
        return None
    gb = size_bytes / (1024 ** 3)
    if gb >= 1:
        return f"{gb:.2f}GB"
    mb = size_bytes / (1024 ** 2)
    return f"{mb:.0f}MB"


async def _fetch_modelscope(client: httpx.AsyncClient, repo_id: str) -> dict | None:
    """Fetch metadata from ModelScope API."""
    try:
        resp = await client.get(f"{MODELSCOPE_API}/{repo_id}", timeout=15)
        if resp.status_code != 200:
            return None
        data = resp.json().get("Data", {})
        model_infos = data.get("ModelInfos", {})
        # Extract model size from safetensor or first available key
        model_size = None
        for info in model_infos.values():
            if "model_size" in info:
                model_size = info["model_size"]
                break
        tensor_types = None
        for info in model_infos.values():
            if "tensor_type" in info:
                tensor_types = info["tensor_type"]
                break

        org = data.get("Organization", {})
        return {
            "organization": org.get("Name") if isinstance(org, dict) else None,
            "model_size_bytes": model_size,
            "frameworks": data.get("Frameworks"),
            "libraries": data.get("Libraries"),
            "license": data.get("License") or None,
            "languages": None,  # ModelScope doesn't have a standard languages field
            "tags": data.get("Tags"),
            "tensor_types": tensor_types,
            "description": data.get("ChineseName"),
        }
    except Exception as e:
        logger.warning("ModelScope fetch failed for %s: %s", repo_id, e)
        return None


async def _fetch_huggingface(client: httpx.AsyncClient, repo_id: str) -> dict | None:
    """Fetch metadata from HuggingFace API."""
    try:
        resp = await client.get(f"{HF_API}/{repo_id}", timeout=15)
        if resp.status_code != 200:
            return None
        data = resp.json()
        tags = data.get("tags", [])
        # Extract structured info from tags
        frameworks = [t for t in tags if t in ("pytorch", "safetensors", "onnx", "tensorflow", "jax", "transformers")]
        license_tag = next((t.replace("license:", "") for t in tags if t.startswith("license:")), None)
        lang_tags = [t for t in tags if len(t) == 2 and t.isalpha()]
        card = data.get("cardData") or {}

        return {
            "organization": data.get("author"),
            "model_size_bytes": data.get("usedStorage"),
            "frameworks": frameworks or None,
            "libraries": None,
            "license": license_tag or card.get("license"),
            "languages": card.get("language") or lang_tags or None,
            "tags": [t for t in tags if t not in frameworks and not t.startswith("license:") and len(t) > 2],
            "tensor_types": None,
            "description": None,
        }
    except Exception as e:
        logger.warning("HuggingFace fetch failed for %s: %s", repo_id, e)
        return None


async def fetch_and_store(session: AsyncSession, engine_key: str, cfg: dict) -> ModelMetadata | None:
    """Fetch metadata for one engine and store in DB. Prefer ModelScope, fallback HF."""
    ms_id = cfg.get("modelscope_id")
    hf_id = cfg.get("hf_id")
    if not ms_id and not hf_id:
        return None

    async with httpx.AsyncClient() as client:
        meta = None
        if ms_id:
            meta = await _fetch_modelscope(client, ms_id)
        if meta is None and hf_id:
            meta = await _fetch_huggingface(client, hf_id)

    if meta is None:
        return None

    row = ModelMetadata(
        engine_key=engine_key,
        modelscope_id=ms_id,
        hf_id=hf_id,
        **meta,
    )
    session.add(row)
    await session.commit()
    await session.refresh(row)
    return row


async def get_all_metadata(session: AsyncSession) -> dict[str, ModelMetadata]:
    """Return all cached metadata keyed by engine_key."""
    result = await session.execute(select(ModelMetadata))
    return {row.engine_key: row for row in result.scalars().all()}


async def sync_metadata(session: AsyncSession) -> dict[str, ModelMetadata]:
    """Check configs, fetch metadata for any engine not yet in DB."""
    configs = load_model_configs()
    existing = await get_all_metadata(session)
    for key, cfg in configs.items():
        if key not in existing:
            row = await fetch_and_store(session, key, cfg)
            if row:
                existing[key] = row
    return existing


async def refresh_metadata(session: AsyncSession, engine_key: str) -> ModelMetadata | None:
    """Force re-fetch metadata for a specific engine."""
    configs = load_model_configs()
    cfg = configs.get(engine_key)
    if not cfg:
        return None
    # Delete existing
    result = await session.execute(
        select(ModelMetadata).where(ModelMetadata.engine_key == engine_key)
    )
    old = result.scalar_one_or_none()
    if old:
        await session.delete(old)
        await session.commit()
    return await fetch_and_store(session, engine_key, cfg)


def scan_local_models() -> set[str]:
    """Scan LOCAL_MODELS_PATH and return set of local_path dirs that exist.

    Layout:
      llm/<MODEL>                          — depth 2
      tts/<MODEL>                          — depth 2
      image/diffusion_models/<MODEL>       — depth 3 (transformer dirs)
      image/<sub>/<MODEL>                  — depth 3 (vae/, text_encoders/ etc)
    """
    settings = get_settings()
    base = Path(settings.LOCAL_MODELS_PATH)
    if not base.exists():
        return set()
    found = set()
    for type_dir in base.iterdir():
        if not type_dir.is_dir():
            continue
        # Image holds component subdirectories (diffusion_models / vae /
        # text_encoders); each child is one component dir holding
        # single-file safetensors. Walk one extra level so the registered
        # `image/diffusion_models/<MODEL>` paths are matchable.
        if type_dir.name == "image":
            for sub_dir in type_dir.iterdir():
                if not sub_dir.is_dir():
                    continue
                for component_dir in sub_dir.iterdir():
                    if component_dir.is_dir():
                        found.add(f"{type_dir.name}/{sub_dir.name}/{component_dir.name}")
            continue
        for model_dir in type_dir.iterdir():
            if model_dir.is_dir():
                found.add(f"{type_dir.name}/{model_dir.name}")
    return found
