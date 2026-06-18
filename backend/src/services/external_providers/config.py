"""provider 配置加载 + GovernedProvider(provider × governor 绑定)+ 全局 registry。

配置:backend/configs/external_providers.yaml(`NOUS_EXTERNAL_PROVIDERS_CONFIG` 可 override,
测试用)。registry 懒构建并缓存;reset_registry() 供测试清缓存。
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from src.services.external_providers.base import (
    ExternalCliProvider,
    ExternalGenRequest,
    ExternalGenResult,
)
from src.services.external_providers.codex import CodexProvider
from src.services.external_providers.dreamina import DreaminaProvider
from src.services.external_providers.governor import ProviderGovernor

logger = logging.getLogger(__name__)

_BACKEND_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_CONFIG = _BACKEND_ROOT / "configs" / "external_providers.yaml"

# provider 名 → 实现类。新增 provider 在此登记。
_PROVIDER_CLASSES: dict[str, type[ExternalCliProvider]] = {
    "dreamina": DreaminaProvider,
    "codex": CodexProvider,
}


class ProviderConfig(BaseModel):
    enabled: bool = True
    executable: str = ""
    modalities: list[str] = Field(default_factory=list)
    concurrency: int = 1
    rate_per_min: float = 0.0
    min_interval_s: float = 0.0
    queue_capacity: int = 16
    cache_ttl_s: float = 0.0
    # provider 专属可选项(如 dreamina poll_seconds),透传构造函数。
    params: dict = Field(default_factory=dict)


class GovernedProvider:
    """一个 provider + 它的 governor。节点/路由只跟这层打交道。"""

    def __init__(self, provider: ExternalCliProvider, governor: ProviderGovernor) -> None:
        self.provider = provider
        self.governor = governor

    @property
    def name(self) -> str:
        return self.provider.name

    async def probe_status(self):
        return await self.provider.probe_status()

    async def login_start(self):
        return await self.provider.login_start()

    async def generate(self, req: ExternalGenRequest) -> ExternalGenResult:
        cache_key = _request_cache_key(self.name, req)
        return await self.governor.run(lambda: self.provider.generate(req), cache_key=cache_key)


def _request_cache_key(provider: str, req: ExternalGenRequest) -> str:
    payload = req.model_dump()
    blob = json.dumps({"p": provider, **payload}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _config_path() -> Path:
    override = os.environ.get("NOUS_EXTERNAL_PROVIDERS_CONFIG")
    return Path(override).expanduser() if override else _DEFAULT_CONFIG


def load_config(path: Path | None = None) -> dict[str, ProviderConfig]:
    cfg_path = path or _config_path()
    if not cfg_path.is_file():
        logger.info("external_providers config 不存在,跳过:%s", cfg_path)
        return {}
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    providers = data.get("providers") or {}
    out: dict[str, ProviderConfig] = {}
    for name, raw in providers.items():
        try:
            out[str(name)] = ProviderConfig(**(raw or {}))
        except Exception:  # noqa: BLE001 — 单个 provider 配置坏不该拖垮整体
            logger.exception("external_providers: provider %s 配置无效,已跳过", name)
    return out


def _build_registry(path: Path | None = None) -> dict[str, GovernedProvider]:
    registry: dict[str, GovernedProvider] = {}
    for name, cfg in load_config(path).items():
        if not cfg.enabled:
            continue
        cls = _PROVIDER_CLASSES.get(name)
        if cls is None:
            logger.warning("external_providers: 未知 provider %s(无实现类),跳过", name)
            continue
        provider = cls(executable=cfg.executable, **cfg.params)
        if cfg.modalities:
            provider.modalities = set(cfg.modalities)
        governor = ProviderGovernor(
            name,
            concurrency=cfg.concurrency,
            rate_per_min=cfg.rate_per_min,
            min_interval_s=cfg.min_interval_s,
            queue_capacity=cfg.queue_capacity,
            cache_ttl_s=cfg.cache_ttl_s,
        )
        registry[name] = GovernedProvider(provider, governor)
    return registry


_registry: dict[str, GovernedProvider] | None = None


def get_registry() -> dict[str, GovernedProvider]:
    global _registry
    if _registry is None:
        _registry = _build_registry()
    return _registry


def reset_registry() -> None:
    """清缓存(测试用,或配置热更后)。"""
    global _registry
    _registry = None
