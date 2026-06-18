"""外部 CLI 生成 provider 子系统(spec 2026-06-18-external-cli-generation-providers)。

把账号登录态的外部 CLI(即梦 dreamina / codex)抽成统一 provider,经 subprocess
转发生成图像/视频。**纯云转发,不吃本地 GPU,跑在主后端进程**(对应工作流里的 inline
节点,不进 GPU runner)。每 provider 套一层 ProviderGovernor 护栏(并发/限速/节流/
满 503/可选缓存),把账号在云侧的画像锁在「单人低频串行」——这是规避封号的核心。

公开面:
- ExternalCliProvider:provider ABC
- ProviderGovernor:并发节流护栏
- ExternalGenRequest / ExternalGenResult / ProviderStatus / ArtifactRef:契约
- get_registry():按 configs/external_providers.yaml 构建的 {name: GovernedProvider}
"""
from __future__ import annotations

from src.services.external_providers.base import (
    ArtifactRef,
    ExternalCliProvider,
    ExternalGenRequest,
    ExternalGenResult,
    ProviderStatus,
)
from src.services.external_providers.config import (
    GovernedProvider,
    ProviderConfig,
    get_registry,
    reset_registry,
)
from src.services.external_providers.governor import GovernorBusyError, ProviderGovernor

__all__ = [
    "ArtifactRef",
    "ExternalCliProvider",
    "ExternalGenRequest",
    "ExternalGenResult",
    "ProviderStatus",
    "ProviderGovernor",
    "GovernorBusyError",
    "GovernedProvider",
    "ProviderConfig",
    "get_registry",
    "reset_registry",
]
