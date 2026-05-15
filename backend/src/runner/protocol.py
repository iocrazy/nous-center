"""GPU Runner IPC 协议 —— 主进程 <-> image/TTS runner 子进程的 wire format.

spec §3.3。走 multiprocessing.Pipe，msgpack 编码（dev 模式 NOUS_IPC_FORMAT=json
fallback 便于 journalctl 调试）。**仅 image/TTS runner 走此协议**；LLM runner 不
收 RunNode，主进程直连其 vLLM HTTP 端口（Lane E）。

消息是 frozen dataclass —— 跨进程边界传不可变值，避免别名 bug。每个消息有一个
`kind` 字面量做判别式，`decode` 按 kind 路由回正确的 dataclass。
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Literal


class ProtocolError(Exception):
    """编解码失败 / 未知消息 kind。"""


# ------------------------------------------------------------------
# 主进程 -> image/TTS runner
# ------------------------------------------------------------------


@dataclass(frozen=True)
class LoadModel:
    model_key: str
    config: dict[str, Any] = field(default_factory=dict)
    kind: Literal["load_model"] = "load_model"


@dataclass(frozen=True)
class UnloadModel:
    model_key: str
    kind: Literal["unload_model"] = "unload_model"


@dataclass(frozen=True)
class RunNode:
    task_id: int
    node_id: str
    node_type: str  # 仅 "image" / "tts"
    model_key: str | None
    inputs: dict[str, Any]
    is_deterministic: bool = False
    kind: Literal["run_node"] = "run_node"


@dataclass(frozen=True)
class Abort:
    task_id: int
    node_id: str | None = None
    kind: Literal["abort"] = "abort"


@dataclass(frozen=True)
class Ping:
    kind: Literal["ping"] = "ping"


# ------------------------------------------------------------------
# image/TTS runner -> 主进程
# ------------------------------------------------------------------


@dataclass(frozen=True)
class Ready:
    """runner 子进程 event loop 起来后发的第一个消息（spec §4.2 生命周期图）。"""

    runner_id: str
    group_id: str
    gpus: list[int]
    kind: Literal["ready"] = "ready"


@dataclass(frozen=True)
class NodeResult:
    task_id: int
    node_id: str
    status: Literal["completed", "failed", "cancelled"]
    outputs: dict[str, Any] | None
    error: str | None
    duration_ms: int
    kind: Literal["node_result"] = "node_result"


@dataclass(frozen=True)
class NodeProgress:
    task_id: int
    node_id: str
    progress: float  # 0.0 ~ 1.0
    detail: str | None = None
    kind: Literal["node_progress"] = "node_progress"


@dataclass(frozen=True)
class ModelEvent:
    event: Literal["loaded", "unloaded", "load_failed"]
    model_key: str
    error: str | None = None
    kind: Literal["model_event"] = "model_event"


@dataclass(frozen=True)
class Pong:
    runner_id: str
    loaded_models: list[str] = field(default_factory=list)
    kind: Literal["pong"] = "pong"


# kind 字面量 -> dataclass 类的路由表
_KIND_TO_CLASS: dict[str, type] = {
    "load_model": LoadModel,
    "unload_model": UnloadModel,
    "run_node": RunNode,
    "abort": Abort,
    "ping": Ping,
    "ready": Ready,
    "node_result": NodeResult,
    "node_progress": NodeProgress,
    "model_event": ModelEvent,
    "pong": Pong,
}

# 类型注解仅供调用方做 isinstance / match —— 任意消息的联合类型
Message = (
    LoadModel | UnloadModel | RunNode | Abort | Ping
    | Ready | NodeResult | NodeProgress | ModelEvent | Pong
)


def default_format() -> str:
    """wire format：环境变量 NOUS_IPC_FORMAT，默认 msgpack。"""
    fmt = os.getenv("NOUS_IPC_FORMAT", "msgpack").strip().lower()
    return fmt if fmt in ("msgpack", "json") else "msgpack"


def encode(msg: Any, *, fmt: str | None = None) -> bytes:
    """把消息 dataclass 编成 bytes。"""
    fmt = fmt or default_format()
    payload = asdict(msg)
    if fmt == "json":
        return json.dumps(payload).encode("utf-8")
    import msgpack

    return msgpack.packb(payload, use_bin_type=True)


def decode(raw: bytes, *, fmt: str | None = None) -> Any:
    """把 bytes 解回对应的消息 dataclass。未知 kind 抛 ProtocolError。"""
    fmt = fmt or default_format()
    try:
        if fmt == "json":
            payload = json.loads(raw.decode("utf-8"))
        else:
            import msgpack

            payload = msgpack.unpackb(raw, raw=False)
    except Exception as e:  # noqa: BLE001 — 任何解码异常统一包成 ProtocolError
        raise ProtocolError(f"failed to decode {fmt} payload: {e}") from e

    if not isinstance(payload, dict) or "kind" not in payload:
        raise ProtocolError(f"decoded payload is not a tagged message: {payload!r}")
    kind = payload["kind"]
    cls = _KIND_TO_CLASS.get(kind)
    if cls is None:
        raise ProtocolError(f"unknown message kind: {kind!r}")
    # 只取该 dataclass 声明的字段，多余 key 忽略（向前兼容）
    known = {f.name for f in fields(cls)}
    kwargs = {k: v for k, v in payload.items() if k in known}
    try:
        return cls(**kwargs)
    except TypeError as e:
        raise ProtocolError(f"payload missing fields for {kind!r}: {e}") from e
