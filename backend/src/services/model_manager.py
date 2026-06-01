from __future__ import annotations

import asyncio
import importlib
import logging
import time
from pathlib import Path
from typing import Awaitable, Callable

from pydantic import BaseModel, ConfigDict, Field

from src.errors import ModelLoadError, ModelNotFoundError
from src.services.inference.base import InferenceAdapter
from src.services.inference.registry import ModelRegistry, ModelSpec
from src.services.gpu_allocator import GPUAllocator

logger = logging.getLogger(__name__)


def _modular_repo_from_components(resolved: dict) -> str:
    """从细粒度图组件推 HF-layout repo(含 model_index.json 的目录)。

    ModularPipeline.from_pretrained 要 repo(提供 config/scheduler + clip/vae)。依次从
    unet/clip/vae 组件文件向上找 model_index.json —— **unet 是 comfy 量化单文件(无 repo)时,
    从 clip/vae(指向 HF text_encoder/vae)推 repo**(PR-2:transformer 由桥接 override)。
    """
    for comp_key in ("diffusion_models", "clip", "vae"):
        spec = resolved.get(comp_key)
        if spec is None:
            continue
        f = Path(spec.file)
        for cand in (f.parent, f.parent.parent, f.parent.parent.parent):
            if (cand / "model_index.json").exists():
                return str(cand)
    # 全单文件无 HF repo → 用「架构参考整模型」(单文件装配 PR-2):借其 config/scheduler/tokenizer,
    # 三组件由桥接 override(build_bridged_*)灌单文件权重。架构取 unet 的 adapter_arch。
    unet = resolved.get("diffusion_models")
    arch = getattr(unet, "adapter_arch", None) or "flux2"
    ref = _reference_repo_for_arch(arch)
    if ref:
        return ref
    raise ValueError(
        f"modular 引擎需 HF-layout repo;组件均为单文件且找不到架构 {arch!r} 的参考整模型"
        f"(在 image/diffusers/ 放一个对应整模型作 config 参考)。"
    )


def _is_comfy_single_file_unet(unet_spec) -> bool:
    """unet 是 repo 外的 comfy 量化单文件(需桥接 override)而非 HF-layout transformer。

    HF-layout transformer 目录有 config.json;comfy 单文件(diffusion_models/flux/)没有。
    """
    return not (Path(unet_spec.file).parent / "config.json").exists()


def _is_standalone_single_file(spec) -> bool:
    """任意组件:文件是 repo 外单文件(parent 无 config.json)→ 需桥接 override。
    HF-layout 组件(diffusers/<m>/{transformer,text_encoder,vae}/)parent 有 config.json。"""
    return not (Path(spec.file).parent / "config.json").exists()


def _reference_repo_for_arch(arch: str) -> str | None:
    """架构 → 配置目录(单文件装配:借它的 tokenizer + 各组件 config)。

    **PR-B 起优先返回仓内 bundle**(`backend/configs/image_arch/<arch>/`,几 MB)——
    用户库可不再放参考整模型(18GB)即能跑同架构单文件。Fallback 扫 LOCAL_MODELS_PATH/image/diffusers/
    保留向后兼容(老用户仍能用)。

    支持架构:flux2(已 bundle)。新增 arch 经 PR-C 的 ImageArchSpec 注册表 + 在 configs/ 加 bundle 一并接入。
    """
    import json  # noqa: PLC0415
    from src.config import get_settings  # noqa: PLC0415

    arch_lower = (arch or "").lower()
    if not arch_lower:
        return None

    # 1. 仓内 bundled config(首选,几 MB,完全 self-contained)
    # model_manager.py 在 backend/src/services/ → parents[2] = backend/。
    backend_root = Path(__file__).resolve().parents[2]
    bundled = backend_root / "configs" / "image_arch" / arch_lower
    if (bundled / "transformer" / "config.json").is_file():
        return str(bundled)

    # 2. Fallback:扫 LOCAL_MODELS_PATH/image/diffusers/*/model_index.json(向后兼容)
    base = Path(get_settings().LOCAL_MODELS_PATH) / "image" / "diffusers"
    if not base.is_dir():
        return None
    hint = {"flux2": "flux2", "flux1": "flux", "ernie": "ernie"}.get(arch_lower, arch_lower)
    if not hint:
        return None
    for d in sorted(base.iterdir()):
        mi = d / "model_index.json"
        if not mi.is_file():
            continue
        try:
            cls = str(json.loads(mi.read_text()).get("_class_name", "")).lower()
        except Exception:  # noqa: BLE001
            continue
        if hint in cls:
            return str(d)
    return None

# Re-export so existing `from src.services.model_manager import
# ModelLoadError, ModelNotFoundError` keeps working (these moved to
# src.errors so they share the NousError envelope path).
__all__ = ["ModelLoadError", "ModelManager", "ModelNotFoundError"]


class LoadedModel(BaseModel):
    """Runtime entry per loaded model: spec + adapter instance + GPU placement
    + LRU bookkeeping. Mutable (touch() updates last_used)."""

    spec: ModelSpec
    adapter: InferenceAdapter
    gpu_index: int  # primary GPU (for single-GPU models)
    gpu_indices: list[int] = Field(default_factory=list)  # all GPUs (for tensor-parallel)
    loaded_at: float = Field(default_factory=time.monotonic)
    last_used: float = Field(default_factory=time.monotonic)

    # InferenceAdapter is a non-pydantic ABC instance; allow as field value.
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def touch(self) -> None:
        self.last_used = time.monotonic()


# PR-1 Task 6: components are lighter than full LoadedModel — they're just a loaded
# state_dict/module dict + metadata. Stored in ModelManager._components.
LoadedComponent = dict  # opaque to ModelManager: {_state_dict, spec, loaded_at, ...}


class ModelManager:
    """Unified model lifecycle manager: load, unload, evict, reference-count."""

    def __init__(self, registry: ModelRegistry, allocator: GPUAllocator) -> None:
        self._registry = registry
        self._allocator = allocator
        self._models: dict[str, LoadedModel] = {}
        self._references: dict[str, set[str]] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        # device=auto 粘性放置(#199):device-independent combo identity → 上次落的 gpu_index。
        # 同一工作流二次跑 auto 解析回同一张卡,避免 combo_key 翻卡 → cache miss → 重载。
        self._image_stick: dict[tuple, int] = {}
        # 反向:model_id → stick identity。unload/evict 时据此清 _image_stick,避免 stale
        # stick 指向已卸载/已满的卡 + 绕过 VRAM 守卫 → 反复 OOM(#210 回归修复)。
        self._image_stick_keys: dict[str, tuple] = {}
        # 正在 infer 的 model_id —— node-executor 在 adapter.infer 期间标记。unload/evict
        # 绝不卸载在用的(即使 force):denoise 在 to_thread 工作线程跑,卸载 = 释放正在用的
        # CUDA 权重 → 工作线程撞已释放显存 → segfault。
        self._in_use: set[str] = set()
        # Per-model load failures (set by background preload tasks or prior
        # failed load_model attempts). get_loaded_adapter raises ModelLoadError
        # when a record exists. Cleared on successful load.
        self._load_failures: dict[str, str] = {}
        # round3 #2:load_model 自动选卡(spec.gpu is None → get_best_gpu)后,实际落卡
        # index 是局部变量;OOM 时 load_model raise、还没写进 _models → get_or_load 拿不到
        # 真正 OOM 的卡,退成 evict_lru(None) 驱全局 LRU(可能驱了另一张没满的卡,OOM 的卡
        # 仍满 → 重试再 OOM → 永久毒化)。这里记每个 model 上次尝试的落卡,OOM 时据它精确驱逐。
        self._last_attempt_gpu: dict[str, int] = {}
        # PR-1 Task 6: component-level cache (parallel to _models). Old yaml-driven
        # adapters keep using _models; PR-2's ImageSampler will use _components.
        # Forward-ref the ComponentKey type as a tuple alias so the module-top
        # import stays clean (component_spec doesn't import ModelManager).
        from src.services.inference.component_spec import ComponentKey  # noqa: F401
        # PR-D4(2026-05-28):删 `_image_adapters` / `_image_adapter_locks` 双套
        # 路径。image adapter 走 derived model_id 入 `_models` 统一字典,LRU
        # 驱逐 / 引用计数 / 状态可见 / pid_map 全部 ModelManager 底层自动覆盖,
        # 不再单独管 image。derive 规则见 `_derive_image_model_id`。

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _lock_for(self, model_id: str) -> asyncio.Lock:
        return self._locks.setdefault(model_id, asyncio.Lock())

    def _instantiate_adapter(self, spec: ModelSpec) -> InferenceAdapter:
        """Dynamically import and instantiate adapter from spec.adapter_class dotted path.

        v2: passes `paths: dict[str, str]` to the adapter __init__. Single-component
        adapters (vLLM/SGLang/TTS) read `paths['main']`; image-class adapters
        read `paths['transformer']`, `paths['text_encoder']`, `paths['vae']`.

        For image specs, lora_paths is auto-injected from the lora_scanner
        unless the yaml entry already supplied one. This means yaml never
        has to enumerate individual LoRAs — drop a .safetensors into a
        configured LORA_PATHS dir and it's available next adapter load.
        """
        dotted = spec.adapter_class
        module_path, _, class_name = dotted.rpartition(".")
        if not module_path:
            raise ImportError(
                f"adapter_class '{dotted}' must be a fully-qualified dotted path"
            )
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)

        params = dict(spec.params)
        if spec.model_type == "image" and "lora_paths" not in params:
            # Inject ALL scanned LoRAs (no arch filter). Pre-existing
            # workflows that reference an "incompatible" LoRA name should
            # still be loadable here so the adapter can produce its own
            # friendly error at apply-time (image_diffusers.py:243 catches
            # "zero matching weights" and explains the architecture
            # mismatch). Filtering at injection-time silently breaks those
            # workflows with a confusing "not in registered lora_paths"
            # message — see post-mortem in PR #75 successor.
            #
            # The chip count on /api/v1/engines stays arch-aware (via
            # `count_loras_for_arches(accepts)` in engines.py) so the UI
            # still tells the truth about how many LoRAs are likely usable.
            from src.services.lora_scanner import get_lora_paths
            params["lora_paths"] = get_lora_paths()
        # accepts_lora_archs is yaml-only metadata, never passed to adapter
        params.pop("accepts_lora_archs", None)

        return cls(paths=spec.paths, **params)

    def _detect_vllm_gpus_for_adapter(self, adapter) -> list[int]:
        """Map the adapter's subprocess (and its children) to GPU indices
        via nvidia-smi. Runs AFTER load() — before that, vLLM hasn't
        allocated any devices yet."""
        try:
            # Collect pids belonging to this adapter: main process + descendants.
            root_pid = None
            proc = getattr(adapter, "_process", None)
            if proc is not None:
                root_pid = proc.pid
            adopted = getattr(adapter, "_adopted_pid", None)
            if adopted:
                root_pid = adopted
            if not root_pid:
                return []

            import subprocess
            # Descendants (children + grandchildren); tp>1 spawns worker procs.
            try:
                out = subprocess.run(
                    ["ps", "-o", "pid", "--no-headers", "--ppid", str(root_pid)],
                    capture_output=True, text=True, timeout=3,
                ).stdout
                pids: set[int] = {root_pid}
                for line in out.splitlines():
                    s = line.strip()
                    if s.isdigit():
                        pids.add(int(s))
                # One more hop (tp uses spawn → grandchildren)
                for child in list(pids):
                    out2 = subprocess.run(
                        ["ps", "-o", "pid", "--no-headers", "--ppid", str(child)],
                        capture_output=True, text=True, timeout=3,
                    ).stdout
                    for line in out2.splitlines():
                        s = line.strip()
                        if s.isdigit():
                            pids.add(int(s))
            except Exception:
                pids = {root_pid}

            result = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode != 0:
                return []
            gpu_result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5,
            )
            uuid_to_idx: dict[str, int] = {}
            for line in gpu_result.stdout.strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 2:
                    uuid_to_idx[parts[1]] = int(parts[0])

            hits: set[int] = set()
            for line in result.stdout.strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 2:
                    continue
                try:
                    pid_int = int(parts[0])
                except ValueError:
                    continue
                if pid_int in pids:
                    idx = uuid_to_idx.get(parts[1])
                    if idx is not None:
                        hits.add(idx)
            return sorted(hits)
        except Exception:
            return []

    def _detect_vllm_gpus(self, spec: ModelSpec) -> list[int]:
        """Detect which GPUs vLLM is using by checking nvidia-smi."""
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode != 0:
                return [0]
            gpu_result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5,
            )
            uuid_to_idx: dict[str, int] = {}
            for line in gpu_result.stdout.strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 2:
                    uuid_to_idx[parts[1]] = int(parts[0])
            vllm_gpus: set[int] = set()
            for line in result.stdout.strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 2:
                    idx = uuid_to_idx.get(parts[1])
                    if idx is not None:
                        vllm_gpus.add(idx)
            return sorted(vllm_gpus) if vllm_gpus else [0]
        except Exception:
            return [0]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_loaded(self, model_id: str) -> bool:
        entry = self._models.get(model_id)
        return entry is not None and entry.adapter.is_loaded

    def get_adapter(self, model_id: str) -> InferenceAdapter | None:
        entry = self._models.get(model_id)
        if entry is None:
            return None
        entry.touch()
        return entry.adapter

    async def get_loaded_adapter(self, model_id: str) -> InferenceAdapter:
        """Get adapter, loading on demand if needed.

        v2 unified path for all node-layer adapter calls. Replaces the
        4-line "get_adapter → check is_loaded → load_model → get_adapter"
        pattern duplicated across nodes/llm.py + nodes/audio.py.

        Raises:
            ModelNotFoundError: model_id has no spec (yaml + scan miss).
                                Maps to HTTP 404.
            ModelLoadError:     load failed (recorded in `_load_failures`).
                                Maps to HTTP 503.
        """
        # Fast path: already loaded
        adapter = self.get_adapter(model_id)
        if adapter is not None and adapter.is_loaded:
            return adapter

        # Check for prior failure (set by background preload or a previous
        # load_model attempt). Don't retry indefinitely — admin must
        # restart or call load_model explicitly to clear.
        if model_id in self._load_failures:
            raise ModelLoadError(model_id, self._load_failures[model_id])

        # Lazy load. load_model raises ValueError("Unknown model") on
        # spec miss; convert to typed ModelNotFoundError.
        try:
            await self.load_model(model_id)
        except ValueError as e:
            if "Unknown model" in str(e):
                raise ModelNotFoundError(model_id) from e
            # Other ValueErrors (e.g. adapter-without-config) are load failures
            self._load_failures[model_id] = str(e)
            raise ModelLoadError(model_id, str(e)) from e
        except Exception as e:
            self._load_failures[model_id] = f"{type(e).__name__}: {e}"
            raise ModelLoadError(model_id, str(e)) from e

        adapter = self.get_adapter(model_id)
        if adapter is None or not adapter.is_loaded:
            self._load_failures[model_id] = "load_model returned but adapter is not loaded"
            raise ModelLoadError(model_id, self._load_failures[model_id])
        return adapter

    @staticmethod
    def _is_oom(exc: BaseException) -> bool:
        """判定异常是不是 CUDA OOM。

        不能在模块顶层 import torch（runner venv 测试里 torch 是 MagicMock，
        且 ModelManager 应能在无 torch 的纯逻辑测试中跑）。改用类名 + 文本判定：
        torch.cuda.OutOfMemoryError 的类名就是 'OutOfMemoryError'；其它库的
        OOM 一般文案里有 'out of memory'。
        """
        name = type(exc).__name__
        if "OutOfMemoryError" in name:
            return True
        return "out of memory" in str(exc).lower()

    async def get_or_load(
        self,
        model_id: str,
        adapter_factory: Callable[[ModelSpec], InferenceAdapter] | None = None,
    ) -> InferenceAdapter:
        """Get adapter, loading on demand with OOM-evict-retry (spec §4.3).

        在 `get_loaded_adapter` 的 lazy-load 之上加一层 OOM 韧性：首次 load 撞
        CUDA OOM → evict 同 GPU 的 LRU 非 resident / 非 referenced 模型 → 重试
        一次。重试仍 OOM（或非 OOM 异常）→ 落 `_load_failures` 并 raise
        ModelLoadError。**runner 子进程的 node-executor 唯一的加载入口** —— OOM
        重试逻辑全收在这里，调用方不感知。

        Raises:
            ModelNotFoundError: model_id 无 spec (HTTP 404).
            ModelLoadError:     load 失败 / 二次 OOM (HTTP 503).
        """
        # Fast path: already loaded
        adapter = self.get_adapter(model_id)
        if adapter is not None and adapter.is_loaded:
            return adapter
        # Prior failure check —— 不重试
        if model_id in self._load_failures:
            raise ModelLoadError(model_id, self._load_failures[model_id])

        spec = self._registry.get(model_id)
        if spec is None:
            spec = self._registry.add_from_scan(model_id)
        if spec is None:
            raise ModelNotFoundError(model_id)

        last_err: BaseException | None = None
        for attempt in range(2):
            try:
                await self.load_model(model_id, adapter_factory=adapter_factory)
                loaded = self.get_adapter(model_id)
                if loaded is None or not loaded.is_loaded:
                    self._load_failures[model_id] = (
                        "load_model returned but adapter is not loaded"
                    )
                    raise ModelLoadError(model_id, self._load_failures[model_id])
                self._load_failures.pop(model_id, None)
                return loaded
            except ModelLoadError:
                raise
            except Exception as e:  # noqa: BLE001
                last_err = e
                if self._is_oom(e) and attempt == 0:
                    # 配置固定卡用 spec.gpu;自动分配用 load_model 记下的实际落卡
                    # (round3 #2)—— 否则 evict_lru(None) 驱全局、放跑了真正 OOM 的卡。
                    if isinstance(spec.gpu, int):
                        gpu = spec.gpu
                    else:
                        gpu = self._last_attempt_gpu.get(model_id)
                    evicted = await self.evict_lru(gpu_index=gpu)
                    logger.warning(
                        "get_or_load(%r): OOM on first load, evicted %r, retrying",
                        model_id, evicted,
                    )
                    continue
                msg = (
                    f"OOM after evict: {e}" if self._is_oom(e)
                    else f"{type(e).__name__}: {e}"
                )
                self._load_failures[model_id] = msg
                raise ModelLoadError(model_id, msg) from e
        # 防御：循环必定 return 或 raise
        self._load_failures[model_id] = f"{type(last_err).__name__}: {last_err}"
        raise ModelLoadError(model_id, self._load_failures[model_id])

    def get_references(self, model_id: str) -> set[str]:
        return set(self._references.get(model_id, set()))

    def add_reference(self, model_id: str, ref_id: str) -> None:
        self._references.setdefault(model_id, set()).add(ref_id)

    def remove_reference(self, model_id: str, ref_id: str) -> None:
        refs = self._references.get(model_id)
        if refs:
            refs.discard(ref_id)

    async def load_model(
        self,
        model_id: str,
        adapter_factory: Callable[[ModelSpec], InferenceAdapter] | None = None,
    ) -> None:
        """Load *model_id* onto the best available GPU.

        Parameters
        ----------
        adapter_factory:
            Optional callable ``(ModelSpec) -> InferenceAdapter``.  When
            omitted the adapter is instantiated via dynamic import from
            ``spec.adapter_class``.
        """
        spec = self._registry.get(model_id)
        if spec is None:
            # Fallback: model wasn't in models.yaml at startup but the
            # disk scanner discovered it later (auto-detected LLM / VL).
            # Synthesize a ModelSpec on the fly so newly-dropped LLM
            # checkpoints don't require a yaml edit + restart.
            spec = self._registry.add_from_scan(model_id)
        if spec is None:
            raise ValueError(f"Unknown model: {model_id!r}")

        async with self._lock_for(model_id):
            if self.is_loaded(model_id):
                self._models[model_id].touch()
                return

            # Determine device and GPU indices
            detect_after_load = False
            if spec.gpu is not None:
                # Use configured GPU(s)
                if isinstance(spec.gpu, list):
                    gpu_indices = spec.gpu
                    gpu_index = spec.gpu[0]
                else:
                    gpu_indices = [spec.gpu]
                    gpu_index = spec.gpu
            elif spec.vram_mb > 0:
                gpu_index = self._allocator.get_best_gpu(spec.vram_mb)
                gpu_indices = [gpu_index] if gpu_index >= 0 else []
                # Image adapters use diffusers `enable_model_cpu_offload` —
                # weights live in CPU RAM and only the active block enters
                # GPU during inference. Allocator's "no GPU has 24GB free"
                # check (`get_best_gpu` → -1) doesn't mean we can't run; it
                # means we need to share an existing GPU. Fall back to GPU 0
                # so /api/v1/engines doesn't surface the placeholder -1
                # ("running · GPU -1" in the UI). This matches what
                # `enable_model_cpu_offload(gpu_id=0)` will use anyway.
                if gpu_index < 0 and spec.model_type == "image":
                    logger.warning(
                        "Image model %r: allocator found no GPU with %dMB free; "
                        "falling back to GPU 0 (cpu_offload mode shares it).",
                        model_id, spec.vram_mb,
                    )
                    gpu_index = 0
                    gpu_indices = [0]
                detect_after_load = False
            else:
                # External service (e.g. vLLM) — detect GPUs AFTER the process
                # has actually claimed them (pre-load nvidia-smi returns nothing).
                gpu_index = 0
                gpu_indices = []
                detect_after_load = True

            device = f"cuda:{gpu_index}" if gpu_index >= 0 else "cpu"
            # round3 #2:记录本次实际落卡,供 get_or_load 在 OOM 时精确驱逐该卡
            # (detect_after_load 的外部服务 gpu_index 是占位 0,记了也无害)。
            if gpu_index >= 0:
                self._last_attempt_gpu[model_id] = gpu_index

            # Build adapter
            if adapter_factory is not None:
                adapter = adapter_factory(spec)
            else:
                adapter = self._instantiate_adapter(spec)

            await adapter.load(device)

            if detect_after_load:
                gpu_indices = self._detect_vllm_gpus_for_adapter(adapter) or [0]
                gpu_index = gpu_indices[0] if gpu_indices else 0

            self._models[model_id] = LoadedModel(
                spec=spec,
                adapter=adapter,
                gpu_index=gpu_index,
                gpu_indices=gpu_indices,
            )
            self._references.setdefault(model_id, set())
            # Clear prior failure record on successful load; lets admin
            # retry by re-calling load_model after fixing the underlying issue.
            self._load_failures.pop(model_id, None)
            logger.info("Loaded model %r on %s", model_id, device)

    def _model_id_for_adapter(self, adapter: InferenceAdapter) -> str | None:
        """given adapter instance → its model_id(_models 里反查,n 很小)。"""
        for mid, entry in self._models.items():
            if entry.adapter is adapter:
                return mid
        return None

    def mark_adapter_in_use(self, adapter: InferenceAdapter) -> None:
        """node-executor 在 adapter.infer 前调:标记其 model_id 正在用,unload/evict 跳过。"""
        mid = self._model_id_for_adapter(adapter)
        if mid is not None:
            self._in_use.add(mid)

    def release_adapter(self, adapter: InferenceAdapter) -> None:
        """infer 收尾(成功/异常)调:清正在用标记。"""
        mid = self._model_id_for_adapter(adapter)
        if mid is not None:
            self._in_use.discard(mid)

    async def unload_model(self, model_id: str, force: bool = False) -> None:
        """Unload *model_id*.

        If the model is *resident* or has active references it will NOT be
        unloaded unless *force=True*.
        """
        async with self._lock_for(model_id):
            entry = self._models.get(model_id)
            if entry is None:
                return

            # in-use 是硬守卫,**强于 force**:绝不卸载正在 infer 的 adapter —— denoise 在
            # to_thread 工作线程跑,卸载会释放它正在用的 CUDA 权重 → segfault。
            if model_id in self._in_use:
                logger.warning(
                    "Skipping unload of in-use model %r(正在 infer,卸载会 segfault)", model_id)
                return

            if not force:
                if entry.spec.resident:
                    logger.debug("Skipping unload of resident model %r", model_id)
                    return
                refs = self._references.get(model_id, set())
                if refs:
                    logger.debug(
                        "Skipping unload of referenced model %r (refs=%s)",
                        model_id,
                        refs,
                    )
                    return

            entry.adapter.unload()
            del self._models[model_id]
            # 清 device=auto 粘性(#210 回归):adapter 没了,stick 不能再指向它的卡并跳 VRAM
            # 守卫 —— 否则下次同工作流粘回已卸载/可能已满的卡反复 OOM。
            sk = self._image_stick_keys.pop(model_id, None)
            if sk is not None:
                self._image_stick.pop(sk, None)
            logger.info("Unloaded model %r", model_id)

    @property
    def loaded_model_ids(self) -> list[str]:
        return [mid for mid, entry in self._models.items() if entry.adapter.is_loaded]

    def get_pid_map(self) -> dict[int, str]:
        """Return {pid: model_id} for all managed processes that have a PID."""
        result: dict[int, str] = {}
        for mid, entry in self._models.items():
            pid = getattr(entry.adapter, "pid", None)
            if pid is not None:
                result[pid] = mid
        return result

    def get_status(self) -> dict:
        """Return current model manager status."""
        return {
            "loaded": self.loaded_model_ids,
            "references": {k: list(v) for k, v in self._references.items() if v},
            "last_used": {
                mid: entry.last_used for mid, entry in self._models.items()
            },
        }

    def get_model_dependencies(self, workflow: dict) -> list[dict]:
        """Extract model dependencies from workflow nodes."""
        deps: list[dict] = []
        seen: set[str] = set()
        for node in workflow.get("nodes", []):
            node_type = node.get("type", "")
            data = node.get("data", {})
            model_key: str | None = None
            if node_type == "tts_engine":
                model_key = data.get("engine")
            elif node_type == "llm":
                model_key = data.get("model_key")
            if model_key and model_key not in seen:
                spec = self._registry.get(model_key)
                if spec is not None:
                    seen.add(model_key)
                    deps.append({"key": model_key, "type": spec.model_type})
        return deps

    async def preload_residents(
        self,
        on_loaded: Callable[[str], Awaitable[None]] | None = None,
    ) -> None:
        """Startup preload of resident models, ordered by `preload_order` (spec 4.2).

        遍历 registry 里所有 `resident:true` 的 spec，按 `preload_order` 升序
        load。`preload_order` 为 None 的排在最后（保持 registry 的 FIFO 顺序）。

        **Fail-soft（spec 4.3）**：单个模型 `load_model` 抛任何异常 → 把原因写进
        `_load_failures[model_id]` + 继续下一个模型，**绝不向上抛**。这样某个
        resident 模型 OOM / 文件损坏不会阻断 API server 启动，也不会阻断后面
        其它 resident 模型的 preload。失败的模型在 `/health` 的 `load_failures`
        里可见，Dashboard 据此显示 degraded banner + Retry。

        Parameters
        ----------
        on_loaded:
            可选回调，每个模型成功 load 后以 `model_id` 调用一次。`main.py` 用它
            做「invalidate engines/models cache + 推 ws/models 事件」。回调本身
            抛异常会被吞掉（best-effort，不影响 preload 流程）。

        LLM 类模型不在此处 preload —— vLLM 的 spawn / health 是 LLM Runner 的
        职责（spec 4.2），走另一条路。本方法只处理 image / tts 等进 image/TTS
        runner 的 resident 模型。
        """
        residents = [s for s in self._registry.specs if s.resident and s.model_type != "llm"]
        # 升序 key：preload_order 有值的在前（按值升序），None 的统一排到最后。
        # (0, order) < (1, 0) 保证所有 None 都在所有有值的之后。
        residents.sort(
            key=lambda s: (1, 0) if s.preload_order is None else (0, s.preload_order)
        )
        if residents:
            logger.info(
                "Preloading %d resident model(s) in order: %s",
                len(residents), [s.id for s in residents],
            )
        for spec in residents:
            try:
                await self.load_model(spec.id)
            except Exception as e:  # noqa: BLE001 — fail-soft is the whole point
                detail = f"{type(e).__name__}: {e}"
                self._load_failures[spec.id] = detail
                logger.warning("Resident preload failed for %s: %s", spec.id, detail)
                continue
            logger.info("Resident preload succeeded: %s", spec.id)
            if on_loaded is not None:
                try:
                    await on_loaded(spec.id)
                except Exception:  # noqa: BLE001 — callback is best-effort
                    logger.exception("preload_residents on_loaded callback failed for %s", spec.id)

    async def check_idle_models(self) -> None:
        """Unload models that have been idle too long with no references."""
        now = time.monotonic()
        to_unload: list[str] = []
        for mid, entry in list(self._models.items()):
            if entry.spec.resident:
                continue
            if self._references.get(mid):
                continue
            ttl = entry.spec.ttl_seconds
            if ttl <= 0:
                continue
            if now - entry.last_used > ttl:
                to_unload.append(mid)
        for mid in to_unload:
            logger.info("TTL expired: unloading %s", mid)
            await self.unload_model(mid)

    async def evict_lru(self, gpu_index: int | None = None) -> str | None:
        """Evict the least-recently-used non-resident, non-referenced model.

        Parameters
        ----------
        gpu_index:
            When provided, only consider models loaded on that GPU.

        Returns
        -------
        The evicted model_id, or ``None`` if nothing was evicted.
        """
        candidates = [
            entry
            for mid, entry in self._models.items()
            if not entry.spec.resident
            and not self._references.get(mid)
            and mid not in self._in_use  # 不驱逐正在 infer 的(否则 segfault)
            and (gpu_index is None or entry.gpu_index == gpu_index)
        ]

        if not candidates:
            return None

        lru = min(candidates, key=lambda e: e.last_used)
        model_id = lru.spec.id
        await self.unload_model(model_id, force=True)
        logger.info("Evicted LRU model %r from gpu %s", model_id, lru.gpu_index)
        return model_id

    # --- PR-1 Task 6: component-level cache APIs -----------------------------
    #
    # Per spec §5.5 these coexist with the legacy load_model/is_loaded/unload_model.
    # PR-2's ImageSampler will exercise these directly without going through yaml.

    # PR-4 收官:legacy 组件 L1 缓存(_component_lock_for / is_component_loaded /
    # get_or_load_component / _load_component_impl / _load_component_module /
    # unload_component / _base_spec)已删 —— 仅服务自写 DiffusersImageBackend 引擎,
    # 已随 image_diffusers/image_sampler 删除。modular 引擎走 ComponentsManager +
    # build_bridged_transformer(image_modular),不用此缓存。

    _VRAM_EST_MB = {"diffusion_models": 18000, "clip": 6000, "vae": 1000}

    def _resolve_component_device(self, spec):
        """Resolve device='auto' → 'cuda:N' via allocator. Returns a NEW spec
        (model_copy keeps validators) so the original descriptor is untouched."""
        if spec.device != "auto":
            return spec
        # 优先用真实文件大小(×1.3)估需求 —— 比固定 _VRAM_EST_MB 表准:anima 2B(~10GB)≠
        # Flux2 9B(18GB),套固定表会把 anima 当 18GB 误拦/落错卡。拿不到文件则回退表。
        need = self._VRAM_EST_MB.get(spec.kind, 8000)
        try:
            import os
            sz_mb = int(os.path.getsize(spec.file) / (1024 * 1024) * 1.3)
            if sz_mb > 0:
                need = sz_mb
        except (OSError, AttributeError, TypeError):
            pass
        idx = self._allocator.get_best_gpu(need)
        resolved = f"cuda:{idx}" if idx >= 0 else "cpu"
        return spec.model_copy(update={"device": resolved})

    @staticmethod
    def _free_vram_mb(device: str) -> int | None:
        """目标卡空闲显存(MB)。'cpu'/'auto'/无 GPU/查询失败 → None(跳过保护)。
        用 nvidia-smi(避免 import torch);best-effort,失败不阻塞。"""
        if not device.startswith("cuda:"):
            return None
        try:
            idx = int(device.split(":", 1)[1])
        except (ValueError, IndexError):
            return None
        try:
            import subprocess
            out = subprocess.run(
                ["nvidia-smi", f"--id={idx}",
                 "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            if out.returncode != 0:
                return None
            return int(out.stdout.strip().splitlines()[0])
        except Exception:  # noqa: BLE001 — best-effort,任何失败都跳过保护
            return None

    @staticmethod
    def _estimate_image_vram_mb(resolved: dict) -> int | None:
        """粗估整模型显存需求(MB)= 三组件 resident bytes 之和 * 余量系数。任一文件不存在
        (纯逻辑测试用 stub 路径)→ None(无法估,跳过保护)。

        fp8 weight-only(torchao):transformer + clip 权重 fp8 存储 ≈ bf16 文件的一半,
        所以这两件按 file bytes * 0.5 估(否则按 bf16 文件字节会高估 → 误拦本可装下的 fp8)。
        """
        import os
        total = 0
        for k in ("diffusion_models", "clip", "vae"):
            try:
                sz = os.path.getsize(resolved[k].file)
            except OSError:
                return None
            # fp8 量化只作用于 transformer/clip(vae 不量化)
            if k in ("diffusion_models", "clip") and (resolved[k].dtype or "").lower().startswith("fp8"):
                sz //= 2
            total += sz
        # LoRA 权重注入 transformer → 真占显存,旧估算漏了(守卫会低估 → 放过实际会 OOM 的装载)。
        # path 有且文件在才加(name-only / 不存在的不计)。
        for lora in (getattr(resolved["diffusion_models"], "loras", None) or []):
            lp = getattr(lora, "path", None)
            if lp:
                try:
                    total += os.path.getsize(lp)
                except OSError:
                    pass
        # 1.3x 余量(激活/中间张量;含 VAE decode 峰值的粗略 headroom);bytes → MB
        return int(total / (1024 * 1024) * 1.3)

    @staticmethod
    def _explain_image_combo_key(combo_key: tuple) -> dict:
        """把 combo_key 拆成 readable dict 用于日志/调试,让 cache miss 时一眼看出
        哪个字段变化导致 derived_id 不同。PR-D5 诊断 cache 命中率用。

        combo_key shape:
          (pipeline_class, offload, transformer_key, clip_key, vae_key)
        每个 component_key shape:
          (file, device, dtype, frozenset[(lora_name, strength), ...])
        """
        def _explain_comp(ck: tuple) -> dict:
            if not isinstance(ck, tuple) or len(ck) < 4:
                return {"raw": repr(ck)}
            file, device, dtype, loras = ck
            return {
                "file": file,
                "device": device,
                "dtype": dtype,
                "loras": sorted(f"{n}@{s}" for n, s in (loras or set())),
            }
        if not isinstance(combo_key, tuple) or len(combo_key) < 5:
            return {"raw": repr(combo_key)}
        pclass, offload, t_key, c_key, v_key = combo_key
        return {
            "pipeline_class": pclass,
            "offload": offload,
            "transformer": _explain_comp(t_key),
            "clip": _explain_comp(c_key),
            "vae": _explain_comp(v_key),
        }

    @staticmethod
    def _derive_image_model_id(combo_key: tuple) -> str:
        """combo_key = (pipeline_class, offload, transformer_key, clip_key, vae_key)
        → 一个稳定可读的 model_id 串,作为 `_models` 字典的键。

        命名:`image:<pipeline_class>:<transformer_basename>:<short_hash>`
        - pipeline_class 让人看出引擎家族(Flux2Klein / Anima);
        - transformer_basename 是单文件名或 HF repo 末段,辨识具体模型;
        - short_hash 兜底 dtype/offload/LoRA 等剩余差异,**保证 1-1 对应** combo_key。

        derived id 不进 yaml registry — 它只活在 `_models` 运行时字典里,被
        evict_lru / loaded_model_ids / unload_model / get_pid_map 自动消费。
        """
        import hashlib
        pipeline_class, offload, t_key, *_rest = combo_key
        # transformer key shape: (file, device, dtype, frozenset(loras))
        t_file = (t_key[0] if isinstance(t_key, tuple) and t_key else "unknown") or "unknown"
        # 取文件名末段(/path/to/Flux2-Klein-9B-True-v2-bf16.safetensors → Flux2-...-bf16)
        from os.path import basename, splitext
        t_base = splitext(basename(str(t_file)))[0] or "main"
        # 短 hash 区分 dtype/offload/clip/vae/LoRA 等剩余维度
        payload = repr(combo_key).encode("utf-8")
        short_hash = hashlib.sha256(payload).hexdigest()[:8]
        return f"image:{pipeline_class}:{t_base}:{short_hash}"

    def _synthesize_image_spec(
        self,
        model_id: str,
        adapter_class_path: str,
        target_device: str,
        vram_mb: int,
        pipeline_class: str,
        source_files: list[str] | None = None,
    ) -> "ModelSpec":
        """构造 in-flight ModelSpec 给 derived image adapter 用 — 不入 yaml registry,
        只让 LoadedModel.spec 字段有合规对象,这样 evict_lru / loaded_model_ids
        都能正常工作(它们都读 entry.spec.resident / entry.spec.id 等字段)。

        resident=False 让 image adapter 默认可 LRU 驱逐;ttl_seconds 设极大值
        (image adapter 不靠 TTL 释放,靠 LRU + 手动 unload + reference)。

        source_files:组装该 adapter 的源组件文件(unet/clip/vae)。image adapter
        加载在 runner 子进程,主进程靠 `loaded_models_snapshot()` → Pong 上报后,
        要用这些文件把 runner 里的 adapter 映射回引擎库卡片(card 是文件,adapter
        是 combo hash id)。存进 params 随 snapshot 一起过进程边界。
        """
        return ModelSpec(
            id=model_id,
            model_type="image",
            adapter_class=adapter_class_path,
            paths={},  # paths 已被 adapter 内部 build_bridged_* 消化,registry 不再需要
            vram_mb=vram_mb,
            params={"pipeline_class": pipeline_class, "source_files": source_files or []},
            resident=False,
            ttl_seconds=10**9,  # 不靠 TTL 回收(LRU + 手动 unload 才是主路径)
        )

    def loaded_models_snapshot(self) -> list[dict]:
        """当前已加载模型的结构化快照,用于跨进程上报(runner 子进程 → 主进程,
        经 Pong)。image/tts adapter 真加载在 runner 自己的 ModelManager._models 里,
        主进程的 _models 看不到 —— runner 每次 Pong 带上这份快照,主进程聚合后供
        `/image-cache`、系统状态「已加载模型」、引擎库 loaded 视图读(单一真相来源)。

        每条字段都为「过进程边界 + 还原 UI」够用:model_id(combo hash)、model_type、
        gpu、vram、pipeline_class、source_files(映射回引擎卡)、last_used_ago_sec。
        """
        now = time.monotonic()
        out: list[dict] = []
        for mid, entry in self._models.items():
            params = entry.spec.params or {}
            out.append({
                "model_id": mid,
                "model_type": entry.spec.model_type,
                "gpu_index": entry.gpu_index,
                "gpu_indices": list(entry.gpu_indices),
                "vram_mb": entry.spec.vram_mb,
                "pipeline_class": params.get("pipeline_class"),
                "source_files": list(params.get("source_files") or []),
                "last_used_ago_sec": round(now - entry.last_used, 2),
            })
        return out

    async def get_or_load_image_adapter(self, components: dict, pipeline_class: str = "Flux2KleinPipeline", on_event=None, offload: str = "none"):
        """PR-4 entry for the runner component path. Resolves auto devices,
        loads/reuses base modules via the component L1 cache, assembles (or
        reuses) a DiffusersImageBackend keyed by the full 3-component combo.

        OOM resilience: on first-load CUDA OOM, evict legacy LRU then retry once.

        PR-5a: on_event(key, state, error) — 可选异步回调，每个组件触发：
          - "loading": 开始加载 base 模块之前
          - "loaded":  加载成功后（combo cache HIT 时三个组件都触发）
          - "failed":  加载抛异常时（随后 re-raise）
        on_event=None 时行为与 PR-4 完全一致（_emit 是 no-op）。
        """
        from src.services.inference.component_spec import to_component_key, component_state_key

        async def _emit(spec, state, error=None):
            if on_event is not None:
                await on_event(component_state_key(spec), state, error)

        resolved = {k: self._resolve_component_device(s) for k, s in components.items()}
        # device=auto 粘性放置(#199 根治):同一工作流(同 file/dtype/loras)二次跑,把
        # diffusion_models 的 auto 解析粘回上次落的卡 —— 否则 get_best_gpu 按「此刻最空」
        # 会翻卡(run1 占了 A,run2 落 B)→ combo_key 变 → cache miss → 同模型重载(70G 累积)。
        # clip/vae 随 target(下方强制),所以只粘 diffusion_models 即可。
        stick_key = self._image_stick_identity(components, pipeline_class, offload)
        sticky = (components["diffusion_models"].device == "auto"
                  and stick_key in self._image_stick)
        if sticky:
            resolved["diffusion_models"] = resolved["diffusion_models"].model_copy(
                update={"device": f"cuda:{self._image_stick[stick_key]}"})
        # 整模型单卡不变式(spec 2026-05-21 rev 2):device=auto 会让三组件各自 resolve
        # 到不同卡 —— 以 unet 解析出的卡为准,强制 clip/vae 落同一张卡。
        target = resolved["diffusion_models"].device
        for k in ("clip", "vae"):
            if resolved[k].device != target:
                resolved[k] = resolved[k].model_copy(update={"device": target})
        # LLM 卡保护:目标卡空闲显存不足(常驻 LLM 占着)→ 装载前清晰报错,不静默 OOM。
        # PR-D:offload=cpu 时跳过这检查 — accelerate hooks 会按需挪 CPU,峰值 VRAM 远低于估算。
        # 粘性命中(sticky):预期 cache hit、不需新显存 —— 跳过守卫,否则「卡已被自己占满」
        # 会误判 OOM。真需重载(被 evict 过)时下游 OOM-retry 兜底。
        if offload == "none" and not sticky:
            free_mb = self._free_vram_mb(target)
            need_mb = self._estimate_image_vram_mb(resolved)
            if free_mb is not None and need_mb is not None and free_mb < need_mb:
                raise RuntimeError(
                    f"{target} 空闲显存不足({free_mb}MB < 约需 {need_mb}MB)—— "
                    f"该卡可能被常驻 LLM 占用。换张卡(device)、用更低精度(fp8)、"
                    f"或启用 offload=cpu(让大模型自动倒换 CPU)。"
                )
        # combo_key 包含 offload:不同 offload 模式是不同的 pipe 实例(enable_model_cpu_offload 改了内部 hook,
        # 不能复用为 offload=none 的 adapter)。
        combo_key = (pipeline_class, offload) + tuple(
            to_component_key(resolved[k]) for k in ("diffusion_models", "clip", "vae"))
        # PR-anima-6 engine 集成:pipeline_class="AnimaPipeline" → 走 AnimaImageBackend
        # (Anima 是 2B 自定义 DiT,跟 Flux2KleinPipeline 走不同路径)。
        if pipeline_class == "AnimaPipeline":
            adapter = await self._get_or_load_anima_adapter(
                resolved, combo_key, target, _emit, offload)
        else:
            adapter = await self._get_or_load_modular_adapter(
                resolved, combo_key, pipeline_class, target, _emit, offload)
        # 记住这工作流落的卡 —— 下次 auto 粘回来(load 与 cache hit 都刷新,target 即当前卡)。
        if target.startswith("cuda:"):
            self._image_stick[stick_key] = int(target.split(":", 1)[1])
            # 反向登记,unload/evict 时据此清 stick(防 stale)。
            self._image_stick_keys[self._derive_image_model_id(combo_key)] = stick_key
        return adapter

    @staticmethod
    def _image_stick_identity(components: dict, pipeline_class: str, offload: str) -> tuple:
        """device-independent combo identity:同一工作流(同 file/dtype/loras/pipeline/offload)
        无论解析到哪张卡都同一 identity。用作 _image_stick 的 key,让 device=auto 二次跑粘回
        同卡 → combo_key 稳定 → cache hit。"""
        parts = []
        for k in ("diffusion_models", "clip", "vae"):
            s = components[k]
            loras = frozenset(
                (lo.name, float(lo.strength)) for lo in (getattr(s, "loras", None) or []))
            parts.append((s.file, s.dtype, loras))
        return (pipeline_class, offload, tuple(parts))

    async def _get_or_load_anima_adapter(self, resolved, combo_key, target, _emit, offload: str):
        """Anima(2B 自定义 DiT)adapter 装配 —— spec 2026-05-26-anima-port,PR-anima-6 集成。

        跟 _get_or_load_modular_adapter 不同:不走桥接 build_bridged_*(那是 Flux2 转换 ComfyUI
        单文件给 diffusers 用的),anima 直接由 arch_anima.AnimaPipeline.from_components 装配
        anima 单文件 + qwen3 + qwen-image VAE。tokenizer 路径走 env NOUS_ANIMA_QWEN_TOKENIZER。
        """
        from src.services.inference.image_anima import AnimaImageBackend  # noqa: PLC0415

        model_id = self._derive_image_model_id(combo_key)
        async with self._lock_for(model_id):
            # cache hit → touch LRU + return
            entry = self._models.get(model_id)
            if entry is not None:
                logger.info("image adapter HIT id=%s (anima)", model_id)
                entry.touch()
                for k in ("diffusion_models", "clip", "vae"):
                    await _emit(resolved[k], "loaded")
                return entry.adapter
            # cache miss — dump combo_key 字段,方便复盘是哪个字段导致不复用
            logger.info(
                "image adapter MISS id=%s (anima) combo=%s",
                model_id, self._explain_image_combo_key(combo_key),
            )

            for k in ("diffusion_models", "clip", "vae"):
                await _emit(resolved[k], "loading")
            try:
                paths = {
                    "transformer": resolved["diffusion_models"].file,
                    "text_encoder": resolved["clip"].file,
                    "vae": resolved["vae"].file,
                }
                adapter = AnimaImageBackend(
                    paths=paths,
                    device=target,
                    dtype=resolved["diffusion_models"].dtype,
                    pipeline_class="AnimaPipeline",
                    offload=offload,
                )
                await adapter.load(target)
                # 预热在首次 infer 时 lazy(_ensure_pipe);不像 Flux2 在装 adapter 时就构建 pipe
                # —— anima pipe 装配 = 4.5s 加载,留首次 infer 一起付,UX 不卡 startup。
            except Exception as e:  # noqa: BLE001
                for k in ("diffusion_models", "clip", "vae"):
                    await _emit(resolved[k], "failed", f"{type(e).__name__}: {e}")
                raise

            for k in ("diffusion_models", "clip", "vae"):
                await _emit(resolved[k], "loaded")
            gpu_idx = int(target.split(":")[1]) if target.startswith("cuda:") else 0
            self._models[model_id] = LoadedModel(
                spec=self._synthesize_image_spec(
                    model_id=model_id,
                    adapter_class_path="src.services.inference.image_anima.AnimaImageBackend",
                    target_device=target,
                    vram_mb=self._estimate_image_vram_mb(resolved) or 0,
                    pipeline_class="AnimaPipeline",
                    source_files=[resolved[k].file for k in ("diffusion_models", "clip", "vae")],
                ),
                adapter=adapter,
                gpu_index=gpu_idx,
                gpu_indices=[gpu_idx],
            )
            return adapter

    async def get_or_load_seedvr2_adapter(
        self,
        model_dir: str,
        dit_model: str | None = None,
        vae_model: str | None = None,
        device: str = "cuda",
    ):
        """SeedVR2 超分 adapter 装配/复用 —— 跟 anima/modular 不同:SeedVR2 不是「三组件
        (diffusion_models/clip/vae)」模型,是「DiT + 专用 VAE 整套自带」的上采样器(by
        model_dir,非 component combo)。所以走**独立 by-key 路径**,不经 component-combo
        机制(get_or_load_image_adapter 那套 device-auto/sticky/三组件强制同卡都不适用)。

        但仍登记进统一 `_models` 字典 → 复用 LRU 驱逐 / loaded_models_snapshot 跨进程上报 /
        手动 unload,跟 Flux2/anima 同一套生命周期管理。

        model_id = `image:SeedVR2:<dit_base>:<short_hash>`(hash 兜底 model_dir/vae/device)。
        """
        import hashlib  # noqa: PLC0415
        from os.path import basename, splitext  # noqa: PLC0415

        from src.services.inference.image_seedvr2 import (  # noqa: PLC0415
            DEFAULT_DIT,
            DEFAULT_VAE,
            SeedVR2UpscaleBackend,
        )

        dit = dit_model or DEFAULT_DIT
        vae = vae_model or DEFAULT_VAE
        # device=auto:SeedVR2 不走 component sticky;直接挑最空的卡(7B 给 Pro 6000)。
        # get_best_gpu 返 -1 = 没卡装得下 → 回退 cuda:0,让下游 OOM 报真错(不静默)。
        target = device
        if device in ("auto", "cuda"):
            best = self._allocator.get_best_gpu(SeedVR2UpscaleBackend.estimated_vram_mb)
            target = f"cuda:{best}" if best is not None and best >= 0 else "cuda:0"

        dit_base = splitext(basename(str(dit)))[0] or "main"
        payload = repr((model_dir, dit, vae, target)).encode("utf-8")
        short_hash = hashlib.sha256(payload).hexdigest()[:8]
        model_id = f"image:SeedVR2:{dit_base}:{short_hash}"

        async with self._lock_for(model_id):
            entry = self._models.get(model_id)
            if entry is not None:
                logger.info("image adapter HIT id=%s (seedvr2)", model_id)
                entry.touch()
                return entry.adapter
            logger.info("image adapter MISS id=%s (seedvr2) dir=%s dit=%s", model_id, model_dir, dit)

            adapter = SeedVR2UpscaleBackend(
                paths={"model_dir": model_dir, "dit": dit, "vae": vae},
                device=target,
            )
            await adapter.load(target)

            gpu_idx = int(target.split(":")[1]) if target.startswith("cuda:") else 0
            self._models[model_id] = LoadedModel(
                spec=self._synthesize_image_spec(
                    model_id=model_id,
                    adapter_class_path="src.services.inference.image_seedvr2.SeedVR2UpscaleBackend",
                    target_device=target,
                    vram_mb=SeedVR2UpscaleBackend.estimated_vram_mb,
                    pipeline_class="SeedVR2",
                    source_files=[dit, vae],
                ),
                adapter=adapter,
                gpu_index=gpu_idx,
                gpu_indices=[gpu_idx],
            )
            return adapter

    async def _get_or_load_modular_adapter(self, resolved, combo_key, pipeline_class, target, _emit, offload: str = "none"):
        """图像引擎:建/复用 ModularImageBackend(class 名是历史,实际是标准 diffusers pipeline)。

        与 legacy 共用 `_image_adapters` combo 缓存 + 四态事件(coarse:加载前 loading、
        建好 loaded;细粒度 ComponentsManager 事件留后续)。comfy 单文件量化 → PR-2。
        重 load 经 `asyncio.to_thread` 不阻塞 runner 事件循环。
        """
        from src.services.inference.image_modular import ModularImageBackend

        model_id = self._derive_image_model_id(combo_key)
        async with self._lock_for(model_id):
            # cache hit → touch LRU + return adapter from unified _models dict
            entry = self._models.get(model_id)
            if entry is not None:
                logger.info("image adapter HIT id=%s (modular)", model_id)
                entry.touch()
                for k in ("diffusion_models", "clip", "vae"):
                    await _emit(resolved[k], "loaded")
                return entry.adapter
            # cache miss — dump combo_key 字段,方便复盘是哪个字段导致不复用
            logger.info(
                "image adapter MISS id=%s (modular) combo=%s",
                model_id, self._explain_image_combo_key(combo_key),
            )

            repo = _modular_repo_from_components(resolved)
            for k in ("diffusion_models", "clip", "vae"):
                await _emit(resolved[k], "loading")
            try:
                # 单文件桥接 override:repo 外的单文件组件各自桥接(dequant + from_config of repo);
                # HF-layout 组件(diffusers/<m>/.../config.json)则 None,由 from_pretrained 加载。
                from src.services.inference.image_modular import (  # noqa: PLC0415
                    build_bridged_text_encoder,
                    build_bridged_transformer,
                    build_bridged_vae,
                )
                # PR-D / PR-D2:桥接 load_device 按 offload 模式选 ——
                #   cpu  → 'cpu'(enable_model_cpu_offload 接管;直接 cuda 加载会爆 34GB > 24GB)
                #   cuda:N → 'cuda:N'(stash 卡;hook 把组件移到 compute 时再倒)
                #   none → target(compute 卡;全程在那)
                if offload == "cpu":
                    load_device = "cpu"
                elif offload.startswith("cuda:"):
                    load_device = offload  # stash 卡
                else:
                    load_device = target
                t_ov = c_ov = v_ov = None
                if _is_standalone_single_file(resolved["diffusion_models"]):
                    t_ov = await asyncio.to_thread(
                        build_bridged_transformer, resolved["diffusion_models"], repo, load_device)
                if _is_standalone_single_file(resolved["clip"]):
                    c_ov = await asyncio.to_thread(
                        build_bridged_text_encoder, resolved["clip"], repo, load_device)
                if _is_standalone_single_file(resolved["vae"]):
                    v_ov = await asyncio.to_thread(
                        build_bridged_vae, resolved["vae"], repo, load_device)
                def _build_adapter():
                    return ModularImageBackend(
                        repo=repo,
                        device=target,
                        dtype=resolved["diffusion_models"].dtype,
                        pipeline_class=pipeline_class,
                        offload=offload,
                        transformer_override=t_ov,
                        text_encoder_override=c_ov,
                        vae_override=v_ov,
                    )

                # PR-D4 OOM 重试一次:加载 / _ensure_pipe 抛 CUDA OOM →
                # evict 同卡 LRU(可能是上次跑剩下的旧 image adapter)→ 重试。
                # 跟 get_or_load(L429-L438)的 retry 套路一致,但作用于 image。
                gpu_idx_target = int(target.split(":")[1]) if target.startswith("cuda:") else None
                try:
                    adapter = _build_adapter()
                    await adapter.load(target)
                    await asyncio.to_thread(adapter._ensure_pipe)
                except Exception as e:  # noqa: BLE001
                    if self._is_oom(e) and gpu_idx_target is not None:
                        evicted = await self.evict_lru(gpu_index=gpu_idx_target)
                        if evicted is not None:
                            logger.info(
                                "image adapter %r: OOM on first load, evicted %r on gpu %s, retrying",
                                model_id, evicted, gpu_idx_target,
                            )
                            try:
                                adapter = _build_adapter()
                                await adapter.load(target)
                                await asyncio.to_thread(adapter._ensure_pipe)
                            except Exception as e2:  # noqa: BLE001
                                for k in ("diffusion_models", "clip", "vae"):
                                    await _emit(resolved[k], "failed", f"{type(e2).__name__}: {e2}")
                                raise
                        else:
                            for k in ("diffusion_models", "clip", "vae"):
                                await _emit(resolved[k], "failed", f"OOM and nothing evictable: {e}")
                            raise
                    else:
                        for k in ("diffusion_models", "clip", "vae"):
                            await _emit(resolved[k], "failed", f"{type(e).__name__}: {e}")
                        raise
            except Exception as e:  # noqa: BLE001
                # 桥接 build_bridged_* / 其它非 OOM 路径 — 兜底 emit failed 再抛。
                # 内层 OOM 重试块已经发了 emit failed,这里只 cover 它没覆盖的路径。
                for k in ("diffusion_models", "clip", "vae"):
                    await _emit(resolved[k], "failed", f"{type(e).__name__}: {e}")
                raise
            for k in ("diffusion_models", "clip", "vae"):
                await _emit(resolved[k], "loaded")
            gpu_idx = int(target.split(":")[1]) if target.startswith("cuda:") else 0
            self._models[model_id] = LoadedModel(
                spec=self._synthesize_image_spec(
                    model_id=model_id,
                    adapter_class_path="src.services.inference.image_modular.ModularImageBackend",
                    target_device=target,
                    vram_mb=self._estimate_image_vram_mb(resolved) or 0,
                    pipeline_class=pipeline_class,
                    source_files=[resolved[k].file for k in ("diffusion_models", "clip", "vae")],
                ),
                adapter=adapter,
                gpu_index=gpu_idx,
                gpu_indices=[gpu_idx],
            )
            return adapter
