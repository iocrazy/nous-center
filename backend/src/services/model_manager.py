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
    for comp_key in ("unet", "clip", "vae"):
        spec = resolved.get(comp_key)
        if spec is None:
            continue
        f = Path(spec.file)
        for cand in (f.parent, f.parent.parent, f.parent.parent.parent):
            if (cand / "model_index.json").exists():
                return str(cand)
    raise ValueError(
        "modular 引擎需 HF-layout repo(model_index.json);unet/clip/vae 组件文件均不在 "
        "HF repo 下 —— comfy 全单文件(无 HF clip/vae)暂不支持"
    )


def _is_comfy_single_file_unet(unet_spec) -> bool:
    """unet 是 repo 外的 comfy 量化单文件(需桥接 override)而非 HF-layout transformer。

    HF-layout transformer 目录有 config.json;comfy 单文件(diffusion_models/flux/)没有。
    """
    return not (Path(unet_spec.file).parent / "config.json").exists()

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
        # Per-model load failures (set by background preload tasks or prior
        # failed load_model attempts). get_loaded_adapter raises ModelLoadError
        # when a record exists. Cleared on successful load.
        self._load_failures: dict[str, str] = {}
        # PR-1 Task 6: component-level cache (parallel to _models). Old yaml-driven
        # adapters keep using _models; PR-2's ImageSampler will use _components.
        # Forward-ref the ComponentKey type as a tuple alias so the module-top
        # import stays clean (component_spec doesn't import ModelManager).
        from src.services.inference.component_spec import ComponentKey  # noqa: F401
        # 图像 adapter 缓存(key = pipeline_class + 3 组件 key)。PR-4 删了 legacy 组件
        # L1 缓存(_components 等);modular 引擎自管组件(ComponentsManager)。
        self._image_adapters: dict = {}
        self._image_adapter_locks: dict = {}
        # PR-1 modular 引擎:runner 持一个 ComponentsManager 单例(跨请求共享/缓存)。
        # lazy 建(_import 在 image_modular,避免顶层 import diffusers/torch)。
        self._modular_cm = None

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
                    gpu = spec.gpu if isinstance(spec.gpu, int) else None
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

    async def unload_model(self, model_id: str, force: bool = False) -> None:
        """Unload *model_id*.

        If the model is *resident* or has active references it will NOT be
        unloaded unless *force=True*.
        """
        async with self._lock_for(model_id):
            entry = self._models.get(model_id)
            if entry is None:
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

    _VRAM_EST_MB = {"unet": 18000, "clip": 6000, "vae": 1000}

    def _resolve_component_device(self, spec):
        """Resolve device='auto' → 'cuda:N' via allocator. Returns a NEW spec
        (model_copy keeps validators) so the original descriptor is untouched."""
        if spec.device != "auto":
            return spec
        idx = self._allocator.get_best_gpu(self._VRAM_EST_MB.get(spec.kind, 8000))
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
        """粗估整模型显存需求(MB)= 三组件文件 bytes 之和 * 余量系数。任一文件不存在
        (纯逻辑测试用 stub 路径)→ None(无法估,跳过保护)。"""
        import os
        total = 0
        for k in ("unet", "clip", "vae"):
            try:
                total += os.path.getsize(resolved[k].file)
            except OSError:
                return None
        # 1.3x 余量(激活/中间张量);bytes → MB
        return int(total / (1024 * 1024) * 1.3)

    def _image_adapter_lock_for(self, key) -> asyncio.Lock:
        return self._image_adapter_locks.setdefault(key, asyncio.Lock())

    async def get_or_load_image_adapter(self, components: dict, pipeline_class: str = "Flux2KleinPipeline", on_event=None):
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
        # 整模型单卡不变式(spec 2026-05-21 rev 2):device=auto 会让三组件各自 resolve
        # 到不同卡 —— 以 unet 解析出的卡为准,强制 clip/vae 落同一张卡。
        target = resolved["unet"].device
        for k in ("clip", "vae"):
            if resolved[k].device != target:
                resolved[k] = resolved[k].model_copy(update={"device": target})
        # LLM 卡保护:目标卡空闲显存不足(常驻 LLM 占着)→ 装载前清晰报错,不静默 OOM。
        # 无 GPU / 文件不存在(纯逻辑测试)时 free/need 为 None → 跳过,不阻塞。
        free_mb = self._free_vram_mb(target)
        need_mb = self._estimate_image_vram_mb(resolved)
        if free_mb is not None and need_mb is not None and free_mb < need_mb:
            raise RuntimeError(
                f"{target} 空闲显存不足({free_mb}MB < 约需 {need_mb}MB)—— "
                f"该卡可能被常驻 LLM 占用。换张卡(device)、用更低精度(fp8),"
                f"或先释放该卡。"
            )
        combo_key = (pipeline_class,) + tuple(
            to_component_key(resolved[k]) for k in ("unet", "clip", "vae"))
        # PR-4 收官:唯一引擎 = Modular Diffusers(自写 ImageSampler/DiffusersImageBackend 已删)。
        return await self._get_or_load_modular_adapter(
            resolved, combo_key, pipeline_class, target, _emit)

    async def _get_or_load_modular_adapter(self, resolved, combo_key, pipeline_class, target, _emit):
        """PR-1 modular 引擎:建/复用 ModularImageBackend(HF-layout)。

        与 legacy 共用 `_image_adapters` combo 缓存 + 四态事件(coarse:加载前 loading、
        建好 loaded;细粒度 ComponentsManager 事件留后续)。comfy 单文件量化 → PR-2。
        重 load 经 `asyncio.to_thread` 不阻塞 runner 事件循环。
        """
        from src.services.inference.image_modular import ModularImageBackend, _import_modular

        async with self._image_adapter_lock_for(combo_key):
            cached = self._image_adapters.get(combo_key)
            if cached is not None:
                for k in ("unet", "clip", "vae"):
                    await _emit(resolved[k], "loaded")
                return cached

            repo = _modular_repo_from_components(resolved)
            for k in ("unet", "clip", "vae"):
                await _emit(resolved[k], "loading")
            try:
                if self._modular_cm is None:
                    _, components_manager_cls = _import_modular()
                    self._modular_cm = components_manager_cls()
                # comfy 量化单文件 unet → 桥接 override(dequant+转键+from_config);HF-layout 则 None
                override = None
                if _is_comfy_single_file_unet(resolved["unet"]):
                    from src.services.inference.image_modular import build_bridged_transformer
                    override = await asyncio.to_thread(
                        build_bridged_transformer, resolved["unet"], repo, target)
                adapter = ModularImageBackend(
                    repo=repo,
                    device=target,
                    dtype=resolved["unet"].dtype,
                    components_manager=self._modular_cm,
                    transformer_override=override,
                )
                await adapter.load(target)
                await asyncio.to_thread(adapter._ensure_pipe)  # 预热(blocking load 进线程)
            except Exception as e:  # noqa: BLE001
                for k in ("unet", "clip", "vae"):
                    await _emit(resolved[k], "failed", f"{type(e).__name__}: {e}")
                raise
            for k in ("unet", "clip", "vae"):
                await _emit(resolved[k], "loaded")
            self._image_adapters[combo_key] = adapter
            return adapter
