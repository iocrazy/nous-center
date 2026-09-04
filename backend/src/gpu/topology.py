"""GPU 拓扑与放置规则 —— `gpu`/`gpus` 优先级、组校验、组预算、可用组候选。

**权威来源是 `configs/hardware.yaml`**(经 `GPUAllocator._build_groups` 解析,这里不另起
一套解析)。那份 yaml 记着运维约束 —— 比如本机 GPU 0 是显示卡,虽然与 GPU 2 之间有
NVLink,但在把显示器挪走之前不该拿来跑张量并行。所以「哪些卡可以组成一个 TP 单元」
只认 yaml 里声明的多卡 group;`nvidia-smi topo -m` 在这里**只用于校验/补缺**
(yaml 没写 nvlink 时探测一下、写了但探不到时告警),不构成新的候选来源。

张量并行(tp>1)的两条硬规则:
  1. **只在同型号卡之间**:vLLM 按卡数均分权重,组内最小的卡决定整组上限。96G 的
     Pro 6000 和 24G 的 3090 混一组,要么按 24G 算白扔 72G,要么启动即 OOM。
  2. **组大小是 2 的幂**:tp 必须整除注意力头数,非 2 的幂在真实模型上基本起不来。

放置决策本身**只发生在 `ModelManager._resolve_placement` 一处**;适配器只执行传下来的
device / gpus(2026-09-03 审查)。本模块提供它需要的纯函数。
"""

from __future__ import annotations

import logging
import re
import subprocess
import threading
import time
from itertools import combinations

logger = logging.getLogger(__name__)

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_GPU_ROW_RE = re.compile(r"^GPU(\d+)\b")
_NVLINK_CELL_RE = re.compile(r"^NV\d+$")

# 允许的 TP 组大小(2 的幂;vLLM 要求 tp 整除注意力头数)。
_ALLOWED_GROUP_SIZES = (2, 4, 8)
# 候选组返回上限 —— 菜单不该被几十项撑爆。
_MAX_CANDIDATES = 8

# 拓扑缓存。**失败不永久缓存**:探测失败(nvidia-smi 缺失/超时)只压 30s 负缓存,
# 否则一次启动期抖动就让整个进程余生都以为"没有 NVLink"(审查 #18)。
_NEGATIVE_TTL_S = 30.0
_cache_lock = threading.Lock()
_nvlink_cache: tuple[float, set[frozenset[int]]] | None = None  # (expires_at, pairs)


# ---------------------------------------------------------------------------
# nvidia-smi topo -m —— 只用于 nvlink 的校验/补缺
# ---------------------------------------------------------------------------

def parse_topo_matrix(text: str) -> set[frozenset[int]]:
    """解析 ``nvidia-smi topo -m`` → NVLink 直连的 GPU index 对集合。

    行形如 ``GPU0\\t X \\tNODE\\tNV4\\t0-47\\t...``:第 0 列是行标签,随后 N 列是与
    GPU0..GPU{N-1} 的连接类型,再往后是 CPU/NUMA affinity 等无关列。N 由 GPU 行数
    决定(比数表头列更稳:表头带 ANSI 转义且各驱动版本列名不一)。
    """
    rows: list[tuple[int, list[str]]] = []
    for raw in text.splitlines():
        line = _ANSI_RE.sub("", raw).strip()
        m = _GPU_ROW_RE.match(line)
        if not m:
            continue
        cells = line.split()
        rows.append((int(m.group(1)), cells[1:]))
    n = len(rows)
    pairs: set[frozenset[int]] = set()
    for idx, cells in rows:
        for j, cell in enumerate(cells[:n]):
            if j != idx and _NVLINK_CELL_RE.match(cell):
                pairs.add(frozenset((idx, j)))
    return pairs


def nvlink_pairs(refresh: bool = False) -> set[frozenset[int]]:
    """有 NVLink 直连的 GPU index 对。探测不到 → 空集(视为无 NVLink,只降级不阻塞)。

    成功的结果长期缓存(拓扑不会中途变);**失败只缓存 30 秒**,下次调用会重试。
    """
    global _nvlink_cache
    now = time.monotonic()
    with _cache_lock:
        if not refresh and _nvlink_cache is not None and _nvlink_cache[0] > now:
            return set(_nvlink_cache[1])

    pairs: set[frozenset[int]] = set()
    ok = False
    try:
        r = subprocess.run(
            ["nvidia-smi", "topo", "-m"], capture_output=True, text=True, timeout=5
        )
        if r.returncode == 0:
            pairs = parse_topo_matrix(r.stdout)
            ok = True
    except (FileNotFoundError, OSError, subprocess.SubprocessError) as e:
        logger.warning("nvidia-smi topo -m 探测失败(%s)—— 暂按无 NVLink 处理,%.0fs 后重试",
                       type(e).__name__, _NEGATIVE_TTL_S)
    except Exception as e:  # noqa: BLE001 — 解析异常同样降级
        logger.warning("解析 nvidia-smi topo -m 失败,暂按无 NVLink 处理:%s", e)

    with _cache_lock:
        # 成功 → 永久(float("inf"));失败 → 30s 后重试。
        _nvlink_cache = (float("inf") if ok else now + _NEGATIVE_TTL_S, pairs)
    return set(pairs)


def reset_cache() -> None:
    """清拓扑缓存(测试用)。"""
    global _nvlink_cache
    with _cache_lock:
        _nvlink_cache = None


def group_is_nvlinked(gpus: list[int], pairs: set[frozenset[int]] | None = None) -> bool:
    """组内**任意两卡**都有 NVLink 直连才算 nvlink 组(单卡组恒 False)。"""
    if len(gpus) < 2:
        return False
    if pairs is None:
        pairs = nvlink_pairs()
    return all(frozenset((a, b)) in pairs for a, b in combinations(gpus, 2))


def warm_caches() -> None:
    """启动时预热拓扑/卡信息(同步 subprocess —— 调用方丢 to_thread)。

    之后 `nvlink_pairs()` / `gpu_names()` 都是缓存命中,不会在请求路径或 load 路径上
    再吃一次 nvidia-smi 的几十毫秒(审查 #17)。
    """
    try:
        nvlink_pairs(refresh=True)
        gpu_names()
    except Exception as e:  # noqa: BLE001 — 预热失败不该拖垮启动
        logger.warning("GPU 拓扑预热失败(不影响启动):%s", e)


# ---------------------------------------------------------------------------
# 卡信息 —— 复用 detector 的缓存,不再自己 shell-out
# ---------------------------------------------------------------------------

def gpu_names() -> dict[int, str]:
    """``{index: 型号名}``,来自 `detector.get_gpus()`(进程内缓存)。同型号判定的依据。"""
    from src.gpu.detector import get_gpus  # noqa: PLC0415

    try:
        return {g.index: g.name for g in get_gpus()}
    except Exception:  # noqa: BLE001 — 无 torch / 无 GPU 环境
        return {}


def gpu_totals_gb() -> dict[int, float]:
    """``{index: 总显存 GB}``,同样来自 `detector.get_gpus()` 的缓存。"""
    from src.gpu.detector import get_gpus  # noqa: PLC0415

    try:
        return {g.index: g.vram_total_gb for g in get_gpus()}
    except Exception:  # noqa: BLE001
        return {}


def display_gpu_indices() -> set[int]:
    """驱动显示服务的卡 —— 与单卡路径同一个探测(`detector.get_display_gpu_indices`)。"""
    from src.gpu.detector import get_display_gpu_indices  # noqa: PLC0415

    return get_display_gpu_indices()


# ---------------------------------------------------------------------------
# gpu / gpus 优先级 —— 唯一实现
# ---------------------------------------------------------------------------

def _field(obj, key: str):
    """从 dict(models.yaml cfg)或对象(ModelSpec)取字段,两种形状同一套读法。"""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def resolve_gpus(config_or_spec) -> list[int]:
    """该模型显式配置的落卡列表 —— **`gpu`/`gpus` 优先级的唯一实现**。

    `gpus`(组)优先于 `gpu`(单卡);都没有 → ``[]``(= 交给自动选卡)。
    `gpus: []` 是「显式清空组」的哨兵(运行时覆盖用它区分"没设"和"设为无组"),
    此时回落到 `gpu`。
    返回值永远是去重后的 int 列表,顺序保持配置里的顺序。
    """
    raw_group = _field(config_or_spec, "gpus")
    out: list[int] = []
    if isinstance(raw_group, (list, tuple)):
        for v in raw_group:
            try:
                iv = int(v)
            except (TypeError, ValueError):
                continue
            if iv >= 0 and iv not in out:
                out.append(iv)
        if out:
            return out

    single = _field(config_or_spec, "gpu")
    if isinstance(single, bool) or single is None:
        return []
    if isinstance(single, int):
        return [single] if single >= 0 else []
    if isinstance(single, (list, tuple)):  # 历史形状:gpu 也可能是 list
        for v in single:
            try:
                iv = int(v)
            except (TypeError, ValueError):
                continue
            if iv >= 0 and iv not in out:
                out.append(iv)
        return out
    if isinstance(single, str) and single.startswith("cuda"):
        try:
            return [int(single.split(":")[-1])]
        except ValueError:
            return []
    return []


# ---------------------------------------------------------------------------
# 组校验 —— HTTP 路径与 YAML 加载路径共用的唯一实现
# ---------------------------------------------------------------------------

def validate_gpu_group(
    gpus, *, groups: list | None = None, check_warnings: bool = True
) -> tuple[list[int], list[str], list[str]]:
    """校验一个 GPU 组。→ ``(归一化后的组, errors, warnings)``。

    errors 非空 = 这个组不能用(HTTP 400 / YAML 加载时忽略并 log error):
      * 不是整数列表 / 有重复 / 少于 2 张 / 大小不是 2 的幂
      * 卡不存在
      * **异构**(型号不一致)
    warnings = 能用但要提醒(与单卡路径行为一致 —— 单卡钉到显示卡也只是 warning):
      * 组内有卡在驱动显示服务
      * 组没在 hardware.yaml 里声明(绕过了拓扑权威的软偏好)
      * 组内并非两两 NVLink 直连(TP 走 PCIe,慢)

    `check_warnings=False` 只跑 errors —— 那部分全部走 `gpu_names()` 的进程内缓存,
    没有 subprocess。YAML 加载路径(同步、可能在事件循环上)用它,别在那儿吃
    `nvidia-smi` 的几十毫秒;HTTP 路径走完整版(路由已丢 to_thread)。
    """
    errors: list[str] = []
    warnings: list[str] = []

    try:
        norm = [int(i) for i in gpus]
    except (TypeError, ValueError):
        return [], ["gpus must be a list of integers"], []
    if len(set(norm)) != len(norm):
        errors.append("gpus contains duplicate indices")
    if len(norm) < 2:
        errors.append("a GPU group needs at least 2 cards")
    elif len(norm) not in _ALLOWED_GROUP_SIZES:
        errors.append(
            f"GPU 组大小必须是 2 的幂({', '.join(map(str, _ALLOWED_GROUP_SIZES))}):"
            f"tp 要整除注意力头数,{len(norm)} 张卡的组起不来"
        )
    if errors:
        return [], errors, warnings

    norm = sorted(norm)
    names = gpu_names()
    if names:
        missing = [i for i in norm if i not in names]
        if missing:
            errors.append(f"unknown GPU index: {missing}")
            return [], errors, warnings
        distinct = {names[i] for i in norm}
        if len(distinct) > 1:
            errors.append(
                "GPU 组必须同型号(异构卡做张量并行会按最小卡算或直接 OOM):"
                f"{sorted(distinct)}"
            )
            return [], errors, warnings

    if not check_warnings:
        return norm, errors, warnings

    on_display = sorted(set(norm) & display_gpu_indices())
    if on_display:
        warnings.append(
            f"GPU 组 {norm} 里 {on_display} 在驱动显示服务 —— 重负载会把桌面挤崩"
        )
    if groups is None:
        groups = hardware_groups()
    declared = {tuple(sorted(g.gpus)) for g in groups if len(g.gpus) > 1}
    if declared and tuple(norm) not in declared:
        warnings.append(
            f"GPU 组 {norm} 未在 hardware.yaml 里声明(已声明的多卡组:"
            f"{[list(d) for d in sorted(declared)]})"
        )
    if not group_is_nvlinked(norm):
        warnings.append(f"GPU 组 {norm} 并非两两 NVLink 直连 —— TP 同步走 PCIe,较慢")
    return norm, errors, warnings


def sanitize_config_gpus(model_id: str, config_or_entry) -> list[int] | None:
    """YAML / 覆盖里读到的 `gpus` 过一遍校验。非法 → log error 并返回 None(忽略该字段,
    **不阻塞启动**);合法 → 返回归一化后的组。单卡 `gpu` 不走这里。"""
    raw = _field(config_or_entry, "gpus")
    if not isinstance(raw, (list, tuple)) or not raw:
        return None
    norm, errors, warnings = validate_gpu_group(list(raw), check_warnings=False)
    if errors:
        logger.error("模型 %s 的 gpus=%s 非法,已忽略(按单卡 gpu / 自动选卡处理):%s",
                     model_id, list(raw), "; ".join(errors))
        return None
    for w in warnings:
        logger.warning("模型 %s 的 GPU 组:%s", model_id, w)
    return norm


# ---------------------------------------------------------------------------
# 组预算 —— 组内最小 total/free(唯一实现)
# ---------------------------------------------------------------------------

def group_budget_gb(idxs: list[int], stats: list[dict] | None = None) -> tuple[float, float]:
    """一组卡的 ``(total_gb, free_gb)`` —— 取组内**最小**,不是求和。

    TP 均分权重、`gpu_memory_utilization` 是**每卡**比例,组内最小的卡先见底。按 sum
    算会让预算端点放行一个单卡装不下的绝对值,适配器再照着它起 → 启动即 OOM。
    预算端点与适配器都必须走这一个实现(数据源统一为 nvidia-smi 的 MB,不混 torch 的
    四舍五入 GB)。查不到 → (0.0, 0.0),调用方按"未知"处理。
    """
    if not idxs:
        return 0.0, 0.0
    if stats is None:
        from src.services.gpu_monitor import poll_gpu_stats  # noqa: PLC0415

        stats = poll_gpu_stats()
    by_index = {int(s["index"]): s for s in stats}
    cards = [by_index[i] for i in idxs if i in by_index]
    if not cards:
        return 0.0, 0.0
    return (
        min(float(c.get("total_mb") or 0) for c in cards) / 1024,
        min(float(c.get("free_mb") or 0) for c in cards) / 1024,
    )


# ---------------------------------------------------------------------------
# 候选组 —— 权威来源是 hardware.yaml
# ---------------------------------------------------------------------------

def hardware_groups(hardware_config: dict | None = None) -> list:
    """hardware.yaml 的 group 拓扑,**复用 `GPUAllocator` 的解析**(不另起一套)。

    读不到 / 解析失败 → `[]`(候选组为空,UI 就只剩单卡项 —— 这正是"没声明过多卡组"
    该有的样子)。
    """
    try:
        from src.config import load_hardware_config  # noqa: PLC0415
        from src.services.gpu_allocator import GPUAllocator  # noqa: PLC0415

        cfg = load_hardware_config() if hardware_config is None else hardware_config
        return GPUAllocator(poll_fn=lambda: [], hardware_config=cfg).groups()
    except Exception as e:  # noqa: BLE001 — 拓扑读不到不该拖垮调用方
        logger.warning("读 hardware.yaml 组拓扑失败,按无多卡组处理:%s", e)
        return []


def candidate_groups(groups: list | None = None) -> list[dict]:
    """可用于张量并行的候选组 —— **hardware.yaml 里声明的多卡 group**,过完同型号/大小校验。

    返回
    ``[{"gpus":[0,2], "name":"RTX 3090", "nvlink":true, "total_gb":48.0, "display_gpus":[0]}]``,
    NVLink 组优先、卡数少的优先。异构组、大小非 2 的幂的组会被丢掉并 log error ——
    它们不是"可选但不推荐",是根本起不来。

    `nvlink` 以 yaml 为准,yaml 没标(False)时用 `topo -m` **补缺**;yaml 标了 true 但
    探测不到则保留 yaml 的值并告警(拓扑可能只是查不到)。
    """
    if groups is None:
        groups = hardware_groups()
    names = gpu_names()
    totals = gpu_totals_gb()
    pairs = nvlink_pairs()
    display = display_gpu_indices()

    out: list[dict] = []
    for g in groups:
        gpus = sorted(int(i) for i in g.gpus)
        if len(gpus) < 2:
            continue
        norm, errors, _ = validate_gpu_group(gpus, groups=groups)
        if errors:
            logger.error("hardware.yaml 的组 %s(%s)不可用于张量并行:%s",
                         g.id, gpus, "; ".join(errors))
            continue
        detected = group_is_nvlinked(norm, pairs)
        if g.nvlink and not detected:
            logger.warning("hardware.yaml 的组 %s 标了 nvlink,但 topo -m 没探到直连", g.id)
        out.append({
            "id": g.id,
            "gpus": norm,
            "name": names.get(norm[0], f"GPU{norm[0]}"),
            "nvlink": bool(g.nvlink or detected),
            "total_gb": round(sum(totals.get(i, 0.0) for i in norm), 1),
            "display_gpus": sorted(set(norm) & display),
        })
    out.sort(key=lambda d: (not d["nvlink"], len(d["gpus"]), d["gpus"]))
    return out[:_MAX_CANDIDATES]


def select_tp_group(
    model_size_gb: float,
    *,
    groups: list | None = None,
    candidates: list[dict] | None = None,
    exact_size: int | None = None,
    stats: list[dict] | None = None,
    headroom: float = 1.2,
) -> list[int]:
    """给一个装不下单卡的模型挑一组卡做张量并行。挑不到返回 ``[]``。

    只在 `candidate_groups()`(= hardware.yaml 声明过、且过了同型号校验的组)里挑 ——
    **不自己枚举卡的组合**,否则就绕过了 yaml 里的运维约束(比如"GPU 0 是显示卡,
    腾空前别拿它做 tp")。

    过滤:组内每张卡都要装得下自己那份分片,``min(free) * size >= model_size * headroom``
    (按组内**最小** free 算,不是求和 —— 最小的卡先炸)。
    排序:NVLink 优先 → 卡数少优先 → 组内总 free 多优先。
    """
    if candidates is None:
        candidates = candidate_groups(groups)
    if stats is None:
        from src.services.gpu_monitor import poll_gpu_stats  # noqa: PLC0415

        stats = poll_gpu_stats()

    best: tuple[tuple, list[int]] | None = None
    for cand in candidates:
        gpus = list(cand["gpus"])
        if exact_size is not None and len(gpus) != exact_size:
            continue
        _total, free = group_budget_gb(gpus, stats)
        if free * len(gpus) < model_size_gb * headroom:
            continue
        key = (bool(cand["nvlink"]), -len(gpus), free * len(gpus))
        if best is None or key > best[0]:
            best = (key, gpus)
    return list(best[1]) if best else []
