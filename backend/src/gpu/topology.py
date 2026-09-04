"""GPU 拓扑:NVLink 直连关系 + 「同型号可组队」候选组。

为什么要这个模块 —— 张量并行(tensor parallel, tp>1)**只在同型号卡之间成立**:
vLLM 把权重按卡数均分,组内最小的卡决定整组上限。把 96G 的 Pro 6000 和 24G 的 3090
混做一组,要么按最小卡 24G 算把大卡浪费掉,要么直接 OOM。所以自动选组必须先按
``name`` 分组,**绝不混异构**。

NVLink 是次级偏好:``nvidia-smi topo -m`` 里两卡交叉格是 ``NV<n>``(n=链路数)时,
它们之间的 all-reduce 走 NVLink 而不是 PCIe/主机桥,TP 的每步同步开销小一个量级。
拿不到拓扑(没有 nvidia-smi / 解析失败)时一律视为「无 NVLink」——降级成"能跑但慢",
不阻塞选组。

放在独立文件而非 detector.py:detector 是 torch 侧的设备发现,这里全部走 nvidia-smi
文本(测试里 torch 被 mock,拓扑仍要能查)。
"""

from __future__ import annotations

import itertools
import logging
import re
import subprocess

logger = logging.getLogger(__name__)

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_GPU_ROW_RE = re.compile(r"^GPU(\d+)\b")
_NVLINK_CELL_RE = re.compile(r"^NV\d+$")

# 进程内缓存:拓扑与卡名在进程生命周期里不变(热插拔 GPU 不在支持范围)。
_cached_nvlink: set[frozenset[int]] | None = None
_cached_names: dict[int, str] | None = None


def _run_smi(args: list[str], timeout: float = 5.0) -> str | None:
    try:
        r = subprocess.run(
            ["nvidia-smi", *args], capture_output=True, text=True, timeout=timeout
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return None
    if r.returncode != 0:
        return None
    return r.stdout


def parse_topo_matrix(text: str) -> set[frozenset[int]]:
    """解析 ``nvidia-smi topo -m`` 输出 → NVLink 直连的 GPU index 对集合。

    行形如 ``GPU0\t X \tNODE\tNV4\t0-47\t...``:第 0 列是行标签,随后 N 列是与
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
            if j == idx:
                continue
            if _NVLINK_CELL_RE.match(cell):
                pairs.add(frozenset((idx, j)))
    return pairs


def nvlink_pairs(refresh: bool = False) -> set[frozenset[int]]:
    """有 NVLink 直连的 GPU index 对(``{frozenset({0, 2}), ...}``)。

    拿不到拓扑 → 空集(= 视为无 NVLink)。**绝不因此报错**:NVLink 只是偏好,
    没有它 TP 照样能跑。
    """
    global _cached_nvlink
    if _cached_nvlink is not None and not refresh:
        return set(_cached_nvlink)
    out = _run_smi(["topo", "-m"])
    if out is None:
        _cached_nvlink = set()
        return set()
    try:
        _cached_nvlink = parse_topo_matrix(out)
    except Exception as e:  # noqa: BLE001 — 拓扑解析失败降级成「无 NVLink」
        logger.warning("解析 nvidia-smi topo -m 失败,视为无 NVLink:%s", e)
        _cached_nvlink = set()
    return set(_cached_nvlink)


def gpu_names(refresh: bool = False) -> dict[int, str]:
    """``{index: 型号名}``。同型号判定的依据(自动组队只在同名卡之间做)。"""
    global _cached_names
    if _cached_names is not None and not refresh:
        return dict(_cached_names)
    out = _run_smi(["--query-gpu=index,name", "--format=csv,noheader"])
    names: dict[int, str] = {}
    if out:
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",", 1)]
            if len(parts) < 2:
                continue
            try:
                names[int(parts[0])] = parts[1]
            except ValueError:
                continue
    _cached_names = names
    return dict(names)


def reset_cache() -> None:
    """清进程内缓存(测试用)。"""
    global _cached_nvlink, _cached_names
    _cached_nvlink = None
    _cached_names = None


def group_is_nvlinked(gpus: list[int], pairs: set[frozenset[int]] | None = None) -> bool:
    """组内**任意两卡**都有 NVLink 直连才算 nvlink 组(单卡组恒 False)。"""
    if len(gpus) < 2:
        return False
    if pairs is None:
        pairs = nvlink_pairs()
    return all(frozenset((a, b)) in pairs for a, b in itertools.combinations(gpus, 2))


def candidate_groups(
    stats: list[dict] | None = None,
    names: dict[int, str] | None = None,
    pairs: set[frozenset[int]] | None = None,
    max_size: int = 4,
) -> list[dict]:
    """可用于张量并行的**同型号**候选组(size ≥ 2)。

    返回 ``[{"gpus": [0, 2], "name": "RTX 3090", "nvlink": True, "total_gb": 48.0}, ...]``,
    NVLink 组优先、其次卡数少的、其次 index 小的。前端「GPU 分配 → 组合」子菜单直接吃这个。
    异构卡永不成组 —— 这正是本函数存在的意义。
    """
    if stats is None:
        from src.services.gpu_monitor import poll_gpu_stats  # noqa: PLC0415

        stats = poll_gpu_stats()
    if names is None:
        names = gpu_names()
    if pairs is None:
        pairs = nvlink_pairs()

    total_by_idx = {int(s["index"]): float(s.get("total_mb") or 0) / 1024 for s in stats}
    by_name: dict[str, list[int]] = {}
    for idx in sorted(total_by_idx):
        by_name.setdefault(names.get(idx, f"GPU{idx}"), []).append(idx)

    out: list[dict] = []
    for name, idxs in by_name.items():
        if len(idxs) < 2:
            continue
        for size in range(2, min(len(idxs), max_size) + 1):
            for combo in itertools.combinations(idxs, size):
                gpus = list(combo)
                out.append(
                    {
                        "gpus": gpus,
                        "name": name,
                        "nvlink": group_is_nvlinked(gpus, pairs),
                        "total_gb": round(sum(total_by_idx.get(i, 0.0) for i in gpus), 1),
                    }
                )
    out.sort(key=lambda g: (not g["nvlink"], len(g["gpus"]), g["gpus"]))
    return out


def select_tp_group(
    stats: list[dict],
    model_size_gb: float,
    *,
    names: dict[int, str] | None = None,
    pairs: set[frozenset[int]] | None = None,
    exact_size: int | None = None,
    headroom: float = 1.2,
    max_size: int = 4,
) -> list[int]:
    """给一个装不下单卡的模型挑一组卡做张量并行。挑不到返回 ``[]``。

    规则(顺序即优先级):
      1. **只在同型号卡之间挑** —— 异构组队要么按最小卡算白扔大卡显存,要么启动即 OOM;
      2. 组内每张卡都要装得下自己那份分片:``min(free) * size >= model_size * headroom``
         (按组内**最小** free 算,不是求和 —— TP 是均分,最小的卡先炸);
      3. NVLink 全连通的组优先;
      4. 同为 NVLink(或同为非 NVLink)时,卡数少的优先(省卡),再按组内总 free 多的优先。

    ``exact_size`` 非空时只考虑该卡数的组(用于「用户显式配了 tensor_parallel_size=N
    但没配 gpus」的场景)。
    """
    if names is None:
        names = gpu_names()
    if pairs is None:
        pairs = nvlink_pairs()

    by_name: dict[str, list[dict]] = {}
    for s in stats:
        idx = int(s["index"])
        by_name.setdefault(names.get(idx, f"GPU{idx}"), []).append(s)

    best: tuple[tuple, list[int]] | None = None
    for cards in by_name.values():
        if len(cards) < 2:
            continue
        cards = sorted(cards, key=lambda c: int(c["index"]))
        sizes = [exact_size] if exact_size else range(2, min(len(cards), max_size) + 1)
        for size in sizes:
            if not size or size < 2 or size > len(cards):
                continue
            for combo in itertools.combinations(cards, size):
                idxs = [int(c["index"]) for c in combo]
                min_free_gb = min(float(c.get("free_mb") or 0) for c in combo) / 1024
                if min_free_gb * size < model_size_gb * headroom:
                    continue
                total_free = sum(float(c.get("free_mb") or 0) for c in combo)
                key = (group_is_nvlinked(idxs, pairs), -size, total_free)
                if best is None or key > best[0]:
                    best = (key, idxs)
    return list(best[1]) if best else []
