# 图像显存守卫「先腾后载」— 借鉴 ComfyUI free_memory 的按卡 LRU 驱逐

**Status**: Draft
**Author**: heygo(设计 by Claude)
**Date**: 2026-06-07

## 0. 背景与问题

用户报告:工作流节点四态全显「已加载」,但 Run 报 `cuda:1 空闲显存不足`。两轮修复:
- **#354** 修了「组件已在 L1 池却被守卫按全新载入估满尺寸」的误拦(守卫跳过已在 `_components` 池的组件)。
- 本 spec 补另一类:**该卡被「其它可驱逐的 image adapter」占着、本可腾出来载入当前请求**,但守卫直接硬报错,从不尝试腾。

**根因(机制差异)**:对照 ComfyUI(读了 `comfy/model_management.py`):
- ComfyUI **载入前主动腾**:`load_models_gpu`(:713)对每张目标卡调 `free_memory(需要量×1.1+reserve, device)`(:658),**按 LRU 卸掉该卡上最久没用的模型腾够空间**,再载;还塞不下才退 lowvram 分块上卡。**从不"显存不足"报错**(除非真 OOM)。
- nous **载入前硬报错**:`_guard_image_vram_per_card`(model_manager.py)发现 `free < need` 直接 `raise RuntimeError`。已有的驱逐 `evict_lru(gpu_index)` 只在**真 OOM 之后**被动重试一次(`_get_or_load_modular_adapter` OOM-retry),被守卫抢在前面、根本没机会跑。

## 1. 目标 / 非目标

### 目标
- 守卫从「不够即报错」改为「**不够先按卡 LRU evict 腾,腾够再载;腾不出才报清晰错误**」(对齐 ComfyUI free_memory 语义)。
- 复用现有 `evict_lru(gpu_index)` 原语(已正确跳过 resident / 被引用 / in-use 的模型),不新造驱逐逻辑。
- 保留 reserve 余量(像 ComfyUI 留 activation headroom),避免腾到刚好却 forward OOM。
- 零回归:够装时行为不变(不 evict);#354 的「跳过已在池组件」继续生效。

### 非目标
- **lowvram 逐层分块上卡**(ComfyUI `lowvram_model_memory`):更深的引擎改动,本 spec 不做;已有 offload=cpu/cuda:N 作粗粒度替代(逐组件选卡 spec 2026-06-04)。列 future。
- 跨 runner / 驱逐 LLM:image runner 的 `_models` 只含 image adapter,**vLLM/LLM 不在其中、不可驱逐**(也不该驱逐用户的 LLM)。所以「卡被 vLLM 占满」时 evict 腾不出 → 仍报错(正确:提示换卡 / offload)。

### 成功标准
- 卡上有可驱逐的旧 image adapter + 当前请求装不下 → **自动 evict 腾出再载,不报错**。
- 卡被 resident / in-use / vLLM 占满、腾不出 → 报清晰错误(换卡 / fp8 / offload / 分卡)。
- 单测:可驱逐→腾后成功;不可驱逐→报错;够装→不 evict。
- 既有 image / component 测试全绿。

## 2. 设计

### 2.1 新增 `_free_image_vram_on_card(card, need_mb)`(对照 ComfyUI free_memory)
```python
async def _free_image_vram_on_card(self, card: str, need_mb: int) -> int | None:
    """按卡 LRU 驱逐,直到空闲 ≥ need_mb 或无可驱逐。返回最终空闲 MB(None=查不到→不阻塞)。
    复用 evict_lru(gpu_index)(只驱逐 非resident/非引用/非in-use 的 image adapter)。"""
    idx = int(card.split(":")[1]) if card.startswith("cuda:") else None
    if idx is None:
        return None
    free = self._free_vram_mb(card)
    while free is not None and free < need_mb:
        evicted = await self.evict_lru(gpu_index=idx)
        if evicted is None:        # 该卡无可驱逐的了(只剩 resident/in-use/vLLM)
            break
        free = self._free_vram_mb(card)
    return free
```

### 2.2 守卫改「先腾后载」(`_guard_image_vram_per_card`)
当前:逐卡算 need(已跳过 offload≠none + 已在池组件,#354),`free < need` → raise。
改为:逐卡算 need 后,先 `await self._free_image_vram_on_card(card, need_mb + RESERVE_MB)`,**腾完仍 `free < need` 才 raise**(错误文案沿用,补一句「已尝试驱逐空闲模型仍不足」)。
- `RESERVE_MB`:固定余量(ComfyUI 默认 ~1GB inference + 几百 MB reserved)。建议 `RESERVE_MB = 1024`,叠加现有 need 的 1.3× headroom。
- 守卫从同步改 async(它已在 async `get_or_load_image_adapter` 内调用,改 `await` 即可)。

### 2.3 与现有 OOM-retry 的关系
- 守卫主动腾后,绝大多数情况不会再撞 OOM。
- 保留 `_get_or_load_modular_adapter` 的 OOM-retry(evict_lru + 重试一次)作**最后兜底**(估算不准 / 碎片 / 突发占用)。两层互补:守卫=事前按估算腾,OOM-retry=事后真 OOM 兜。

### 2.4 安全不变式(evict_lru 已保证,无需额外)
- 不驱逐 `resident` / 被 combo 引用(`_references`)/ 正在 infer(`_in_use`,防 segfault)的模型。
- 只动当前目标卡(`gpu_index`),不误伤别卡。
- vLLM/LLM 不在 image runner `_models` → 天然不被驱逐。

## 3. PR 拆分
**PR-1(单一后端 PR)**:
- `_free_image_vram_on_card` 新增;`_guard_image_vram_per_card` 改 async + 先腾后载 + RESERVE_MB。
- `get_or_load_image_adapter` 调用处 `await`。
- 单测:
  - 卡满 + 有可驱逐 LRU adapter → 守卫驱逐后放行(mock `_free_vram_mb` 随驱逐变大、mock `evict_lru` 计数)。
  - 卡满 + 无可驱逐(全 resident/in-use)→ 仍 raise。
  - 够装 → 不调 evict(零回归)。
- 不碰 `image_modular.py` → 不触发真模型 smoke 闸门(CLAUDE.md);可选真机验「填满一张卡→跑另一工作流→自动腾出跑通」。

## 4. 风险
- **缓存抖动**:evict 了马上又要用的 adapter → 下次重载。ComfyUI 接受此 LRU 取舍;nous image runner 串行,驱逐 idle 非引用 adapter 来装当前请求是对的。
- **估算偏差**:`_free_vram_mb`(nvidia-smi)与真实碎片有差 → 故留 OOM-retry 兜底。
- **async 化守卫**:确认所有调用点都在 async 上下文(目前唯一调用点 `get_or_load_image_adapter` 是 async)。

## 5. Future
- lowvram 逐层分块上卡(ComfyUI `lowvram_model_memory`):单工作流超单卡时分块流式,比整组件 offload 更细。
- reserve 余量按 nvidia-smi 动态(ComfyUI 按 OS / 平台调 `EXTRA_RESERVED_VRAM`)。
