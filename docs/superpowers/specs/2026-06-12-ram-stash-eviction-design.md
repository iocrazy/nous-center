# RAM stash 驱逐:卸载/驱逐默认降级为「挪内存」而非销毁

日期:2026-06-12 | 状态:spec(用户已拍「需要做」)| 来源:Ideogram-4 接入后与用户讨论 ComfyUI 对比

## 动机

nous 当前驱逐 = 销毁 adapter + 出池组件,再次使用要从磁盘冷载(Ideogram-4 fp8 量化路径
真机 95s;Klein ~22s)。ComfyUI 的等价机制(comfy/model_patcher.py:ModelPatcher(load_device,
offload_device) + model_management.free_memory)是**权重常驻 RAM、GPU 只放工作集**:驱逐 =
`.to(offload_device)` 挪回内存,再用秒级搬回。本机 128G RAM 装下全部常用图像模型绰绰有余。

目标:模型切换从「分钟级冷载」变「秒级搬运」,显存语义不变(stash 后显存立即归还)。

## 设计

### 两层 stash(对应两条加载路线)

1. **组件 L1 池 stash(细粒度路线,Flux2 等)**
   - `_release_combo_components`:refs 清空的组件**不出池**,改为 `module.to("cpu")` +
     标记 `stashed=True`,留在 `_components`。
   - `get_or_load`/L1 命中:池中 stashed 组件 → `.to(target_device)` 搬回 + `stashed=False`
     → 正常组装 combo(装配本来就支持 L1 HIT 重组,无磁盘 IO)。
   - 共享组件天然安全:refs 非空的组件根本不会走到 stash(现行 refcount 语义不变)。
   - **设备键语义**:L1 键含 device。stash 后权重在 CPU 但键不变(它是「身份」,恢复目标卡
     就是键里的卡)。跨卡复用(键不同)仍是 MISS——与现状一致,不在本 spec 扩。

2. **adapter 级 stash(整模型路线,Z-Image / Qwen-Edit / Ideogram-4)**
   - `unload_model(mode="stash")`:不调 `adapter.unload()` 销毁,改调新的
     `adapter.stash()`(引擎实现 `pipe.to("cpu")` + empty_cache),`_models` entry 保留,
     标记 `entry.stashed=True`;从显存记账/可驱逐统计中按「已腾出」处理。
   - `get_or_load_image_adapter` 命中 stashed entry → `adapter.restore(device)`
     (`pipe.to(device)`)→ 活回。
   - anima(自定义 DiT)与 modular 同接口(`stash/restore` 进引擎 ABC,默认实现 =
     pipe/module `.to()`)。

### RAM 水位与 stash 层 LRU

- `psutil.virtual_memory().available`,水位线 `NOUS_STASH_RAM_RESERVE_GB`(默认 24,
  给 PG/前端/系统留余量)。
- stash 前检查:`available - 待 stash 字节 < 水位` → 不 stash,直接销毁(现行为)。
- stash 层记账:`_stashed_bytes` 估算用现成 `_component_bytes`(分片感知);
  超水位时按 stash 时间 LRU 真销毁最旧者直至回到水位内。
- 驱逐顺序:GPU 紧张 → 先 stash(腾显存);RAM 紧张 → 销毁最旧 stash(腾内存)。

### 接入点

- `evict_lru` / 显存守卫「先腾后载」:驱逐动作默认 `mode="stash"`(in_use/resident/refs
  守卫全部沿用,**强于** stash)。
- 手动卸载 API(`/component/unload`、`/loaded-adapter/unload`、unload-image-adapters):
  默认 stash;请求加 `destroy=true` 才真销毁(给用户「彻底清掉」的出口)。
- 四态/UI:stashed 映射为 `unloaded`(语义=不占显存),不加第五态;
  结构化快照加 `stashed: true` 字段供 ModelsOverlay 将来显示「(内存待命)」。

### 风险与验证门禁

- **torchao fp8 量化权重 `.to()` 往返**:tensor subclass 的 cpu↔cuda 搬运需真机 spike
  先验(PR-1 第一步);不行则 fp8 量化组件 stash 退化为销毁(逐组件判断)。
- **段路/手写循环**:stash 只在无 in_use 时发生,与采样互斥(沿用 in_use 硬守卫)。
- **golden 回归**:stash→restore→出图必须与直载 bit 一致(SSIM=1.0,smoke_image_ab
  流程);Ideogram-4 用 smoke_ideogram4 同验。
- **vLLM 不在体系内**(独立进程,自管显存),不受影响。

## PR 链

- PR-1:组件 L1 池 stash/restore + RAM 水位 + stash LRU(细粒度路线)+ fp8 `.to()` spike
- PR-2:adapter stash/restore(整模型路线,引擎 ABC `stash/restore`)
- PR-3:evict_lru/显存守卫/卸载 API 接入(默认 stash)+ 真机时延对比测试
  (验收:Ideogram-4 95s 冷载 → stash 恢复 ≤10s;Klein 22s → ≤5s)

## Follow-up(本 spec 记录,不在首批)

- **整模型组件注册进 L1**(用户 2026-06-12 问的 qwen CLIP 复用):`from_pretrained` 后把
  transformer/text_encoder/vae 注册进 `_components`(refcount 接管生命周期),加载时支持
  组件注入(`from_pretrained(..., vae=共享实例)`)。当前模型组合 TE 全不同、仅 flux2-vae
  0.32G 重叠,收益为零;出现共享大 TE 的架构组合时实施。
- 跨卡 stash 恢复(键含 device 的放宽)。
