# 组件级 L1 缓存 + 跨 combo 复用 + 预加载/常驻设计

状态:设计(2026-06-02)。用户需求:**同一组件(file+device+dtype = 一个 id)被多个工作流共享时
只加载一份、跨 combo 复用,可从引擎库预加载 + 常驻 pin**。落地分多 PR。改图像加载核心,**必真模型 smoke**
(CLAUDE.md)。前置:[[project_unified_model_mgmt_gap]](统一引擎库已做 PR-1/2/3)。

## 用户需求(原话场景)
- 工作流 A 用(模型X-bf16 + clipY + vaeZ)
- 工作流 B 用(模型X-bf16 + clipW + vaeZ)
→ **X-bf16 和 vaeZ 在 A、B 间复用,不重载;只有 clip 不同各加载各的。**
+ 引擎库能预加载单个组件(指定精度/卡 = 一个 id)+ 常驻 pin(大模型 16GB 钉住免重载)。

## 现状:只有 combo 级缓存,无组件级复用
- `get_or_load_image_adapter`(model_manager.py:974):combo_key = (pipeline_class, offload) +
  (to_component_key(diffusion_models/clip/vae))。**整套 adapter 按 combo_key 缓存进 `_models`**(L2)。
- combo_key 任一组件不同 → MISS → `_get_or_load_modular_adapter` 调 `build_bridged_transformer/
  text_encoder/vae`(image_modular.py)**各自从单文件重建**,无单组件缓存。
- **diffusers ComponentsManager 已删**(image_modular.py:294 "PR-A 删了 components_manager")→ 没有
  原生组件共享,现在是标准 Flux2KleinPipeline。
- **组件 id 基础设施已有**:`to_component_key(spec)` = (file, device, dtype, loras)(component_spec.py:80);
  `component_state_key` 给加载器节点四态点用。`_components` dict 字段存在但 vestigial(单组件加载方法已删,
  model_manager.py:781)。
→ 所以「X-bf16 跨 combo 复用」**现在不成立**(combo 不同就全重建)。差的是组件级 L1 缓存 + 复用层。

## 设计

### 1. 组件级 L1 缓存(`_components`)
- `_components: dict[ComponentKey, LoadedComponent]`,key = `to_component_key`(file|device|dtype|loras)。
- `_get_or_load_modular_adapter`:装配前,对 transformer/clip/vae **逐个查 L1**:命中 → 复用已加载模块;
  未命中 → `build_bridged_*` 建好 + 存 L1。然后把(可能共享的)三模块组装成 Flux2KleinPipeline。
- combo L2 缓存保留(整套命中最快);combo miss 但组件 L1 命中 = 部分复用(用户场景:只重建 clip)。

### 2. 引用计数(关键正确性)
- 共享组件被多个 combo 引用 → `_components[key].refs: set[combo_id]`。evict/unload 一个 combo 时,
  组件 refs 减;**refs 空 + 非 resident 才真释放**。否则 X-bf16 被 B 用着、卸 A 不能释放它。

### 3. ⚠️ 最大风险:共享模块的设备/offload 管理
- 两个 combo 共享同一 transformer 模块实例,但 offload 模式可能不同(A=cpu offload,B=none)→
  A 的 `enable_model_cpu_offload` hook 会把共享模块挪 CPU,B 在 GPU 用就崩。
- **决策**:组件级复用**只在相同 (device, offload) 下共享**;不同 offload → 不共享(算不同 L1 entry 或
  退回 combo 级)。或共享组件强制 offload=none(常驻 GPU),offload 只作用于非共享部分。**这块设计要
  在 PR-1 真模型验**(offload + 共享的交互是 segfault 高发区,见 runner bug hunt 历史)。

### 4. 预加载 + 常驻(组件级)
- 新端点 `POST /engines/component/preload`(file+role+device+dtype → 派 image runner →
  L1 加载该组件)+ resident pin(`_components[key].resident=True` → 不被 LRU evict)。
- 引擎库组件卡:选精度/卡 → 预加载 + 常驻 pin(撤销 PR-2 的 has_adapter=False 门控,对组件开放
  「预加载/常驻」动作)。loaded 状态多键匹配已有(unified-engine-library)。
- SeedVR2 常驻 pin 顺带做(by-key 完整模型,简单)。

## 接入点(file:line)
- `model_manager.py:974 get_or_load_image_adapter` / `:1121 _get_or_load_modular_adapter`(加 L1 查/存 + refcount)。
- `image_modular.py build_bridged_*`(:86/:123/:150)(返模块;L1 缓存这些)。
- `component_spec.py:80 to_component_key`(L1 key,已有)。
- `_components` dict(model_manager.py:129/162,resurrect)。
- evict/unload(model_manager.py:596 区 resident)+ 组件 refcount。
- 新:component preload 端点 + 运行时 runner 消息(照 PreloadComponents/PreloadSeedVR2)。
- 前端:引擎库组件卡开放预加载/常驻动作 + 选精度。

## PR 拆分
- PR-1 组件级 L1 缓存 + 复用 + refcount(核心):`_get_or_load_modular_adapter` 逐组件 L1。**真模型 smoke:
  A(X-bf16+Y+Z)→ B(X-bf16+W+Z),验 X/Z 不重载(日志/计时)+ 出图正确 + 共享 offload 不崩。**
- PR-2 组件预加载 + 常驻 pin(端点 + runner 消息 + resident)+ SeedVR2 常驻 pin。
- PR-3 前端:引擎库组件卡开放预加载/常驻 + 选精度。

## 风险/坑
- 共享模块 offload/device(§3)—— 最大坑,真机验。
- refcount 错 → 卸了在用的组件 → segfault(runner bug 历史多发)。
- LRU 显存压力:组件级缓存更多小实体,evict 策略要按组件 + 引用 + resident。
- 改图像引擎必真模型 smoke(CLAUDE.md);CI mock torch 测不了。
参见 [[project_unified_model_mgmt_gap]]、[[project_image_component_multigpu]]、[[feedback_verify_real_model]]、
[[feedback_long_term_robustness]]、[[feedback_push_before_impl]]。
