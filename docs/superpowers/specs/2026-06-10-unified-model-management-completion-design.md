# 统一模型管理收尾(设计)

**Status**: Draft
**Author**: heygo(设计 by Claude)
**Date**: 2026-06-10

## 0. 动机 / 现状校准(读真代码 + 活 /engines,2026-06-10)

用户看 image 引擎组织,问「不同模型(Flux/Z-Image)一个文件吗?好集成/卸载/管理吗?」。**核实后:统一引擎库
当年标的"大改"(`project_unified_model_mgmt_gap`)其实已大部分做完(#303/304/305),本 spec 是收尾打通,不是大改。**

### 已做(确认在线)
- **代码层引擎结构干净**:`image_modular.py`(`ModularImageBackend`)用 `pipeline_class` 分派 **Flux2 / Z-Image /
  Qwen-Edit**(共一文件,`_build_{klein,zimage,qwen_edit}_pipe`);Anima/SeedVR2 因非 diffusers 系单独成文件。
  加新 diffusers 模型 = `IMAGE_ARCH_REGISTRY` 加一行 + 一个 `_build_*_pipe` 分支。
- **统一目录**:`engine_catalog.py` 把 SeedVR2(kind=upscale)+ 单文件组件(diffusion_models/clip/vae=component,
  loras=lora)并入 `/api/v1/engines`。活查 50 条:18 model + 1 upscale + 23 component + 8 lora,**本 session 新下的
  Z-Image-base/turbo / GGUF 编码器 / 全 LoRA / anima 单文件全在列**,带 VRAM 残留状态(aggregate_runner_loaded 多键匹配)。
- **载卸能力**:SeedVR2 引擎库可独立载卸(`/engines/seedvr2/{preload,unload}`,#305);组件 **后端 preload 端点已存在**
  (`/engines/component/preload` + `/api/v1/models/components/preload`,L1 缓存 + resident);`unload_model(model_id)` /
  `evict_lru` / `loaded_models_snapshot` / `/engines/loaded-adapters` 齐。

### 剩的 gap(本 spec 范围)
1. **组件载卸能力与 UI 脱节**:后端能 preload 单文件组件(diffusion_models/clip/vae)+ resident 钉,但 catalog 标
   `has_adapter=False` → 引擎库 UI 不给载/卸/常驻按钮。用户看得到、动不了(只能靠工作流隐式加载)。
2. **anima 非可加载条目**:anima 单文件只作 `component:diffusion_models` 显示,没有像 SeedVR2 的独立可加载 adapter 入口。
3. **已加载 combo 不能从引擎库卸单个**:Z-Image 双采等 ad-hoc combo(`_models` 字典 by combo-hash)在 Dashboard/
   loaded-adapters 可见,但引擎库列表按 catalog 条目组织,卸单个 combo 没入口(只能整卡 evict)。
4. **50+ 条目无分组**:component(23)+ lora(8)平铺,缺按角色(diffusion_models/clip/vae/loras)分组折叠。
5. **真机未验**:SeedVR2/组件 引擎库载→显存涨/卸→掉 跨进程链路 #305 当时标「待真机验」,至今未确认。

## 1. 设计原则
- **不重复造**:复用已有 preload/unload 端点 + catalog;只补「能力↔UI」缺的线 + 分组 + 验证。
- **组件载卸语义老实**:单文件组件**不能脱离 pipeline 独立推理**(clip/vae 要配 transformer),"加载"= 预热进
  L1 缓存常驻(下次 combo 装配命中,免重载),"卸载"= 出 L1 + 释放该组件显存。catalog `has_adapter` 改成
  区分「可独立推理(SeedVR2)」vs「可预热常驻(组件)」两种可管理性,别用一个 bool 糊。
- **长远**:三来源(registry / 组件扫描 / by-key)已合到 catalog 单一出口;保持引擎库 + 节点下拉共用它。

## 2. PR 拆分(独立分支 + CI 绿)

### PR-1:组件预热/常驻接进引擎库(能力↔UI 打通)
- `EngineInfo` 的 `has_adapter: bool` → 加 `manageable: "loadable" | "preloadable" | "view_only"`
  (SeedVR2/整模型=loadable;单文件组件=preloadable;无 adapter 兜底=view_only)。保留 has_adapter 兼容。
- catalog 组件条目带其 `ComponentSpec`(role/file/arch 推断)→ 前端「预热常驻 / 出缓存」按钮调已有
  `/engines/component/preload`(resident=true)+ 新增 `/engines/component/unload`(包 `unload_model` /
  出 L1)。loaded 状态读 aggregate_runner_components(组件 L1 快照,已存在)。
- **验证**:引擎库点组件「预热」→ runner L1 出现该组件 + 显存涨;「出缓存」→ 掉。真机(跨进程)。

### PR-2:anima 可加载条目 + 已加载 combo 卸载入口
- `engine_catalog` 加 anima 条目(kind=image/model,has adapter `AnimaImageBackend`,like SeedVR2 by-key)。
- 引擎库「已加载」分区列 `loaded_models_snapshot` 的 combo(by hash,显示组成组件 + 卡 + LRU 位)→ 每条给
  「卸载」调 `/engines`→`unload_model(model_id)`。整卡「释放显存」调 evict 链。
- **验证**:真机加载 anima 出图 + 引擎库可卸;跑 Z 双采后引擎库「已加载」列出 combo + 可单卸。

### PR-3:前端分组折叠 + 收尾
- ModelsOverlay 按 kind/role 分组折叠(diffusion_models / clip / vae / loras / 整模型 / 超分 / 已加载);
  组件多时默认折叠,搜索过滤。
- **验证**:50+ 条目分组清爽;chrome-devtools 巡检载卸状态徽标正确。

## 3. 不做 / 风险
- **不**把组件做成「独立推理引擎」(clip/vae 物理上要配 transformer)—— 只做预热常驻管理。
- 组件很多(loras 易上百)→ 分组折叠 + 搜索必须(PR-3),否则引擎库被刷屏。
- 跨进程载卸 CI 测不了(runner 子进程 mock torch)→ 每个 PR 真机验(引擎库点→显存增减),对齐 #305 教训。
- resident 钉是 in-memory(runner 重启失效),非 yaml 持久 —— 维持现状(组件非 registry 模型)。

## 4. 范围
执行顺序:PR-1(组件载卸打通)→ PR-2(anima + combo 卸载)→ PR-3(分组 UX)。每 PR 独立分支 + CI 绿 + 真机验载卸。
参见 [[project_unified_model_mgmt_gap]]、[[project_component_l1_cache]]、[[project_service_api_layer]]、
[[project_seedvr2_three_node]]、[[feedback_long_term_robustness]]。
