# 统一引擎库设计 — 全模型 + VRAM 常驻/加载状态

状态:设计(2026-06-02)。用户多次指出引擎库只覆盖一半模型,要求**所有图像模型(SeedVR2/单文件
组件 diffusion_models/clip/vae/LoRA/anima)都进引擎库 + 看到哪些常驻显存/已加载**。落地分多 PR。
前置背景见 [[project_unified_model_mgmt_gap]]。

## 问题:三套模型目录,引擎库只列一套
当前 `/api/v1/engines`(`list_all_engines`,engines.py:154)= `scan_models()` + 磁盘存在过滤 +
loaded 状态叠加(`mgr._models.get(name)`,engines.py:59/68,**按 registry name 匹配**)。

| 目录源 | 内容 | 在引擎库? | 现在怎么管 |
|---|---|---|---|
| **registry + 自动发现** | models.yaml(LLM/TTS/VL)+ `scan_models` 自动测 `image/diffusers/<整模型>`(model_index.json)+ LLM config.json | ✅ | 引擎库(idle/loaded/on-demand/resident/GPU) |
| **单文件组件** | `image/{diffusion_models,text_encoders,vae}/*.safetensors` + LoRA | ❌ | 仅节点下拉(component_scanner;model_scanner.py:17 `_IMAGE_COMPONENT_SUBDIRS` **显式 skip**) |
| **by-key** | SeedVR2(`image/SEEDVR2`,get_or_load_seedvr2_adapter,id=`image:SeedVR2:<hash>`)、anima | ❌ | 仅工作流加载;加载后进 `_models`+`loaded_models_snapshot`(Dashboard #281 聚合能看,引擎库按 name 匹配不上 hash id → 不列) |

→ 用户痛点:SeedVR2/组件/LoRA **空闲时引擎库完全不可见**,无法统一看「是否常驻显存 / 是否已加载 / 在哪张卡」。

## 设计目标
引擎库成为**单一模型管理面**:三来源合并的目录 + 每条目的 **VRAM 残留状态**(已加载 + 常驻 + 卡 + VRAM),
所有图像模型(含组件/LoRA/SeedVR2/anima)可见可管。

## 架构

### 1. 统一模型目录服务(catalog)
新增/扩展一个 catalog,合并三来源(去重、统一 schema):
- registry + 自动发现(现 `scan_models`/`scan_local_models`)
- 单文件组件(`component_scanner.scan_components` 的 diffusion_models/clip/vae/loras)+ SeedVR2(`seedvr2_dit_models_with_disk_status` + VAE)
- by-key 可加载单元(SeedVR2/anima)作为「整套」条目
统一 entry:`{id, name, category(语言/图像/视觉/TTS/组件/LoRA/超分), kind(整模型/组件/by-key), file/dir, vram_mb, local_exists, loadable(bool)}`。
**loadable 区分**:整模型/by-key(SeedVR2/anima)可独立加载;单文件组件(clip/vae/单 LoRA)**不独立可加载**
(随 pipeline 加载)→ 显示但「加载」动作可能 N/A 或「随用加载」。诚实标注,别给假按钮。

### 2. VRAM 残留状态叠加(核心:用户要的)
合并 `model_manager.loaded_models_snapshot()`(返 `{model_id, model_type, gpu_index, gpu_indices,
vram_mb, pipeline_class, source_files, last_used_ago_sec}`,跨 runner 子进程经 Pong 聚合)到 catalog:
- **匹配键**:现引擎库只按 registry name(engines.py:59)。改成**多键匹配** —— registry name / by-key
  model_id(`image:SeedVR2:<hash>`)/ `source_files`(组件文件路径,匹配单文件组件条目「随哪个已加载
  adapter 在用」)。
- 每条目算出:`loaded`(在显存)/ `resident`(常驻 flag,registry `resident:true`;by-key 默认非常驻)/
  `gpu` / `vram_mb` / `last_used`。

### 3. 引擎库 UI
- 分类 tab 扩展:图像下分「整模型 / 组件 / LoRA / 超分(SeedVR2)」或加 tab。
- 每卡显示残留状态徽标:**已加载@cuda:N / 常驻 / 空闲 / 按需**。「已加载」tab 列全部当前在显存的(含
  SeedVR2/组件),从 `loaded_models_snapshot`(单一真相)。
- 单文件组件卡:标「组件(随 pipeline 加载)」,不给独立加载按钮(避免假操作)。

## 关键接入点(已调研 file:line)
- `engines.py:154 list_all_engines` —— 组装入口,要合 catalog + loaded 多键匹配。
- `model_scanner.py:42 scan_models` / `:17 _IMAGE_COMPONENT_SUBDIRS`(现 skip 组件,要纳入或并行 catalog)。
- `component_scanner.py scan_components(role)` —— 单文件组件来源(diffusion_models/clip/vae/loras)。
- `image_seedvr2.seedvr2_dit_models_with_disk_status` —— SeedVR2 DiT 磁盘状态(已有);VAE 同补。
- `model_manager.loaded_models_snapshot()` —— 残留状态来源;`_models` by-key id。
- registry adapter 构造 `model_manager.py:218 cls(paths=spec.paths,**params)` —— SeedVR2 若注册进 registry,
  paths={model_dir,dit,vae} 可直接构造(但 model_dir 需绝对路径解析,现 yaml paths 是相对)。
- 前端:`useEngines` / 引擎库分类 tab / EngineCard 残留状态徽标。

## 决策(2026-06-02 用户「按你推荐的来」拍板)
- ✅ SeedVR2/anima 进引擎库:**catalog 动态发现**(不进 registry)—— 避免「引擎库 load」与「工作流
  by-key load」双 model_id 混乱;loaded 状态纯靠 snapshot 多键匹配。代价:引擎库 load/unload 单接 by-key(PR-3)。
- ✅ 单文件组件:**只显示 + has_adapter=False(UI 禁用加载按钮)**,不给假「加载」操作;标「随 pipeline 加载」。
- ✅ 常驻分两概念都露:`resident`(配置:空闲不卸)+ `loaded@cuda:N`(此刻在显存)。
- ✅ 扩展现有 `/api/v1/engines`(不另开端点)。

## 关键发现(精简 PR 范围)
「已加载」基础设施**已存在**:`aggregate_runner_loaded(app.state)` + `/api/v1/engines/loaded-adapters`
(engines.py:350)已聚合 runner 加载的 adapter(SeedVR2/combo,按 source_files)。截图「已加载 0」=
当时没加载。所以残留状态来源现成,PR-1 真正补的是 **idle 目录**(空闲时也列 SeedVR2/组件)+ 把 loaded
状态并进主 engines 列表(多键匹配)。

## ✅ PR-1 完成:后端 catalog + 残留状态多键匹配
`src/services/engine_catalog.py`:seedvr2_catalog_entries(磁盘已有的 DiT,kind=upscale,has_adapter=True)
+ component_catalog_entries(diffusion_models/clip/vae/loras,kind=component/lora,has_adapter=False)+
catalog_extra_engines(type 过滤);loaded/gpu 从 aggregate_runner_loaded 多键匹配(SeedVR2 按 model_id
前缀 image:SeedVR2: + DiT 文件名;组件按 source_files basename)。EngineInfo 加 `kind` 字段。engines.py
list_all_engines 末尾 extend。测试 test_engine_catalog(5,含残留多键匹配)+ 引擎库套 59 passed 无回归。

## PR 拆分(每个独立绿门控)
- PR-1 后端 catalog + 残留状态多键匹配:统一目录(三来源合并)+ loaded_models_snapshot 多键匹配 →
  `/api/v1/engines`(或新 `/api/v1/models/catalog`)返全模型 + loaded/resident/gpu。CI 安全 wiring 测试。
- PR-2 前端引擎库:分类/tab 扩展 + 残留状态徽标 + 「已加载」聚合全类型 + 组件「不可独立加载」标注。
- PR-3(按需)SeedVR2/anima 可从引擎库预热/卸载(load/unload 动作接 by-key path)。
- 真机验:引擎库列出 SeedVR2/组件 + 跑工作流后「已加载」实时反映。

## 约束/坑
- loaded 在 runner 子进程,主进程靠 Pong 聚合 `loaded_models_snapshot`(单一真相,别在主进程另算)。
- 单文件组件不是独立 engine(clip 单文件不能单独跑)—— 别硬塞「加载」按钮(假操作)。
- 跨进程可见性是历史复发 bug(见 [[project_workflow_ui_bugs]] useEngines 漏 runner adapter / #281)——
  多键匹配 + 单一真相(snapshot)要测真机。
参见 [[project_unified_model_mgmt_gap]]、[[project_output_delivery_service_layer]](同类管理面统一)、
[[feedback_long_term_robustness]]、[[feedback_push_before_impl]]。
