# 模型加载对齐 ComfyUI:单文件组件 vs 整模型分离 Plan

> REQUIRED SUB-SKILL: executing-plans。

**Goal(用户确认的设计)**:区分两类模型的存放 + 加载,对齐 ComfyUI:
- **DiT 单文件**(comfy 量化 / 原生扩散模型本体)→ `diffusion_models/`(+`unet/` 别名)→ **Load Diffusion Model**(组件,只出 MODEL)。
- **整模型**(HF-layout diffusers)→ `diffusers/`→ **Load Checkpoint**(扫 diffusers/ 目录自动列,出 MODEL+CLIP+VAE)。
- **弃用 checkpoints** 文件夹概念。
- 启动时扫描 + 自检(日志报每类找到啥 / diffusers 整模型子目录完整性)。

**当前混淆(实测)**:① 组件角色(diffusion_models/clip/vae)扫了 `diffusers/*/{transformer,text_encoder,vae}`(整模型子组件);② 注册表把 comfy 单文件也当"模型"列进 Load Checkpoint;③ 注册表 image 模型 adapter 仍写已删的 `DiffusersImageBackend`(stale)。

**Branch**:`feat/model-loading-comfyui-aligned`。

---

## Task 1:组件角色只扫单文件夹(去 diffusers 子组件)✅

**Files**:`backend/configs/model_paths.yaml` + `backend/tests/test_component_scanner.py`

- [x] model_paths.yaml:diffusion_models=`diffusion_models/**`+`unet/**`;clip=`text_encoders/**`+`clip/**`;vae=`vae/**`;均删 `diffusers/*/{transformer,text_encoder,vae}`。
- [x] 测试:`test_diffusers_subcomponents_excluded_from_component_roles`。

## Task 2:新 "checkpoint" 角色 —— 扫 diffusers/ 整模型目录 ✅

**Files**:`backend/src/services/component_scanner.py` + 测试

- [x] `ROLE_DIRS` 加 `"checkpoint"`;`_scan_checkpoints()` 列 `image/diffusers/*/`(含 model_index.json)目录,quant_type="checkpoint"。
- [x] model_paths.yaml 加 `checkpoint: []`;`_scan_all` 特判 checkpoint → 目录扫描(非 glob)。
- [x] 测试:`test_scan_checkpoints_lists_only_complete_diffusers_dirs`(有 model_index 才列)。

## Task 3:Load Checkpoint 节点改 component_select(checkpoint)+ 目录→组件 resolver ✅

**Files**:`backend/nodes/flux2-components/node.yaml` + `executor.py` + 测试

- [x] node.yaml:`file`(component_select, role: checkpoint)。
- [x] `exec_load_checkpoint`:diffusers 目录 → `<dir>/{transformer,text_encoder,vae}` 首片 → 三描述符(同 device/dtype);_first_safetensors 助手。
- [x] 测试:`test_flux2_checkpoint_resolve.py` 4 个(三组件 / 默认 bf16 / 缺 file / 缺子目录)。

## Task 4:启动扫描 + 自检 ✅

**Files**:`backend/src/api/main.py`(lifespan)+ `component_scanner.selfcheck_report()`

- [x] `selfcheck_report()`:每角色计数 + 整模型缺 transformer/text_encoder/vae 告警(不抛)。lifespan 调用 + log,fail-soft 不阻塞。测试 2 个。

## Task 5:清 stale 注册表 adapter(顺带)✅

- [x] 用户定:**删除**。从 models.yaml 移除 3 条指向已删 `DiffusersImageBackend` 的 image 条目(图像走组件路径,registry 不再是来源)。测试改 `test_models_yaml_has_no_image_entries`;更新 models.py 路由 stale 文档。
  理由:ModularImageBackend 构造签名与 `_instantiate_adapter` 不兼容,「改 adapter 字符串」并不能让 load_model 真跑通 —— 删除是唯一诚实/健壮选项。

## Task 6:真模型验证 + PR(进行中)

- [x] standalone smoke `smoke_load_checkpoint_dir.py`:exec_load_checkpoint(目录)→ 摊平 → get_or_load_image_adapter → 出图(cuda:1)。
- [x] 后端全套(968 passed/8 skipped,改完 1 个 route 角色断言)+ 前端 tsc/vitest(95)+ ruff src/tests。
- [ ] vite build;PR → CI 全 pass → merge。
