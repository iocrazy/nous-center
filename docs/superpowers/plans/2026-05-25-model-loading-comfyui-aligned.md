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

## Task 1:组件角色只扫单文件夹(去 diffusers 子组件)

**Files**:`backend/configs/model_paths.yaml` + `backend/tests/test_component_scanner.py`

- [ ] model_paths.yaml:
  - `diffusion_models`: 留 `image/diffusion_models/**/*.{safetensors,gguf}` + `image/unet/**/*.{safetensors,gguf}`;**删 `image/diffusers/*/transformer/*`**。
  - `clip`: `image/text_encoders/**` + `image/clip/**`;**删 `image/diffusers/*/text_encoder/*`**。
  - `vae`: `image/vae/**`;**删 `image/diffusers/*/vae/*`**。
- [ ] 测试:断言组件角色不再含 diffusers/ 子组件(只单文件夹)。

## Task 2:新 "checkpoint" 角色 —— 扫 diffusers/ 整模型目录

**Files**:`backend/src/services/component_scanner.py` + 测试

- [ ] `ROLE_DIRS` 加 `"checkpoint"`。新 `_scan_checkpoints()`:列 `image/diffusers/*/` **目录**(含 `model_index.json` 的=整 diffusers 模型),entry = {filename: 目录名, abs_path: 目录路径, kind: "checkpoint"}。
- [ ] model_paths.yaml 加 `checkpoint:` 角色(标记 diffusers 根)或在 scanner 特判 diffusers 目录扫描。
- [ ] 测试:diffusers/<model>/(有 model_index.json)被列;无 model_index 的不列。

## Task 3:Load Checkpoint 节点改 component_select(checkpoint)+ 目录→组件 resolver

**Files**:`backend/nodes/flux2-components/node.yaml` + `executor.py` + 测试

- [ ] node.yaml `flux2_load_checkpoint`:`model_key`(model_select)→ `file`(component_select, role: checkpoint)。
- [ ] `exec_load_checkpoint`:输入 diffusers 目录路径 → 解析 `<dir>/transformer`(首片)、`<dir>/text_encoder`、`<dir>/vae` → 三组件描述符(同 device/dtype)。替代 registry.get(model_key)+expand_legacy_image_spec。
- [ ] 测试:目录 → 三组件 spec(HF-layout 结构)。

## Task 4:启动扫描 + 自检

**Files**:`backend/src/api/main.py`(lifespan)

- [ ] 启动时 `component_scanner` 扫一遍 + 日志自检:每角色找到几个、diffusers 整模型缺子目录(transformer/text_encoder/vae)告警。不阻塞启动。

## Task 5:清 stale 注册表 adapter(顺带)

- [ ] 注册表 image 模型的 `adapter: DiffusersImageBackend`(已删)→ 改 modular 或移除(确认 generate.py 等不依赖它出错)。Load Checkpoint 不再用注册表后,评估 image 注册表条目是否还需要。

## Task 6:真模型验证 + PR

- [ ] standalone smoke:Load Checkpoint 选 `diffusers/Flux2-klein-9B` 整模型 → 出图(走目录→组件→modular)。Load Diffusion Model 下拉只剩 diffusion_models/ 单文件。
- [ ] 后端全套 + 前端 tsc/vitest/build;真机硬刷新:Load Diffusion Model 只 DiT 单文件、Load Checkpoint 列 diffusers/ 整模型。
- [ ] PR → CI 全 pass → merge。
