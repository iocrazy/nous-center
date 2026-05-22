# Image 细粒度图收敛 — PR-4(删 Family B + PortType 收敛)Plan

> REQUIRED SUB-SKILL: superpowers:executing-plans。删除型 PR:删 → 跑全套 → 按 test 输出迭代清理。

**Goal:** 删除重复的 Family B 图像节点(`image_generate` + `image_unet_load`/`image_clip_load`/`image_vae_load`/`image_lora_apply`),节点库只剩一套 ComfyUI 风格细粒度图(flux2-components)。**保留** runner 的 ImageRequest/`get_or_load_image_adapter`/`ImageSampler` 执行引擎(细粒度图复用)+ `image_output` 终端节点 + `expand_legacy_image_spec`(Load Checkpoint 复用)+ `loras` 路由(Load LoRA 用)+ `image_l2_cache`/`image_files`(细粒度图复用)。

**Branch:** `feat/image-granular-convergence-pr4`。**Spec:** §6 删除清单。**前置:** PR-1/2/3 merged。

**风险:** 删除牵连面大(~8 src + ~14 test)。策略:先删 + 改 src,再 `pytest -q` 看 fallout,**Family-B-only 测试删除、混用测试改成 flux2/flux2_vae_decode**。每步小 commit。

---

## Task 1: 后端删 Family B 节点 + 注册

- [ ] **删** `backend/src/services/nodes/image_components.py`(4 个 loader inline 节点整文件)。
- [ ] **改** `backend/src/services/nodes/image.py`:删 `ImageGenerateNode` + `@register("image_generate")`(行 41-126),**保留** `ImageOutputNode`(`image_output` 终端,细粒度图 VAE Decode → image_output 仍用)。
- [ ] **改** `backend/src/services/workflow_executor.py`:
  - 行 14 import 去掉 `image_components`。
  - 行 41 `_NODE_TYPE_TO_GROUP_ID` 删 `"image_generate": "image"`(留 flux2_vae_decode + tts_engine)。
  - 行 232-246 老格式 `image_generate` inline 展开块整段删(`expand_legacy_image_spec` 函数留着给 Load Checkpoint)。
- [ ] **改** `backend/src/services/node_routing.py`:`DISPATCH_NODE_TYPES` 去 `image_generate`(留 `tts_engine` + `flux2_vae_decode`)。
- [ ] **改** `backend/src/runner/runner_process.py`:`_build_request` image 分支删 Family B **flat `unet/clip/vae` 路径** + 老 `model_key`/loras 兜底(image_generate 没了,只剩 granular 嵌套 latent 路径)。granular 分支保留。
- [ ] **改** `backend/src/api/routes/workflow_publish.py`:`_IMAGE_NODE_TYPES` 去 `image_generate`(留 `flux2_vae_decode`)。
- [ ] **注释更新(可选,防误导)**:`loras.py`/`image_l2_cache.py`/`image_files.py`/`component_spec.py` 注释里 image_generate → 细粒度图措辞。
- [ ] **跑 + 清理**:`cd backend && uv run pytest -q`,按 fallout:
  - **删**:`test_image_component_nodes.py`、`test_image_components_e2e.py`、`test_workflow_executor_legacy_expand.py`(均 Family-B-only)。
  - **改**:`test_image_node.py`(去 image_generate 用例留 image_output)、`test_workflow_executor_is_deterministic.py` / `test_workflow_executor_cached_event.py` / `test_workflow_executor_split.py`(fixture 用 image_generate → 换 flux2_vae_decode 或别的 dispatch)、`test_runner_build_request.py`(flat 路径用例删)、`test_runner_build_request_granular.py`(删 `test_legacy_flat_components_still_work`)、`test_node_routing.py`(DISPATCH 集合断言去 image_generate)、`test_workflow_publish_image.py`(image_generate workflow → 只留 flux2)、`test_tts_node_is_dispatch.py` / `test_image_diffusers.py` / `test_image_output_storage.py` / `conftest.py`(逐个看引用,多半是 fixture 节点类型,换掉)。
- [ ] `ruff check src tests` 干净。
- [ ] **Commit** `feat(image): PR-4 — 删 Family B 后端(image_generate + image_*_load),保留执行引擎/image_output/Load Checkpoint`

---

## Task 2: 前端删 Family B + PortType 收敛

- [ ] **改** `frontend/src/models/workflow.ts`:
  - `NODE_DEFS` 删 `image_generate`/`image_unet_load`/`image_clip_load`/`image_vae_load`/`image_lora_apply`(留 `image_output`)。
  - `BuiltinNodeType` union 去这 5 个。
  - `PortType` 去小写 `'unet' | 'clip' | 'vae'`(仅 Family B 用;细粒度图用大写 MODEL/CLIP/VAE,经 plugin defs 走字符串)。
- [ ] **改** `frontend/src/models/nodeRegistry.ts`:`DECLARATIVE_NODES` 删这 5 个;`NODE_CATEGORIES` 「图像」组去 `image_generate`(留 `image_output`;flux2 插件节点 merge 进来)。
- [ ] **改** `frontend/src/components/panels/NodeLibraryPanel.tsx`:删 `image_loading`「组件加载」category。
- [ ] **改** `frontend/src/components/nodes/DeclarativeNode.tsx`:若有 image_generate/Family-B 专属分支(如 imageStage 仅 image_generate)→ 改成 flux2_vae_decode/通用(grep 确认)。
- [ ] **跑 + 清理**:`tsc -b` + `vitest run`,按 fallout 改/删测试(`runners.test.tsx`/`CachedHint.test.tsx` 若用 image_generate → 换 flux2_vae_decode)。`npm run build`。
- [ ] **Commit** `feat(image): PR-4 — 删 Family B 前端 + PortType 收敛(节点库只剩一套细粒度图)`

---

## Task 3: 真机 + 收尾
- [ ] **grep 确认零残留**:`grep -rn "image_generate\|image_unet_load\|image_clip_load\|image_vae_load\|image_lora_apply" backend/src frontend/src`(只剩注释/无)。
- [ ] **真机**:起 backend + 打开编辑器 → 节点库「图像」只剩 Load Checkpoint/Diffusion/CLIP/VAE/LoRA/Encode/KSampler/VAE Decode + 图像输出,**无「图像生成」「组件加载」分类**。已存的细粒度 workflow(granular-smoke-pr1pr2 / pr3-clip-single)仍能跑出图。
- [ ] **更新 2026-05-19 spec Status**:标注 Family B 已被 2026-05-21 收敛取代。
- [ ] PR → CI 绿 → auto-merge。**收敛完成。**
