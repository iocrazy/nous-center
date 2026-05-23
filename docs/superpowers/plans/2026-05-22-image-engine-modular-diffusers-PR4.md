# PR-4 — 切默认 modular + 删自写 legacy 引擎(收官)Plan

> REQUIRED SUB-SKILL: executing-plans。
> Spec: `2026-05-22-image-engine-modular-diffusers-design.md`。前置 PR-1/2/3(#128-131)。用户选 D1=A 完整迁移、PR-4 一步切+删。

**Goal**:默认引擎 legacy→modular;删自写 ImageSampler + DiffusersImageBackend + legacy 组件加载链。迁移完只剩一套引擎。

**Branch**:`feat/image-modular-delete-legacy-pr4`。

## 删 / 留 / 搬(依赖图实测)

**删**:
- `image_sampler.py`(ImageSampler,只被 DiffusersImageBackend 用)
- `image_diffusers.py` 的 `DiffusersImageBackend` 类 + 其 helper(`load_diffusers_pipeline`/`load_quantized_transformer`/`_load_with_quantized_transformer`/`_build_empty_transformer_from_dir`/`load_component_module`/`_load_hf_or_quant`/`_torch_dtype_from` 等只服务 legacy 的)
- `model_manager`:`get_or_load_image_adapter` 的 **legacy 分支**(保留 modular 分支 + 改默认)、`get_or_load_component`/`_load_component_module`、组件 L1 缓存(`_components`/`_component_locks`/`_component_failures`,若仅 legacy 用)
- `_select_image_engine`:默认改 modular;或干脆去掉 selector + NOUS_IMAGE_ENGINE,image 永远 modular(收官后无并存)

**搬**(删 image_diffusers 前):
- `_maybe_convert_comfy_flux2_lora`(image_diffusers:321,被 image_modular 用)→ 移到 `image_modular.py`(或新 `lora_convert.py`)。

**留**:`image_modular`(ModularImageBackend/build_bridged_transformer)、`quant_loaders`(dequant_and_convert)、ComponentsManager、`_modular_repo_from_components`/`_is_comfy_single_file_unet`、runner 派发、component_scanner、editor、image_output_storage、四态/计费/服务发布。

## Tasks(每步 ruff + 相关测试,逐步验证)

- [ ] **T1 搬 `_maybe_convert_comfy_flux2_lora`** → image_modular(或 lora_convert);image_modular._apply_loras 改本地引用。跑 wiring + #125 转换相关测试。commit。
- [ ] **T2 默认切 modular**:`_select_image_engine` 默认 "modular"(或去 selector,image 恒 modular)。改 test_image_engine_selector 的默认断言。commit。
- [ ] **T3 删 DiffusersImageBackend + image_sampler** + image_diffusers 仅服务 legacy 的 helper。改 nodes/image.py 注释。commit。
- [ ] **T4 剪 model_manager**:删 get_or_load_image_adapter legacy 分支(保留 modular,简化:不再需要 engine 分支)、get_or_load_component/_load_component_module/组件 L1 缓存(确认仅 legacy 用)。commit。
- [ ] **T5 测试清理**:13 个引用 legacy(ImageSampler/DiffusersImageBackend/from_loaded_components/get_or_load_component)的测试 —— 删 legacy-only 的、改混用的到 modular。grep 确认 src 无 legacy 残留。
- [ ] **T6 真模型 + PR**:NOUS_IMAGE_ENGINE 不设(默认 modular)→ HF-layout 出图 + comfy fp8 + LoRA 各 smoke 过(复用现有 smoke)。后端全套 + ruff + 前端不涉及。PR → CI 全 pass(逐项)→ merge。grep 全仓无 image_sampler/DiffusersImageBackend 残留。

## 风险
- 大删除 + 13 测试:逐步小 commit,每步跑相关测试,别一次性大改。
- 默认切 modular = 生产上 experimental API(D5 钉 commit + smoke 门已在)。一步切+删不可逆(用户 D1=A 接受)。
- 删后 image 永远 modular:确认 ERNIE 等其它 image 模型也走 modular(ernie_image 官方 blocks 已在;若有 legacy-only image 模型会断 —— 实测确认)。
