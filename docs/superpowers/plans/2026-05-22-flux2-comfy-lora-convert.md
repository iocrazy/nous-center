# Flux2 ComfyUI/Kohya LoRA 格式支持(键重映射)Plan

> REQUIRED SUB-SKILL: executing-plans。

**Goal:** 让用户磁盘上的 ComfyUI/BFL 格式 Flux2 LoRA(`diffusion_model.double_blocks.N.img_attn.qkv.lora_down.weight` 这套)真能加载生效。当前 `_apply_loras → pipe.load_lora_weights(path)` 被 diffusers 的 `Flux2LoraLoaderMixin.lora_state_dict` 误判:`is_kohya = any(".lora_down.weight" in k)` 先命中 → 走 `_convert_kohya_flux2`(认 `lora_unet_` 前缀)→ 零匹配 → 我们的防御 `raise "loaded zero matching weights"`。

**根因(已查实):** diffusers `lora_pipeline.py:5677` 的 `is_kohya` 启发式太宽 —— ComfyUI/BFL LoRA **同时**有 `diffusion_model.` 前缀(应走 `_convert_non_diffusers_flux2`)和 `.lora_down.weight` 后缀(被 is_kohya 抢先)。`_convert_non_diffusers_flux2_lora_to_diffusers` **正好**处理这套(已实测:242 键→306 个有效 `transformer.*` diffusers 键,qkv 拆分/mlp/modulation 全映射)。

**修法:** `_apply_loras` 加载前检测该格式 → 自己调 `_convert_non_diffusers_flux2_lora_to_diffusers` 预转换 → `load_lora_weights(converted_dict)`(转换后键纯 diffusers 格式,不会被二次 mangle)。绕过 diffusers 的 dispatch bug。仅对 Flux2 pipe + 命中该格式时转换;其它(ERNIE/已是 diffusers 格式)走原路径。

**Branch:** `feat/flux2-comfy-lora-convert`。**前置:** 收敛 PR-1..4 merged。

---

## Task 1: ComfyUI Flux2 LoRA 预转换注入

**Files:** `backend/src/services/inference/image_diffusers.py` + `backend/tests/test_flux2_comfy_lora_convert.py`

- [ ] **Step 1: 失败测试** —`_comfy_flux2_lora_state_dict(path_or_sd) -> dict|None`:命中 ComfyUI/BFL 格式返回转换后 dict,否则 None。
```python
def test_detects_and_converts_comfy_flux2_lora():
    from src.services.inference.image_diffusers import _maybe_convert_comfy_flux2_lora
    sd = {"diffusion_model.double_blocks.0.img_attn.qkv.lora_down.weight": <t>,
          "diffusion_model.double_blocks.0.img_attn.qkv.lora_up.weight": <t>, ...}
    conv = _maybe_convert_comfy_flux2_lora(sd)
    assert conv is not None
    assert all(k.startswith("transformer.") for k in conv)
    assert any("lora_A" in k for k in conv)

def test_passthrough_non_comfy():
    # 已是 diffusers 格式 / kohya lora_unet_ → 不转(返回 None,走原路径)
    assert _maybe_convert_comfy_flux2_lora({"transformer.x.lora_A.weight": <t>}) is None
    assert _maybe_convert_comfy_flux2_lora({"lora_unet_double_blocks_0_...": <t>}) is None
```
> 用小的真 tensor(torch.zeros)构造最小 double_blocks LoRA,验转换器不崩 + 输出形态。也可直接对真文件 `klein_9B_Turbo_r128.safetensors` 跑(已实测 242→306)。

- [ ] **Step 2: 跑确认失败**
- [ ] **Step 3: 实现** —新 helper(模块级,detection 不依赖 torch 重逻辑):
```python
def _maybe_convert_comfy_flux2_lora(state_dict: dict):
    """ComfyUI/BFL 格式 Flux2 LoRA(diffusion_model. 前缀 + lora_down/up)→ 转成
    diffusers transformer.* 格式;否则返回 None(走原 load_lora_weights 路径)。
    绕过 diffusers Flux2LoraLoaderMixin 的 is_kohya 误判(它把这套路由到 lora_unet_
    转换器导致零匹配)。"""
    if not any(k.startswith("diffusion_model.") for k in state_dict):
        return None
    if not any(".lora_down.weight" in k or ".lora_up.weight" in k for k in state_dict):
        return None
    from diffusers.loaders.lora_conversion_utils import _convert_non_diffusers_flux2_lora_to_diffusers
    return _convert_non_diffusers_flux2_lora_to_diffusers(dict(state_dict))
```
`_apply_loras` 第 669 行改:
```python
                converted = None
                if isinstance(lora_path, str) and lora_path.endswith(".safetensors") \
                        and type(self._pipe).__name__.startswith("Flux2"):
                    from safetensors.torch import load_file
                    converted = _maybe_convert_comfy_flux2_lora(load_file(lora_path))
                if converted is not None:
                    self._pipe.load_lora_weights(converted, adapter_name=spec.name)
                else:
                    self._pipe.load_lora_weights(lora_path, adapter_name=spec.name)
```
- [ ] **Step 4: 跑通 + 回归** `pytest tests/test_flux2_comfy_lora_convert.py tests/test_from_loaded_components.py -q`
- [ ] **Step 5: ruff + Commit** `feat(image): Flux2 ComfyUI/BFL LoRA 格式预转换(绕 diffusers is_kohya 误判)`

---

## Task 2: 真模型验证 + PR

- [ ] **真模型 smoke**(standalone,需 GPU):细粒度图 + Load LoRA(klein_9B_Turbo_r128)→ get_or_load_image_adapter → infer。验:① 不再抛 "zero matching weights";② `get_active_adapters()` 含该 LoRA;③ 出图有效 + 与无 LoRA 同 seed 对比有差异(LoRA 真生效)。复用 smoke_granular_pr1.py 的 SMOKE_LORA。
- [ ] 全套 `pytest -q` + `ruff` + 前端不涉及(纯后端)。
- [ ] PR → CI 绿 → auto-merge。
- [ ] 真机(可选):编辑器拖 Load LoRA(turbo)接进细粒度图 → Run → 出图。
