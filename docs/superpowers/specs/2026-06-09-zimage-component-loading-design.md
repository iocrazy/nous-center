# Z-Image 引擎地基:组件化加载 + 采样器放开(设计)

**Status**: Draft
**Author**: heygo(设计 by Claude)
**Date**: 2026-06-09

## 0. 动机

用户「Zimage 双采 + Flux2-Klein 三采」人像写真工作流(`~/Downloads/...还原真实皮肤质感.json`)
真机复刻被 nous **引擎能力**卡死(模型文件另说,见 §6)。读真工作流(89 节点)发现 Z-Image 段:

- **分开载入**:`UNETLoader` 单文件(base=`Z-Image-base`、refiner=`z_image_turbo_bf16`,**两个不同 UNet**)
  + 独立 `VAELoader`(`ae.safetensors`=Flux1 VAE)+ `CLIPLoaderGGUF`(`Qwen3-4b-Z-Image-Engineer` GGUF 微调编码器)。
- **采样器**:base=`euler+simple`、refiner=`euler_ancestral+beta`(留噪 latent 接力双采)。

nous 现状(两处硬限制):
1. **Z-Image 是「整模型 only」**:`image_modular._build_zimage_pipe` 只走 `ZImagePipeline.from_pretrained`,
   **无组件 override 路径**(Flux2 的 `_build_klein_pipe` 有 `len(overrides)==3` 的单文件桥接装配,Z-Image 没有)。
   ⇒ 单文件 UNet / 独立编码器·VAE / base≠refiner 不同模型 全加载不了。
2. **Z-Image 采样器锁死**:`ZImageTurboArchAdapter.supported_samplers={euler}` / `supported_schedulers={normal}`;
   引擎 Z-Image 分段循环(PR-B2)用 `scheduler.set_timesteps`(normal/shift=3)。⇒ simple/beta/euler_ancestral 全拒。

本 spec 补这两块**引擎地基**(不依赖缺失模型文件即可落地 + 真机验证,用 Z-Image-Turbo 整模型的组件子目录自比)。

## 1. 已钉的事实(读真代码)

- Flux2 单文件桥接装配已成熟:`_build_klein_pipe` overrides==3 → `klein_cls(transformer=, text_encoder=,
  tokenizer=, vae=, scheduler=, is_distilled=False)`;桥接 loader `build_bridged_{transformer,text_encoder,vae}`
  (`image_modular.py`)—— **但写死 Flux2**:`Flux2Transformer2DModel` / `AutoencoderKLFlux2`。
- `ZImagePipeline.__init__(transformer, text_encoder, tokenizer, vae, scheduler)`(diffusers 钉的 commit)——
  和 Klein 同形,可同法组件装配。Z-Image transformer 类=`ZImageTransformer2DModel`;VAE=`AutoencoderKL`
  (Flux1 16ch,非 Flux2 的 `AutoencoderKLFlux2`)。
- PR-B2 已给 Z-Image **手写去噪循环**(`_run_zimage_segmented`)——euler_ancestral(ancestral 重加噪)可在此循环内实现
  (diffusers `FlowMatchEulerDiscreteScheduler` 无 ancestral;但我们已手写循环,加 x0→re-noise 一步即可)。
- `sigma_schedules.py` 已有 simple/beta/sgm_uniform/... 的 sigma 算法(ComfyUI ground-truth 验过),
  Z-Image 整段/分段都可改用 `compute_sigmas(scheduler, steps, shift)` 取代写死 `set_timesteps(normal)`。
- 组件预加载/combo 装配走 `build_bridged_*` 的 dispatch(`model_manager.preload_image_component` +
  `_get_or_build_image_component`),按 role 选 build_fn —— 目前全 Flux2 桥接。

## 2. PR 拆分(独立分支 + CI 绿 + 引擎改真机 smoke)

### PR-1:Z-Image 采样器放开(scheduler + ancestral)
- `ZImageTurboArchAdapter.supported_schedulers` 扩到 9 个(对齐 Flux2:normal/karras/exponential/beta/
  simple/sgm_uniform/ddim_uniform/linear_quadratic/kl_optimal);`supported_samplers` 加 `euler_ancestral`。
- 引擎 Z-Image 路径(整段 + 分段)算 sigma 改 `compute_sigmas(scheduler, steps, shift=3)`(取代写死
  `set_timesteps(normal)`)→ 覆盖 `scheduler.sigmas/timesteps`(分段已这么做,整段也统一)。`normal` 默认零回归。
- `euler_ancestral`:在 `_run_zimage_segmented` 手写循环里实现 ancestral 步(每步算 x0 → 按 sigma_down 去噪 +
  sigma_up 重加噪,k-diffusion 公式;对照 ComfyUI `sampling.sample_euler_ancestral` 逐式)。非 ancestral 走原 euler。
- **验证**:`smoke_zimage_split.py` normal 路径 golden 不变(SSIM 1.0);新增 simple/beta 出图正确;
  euler_ancestral 出图正确(与 ComfyUI 同参对比构图一致)。`smoke_image_ab.py` Flux2 零回归。

### PR-2:Z-Image 组件 override 加载(分开载入)
- `_build_zimage_pipe` 加 overrides 分支:有 transformer/text_encoder/vae override → `ZImagePipeline(...)` 组件装配
  (tokenizer/scheduler 从参考库;同 Klein 路径)。
- 桥接 loader 去 Flux2 硬编码:按 arch 选 transformer 类(`ZImageTransformer2DModel` vs `Flux2Transformer2DModel`)
  + VAE 类(`AutoencoderKL` vs `AutoencoderKLFlux2`)。`build_bridged_{transformer,vae}` 加 arch 参数 /
  `model_arch_adapter` 注册各 arch 的 module 类(更收口)。
- **验证**:把 Z-Image-Turbo 整模型的 `transformer/`·`text_encoder/`·`vae/` 子目录权重当单文件**分开装配** →
  与 `from_pretrained` 整模型出图 **SSIM 1.0**(同权重不同装配路径,真机自比,不依赖缺失文件)。

### PR-3:GGUF 文本编码器(2026-06-10 调研后定:走 transformers 原生 GGUF)
- 目标:`CLIPLoaderGGUF` 等价 —— 加载 `Qwen3-4b-Z-Image-Engineer-V4-Q8_0.gguf`(原工作流的 Z-Image 编码器)。
- **调研结论(读真文件 + ComfyUI/transformers 源)**:GGUF=qwen3 架构,Q8_0(253 张量)+ F32(145 norm),
  **键用 llama.cpp 命名**(`blk.0.attn_k.weight` 非 HF `model.layers.0.self_attn.k_proj.weight`)。
  **→ 自写 dequant 还得手动 remap 键;transformers 原生 `from_pretrained(..., gguf_file=)` 内部做 dequant +
  key 重映射,且 qwen3 已在 `GGUF_SUPPORTED_ARCHITECTURES`** → 走原生路径,~20 LOC。
- 改点:`quant_loaders.py` 删 `reject_gguf` 桩(行 92-97);`build_bridged_text_encoder`(`image_modular.py`
  行 191-215)加 `.gguf` 分支 —— 不走 `dequant_comfy_mixed`,直接
  `AutoModelForCausalLM.from_pretrained(repo/text_encoder, gguf_file=clip_spec.file, torch_dtype=bf16)`。
- 风险:transformers GGUF 支持仍演进;dtype(Q8_0→fp16→bf16 cast)；tokenizer 仍从参考库(GGUF 内 tokenizer 不用)。
- **验证**:`smoke_zimage_gguf.py` 用真 GGUF 装配出图 vs 普通 `qwen_3_4b.safetensors` 出图(同 prompt;
  编码器是不同微调,不要求 SSIM=1,要求出图正确 + 风格朝 Engineer 偏);Flux2 零回归。

### PR-4:ColorMatch 后处理节点(2026-06-10 调研后加)
- 目标:复刻原工作流 `ColorMatch ['mkl', 0.65, True]`(肤色保持)= image→image inline 节点。
- **调研结论**:源 = ComfyUI-KJNodes `image_nodes.py:69`,依赖 `color-matcher` PyPI 包(MIT,纯 numpy/scipy 无模型);
  mkl=Monge-Kantorovich Linearization;strength 线性插值 `src + strength*(matched-src)`。
- 改点:新建 `backend/nodes/image-color-match/{node.yaml,executor.py}`(照 `image-io`/`seedvr2` inline 模式:
  上游两路 image_url → runner `_resolve_input_image_path` 解析本地 → `ColorMatcher().transfer(src,ref,method)` →
  strength 混合 → `write_image` 落盘签 URL)。`pyproject.toml` 加 `color-matcher`。
- **验证**:真后端 e2e:终图 + 参考图 → ColorMatch → 肤色朝参考偏移、强度可调。

### 不做(2026-06-10 调研定,工作量/架构边界):
- **LCS 四节点(SharpnessIntervene/ColorAnchor/Calibrate/LoadData)≠ 后处理,做不了 image→image**:
  源 = `comfyui-lcs` 自定义包,本质是 **FLUX 采样期 post-CFG hook**(`set_model_sampler_post_cfg_function`,
  在去噪循环逐步沿锐化/保色方向改 latent,依赖 sigma schedule)。生成完再用 = 无效。复刻 = 给 nous 采样循环加
  **post-CFG hook 机制**(model patch 注入 + sigma 感知)= 独立架构工程,本 spec 不含。需用户单拍。
- **ReferenceLatent 已等价复刻,无需新建**:nous Flux2 `image=`(strength=1 默认,非 img2img)直接喂
  `Flux2KleinPipeline.__call__`,pipeline 内部把参考图编码 latent **token concat 进主序列**(kontext 式,
  pipeline_flux2_klein.py:838-840)= 与 ComfyUI `VAEEncode→ReferenceLatent` 同机制。差异仅:原工作流 steps=4
  (我 e2e 填 8,改参即可)+ 原图带第二个「负向潜空间抑制」ReferenceLatent(精修,可选)。

## 3. 不做 / 风险

- **不** bump diffusers。
- euler_ancestral 是手写 ancestral(对照 ComfyUI 逐式),需真机核构图;`FlowMatchEuler` 无原生 ancestral。
- Z-Image VAE 是 Flux1 `AutoencoderKL`(16ch),桥接类选错(套 Flux2 的 `AutoencoderKLFlux2`)会 shape 错 —— PR-2 按 arch 派发。
- 缺失模型文件(6 LoRA / base·refiner 单文件 / True-V2 / GGUF 编码器)**不在本 spec 范围**(地基补齐后用户自备文件即可复刻)。

## 4. 范围(用户 2026-06-09 拍:先补引擎能力)

执行顺序:PR-1(采样器)✅ → PR-2(分开载入)✅ → PR-3(GGUF 编码器)→ PR-4(ColorMatch 节点)。
2026-06-10 用户拍:补齐距 1:1 的差距(逐节点比对见 [[project_zimage_portrait_replication]]):(a)✅CFG 修正已验
(b)PR-3 GGUF + PR-4 ColorMatch(c)LCS post-CFG hook = 独立架构工程待单拍、ReferenceLatent 已等价。
每 PR 独立分支 + CI 绿 + 引擎改真机 smoke。
参见 [[project_unified_model_mgmt_gap]]、[[project_multi_sampling_cross_model]]、[[feedback_read_comfyui_source]]。
