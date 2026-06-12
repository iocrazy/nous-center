# Ideogram-4 双 DiT 单文件精细路(整装/精细双线对齐)

日期:2026-06-12 | 状态:spec | 来源:用户「都补一下」双线并行 arc(Z-Image #512 之后,
ideogram4 单文件补齐);可行性已 spike 验过(分支 `spike/ideogram4-singlefile`,真机出图)

## 动机

[[project_unified_model_mgmt_gap]] 的双线盘点:整装(diffusers 整模型,Load Checkpoint)
vs 精细(comfy 单文件 DiT + clip/vae 组装,Load Diffusion Model)。Z-Image 已补齐双线
(#512)。**ideogram4 现在只有整装**(`diffusers/Ideogram-4-bf16`,58G bf16),但用户有
完整 comfy 单文件集(`【83】Ideogram4全自动流程/models/`):

- `ideogram4_fp8_scaled.safetensors`(9.3G,conditional DiT)
- `ideogram4_unconditional_fp8_scaled.safetensors`(9.3G,unconditional DiT —— 双模型非对称 CFG)
- `qwen3vl_8b_fp8_scaled.safetensors`(10.6G,Qwen3-VL 文本编码器)
- `flux2-vae.safetensors`(0.3G,= Flux2 同款 `AutoencoderKLFlux2`)

**价值**:单文件 fp8 全套 ~29G,整模型 bf16 58G。fp8 直接塞 48G(2×3090 跨卡)/ 配
[[project_lowvram_streaming]] 流式分块塞 24G 单卡。给小卡跑 ideogram4 的现实路径。

## 为什么比 Z-Image 难:双 DiT 破「一 DiT → 一 MODEL」假设

Z-Image #512 几乎零成本(后端 PR-2 分开载入早接好,只暴露 loader 下拉)。ideogram4 不同:

1. **diffusers 钉的 commit 没注册 Ideogram4 的 `from_single_file`**(spike 实测报
   `FromOriginalModelMixin is currently only compatible with ...`,列表无 Ideogram4Transformer2DModel)。
   → 不能走 z-image 的 `from_single_file`,改 flux2 式 `dequant_comfy_mixed` + 手写键转换。
2. **双 DiT**:`Ideogram4Pipeline` 需 `transformer` + `unconditional_transformer` 两个 DiT。
   现有组件/loader 系统假设「一个 diffusion_models 单文件 → 一个 MODEL 端口」——**没有第二个 DiT 的槽位**。
   `_build_ideogram4_pipe` 当前只 `from_pretrained`,无 override 装配路径。
3. **Qwen3-VL TE**:`Qwen3VLModel`(非普通 Qwen3 CausalLM),`build_bridged_text_encoder` 用
   `AutoModelForCausalLM` 接不住。

## 侦察结论(spike 真机验过,`spike/ideogram4-singlefile`)

转换器**纯机械、key-exact**(CPU load missing=0 / unexpected=0,真机装配出图连贯正确):

- **DiT comfy → diffusers**:`attention.qkv`(融合)→ `to_q`/`to_k`/`to_v`(`chunk(3, dim=0)`);
  `attention.o` → `attention.to_out.0`。其余 27 pattern 同名。`dequant_comfy_mixed`(fp8_scaled→bf16)直接复用。
  - 权重 weight-diff 坐实:转换后单文件 DiT vs 整模型 526 张量 mean rel err **2%**(纯 fp8 量化噪声);
    **to_q/k/v 不在误差头部**(头部是 adaln/final_layer 12-16%)→ qkv 切分顺序正确。
- **TE Qwen3-VL comfy → diffusers**:`model.visual.` → `visual.`;`model.` → `language_model.`;
  丢 `lm_head.weight`(整模型无)。用 `AutoModel`(非 `AutoModelForCausalLM`)→ `Qwen3VLModel`。load 0/0。
- **VAE**:`flux2-vae` = `AutoencoderKLFlux2` → 复用 `build_bridged_vae` 现成 flux2 路,零改。
- **坑**:① `Ideogram4Pipeline` `guidance_scale` 与 `guidance_schedule` **互斥**,且 schedule 必须
  长度 == `num_inference_steps`(默认 48-tuple `(7,)*45+(3,)*3`)—— 传标量须显式 `guidance_schedule=None`,
  传 schedule 须等长。② `enable_model_cpu_offload` 对 Qwen3-VL 嵌套 `embed_tokens` 有 device 错配(spike
  大卡直载绕开;产品化逐组件 hook 须处理)。

## 设计

三 PR。每个独立分支 + 真机 smoke 后合。引擎层(PR-1)是核心,loader UX(PR-2)是「破单 DiT 假设」的架构决策,
runner 管线(PR-3)收口让工作流真能跑。

### PR-1:引擎层 —— ideogram4 单文件装配(纯后端,不动 loader/runner)

把 spike 验过的转换器接进产品引擎,使 `ModularImageBackend` 能用 4 个 override 装配 `Ideogram4Pipeline`。

- **`build_bridged_transformer` 加 ideogram4 分支**:`_ref_class_name` == `Ideogram4Transformer2DModel`
  → `dequant_comfy_mixed` + DiT 键转换(qkv 三分 + o→to_out.0)+ `from_config` + `load_state_dict`。
  (z-image 走 from_single_file,flux2 走 dequant_and_convert,ideogram4 走「dequant + 自定义键转」第三路。)
- **`build_bridged_text_encoder` 接 Qwen3-VL**:检测 repo `text_encoder/config.json` 的 `model_type==qwen3_vl`
  / `_class_name==Qwen3VLModel` → 用 `AutoModel`(非 CausalLM)+ TE 前缀键转换(`model.`→`language_model.` /
  `model.visual.`→`visual.` / 丢 lm_head)。其余架构走原 `AutoModelForCausalLM` 路(零回归)。
- **`ModularImageBackend` 加第二 DiT override 槽** `unconditional_transformer_override`;
  **`_build_ideogram4_pipe` 加 override 装配分支**(对齐 `_build_zimage_pipe`):4 override 齐
  (transformer + unconditional_transformer + text_encoder + vae)→ `Ideogram4Pipeline(...)` 直接构造
  (tokenizer/scheduler 从参考库);不齐 → `from_pretrained`(零回归)。
- **guidance 互斥坑**:已在 `infer` 处理(`guidance_schedule=None` 走标量),override 路复用,不重复。
- **仓内 bundle** `backend/configs/image_arch/ideogram4/`(config + tokenizer + scheduler + 双 transformer config),
  对齐 z-image bundle,让单文件装配不依赖 58G 整模型在场。`_reference_repo_for_arch` 加 ideogram4 hint。
- **真机 smoke `smoke_ideogram4_singlefile.py`**:单文件双 DiT 装配出图 vs 整模型基线 SSIM。
  **判据**:fp8 单文件 vs bf16 整模型本就不到 0.9(spike ~0.67@12步);故基线也用 **fp8 整模型**
  (`Ideogram-4-fp8` 若补全,或对单文件再 torchao fp8)同档比,或仅验「出图连贯 + 双 DiT 都参与
  (uncond 置零 → 图崩验证非对称 CFG 真生效)」。独立进程跑基线避免显存不释放 OOM(spike 教训)。

### PR-2:Loader UX —— 表达两个 DiT(架构决策)

破「一 diffusion_models 单文件 → 一 MODEL」假设。

**ComfyUI 上游怎么做(读了用户 `【83】Ideogram4全自动流程` 工作流 JSON,feedback_read_comfyui_source)**:
**两个独立 `UNETLoader`**(各加载 `ideogram4_fp8_scaled` / `ideogram4_unconditional_fp8_scaled`,各出一个 MODEL)
+ **`DualModelGuider` 合并节点**(吃两个 MODEL + cfg 值 → 喂采样器的非对称 CFG guider)。即上游是
「两 loader + 合并节点」,**不是**单 loader 双 widget。CLIPLoader type=`ideogram4`、VAELoader=flux2-vae。

四选一,**推荐方案 D(对齐 ComfyUI)**:

- **方案 D(两 loader + 合并节点,= ComfyUI)推荐**:复用现有 `flux2_load_diffusion_model`(arch=ideogram4)
  各加载一个 DiT → 各出 MODEL;新增 `ideogram4_dual_guider` 合并节点(两 MODEL 输入 → 一 MODEL,
  spec 带 `unconditional_file` = 第二 DiT 文件)喂 KSampler。**用户已熟悉这个连法**(ComfyUI 同构),
  单 loader 不变(零回归),双 DiT 的「不对称」语义显式落在合并节点。`_ARCH_CLIP_COMPAT` 加 `ideogram4→{qwen}`。
- **方案 B(单 loader 条件第二槽)**:`flux2_load_diffusion_model` 加仅 ideogram4 显示的 `unconditional_file`
  widget。最省节点,但与 ComfyUI 连法分叉、且 loader 渐成 arch-special。
- **方案 A(专用节点)**:新 `ideogram4_load_checkpoint`,两 file widget → 一 MODEL。清晰但分叉 combo/runner。
- **方案 C(约定推断)否决**:`ideogram4_fp8_scaled` 自动找同目录 `..._unconditional_...`。命名约定脆,
  违 [[feedback_long_term_robustness]]。

方案 D 的前端:新合并节点(两 MODEL 输入端口 + 一 MODEL 输出),节点定义从 node.yaml 动态渲染;
`exec_ideogram4_dual_guider` 把第二 MODEL 的 spec.file 写进第一 MODEL 的 `unconditional_file`。
`engine_catalog._infer_arch` 加 `ideogram4_*` → ideogram4(单文件选 DiT 时 loader 下拉默认对)。
**实施 PR-2 前用 AskUserQuestion 跟用户确认 D vs B**(loader UX 是口味决策,且影响用户既有连图习惯)。

### PR-3:Runner / 组件管线 —— 第二 DiT 落地 + 卸载

让工作流真能派发双 DiT 单文件出图。

- **components 契约**:`run_published_workflow` / `exec_ksampler` / runner `_build_request` 把
  MODEL spec 的 `unconditional_file` 透传成第二个 diffusion_models ComponentSpec(`kind="diffusion_models"`,
  `role="unconditional"` 或独立键 `unconditional_transformer`)。
- **`get_or_load_image_adapter`**:`_is_standalone_single_file` 对第二 DiT 也 `build_bridged_transformer`
  → `unconditional_transformer_override`。L1 组件缓存键含第二 DiT 文件(否则错命中)。combo key 含两 DiT。
- **显存 / 卸载**:两 DiT 各 ~9.3G fp8(dequant→bf16 各 18.6G;torchao fp8 回 ~9.3G)。fp8 模式
  (`weight_dtype=fp8_e4m3`)经 `_quantize_fp8_weight_only`(名单已含 `unconditional_transformer`,#493)
  → 双 DiT 都 fp8。offload / 逐组件选卡 / lowvram 流式([[project_lowvram_streaming]])对第二 DiT 一并生效
  (`model_cpu_offload_seq` / footprint 估算把 uncond DiT 计入)。卸载链 [[project_unified_model_mgmt_completion]]
  对两 DiT 都释放。
- **真机 e2e smoke**:画布双 DiT 单文件工作流 → 出图;卸载→显存真降。

## 验证总纲(每 PR 真机,feedback_verify_real_model)

- PR-1:`smoke_ideogram4_singlefile`(装配出图 + 非对称 CFG 生效判据 + 同档基线比)。
- PR-2:节点 UI + `_check_arch_compat` 单测(ideogram4 DiT 配 qwen 放行 / 配 flux2 拒);前端 tsc/eslint/build。
- PR-3:画布全链路真机出图 + 卸载显存验证;runner components 透传单测。

## 风险 / 未决

- **判据**:fp8 单文件无法对 bf16 整模型 SSIM≥0.9(量化差),smoke 用「连贯出图 + 非对称 CFG 生效 +
  weight-diff fp8 噪声档」三判据替代纯 SSIM 闸门(spike 已立此判据)。
- **cpu offload × Qwen3-VL device 错配**:PR-3 逐组件 hook 须特判 TE 嵌套 `embed_tokens`(spike 大卡直载绕开)。
- **bundle 体积**:双 transformer config + Qwen3-VL config + tokenizer,几 MB,可接受(对齐 z-image bundle)。
- 方案 B 若实施中发现 MODEL spec 双文件透传过侵入,回退方案 A(专用节点),不改判据。

关联 [[project_unified_model_mgmt_gap]] [[project_ideogram4_integration]] [[project_lowvram_streaming]]
[[feedback_verify_real_model]] [[feedback_long_term_robustness]];spike 分支 `spike/ideogram4-singlefile`。
