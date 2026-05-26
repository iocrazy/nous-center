# 图像采样控制 + 逐步进度 Plan(对齐 ComfyUI)

> REQUIRED SUB-SKILL: executing-plans。
> spec:`docs/superpowers/specs/2026-05-25-image-sampling-control-and-progress-design.md`。

**Goal**:nous 图像出图对齐 ComfyUI 的(1)采样控制(cfg/negative/scheduler 选择,质量可控)
(2)逐步进度(任务感知)。nous 用 diffusers,采样器=diffusers scheduler(不移植 k-diffusion)。

**Branch**:每 PR 独立。

---

## PR-1:Flux2 comfy 单文件走 true-CFG(根因修复,已真模型证伪)

> **根因翻案(2026-05-26 真模型 A/B 已证)**:上个 session「cfg/质量无关」错了。用户实际跑
> `diffusion_models/flux/Flux2-Klein-9B-True-v2`(comfy 类别,**true-CFG / 去蒸馏**),但 nous 单文件装配
> 借 `diffusers/Flux2-klein-9B`(官方蒸馏整模型,`is_distilled:true`)的 config,modular 据此走**蒸馏 block**
> (`Flux2KleinAutoBlocks` → `guidance=None`、无 negative 分支)→ cfg/negative 全被掐。spike
> `backend/tests/manual/spike_true_cfg.py`:同 prompt/seed42/25步,标准 `Flux2KleinPipeline(is_distilled=False)`
> 出图 **SSIM(cfg1, cfg3.5/5/7/4+neg)=0.68/0.64/0.60/0.64**(对比蒸馏管线 1.0000)→ cfg 经 true-cfg 巨大生效。
> ComfyUI 侧印证:`supported_models.Flux2` **无 is_distilled gate**,cfg/negative 用户控。

**修复 = comfy Flux2 单文件改走标准 `Flux2KleinPipeline(is_distilled=False)`**(我 spike 已用 nous 桥接组件
`build_bridged_*` 喂标准 pipe 真模型跑通;桥接/fp8/LoRA 全 pipe 无关、保留)。这同时是「modular 退役」的第一刀
(见架构收口 spec)。

**Files**:`image_modular.py`(`_ensure_pipe` 建标准 pipe / `infer` 接 negative)、`model_manager.py`
(透传 `pipeline_class`)、`nodes/flux2-components/node.yaml`(Encode Prompt negative 注释)、wiring 测试。

- [ ] `image_modular`:加 lazy seam `_import_klein_pipeline()`(CI monkeypatch);`ModularImageBackend` 收
  `pipeline_class`;`_ensure_pipe`:`pipeline_class=="Flux2KleinPipeline"` → 标准 `Flux2KleinPipeline`
  (单文件 override 三组件 + repo tokenizer/scheduler,`is_distilled=False`;HF-layout → `from_pretrained` 尊重 model_index)。
  非 Flux2(ERNIE)留 modular fallback。fp8/LoRA 路径不变(标准 pipe 同 API)。
- [ ] `infer`:`guidance_scale=cfg`(已);`req.negative_prompt` 非空且 cfg>1 → `negative_prompt_embeds=
  pipe.encode_prompt(neg)[0]`(标准 klein __call__ 无 negative 字符串入参,走预编码 embeds + true-cfg)。
- [ ] **真模型验**(关键):经**生产路径** `get_or_load_image_adapter → infer`,cfg=1 vs cfg=4+neg SSIM<1 + 出图更好;
  LoRA / fp8 经标准 pipe 仍正确(扩 `smoke_single_file_prod` 或新 smoke)。
- [ ] 单元(CI mock):infer 把 cfg→guidance_scale、negative→negative_prompt_embeds 传进 pipe();
  `_ensure_pipe` 建标准 pipe(monkeypatch seam)。

## PR-2:scheduler 选择(diffusers,**仅 flow-matching**)

**Files**:`nodes/flux2-components/node.yaml`(KSampler 加 scheduler 下拉)+ `executor.py` +
`runner` 透传 + `image_modular`(换 pipe.scheduler)

> **范围现实(已查证)**:diffusers 51 个 scheduler 类绝大多数是扩散模型的(epsilon/v-pred),**Flux2 是
> flow-matching,只有 3 个真正兼容**:`FlowMatchEulerDiscreteScheduler`(默认)、`FlowMatchHeunDiscreteScheduler`、
> `FlowMatchLCMScheduler`。套扩散 scheduler 公式不对会出垃圾。**不追 ComfyUI 那 40 个 k-diffusion 采样器**
> (不同引擎 + flow 模型本就采样器空间小)。

- [ ] **复刻 ComfyUI KSampler 的两个下拉**(用户要;结构对齐,选项是 diffusers 的):
  - **`sampler_name`** = scheduler **类**:`FlowMatchEulerDiscrete`(默认)/ `FlowMatchHeunDiscrete`(/ `FlowMatchLCM`)。
  - **`scheduler`** = sigma 调度 **config**:`normal`(默认)/ `karras` / `exponential` / `beta`
    (映射到 FlowMatchEuler 的 `use_karras_sigmas`/`use_exponential_sigmas`/`use_beta_sigmas` 互斥开关;已查证 Flux2
    scheduler config 支持这些 + dynamic_shifting/shift=3.0)。
  - `denoise`:img2img 强度,nous 现 txt2img → 固定 1.0,留 img2img 时接。
- [ ] node.yaml KSampler 加 `sampler_name` + `scheduler` 两下拉;descriptor 带这俩;runner `_build_request` 透传
  到 ImageRequest(加字段 sampler_name/scheduler)。
- [ ] `_ensure_pipe` 后:`cls = {FlowMatchEulerDiscrete/Heun/LCM}[sampler_name]`;
  `pipe.scheduler = cls.from_config({**pipe.scheduler.config, use_karras_sigmas/exponential/beta: ...})`。
- [ ] 真模型验:换 sampler/scheduler 出图正确 + 风格差异。单元:类映射 + sigma 开关 + 透传。

## PR-3:逐步进度(任务感知)

**Files**:`image_modular`(progress 覆盖)+ `runner_process` / protocol(denoise_progress 事件)+
`workflow_executor`(转发)+ 前端(ImageOutputNode / DeclarativeNode / TaskPanel 进度条)

> **PR-1 换标准 pipe 后简化**:标准 `Flux2KleinPipeline.__call__` 自带 `callback_on_step_end(pipe, i, t, kw)`
> —— 直接用它拿 (step, total),**不用再 hack `progress_bar`**(那是 modular 才需要的)。

- [ ] `infer(on_step=...)`:传 `callback_on_step_end` → 每步算 pct → `on_step(pct, i+1, total)`(节流 ~0.5%)。
- [ ] runner 出图任务接 on_step → RunnerClient → 主进程 → WS `denoise_progress {node_id, percent, step, total}`
  走现有 channel(复用 #141 progress WS / openProgressChannel)。
- [ ] 前端:dispatch 节点 / ImageOutputNode 监听 `node-progress` 的 denoise_progress → 进度条 + %;
  TaskPanel 泳道更新。
- [ ] 真机验:浏览器 Run 看到逐步 %(像 ComfyUI)。tsc/vitest/build。

---

## 不做 / future
- 不移植 ComfyUI k-diffusion 采样栈(用 diffusers scheduler)。
- latent 实时预览图(ComfyUI 的 preview)留 future;先做数值进度。
- RandomNoise/CFGGuider/SamplerCustomAdvanced 那种**拆分节点**不做(nous KSampler 保持 bundled +
  加 scheduler/cfg 控件即可;拆分是 ComfyUI 习惯,非必需)。
