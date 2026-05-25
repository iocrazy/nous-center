# 图像采样控制 + 逐步进度(对齐 ComfyUI)— 设计

> 状态:**草稿,待 review + push 后写 plan**。
> 接单文件装配(#142)。依据 [[feedback-long-term-robustness]] [[feedback-verify-real-model]]。
> 用户诉求:nous 出图(1)无逐步进度(只节点级,任务无感知);(2)质量/控制不如 ComfyUI
> (cfg 被忽略、无采样器/调度器/negative 选择)。

## 现状(代码事实)

- `image_modular.infer` 只给 modular pipe 传 `prompt / num_inference_steps / width / height / generator`:
  **不传 `guidance_scale`(cfg)、不传 negative、无采样器/调度器选择、无逐步回调**
  (代码注释:"cfg 映射留 PR-3")。
- `ImageRequest` 有 `cfg_scale`(默认 7.0)/ `steps` 但 cfg 没用;KSampler 节点有 cfg/seed/steps/宽高 widget。
- 进度:`workflow_executor` 发**节点级** node_start/complete(KSampler+VAEDecode 是一个 dispatch 节点 →
  整段采样只一个"节点运行中")。无逐步 %。

## ComfyUI 参考(读 /home/heygo/sites/ComfyUI)

**进度**:
- `comfy/utils.py:1229 ProgressBar.update_absolute(value, total, preview)` → 经
  `set_progress_bar_global_hook`(1221)注册的全局 hook → server 发 WS `progress` 消息(throttle 0.5%)。
- `comfy/samplers.py:749`:采样器每步 `k_callback = lambda x: callback(x["i"], x["denoised"], x["x"], total_steps)`
  → callback 调 `pbar.update_absolute(i+1, total, preview)`。即**采样器逐步回调 → 进度条 → WS**。

**采样控制节点**(`comfy_extras/nodes_custom_sampler.py` + `comfy/samplers.py`):
- `KSamplerSelect`(sampler_name ∈ `SAMPLER_NAMES`=euler/dpmpp_2m/...)、`BasicScheduler`(scheduler ∈
  `SCHEDULER_NAMES`)→ sigmas、`CFGGuider`(model+正/负条件+cfg)、`RandomNoise`(seed)、
  `SamplerCustomAdvanced`(NOISE+GUIDER+SAMPLER+SIGMAS+LATENT → 出图)。**ComfyUI 是 k-diffusion 采样栈。**

## 关键适配:nous 是 diffusers,不移植 k-sampler

- **采样器/调度器**:ComfyUI 的 k-sampler/scheduler **不能直接用**。nous 对应 = **选 diffusers scheduler**
  (FlowMatchEulerDiscreteScheduler / DPMSolverMultistep / Euler / ...),运行时**换 pipe 的 scheduler 组件**。
- **cfg**:Flux2 modular pipe 有 `guidance_scale` InputParam(默认 4.0,distilled guidance)+ negative(true-CFG)。
  → infer 传 `guidance_scale=req.cfg_scale` + `negative_prompt=req.negative_prompt`。
- **逐步进度**:modular Flux2 denoise loop 用 `self.progress_bar(total)` tqdm + 每步 `update()`,
  **无 `callback_on_step_end`**。→ **覆盖 pipe 的 `progress_bar`**(自定义 CM,每步 update 调注入的回调)
  得到 (step, total) → 经现有 channel WS 发逐步事件。

## 设计

### A. 采样控制 + 质量
1. `image_modular.infer`:传 `guidance_scale=req.cfg_scale` + `negative_prompt`(pipe 支持的话;Flux2Klein
   distilled 用 guidance,negative 走 true-cfg)。**先单独验:传 cfg vs 不传对 Flux2-klein 出图的影响**(distilled
   可能 cfg 默认 4.0 最佳;真模型对比)。
2. **scheduler 选择**:KSampler(或新 Sampler 节点)加 `scheduler` 下拉(diffusers scheduler 白名单)→
   runner 透传 → `_ensure_pipe` 后 `pipe.scheduler = <chosen>.from_config(pipe.scheduler.config)`。
3. KSampler 节点补:negative 已在 Encode Prompt;cfg/steps 生效;scheduler 新增。

### B. 逐步进度(ComfyUI 式任务感知)
1. `ModularImageBackend.infer(on_step=...)`:覆盖 `pipe.progress_bar`(或 set 一个 wrapper)→ 每步算
   `pct = (i+1)/total` → `await on_step(pct, i+1, total)`(节流,类似 ComfyUI 0.5%)。
2. 运行链:`runner_process` 出图任务把 on_step 接到 `RunnerClient` → 主进程 → 新 WS 事件
   `denoise_progress {node_id, percent, step, total}` 走 channel(复用 #141 的 progress WS)。
3. 前端:dispatch 节点(VAE Decode)/ ImageOutputNode 收 `denoise_progress` → 显示进度条 + %
   (类似 DeclarativeNode 的 node_stream);TaskPanel 泳道也更新。

## PR 拆分(plan 细化)

- **PR-1**:采样参数 —— infer 传 cfg/negative;真模型验 cfg 对质量的影响(对比 ComfyUI 同参)。
- **PR-2**:scheduler 选择(节点下拉 + runner 透传 + pipe scheduler 换)。
- **PR-3**:逐步进度(progress_bar 覆盖 → WS denoise_progress → 前端进度条)。

## 不做
- 不移植 ComfyUI k-diffusion 采样栈(nous 用 diffusers scheduler)。
- 不做 latent 预览图(ComfyUI 的 preview;先做数值进度,预览留 future)。
