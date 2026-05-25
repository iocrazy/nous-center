# 图像采样控制 + 逐步进度 Plan(对齐 ComfyUI)

> REQUIRED SUB-SKILL: executing-plans。
> spec:`docs/superpowers/specs/2026-05-25-image-sampling-control-and-progress-design.md`。

**Goal**:nous 图像出图对齐 ComfyUI 的(1)采样控制(cfg/negative/scheduler 选择,质量可控)
(2)逐步进度(任务感知)。nous 用 diffusers,采样器=diffusers scheduler(不移植 k-diffusion)。

**Branch**:每 PR 独立。

---

## PR-1:采样参数(cfg + negative)生效

**Files**:`src/services/inference/image_modular.py`(infer)

- [ ] infer 传 `guidance_scale=req.cfg_scale`;若 `req.negative_prompt` 非空传 `negative_prompt`
  (Flux2Klein:guidance distilled + negative 走 true-cfg;签名用 inspect 过滤不支持的 kwarg)。
- [ ] **真模型验**(关键,feedback-verify-real-model):同 prompt/seed,对比 (a) 不传 cfg(现状默认 4.0)
  (b) 传 cfg=1/4/7 + negative —— 看 Flux2-klein 质量差异,定默认值。对照 ComfyUI 同参出图。
- [ ] 单元(CI mock):infer 把 cfg_scale→guidance_scale、negative_prompt 传进 pipe()。

## PR-2:scheduler 选择(diffusers,**仅 flow-matching**)

**Files**:`nodes/flux2-components/node.yaml`(KSampler 加 scheduler 下拉)+ `executor.py` +
`runner` 透传 + `image_modular`(换 pipe.scheduler)

> **范围现实(已查证)**:diffusers 51 个 scheduler 类绝大多数是扩散模型的(epsilon/v-pred),**Flux2 是
> flow-matching,只有 3 个真正兼容**:`FlowMatchEulerDiscreteScheduler`(默认)、`FlowMatchHeunDiscreteScheduler`、
> `FlowMatchLCMScheduler`。套扩散 scheduler 公式不对会出垃圾。**不追 ComfyUI 那 40 个 k-diffusion 采样器**
> (不同引擎 + flow 模型本就采样器空间小)。

- [ ] scheduler 白名单 = flow-matching 3 个(FlowMatchEuler 默认 / Heun / LCM);按模型类型 gate(非 flow
  模型未来另列)。
- [ ] KSampler widget `scheduler`;descriptor 带 scheduler;runner _build_request 透传到 ImageRequest。
- [ ] `_ensure_pipe` 后:若指定且 != 默认,`pipe.scheduler = SchedulerCls.from_config(pipe.scheduler.config)`。
- [ ] 真模型验:换 scheduler 出图正确 + 风格差异。单元:scheduler 映射 + 透传。

## PR-3:逐步进度(任务感知)

**Files**:`image_modular`(progress 覆盖)+ `runner_process` / protocol(denoise_progress 事件)+
`workflow_executor`(转发)+ 前端(ImageOutputNode / DeclarativeNode / TaskPanel 进度条)

- [ ] `ModularImageBackend.infer(on_step=...)`:覆盖 `pipe.progress_bar` → 每步算 pct → `on_step(pct, i, total)`
  (节流 ~0.5%,类似 ComfyUI)。
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
