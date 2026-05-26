# 图像引擎 + 任务 UX 收口 Plan — 续

> 接已合并的 #144-#149(本 session 已完成):
> - #144 真 CFG 修复(质量根因)
> - #145 架构收口 spec
> - #146 采样器/调度器两下拉
> - #147 下拉样式对齐 paperclip
> - #148 逐步进度 + 任务中止(修 HTTP cancel 桥)
> - #149 TaskPanel 对齐 ComfyUI(dock/float)
>
> 依据 [[feedback-long-term-robustness]] [[feedback-verify-real-model]] [[feedback-pr-per-change]]。
> 每个 PR 独立 branch、真模型/真浏览器验证、CI 绿后 auto-merge。

**Goal**:把图像引擎和任务 UX 完整收口 —— 引擎对齐架构 spec(#145)、UX 对齐业界专业标准
(Vercel / GitHub Actions / Replicate)、补 ComfyUI 那个 latent preview 杀手锏。

---

## PR-A:modular 退役第二刀(ERNIE 也走标准 pipeline)

**Files**:`src/services/inference/image_modular.py`(删 `_build_modular_pipe` + `_import_modular`)
+ ERNIE 用 `ErnieImagePipeline.from_pretrained` + tests。

- [ ] `image_modular`:删 `_import_modular`、删 `_build_modular_pipe`。`_ensure_pipe` 简化:
  对非 Flux2 的 pipeline_class(`ErnieImagePipeline` 等)用 lazy import 加载标准 pipeline 类
  + `from_pretrained(repo)`(没 modular 那套 components_manager 了)。
- [ ] `pyproject.toml`:`diffusers` 仍钉 commit;现在 `diffusers.modular*` 完全不引用,只剩标准 pipeline 路径。
- [ ] tests:wiring 测试改测「非 Flux2 → 标准 ErnieImagePipeline」分支(monkeypatch `_import_ernie_pipeline`)。
- [ ] **真模型验**:如有 ERNIE 模型,跑一遍 smoke;无则单元够(ERNIE 这条 PR 后续 spec 真接时再 smoke)。
- [ ] 单元 + ruff + tsc + vite build。

> **理由**:留着 modular fallback 是死代码且 experimental(blast radius)。先删干净再做后面。

---

## PR-B:`from_single_file` 自检架构 + 内置 config,砍掉 per-model 18GB 参考库依赖

**Files**:`src/services/inference/image_modular.py`(`_build_klein_pipe` 不再依赖 `Path(repo)/tokenizer`)
+ `src/services/model_manager.py`(`_modular_repo_from_components` / `_reference_repo_for_arch` 改造)+
仓内 bundle `backend/configs/image_arch/<arch>/`(tokenizer + scheduler config + transformer config + vae config 的小 json,**无权重**)。

- [ ] **内置 per-arch config**:`backend/configs/image_arch/flux2/{tokenizer,scheduler,transformer/config.json,vae/config.json}`
  ——从 `Flux2-klein-9B` 复制小 json(几 KB),`.safetensors` 不带。
- [ ] `image_modular`:全单文件路径用内置 config 目录代替 `self.repo`:tokenizer/scheduler 从 bundle 取;
  diffusers `Flux2Transformer2DModel.from_single_file` 也能传 `config=` 指向 bundle。
- [ ] `model_manager._modular_repo_from_components` 简化:全单文件 → 返回内置 config 目录;
  HF-layout 整模型 → 原参考库逻辑。
- [ ] **真模型验**:把 `image/diffusers/Flux2-klein-9B` 临时改名,确认 True-v2 单文件仍能出图
  (证明真砍掉了参考库依赖)。
- [ ] 单元 + ruff + tsc。

> **理由**:用户的核心痛点 ——「anima 这种新模型,得先放对应 18GB 整模型才能跑」。这条 PR 砍掉。

---

## PR-C:per-arch 注册表 + Qwen-Image/AuraFlow → **anima 能跑**

**Files**:新 `src/services/inference/image_arch_registry.py`(扩 / 取代 `model_arch_adapter.py`)
+ `image_modular`(按 arch dispatch 不同 pipeline class + bundle config)+ `quant_loaders`(可能需新 arch)
+ 内置 config(`backend/configs/image_arch/qwenimage/`、`auraflow/`)+ smoke。

- [ ] **arch 注册表**(对齐 #145 spec):`ImageArchSpec` 单例
  ```
  arch → { pipeline_cls, config_dir, text_encoder_cls, vae_cls,
            supported_samplers, supported_schedulers,
            default_shift, default_steps, default_cfg }
  ```
- [ ] 注册 `flux2`(已有,改造接入)、`qwenimage`、`auraflow`。
- [ ] **arch 自检**:diffusers `from_single_file` 的 `CHECKPOINT_KEY_NAMES`(flux2/auraflow/qwenimage)
  从单文件权重键自检;在 ModularImageBackend `_ensure_pipe` 早期识别 arch。
- [ ] 内置 QwenImage / AuraFlow 的 config 目录(参考 PR-B 套路)。
- [ ] **真模型验**:把 anima 单文件丢进 `diffusion_models/anima/`,**不**提供任何参考库,nous 应自检 AuraFlow
  + 出图(对照 anima 自带 ComfyUI workflow 的 cfg/negative/shift 设置)。
- [ ] 单元 + ruff + tsc。

> **anima 验收**:这条 PR 后,把 anima 模型选进 KSampler/Load 节点,Run → 出图。这是 #145 spec 的核心承诺。
> AuraFlow `ModelSamplingAuraFlow shift=3.0`(读 ComfyUI 节点源码已证 = diffusers FlowMatchEuler shift)写进
> arch 注册表的 default_shift。

---

## PR-D:offload 多目标(cpu / cuda:0/1/2)

**Files**:`image_modular.py`(infer 路径加 offload kwarg + cross-device hook)+ `base.py`(ImageRequest 加字段)
+ runner _build_request 透传 + node.yaml(`flux2_load_diffusion_model` 加 `offload` 下拉 + device 加 `cpu`)+
executor.py(flux2_model bundle 带 offload)+ smoke。

**两个 widget,语义解耦**:
- `device` ∈ [auto, cuda:0, cuda:1, cuda:2, **cpu**]:**计算所在设备**(运行 forward 的地方)。
- `offload` ∈ [**none**, cpu, cuda:0, cuda:1, cuda:2]:**权重不用时挪去哪**(空闲 stash)。

**组合语义**:
- `device=cuda:0, offload=none`:全程 cuda:0(现状)。
- `device=cuda:0, offload=cpu`:diffusers `pipe.enable_model_cpu_offload(gpu_id=0)` —— transformer/text_encoder/vae 按需
  CPU↔cuda:0 倒换。慢但塞得下(34GB 模型在 24GB 3090 也能跑)。
- `device=cuda:0, offload=cuda:1`:**跨卡 offload**(diffusers 没现成 API):权重常驻 cuda:1,每个 forward 前
  `module.to('cuda:0')` 运行后 `to('cuda:1')`。手写 hook(per-component pre_forward / post_forward)。
  用户价值:2 张 3090 协作跑大模型(34GB 在 一张 24GB 不够,但 24+24=48GB 够)。
- `device=cpu, offload=none`:纯 CPU 推理(极慢,几十秒/step,只为兜底 / 调试)。

- [x] node.yaml 显卡下拉加 `cpu` 选项 + `offload` select 下拉(`#152` PR-D)。
- [x] executor.py `exec_load_diffusion_model` 把 offload 字段塞 flux2_model bundle(`#152`)。
- [x] runner_process `_build_request` 把 offload 透传到 ImageRequest(`#152`)。
- [x] `image_modular._ensure_pipe` 接 offload(`#152` none/cpu, PR-D2 cuda:N):
  - `offload == 'none'`:`pipe.to(device)`。
  - `offload == 'cpu'`:`pipe.enable_model_cpu_offload(gpu_id=N)`。
  - `offload == 'cuda:N'`:**子类化 accelerate `CpuOffload`**(`init_hook` destination 改 stash),
    用 `UserCpuOffloadHook(model, hook)` 包给下一个 hook 的 `prev_module_hook` —— accelerate
    `CpuOffload.pre_forward` 用 `isinstance(prev, UserCpuOffloadHook)` 严格检查,传 `CpuOffload`
    子类不通过会让 offload chain 失效;链按 `pipe.model_cpu_offload_seq` 顺序绑。
    `pipe._execution_device` 通过 `_hf_hook.execution_device` 解析到 compute(不是组件实际所在的 stash),
    latents/noise 在 compute,跟 forward output 一致 → scheduler.step 不再触发跨设备相加。
- [x] **真模型验**(2026-05-26 实测 / `smoke_cross_gpu_offload.py`):
  公平对比 device=cuda:0 + offload=cpu(baseline) vs device=cuda:0 + offload=cuda:1(cross-gpu)—
  同 device → RNG 一致 → 应等价。
  | 指标 | baseline (cpu) | cross-gpu (cuda:1) |
  |---|---|---|
  | peak cuda:0(3090) | 18.1 GB | 18.1 GB ≤ 24GB ✓ |
  | peak cuda:1(Pro 6000) | 0 | 33.1 GB |
  | latency | 66s | **53s**(快 20%) |
  | SSIM vs baseline | — | **1.0000** ✓ |
- [x] 单元(routing)+ 真模型 smoke(`tests/manual/smoke_cross_gpu_offload.py`)。
- [ ] 注:与 fp8 weight-only(#139)是两条独立路径,可叠加(后续验证)。
- [ ] **2×3090 协作跑 34GB 模型** 不在 PR-D2 范围 —— stash 卡需装下所有组件常驻
  (transformer 18 + TE 16 = 34GB > 单 3090 24GB)。需 layer-level offload(单组件再切块,
  accelerate `dispatch_model` + `device_map`),独立 PR。

> **理由**:你 3 张卡(Pro 6000 + 2×3090),offload 后 3090 也能跑大模型(慢但行)。
> 跨卡 `offload=cuda:N` 让小卡借大卡当「显存银行」(Pro 6000 96GB 装权重,3090 24GB 跑算力)。
> 真实测**比 CPU offload 还快 20%**(GPU→GPU PCIe 比 GPU→CPU host RAM 快)。

---

## PR-E:UX 重构 — Topbar 全局 task chip + dropdown popover(Vercel 风)

**Files**:新 `frontend/src/components/layout/TaskMenuButton.tsx`(顶栏入口 + popover)+ `Topbar.tsx`
(插入 chip)+ `TaskPanel.tsx`(改成「查看全部」入口打开的厚详情页,不再 IconRail 直点)+ IconRail 调整。

- [ ] **3 层渐进披露**(对齐 Vercel deployments / GitHub Actions runs / Linear inbox):
  - 顶栏 chip:`○ N running` / `○ idle`;状态点 + 计数 + 旋转图标(running 时)。
  - 点开 popover:最近 5-10 个任务,每项 = 缩略图 + 名 + 状态点 + 耗时 + 行内 cancel(running) + 「查看全部 →」。
  - 「查看全部」→ 大 drawer(PR-5 的 TaskPanel,作为厚详情页)或 `/tasks` 全屏页(待定)。
- [ ] popover 用现成 `NodeSelectPopover` 模式(PR-4 那套 paperclip 风格)。
- [ ] localStorage 持久「最后打开模式」(popover / drawer)。
- [ ] **真浏览器验**:截图三态(关闭 / popover 打开 / drawer 全详情)。
- [ ] tsc + vitest + vite build。

> **理由**:用户原话「上方应该有个全局 task,展示总任务数,点开下拉才是这个 dock 抽屉详情」。
> 这正是业界主流(progressive disclosure)。当前 IconRail 直点弹 460px modal 抽屉打断流。

---

## PR-F:TAESD latent live preview(ComfyUI 杀手锏)

**Files**:`backend/models/vae_approx/taef1.safetensors`(~5MB,新 ship)+ `src/services/inference/latent_preview.py`
(新,TAESD 解码 + JPEG 编码)+ `image_modular.callback_on_step_end`(接 latents → preview)+ protocol
(`NodeProgress.preview_url?: str`)+ 前端 `DeclarativeNode`(渲染节点上叠的 preview thumbnail)。

- [ ] **TAESD 权重**(Flux2 兼容:`taef1` for Flux family):放 `backend/models/vae_approx/`,
  load 一次常驻 GPU(~5MB),`decode(x0) → small JPEG`(~96px,quality=70)。
- [ ] `image_modular.callback_on_step_end`:除现有 progress + cancel 外,加 latents → TAESD → base64 JPEG;
  **节流**:对齐 ComfyUI 的 100ms + 0.5% AND 门,但 preview 帧绕过节流(首/末步 + 第 N 步必发)。
- [ ] protocol P.NodeProgress 加 `preview_url: str | None`(data URI)。
- [ ] 前端 `DeclarativeNode` 监听 node_progress 的 preview_url → 节点上叠 96x96 缩略图(覆盖在 imageStage 区)。
- [ ] **真模型验**:浏览器 Run → 节点上看到图慢慢长出来(像 ComfyUI 那种"哇")。WS 带宽 < 200KB/s 不卡顿。
- [ ] 单元 + tsc + vitest + 真机。

> **理由**:用户的「ComfyUI 进度条很好」核心就是 live preview。数字 % 是开发者向,preview 是给眼睛看的杀手锏。

---

## PR-G:真模型端到端 PR-3 验证(留下来的尾巴)

**Files**:`backend/tests/manual/smoke_progress_cancel_e2e.py`(新)+ memory 更新。

- [ ] **smoke 经生产路径** get_or_load_image_adapter → infer:
  (a) 提供 progress_callback,验证每步被调用(SSIM 之外的契约级断言)。
  (b) 中途置 cancel_flag,验证下一步 raise + 整次 infer 在 <500ms 内退出。
  (c) cancel 后再 infer,验证 flag 复位 + 正常出图。
- [ ] 记录到 memory:PR-3 真模型已端到端验证。
- [ ] 不开 PR,直接 master 合;manual smoke 不进 CI。

> **理由**:#148 单元全过 + 真模型路径来自 PR-1 已验,但 cancel/progress 这俩契约**没经真模型端到端过一遍**。
> 用户启动 backend 试用之前补上这个 smoke,免得有 bug 等到生产现场抓。

---

## 实施顺序(我的建议,你可调)

```
PR-A(modular 退役)  ─┐
                    ├→ PR-B(from_single_file + 内置 config)
                    │     └→ PR-C(per-arch 注册表 + anima)  ←─ 用户核心价值
                    │           └→ PR-D(offload 落小卡)
                    │
                    └→ PR-E(UX 重构 Topbar chip + dropdown)  ←─ 独立,可并行
                          └→ PR-F(TAESD latent preview)       ←─ 依赖 E 的进度通路完整

                    PR-G(真模型 PR-3 e2e smoke)               ←─ 任何时候,独立
```

**关键路径**:A → B → C(让 anima 能跑,3 个 PR)。这是 #145 spec 的核心承诺。
**独立轨道**:E → F(UX + 视觉杀手锏,2 个 PR)。
**收尾**:D(offload,小)+ G(smoke,小)。

## 不做 / future

- 不移植 ComfyUI 的 k-diffusion 采样栈(继续用 diffusers FlowMatch);PR-2(#146)已定调。
- 多 backend 实例 → Redis pub-sub 跨进程 WS;单 backend 不需要(spec scope 外)。
- ERNIE 真模型 smoke 留 PR-A 之后单独跑(没真模型就先靠单元)。
- 多卡张量并行图像生成(transformer 跨卡分片)——offload 比这简单,先做 D。

---

**验证矩阵**(每 PR 都要过的最小集):

| | ruff | tsc | vitest | vite build | 真模型 smoke |
|---|---|---|---|---|---|
| PR-A | ✓ | — | — | — | ERNIE(如有)/ 单元 |
| PR-B | ✓ | — | — | — | True-v2 无参考库 |
| PR-C | ✓ | — | — | — | **anima 出图** |
| PR-D | ✓ | — | — | — | True-v2 在 3090 |
| PR-E | — | ✓ | ✓ | ✓ | 浏览器三态截图 |
| PR-F | ✓ | ✓ | ✓ | ✓ | 浏览器看图长出来 |
| PR-G | ✓ | — | — | — | (本身就是 smoke) |
