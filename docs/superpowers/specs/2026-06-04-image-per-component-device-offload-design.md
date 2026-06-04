# Image 逐组件选卡 + 逐组件 Offload — 点亮 2026-05-21 §9 休眠项

**Status**: Draft
**Author**: heygo (设计 by Claude)
**Date**: 2026-06-04

**关联 / 取代**:
- **点亮** `2026-05-21-image-granular-convergence-design.md` §9「逐组件跨卡点亮」休眠项。该 spec rev2 因「image 组只有一张卡 + fp8 可单卡装下」放弃逐组件跨卡,把 cross-device 能力转休眠。本 spec 在用户明确诉求(「我可以自选〔卡〕诶～同时加类似 offload」)下重新点亮。
- **架构漂移修正**:2026-05-21 §0.14 假设「保留休眠的 cross-device `ImageSampler`」。但 Modular Diffusers 迁移(#128-132,`2026-05-22-image-engine-modular-diffusers-design.md`)**删了自写 `ImageSampler`/`image_sampler.py`/`image_diffusers.py`**,引擎只剩 `ModularImageBackend`(`src/services/inference/image_modular.py`)+ `image_anima.py`。所以休眠的 cross-device 路径**已不存在**;本 spec 在 `ModularImageBackend` 上**新建**逐组件放置,复用其已有的 `_enable_cross_gpu_offload`(跨卡 stash/compute 钩子)作积木。

## 0. 诉求与现状

### 用户诉求
1. 除 Load Diffusion Model 外,**Load CLIP / Load VAE(及 Load Checkpoint)节点也要能各自分配显卡**(自由选 cuda:0/1/2),不再强制跟随 transformer 卡。
2. 这些节点也要有**类似 Diffusion Model 的 Offload 旋钮**(cpu / 跨卡 stash),确保「工作流正常运行」(大组件 / 小卡场景不 OOM)。
3. (连带)Load CLIP / Load VAE 的**「已加载」四态在节点上正确显示** —— 现在恒显「未加载」。

### 现状(确认自源码)
- **强制同卡**:`src/runner/runner_process.py:351-362` 在 `flux2_vae_decode` dispatch 摊平 ImageRequest 时,`device = unet_spec["device"]`,随后 `clip_spec["device"] = device`、`vae_spec["device"] = device` —— **clip/vae 的 device 被强制覆盖成 unet 的卡**。
- **节点缺控件**:`backend/nodes/flux2-components/node.yaml` 里 `flux2_load_clip` / `flux2_load_vae` 没有 `device` / `offload` widget(只有 Load Diffusion Model 有)。
- **四态恒「未加载」根因**:前端 `ComponentStatusHeader`(`DeclarativeNode.tsx:257`)用 `componentStateKey({file, device: data.device||'auto', dtype})` 算 state-key。VAE/CLIP 节点无 device → key 恒为 `…|auto|…`;但后端在上述强制同卡后,以 `…|cuda:1|…` 注册/广播组件状态(`component_spec.py:component_state_key`)→ **两边 key 永不相等 → VAE/CLIP 永远 cold(未加载)**。给 VAE/CLIP 加上真实的 device 控件后,两边用同一 device 串,四态自然对上。
- **引擎单卡**:`ModularImageBackend._ensure_pipe()`(`image_modular.py:356`)`pipe.to(self.device)` 把 transformer+text_encoder+vae **整体**搬到一张卡。`offload` 字段当前是**整管线级**(`none` / `cpu` enable_model_cpu_offload / `cuda:N` 跨卡 stash via `_enable_cross_gpu_offload`),非逐组件。
- **build_bridged_* 已具备落任意卡的能力**:`build_bridged_transformer/text_encoder/vae(spec, repo, device)` 各自 `.to(device)`,只是当前都传同一个 `self.device`。

## 1. 目标 / 非目标 / 成功标准

### 目标
- **逐组件 device**:Load CLIP / Load VAE / Load Checkpoint 节点各加 `device`(auto/cuda:0/1/2)widget;`auto` = 跟随 transformer 卡(保持旧默认行为,零回归);显式选卡 = 该组件落到指定卡。
- **逐组件 offload**:同三节点各加 `offload`(none / cpu / cuda:N stash)widget,语义对齐 Load Diffusion Model 的 Offload。
- **引擎逐组件放置**:`ModularImageBackend` 支持 text_encoder / vae 落到与 transformer **不同**的卡,并正确协调跨卡张量流(prompt embeds: TE 卡 → transformer 卡;latents: transformer 卡 → VAE 卡)。
- **四态修复**:CLIP/VAE 节点四态 header 正确显示 已加载/加载中/未加载/失败。
- **零回归**:全 `auto` 的既有工作流行为 = 改动前(整模型单卡),SSIM 1.0 对既有 golden。

### 非目标
- **一卡一 runner / 并发多卡跑多图**:仍单 image runner 串行(沿用 2026-05-21 非目标)。本 spec 只做「一个图内组件跨卡」,不做「多图并发跨卡」。
- **多编码器执行**:clip_stack 多条仍 gated(沿用)。逐组件 device 作用于 CLIP 节点整体(该节点所有 encoder 同卡)。
- **张量并行切单组件跨多卡**:单个 transformer 横跨两张卡(模型并行)不在本 spec;本 spec 是「不同组件放不同卡」,非「一个组件切两卡」。横跨两卡的大 transformer 仍靠 offload(cpu / cuda:N stash)解决。
- **LLM 卡保护重做**:沿用 2026-05-21 §4.4 的前置显存检查。

### 成功标准
- Load CLIP / VAE / Checkpoint 显示 device + offload 控件;四态正确。
- 真机 smoke(用户 3 卡:Pro6000=cuda:1 96G,2×3090=cuda:0/cuda:2):
  1. **全 auto**(回归):Flux2 整模型单卡出图,SSIM 1.0 对既有 golden。
  2. **跨卡放置**:transformer→cuda:1(Pro6000),text_encoder→cuda:0,vae→cuda:2,出图正确(与单卡同 prompt/seed 视觉一致,SSIM≥0.97)。
  3. **逐组件 offload**:某组件 offload=cpu 或 cuda:N stash,出图正确不 OOM。
- 后端测试 + 前端 `tsc`/`vite build` 全绿(每 PR 独立绿)。

## 2. 数据流设计

### 2.1 描述符携带 per-component device/offload
- `exec_load_clip`(`executor.py:85`)产出的 clip 描述符,每个 encoder 带 `device` + `offload`(节点级,套用到所有 encoder)。
- `exec_load_vae`(`executor.py:99`)产出的 vae 描述符 spec 带 `device` + `offload`。
- `exec_load_checkpoint`(`executor.py`)三组件可同卡(现状)或保留单 device(便捷节点,本 spec 给它一个 device 即可,三件同卡)。
- `ComponentSpec`(`component_spec.py`)**已有 `device` 字段**;**新增 `offload` 字段**(默认 `none`),并入 `component_state_key`? —— **不并入**:state-key 当前是 `file|device|dtype|loras`,offload 不改变「加载了哪个权重」的身份(它是放置策略),纳入会让缓存/四态 key 漂移。offload 仅作放置参数透传,不进 cache key。(若日后 offload 改变常驻显存占用需区分,再议。)

### 2.2 runner 不再强制同卡
`runner_process.py:351-362` 改为:
```python
unet_device = unet_spec["device"]            # transformer 卡(整张图的 compute 锚点)
def _resolve(dev):                            # auto → 跟随 transformer 卡(零回归)
    return unet_device if (dev in (None, "auto")) else dev
clip_spec["device"] = _resolve(clip_spec.get("device"))
vae_spec["device"]  = _resolve(vae_spec.get("device"))
# offload 各自透传(none/cpu/cuda:N)
```
- 关键:`auto` 解析成 transformer 卡,但**写回 spec 的是解析后的具体卡**?还是保留 `auto`?
  - **保留 `auto` 直到引擎层**:这样组件 state-key 用 `auto`,前端节点(device=auto)四态对得上;引擎 `_ensure_pipe` 内把 `auto` 解析成 transformer 卡再 `.to()`。显式选卡(cuda:1)则两边都用 cuda:1。→ **四态在 auto 与显式两种情形都对上**。L1 cache key 用 `auto` 在「同一工作流 transformer 卡固定」下不歧义;跨工作流 transformer 卡不同导致 `auto` 歧义属边缘场景(组件复用 spec §,接受;或在 to_component_key 解析,后续再优化)。

### 2.3 引擎逐组件放置(ModularImageBackend)
当前 `_ensure_pipe`:`pipe = _build_klein_pipe(); pipe.to(self.device)`(整体单卡)。
改为:
- `_build_klein_pipe` 用各 override 的 `device` 分别 build(`build_bridged_transformer(...,unet_dev)`、`build_bridged_text_encoder(...,te_dev)`、`build_bridged_vae(...,vae_dev)`)。
- **不再** `pipe.to(self.device)` 整体搬;改逐组件 `.to(各自 device)`(build 时已落卡,这里确认/补)。
- `self.device`(= transformer/compute 卡)仍是 `_execution_device`:diffusers `Flux2KleinPipeline.__call__` 在 transformer 卡上跑 denoise;
  - **prompt embeds 跨卡**:text_encoder 在 te_dev 出 embeds → 需 `.to(transformer_dev)` 再进 denoise。Flux2 pipeline 在 `encode_prompt` 后用 `_execution_device`;若 TE 在别的卡,需在 backend 包一层:encode 后把 `prompt_embeds`/`pooled` 搬到 transformer 卡。(`infer()` 里已手动 `encode_prompt(..., device=self.device)` —— 见 `image_modular.py:610`;改为在 te_dev encode 再搬。)
  - **latents 跨卡**:denoise 在 transformer 卡产 latents → VAE decode 前 `.to(vae_dev)`。Flux2 标准 pipe 把 vae decode 内嵌在 `pipe()` 末尾,假设 vae 同卡。若 vae 在别卡,需:(a) 用 `output_type="latent"` 让 pipe 只出 latent,backend 手动 `latents.to(vae_dev)` 再 `vae.decode`;或 (b) 给 vae 挂 execution-device 钩子。**选 (a)**:更可控、与逐步进度/preview 解耦清晰。
- **逐组件 offload**:每组件按自身 offload 应用:
  - `none`:常驻自己的 device。
  - `cpu`:该组件 `enable_model_cpu_offload` 等价(用时上卡,闲时回 CPU)。对单组件用 accelerate 的 `cpu_offload(module, execution_device=该组件 device)`。
  - `cuda:N`:复用 `_enable_cross_gpu_offload` 的 `_GpuStashOffload` 思路,但作用于单组件(常驻 N 卡,forward 时上该组件 compute 卡)。把现有整管线版重构成可对单 module 应用。

> 复杂度集中在 `image_modular.py`:跨卡 embeds/latents 搬运 + 逐组件 offload 钩子。这是本 spec 的核心风险点(diffusers 跨设备协调脆弱 —— 正是 2026-05-21 当初放弃的原因)。靠真机 smoke 兜底。

## 3. Node Schema 改动(`node.yaml`)

### 3.1 Load CLIP(`flux2_load_clip`)
```yaml
componentRole: clip          # 新增 → 节点级四态 header(现在没有,只有 per-row 点)
widgets:
  - { name: clips,  label: "CLIP", widget: clip_stack }
  - { name: device, label: "显卡", widget: select, options: [auto, "cuda:0", "cuda:1", "cuda:2"], default: auto }
  - { name: offload, label: "Offload", widget: select, default: none, options: [...同 Diffusion Model...] }
  - { name: type,   label: "架构", widget: select, options: [flux2, flux1, sdxl, sd3, qwen], default: flux2 }
```
- device/offload 节点级,套用到该节点所有 encoder(本 spec 非目标:逐 encoder 不同卡)。
- `componentRole: clip` 让节点显示四态 header。但 CLIP 是多 encoder —— header 取「主 encoder(第一条)」的状态,或聚合(全 loaded 才 loaded)。**取聚合**:全部 encoder loaded → loaded;任一 loading → loading;任一 failed → failed;否则 cold。per-row 点保留(细粒度)。

### 3.2 Load VAE(`flux2_load_vae`)
```yaml
componentRole: vae           # 已有
widgets:
  - { name: file,         label: "文件", widget: component_select, role: vae }
  - { name: weight_dtype, label: "精度", widget: select, options: [default, bfloat16, float16], default: bfloat16 }
  - { name: device,  label: "显卡", widget: select, options: [auto, "cuda:0", "cuda:1", "cuda:2"], default: auto }
  - { name: offload, label: "Offload", widget: select, default: none, options: [...] }
```

### 3.3 Load Checkpoint(`flux2_load_checkpoint`)
- 已有 device;补 offload(三件同卡同 offload,便捷节点)。

## 4. 前端改造

### 4.1 四态 key 修复(核心)
- `ComponentStatusHeader` 已用 `data.device`(若节点有 device widget 就用选的卡,否则 auto)。给 VAE/CLIP 加 device widget 后,**key 自动用真实 device**,与后端对上 → 四态修复。无需改 `componentStateKey` 逻辑本身。
- CLIP 节点:新增节点级聚合四态 header(`componentRole: clip`)。聚合逻辑:对 `clips[]` 每条算 key、取 state、聚合。新增 `ClipAggregateStatusHeader` 组件(或扩展 `ComponentStatusHeader` 支持多 key)。per-row `ClipStateDot` 保留。

### 4.2 device/offload 控件
- 走现有 `select` widget(node.yaml 声明即自动渲染,`WidgetRenderer` 已支持 select + 对象 options/description)。无需新 widget 类型。

## 5. PR 拆分(每个独立分支 + CI + 绿后 auto-merge)

### PR-A 后端引擎:ModularImageBackend 逐组件放置 + 逐组件 offload
- `ComponentSpec` 加 `offload` 字段(不进 state-key)。
- `image_modular.py`:`_build_klein_pipe` 逐组件落卡;`infer()` encode 在 te_dev 后搬 embeds 到 transformer 卡;`output_type=latent` + 手动 `latents.to(vae_dev)` + vae.decode;逐组件 offload 钩子(重构 `_enable_cross_gpu_offload` 为可单 module)。
- `runner_process.py` 不再强制同卡(auto→transformer 卡解析,保留 auto 串)。
- **单测**:mock 组件,断言逐组件 device 透传 + auto 解析;断言 offload 透传。
- **真机 smoke**(`tests/manual/`):全 auto 回归 SSIM 1.0;跨卡放置出图 SSIM≥0.97;逐组件 offload 出图。(遵守 CLAUDE.md:`CUDA_DEVICE_ORDER=PCI_BUS_ID` 前置;改 image_modular 前后必跑 smoke_image_ab。)

### PR-B 节点 schema + executor 透传 per-component device/offload
- `node.yaml`:CLIP/VAE/Checkpoint 加 device/offload;CLIP 加 componentRole。
- `executor.py`:`exec_load_clip`/`exec_load_vae`/`exec_load_checkpoint` 描述符带 device/offload。
- 单测:executor 产出带新字段。

### PR-C 前端:四态修复 + CLIP 聚合 header
- VAE/CLIP 四态对上(加 device widget 后自然修复,补 CLIP 聚合 header)。
- `tsc` + `vite build` 绿;真机 chrome-devtools 核四态实显「已加载」。

> PR-A/PR-B 有顺序耦合(B 的描述符喂 A 的引擎),但 A 可先用单测 + smoke 自带固定 spec 验证;B 落地后端到端通。C 依赖 B(node.yaml 的 device widget)。建议落序 B → A → C,或 A 与 B 合并为一个后端 PR(描述符 + 引擎一起,end-to-end 真机验)。**采用 A+B 合并为单后端 PR**(逐组件放置是 end-to-end 能力,拆开各自无法真机验),C 独立前端 PR。

## 6. Test Plan
- 后端单测:auto 解析、逐组件 device/offload 透传、state-key 不含 offload。
- 真机 standalone smoke(必须,CI 跑不了真模型):
  1. 全 auto 回归(SSIM 1.0 对 0522 golden)。
  2. 三组件分三卡出图(SSIM≥0.97 对单卡同 seed)。
  3. 逐组件 offload(cpu / cuda:N stash)出图不 OOM。
- 前端:四态实显(chrome-devtools 真机巡检,对照 `project_realmachine_ui_audit` 方法论 —— 跨进程可见性是复发坑)。

## 7. 风险
- **diffusers 跨设备协调脆弱**(2026-05-21 放弃的原因):embeds/latents 手动搬运若漏一处 → device mismatch 崩或静默错图。靠真机 smoke 三场景兜底,且 auto 路径零改动保回归。
- **逐组件 offload 钩子**:`_enable_cross_gpu_offload` 现是整管线;重构为单 module 要保证 `_execution_device` / latents 分配仍正确(注释 `image_modular.py:203-266` 已记录该坑)。
- **CLIP 多 encoder + device**:本 spec 限定节点级单卡(所有 encoder 同卡),避免逐 encoder 跨卡的组合爆炸。
- **state-key auto 歧义**:跨工作流 transformer 落不同卡时 auto 组件 cache 可能误命中。本 spec 接受(组件复用本就同 spec 才命中;auto 在单工作流内确定)。
