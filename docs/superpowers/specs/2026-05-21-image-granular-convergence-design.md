# Image 细粒度图收敛 — 单一图 · 整模型单卡 · 工作流选卡

**Status**: Draft (rev 2 — 架构改向:放弃逐组件跨卡,改整模型单卡 + 工作流选卡)
**Author**: heygo
**Date**: 2026-05-21
**Revision history**:
- rev 1(2026-05-21):收敛到细粒度图 + 逐组件跨卡(沿用 2026-05-19 的 cross-device ImageSampler)。
- **rev 2(2026-05-21,本版本)**:writing-plans 期排查发现两处硬伤 → 架构改向:
  1. flux2 细粒度图节点全是 **inline**,在**主进程**吃 GPU(`node_routing.DISPATCH_NODE_TYPES` 只有 image_generate / tts_engine),绕过 runner 子进程隔离 —— 正是 V1.5 要消灭的 GPU race。
  2. 真实 `configs/hardware.yaml` 给 **image 组只有一张卡**(3090#0);逐组件跨卡(transformer/clip/vae 分卡)会侵占 llm(Pro 6000)/tts(3090#2)的卡,打破"三组独立并发"前提 → 在本机拓扑下根本用不起来。
  决策(与用户确认):**整模型单卡**(不拆组件),**`weight_dtype`/量化是塞进单卡的核心旋钮**(fp8 让 Flux2 ~25GB→~12GB 装进一张 3090),细粒度图**编译成一次 ImageRequest 派发到所选卡的 runner**。
**Supersedes**:
- `2026-05-19-image-component-multi-gpu-design.md`(rev 2)的 **Family B 路线**(`image_generate` 全家桶 + `image_unet_load`/`clip_load`/`vae_load`/`lora_apply`)与**逐组件跨卡目标**(transformer/clip/vae 分卡 + cross-device `.to()`)。
- 保留并复用其**后端 infra**:§5.1 `ComponentSpec`、§5.3 Quant loaders、§5.4 Runner Protocol、§5.5 ModelManager 复合 key + `get_or_load_image_adapter`、§5.6 `ImageSampler`(同卡路径,跨设备 `.to()` 退化为 no-op)、§4.6 `component_scanner`、§6 四态加载状态、L1/L2 缓存。cross-device 能力**转休眠**(代码留着但不走;未来"模型量化后仍装不下单卡"才点亮)。

## 0. 这版要解决什么

用户原始诉求(对照 ComfyUI `UNETLoaderMultiGPU` 截图):图像「Load Diffusion Model」节点缺 `weight_dtype`(精度)和 `device`(显卡)。排查后确定:
- 仓库里**两套并存**的图像 loader(详见 rev 1 §0):Family A = `flux2-components` 插件的 ComfyUI 细粒度图(单卡、只存 model_key);Family B = 内置 `image_generate` + `image_*_load`(多卡、dispatch 到 runner)。
- 用户选**收敛到 Family A 的细粒度图**(ComfyUI 手感)+ 删 Family B。
- rev 2 进一步定:细粒度图不仅要补 device/dtype,还要**修正它的执行架构**(inline 主进程 → dispatch runner)+ **改成整模型单卡 + 工作流选卡**(而非逐组件跨卡)。

## 1. 目标 / 非目标 / 成功标准

### 目标
- 细粒度图(Load Diffusion / CLIP / VAE → Encode → KSampler → VAE Decode)成为**唯一**一套图像节点,删 Family B。
- **Load Diffusion Model** 补齐 ComfyUI 缺的两项:`file`(选量化文件)+ `weight_dtype`(精度)+ `device`(**这张图整体跑哪张卡**)。`weight_dtype` 是把大模型塞进单卡的核心旋钮(fp8 → Flux2 装进一张 3090)。
- **Load CLIP / Load VAE** 补 `file` + `weight_dtype`(无 device —— **跟随 Load Diffusion Model 选的卡**,整模型单卡)。
- **工作流选卡**:`device` 选 cuda:0/1/2,整模型(transformer+clip+vae+lora)全装那张卡,**换卡只改下拉、不重启**。复用现有 image runner(它已能看到所有卡)。
- **修正执行架构**:细粒度图节点不再 inline 在主进程吃 GPU;改成 inline 描述符产出 + 末端 VAE Decode **dispatch 一次 ImageRequest 到 image runner**,整模型在所选卡跑 `ImageSampler`,中间张量不跨进程。
- **Load CLIP 动态多 CLIP**:用户可**自己增删** CLIP 条目(每条 = file + weight_dtype)+ `type`(架构)选择器。UI + bundle + 合并框架现在做;**多编码器执行 gated**(无真模型可验)。
- Load LoRA 串联,跟随 transformer 同卡。
- 三个 loader 拿到 `component_select` 文件下拉 + 四态加载 header(复用 2026-05-19 §6)。

### 非目标(本 spec 不实施)
- **逐组件跨卡**(transformer/clip/vae 分卡):本机 image 组单卡 + fp8 足以单卡装下,跨卡会破坏角色组隔离。cross-device infra 留休眠,未来"量化后仍超单卡"再点亮(§9)。
- **一卡一 runner / 并发多卡跑图**:现有单 image runner(串行队列)足够"选卡 + 测调配"(测试本就串行)。并发多卡是未来项(§9)。
- **多编码器执行**(flux1/sdxl/sd3):磁盘上只有单编码器模型(Flux2=1 Qwen3,ERNIE=1 text_encoder 走整管线),无真模型可验。只做 UI + bundle + 框架,执行 gated。
- **老 workflow 兼容**:直接删旧节点(3 个旧存档手动重连,无线上服务依赖)。
- GGUF 执行、跨进程持久化缓存、batch>1、采样中间插自定义节点 —— 沿用 2026-05-19 非目标。

### 成功标准
- Load Diffusion Model 显示 file + weight_dtype + device;Load CLIP/VAE 显示 file + weight_dtype;三节点显示四态。
- 同一张细粒度图,`device` 改 cuda:0 / cuda:1 都能跑(整模型落到对应卡),**不重启**。真模型 smoke:Flux2-fp8mixed 落一张 3090 出图(不 offload);Flux2-bf16 落 Pro 6000 出图(需先给 Pro 6000 腾出显存)。
- 细粒度图执行时 GPU 工作发生在 **image runner 子进程**(非主进程),主进程不加载 GPU 模型。
- LoRA 串联两条,strength 各自生效,落 transformer 同卡。
- Load CLIP 能在前端增删 CLIP 行;单条 flux2(Qwen3)端到端出图;两条配置保存正常 + 执行报清晰 gated 错误。
- 节点库「图像」只剩一套;前后端无 `image_generate`/`image_*_load` 残留。
- 后端测试 + 前端 `tsc` + `vite build` 全绿(每 PR 独立绿)。

## 2. 架构:细粒度图「编译 → 一次派发」

### 2.1 关键判断
- **现有 image runner 已能看到所有卡**(`.env` 不设 `CUDA_VISIBLE_DEVICES`,supervisor `_spawn` 也不设;只有 LLM 引擎被隔离)。→ 整模型装哪张卡,由 ImageRequest 的 `device` 决定,**不需要改 runner 拓扑**。
- 细粒度图是**线性链**(Load* → Encode → KSampler → VAE Decode),且非目标排除了"采样中间插自定义节点"。→ 可以**懒求值**:上游节点只产"计划描述符",末端节点一次性派发执行(类似 ComfyUI 从输出节点反向拉取)。

### 2.2 节点执行分流(对齐 Family B 的 inline-loader + dispatch-terminal 模式)

```
Load Diffusion Model ─(unet 描述符: file/dtype/device/loras)─┐
Load LoRA (链式)      ─(往 unet.loras append)────────────────┤
                                                             ▼
Load CLIP ─(clip 描述符: type + encoders[])─→ Encode Prompt ─(conditioning 描述符: {clip, text, neg})─┐
                                                                                                      ▼
                                                                              KSampler ─(latent 描述符: {model, conditioning, w/h/steps/cfg/seed})─┐
                                                                                                                                                  ▼
Load VAE ─(vae 描述符)──────────────────────────────────────────────────────────────────────────→ VAE Decode  ★dispatch★
                                                                                                                  │
                                                          组装完整 ImageRequest(components{unet,clip,vae} + loras + prompt + 采样参数 + device)
                                                                                                                  ▼
                                                                                            image runner 子进程: get_or_load_image_adapter(整模型→device) → ImageSampler.sample() → image
```

- **inline 描述符产出节点**(主进程,不碰 GPU):Load Diffusion / Load CLIP / Load VAE / Load LoRA / Encode Prompt / KSampler。每个只产/累积 plain dict 描述符(嵌套上游计划),无张量。
- **dispatch 终端节点**(→ image runner):`flux2_vae_decode`。它从 vae 描述符 + latent 描述符(内含 model/conditioning/采样参数)**flatten 出完整 ImageRequest**,派发到 runner。
- runner 内复用 **Family B 已建好的执行引擎**:`get_or_load_image_adapter(components, pipeline_class)` 把三组件**全部装到 `device` 这一张卡**(cross-device `.to()` 退化 no-op)→ `ImageSampler.sample()` 端到端 encode→denoise→decode → 出图。中间张量全在 runner 内,不跨进程。
- `device`:取 unet 描述符的 device(Load Diffusion Model 上选的);clip/vae 在装配时强制用同一 device(整模型单卡)。

> 这等于:**编辑器是细粒度图(ComfyUI 手感),运行时是一次整模型派发(复用 image_generate 的 runner 路径)**。把 `image_generate` 编辑器节点删掉,但它在 runner 里的执行引擎(ImageRequest + get_or_load_image_adapter + ImageSampler)保留为细粒度图的后端。

### 2.3 device 路由与 LLM 卡保护
- 现有 dispatch 按 role 把 image 任务投到 image runner;该 runner 据 ImageRequest.device 把整模型装到那张卡。v1 保持单 image runner(串行)。
- **保护**:若 `device` 命中常驻 LLM 占用的卡(Pro 6000 当前跑 vLLM),装载前给出清晰错误/告警(显存不足前置检查),提示用户先腾出该卡。避免静默 OOM。
- `hardware.yaml` 角色变轻:仍用于起 runner,但 image 落卡以工作流 `device` 为准(role 分组从"硬钉死"变"默认/建议")。本 spec 不强求改 hardware.yaml 结构;只加 device 透传 + LLM 卡保护。

## 3. Node Schema 改动(`backend/nodes/flux2-components/node.yaml`)

### 3.1 Load Diffusion Model(`flux2_load_diffusion_model`)
```yaml
widgets:
  - { name: file,         label: "文件",   widget: component_select, role: unet }
  - { name: weight_dtype, label: "精度",   widget: select, options: [default, bfloat16, float16, fp8_e4m3], default: default }
  - { name: device,       label: "显卡",   widget: select, options: [auto, cuda:0, cuda:1, cuda:2], default: auto }
  - { name: adapter_arch, label: "架构",   widget: select, options: [flux2, flux1], default: flux2 }
```
- `device` = **整张图跑哪张卡**(clip/vae/lora 跟随)。`weight_dtype: default` = 文件原生精度。
- 节点声明 `componentRole: unet`。

### 3.2 Load CLIP(`flux2_load_clip`)— 动态多 CLIP
```yaml
widgets:
  - { name: clips, label: "CLIP", widget: clip_stack }   # [{file, weight_dtype}, ...] 可增删
  - { name: type,  label: "架构", widget: select, options: [flux2, flux1, sdxl, sd3, qwen], default: flux2 }
```
- `clip_stack`:可增删条目,每条 `{file, weight_dtype}`(**无 device**,跟随 transformer 卡)。
- `type`:多编码器合并规则;默认 flux2(单 Qwen3,默认一条)。声明 `componentRole: clip`。

### 3.3 Load VAE(`flux2_load_vae`)
```yaml
widgets:
  - { name: file,         label: "文件", widget: component_select, role: vae }
  - { name: weight_dtype, label: "精度", widget: select, options: [default, bfloat16, float16], default: default }
```
- 无 device(跟随 transformer 卡)。声明 `componentRole: vae`。

### 3.4 Load LoRA(`flux2_load_lora`)
- 保持 `lora`(lora_select)+ `strength`。无 device(跟随上游 MODEL 卡)。串联语义已在 `exec_load_lora`(append 到 `loras`),保留。

### 3.5 Load Checkpoint(`flux2_load_checkpoint`,保留便捷)
- 单合并 spec 三件同卡。收敛后须产 ComponentSpec 形态 bundle(`model_key → 三组件文件` resolver,三件同 device);并入 PR-1。若 resolver 偏复杂,降级方案=本轮删 Load Checkpoint(便捷节点,核心是三个独立 loader)。

### 3.6 Encode / KSampler / VAE Decode
- 控件不变。executor 改为**描述符产出**(Encode/KSampler)+ **dispatch 组装 ImageRequest**(VAE Decode)。

## 4. 后端改造

### 4.1 Loader / 中间节点 → 描述符(inline)
- `exec_load_diffusion_model` → `{"_type":"flux2_model","spec":{kind:unet, file, device, dtype, adapter_arch}, "loras":[]}`。
- `exec_load_clip` → `{"_type":"flux2_clip","type": <arch>, "encoders":[{kind:clip, file, dtype}, ...]}`。
- `exec_load_vae` → `{"_type":"flux2_vae","spec":{kind:vae, file, dtype}}`。
- `exec_load_lora` → append `{name, path, strength}` 到上游 unet 的 `loras`(已实现,保留;补 `path`)。
- `exec_encode_prompt` → `{"_type":"flux2_conditioning","clip": <clip bundle>, "text", "negative"}`(**不再在主进程编码张量**,只记计划)。
- `exec_ksampler` → `{"_type":"flux2_latent","model": <unet bundle>, "conditioning": <cond bundle>, "width","height","steps","cfg_scale","seed"}`(**不再在主进程采样**)。
- 全部登记为 inline(`node_routing` 默认 inline 即可,无需进 DISPATCH_NODE_TYPES)。

### 4.2 VAE Decode → dispatch ImageRequest
- `flux2_vae_decode` 进 `DISPATCH_NODE_TYPES`。它的输入(vae 描述符 + **嵌套的** latent 描述符,latent 内含 model/conditioning/采样参数)随 RunNode 投到 runner;**flatten 在 runner `_build_request` 内做**(派发节点的输入不经主进程 executor 变换),走 latent→model/conditioning 把嵌套描述符摊平成 ImageRequest:
  ```python
  ImageRequest(
    components={"unet": ComponentSpec(... device=unet.device),
                "clip": ComponentSpec(... device=unet.device),   # 跟随 transformer 卡
                "vae":  ComponentSpec(... device=unet.device)},
    prompt=cond.text, negative_prompt=cond.negative,
    width, height, steps, cfg_scale, seed,
    pipeline_class="Flux2KleinPipeline",
  )
  ```
  → 经 RunnerClient 投到 image runner;runner `get_or_load_image_adapter` 整模型装 `device` 单卡 → `ImageSampler.sample()` → 出图(schema 同 image_generate:image_url/media_type/width/height/uuid/expires)。
- **多 CLIP**:`encoders` 多条时,runner 侧 encode 走合并框架;单条 flux2/qwen 走已验证路径;多编码器分支 gated 报错(§4.3)。

### 4.3 多 CLIP 合并框架(执行 gated)
- ImageRequest 的 clip 部分携带 `type` + `encoders[]`。runner encode 阶段:
  - `len(encoders)==1` 且 `type in {flux2,qwen}`:已验证单编码器路径。
  - `len>1` 或 `type in {flux1,sdxl,sd3}`:进合并框架(逐编码器 encode + 按 type 合并),但**无对应 backend → 抛清晰错误**:`多编码器架构 '<type>'(<n> 编码器)执行未就绪,需对应多编码器模型 backend(见 §9);当前可用 flux2/qwen 单编码器`。
- 合并接口 `merge_conditionings(type, per_encoder_embeds)` 现在建好,多编码器分支 gated。

### 4.4 device 透传 + LLM 卡保护
- ImageRequest.device 已有(2026-05-19);确认从 VAE Decode 一路透传到 `get_or_load_image_adapter` 的单卡装载(整模型,非拆分)。
- 装载前若 `device` 命中常驻 LLM 卡且显存不足 → 前置清晰错误(不静默 OOM)。

## 5. 前端改造

### 5.1 component_select + 四态接到 flux2 loaders
- `/api/v1/nodes/definitions` 透传 widget 的 `role` + 节点的 `componentRole`;`loadPluginDefinitions()` 拷进 `DECLARATIVE_NODES` / `WidgetDef`(现在没拷,补上)。flux2 三 loader 自动获得 component_select + 四态 header。

### 5.2 `clip_stack` 控件(新)
- 新 `WidgetType: 'clip_stack'`:可增删条目表(每行 file(component_select role=clip)+ weight_dtype + 删除)+「+ 添加 CLIP」。数据 → `data.clips=[{file,weight_dtype},...]`。

### 5.3 device 控件
- Load Diffusion Model 的 `device` 用普通 select(auto/cuda:0/1/2);Load CLIP/VAE 无 device(UI 不显示,后端跟随 transformer)。

### 5.4 PortType 收敛 + 删 Family B 前端
- `PortType` union 去掉小写 `unet`/`clip`/`vae`(仅 Family B 用;细粒度图用大写端口 MODEL/CLIP/VAE/CONDITIONING/LATENT);确认大写端口配色。
- 删 `NODE_DEFS`/`DECLARATIVE_NODES` 的 `image_generate`/`image_unet_load`/`image_clip_load`/`image_vae_load`/`image_lora_apply` + `BuiltinNodeType` 条目 + `NodeLibraryPanel` 的 `image_loading` 分类。

## 6. 删除清单(Family B 编辑器侧,直接删)

| 层 | 删除 | 保留 |
|---|---|---|
| 后端 | `image_generate` 编辑器节点 executor + `image_*_load` 注册 | runner 的 ImageRequest / `get_or_load_image_adapter` / ImageSampler 执行引擎(细粒度图复用) |
| 前端 workflow.ts | 5 节点 + BuiltinNodeType + 小写 unet/clip/vae PortType | — |
| 前端 nodeRegistry.ts | 5 节点 DECLARATIVE + 图像组 image_generate | — |
| 前端 NodeLibraryPanel.tsx | image_loading 分类 | — |

> 旧存档(`新工作流` published / `wikeeyang-test` / `image-e2e-test`)打开显示未知节点,手动重连(已确认,无线上服务依赖)。

## 7. PR 拆分

每个 PR 独立分支、走 CI/CD、绿后 auto-merge。

### PR-1 后端:细粒度图编译 → 一次派发 + 整模型单卡
- inline 节点改产描述符(Load*/Encode/KSampler);`flux2_vae_decode` 进 DISPATCH,组装 ImageRequest;runner 走 `get_or_load_image_adapter` 整模型单卡(device 透传)+ ImageSampler;node.yaml Load Diffusion 加 file/weight_dtype/device、VAE 加 file/weight_dtype;Load Checkpoint resolver(或删);LLM 卡保护。
- **真模型 smoke**:同图换 device(cuda:0 fp8 / cuda:1 bf16)出图;LoRA 串联;确认 GPU 工作在 runner 子进程。

### PR-2 前端:component_select + 四态 + device 控件 + PortType 收敛
- 透传 role/componentRole;flux2 loaders 显示 file/weight_dtype + 四态;Load Diffusion 显示 device;PortType 收敛 + 大写端口配色。`tsc`+`vite build` 绿。

### PR-3 动态多 CLIP
- 前端 `clip_stack` + `type`;后端 clip bundle(type + encoders)+ encode 单编码器走通 + 多编码器框架 gated。真模型 smoke:单条 flux2 出图;两条保存正常 + 执行 gated 报错。

### PR-4 删除 Family B + 文档
- 按 §6 删前后端 Family B(保留 runner 执行引擎);节点库只剩一套;更新 2026-05-19 spec Status 标注被本 spec 收敛取代。

## 8. Test Plan
- PR-1:单测 inline 节点产描述符 + VAE Decode flatten ImageRequest;单测 runner 整模型单卡装载(mock 组件缓存);真模型 standalone smoke(换卡出图 + LoRA 串联 + 确认子进程执行)。沿用 2026-05-19 §9 standalone 真模型测法。
- PR-2:前端组件渲染/交互测;`tsc`+`vite build`;手动核四态 + device 下拉。
- PR-3:clip_stack 增删交互测;后端 encode 单/多编码器分发(多编码器断言 gated);真模型单条出图。
- PR-4:grep 确认无 `image_generate`/`image_*_load` 残留;节点库只剩一套;后端套件全绿。
- 全程遵守"核心假设用真模型验"(换卡出图、LoRA 串联、单 CLIP 路径、子进程执行)。

## 9. Future Work
- **一卡一 runner / 并发多卡跑图**:要同时在多张卡跑多个图像任务时,把单 image runner 拆成按卡的多 runner + 按 device 路由。
- **逐组件跨卡点亮**:当出现"量化后仍装不下任何单卡"的模型(>96GB,或 24–48GB 要横跨两张 3090)时,启用休眠的 cross-device ImageSampler + 真模型验证。
- **多编码器执行点亮**:有 flux.1/sdxl 等多编码器模型 + 散件时,实现对应 `merge_conditionings` 分支 + 真模型验证,打开 §4.3 gated。
- `hardware.yaml` 进一步轻量化(列卡 + 标记 LLM 占用);GGUF / 持久化缓存 / DisTorch 沿用 2026-05-19 §10。
