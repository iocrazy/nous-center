# Image 细粒度图收敛 — 单一 ComfyUI 风格组件多卡图

**Status**: Draft (rev 1)
**Author**: heygo
**Date**: 2026-05-21
**Supersedes**:
- `2026-05-19-image-component-multi-gpu-design.md`(rev 2)的 **Family B 路线**(`image_generate` 全家桶 + `image_unet_load` / `image_clip_load` / `image_vae_load` / `image_lora_apply`)。该 spec 的**后端 infra**(§5 ComponentSpec / DiffusersImageBackend / Quant loaders / Runner Protocol / ModelManager 复合 key / ImageSampler、§4.6 component_scanner、§6 四态加载状态、L1/L2 缓存)**全部保留并复用**;被取代的只是「由哪些节点驱动这套 infra」+「老 workflow 零改动」目标。
- 具体被取代的小节:§3.2 节点拓扑、§4.1–4.5 Node Schema、§7.4「老 workflow 兼容(后端 inline 展开)」、§8 PR 拆分。
**Depends on**: 2026-05-19 spec 已落地的 PR-1..6(component_scanner / ImageSampler / 组件 L1 / 四态 / L2 输出缓存,均已 merge)。

## 0. 这版要解决什么

用户在 workflow 编辑器里看到的「Load Diffusion Model」节点(来自 `flux2-components` 插件)只有一个「模型」下拉,**没有 ComfyUI `UNETLoaderMultiGPU` 那样的 `weight_dtype`(精度)和 `device`(显卡)选择**;Load CLIP / Load LoRA 同样没有选卡能力。

排查发现仓库里**并存两套图像 loader**:

| | Family A(`flux2-components` 插件) | Family B(内置组件 loader) |
|---|---|---|
| 图形态 | ComfyUI 细粒度图(Load Diffusion / CLIP / VAE → Encode → KSampler → VAE Decode) | 全家桶 `image_generate` + `image_*_load` 喂入 |
| 多卡 | ❌ 只存 `model_key`,整管线单卡 | ✅ file + device + dtype(2026-05-19 spec 的产物) |
| 端口 | MODEL/CLIP/VAE/CONDITIONING/LATENT(大写) | unet/clip/vae(小写) |
| 缓存/四态 | ❌ 走 `get_loaded_adapter`(整管线) | ✅ 组件 L1/L2 + 四态 header |

两套都出现在「图像」节点库分组里,功能重叠、互不通。给 Family A 的节点单独加 device/dtype 控件是**安慰剂**——它的 executor 走整管线单卡路径,加了也不生效。

**决策(已与用户确认)**:收敛到**单一细粒度图**(Family A 的拓扑,ComfyUI 风格),把它升级成**组件级多卡**(复用 Family B 已建好的 infra),然后**删除 Family B**。

## 1. 目标 / 非目标 / 成功标准

### 目标
- `flux2-components` 的三个 loader(Load Diffusion Model / Load CLIP / Load VAE)各自能选 **文件 + 显卡(device)+ 精度(weight_dtype)**,对齐 ComfyUI `*LoaderMultiGPU`。transformer / text_encoder / VAE 可落在不同卡(如 VAE 0.4GB 单独丢 cuda:2)。
- Load LoRA **跟随上游 MODEL 的卡**(不单独选卡,跨卡 merge 会崩),多个 Load LoRA **串联**叠在同一 transformer 上。
- Load CLIP 支持**动态多 CLIP**:用户可**自己增删** CLIP 条目(每条 = 文件 + 显卡 + 精度),外加一个 `type`(架构)选择器决定多个编码器如何合并。
- 三个 loader 拿到 **`component_select` 组件下拉 + 四态加载 header**(loaded / cold / loading / failed),复用 2026-05-19 spec §6 的状态机。
- 细粒度图的采样三节点(Encode / KSampler / VAE Decode)走**组件 L1 缓存 + 自写 ImageSampler 跨卡**,而非 `get_loaded_adapter(整管线·单卡)`。
- **删除 Family B**(`image_generate` + `image_*_load`),节点库「图像」分组只剩一套。

### 非目标(本 spec 不实施)
- **多编码器执行**:磁盘上现有图像模型都是单文本编码器(Flux2-Klein = 1 个 Qwen3;ERNIE-Image = 1 个 text_encoder 且走整管线),没有 flux.1(CLIP-L+T5-XXL)/ sdxl(CLIP-L+CLIP-G)这类多编码器模型的散件可供验证。因此**「同时跑 2+ 编码器」不在本 spec 实施范围**:本 spec 只做多 CLIP 的 **UI(增删 + type)+ bundle(list+type)+ 通用合并框架**,单编码器(flux2/qwen)路径用真模型验证;多编码器执行路径**明确 gated**(运行到就报清晰错误),等有真模型 + 对应 backend 时点亮(见 §9)。
- 老 workflow 兼容/迁移:**直接删除**旧节点,不做后端 inline 展开,也不做前端自动迁移。DB 里 3 个用到旧节点的存档(1 个 published `新工作流`、2 个 draft `wikeeyang-test (无 LoRA)` / `image-e2e-test`,均为用户自己的测试,**无线上服务依赖**——2 个 active service_instance 是 LLM/推理,非图像)打开后显示「未知节点」,用户手动用新图重连。
- GGUF 量化执行、跨进程持久化缓存、batch>1、采样中间插自定义节点 —— 沿用 2026-05-19 spec 的非目标。

### 成功标准
- Load Diffusion Model + Load CLIP + Load VAE 三节点各自能选 file/device/dtype,UI 显示四态;细粒度图三组件分卡(transformer→cuda:1, clip→cuda:0, vae→cuda:2)走通出图,SSIM 同架构 > 0.99 / 跨架构 ≈ 0.98(沿用 2026-05-19 §2 修正)。
- Load LoRA 串联两条,strength 各自生效;LoRA 落在上游 transformer 同卡。
- Load CLIP 能在前端**点「+」加行、点删除减行**(UI 可验);**单条 flux2(Qwen3)配置端到端出图**(真模型验证);**两条配置保存正常,但执行时报清晰 gated 错误**(不是崩溃、不是静默错图)。
- 节点库「图像」分组**只剩一套**细粒度节点;`image_generate` / `image_*_load` 在前后端都不存在。
- 后端测试 + 前端 `tsc` + `vite build` 全绿(每个 PR 独立绿)。

## 2. 收敛后的节点集

全部来自 `backend/nodes/flux2-components/`(升级),端口沿用大写 ComfyUI 约定。

| 节点 | 输入 | 输出 | 控件(收敛后) |
|---|---|---|---|
| `flux2_load_diffusion_model` Load Diffusion Model | — | MODEL | **file**(component_select role=unet) · **device** · **weight_dtype** · adapter_arch |
| `flux2_load_clip` Load CLIP | — | CLIP | **clip_stack**(多行: file/device/weight_dtype,增删) · **type**(架构) |
| `flux2_load_vae` Load VAE | — | VAE | **file**(component_select role=vae) · **device** · **weight_dtype** |
| `flux2_load_lora` Load LoRA | MODEL | MODEL | lora(lora_select) · strength(跟随上游 MODEL 卡,可串联) |
| `flux2_load_checkpoint` Load Checkpoint(保留便捷) | — | MODEL+CLIP+VAE | model_key(单合并 spec,三件同卡) |
| `flux2_encode_prompt` Encode Prompt | CLIP + text | CONDITIONING | text · negative_prompt |
| `flux2_ksampler` KSampler | MODEL + CONDITIONING | LATENT | width/height/steps/cfg_scale/seed |
| `flux2_vae_decode` VAE Decode | VAE + LATENT | IMAGE(`image`) | url_ttl_seconds |

> Load Checkpoint 保留作单卡便捷入口(一个合并 spec 三件同卡),不做跨组件分卡——分卡用三个独立 loader。本 spec 不给它加 device/dtype(三件本就同卡)。**但收敛后它必须产 ComponentSpec 形态的 bundle**(否则与新采样路径的两种 bundle 格式打架):需要一个 `model_key → 三个组件文件` 的 resolver(查模型 yaml / component_scanner,三件落同一 auto device)。resolver 并入 PR-1;若 resolver 比预期复杂,降级方案是本轮一并删掉 Load Checkpoint(它只是便捷节点,用户的核心诉求是三个独立 loader)。

## 3. Node Schema 改动

### 3.1 Load Diffusion Model(`flux2_load_diffusion_model`)
`node.yaml` widgets 从单 `model_key` 改为:
```yaml
widgets:
  - { name: file,        label: "文件",     widget: component_select, role: unet }
  - { name: device,      label: "显卡",     widget: select, options: [auto, cuda:0, cuda:1, cuda:2, cpu], default: auto }
  - { name: weight_dtype,label: "精度",     widget: select, options: [default, bfloat16, float16, fp8_e4m3], default: default }
  - { name: adapter_arch,label: "架构",     widget: select, options: [flux2, flux1], default: flux2 }
```
- `weight_dtype` 的 `default` = 用文件原生精度(对齐 ComfyUI 的 `default` 项)。
- 节点声明 `componentRole: unet`(供前端四态 header + component_select 过滤)。

### 3.2 Load CLIP(`flux2_load_clip`)— 动态多 CLIP
- 新 widget `clip_stack`:一个**可增删的条目列表**,每条 `{ file, device, weight_dtype }`;前端渲染「+ 添加 CLIP」+ 每行删除按钮(类比现有 `lora_stack`)。
- 节点级 `type` 选择器(架构):`flux2`(默认,单 Qwen3)/ `flux1`(CLIP-L+T5-XXL)/ `sdxl`(CLIP-L+CLIP-G)/ `sd3` / `qwen` …(与 ComfyUI `DualCLIPLoaderMultiGPU` 的 type 对齐,可扩展)。
- 默认 `clip_stack` 含一条(Flux2 的 Qwen3),用户可加第二、第三条。
```yaml
widgets:
  - { name: clips, label: "CLIP", widget: clip_stack }   # [{file, device, weight_dtype}, ...]
  - { name: type,  label: "架构", widget: select, options: [flux2, flux1, sdxl, sd3, qwen], default: flux2 }
```
- 声明 `componentRole: clip`。

### 3.3 Load VAE(`flux2_load_vae`)
```yaml
widgets:
  - { name: file,        label: "文件", widget: component_select, role: vae }
  - { name: device,      label: "显卡", widget: select, options: [auto, cuda:0, cuda:1, cuda:2, cpu], default: auto }
  - { name: weight_dtype,label: "精度", widget: select, options: [default, bfloat16, float16], default: default }
```
- 声明 `componentRole: vae`。

### 3.4 Load LoRA(`flux2_load_lora`)
- 保持现状:`lora`(lora_select)+ `strength`(slider)。**不加 device**——LoRA 跟随上游 MODEL 的卡;跨卡 merge 会 `Expected all tensors on same device` 崩。
- 串联语义已在 `executor.exec_load_lora` 实现(append 到上游 bundle 的 `loras` 列表),保留。

### 3.5 Encode / KSampler / VAE Decode
控件不变。executor 改走组件路径(§4)。

## 4. 后端改造

### 4.1 Loader bundle 携带 ComponentSpec
现状 `executor.py` 的 bundle 只存 `model_id`(yaml key)。改为携带 **2026-05-19 spec §5.1 的 `ComponentSpec` 字段**(file / device / dtype / loras / adapter_arch / clip_arch),loader 仍是 cheap(只 stash 描述符,不 load 权重):

```python
# Load Diffusion Model
{"_type": "flux2_model", "spec": {"kind": "unet", "file": <abs>, "device": ..., "dtype": ..., "adapter_arch": ...}, "loras": []}

# Load CLIP(动态多 CLIP)
{"_type": "flux2_clip", "type": "flux2", "encoders": [{"kind": "clip", "file": <abs>, "device": ..., "dtype": ...}, ...]}

# Load VAE
{"_type": "flux2_vae", "spec": {"kind": "vae", "file": <abs>, "device": ..., "dtype": ...}}
```
- `file` 由 component_scanner 的 `GET /api/v1/components?role=` 提供的 `abs_path`(2026-05-19 §4.6 已落地)。
- `device: auto` 沿用 ModelManager.get_best_gpu 选卡逻辑。
- `dtype: default` = 文件原生精度。

### 4.2 采样三节点走组件缓存 + ImageSampler
当前 `exec_encode_prompt` / `exec_ksampler` / `exec_vae_decode` 走 `_acquire_adapter(model_id)`(整管线单卡)。改为按各自 bundle 的 ComponentSpec **经组件 L1 缓存 `get_or_load_image_adapter` 装组件**,调用 ImageSampler 的**分解子步骤**,跨组件边界显式 `.to()`:

- **Encode Prompt**:按 CLIP bundle 装 text_encoder 组件 → encode 同一 prompt → 产 CONDITIONING(embeds)。单编码器(flux2/qwen,`len(encoders)==1`)走已验证路径;多编码器见 §4.3。
- **KSampler**:按 MODEL bundle 装 transformer 组件,应用 `loras`(`set_active_loras`),把 CONDITIONING 的 embeds `.to(transformer.device)`,跑 ImageSampler denoise 循环 → LATENT。
- **VAE Decode**:按 VAE bundle 装 vae 组件,把 LATENT `.to(vae.device)`,decode → IMAGE(签名 URL,schema 不变)。

> **实施细化留给 writing-plans**:ImageSampler(2026-05-19 §5.6)目前是 `encode→denoise→decode` 单体(为 `image_generate` 设计)。细粒度图三节点各驱动一个阶段,需把这三步**暴露为可独立调用的子步骤**;组件装配在哪一层(DiffusersImageBackend 组件态 adapter vs 节点直接调)由 PR-1 的 plan 定。
> **model_id 一致性检查**:现状 KSampler / VAEDecode 校验 `model["model_id"] == cond["model_id"]`。收敛后 MODEL / CONDITIONING / VAE 是独立组件,无共享 `model_id`;改为**架构/维度兼容性校验**(embeds 维度对得上 transformer),或放宽并文档化风险。

### 4.3 多 CLIP 合并框架(执行 gated)
`exec_encode_prompt` 按 CLIP bundle 的 `type` + `encoders`:
- `len(encoders)==1` 且 `type in {flux2, qwen}`:**已验证单编码器路径**(装一个 text_encoder 组件,encode)。
- `len(encoders)>1` 或 `type in {flux1, sdxl, sd3}`:进入**通用合并框架**——逐编码器装组件 + encode,按 type 规则合并 embeds(concat / pooled+seq)。但**当前无对应多编码器模型 + backend**,执行到此**抛清晰错误**:
  ```
  RuntimeError: 多编码器架构 '<type>'(<n> 个编码器)执行未就绪 —
  需要对应多编码器模型 backend(见 spec 2026-05-21 §9)。
  当前可用:flux2 / qwen 单编码器。
  ```
- 合并框架的**接口 + 分发**现在就建好(`merge_conditionings(type, per_encoder_embeds)`),只是多编码器分支 gated,等真模型点亮。

## 5. 前端改造

### 5.1 component_select + 四态接到 flux2 loaders
现状 `loadPluginDefinitions()`(`nodeRegistry.ts`)注册插件节点时**不透传** `role` / `componentRole`,所以 flux2 loader 拿不到 component_select 下拉和四态 header。改动:
- 后端 `/api/v1/nodes/definitions` 在 widget 上输出 `role`、在节点上输出 `componentRole`(node.yaml 已声明则透传)。
- `loadPluginDefinitions()` 把 `def.componentRole` 拷进 `DECLARATIVE_NODES[type]`,把每个 widget 的 `role` 拷进 `WidgetDef`。
- 这样 flux2 三 loader 自动获得 2026-05-19 §6 的四态 header + component_select 行为(无需为插件节点写专用前端)。

### 5.2 `clip_stack` 控件(新)
- 新增 `WidgetType: 'clip_stack'`,渲染一个可增删的条目表(每行 file/device/weight_dtype 三个子控件 + 删除按钮)+ 底部「+ 添加 CLIP」。file 子控件复用 component_select(role=clip)。
- 数据形态:`data.clips = [{file, device, weight_dtype}, ...]`,直接喂给后端 bundle 的 `encoders`。

### 5.3 PortType 收敛
- `PortType` union 去掉 Family B 专用的小写 `unet`/`clip`/`vae`(仅 Family B 用;细粒度图用大写端口);保留 `text`/`image`/`audio`/`data`/`any` 等。
- 细粒度图用 node.yaml 的大写端口(MODEL/CLIP/VAE/CONDITIONING/LATENT/IMAGE);确认前端端口配色对这些大写类型有映射(flux2 节点已能渲染,补齐配色即可)。

### 5.4 删除 Family B(前端)
- `workflow.ts`:删 `NODE_DEFS` 里 `image_generate` / `image_unet_load` / `image_clip_load` / `image_vae_load` / `image_lora_apply`;从 `BuiltinNodeType` union 移除。
- `nodeRegistry.ts`:删 `DECLARATIVE_NODES` 里上述 5 个;`NODE_CATEGORIES` 的「图像」分组移除 `image_generate`(只留 `image_output` + 让 flux2 插件节点 merge 进来)。
- `NodeLibraryPanel.tsx`:删 `image_loading`「组件加载」分类。

## 6. 删除清单(Family B,直接删)

| 层 | 删除项 |
|---|---|
| 后端 executor | `image_generate` 及其 `image_unet_load`/`image_clip_load`/`image_vae_load`/`image_lora_apply` 注册与实现 |
| 后端 schema | 相关 NodeDef / 端口 unet 类型(若仅 Family B 用) |
| 前端 `workflow.ts` | `NODE_DEFS` 5 节点 + `BuiltinNodeType` 条目 + `PortType` 的小写 `unet`/`clip`/`vae` |
| 前端 `nodeRegistry.ts` | `DECLARATIVE_NODES` 5 节点 + 「图像」分组 `image_generate` |
| 前端 `NodeLibraryPanel.tsx` | `image_loading` 分类 |

> 保留:component_scanner、ComponentSpec、DiffusersImageBackend、ImageSampler、组件 L1/L2 缓存、四态事件——这些是 infra,由 flux2 节点接管驱动。
> 旧存档:3 个 workflow 打开显示未知节点,用户手动重连(已确认接受)。

## 7. PR 拆分

每个 PR 独立分支、走 CI/CD、绿后 auto-merge。

### PR-1 后端:细粒度图组件化 + 跨卡 assembly
- `node.yaml`:Load Diffusion / VAE 加 file/device/weight_dtype/arch 控件;声明 componentRole。
- `executor.py`:loader bundle 携带 ComponentSpec;Encode/KSampler/VAEDecode 改走 `get_or_load_image_adapter` 组件缓存 + ImageSampler 子步骤,跨卡 `.to()`;放宽/改写 model_id 一致性校验。
- ImageSampler 暴露 encode/denoise/decode 子步骤(若需要)。
- **真模型 smoke**:transformer/clip/vae 分卡出图(SSIM 验证),LoRA 串联生效。

### PR-2 前端:component_select + 四态 + PortType 收敛
- `/api/v1/nodes/definitions` 透传 role/componentRole;`loadPluginDefinitions()` 拷贝之。
- flux2 三 loader 显示 component_select + 四态 header。
- PortType union 收敛;大写端口配色补齐。
- `tsc` + `vite build` 绿。

### PR-3 动态多 CLIP
- 前端 `clip_stack` 控件 + Load CLIP 的 `type` 选择器。
- 后端 Load CLIP bundle(`type` + `encoders` list);`exec_encode_prompt` 单编码器路径走通,多编码器框架 gated(清晰报错)。
- **真模型 smoke**:单条 flux2(Qwen3)出图;两条配置保存正常 + 执行报 gated 错误。

### PR-4 删除 Family B + 文档
- 按 §6 删除清单删前后端 Family B。
- 节点库「图像」分组只剩一套。
- 更新 2026-05-19 spec 的 Status 标注「Family B 已被 2026-05-21 收敛取代」。

## 8. Test Plan

- **PR-1**:单测 loader bundle 含 ComponentSpec;单测采样节点装组件路径(mock 组件缓存);真模型 standalone smoke(分卡出图 + SSIM + LoRA 串联)。沿用 2026-05-19 §9 的 standalone 真模型测法(不进 pytest 默认套件)。
- **PR-2**:前端组件渲染快照 / 交互测;`tsc` + `vite build`;手动核 flux2 loader 出现 file/device/dtype + 四态。
- **PR-3**:前端 clip_stack 增删交互测;后端单测 encode 单/多编码器分发(多编码器断言抛 gated 错误);真模型单条出图 smoke。
- **PR-4**:确认前后端无 `image_generate` / `image_*_load` 残留(grep);节点库只剩一套;现有后端套件全绿。
- 全程遵守 `feedback_verify_real_model`:核心假设(组件分卡出图、LoRA 串联、单 CLIP 路径)用真模型验,不靠 stub 充数。

## 9. Future Work

- **多编码器执行点亮**:当磁盘上有 flux.1(CLIP-L+T5-XXL)/ sdxl(CLIP-L+CLIP-G)等多编码器模型 + 散件时,实现对应 backend 的 `merge_conditionings` 分支 + 真模型验证,把 §4.3 的 gated 分支打开。届时 Load CLIP 的多条配置即可端到端运行。
- 前端编辑器侧老 workflow 自动迁移(本 spec 选择直接删,未来若有大量历史图可加迁移器)。
- 沿用 2026-05-19 §10 的 GGUF / 持久化缓存 / DisTorch 等 Future。
