# Image Component Multi-GPU Loader (ComfyUI-style)

**Status**: Draft
**Author**: heygo
**Date**: 2026-05-19
**Supersedes**: 部分(`DiffusersImageBackend` 整套装一张卡的策略)
**Depends on**: V1.5 Lane K(runner subprocess wiring,PR #107)、CUDA_DEVICE_ORDER 修复(PR #111)

## 1. 背景与问题

V1.5 Lane G 重写后,`DiffusersImageBackend` 把 Flux2 三组件(transformer 17GB + Qwen3 text_encoder 6GB + VAE 0.4GB)整套装一张卡。在 Pro 6000 96GB + 双 3090 24GB 的三卡布局下,这导致:

1. **Pro 6000 闲置**:Flux2 默认走 image runner (V1.5 hardware.yaml 把 image group 钉在 3090),Pro 6000 长期 0 占用
2. **无法跨卡 fine-grained 拆分**:即便手动调,也只能让"模型 A 整套在 GPU 1、模型 B 整套在 GPU 0",无法让单个模型的不同组件落不同卡
3. **A/B 调试浪费**:改 LoRA strength 就得重 load 整个 17GB transformer,实际不变的 text_encoder/VAE 也跟着重来
4. **ComfyUI 量化生态利用率低**:磁盘上躺着 6 个 Flux2 量化文件(fp8mixed/mxfp8mixed/nvfp4mixed/4 个 GGUF)总共 ~38GB,代码只跑通了 fp8mixed,其余浪费

ComfyUI 的 `UNETLoaderMultiGPU` / `CLIPLoaderMultiGPU` / `VAELoaderMultiGPU` 三节点解决了上述问题——每组件独立选 device + 文件,且节点级缓存让 A/B 调参近乎免费。

## 2. 目标 / 非目标

### 目标
- 用户在 workflow 编辑器里**独立**选择 transformer / text_encoder / VAE 三组件的文件 + device
- 同样独立地链式叠加 LoRA(对应 ComfyUI 的「加载 LoRA」节点)
- 每个 loader 节点的 UI 反馈**真实加载状态**(loaded / cold / loading / failed)
- 确定性 seed 的 `image_generate` 二次跑命中缓存秒过
- ComfyUI 量化生态可用:safetensors 全格式(bf16 / fp16 / fp8mixed / mxfp8mixed / nvfp4mixed)
- 老 workflow(单 model_key `image_generate`)**零改动可跑**(后端 inline 展开)
- Loader 节点对所有 diffusers Pipeline 通用(Flux2 / Z-Image / SDXL / Qwen-Image-Edit 等共享 loader)

### 非目标(本 spec 不实施)
- GGUF 量化处理(V2 PR-6,见 § 10.1)
- 跨 backend 重启的持久化缓存(V2 PR-8,见 § 10.3)
- 老 workflow 在**前端编辑器**侧自动展开成新节点(后端 inline 展开兜底已够;前端展开 V2 加)
- `image_edit` / `video_generate` 等针对其他模型架构的 generate 节点(本 spec 只做 `image_generate` for text-to-image 通用路径)
- ComfyUI DisTorch 风格的「单组件按 byte/ratio 拆到多 device」(Pro 6000 96GB 装 Flux2 全套无压力,后续模型变大再回头加)

### 成功标准
- Flux2-bf16 三组件分卡(unet→cuda:1, clip→cuda:0, vae→cuda:2)走通,出图 ≤ 35s
- Flux2-fp8mixed 单卡 Pro 6000 出图 ≤ 25s
- 同 seed image_generate 二跑 ≤ 0.1s(L2 cache 命中)
- 改 LoRA strength 二跑 < 5s(L2 cache miss + 仅 lora_apply 重 patch)
- master 上现有两个老 workflow(309542918354374656 / 308084173191516160)零改动跑通

## 3. 架构

### 3.1 设计取舍:我们 vs ComfyUI

| 维度 | ComfyUI | nous-center(本 spec 后) |
|---|---|---|
| Pipeline 容器 | 自写 `comfy/ldm/` + 自写 sampler | HF diffusers Pipeline 类(Flux2Pipeline / SDXLPipeline 等) |
| 组件 loader | 自写 UNETLoader 等 | **diffusers 组件 `.from_single_file` + 自写 quant dequant** |
| 量化兼容 | 自家 `comfy_quant` metadata + city96 GGUF | 兼容 `comfy_quant`(已有 fp8mixed)+ 本 spec 加 mxfp8/nvfp4 + V2 加 GGUF |
| 多 GPU 拆组件 | `ComfyUI-MultiGPU` custom_nodes | **原生**(本 spec) |
| 跨节点缓存 | 单进程 IS_CHANGED hash,重启丢 | runner 进程内 L1(loader)+ L2(image_generate output),重启丢 |
| 新模型支持 | 等社区移植(常常几周) | diffusers 一发 Pipeline 即可,Day 1 |
| 推理流程自由度 | 极高(每 step 可插节点) | 低(Pipeline 黑盒)|

**关键判断**:我们押 diffusers 维护速度,代价是少一些纸面自由度,换来"广覆盖 + 低维护";不阉割 ComfyUI 量化生态(它产出的 fp8mixed/GGUF 文件我们要能用)。

### 3.2 节点拓扑

```
image_unet_load (Flux2-fp8mixed, cuda:1) ──┐
            │                              │
            ▼                              │
image_lora_apply (style-xl, 0.8)         ──┤
            │                              │
            ▼                              ├──→ image_generate ──→ image_output
image_lora_apply (detail, 0.4)           ──┤    (steps/cfg/seed/...)
                                           │
image_clip_load (Qwen3-fp8, cuda:0)      ──┤
                                           │
image_vae_load (flux2-vae, cuda:2)       ──┘
                                           │
text_input (prompt)                      ──┘
```

- 每个 loader 节点输出**纯描述符 dict**(`{kind, file, device, dtype, ...}`),不传张量 handle(跨子进程不可能)
- `image_lora_apply` 接 unet 描述符 → 在 loras 列表 append 一条 → 输出新 unet 描述符
- `image_generate` 接 3 个组件 + prompt → runner 子进程拿到所有描述符,组装 Flux2Pipeline,各组件 `.to(device)`,跑 sampler

### 3.3 加载与缓存模型

- **L1**:runner ModelManager 按 `(file, device, lora_set)` 复合 key 缓存已加载组件;同样的 unet+lora 组合在不同 workflow / 同 workflow 反复跑均命中
- **L2**:runner 内 image_generate output cache(LRU 50 条),hash 输入(全部组件描述符 + prompt + 采样参数 + seed)。`is_deterministic=True`(seed 非空)才参与;随机 seed 不缓存
- L1/L2 均为 in-memory;backend 重启丢

## 4. Node Schema

### 4.1 `image_unet_load`
```yaml
form:
  file:    enum(component_index['unet'])     # 来自 component_scanner
  device:  enum["auto","cpu","cuda:0","cuda:1","cuda:2"]
  dtype:   enum["bfloat16","float16","fp8_e4m3"]
  adapter_arch: enum["flux2","flux1"]        # LoRA arch 校验用
output_ports:
  unet: dict {kind:"unet", file, device, dtype, adapter_arch, loras:[]}
```

### 4.2 `image_clip_load`
```yaml
form:
  file:    enum(component_index['clip'])
  device:  enum
  dtype:   enum["bfloat16","fp8_e4m3"]
  tokenizer_type: enum["flux2","flux1","sdxl","qwen"]
output_ports:
  clip: dict {kind:"clip", file, device, dtype, tokenizer_type}
```

### 4.3 `image_vae_load`
```yaml
form:
  file:    enum(component_index['vae'])
  device:  enum
  dtype:   enum["bfloat16","float16"]
output_ports:
  vae: dict {kind:"vae", file, device, dtype}
```

### 4.4 `image_lora_apply`(可链式叠多个)
```yaml
input_ports:
  unet:    required  ← from image_unet_load 或上一个 image_lora_apply
form:
  lora_file:  enum(component_index['loras'])
  strength:   float [0..2], default 1.0
  bypass:     bool, default false
  adapter_arch_hint: enum["flux2","flux1","sdxl","auto"]
output_ports:
  unet:    dict {kind:"unet", ...上游所有 + (本节点 file, strength)}
```

`bypass=true` → 直接透传 input,不 append。

### 4.5 改造 `image_generate`
```yaml
input_ports:
  unet:   required ← image_unet_load 或 image_lora_apply
  clip:   required ← image_clip_load
  vae:    required ← image_vae_load
  prompt: text     ← text_input
  negative_prompt: text (optional)
form:
  steps:    int 25, [1..200]
  width:    int 1024, [64..4096]
  height:   int 1024, [64..4096]
  cfg_scale: float 7.0, [0..30]
  seed:     int (空=随机,影响 is_deterministic 标志)
  url_ttl_seconds: int 3600
output_ports:
  image:    dict {image_url, image_uuid, image_expires, width, height, media_type}
  meta:     dict {steps, seed, loras, ...}
```

**校验**:任一组件端口未连 → 节点头部红色「● 缺少 vae 输入」,workflow 提交时阻止发送。

### 4.6 文件来源:`component_scanner`

新增配置 `backend/configs/model_paths.yaml`:
```yaml
base_path: ${LOCAL_MODELS_PATH}   # 默认 /media/heygo/Program/models/nous

roles:
  unet:
    - image/diffusion_models/
    - image/diffusers/*/transformer/
  clip:
    - image/text_encoders/
    - image/diffusers/*/text_encoder/
  vae:
    - image/vae/
    - image/diffusers/*/vae/
  loras:
    - image/loras/
```

`backend/src/services/component_scanner.py`(新建):
- 启动时 glob 上述模式,产 `{role: [{filename, abs_path, size_mb, quant_type, mtime}]}` 缓存进 `app.state.component_index`
- `quant_type` 探测:扫文件名(`fp8mixed` / `mxfp8mixed` / `nvfp4mixed` 子串) + 必要时打开 safetensors header 检查 `comfy_quant` metadata。GGUF 也识别(标记为 `gguf`,V1 dropdown 显示但选中报"V2 支持")
- `GET /api/v1/components?role=unet` 返回该 role 列表
- `POST /api/v1/components/scan` 手动 rescan(admin only)
- WS `component_index_changed` 广播

## 5. Adapter / Runner 改造

### 5.1 `ComponentSpec`(新)

```python
# src/services/inference/base.py
class ComponentSpec(BaseModel):
    kind: Literal["unet", "clip", "vae"]
    file: str                  # 绝对路径(scanner 已 resolve)
    device: str                # "cuda:0" / "cuda:1" / "cuda:2" / "auto"
    dtype: str                 # "bfloat16" / "float16" / "fp8_e4m3"
    loras: list[LoRASpec] = [] # 仅 kind=unet
    adapter_arch: str | None = None    # 仅 unet:"flux2"/"flux1"
    tokenizer_type: str | None = None  # 仅 clip
```

### 5.2 `DiffusersImageBackend` 重写

```python
class DiffusersImageBackend(InferenceAdapter):
    modality = MediaModality.IMAGE
    
    def __init__(self, components: dict[str, ComponentSpec], **kwargs):
        # components = {"unet": ..., "clip": ..., "vae": ...}
        self._components = components
        self._pipe: Flux2Pipeline | None = None
    
    async def load(self) -> None:
        # 1. quant loader 注册表分发(by extension + metadata)
        transformer = QUANT_LOADERS.dispatch(self._components["unet"])
        text_encoder = QUANT_LOADERS.dispatch(self._components["clip"])
        vae = QUANT_LOADERS.dispatch(self._components["vae"])
        # 2. 各组件 .to(device);跨卡 Pipeline 装配
        transformer.to(self._components["unet"].device)
        text_encoder.to(self._components["clip"].device)
        vae.to(self._components["vae"].device)
        # 3. PEFT 一次性 set_active_loras(应用全 list,不分批 patch)
        if self._components["unet"].loras:
            set_active_loras(transformer, self._components["unet"].loras)
        # 4. 不调 enable_model_cpu_offload —— 各组件已落实卡
        self._pipe = Flux2Pipeline(transformer=transformer,
                                   text_encoder=text_encoder, vae=vae)
```

### 5.3 Quant Loader 注册表

```python
# src/services/inference/quant_loaders.py(新建)
QUANT_LOADERS = QuantLoaderRegistry()

# 注册顺序 = 匹配优先级,从特殊到通用(first-match-wins)。
# safetensors 多种量化共享扩展名,只能靠文件名 substring + safetensors header
# 里的 `comfy_quant` metadata 区分,所以特殊格式必须排在 plain 之前。

@QUANT_LOADERS.register(match=lambda spec: "nvfp4mixed" in Path(spec.file).name.lower())
def load_nvfp4mixed(spec: ComponentSpec):
    # 新增:4-bit nf4 dequant
    ...

@QUANT_LOADERS.register(match=lambda spec: "mxfp8mixed" in Path(spec.file).name.lower())
def load_mxfp8mixed(spec: ComponentSpec):
    # 新增:scale 格式跟 fp8mixed 不同,实现见 ComfyUI-MultiGPU distorch_2.py 参考
    ...

@QUANT_LOADERS.register(match=lambda spec: "fp8mixed" in Path(spec.file).name.lower()
                                        or _has_comfy_quant_metadata(spec.file))
def load_fp8mixed(spec: ComponentSpec):
    # 复用现有 load_quantized_transformer(image_diffusers.py:105)—— wikeeyang
    # 风格,scan safetensors header 看 `.comfy_quant` 字段确认。
    ...

@QUANT_LOADERS.register(match=lambda spec: spec.file.endswith(".safetensors"))
def load_safetensors_plain(spec: ComponentSpec):
    # 兜底:bf16/fp16 plain safetensors,走 diffusers from_single_file 原生路径。
    ...

# GGUF 注册延后到 V2 PR-6(本 spec § 10.1)。dropdown 选中 .gguf 时,scanner
# 标 quant_type="gguf",runner 端报「GGUF 暂未支持,V2 加」拒载。
```

### 5.4 Runner Protocol

`P.RunNode` 不变(`is_deterministic` 字段已存在),`inputs` dict 携带 3 个组件描述符。`runner_process._build_request`:

```python
if node.node_type == "image":
    if all(k in node.inputs for k in ("unet", "clip", "vae")):
        return ImageRequest(
            request_id=f"task-{node.task_id}",
            prompt=str(node.inputs.get("prompt", "")),
            negative_prompt=str(node.inputs.get("negative_prompt", "")),
            steps=int(node.inputs.get("steps") or 25),
            width=int(node.inputs.get("width") or 1024),
            height=int(node.inputs.get("height") or 1024),
            cfg_scale=float(node.inputs.get("cfg_scale") or 7.0),
            seed=...,
            components={
                "unet": ComponentSpec(**node.inputs["unet"]),
                "clip": ComponentSpec(**node.inputs["clip"]),
                "vae":  ComponentSpec(**node.inputs["vae"]),
            },
        )
    # 老路径:有 model_key 无 components → yaml 展开
    return ImageRequest(..., model_key=node.model_key)
```

### 5.5 ModelManager 复合 key

`ModelManager._models` 由 `dict[str, LoadedModel]`(key=model_key)改为 `dict[ComponentKey, LoadedModel]`:

```python
ComponentKey = tuple[
    str,                            # file 绝对路径
    str,                            # device,如 "cuda:1"
    frozenset[tuple[str, float]],   # lora_set,每条 (lora_file, strength) 排序后 frozenset
]
```

- 老 `model_key` 路径在 load 时翻译成等价的 ComponentKey(yaml `paths.transformer/text_encoder/vae` → 3 个 ComponentKey)
- `is_loaded(model_key)` 老 API 兼容:翻译后查询(三组件全 loaded → True)
- 新 `is_component_loaded(key: ComponentKey) -> Literal["loaded","loading","cold","failed"]` API
- `evict_lru(gpu_index)` 不变(按 entry.gpu_index 维度淘汰)

## 6. Loader 节点状态展示

### 6.1 四态视觉

| 状态 | 视觉 | 含义 |
|---|---|---|
| `loaded` | 亮色 + 绿点 + "✓ N GB" | 已加载,workflow 跑会秒过 |
| `cold` | 灰色 + 灰点 + "未加载" | 未装,首跑触发 load |
| `loading` | 黄色脉冲 + 进度环 | 正在装 |
| `failed` | 红色 + 错误文案 | 上次 load 失败,可点 retry |

### 6.2 加载触发

- **默认**:workflow 跑 → runner 发现组合不在 L1 cache → 先 load 再采样
- **辅助 prewarm**:节点 UI 右上角图标 → `POST /api/v1/models/components/preload {file, device, lora_set}` → runner 主动 load

### 6.3 前端 hook

```typescript
function useComponentState(keys: ComponentKey[]): Record<string, ComponentState>
// - 节点 mount 时 batch GET /api/v1/models/components/state?keys=...
// - WS 订阅 /ws/models 的 component_state_changed,精准更新
// - 节点 unmount 取消订阅
```

## 7. UI / 编辑器改造

### 7.1 Palette 子分类

`frontend/src/components/workflow/NodePalette.tsx`:
```
图像生成
├── image_generate           (改造:多输入端口)
├── image_output             (不变)
└── 组件加载                 ← 新建子分类
    ├── image_unet_load
    ├── image_clip_load
    ├── image_vae_load
    └── image_lora_apply
```

子分类用 `NodeCategory.subcategory: string`,跟现有「语言模型→llm_chat / llm_completion」一致。

### 7.2 Loader 节点表单

参考 ComfyUI `UNETLoaderMultiGPU`:
```
┌─ image_unet_load ──────────────────┐
│ ● loaded · 17.0GB · cuda:1         │  ← useComponentState
├────────────────────────────────────┤
│ Transformer 文件                   │
│  [Flux2-Klein-9B-bf16.safe… ▼]    │  ← dropdown(component_index['unet'])
│ Device      [cuda:1 ▼]             │
│ Dtype       [bfloat16 ▼]           │
│ Adapter arch[flux2 ▼]              │
├────────────────────────────────────┤
│                          unet ●→  │
└────────────────────────────────────┘
```

### 7.3 改造后 image_generate 表单

```
┌─ image_generate ───────────────────┐
│ ● ready · 3/3 组件已连接           │
├────────────────────────────────────┤
│ →● unet     (来自 unet_load)       │
│ →● clip     (来自 clip_load)       │
│ →● vae      (来自 vae_load)        │
│ →● prompt   (来自 text_input)      │
│ →● negative_prompt (可选)          │
├────────────────────────────────────┤
│ Steps      [25      ]              │
│ Width [1024] Height [1024]         │
│ CFG scale  [7.0     ]              │
│ Seed       [(空=随机)]              │
│ URL TTL    [3600 s  ]              │
├────────────────────────────────────┤
│                         image ●→  │
│                         meta  ●→  │
└────────────────────────────────────┘
```

### 7.4 老 workflow 兼容

- **前端**(本 spec 不实施,V2 PR-7 加,见 § 10.2):打开老 workflow 自动展开 3 loader + N lora_apply
- **后端**(本 spec **实施**):`workflow_executor._dispatch_node` 检测老格式
  - node.type==`image_generate` && node.data.model_key && 无 unet/clip/vae 边
  - 从 yaml 读 `paths.transformer / .text_encoder / .vae`(或 `quantized_transformer`)
  - 构造 3 个 ComponentSpec(device="auto"),塞进 inputs.unet/clip/vae
  - LoRA list 转 lora_apply 链(或直接合并进 unet 描述符的 loras)
  - 之后路径与新格式一致

## 8. PR 拆分

| PR | 目标 | 估计行数 |
|---|---|---|
| **PR-1** | `DiffusersImageBackend` 接 ComponentSpec dict;quant loader 注册表(bf16/fp16/fp8mixed 复用+mxfp8mixed/nvfp4mixed 新增);老 yaml 翻译兼容 | ~500 |
| **PR-2** | `component_scanner` 服务 + `GET /api/v1/components` + `POST /scan` + WS `component_index_changed` + `backend/configs/model_paths.yaml` | ~250 |
| **PR-3** | 4 个新节点(`unet_load` / `clip_load` / `vae_load` / `lora_apply`)+ `image_generate` 改造;`runner_process._build_request` 加 components 分支;workflow_executor 老格式 inline 展开 | ~700 |
| **PR-4** | ModelManager 复合 key + `is_component_loaded` API + `/ws/models` 加 `component_state_changed`;前端 `useComponentState` hook + 4 节点 React 组件 + palette 子分类 | ~400 |
| **PR-5** | L2 image_generate output cache(LRU 50)+ `is_deterministic` 标志贯通 + WS `node_cache_hit` + TaskPanel 节点 "(cached)" 角标 | ~200 |

5 个 PR 串行依赖(PR-N 依赖 PR-N-1 全 merge),每个独立可 ship 可灰度。

## 9. Test Plan

| 层 | 覆盖 |
|---|---|
| Unit | ComponentSpec 序列化、quant dequant 每格式 fixture、component_scanner glob、is_component_loaded 状态机、L2 cache LRU |
| Integration | 4 新节点 + image_generate 全链路(fake_adapter=True),描述符流转 + L1 复合 key 缓存命中 |
| Smoke(真模型) | 1) Flux2-bf16 三组件分卡 ≤ 35s;2) Flux2-fp8mixed 单卡 Pro 6000 ≤ 25s;3) 改 LoRA strength 二跑 < 5s;4) 同 seed 二跑 < 0.1s |
| 前端 | vitest:4 节点渲染 + 端口连接 + state hook + dropdown 数据源 |
| Regression | master 上 image-e2e-test(309542918354374656)+ 新工作流(308084173191516160)零改动跑通 |

## 10. Future Work

### 10.1 V2 PR-6:GGUF 量化处理
- 引入 [city96/ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) dequant 逻辑
- 装 `gguf` Python 包
- 加 `load_gguf` 到 quant_loaders 注册表
- 启用磁盘上 4 个 Flux2-Q4/Q5/Q6/Q8 文件(~28GB 节省)

### 10.2 V2 PR-7:前端编辑器自动迁移老 workflow
- 打开老 workflow 检测 `image_generate(model_key=X)` → 弹窗"已检测到老格式,是否展开"
- 自动 insert 3 loader + N lora_apply 节点
- 自动布局到 image_generate 左侧

### 10.3 V2 PR-8:跨进程持久化中间缓存
- `~/.gstack/cache/nodes/<hash>.json` 存 cache 索引(PNG bytes 复用现有 outputs/ 目录)
- Cache key 加版本戳(diffusers_version, adapter_class_version, model_mtime+size)
- LRU bound 1000 条
- backend 重启后命中,二跑秒过

### 10.4 多模型架构 generate 节点
- `image_edit`(Qwen-Image-Edit / IP-Adapter,接 input image)
- `video_generate`(HunyuanVideo / Wan,输出帧序列)
- 共享本 spec 的 loader 节点(因为 diffusers Pipeline 同结构)

### 10.5 ComfyUI DisTorch 风格的单组件按 byte/ratio 拆 device
- 仅当未来上更大模型(70B 级 image,Pro 6000 96GB 装不下)时考虑
- 参考 ComfyUI-MultiGPU `distorch_2.py`(795 行)
