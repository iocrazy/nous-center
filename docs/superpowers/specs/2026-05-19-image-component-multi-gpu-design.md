# Image Component Multi-GPU Loader (ComfyUI-style)

**Status**: Draft (rev 2 — Task 0 risk gate triggered §5.2 fallback,B 路线变主路径)
**Author**: heygo
**Date**: 2026-05-19
**Supersedes**: 部分(`DiffusersImageBackend` 整套装一张卡的策略)
**Depends on**: V1.5 Lane K(runner subprocess wiring,PR #107)、CUDA_DEVICE_ORDER 修复(PR #111)
**Revision history**:
- rev 1(2026-05-19 初稿):假设 diffusers Pipeline.__call__ 跨 device 可行
- **rev 2(2026-05-19,本版本)**:Task 0 实测 `Flux2KleinPipeline.__call__` 在 denoise 循环里 hard-coded 同 device — 跨卡导致 RuntimeError("Expected all tensors to be on same device"). 决策:**自写 ImageSampler 取代 diffusers Pipeline.__call__**(参考 ComfyUI `samplers.py:986` outer_sample 模式 + 每组件边界显式 .to())。spec §5.2 fallback 提升为主路径。

## 1. 背景与问题

V1.5 Lane G 重写后,`DiffusersImageBackend` 把 Flux2 三组件(transformer 18GB + Qwen3 text_encoder 6GB + VAE 0.4GB,合计 ~25GB)**整套装一张卡**。PR #111 修了 CUDA_DEVICE_ORDER + ModelManager.get_best_gpu 后,单模型已经能自动落 Pro 6000,但组件级仍捆绑——三组件必须同卡。这导致 Pro 6000 + 双 3090 三卡布局下:

1. **无法跨卡 fine-grained 拆分**:flux2 整套去 Pro 6000(96GB)后,另外两张 3090 各 24GB 闲;反过来 flux2 落 3090(24GB)时又要靠 cpu_offload 保命,速度 5-6 倍慢于 Pro 6000(实测 PR #111 verified)。本来 VAE 0.4GB 完全可以推给某张 3090,但代码做不到
2. **A/B 调试浪费**:改 LoRA strength 就得重 load 整个 18GB transformer,实际不变的 text_encoder/VAE 也跟着重来。LoRA 切换一次 ~30 秒
3. **ComfyUI 量化生态利用率低**:磁盘上躺着 6 个 Flux2 量化文件(fp8mixed/mxfp8mixed/nvfp4mixed/4 个 GGUF)总共 ~38GB,代码只跑通了 fp8mixed,其余浪费
4. **同 seed 调参反复跑**:确定性参数下二跑结果跟一跑同图,但每次都真采样 30+ 秒;无任何缓存复用
5. **diffusers Pipeline.__call__ 是黑盒** (rev 2 新增):Task 0 实测发现 `Flux2KleinPipeline.__call__` 在 denoise 循环里硬假设所有组件同 device,跨 device 直接 RuntimeError。即便我们把组件 .to() 不同卡装上了,Pipeline 一调用就崩。**这意味着要实现痛点 1 的解决,必须自写采样循环**(类似 ComfyUI `samplers.py:986` 的 `outer_sample` 模式)

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
- **(rev 2 新增)真正的跨 device 推理**:transformer 在 cuda:1、text_encoder 在 cuda:0、vae 在 cuda:2 时 image_generate 能正确出图(SSIM > 0.99 vs 单 device baseline)。实现方式:自写 `ImageSampler`(参考 ComfyUI `samplers.py:986`)取代 diffusers `Pipeline.__call__`

### 非目标(本 spec 不实施)
- GGUF 量化处理(V2 PR-7,见 § 10.1 — rev 2 后 PR 编号顺延)
- 跨 backend 重启的持久化缓存(V2 PR-9,见 § 10.3)
- 老 workflow 在**前端编辑器**侧自动展开成新节点(后端 inline 展开兜底已够;前端展开 V2 加)
- `image_edit` / `video_generate` 等针对其他模型架构的 generate 节点(本 spec 只做 `image_generate` for text-to-image 通用路径)
- ComfyUI DisTorch 风格的「单组件按 byte/ratio 拆到多 device」(Pro 6000 96GB 装 Flux2 全套无压力,后续模型变大再回头加)
- **(rev 2 新增非目标)** 重写 transformer / vae / text_encoder 的 forward(diffusers 模型类自带的 forward 单 device 工作良好,本 spec 只重写采样循环外层)
- **(rev 2 新增非目标)** 批量推理(`batch_size > 1`)— ImageSampler V1 只支持 batch=1
- **(rev 2 新增非目标)** 采样中间插自定义节点(ComfyUI KSampler 节点支持,我们 workflow 粒度停在"完整 image_generate")

### 成功标准
- Flux2-bf16 三组件分卡(unet→cuda:1, clip→cuda:0, vae→cuda:2)走通,出图 ≤ 35s,**且 SSIM 跟单 device baseline > 0.99**(自写采样数学正确性)
- Flux2-fp8mixed 单卡 Pro 6000 出图 ≤ 25s
- 同 seed image_generate 二跑 ≤ 0.1s(L2 cache 命中)
- 改 LoRA strength 二跑 < 5s(L2 cache miss + 仅 lora_apply 重 patch)
- master 上现有两个老 workflow(309542918354374656 / 308084173191516160)零改动跑通
- **Cancel mid-sampler**:25 步采样到第 10 步触发 cancel,500ms 内 NodeResult status=cancelled 回到调用方

## 3. 架构

### 3.1 设计取舍:我们 vs ComfyUI (rev 2)

| 维度 | ComfyUI | nous-center(本 spec 后) |
|---|---|---|
| Model 实现(transformer / vae / text_encoder forward) | 自写 `comfy/ldm/` 整套 | **复用 diffusers 类**(Flux2Transformer2DModel / AutoencoderKLFlux2 / etc.)— 它们 ` __call__` 不涉及跨 device |
| **采样循环**(denoise + scheduler step + 跨组件 tensor 转移) | 自写 `comfy/samplers.py:986` outer_sample | **自写 `ImageSampler`**(本 spec §5.6,**参考 ComfyUI 实现**)— **不**用 diffusers `Pipeline.__call__`(Task 0 实测它跨 device 崩) |
| 组件 loader | 自写 UNETLoader 等 | diffusers `.from_pretrained` / `.from_single_file` + 自写 quant dequant |
| 量化兼容 | 自家 `comfy_quant` metadata + city96 GGUF | 兼容 `comfy_quant`(已有 fp8mixed)+ 本 spec 加 mxfp8/nvfp4 + V2 加 GGUF |
| 多 GPU 拆组件 | `ComfyUI-MultiGPU` custom_nodes(也是靠自写 sampler 跨 device .to())| **原生**(本 spec ImageSampler 在每组件边界 .to()) |
| 跨节点缓存 | 单进程 IS_CHANGED hash,重启丢 | runner 进程内 L1(loader)+ L2(image_generate output),重启丢 |
| 新模型支持 | 等社区移植 ComfyUI(常常几周) | diffusers 发模型类那刻能用 + 加 ~50 行 ModelArchAdapter 注册采样调用方式 |
| 推理流程自由度 | 极高 | 高 — 我们的 ImageSampler 跟 ComfyUI 同档,但**采样循环只有一层不分节点**(不支持中间插自定义节点) |

**关键判断**(rev 2):我们押 diffusers **作为模型权重 + 模型 forward 的源**,自己控制**采样循环 + 跨组件协调**。这是 ComfyUI 同款架构 — 区别在于 ComfyUI 把 transformer/vae 的 forward 也重写了(更深控制),我们停在采样层(避免维护 model forward)。代价是新模型架构(如未来 Flux3 出来 transformer forward 签名变了)要写一层 ModelArchAdapter,~50-100 行;不像 ComfyUI 是整套 model 重写。

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
- `image_generate` 接 3 个组件 + prompt → runner 子进程组装 `ImageSampler`(本 spec §5.6 自写),每组件按描述符的 device 调 `.to()` 装好,跑自写的采样循环。**不**走 diffusers `Flux2KleinPipeline.__call__`(它跨 device 崩,Task 0 实测验证)

### 3.2a Runner 内部数据流(rev 2 新增)

```
ImageRequest (含 3 个 ComponentSpec + prompt + steps + ...)
    ↓
ImageSampler.sample()
    ├── encode_prompt(text_encoder on cuda:0)
    │       prompt → tokenize → text_encoder.forward() → embedding
    │       embedding.to(transformer.device)   ← 跨卡显式转移
    ├── for step in scheduler.timesteps:
    │       latent = latent.to(transformer.device)  ← 已在 transformer device,no-op
    │       noise_pred = transformer.forward(latent, t, embedding)
    │       latent = scheduler.step(noise_pred, latent)
    ├── latent.to(vae.device)                   ← 跨卡显式转移
    └── image = vae.decode(latent)
```

每次跨组件调用的 `.to()` 是显式的,微秒级。这正是 ComfyUI `samplers.py:986` 干的事 — 我们参考实现。

### 3.3 加载与缓存模型

- **L1**:runner ModelManager 按 `(file, device, lora_set)` 复合 key 缓存已加载组件;同样的 unet+lora 组合在不同 workflow / 同 workflow 反复跑均命中
- **L2**:runner 内 image_generate output cache(LRU 50 条),hash 输入(全部组件描述符 + prompt + 采样参数 + seed)。`is_deterministic=True`(seed 非空)才参与;随机 seed 不缓存
- L1/L2 均为 in-memory;backend 重启丢

#### L2 cache entry schema

**关键**:cache 不存 signed URL(URL 自带 expires,默认 3600s,缓存命中时大概率已过期 → 客户端拿到 403)。改存「图片磁盘 anchor + 元数据」,命中时**重签 URL**:

```python
@dataclass
class L2CacheEntry:
    image_uuid: str          # write_image 返回的 uuid hex
    date: str                # outputs/<date>/<uuid>.png 路径用
    ext: str                 # "png"
    meta: dict               # steps, seed, loras, width, height(不含 URL / expires)
    cached_at: float         # 命中频次统计 + LRU 用

def serve_l2_hit(entry: L2CacheEntry, ttl_seconds: int) -> dict:
    # 1. 校验底层 PNG 还在(TTL 清理脚本可能已删)
    path = outputs_root() / entry.date / f"{entry.image_uuid}.{entry.ext}"
    if not path.exists():
        raise L2CacheMiss()  # 触发重跑 + 重写 cache entry
    # 2. 重签 URL —— HMAC 只要几微秒
    expires = int(time.time()) + ttl_seconds
    token = _sign(entry.image_uuid, expires)
    url = f"/files/images/{entry.date}/{entry.image_uuid}.{entry.ext}?token={token}&expires={expires}"
    return {**entry.meta, "image_url": url, "image_uuid": entry.image_uuid, "image_expires": expires}
```

这样 backend 重启不影响(L2 已经丢),但即便 in-memory cache 跨多日存在,签出的 URL 永远是当前时刻起算 TTL 内有效。底层 PNG 也按现有 outputs TTL 清理逻辑走(不与 L2 互锁,L2 miss 就重跑)。

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
  clip_arch: enum["flux2","flux1","sdxl","qwen"]   # 编码器架构,与 unet 节点 adapter_arch 对称
output_ports:
  clip: dict {kind:"clip", file, device, dtype, clip_arch}
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
  unet:    dict  # 见下面语义
```

**输出语义(明确无歧义)**:
```python
# bypass=True → 直接透传上游 unet 描述符,不动 loras 列表
output_unet = input_unet if bypass else {
    **input_unet,
    "loras": [*input_unet["loras"], {"file": lora_file, "strength": strength}],
}
```

也就是说,输出 dict = 上游 unet 描述符的浅复制 + 在 `loras` 列表末尾 append 一条 `{file, strength}`。bypass=True 时连复制都不做(完全透传)。多次链式应用 = loras list 依次扩展。

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
    clip_arch: str | None = None       # 仅 clip:"flux2"/"flux1"/"sdxl"/"qwen"
```

### 5.2 `DiffusersImageBackend` 重写(rev 2)

`DiffusersImageBackend` 不再 wrap 整个 diffusers Pipeline。改成 hold 三个 component 引用 + 一个 ImageSampler 实例。Pipeline 只在 **load** 阶段用作"模型类构造器"(`Flux2Transformer2DModel.from_pretrained()`),不再用于 **inference**(inference 走我们的 ImageSampler)。

```python
class DiffusersImageBackend(InferenceAdapter):
    modality = MediaModality.IMAGE

    def __init__(self, components: dict[str, ComponentSpec], pipeline_class: str, **kwargs):
        # components = {"unet": ..., "clip": ..., "vae": ...}
        # pipeline_class = "Flux2KleinPipeline" / "Flux2Pipeline" / "StableDiffusionXLPipeline" / ...
        self._components = components
        self._pipeline_class = pipeline_class
        self._transformer = None
        self._text_encoder = None
        self._tokenizer = None
        self._vae = None
        self._scheduler = None
        self._sampler: ImageSampler | None = None

    async def load(self) -> None:
        # 1. 各组件按 ComponentSpec 走 quant_loaders 注册表
        self._transformer = QUANT_LOADERS.dispatch(self._components["unet"])
        self._text_encoder, self._tokenizer = _load_text_encoder_and_tokenizer(self._components["clip"])
        self._vae = QUANT_LOADERS.dispatch(self._components["vae"])

        # 2. 各组件 .to(spec.device) — 跨卡安全(组件 forward 自身不跨卡)
        self._transformer.to(self._components["unet"].device)
        self._text_encoder.to(self._components["clip"].device)
        self._vae.to(self._components["vae"].device)

        # 3. LoRA(仅 unet)
        if self._components["unet"].loras:
            set_active_loras(self._transformer, self._components["unet"].loras)

        # 4. scheduler — 从 pipeline_class 对应的 diffusers config 加载(默认值即可)
        self._scheduler = _load_default_scheduler(self._pipeline_class, self._components["unet"])

        # 5. 构造自写 ImageSampler — 持有所有组件引用 + arch adapter
        self._sampler = ImageSampler(
            transformer=self._transformer,
            text_encoder=self._text_encoder,
            tokenizer=self._tokenizer,
            vae=self._vae,
            scheduler=self._scheduler,
            arch_adapter=MODEL_ARCH_REGISTRY.get(self._pipeline_class),
        )

    async def infer(self, req: ImageRequest) -> InferenceResult:
        # ImageSampler.sample() 是我们写的采样循环 —— 自带跨组件 .to() 处理
        return await self._sampler.sample(req)
```

**关键改动 vs rev 1**:
- 不再 `Flux2Pipeline(transformer=..., text_encoder=..., vae=...)` 实例化(那个 Pipeline.__call__ 跨卡崩)
- `Flux2Transformer2DModel` / `AutoencoderKLFlux2` 等**模型类还用 diffusers 的**(它们自身 forward 不涉及跨卡 — 一个 transformer 整体在一张卡内)
- `pipeline_class` 字段决定走哪种采样调用方式(Klein 是 distilled,无 CFG;Dev 25 步有 CFG;SDXL 有 negative prompt CFG 等)
- ImageSampler 是我们自写,详见 §5.6

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

# GGUF 注册延后到 V2 PR-7(本 spec § 10.1)。dropdown 选中 .gguf 时,scanner
# 标 quant_type="gguf",runner 端报「GGUF 暂未支持,V2 加」拒载。
```

### 5.4 Runner Protocol

`P.RunNode` 不变(`is_deterministic` 字段已存在),`inputs` dict 携带 3 个组件描述符。`runner_process._build_request`:

```python
if node.node_type == "image":
    if all(k in node.inputs for k in ("unet", "clip", "vae")):
        # seed 处理:None / 空字符串 / 缺失 = 随机(影响 is_deterministic 标志);
        # 整数 = 确定性(由上游 workflow_executor._dispatch_node 设 is_deterministic=True)。
        raw_seed = node.inputs.get("seed")
        seed: int | None = int(raw_seed) if raw_seed not in (None, "") else None
        return ImageRequest(
            request_id=f"task-{node.task_id}",
            prompt=str(node.inputs.get("prompt", "")),
            negative_prompt=str(node.inputs.get("negative_prompt", "")),
            steps=int(node.inputs.get("steps") or 25),
            width=int(node.inputs.get("width") or 1024),
            height=int(node.inputs.get("height") or 1024),
            cfg_scale=float(node.inputs.get("cfg_scale") or 7.0),
            seed=seed,
            components={
                "unet": ComponentSpec(**node.inputs["unet"]),
                "clip": ComponentSpec(**node.inputs["clip"]),
                "vae":  ComponentSpec(**node.inputs["vae"]),
            },
        )
    # 老路径:有 model_key 无 components → workflow_executor 已在 dispatch 前
    # inline 展开成等价 components,本分支只是 defense-in-depth(理论上走不到)。
    return ImageRequest(..., model_key=node.model_key)
```

### 5.5 ModelManager 复合 key

`ModelManager._models` 由 `dict[str, LoadedModel]`(key=model_key)改为 `dict[ComponentKey, LoadedModel]`:

```python
ComponentKey = tuple[
    str,                            # file 绝对路径
    str,                            # device,如 "cuda:1"(canonicalized — 无 leading zero)
    str,                            # dtype,如 "bfloat16" / "fp8_e4m3" — 必须在 key 里!
                                    # 同 file 在不同 target dtype 下 dequant 出的张量不同
                                    # (bf16 vs fp16 内存表示不同),不能共享 cache entry。
    frozenset[tuple[str, float]],   # lora_set,每条 (lora_file, strength) frozenset
]
```

- 老 `model_key` 路径在 load 时翻译成等价的 ComponentKey(yaml `paths.transformer/text_encoder/vae` → 3 个 ComponentKey)
- `is_loaded(model_key)` 老 API 兼容:翻译后查询(三组件全 loaded → True)
- 新 `is_component_loaded(key: ComponentKey) -> Literal["loaded","loading","cold","failed"]` API
- `evict_lru(gpu_index)` 不变(按 entry.gpu_index 维度淘汰)
- **Device canonicalization**:`cuda:00` / `cuda:007` 等带前导零的写法必须在 `ComponentSpec.device` validator 里规范化为 `cuda:0` / `cuda:7`,否则两 spec 物理同卡但 ComponentKey 不等,造成 cache 双装

**ComponentSpec 跨字段约束**(rev 2 修订 — code review 第 2 条):
- `loras` 非空 → `kind == "unet"`(Flux2 LoRA 只 patch DiT)
- `adapter_arch is not None` → `kind == "unet"`
- `clip_arch is not None` → `kind == "clip"`

违反任一约束 pydantic ValidationError,防止静默丢字段。

### 5.6 ImageSampler(rev 2 新增 — 自写采样循环)

**定位**:取代 diffusers `Pipeline.__call__` 的黑盒采样。整套设计参考 ComfyUI `comfy/samplers.py:986` 的 `outer_sample` 流程,但只覆盖到"采样循环 + 跨组件 tensor 转移",**不重写**组件本身的 forward(那是 diffusers 模型类的责任)。

#### 5.6.1 接口

```python
class ImageSampler:
    """自写采样循环,handle 跨组件 device 转移。

    架构:三段式 — encode_prompt / denoise_loop / vae_decode,每段开头显式
    .to() 把输入搬到目标组件的 device。每段内部完全是单 device 计算(组件
    forward 自身不涉及跨卡)。
    """

    def __init__(
        self,
        transformer,       # diffusers Flux2Transformer2DModel etc.
        text_encoder,      # transformers AutoModelForCausalLM etc.
        tokenizer,         # transformers tokenizer
        vae,               # diffusers AutoencoderKL etc.
        scheduler,         # diffusers FlowMatchEulerDiscreteScheduler etc.
        arch_adapter: "ModelArchAdapter",
    ): ...

    async def sample(self, req: ImageRequest) -> InferenceResult:
        # 1. encode_prompt
        embeds = await self._encode_prompt(req.prompt, req.negative_prompt)

        # 2. init latent
        latent = self._init_random_latent(req)  # on transformer.device

        # 3. denoise loop
        for step_idx, t in enumerate(self._scheduler_timesteps(req.steps)):
            latent = self._denoise_step(latent, embeds, t, req, step_idx)
            await self._on_progress(step_idx, req.steps)

        # 4. vae decode
        image = await self._vae_decode(latent)

        # 5. encode image to PNG bytes + return InferenceResult
        return ImageResult(media_type="image/png", data=image_to_png(image), metadata={...})

    def cancel(self) -> None:
        """spec §G2 mid-sampler cancel — sets internal flag, denoise loop checks per step."""
        self._cancel_flag.set()
```

#### 5.6.2 跨组件 .to() 钩子(关键)

```python
async def _encode_prompt(self, prompt: str, negative_prompt: str | None) -> dict:
    # 输入(prompt str)是 CPU 数据,tokenize 在 CPU,然后送 text_encoder.device
    inputs = self._tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(self._text_encoder.device) for k, v in inputs.items()}
    embeds = self._text_encoder(**inputs).last_hidden_state
    # 跨组件转移:embedding 从 text_encoder.device 搬到 transformer.device
    return embeds.to(self._transformer.device)

def _denoise_step(self, latent, embeds, t, req, step_idx):
    # latent 和 embeds 已经在 transformer.device(由 _encode_prompt 保证)
    # transformer.forward 内部完全单 device,不跨卡
    noise_pred = self._transformer(
        hidden_states=latent,
        timestep=t.to(latent.device),
        encoder_hidden_states=embeds,
    ).sample
    # scheduler.step 是纯数学,跟 device 无关
    return self._scheduler.step(noise_pred, t, latent).prev_sample

async def _vae_decode(self, latent):
    # 跨组件转移:latent 从 transformer.device 搬到 vae.device
    latent = latent.to(self._vae.device)
    # vae.decode 内部完全在 vae.device 上
    return self._vae.decode(latent / self._vae.config.scaling_factor).sample
```

#### 5.6.3 ModelArchAdapter(rev 2 新增)

不同 diffusers Pipeline 的采样流程有差异:
- **Klein** (distilled):9 步,无 CFG,无 negative prompt
- **Flux2 Dev**:25 步,有 guidance scale(虽然是 distilled 模式),有 negative prompt
- **SDXL**:有 CFG,有 negative prompt,timestep + sigma 计算稍不同
- **Z-Image-Turbo** (distilled):4-9 步,无 CFG

ImageSampler 主体逻辑同,**步进/CFG/negative prompt 处理**由 `ModelArchAdapter` 注入:

```python
class ModelArchAdapter(Protocol):
    """每种 Pipeline 一个 adapter,告诉 ImageSampler 怎么调采样。"""
    def supports_cfg(self) -> bool: ...
    def supports_negative_prompt(self) -> bool: ...
    def encode_negative_prompt(self, tokenizer, text_encoder, prompt: str) -> torch.Tensor: ...
    def transformer_forward(self, transformer, latent, t, embeds, **kw) -> torch.Tensor: ...
    def vae_decode(self, vae, latent) -> torch.Tensor: ...

# 注册
MODEL_ARCH_REGISTRY: dict[str, ModelArchAdapter] = {
    "Flux2KleinPipeline": FluxKleinArchAdapter(),       # distilled,9 步
    "Flux2Pipeline":      FluxDevArchAdapter(),         # mainline
    # ...
}
```

V1 本 spec 只实现 `FluxKleinArchAdapter`(磁盘上模型是 Klein)。其他 adapter 留接口,future 添加。

#### 5.6.4 Cancel 与 progress

- `cancel()` 设 internal flag,denoise loop 每步检查 → 抛 `NodeCancelled`(spec §G2)
- 每步 await `_on_progress(step_idx, total)` → 通过 callback 发 NodeProgress 到 RunnerClient

#### 5.6.5 不支持的(明确标记非目标)

- 在采样中间插自定义节点(ComfyUI 的 KSampler 支持,我们不支持 — workflow 编辑器粒度停在"完整 image_generate"节点)
- DisTorch 风格的"按 layer 拆 transformer 跨 device"(整 transformer 一张卡)
- 批量推理(`batch_size > 1`)— V2 follow-up

#### 5.6.6 参考实现来源

| 我们要写的 | ComfyUI 对应 | 备注 |
|---|---|---|
| `ImageSampler.sample()` | `samplers.py:986 outer_sample()` | 主流程 |
| `_denoise_step()` | `samplers.py:KSampler.inner_sample()` 里的 step | step 调用 |
| Cross-device `.to()` | ComfyUI-MultiGPU `wrappers.py` 的 device override + samplers 内自动迁移 | 我们更显式 |
| `ModelArchAdapter` | ComfyUI 没有(他们每模型自写 ldm/) | 我们更轻,只 dispatch 调用 |

---

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
- **辅助 prewarm**:节点 UI 右上角图标 → `POST /api/v1/models/components/preload` —— **批量接口**:

```http
POST /api/v1/models/components/preload
Content-Type: application/json
Authorization: admin cookie

{
  "components": [
    {"file": "/path/to/unet.safetensors", "device": "cuda:1", "dtype": "bfloat16", "loras": [{"file": "...", "strength": 0.8}]},
    {"file": "/path/to/clip.safetensors", "device": "cuda:0", "dtype": "bfloat16"},
    {"file": "/path/to/vae.safetensors",  "device": "cuda:2", "dtype": "bfloat16"}
  ]
}
```

Backend 给 image runner 串行下发 LoadModel 命令(runner 内 per-model lock 自动串)。返回 `202` + task_id,客户端通过 WS `component_state_changed` 监听最终态。**批量是必须的**——用户场景一定是一次 warm 「unet + clip + vae(+ loras)」整套组合,逐个 endpoint 调三次额外 round-trip。

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

- **前端**(本 spec 不实施,V2 PR-8 加,见 § 10.2):打开老 workflow 自动展开 3 loader + N lora_apply
- **后端**(本 spec **实施**):`workflow_executor._dispatch_node` 检测老格式
  - node.type==`image_generate` && node.data.model_key && 无 unet/clip/vae 边
  - 从 yaml 读 `paths.transformer / .text_encoder / .vae`(或 `quantized_transformer`)
  - 构造 3 个 ComponentSpec(device="auto"),塞进 inputs.unet/clip/vae
  - LoRA list 转 lora_apply 链(或直接合并进 unet 描述符的 loras)
  - 之后路径与新格式一致

## 8. PR 拆分

| PR | 目标 | 估计行数 |
|---|---|---|
| **PR-1**(纯 infra)| (a) `ComponentSpec` + `ComponentKey` 类型;(b) `QuantLoaderRegistry` + 5 loader(plain / fp8mixed 重用 / mxfp8mixed 新 / nvfp4mixed 新 / GGUF 拒载);(c) `ModelManager._components` 缓存 + `get_or_load_component` / `is_component_loaded` / `unload_component` API(parallel 于现有 `_models`,legacy 路径不动)。**不**改 image_diffusers / Pipeline 调用 — 纯类型 + 缓存基础设施 | ~450 |
| **PR-2**(自写 ImageSampler — rev 2 新)| `ImageSampler` 类(§5.6)+ `ModelArchAdapter` 协议 + `FluxKleinArchAdapter` 第一个实现 + `DiffusersImageBackend` 改成 hold 组件引用 + load 阶段构造 ImageSampler + infer 走 ImageSampler.sample。Cancel flag / progress callback 接入。**关键 PR,工程量最大**。包含 Task 0 verify 脚本的"反向版本"——验证自写采样在跨 device 下出图正确性(同 prompt + seed 跟 diffusers Pipeline 单 device 版输出像素级 / SSIM > 0.99) | ~800 |
| **PR-3**(scanner)| `component_scanner` 服务 + `GET /api/v1/components?role=...` + `POST /scan` + WS `component_index_changed` + `backend/configs/model_paths.yaml` | ~250 |
| **PR-4**(workflow 节点)| 4 个新节点(`unet_load` / `clip_load` / `vae_load` / `lora_apply`)+ `image_generate` 改造;`runner_process._build_request` 加 components 分支;workflow_executor 老格式 inline 展开为 ComponentSpec 组合 | ~700 |
| **PR-5**(状态展示 + UI)| `GET /api/v1/models/components/state` 批量查询 + `POST /api/v1/models/components/preload` 批量预热 + WS `/ws/models` 加 `component_state_changed`;前端 `useComponentState` hook + 4 节点 React 组件 + palette 子分类 | ~300 |
| **PR-6**(L2 cache)| L2 image_generate output cache(LRU 50,entry schema 见 §3.3)+ `is_deterministic` 标志贯通 + L2 命中时**重签 URL** + WS `node_cache_hit` + TaskPanel 节点 "(cached)" 角标 | ~250 |

**总 6 个 PR**,串行依赖。每个独立可 ship 可灰度。

### PR-2 风险点(rev 2 关键)

PR-2 是本 spec 的核心工程风险:
- **diffusers 升级耦合**:我们 ImageSampler 调 `Flux2Transformer2DModel.forward` 等内部 API。diffusers 0.39+ 改签名 → PR-2 要 patch。缓解:`ModelArchAdapter` 把 forward 调用方式 isolate 在一层 — 升 diffusers 只动 adapter。
- **数学正确性**:自写采样 vs diffusers Pipeline 必须出**几乎同样**的图(同 prompt + seed)。PR-2 必须有 SSIM 对比测试,baseline 是 `Flux2KleinPipeline.from_pretrained(...).to("cuda:1")(prompt=..., seed=...)` 的输出。SSIM > 0.99 才算正确。
- **Scheduler 行为**:Klein 是 distilled,sigma 序列固定;Dev 是 mainline,sigma 由 step count 算出。`_load_default_scheduler` 必须返回跟 Pipeline 同款 scheduler(从 `scheduler/` 子目录加载即可,但要核对 config)。
- **Cancel 时序**:cancel flag 必须每步 atomic 检查,避免 cancel 到一半但 NodeProgress 还在发(spec §G2 现有规范)。

## 9. Test Plan

| 层 | 覆盖 |
|---|---|
| Unit | ComponentSpec 序列化 / quant dequant 每格式 fixture(bf16/fp16/fp8mixed/mxfp8mixed/nvfp4mixed)/ component_scanner glob / is_component_loaded 状态机 / L2 cache LRU + URL 重签 / lora_apply 输出语义(append + bypass)/ 老 model_key → ComponentKey 翻译 / **`ImageSampler._denoise_step` 单测(给定固定 latent + embed + noise_pred,验出 next latent 跟 diffusers scheduler.step 同 tensor)** / **`ModelArchAdapter` 注册 + dispatch 单测** |
| Integration | (a) 4 新节点 + image_generate 全链路(fake_adapter=True),描述符流转 + L1 复合 key 缓存命中;(b) **Loader 状态四态切换**:cold → preload 触发 loading → loaded → 强制 OOM 模拟 → failed → retry → loaded(每态 WS 事件断言);(c) **PR-2 核心**:`ImageSampler.sample` 跟 `Flux2KleinPipeline.__call__` 单 device 输出 SSIM 对比(同 prompt + seed,SSIM > 0.99)|
| Smoke(真模型) | 1) **跨卡 smoke**:Flux2-bf16 三组件分卡(unet→cuda:1, clip→cuda:0, vae→cuda:2)真出图,SSIM 跟单卡 baseline > 0.99,耗时 ≤ 35s;2) **单卡 smoke**:Flux2-bf16 单卡 Pro 6000(三组件都 .to("cuda:1"))耗时 ≤ 28s;3) 改 LoRA strength 二跑 < 5s;4) 同 seed 二跑 < 0.1s(L2 命中 + URL 重签);5) **联合场景**:三组件分卡 + 2 LoRA + 同 seed 二跑,验 L1 unet 重 patch、clip/vae 复用、L2 命中签新 URL,总耗时 < 2s;6) **Cancel mid-sampler**:25 步采样,t=10 步时调 cancel,500ms 内 NodeResult status=cancelled 到达 |
| 前端 | vitest:4 节点渲染 + 端口连接 + useComponentState hook(订阅/取消订阅时序)+ dropdown 数据源拉取 |
| Regression | master 上 image-e2e-test(309542918354374656)+ 新工作流(308084173191516160)零改动跑通,验 workflow_executor inline 展开正确;**PR-2 新增 SSIM regression test**(每 commit CI 跑,baseline 不动)|

## 10. Future Work

### 10.1 V2 PR-7:GGUF 量化处理
- 引入 [city96/ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) dequant 逻辑
- 装 `gguf` Python 包
- 加 `load_gguf` 到 quant_loaders 注册表
- 启用磁盘上 4 个 Flux2-Q4/Q5/Q6/Q8 文件(~28GB 节省)

### 10.2 V2 PR-8:前端编辑器自动迁移老 workflow
- 打开老 workflow 检测 `image_generate(model_key=X)` → 弹窗"已检测到老格式,是否展开"
- 自动 insert 3 loader + N lora_apply 节点
- 自动布局到 image_generate 左侧

### 10.3 V2 PR-9:跨进程持久化中间缓存
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
