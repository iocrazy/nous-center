# SeedVR2 三节点对齐 ComfyUI 设计

状态:设计（2026-06-02）。落地分多个独立 PR（见末尾拆分）。
前置:SeedVR2 引擎/runner/单节点已落地（#287/#288/#289/#290/#292），本设计把单
`seedvr2_upscale` 节点重构成对齐 ComfyUI 最新 NumZ 的**三节点**,解锁大图分块(tiling)+
3090 显存(blockswap)+ 更多控制。

## 背景与动机

NumZ 最新 ComfyUI 节点(v0.22.x,`interfaces/{dit_model_loader,vae_model_loader,video_upscaler}.py`,
真源码已读,在 `/home/heygo/sites/ComfyUI/custom_nodes/seedvr2_videoupscaler/`)是**三节点**:

1. **SeedVR2 (Down)Load DiT Model** → 产 `SEEDVR2_DIT` **配置 dict**(惰性,不真加载)
2. **SeedVR2 VAE** → 产 `SEEDVR2_VAE` **配置 dict**
3. **SeedVR2 增强**(`image + dit + vae → image`)→ 真加载 + 超分

我们当前单节点把 DiT/VAE 配置 + tiling + blockswap 全藏了 → 大图 VAE OOM、7B 在 3090 OOM、
跟 ComfyUI 脱节。用户(2026-06-02)选「全套三节点对齐」。

**关键事实(已 grounding)**:两个 loader 只产**配置字典**(不真加载),真加载在增强节点;
我们 vendored 的 `prepare_runner` **本就接** `block_swap_config`/`encode_tiled`/`encode_tile_size`/
`decode_tiled`/...(generation_utils.py:431-435),只是现在 adapter 没传。所以本质工作 =
「把参数分三组的 UI + adapter 把配置串进已支持的 prepare_runner」,非重写引擎。

## ComfyUI 真实 config dict 形状(逐字,已读源)

DiT loader `execute` → config:
```
{model, device, offload_device, cache_model, blocks_to_swap, swap_io_components,
 attention_mode, torch_compile_args, node_id}
```
VAE loader `execute` → config:
```
{model, device, offload_device, cache_model, encode_tiled, encode_tile_size,
 encode_tile_overlap, decode_tiled, decode_tile_size, decode_tile_overlap,
 tile_debug, torch_compile_args, node_id}
```
增强 `execute(image, dit, vae, seed, resolution, max_resolution, batch_size,
uniform_batch_size, color_correction, temporal_overlap, prepend_frames,
input_noise_scale, latent_noise_scale, offload_device, enable_debug)`:
- `block_swap_config = {"blocks_to_swap": N, "offload_device": torch.device(...)}`(仅 N>0 或 swap_io)
- `prepare_runner(dit_model, vae_model, model_dir, ..., block_swap_config=..., attention_mode=dit.attention_mode,
   encode_tiled=vae.encode_tiled, encode_tile_size=(N,N), encode_tile_overlap, decode_tiled=..., decode_tile_size=(N,N), ...)`
- DiT device 与 VAE device **独立**(`dit["device"]` / `vae["device"]`)。

## nous-center 映射

| ComfyUI 节点 | nous-center 节点 | 执行类 | 端口/输出 |
|---|---|---|---|
| Load DiT | `seedvr2_load_dit` | **inline**(无 GPU,只 bundle config) | 输出 `dit`(新端口类型 `seedvr2_dit`) |
| Load VAE | `seedvr2_load_vae` | **inline** | 输出 `vae`(新端口类型 `seedvr2_vae`) |
| 增强 | `seedvr2_upscale`(重构现有) | **dispatch**(image runner) | 输入 `image`+`dit`+`vae` → 输出 `image` |

**数据流**:两个 loader inline 产 config dict(像 flux2 loader 产描述符)→ 经 edges 作 inputs 进
`seedvr2_upscale` → runner `_build_request` 构 `UpscaleRequest`(带 dit/vae config + image_url +
增强参数)→ adapter.load() 用 dit/vae config(device/blockswap/attention + tiling)调 prepare_runner →
adapter.upscale() 用增强参数。

## 改动面

### 引擎(`SeedVR2UpscaleBackend`)
- `__init__/load` 现在硬编码单 device + sdpa + 无 blockswap/tiling。改成接 **dit config + vae config**:
  DiT device、blocks_to_swap、swap_io_components、offload_device、attention_mode;VAE device、
  encode/decode tiled + tile_size + overlap。构 `block_swap_config`、把 tiling 传 `prepare_runner`。
- `upscale()` 已有 batch_size/color_correction/noise_scale;补 max_resolution、temporal_overlap、
  prepend_frames、uniform_batch_size。
- DiT/VAE 可落不同卡(prepare_runner 已支持 dit_device/vae_device 分离)。

### Request(`UpscaleRequest`)
- 加 `dit: dict`(DiT config)+ `vae: dict`(VAE config),或扁平化关键字段。倾向带 `dit`/`vae`
  两个 dict(忠实 ComfyUI),adapter 直接消费。保留现有 `image`/`resolution`/`seed`/`color_correction`/
  noise_scale;加 max_resolution/batch_size/temporal_overlap/prepend_frames。

### ModelManager 缓存键
- `get_or_load_seedvr2_adapter` 现在 key=(model_dir,dit,vae,device)。blockswap/tiling 在 prepare_runner
  时生效 → 改 key 纳入 dit_device/vae_device/blocks_to_swap/swap_io/offload(不同配置 = 不同 runner 实例)。
  tiling 是 encode/decode 时参数,可不进 key(同 runner 不同 tiling 仅影响单次推理)—— 但 prepare_runner
  也收 tiling(配 runner),稳妥起见 model_id 一并纳入。**真机验缓存命中/失效行为**。

### 节点包
- `seedvr2_load_dit` / `seedvr2_load_vae` 加进 `nodes/seedvr2/`(node.yaml + executor.py 的 inline executor
  产 config dict)。`seedvr2_upscale` 改:inputs 加 `dit`/`vae`,去掉直接的 dit_model widget(移到 DiT loader)。
- 新端口类型 `seedvr2_dit` / `seedvr2_vae`(前端 NODE_DEFS + 类型校验)。

### 前端
- 全声明式 → node.yaml 自动出节点。VAE model 也做磁盘感知 select(复用 seedvr2_model_select 模式,
  加 VAE 白名单端点)。device 用 select(cuda:0/1/2/cpu)。tiling/blockswap 是 widget。
- 端口类型 `seedvr2_dit`/`seedvr2_vae` 要在前端类型表登记(连线校验)。

## PR 拆分(每个独立绿门控)
- **PR-1 引擎**:adapter 接 dit/vae config(device/blockswap/attention + tiling)+ 增强参数,串进
  prepare_runner。`UpscaleRequest` 扩展。ModelManager 缓存键纳入。**真机 smoke:大图 tiling 不 OOM +
  7B blockswap 塞 3090**。
- **PR-2 三节点**:`seedvr2_load_dit`/`seedvr2_load_vae` inline 节点 + executor + 新端口类型;重构
  `seedvr2_upscale` 取 dit/vae 输入;runner `_build_request` 从 config 构扩展 `UpscaleRequest`。
- **PR-3 前端打磨**:VAE 磁盘感知 select、device select、tiling/blockswap 高级区;更新已建的工作流
  (image_input → load_dit/load_vae → upscale → image_output);**真机端到端**。

## 约束/坑
- 读真源(已做);忠实复刻 config dict 形状 + prepare_runner 接线。
- blockswap 需 offload_device != device 且非 macOS(我们 Linux,OK)。3B 0-32 块、7B 0-36 块。
- DiT/VAE device 独立 → 可 DiT 落 Pro6000、VAE 落别的卡;但 image runner 子进程当前 pin 单卡组,
  跨卡要确认 runner 的 CUDA_VISIBLE_DEVICES 不挡(可能需 runner 见多卡或 device 限同组)。**PR-1 真机验**。
- 改引擎前后必跑真模型 smoke(CLAUDE.md)。
参见 [[project_seedvr2_integration]]、[[project_seedvr2_pr3_design]]、[[feedback_read_comfyui_source]]。
