# 原生复刻四图像功能(去 ComfyUI 依赖)+ 发布为服务 + 测试页

**Status**: Draft
**Author**: heygo(设计 by Claude)
**Date**: 2026-06-07

## 0. 目标

把 Infinite-Canvas(外接 ComfyUI)的四个图像功能,在 **nous 原生引擎**里复刻(不依赖 ComfyUI),
发布为可外部 API 调用的服务,并把它的四功能前端搬进来做服务测试页。

四功能 → 模型:
- 文生图 = **Z-Image-Turbo**(6B distilled,8 步,guidance=0)
- 细节增强 = **SeedVR2 放大**(已有)± Z-Image controlnet(可选)
- 图片编辑 = **Flux2-Klein**(已有引擎)+ 多参考编辑入口
- 角度控制 = **Qwen-Image-Edit-2511** + fal Multiple-Angles-LoRA(96 相机位)

## 1. 可行性结论(已验证)

- **diffusers 已钉的 commit(0.38.0.dev0 / `c8eba433`)已自带** `ZImagePipeline`、`ZImageImg2ImgPipeline`、
  `QwenImageEditPlusPipeline`、`Flux2KleinPipeline` —— **无需 bump diffusers**(避开 golden smoke 风险)。
- nous 已有:`ModularImageBackend`(Flux2)、`AnimaImageBackend`(第二适配器,证明可扩展)、SeedVR2、
  节点包系统、workflow→service 发布、`/v1/apps/{name}/run`、`/v1/images/generations`、通用 Playground。
- 缺:Z-Image / Qwen-Image-Edit 两个架构的适配器 + 节点包 + 权重(公开,代下);多架构派发现为
  runner 里硬编码 if-elif(`arch=="anima"?...:"Flux2KleinPipeline"`),未建注册表。
- 权重盘上没有 → 代下(Z-Image-Turbo 下载中;Qwen-Edit 后续)。

## 2. 架构:ImageArchSpec 多架构注册表(P0)

现状痛点:加一个模型架构要改散落多处——`runner_process.py` 派发 if-elif、`image_modular.py` 的
`pipeline_class` 硬 check、`model_arch_adapter.py` 的 `MODEL_ARCH_REGISTRY`、bundled configs。

P0 收成一张**注册表**(对照 ComfyUI 靠权重 key 自动检测的思路,但我们显式声明):
```python
@dataclass
class ImageArchSpec:
    arch: str                 # "flux2" | "z-image" | "qwen-edit" | "anima"
    pipeline_class: str       # diffusers 类名
    adapter_factory: Callable # 建对应 InferenceAdapter
    supports_cfg / negative / default_steps / default_guidance / samplers / schedulers
    needs_image_input: bool   # 编辑类(qwen-edit / flux2-edit)=True
IMAGE_ARCH_REGISTRY: dict[str, ImageArchSpec] = {...}
```
- runner 派发、引擎 pipeline 选择、采样器校验都从注册表读 → 加架构 = 注册一条。
- 现有 flux2 / anima 迁进注册表(行为不变,零回归;真机 golden smoke 验 Flux2 SSIM≥0.97)。

## 3. P1:Z-Image-Turbo 文生图(端到端模板)

把「加一个文生图架构 + 节点 + 工作流 + 发布服务 + API」走通一遍,作为 P2/P3 的模板。

### 3.1 引擎适配器
- `ModularImageBackend` 支持 `pipeline_class="ZImagePipeline"`:`ZImagePipeline.from_pretrained(repo)`;
  Z-Image distilled → `guidance_scale=0`、默认 8 步;text encoder=qwen3、自带 vae/scheduler。
  整模型 HF-layout(下载的 `diffusers/Z-Image-Turbo/`),走 `_build_*` 的 from_pretrained 路径(非单文件桥接)。
- 注册 `ZImageArchSpec`(supports_cfg=False,default_steps=8,default_guidance=0,samplers/schedulers 按 Z-Image)。
- 复用现有逐组件选卡/offload/守卫先腾后载(image_modular 通用能力)。

### 3.2 节点包 `backend/nodes/z-image/`
- `z_image_load`(整模型 loader → 描述符:repo/device/dtype)+ `z_image_ksampler`(width/height/steps/seed)
  + `z_image_decode`(dispatch 终端 → ImageRequest)。或复用 flux2 的 encode/ksampler 抽象(若通用)。
- executor.py inline 产描述符;dispatch 终端摊平 ImageRequest(pipeline_class=ZImagePipeline)。

### 3.3 工作流 + 发布
- 建一条 nous 工作流(z_image_load → ksampler → decode),exposed_inputs=prompt/size,exposed_outputs=image。
- 经 `/api/v1/workflows/{id}/publish` 发布成 ServiceInstance → `/v1/apps/z-image/run` + `/v1/images/generations`(model=z-image)。

### 3.4 验证(真机)
- standalone smoke:`ZImagePipeline` 出图正确(prompt 一致性、8 步)。
- 服务端到端:`/v1/images/generations` model=z-image 出图。

## 4. P2-P5(P1 模板复制)

- **P2 角度控制**:Qwen-Image-Edit-2511 适配器(`QwenImageEditPlusPipeline`,needs_image_input)+ 多角度 LoRA;
  节点带输入图 + prompt;`Qwen-Image-Edit-2511` ~20B + qwen_2.5_vl_7b,显存大 → 靠逐组件选卡/offload。
- **P3 图片编辑**(Flux2 多参考编辑入口:Flux2KleinPipeline 的 image/reference 输入 + Kontext 多参考)
  + **细节增强**(复用 SeedVR2;Z-Image controlnet 可选)。
- **P4 服务化**:四条都发布 + `/v1/images/generations` 按 model 路由 + Key 授权/配额(已有)。
- **P5 测试页**:把 Infinite-Canvas 四功能 UI 搬进服务测试页(前端):文生图/增强/编辑/角度控制四个 tab,
  「引擎来源」切本地(nous)。现有通用 Playground 之外的 per-service 自定义 UI(需前端新组件)。

## 5. 风险
- **权重大 + 下载慢**:Z-Image ~12GB、Qwen-Edit ~40GB+。代下,P2 前备好。
- **Qwen-Edit 显存**:~20B + 7B VL encoder,单卡紧 → 逐组件跨卡 + offload(已建,见 [[project_image_vram_guard_arc]])。
- **注册表重构动 Flux2 路径**:P0 必须真机 golden smoke 验 Flux2 不回归(SSIM≥0.97,改 image_modular 的硬规则)。
- **图片编辑/Kontext 多参考**:Flux2KleinPipeline 的多参考/图输入 API 要核(可能要 ZImageImg2Img / 专门 edit pipeline)。
- per-service 自定义测试页:前端工作量,P5 单独评估。

## 6. PR 拆分(每 PR 独立 + CI;动 image_modular 的跑真机 smoke)
- PR-P0:ImageArchSpec 注册表 + flux2/anima 迁入(真机 golden smoke)。
- PR-P1a:Z-Image 引擎适配器 + 注册 + standalone smoke。
- PR-P1b:z-image 节点包 + 工作流 + 发布 + `/v1/images/generations` 路由 + 服务端到端验。
- PR-P2/P3/P4/P5:按上。
