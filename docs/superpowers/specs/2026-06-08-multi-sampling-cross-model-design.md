# 多采样 / 跨模型链式采样(对齐 ComfyUI 双采样)设计

**Status**: Draft
**Author**: heygo(设计 by Claude)
**Date**: 2026-06-08

## 0. 目标

让 nous 工作流支持「把上一段采样的结果喂给下一段采样器继续生成」,对齐 ComfyUI 的
双采样 / 多采样:
- **跨模型**(用户主诉:Z-Image + Flux2-Klein 三采样)。
- **同模型 refiner**(Z-Image→Z-Image、Flux→Flux 接着去噪)。

向后兼容是硬约束:现有单采样工作流必须**字节级零改动**。

## 1. 关键事实(已读真代码,先把前提钉死)

### 1.1 Z-Image-Turbo 和 Flux2-Klein 的 latent 空间**物理不兼容**

对照两边 `config.json` 真实数值(`/media/heygo/Program/models/nous/image/diffusers/`):

| | Z-Image-Turbo | Flux2-Klein |
|---|---|---|
| VAE 类 | `AutoencoderKL` | `AutoencoderKLFlux2` |
| latent 通道 | **16** | **32 → patchify 128** |
| Transformer in_channels | 16 | 128 |
| scaling_factor | 0.3611 | 无(batch-norm 归一化) |
| shift_factor | 0.1159 | 无 |

Z-Image-Turbo 的 latent(16ch / scale 0.3611 / shift 0.1159)= ComfyUI 经典 **Flux1** 的
`latent_format`(`comfy/latent_formats.py` Flux 类:latent_channels=16, scale_factor=0.3611,
shift_factor=0.1159)。这就是 Infinite-Canvas `Z-Image.json` 用 `flux ultra vae.safetensors`
的原因——Z-Image-Turbo 复用 Flux1 VAE。

Flux2-Klein 是另一个空间(`AutoencoderKLFlux2`,32→128ch,`flux2-vae.safetensors`)。

**⇒ 16 通道 latent 喂不进 128 通道 transformer。Z-Image↔Flux2-Klein 无法做纯 latent 接力。**
ComfyUI 同理:`comfy/sample.py:fix_empty_latent_channels` 只在 empty latent 补通道,非空跨模型
通道不匹配照样 UNet 报错。latent 不兼容的跨模型,ComfyUI 也只能**过一遍像素空间**
(VAE decode → VAE encode)。

### 1.2 我们 diffusers 的 pipeline 不支持 `denoising_start` / `strength`

钉的 commit(`c8eba433`)里:
- `ZImagePipeline` / `Flux2KleinPipeline` 都**接受 `latents=` 注入**,但**都没有
  `denoising_start` / `strength` 参数**(全文检索无)。
- ⇒ 没法用 diffusers 原生「从第 N 步接着去噪」。要做真 latent 接力(部分去噪)只能像 ComfyUI
  手搓 sigma 截断:注入 latent + 自定义 sigma schedule(我们 `image_modular.py` 已有 injected-sigma
  路径,行 844-851,可复用)。
- 注意:`ZImageImg2ImgPipeline` / `Flux2KleinImg2ImgPipeline`(img2img 变体)**带 `strength`**——
  像素链路直接用它,不用手搓。

### 1.3 现状:latent 是描述符 + 单体 pipe(),从不离开 runner 进程

- `exec_ksampler`(`backend/nodes/flux2-components/executor.py:168`)返回**纯描述符 dict**
  `{"_type":"flux2_latent", model, conditioning, width, height, steps, cfg_scale, sampler_name,
  scheduler, seed, input_image?}` —— **无张量**。
- 真采样在 runner 终端 `flux2_vae_decode`(dispatch 节点)一把 `pipe(**call_kwargs)`
  (`image_modular.py:891`)跑完 encode→denoise→decode。中间 latent 只活在 `pipe()` 内部,
  callback(`cb_kwargs["latents"]`)能读但只投成 96px JPEG 预览,从不导出。
- **节点间传的是签名 URL(图)**,不是张量。runner IPC 走 msgpack(`protocol.py:232`),
  **torch.Tensor 进 msgpack 直接炸**。图像字节落盘 `NAS_OUTPUTS_PATH`,只有 URL 过 pipe。
- `input_image` 字段已存在(qwen-edit 工作流在用 img2img),即「上一段图喂下一段」的管线**已通**。

## 2. 两条路(分清,别混)

### 路 A:跨模型像素链(用户要的 Z-Image+Flux Klein 三采样)

latent 不兼容 ⇒ 必走像素空间。本质是 **img2img 链**:
```
Z-Image 采样 → VAE decode 成图(签名 URL)
            → Flux2-Klein img2img(VAE encode + strength<1 去噪)→ 图
            →(可选第三段)
```
- 复用现有 `input_image` + 签名 URL 管线,**不碰 IPC / msgpack / latent 张量**。
- 引擎侧:`needs_image_input` 的架构走 `*Img2ImgPipeline`(带 `strength`),已在 diffusers commit 里。
- 这是 latent 不兼容跨模型的**唯一物理路径**,且最小破坏。

### 路 B:同空间真 latent 接力(对齐 ComfyUI 双采样原教旨)

仅当两段 latent 空间相同(同模型,或同为 Flux1 16ch 空间的不同模型)才可做,**不过 VAE**:
```
KSampler A(出 latent 张量)→ KSampler B(注入 latent + 截断 sigma 接着去噪)→ VAE Decode
```
- 需要把 latent 张量在节点间传递 ——**但不能进 msgpack**。解法:**latent 落盘**(`.pt` /
  `safetensors`),节点间传**路径引用描述符**(和现在传图 URL 同构),零 IPC 改动。
- 部分去噪靠手搓 sigma 截断(`image_modular.py` injected-sigma 路径复用),非 diffusers 原生。
- 限制:Z-Image distilled 8 步 / guidance=0,作 refiner 余地小;真正受益的是非 distilled 模型。

## 3. 向后兼容(两条路都兼容,核心设计点)

现有单采样工作流的契约:`KSampler → VAE Decode(dispatch 终端)→ image_output`,
终端一把 pipe() 出图。**不能动这条默认路径。**

### 3.1 VAE Decode 加 `output_mode`(路 B 用)
- `VAE Decode` 节点加 `output_mode = "image" | "latent"`,**默认 `"image"` = 现有行为不变**。
- `"latent"` 模式:终端不 decode,把真 latent 张量落盘,产出 latent 描述符
  `{"_type":"latent_ref", path, arch, latent_channels, scale_factor, shift_factor, width, height}`
  (可序列化,路径引用,不进 msgpack)。
- ⇒ 不连下游 latent 节点的老工作流走 `"image"` 分支,字节级一致。

### 3.2 新节点 `sample_from_latent`(路 B 用)
- 读 `latent_ref` 描述符 → 注入 `latents=` + 截断 sigma 接着去噪 → 再产 latent_ref 或交给 VAE Decode。
- 入口校验:`latent_ref.arch` / `latent_channels` 与本段模型不匹配 → **拒绝 + 人话报错**
  (对齐 [[project_anima_arch_mismatch]] 的派发前校验思路,别让 UNet 抛晦涩 shape 错)。

### 3.3 img2img 链(路 A)= 纯工作流编排,几乎零引擎新代码
- 第二段采样节点连 `input_image`(上一段的 image_output URL),设 `denoise/strength<1`。
- 引擎按 `needs_image_input` + 有 `input_image` → 选 `*Img2ImgPipeline`。
- 现有 qwen-edit img2img 已证明这条管线通,主要是**在 Studio UI 暴露 strength + 串接入口**。

## 4. PR 拆分(按 [[feedback_pr_per_change]],逐 PR 独立分支 + CI 绿)

### 路 A(跨模型像素链)——先做,价值直达用户主诉,破坏最小
- **PR-A1 引擎**:`ModularImageBackend` 在 `needs_image_input` + `input_image` present 时走
  `*Img2ImgPipeline`(`strength` 入参);注册表 `ImageArchSpec` 标 img2img pipeline_class。
  真机 golden smoke:Flux2 SSIM≥0.97 零回归 + Z-Image→Flux2-Klein img2img 出图正确。
- **PR-A2 节点/工作流**:采样节点暴露 `strength`(img2img 时生效);建一条 Z-Image→Flux2-Klein
  三采样示例工作流。
- **PR-A3 前端**:Studio UI 暴露 strength + 串接入口(上一段输出→下一段 input_image)。

### 路 B(同空间真 latent 接力)——后做,工作量在引擎内部
- **PR-B1 latent 落盘 + VAE Decode output_mode**:`"latent"` 模式产 `latent_ref` 描述符
  (落盘 + 路径引用),默认 `"image"` 零改动。runner 落盘走现有签名 URL/文件机制(非 msgpack 张量)。
- **PR-B2 sample_from_latent 节点 + 截断 sigma**:注入 `latents=` + sigma 截断部分去噪;
  arch/channels 派发前校验 + 人话报错。
- **PR-B3 前端**:VAE Decode latent 输出端口 + sample_from_latent 节点 + 端口类型 `latent_ref`。

## 5. 验证(真机,CLAUDE.md 强制)

- 改 `image_modular.py` 必跑 `tests/manual/smoke_image_ab.py`(真模型/GPU),Flux2 golden SSIM≥0.97。
- 路 A:Z-Image→Flux2-Klein img2img 真机出图,人工核图(像素链不退化)。
- 路 B:同模型 latent 接力真机出图;latent_ref 落盘→读回→接续去噪 bit 级可复现;
  跨架构 latent 注入触发派发前校验报错(不崩)。
- CI 跑不了真模型(conftest mock torch + 无 GPU),引擎正确性只靠 standalone smoke。

## 6. 不做 / 风险

- **不**为 Z-Image↔Flux2-Klein 强行做 latent 通道转换(16↔128 无意义,物理不同空间;过像素是正解)。
- **不**往 msgpack / `NodeResult` / `ImageRequest` 塞 torch.Tensor(IPC 炸);latent 一律落盘传引用。
- **不** bump diffusers(避开 golden smoke 风险;img2img 变体已在钉的 commit 里)。
- 风险:截断 sigma 部分去噪(路 B)是手搓,需对照 ComfyUI `common_ksampler` 的 sigma 切片逐数值核;
  Z-Image distilled 作 refiner 收益有限(8 步/guidance=0)。

## 7. 范围决策(待用户拍)

spec 覆盖两条路。落地范围三选一:
- 只做路 A(跨模型像素链)= 直达「Z-Image+Flux Klein 三采样」,最小破坏。
- 路 A + 路 B(加同模型真 latent 接力),工作量大。
- 先路 A 真机验通,再评估路 B 是否值得(distilled refiner 收益存疑)。
