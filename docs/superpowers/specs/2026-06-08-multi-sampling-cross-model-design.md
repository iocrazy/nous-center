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

### 1.2 「接上一段图」有两种语义,各模型支持的不同(已核实真 pipeline 源)

钉的 commit(`c8eba433`)diffusers `pipelines/{flux2,z_image,qwenimage}/`:

**(a) 真 img2img(VAE encode 输入图 → 加噪到 strength 点 → 去噪)**——只有带 `strength` 的 Img2Img 变体:
- ✅ `ZImageImg2ImgPipeline` **存在**:`__call__(image=, strength=0.6, num_inference_steps, ...)`,
  `prepare_latents(init_image, ...)` + `get_timesteps(steps, strength)` —— **真 img2img**。
- ❌ **`Flux2KleinImg2ImgPipeline` 不存在**。Flux2 系只有 `Flux2Pipeline` / `Flux2KleinPipeline` /
  `Flux2KleinInpaintPipeline` / `Flux2KleinKVPipeline`,**无 img2img**。
- `QwenImageEditPlusPipeline` **无 strength**。

**(b) 多参考「编辑」条件(把参考图 encode 成 image_latents,denoise loop 里和生成 latent 拼接)**——
不是从输入图加噪起步,是把它当条件:
- `Flux2KleinPipeline.__call__(image=, ...)`(pipeline_flux2_klein.py:613):`image` → `prepare_image_latents`
  → denoise loop `torch.cat([latents, image_latents], dim=1)`(:844)。**这是参考/编辑条件,无 strength**。
- `QwenImageEditPlusPipeline` 同理(edit 参考条件)。
- ⚠️ **我们 image_modular.py:829-840 现有的 `if "image" in signature` 注入,对 Flux2-Klein/Qwen-Edit
  走的就是 (b) 这条** —— 即 Flux2-Klein「接上一段图」**今天就能用,且语义是参考编辑,零引擎改**。

**(c) latents= 注入**:`ZImagePipeline` / `Flux2KleinPipeline` 都接受 `latents=`,但**无 `denoising_start`**
  ⇒ 同空间真 latent 接力(部分去噪)得手搓 sigma 截断(`image_modular.py` injected-sigma 路径,行 844-851)。

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

latent 不兼容 ⇒ 必走像素空间。但「接上一段图」分两种语义(见 §1.2),按模型支持的能力切:

```
Z-Image 采样 → VAE decode 成图(签名 URL)
            → 下一段接这张图,两种接法:
              (A-ref)   参考编辑条件:Flux2-Klein / Qwen-Edit 的 image=(已通,见下)
              (A-img2img) strength<1 加噪重去噪:仅 ZImageImg2ImgPipeline 有
            →(可选第三段)
```

- **A-ref(跨模型,Flux2-Klein 接 Z-Image 图)= 今天就能用,零引擎改**:image_modular.py:829-840 的
  `image=` 注入对 Flux2-Klein 走的就是多参考编辑条件。工作量只在**前端串接**(上一段 image_output →
  下一段 input_image)+ 建示例工作流 + 真机核图。这是用户「Z-Image+Flux Klein 三采样」最直接的落地。
- **A-img2img(strength 加噪重去噪)= 仅 Z-Image 有 pipeline**:需在引擎接 `ZImageImg2ImgPipeline`
  (新 pipeline_class + `strength` 入参 + arch 注册表标 `img2img_pipeline_class`)。Flux2-Klein 无 img2img
  变体,**不做**(强行手搓 = 重写 pipeline,违背不重写引擎原则)。
- 两种都复用 `input_image` + 签名 URL,**不碰 IPC / msgpack / latent 张量**。

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

### 3.3 路 A-ref(跨模型参考编辑链)= 纯工作流编排,零引擎新代码
- 第二段(Flux2-Klein / Qwen-Edit)连 `input_image`(上一段 image_output URL)。
- 引擎现有 `image=` 注入(image_modular.py:829-840)已把它走多参考编辑条件 → **今天就通**。
- 工作量:**Studio UI 串接入口 + 示例工作流 + 真机核图**,无引擎改动。

### 3.4 路 A-img2img(strength 加噪重去噪)= 仅 Z-Image,引擎接新 pipeline
- 仅 `ZImageImg2ImgPipeline`(带 strength)。引擎按「有 input_image + arch 有 img2img 变体」选它。
- arch 注册表加 `img2img_pipeline_class` 字段;无该字段的架构(Flux2-Klein)走 §3.3 参考条件,不做 strength。

## 4. PR 拆分(按 [[feedback_pr_per_change]],逐 PR 独立分支 + CI 绿)

### 路 A-ref(跨模型参考编辑链)——先做,直达用户主诉,**零引擎改**
- **PR-A1 前端 + 工作流**:Studio 串接入口(上一段 image_output → 下一段 input_image)+ 建
  Z-Image→Flux2-Klein 参考编辑示例工作流。真机核:三采样链出图正确。引擎不动(现有 image= 路径)。

### 路 A-img2img(Z-Image strength 重去噪)——次做,引擎接新 pipeline
- **PR-A2 引擎**:`ModularImageBackend` 接 `ZImageImg2ImgPipeline`(`strength` 入参);arch 注册表加
  `img2img_pipeline_class`;有 input_image + 该 arch 有 img2img 变体 → 走它。真机 golden smoke:
  Flux2 SSIM≥0.97 零回归 + Z-Image img2img(strength)出图正确。
- **PR-A3 节点/前端**:采样节点暴露 `strength`(仅有 img2img 变体的架构生效)。

### 路 B(同空间真 latent 接力)——后做,工作量在引擎内部
- **PR-B1 latent 落盘 + VAE Decode output_mode**:`"latent"` 模式产 `latent_ref` 描述符
  (落盘 + 路径引用),默认 `"image"` 零改动。runner 落盘走现有签名 URL/文件机制(非 msgpack 张量)。
- **PR-B2 sample_from_latent 节点 + 截断 sigma**:注入 `latents=` + sigma 截断部分去噪;
  arch/channels 派发前校验 + 人话报错。
- **PR-B3 前端**:VAE Decode latent 输出端口 + sample_from_latent 节点 + 端口类型 `latent_ref`。

## 5. 验证(真机,CLAUDE.md 强制)

- 改 `image_modular.py` 必跑 `tests/manual/smoke_image_ab.py`(真模型/GPU),Flux2 golden SSIM≥0.97。
- 路 A-ref:Z-Image→Flux2-Klein 参考编辑链真机出图,人工核图(无引擎改,但要确认 image= 注入对
  Flux2-Klein 真生效 + 三采样链端到端通)。
- 路 A-img2img:Z-Image `ZImageImg2ImgPipeline`(strength)真机出图。
- 路 B:同模型 latent 接力真机出图;latent_ref 落盘→读回→接续去噪 bit 级可复现;
  跨架构 latent 注入触发派发前校验报错(不崩)。
- CI 跑不了真模型(conftest mock torch + 无 GPU),引擎正确性只靠 standalone smoke。

## 6. 不做 / 风险

- **不**为 Z-Image↔Flux2-Klein 强行做 latent 通道转换(16↔128 无意义,物理不同空间;过像素是正解)。
- **不**往 msgpack / `NodeResult` / `ImageRequest` 塞 torch.Tensor(IPC 炸);latent 一律落盘传引用。
- **不** bump diffusers(避开 golden smoke 风险;ZImageImg2ImgPipeline 已在钉的 commit 里)。
- **不**给 Flux2-Klein 强行做 strength img2img(无对应 pipeline,手搓 = 重写引擎,违背原则;
  Flux2-Klein 接上一段图走参考编辑条件即可)。
- 风险:截断 sigma 部分去噪(路 B)是手搓,需对照 ComfyUI `common_ksampler` 的 sigma 切片逐数值核;
  Z-Image distilled 作 refiner 收益有限(8 步/guidance=0)。

## 7. 范围(用户 2026-06-08 拍:路 A + 路 B 都做)

执行顺序:
- **PR-A1**(路 A-ref,零引擎改)先做 —— Studio 串接 + Z-Image→Flux2-Klein 参考编辑示例工作流,真机核图。
- **PR-A2/A3**(路 A-img2img,Z-Image strength)—— 引擎接 ZImageImg2ImgPipeline + 暴露 strength。
- **PR-B1/B2/B3**(路 B 同空间 latent 接力)—— latent 落盘 + sample_from_latent + sigma 截断 + 前端。
- 每 PR 独立分支、CI 绿、引擎改动真机 smoke。路 B 落地前再评 distilled refiner 实际收益。
