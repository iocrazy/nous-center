# ideogram4 复刻 manifest

来源: `【Work-Fisher】Ideogram4半自动V3版.json`(ComfyUI,Ideogram-4 开源权重 2026-06-03) | 日期: 2026-06-11

原工作流 = Ideogram-4 双 DiT 文生图(DualModelGuider 非对称 CFG)+ 结构化 JSON caption
(文字排版/bbox)。与万物迁移不同,这次是**新模型架构接入**(3 个代码 PR + 工作流/服务)。

## 创建的资源(卸载按从上往下顺序删)

| 资源 | id / 名称 | 卸载 |
|---|---|---|
| API key | `323433244487847936`(ideogram4-external,sk-ideo-7c…) | `DELETE /api/v1/keys/323433244487847936` |
| 服务 | `323433243686735872`(ideogram4,image) | `DELETE /api/v1/services/ideogram4` 或置 retired |
| 工作流 | `323430777943494656`(Ideogram-4 文字海报) | `DELETE /api/v1/workflows/323430777943494656` |
| 模型 | `image/diffusers/Ideogram-4-bf16/`(53.6G,社区转档 CalamitousFelicitousness) | 确认无引用后删目录 |
| 源素材 | `/media/heygo/Program/models/【83】Ideogram4全自动流程(1)/`(comfy 单文件版,nous 用不上) | 可删(diffusers 不支持其单文件加载) |

代码级能力(通用,不随本复刻卸载):
- #490 diffusers bump 784fa626(golden SSIM=1.0 零回归)
- #491 ideogram4 架构注册 + 引擎 builder + guidance 互斥处理 + step 回调
- #493 fp8 量化名单补 unconditional_transformer

## 复刻参数(对照 ComfyUI 原件)

| ComfyUI | nous |
|---|---|
| UNETLoader ×2 + DualModelGuider(cfg 7) | Load Checkpoint(整模型,双 DiT 随 repo)+ ksampler cfg_scale=7(引擎 guidance_schedule=None 走标量) |
| Ideogram4Scheduler [20,1024,1024,0.5,1.75] | steps=20;mu/std 用 pipeline 默认(0/1.5,差异可忽略,需要时引擎再暴露) |
| CLIPLoader qwen3vl(type ideogram4) | 整模型自带 Qwen3-VL TE |
| euler_ancestral | euler(pipeline FlowMatchEuler;ancestral 待引擎扩) |
| caption JSON 工具节点群 | 直接把 JSON 字符串喂 text_input(提示词模版见源文件夹 提示词模版.txt) |

## exposed 契约

`POST /v1/apps/ideogram4`,Bearer key。

| key | 说明 | 必填 | 默认 |
|---|---|---|---|
| prompt | Ideogram caption JSON(或普通文本) | 是 | — |
| width / height | 出图尺寸 | 否 | 1024 |
| steps | 步数 | 否 | 20 |
| cfg_scale | guidance | 否 | 7 |
| seed | 种子 | 否 | 随机 |

返回 `outputs.dec.image_url`。

## 注意事项

- **模型内嵌 safety**(蒸馏进权重,非 pipeline 组件):欠规格/可疑 caption 会确定性输出
  「Image blocked by safety filter」占位图。用完整结构的 caption JSON(参考提示词模版)。
- 权重协议:代码 Apache 2.0,**权重非商用**;商用需向 Ideogram 买授权。
- bf16 双 DiT ~54G;与 resident vLLM(39G)同卡时画布 Load Checkpoint 精度选
  `fp8_e4m3`(torchao 在线量化,#493 后全组件覆盖 ~28G)。官方 fp8 repo 是 gated,
  用户在 HF 同意协议后可换官方版。
- 真机记录:画布 e2e 65s(冷加载+量化+20 步);服务热调 24s;文字渲染按 caption 规格。
