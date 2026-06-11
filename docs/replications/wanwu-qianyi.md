# wanwu-qianyi 复刻 manifest

来源: `Klein - 万物迁移.json`(ComfyUI,B 站 wuli大雄oO 同款) | 日期: 2026-06-11

原工作流 = 万物迁移 LoRA 的 A/B 对比测试:图1 主图 + 图2 迁移参考(光线/画风),
LoRA 条 vs 裸 Klein 基模同种子对比,右上 ImageReel 拼标注长图。

## 创建的资源(卸载按从上往下顺序删)

| 资源 | id / 名称 | 卸载 |
|---|---|---|
| API key | `323306064118288384`(wanwu-qianyi-external,sk-wanw-9d…) | `DELETE /api/v1/keys/323306064118288384` |
| 服务 | `323306062335709184`(wanwu-qianyi,image) | `DELETE /api/v1/services/wanwu-qianyi` 或置 retired |
| 服务版工作流 | `323306061140332544`(万物迁移·服务版(单链)) | `DELETE /api/v1/workflows/323306061140332544` |
| 调试版工作流 | `323110738921000960`(Klein·万物迁移 A/B(LoRA vs 基模)) | `DELETE /api/v1/workflows/323110738921000960` |
| 模型文件 | `image/loras/Klein-万物迁移.safetensors`(279MB) | 共享资产,确认无他人引用再删 |

代码级能力(通用,不随本复刻卸载):#468 image_ref_join 参考图合并节点、
#484 细粒度路径 LoRA 应用修复、#485 LoRA adapter 名消毒。

## 复刻参数(对照 ComfyUI 原件)

Flux2-Klein-9B fp8 + qwen_3_8b_fp8mixed CLIP + flux2-vae;euler / 4 步
(Flux2Scheduler 分辨率 shift)/ cfg=1 / LoRA Klein-万物迁移@0.8;
多参考图经 image_ref_join(A 口=图1 主图,B 口=图2 参考);
**KSampler 宽高必须设主图比例**(锚定主体,= ComfyUI GetImageSize 语义)。

## exposed 契约

`POST /v1/apps/wanwu-qianyi`,Bearer key。

| key | 说明 | 必填 | 默认 |
|---|---|---|---|
| image_1 | 主图(被改的),base64 data URI | 是 | — |
| image_2 | 参考图(风格/光线来源) | 是 | — |
| prompt | 迁移指令 | 否 | 将图1变为图2的画风 |
| width / height | 出图尺寸(设主图比例) | 否 | 1024 / 768 |
| seed | 种子 | 否 | 随机 |
| lora_strength | LoRA 强度 | 否 | 0.8 |

返回 `outputs.dec.image_url`(签名 URL,1h TTL)。

真机验证:2026-06-11 真人照→动漫画风迁移出片(质量同源 up 主档),
外部 Bearer 调用 4s 出图(模型常驻)。
