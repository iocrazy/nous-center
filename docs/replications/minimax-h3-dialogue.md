# minimax-h3-dialogue 复刻 manifest

来源: `【MINIMAX-H3】八月最强文戏-高配版.json`(ComfyUI UI 导出,86 节点 / 20 分组)
日期: 2026-09-03
方式: **ComfyUI 桥(模板即服务)**,非原生复刻 —— 与 `minimax-h3-r2v`、`krea2` 同一条路。
中文标签: **MiniMax-H3 文戏高配版**(桥的 `ComfyTemplate` / `ServiceInstance` 都只有
`name` 一个字段,没有 display label 列,所以中文名只活在本文档与各 exposed_param 的
`label` 里)。

## 为什么走桥而不是原生复刻

工作流依赖 `MiniMaxH3*T8*` 一整套第三方节点(AudioConditioning / DualClockSampler /
LearnedTwoPassParityPlan / LearnedLatentUpscale / TwoPassLatentReconcile /
TwoPassDetailMixer / AVDecode)、`LoraLoaderBypassModelOnly`、
`LayerUtility: ImageScaleByAspectRatio V2`、`ResolutionSelector`、
`ComfyMathExpression`、`VHS_VideoCombine`。这些节点封装的是 H3 双通道(video/audio
两个时钟)采样契约,nous 侧没有对应能力,原生复刻等于把上游节点包重写一遍。

## UI → API 格式转换

桥吃 API 格式(`{node_id: {class_type, inputs}}`),原始文件是 UI 格式。用 comfy-cli
的客户端转换,`object_info` 取自 sidecar(**本机 sidecar 在 `:8888`,不是文档默认的
`:8188`** —— `backend/.env` 的 `NOUS_COMFY_URL=http://127.0.0.1:8888`):

```python
from comfy_cli.workflow_to_api import convert_ui_to_api
api = convert_ui_to_api(ui_workflow, object_info)   # object_info 取自 GET /object_info
```

结果: 86 节点 → **36 节点**。被消除的是 GetNode×24 / SetNode×15(KJNodes 前端虚拟
节点,链路直连)、MarkdownNote×1,以及 14 个 `mode=4`(bypass)节点。

`comfy validate --workflow <api.json> --input <object_info.json>` →
**valid: true, 0 errors, 0 warnings**。

### ⚠️ 同目录已有的 `API运行/…_api.json` 是过期导出,别直接用

新转的 36 节点 vs 已有导出的 39 节点,diff 只有两处,但都说明已有导出比 UI 文件旧
(UI 文件 05:42 保存,导出 05:39):

| 差异 | 说明 |
|---|---|
| 已有导出多 `#32` `MiniMaxH3MemoryEfficientSageAttentionPatch`、`#153` 同类、`#154` `ModelAttentionBackend` | 这三个节点**在当前 UI 文件里根本不存在**,是上一版留下的旁支(而且没有任何下游消费,ComfyUI 也不会执行) |
| `#223`/`#225` `LoadAudio` 多一个 `audioUI: ""` | UI 侧的展示用 widget,不是真入参 |

图结构其余完全一致。**以新转的为准**,已入库的就是新转的这份。

## 节点结构与 6 图 / 3 音频的语义

### MarkdownNote 没有使用说明

工作流里唯一的 `MarkdownNote`(`#166`)**只是一张尺寸对照表**(megapixels × 16:9 →
输出宽高),没有讲玩法。下面的角色划分是从连线和提示词文本反推的。

### 提示词就是"剧本",图/音频靠序号被它引用

`MiniMaxH3AudioConditioningT8` 的 `ref_images` / `ref_audios` 是 autogrow 列表,
节点 tooltip 明说 "Presents drive_audio to Qwen/DiT as an official **`<Audio N>`**
reference"。所以:

- `ref_image_0/1/2` 就是提示词里的 **`<Picture 1>` / `<Picture 2>` / `<Picture 3>`**
- `ref_audio_0/1/2` 就是 **`<Audio 1>` / `<Audio 2>` / `<Audio 3>`**
- **不是"一个角色一张脸 + 一段音频"的固定绑定**。绑定关系全写在提示词文本里,例如
  工作流自带脚本的写法是
  `<Subject1>Gu Yan是<Picture 1>画面左侧男性,保留参考图中的面部特征…`
  —— 一张双人同框图同时锚定两个角色,靠"画面左侧/右侧"区分。
- 说话人用 `S1` / `S2` 标记,台词写成
  `<Subject 2> (S2) says:` + `<d>[Chinese]…</d>`;分镜写成 `[Shot 1]`…`[Shot 5]`。
- `strict_prompt_tags=true`,所以这套标签是硬约束,不是自由文本。

结论:**图和音频是"素材池 + 序号",角色分工由提示词声明**。想加角色就在提示词里多写
一个 `<SubjectN>` 并指向某个 `<Picture N>`。

### ⚠️ 实际只有 3 图 + 2 音频进了服务

UI 文件里确实有 6 个 `LoadImage` + 3 个 `LoadAudio`,但**参考图 4/5/6 和参考音频 3 在
用户保存的这一版里是 bypass(`mode=4`)状态**,连同它们的缩放节点、Preview、Set 节点
一共 14 个节点被 `convert_ui_to_api` 剔除。API 图里只剩:

| UI 分组 | LoadImage/LoadAudio 节点 | 冻结的文件名 | 落到 conditioning 的槽位 |
|---|---|---|---|
| 参考图1 | `#238` → 缩放 `#236` | `ee1374a88b158631.png` | `ref_images.ref_image_0` = `<Picture 1>` |
| 参考图2 | `#237` → 缩放 `#231` | `5 (1).jpg` | `ref_images.ref_image_1` = `<Picture 2>` |
| 参考图3 | `#230` → 缩放 `#232` | `c284078e…cae.png` | `ref_images.ref_image_2` = `<Picture 3>` |
| 参考图4/5/6 | `#218`/`#227`/`#217` | — | **bypass,不在 API 图里** |
| 参考音频1 | `#225` | `h3_reference_audio_2_15s.wav` | `ref_audios.ref_audio_0` = `<Audio 1>` |
| 参考音频2 | `#223` | `h3_reference_audio_1_15s.wav` | `ref_audios.ref_audio_1` = `<Audio 2>` |
| 参考音频3 | `#222` | — | **bypass,不在 API 图里** |

所以服务只暴露 3 图 + 2 音频。要开到 6 图 3 音频,得先在 ComfyUI 里把那几个节点解除
bypass、保存,再重新转 API 并重传模板 + 补映射 —— 本次按"不改工作流文件本身"的约束
没做。

三张图都经 `LayerUtility: ImageScaleByAspectRatio V2`(`original` 比例、`crop`、
lanczos、对齐 32、总像素 1280k)归一后才进 conditioning,所以上传图不必自己裁尺寸。

### 两遍采样(LOW → 潜空间放大 → HIGH 精修)

```
#164 ResolutionSelector ─┐
#165 ComfyMathExpression ┤ (帧数 = 时长×24,向上对齐 17n+5)
#167 CR Prompt Text ─────┤
3×LoadImage → 3×Scale ───┼→ #7  AudioConditioningT8 (LOW,  width/height 来自 #164)
2×LoadAudio ─────────────┘        │
                                  ├→ #8 DualClockSamplerT8 (steps 8, shift 12/3)
                                  ├→ #9 ParityPlan (base 8 / coarse 4 / refine 4)
                                  └→ #12 SamplerCustomAdvanced (LOW 出 latent)
                                        ↓
              #13 LearnedLatentUpscaleT8Advanced (scale_by 1.5, 学习式 3D 潜空间放大)
                                        ↓  (它的 width/height 输出反过来喂 #14)
              #14 AudioConditioningT8 (HIGH,同一套图/音频/提示词,尺寸跟随 #13)
                                        ↓
              #15 TwoPassLatentReconcile → #16 TwoPassDetailMixer → #19 SamplerCustomAdvanced
                                        ↓
              #20 MiniMaxH3AVDecodeT8 → (IMAGE, AUDIO) → #226 VHS_VideoCombine
```

- **模型**: `#46` UNETLoader `FeiHou_MiniMax-H3_Remix_v0.6_int8_convrot_v2.safetensors`
  + `#50` `LoraLoaderBypassModelOnly` 挂 `minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16`
  (强度 0.75)+ `#33` `ModelAttentionBackend` = pytorch attention。
- **CLIP**: `#160` `qwen3vl_32b_minimax_h3_int8_convrot.safetensors`(type=minimax)。
  `#161` 的 nvfp4_awq 版本在 API 图里是**孤儿节点**,没有下游,ComfyUI 不会执行它。
- **VAE**: `#156` video VAE fp16、`#158` audio VAE fp32 —— 两个 VAE 是 H3 音视频双通道的
  必备件。
- **帧数**: `#165` 的表达式
  `max(5, round(a*24)) + (5 - (max(5, round(a*24)) % 17)) % 17`,即时长 `a` 秒 × 24fps
  向上对齐到 `17n+5` 网格(节点 tooltip:"24fps; snapped up to the 17n+5 H3 grid")。
  **对齐会让成片略长于请求值**,调用方按下表预期:

  | `duration_s` | 帧数 | 成片时长 @24fps |
  |---|---|---|
  | 2 | 56 | 2.33s |
  | 4 | 107 | 4.46s |
  | 8 | 192 | 8.00s |
  | 15 | 362 | 15.08s |
- **输出**: `#226` `VHS_VideoCombine`,`video/h265-mp4`、`yuv420p10le`、crf 22、
  frame_rate 24,音轨来自 `#20` 的 AUDIO 输出。**没有 SaveVideo 节点**,产物由
  VHS 落盘后桥去抓。
- **无 CFG**:走 `BasicGuider`(纯条件引导),图里没有任何 cfg 入参,所以 exposed 里
  也没有 CFG 这一项。

## 创建的资源(卸载按从下往上顺序删)

- 服务 + 模板: `minimax-h3-dialogue` / template_id **`353841663996596224`**
  → `DELETE /api/v1/comfy-templates/353841663996596224`(级联删 ServiceInstance)
- API key: 未铸(验证走 `ADMIN_TOKEN` 的 admin-session 回退路径)
- 模型文件: **共享资产,勿删** —— 由 ComfyUI sidecar 自己的 `extra_model_paths.yaml`
  管理,nous 侧没有引用

## exposed 契约

`POST /v1/services/minimax-h3-dialogue/predictions`(注意前缀 `/v1/`,不是 `/api/v1/`)
带 `Prefer: respond-async` → 202 → 轮询 `GET /v1/predictions/{id}`。

| key | 类型 | 节点.入参 | 必填 | 默认 | 说明 |
|---|---|---|---|---|---|
| `prompt` | string | `167.prompt` | **是** | 工作流自带的 5 分镜双人脚本 | 剧本:角色设定 + 音色设定 + 分镜 + 台词。`<Picture N>` / `<Audio N>` / `S1`/`S2` / `[Shot N]` / `<d>[Chinese]…</d>` 是硬标签 |
| `image1` | image | `238.image` | **是** | — | `<Picture 1>` |
| `image2` | image | `237.image` | 否 | — | `<Picture 2>` |
| `image3` | image | `230.image` | 否 | — | `<Picture 3>` |
| `audio1` | audio | `225.audio` | **是** | — | `<Audio 1>`,S1 音色参考,建议 ≤15s 干净人声 |
| `audio2` | audio | `223.audio` | 否 | — | `<Audio 2>`,S2 音色参考 |
| `duration_s` | number 2–15 | `163.value` | 否 | **2** | 秒;内部 ×24fps 后对齐到 17n+5 帧 |
| `aspect_ratio` | string(enum 8) | `164.aspect_ratio` | 否 | `16:9 (Widescreen)` | 画面比例 |
| `megapixels` | number 0.2–2.0 | `164.megapixels` | 否 | 0.5 | 第一遍采样的画布大小 |
| `hires_scale` | number 1.0–2.0 | `13.scale_by` | 否 | 1.5 | 潜空间放大倍数,决定最终成片尺寸 |
| `seed` | integer | `11.noise_seed` | 否 | 随机 | `random=true`,留空即随机 |

**最终分辨率 = ResolutionSelector(aspect_ratio, megapixels) × hires_scale**。
`duration_s` 的默认值被压到最短的 2 秒(便于 smoke 与试参),生产调用请显式传。

### 没暴露什么,以及为什么

- **`steps`(`8.steps` = 8)/ ParityPlan 的 base 8 / coarse 4 / refine 4**:三者与
  `fl2v_turbo_8step` LoRA 是一套契约,单独改 `steps` 会让两遍采样的 sigma 划分对不上。
- **`frame_rate`(`226` = 24)**:`#165` 的帧数表达式把 24fps 写死在里面,改这里会让
  时长与帧数脱钩。
- **DetailMixer 的 Tail / Bias / STG / Restart 开关**:上游默认全关,属于调参实验位。
- **模型/LoRA/VAE/CLIP 文件名、attention backend、`task_type=Ref2VA`、
  `audio_mode=native`、`ref_image_size=max`、`reference_video_policy=official_2_to_15s`**:
  内部常量。

### ⚠️ 已知契约缺陷:选填的图/音频"关不掉"

桥的取值是 `data.get(key, mapping_default)`,再落回**工作流冻结的原值**。`LoadImage` /
`LoadAudio` 是必连输入,桥没法把一条边断开。所以:

- 不传 `image2` / `image3` / `audio2` **不等于"这一路不参与"** —— 工作流里冻结的那张
  示例图 / 那段示例音频仍然会作为 `<Picture 2>` / `<Picture 3>` / `<Audio 2>` 进条件。
- 而 `#238`(参考图1)、`#237`(参考图2)冻结的 `ee1374a88b158631.png` 与 `5 (1).jpg`
  **和 `example.png` 是同一个文件**(md5 `e6d676fb…`,8589 字节的 ComfyUI 自带占位图)。
  也就是说工作流保存时这两个槽位放的是占位图,不是真素材。
- 补救:**三张图都传**,或者在提示词里根本不引用 `<Picture 2>` / `<Picture 3>`。
  和 krea2 一样,这类条件语义只能写进 `label` 带出去。

产物结构(嵌套,不是顶层 `items`):

```
output.outputs.bridge.video_url    → /files/images/<date>/<hash>.mp4?token=…&expires=…
output.outputs.bridge.items[0].url → 同一个 URL
```

注意路径段是 `images` 不是 `videos` —— 桥的落盘走
`image_output_storage.write_image`,那个函数把签名 URL 固定拼成
`/files/images/<date>/<uuid>.<ext>`,只有扩展名随产物类型变。

### 调用示例

```bash
# 1) 建模板 + 服务(已做过,留档)
curl -X POST http://127.0.0.1:8000/api/v1/comfy-templates \
  -H "Authorization: Bearer $ADMIN_TOKEN" -H 'Content-Type: application/json' \
  -d '{"name":"minimax-h3-dialogue","workflow":<api.json>}'

# 2) 设 mapping(已做过,留档)
curl -X PUT http://127.0.0.1:8000/api/v1/comfy-templates/353841663996596224/mapping \
  -H "Authorization: Bearer $ADMIN_TOKEN" -H 'Content-Type: application/json' \
  -d '{"exposed_params":[…]}'

# 3) 异步预测。文件型参数收两种形态:data URI(桥自动上传到 sidecar)
#    或 sidecar input/ 目录里已有的文件名。
curl -X POST http://127.0.0.1:8000/v1/services/minimax-h3-dialogue/predictions \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H 'Prefer: respond-async' -H 'Content-Type: application/json' \
  -d '{"input":{
        "prompt":"<Subject1>是<Picture 1>中的少女…\n<Subject 1> (S1) says:\n<d>[Chinese]今天真冷啊。</d>",
        "image1":"data:image/png;base64,…",
        "audio1":"data:audio/wav;base64,…",
        "duration_s":2, "seed":20260903}}'
# → 202 {"id":"…","status":"starting"}

# 4) 轮询到终态
curl -H "Authorization: Bearer $ADMIN_TOKEN" \
     http://127.0.0.1:8000/v1/predictions/<id>
```

鉴权:`/v1/*` 先按 M:N `InstanceApiKey` 校验,`ADMIN_TOKEN` 不是 InstanceApiKey,
会退回 admin-session 校验 —— 所以脚本化调用直接拿 `ADMIN_TOKEN` 当 bearer 就行,
不必先铸 key。

## 真机验证

2026-09-03,经 nous 服务(不是直连 sidecar),`ffprobe` 断言 video + audio 双流。

| # | prediction | 参数 | predict_time | 端到端 | 产物 |
|---|---|---|---|---|---|
| 1 | `353848502905737216` | `duration_s=2`, `seed=20260903` | **111.8s**(冷,含权重加载) | 115.5s | 293,744 B |
| 2 | `353849263177863168` | 同上,`seed=777` | **59.1s**(热) | 63.1s | 310,438 B |

两次产物规格一致:

| 项 | 值 |
|---|---|
| 视频流 | hevc **1440×832**,56 帧,24/1 fps |
| 音频流 | aac 32000 Hz,**双声道** |
| 时长 | 2.333s(= 56 帧 ÷ 24,与 17n+5 对齐表吻合) |
| URL | `/files/images/2026-09-03/<hash>.mp4?token=…&expires=…` |

**分辨率实测 1440×832**,不是按 ResolutionSelector(0.5MP/16:9/mult 32 → 960×544)
乘 1.5 直推的 1440×816 —— 高度被 `#13` 的潜空间网格又抬了一档。要精确尺寸以实测为准。

**冷热差一倍**:第 1 次要把 `MiniMaxH3TEModel_`(25,882 MB)+ `MiniMaxH3`(32,427 MB)
+ 两个 VAE 从盘搬进显存,`model_type FLOW_AV`;第 2 次权重常驻,只跑采样。生产估算按热态
的 ~59s / 2.3 秒片长。

**语义核对 ✓**:参考图是一张棕色长发、白大衣白围巾的少女;提示词写的是「黄昏雪原、
远处低矮山脊、呼出一小口白气」。抽第 8/30/52 帧看,人物形象与参考图一致,雪原黄昏背景
和呼白气都出现了 —— 图参考、文本分镜、音频通道三条链路端到端都通。

**参数形态覆盖**:`image1` / `audio1` 走 data URI(桥自动上传到 sidecar),
`image2` / `image3` / `audio2` 走"sidecar input/ 里已有的文件名"。两种形态都验证过。
请求体 ~3.1 MB,没有触发任何体积限制。

### 踩到的两个坑(都不是 H3 的问题)

1. **后端重启会把在飞的 prediction 变成永久孤儿**。03:23:49 提交的
   `353847545677484032` 在 22 秒后遇上 `nous-engine-backend` 重启(03:24:11 Stopping /
   03:24:13 Started),执行任务随进程一起没了,但 DB 行永远停在 `processing`,
   ComfyUI 侧从没收到过那张图。只能手动 `POST /v1/predictions/{id}/cancel` 再重发。
2. **别拿 sidecar 的 `got prompt` 反推"我的任务在跑"**。当时 sidecar 日志里确实有一条
   03:24:55 的 `got prompt`、47.4 秒跑完,但那是别人的 krea2(日志里是 `Krea2TEModel_` /
   `WanVAE` / llama-cpp)。判断 H3 任务是否真在跑,要看日志里有没有 `MiniMaxH3*` 的
   `Requested to load` —— `/queue` 的深度和 GPU 利用率都会骗人。

## 运维注意

- sidecar `nous-engine-comfyui` 在 `:8888`(不是文档默认的 `:8188`),绑
  `CUDA_VISIBLE_DEVICES=1` = RTX PRO 6000 96G。H3 实测要搬
  25,882 MB(TE)+ 32,427 MB(UNet)+ 两个 VAE 进显存,只有这张卡装得下。
- `comfy_bridge._SEM(1)` 全局串行:`krea2`、`minimax-h3-r2v`、`minimax-h3-dialogue`
  共用一把信号量,任何一个在渲染时其余全部排队。本次 smoke 前后被 krea2 任务挡过两次。
- **首次调用会因权重冷加载慢一倍**(实测 112s vs 热态 59s)。sidecar 重启后、或被别的
  大模型任务挤掉显存后,第一发都要重新付这笔钱。
- 直接投给 sidecar 的任务(不经 nous)不受这把锁管,会在 ComfyUI 自己的队列里和桥的
  任务混排。
- 时长是主要成本项:`duration_s` 每加 1 秒多 24 帧,两遍采样都要跑。`hires_scale` 直接
  决定第二遍的分辨率,同样线性吃时间和显存。

## 拿不准的点

1. **6 图 / 3 音频的完整语义只能推断**。`MarkdownNote` 只有尺寸表,没有使用说明;
   参考图 4/5/6 与参考音频 3 在保存的这一版里是 bypass 的,连冻结文件名都被
   `convert_ui_to_api` 一起剔除,无从看出作者原本打算给它们什么角色。目前的判断
   ——「素材池 + `<Picture N>`/`<Audio N>` 序号引用,角色分工由提示词声明」—— 来自节点
   tooltip 与自带脚本的写法,没有作者确认。
2. **`<Audio N>` 与 `S1`/`S2` 的对应关系是约定俗成还是硬绑定**,节点 schema 没说。自带
   脚本里 S1 是男声、S2 是女声,而两段参考音频哪段对应谁,提示词文本里没有显式的
   `<Audio 1>` 引用 —— 只有音色的文字描述。`prompt_primary_audio_ordinal=0`(= 关闭)
   也暗示这里没有强制主音源。
3. **是否该暴露 `steps`**。本次按"与 8step turbo LoRA 及 ParityPlan 耦合"判定不暴露,
   如果用户确实想调,需要连 `9.base_steps/coarse_steps/refine_steps` 一起开。
