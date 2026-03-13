# 火山引擎语音服务 API 设计参考

> 来源：火山引擎「开发参考」文档全量阅读（2026-03）
> 用途：为 mind-center 设计提供借鉴

---

## 模块总览

| 模块 | 协议 | 场景 | 核心特点 |
|------|------|------|----------|
| 双向流式 TTS (WebSocket V3) | WS 二进制帧 | 实时对话、低延迟合成 | 4字节头 + 连接/会话两级生命周期 |
| HTTP 流式 TTS (Chunked/SSE) | HTTP | 简单集成、单次合成 | base64 音频块，SSE event-stream |
| 精品长文本 TTS | HTTP 异步 | 长篇有声读物、播报 | submit/query，最大10万字 |
| 播客生成 (Podcast V3) | WS 二进制帧 | 双人播客、多轮对话 | Round 事件模型，断点重试 |
| 同声传译 2.0 | WS + Protobuf | 实时口译、会议翻译 | 三路并行事件流(ASR/翻译/TTS) |
| 流式语音识别 (ASR) | WS 二进制帧 | 实时转写 | 与 TTS 共享帧协议 |
| 豆包语音妙记 | HTTP 异步 | 会议记录、音频分析 | 转写+翻译+摘要+章节提取 |
| 音视频字幕生成 | HTTP 异步 | 视频字幕 | 词级时间戳 |
| 声音复刻 | HTTP | 定制音色 | 一次训练→多模型，状态机管理 |

---

## 核心设计模式

### 1. 统一二进制 WebSocket 帧协议

所有实时 WS 服务共享同一个 4 字节头格式：

```
Byte 0: [protocol_version:4][header_size:4]
Byte 1: [message_type:4][type_flags:4]
Byte 2: [serialization:4][compression:4]
Byte 3: [reserved:8]
```

**字段含义：**
- `message_type`: 0b0001=full client request, 0b1001=full server response, 0b1011=server ACK, 0b1111=server error
- `type_flags`: 0b0001=无序列号, 0b0010=正序列号(POS_SEQ), 0b0011=负序列号(NEG_SEQ, 表示最后一包)
- `serialization`: 0b0000=无, 0b0001=JSON, 0b0010=Thrift, 0b1111=自定义
- `compression`: 0b0000=无, 0b0001=gzip, 0b0010=自定义

**头部之后的 payload 结构：**
```
[header 4B] [sequence:4B if POS/NEG_SEQ] [payload_size:4B] [payload]
```

**启发：** mind-center 的 WebSocket 目前是纯 JSON，切换到二进制帧可以减少解析开销，且音频数据不需要 base64 编码。

---

### 2. 连接/会话两级生命周期

```
StartConnection
  ├─ StartSession (session 1)
  │   ├─ TaskRequest (text chunk 1)
  │   ├─ TaskRequest (text chunk 2)
  │   └─ FinishSession
  ├─ StartSession (session 2)
  │   ├─ TaskRequest ...
  │   └─ FinishSession
  └─ FinishConnection
```

- 一个 WS 连接可复用多个串行会话
- 鉴权仅在 `StartConnection` 做一次
- 每个 session 可独立配置音色、参数
- 支持 `FinishSession` 后立刻 `StartSession` 切换上下文

**启发：** mind-center 当前每次合成都建新连接。可以改为连接复用，降低握手开销，特别是批量合成场景。

---

### 3. 播客 Round 事件模型

```
Client: PodcastStart (input_text/url/dialog_list, speakers)
Server: PodcastRoundStart (speaker_id, round_id)
Server: PodcastRoundResponse (audio_data) × N
Server: PodcastRoundEnd (round_id)
... 重复多轮 ...
Server: PodcastFinish
```

**4种输入模式：**
1. `input_text` — 长文本自动拆分角色
2. `input_url` / `input_file` — URL 或 PDF 自动提取内容
3. `dialog_text_list` — `[{"speaker":0,"text":"..."},...]` 精确控制对话
4. `prompt` — 仅给主题，AI 自动展开

**断点重试：** 失败时返回 `failed_round_id`，客户端可发 `PodcastRetry(retry_round_id)` 从该轮重新生成。

**启发：** mind-center 的批量合成可以借鉴 Round 模型：每个角色的每段台词作为一个 Round，支持单轮重试而非整体重做。

---

### 4. 同声传译三路并行事件流

```
事件 ID 范围:
  源语言 ASR:    650 (start) → 651 (sentence) → 652 (finish)
  翻译结果:      653 (start) → 654 (sentence) → 655 (finish)
  TTS 合成音频:  350 (start) → 351 (sentence) → 352 (finish)
  音频静音通知:  250 (AudioMuted)
```

三路事件独立并行推送：
- ASR 实时输出源语言转写
- 翻译实时输出目标语言文本
- TTS 实时输出翻译后的音频

额外能力：
- `spk_chg` 字段检测说话人切换
- `AudioMuted` 事件通知客户端静音（避免啸叫）
- 支持 S2S（语音到语音）和 S2T（语音到文本）两种模式

**启发：** mind-center 的多步骤流水线（ASR → 处理 → TTS）可以借鉴这种并行事件流模型，让中间结果尽早可见。

---

### 5. 异步任务 submit/query 模式

用于长时间任务（妙记、字幕、长文本TTS）：

```
POST /submit
  Body: { audio_url, language, ... }
  Response: { task_id: "xxx" }

POST /query
  Body: { task_id: "xxx" }
  Response: { status: "running|success|failed", result: {...} }
```

**增强特性：**
- `callback_url` — 任务完成后主动回调
- 长文本 TTS 的 query 返回分片信息：`{ splits: [{text, audio_url, silence_duration}, ...] }`
- 妙记返回结构化结果：`{ transcription, translation, summary, chapters }`

**启发：** mind-center 已有 Celery task，但缺少统一的任务查询 API。可以统一为 submit/query 模式，并增加 callback 支持。

---

### 6. TTS 2.0 自然语言情感控制

**两种方式：**

#### a) context_texts（上下文指令）
```json
{
  "context_texts": ["用痛心的语气说话"],
  "text": "我再也不想看到这种事情了。"
}
```
- 支持中英文自然语言描述
- 最多 30 轮上下文 / 10 分钟有效

#### b) CoT 内联标签
```
<cot text=急促难耐>快点快点，来不及了！</cot>
<cot text=温柔关切>没关系，慢慢来。</cot>
```
- 可以对同一段文字中不同句子使用不同情感
- 支持语速、音量等控制

#### c) section_id 跨会话上下文
```json
{
  "section_id": "chapter-3",
  "context_texts": ["用激动的语气说话"]
}
```
同一个 `section_id` 的请求共享情感上下文（30轮 / 10分钟内有效）。

**启发：** mind-center 可以在节点编辑器中增加「情感」属性，支持自然语言描述情感风格，而非传统的枚举选项。

---

### 7. 声音复刻多模型训练

一次声音复刻训练会产出多个模型变体：

| 模型类型 | 说明 | 适用场景 |
|----------|------|----------|
| ICL 1.0 | 基础克隆 | 快速原型 |
| DiT 标准 | 高质量 | 正式内容生产 |
| DiT 修复 | 含降噪 | 嘈杂录音素材 |
| ICL 2.0 | 最新一代 | 最佳效果 |

**状态机：**
```
NotFound → Training → Success/Failed → Active
```

**启发：** mind-center 的音色管理可以支持「一次上传，多种模型」的概念，让用户选择质量/速度的平衡点。

---

### 8. 音色混合（Voice Mixing）

```json
{
  "voice_type": "BV001_V2",
  "mix_voice_list": [
    { "voice_type": "BV700_V2", "mix_factor": 0.3 },
    { "voice_type": "BV406_V2", "mix_factor": 0.2 }
  ]
}
```

- 最多混合 3 个音色
- `mix_factor` 权重之和为 1.0（主音色权重 = 1 - Σmix_factor）
- 可用于创建新音色风格

**启发：** mind-center 可以在 Voice Presets 中增加混合功能，让用户通过调节滑块混合多个参考音色。

---

### 9. 多粒度字幕与情感预测

```json
{
  "enable_subtitle": 3,  // 0=关, 1=句级, 2=词级, 3=音素级
  "emotion_prediction": true
}
```

**字幕输出结构：**
```json
{
  "words": [
    {
      "text": "你好",
      "start_time": 0,
      "end_time": 320,
      "phonemes": [
        { "phone": "n", "start_time": 0, "end_time": 80 },
        { "phone": "i3", "start_time": 80, "end_time": 160 },
        ...
      ]
    }
  ],
  "emotion": "happy"
}
```

**启发：** mind-center 的输出节点可以增加字幕/时间戳显示，支持词级高亮同步。

---

### 10. 用量计量与缓存

#### 用量统计
请求头加 `X-Control-Require-Usage-Tokens-Return: *`，响应中返回：
```json
{
  "usage": {
    "input_text_tokens": 42,
    "output_audio_tokens": 1280,
    "input_audio_tokens": 0
  }
}
```

#### 结果缓存
```json
{
  "cache_config": {
    "use_cache": true
  }
}
```
- 相同文本 + 相同参数 → 返回缓存音频（1小时 TTL）
- 减少重复合成开销

#### AIGC 水印
- 音频水印：可溯源内容来源
- Header 元数据：`X-AIGC-*` 系列 header

**启发：** mind-center 可以加入 token 用量统计面板（Dashboard），以及合成结果缓存（Redis），避免重复合成相同内容。

---

## 对 Mind-Center 的改进建议（按优先级）

### 高优先级

| # | 建议 | 参考来源 | 实现思路 |
|---|------|----------|----------|
| 1 | **统一异步任务 API** | 妙记/字幕/长文本 TTS | 所有长时任务统一为 `POST /submit` + `POST /query`，返回标准 `{task_id, status, result}` |
| 2 | **WebSocket 连接复用** | 连接/会话两级模型 | 保持 WS 连接，多次合成复用同一连接，减少握手开销 |
| 3 | **情感控制节点** | TTS 2.0 context_texts | 节点编辑器增加「Emotion」属性，支持自然语言描述情感 |

### 中优先级

| # | 建议 | 参考来源 | 实现思路 |
|---|------|----------|----------|
| 4 | **批量合成 Round 模型** | 播客 Round 事件 | 多角色台词逐轮合成，支持单轮重试 |
| 5 | **用量统计面板** | UsageResponse | Dashboard 增加 token/字符用量图表 |
| 6 | **合成结果缓存** | cache_config | Redis 缓存相同参数的合成结果，1h TTL |
| 7 | **音色混合** | mix_voice_list | Voice Presets 增加多音色混合滑块 |

### 低优先级

| # | 建议 | 参考来源 | 实现思路 |
|---|------|----------|----------|
| 8 | **二进制 WS 帧** | 统一帧协议 | 将 JSON WS 升级为二进制帧，音频不需 base64 |
| 9 | **词级字幕/时间戳** | 多粒度字幕 | OutputNode 支持词级高亮播放 |
| 10 | **Callback 通知** | submit/query + callback | 长时任务完成后主动通知前端（WS push 或 webhook） |

---

## TTS 参数完整参考（WebSocket V3）

从火山引擎文档提取的完整参数列表，供 mind-center API 设计参考：

```json
{
  "app": {
    "appid": "string",
    "token": "string",
    "cluster": "string"
  },
  "user": {
    "uid": "string"
  },
  "audio": {
    "encoding": "pcm|ogg_opus|mp3|wav",
    "voice_type": "string",
    "speed_ratio": 1.0,
    "volume_ratio": 1.0,
    "pitch_ratio": 1.0,
    "sample_rate": 24000,
    "bits": 16,
    "channel": 1,
    "emotion": "string",
    "language": "zh|en|ja|...",
    "enable_timestamp": true,
    "enable_subtitle": 0
  },
  "request": {
    "reqid": "uuid",
    "text": "string",
    "text_type": "plain|ssml",
    "operation": "submit|query",
    "silence_duration": 125,
    "with_frontend": true,
    "frontend_type": "unitTson",
    "pure_english_opt": 1,
    "context_texts": ["string"],
    "section_id": "string",
    "cache_config": { "use_cache": true }
  }
}
```
