# Mind Center Console — Design Spec

## Overview

mind-center-console 是一个独立的 React 前端项目，作为 mind-center 后端的开发者控制台。用于模型管理、TTS 调试、Voice Preset 管理等。mind-center 本身是统一的后端 API 服务（FastAPI + Celery + 双 RTX 3090 GPU），为 mediahub 等上层应用提供 AI 能力支撑。

## Architecture

```
┌─────────────────────┐     ┌──────────────────────────────────┐
│ mind-center-console │     │ mediahub / nous / 播客项目         │
│ (React Dev Console) │     │ (上层创作应用)                     │
└────────┬────────────┘     └──────────────┬───────────────────┘
         │ HTTP                            │ HTTP + WebSocket
         ▼                                 ▼
┌──────────────────────────────────────────────────────────────┐
│                    mind-center (FastAPI)                      │
│  /api/v1/engines/*     — 引擎管理                             │
│  /api/v1/tts/synthesize — 同步合成 (debug)                    │
│  /api/v1/tts/generate   — 异步合成 (production)               │
│  /api/v1/tts/batch      — 批量合成 (多角色/播客)               │
│  /api/v1/voices/*       — Voice Preset CRUD                  │
│  /ws/tasks              — 任务状态推送                         │
├──────────────────────────────────────────────────────────────┤
│                    Celery Workers                             │
│  TTS Worker (GPU #1)  │  Image Worker (GPU #0)               │
│  Video Worker (GPU #0 + #1)                                  │
├──────────────────────────────────────────────────────────────┤
│  Redis (broker + result)  │  PostgreSQL (presets, history)    │
└──────────────────────────────────────────────────────────────┘
```

## Dual GPU Design

双 RTX 3090 (各 24GB) 并行使用，不同于 ComfyUI 的单卡限制。

### GPU 分配

| GPU | 用途 | 常驻模型 |
|-----|------|---------|
| #0 (24GB) | 图像生成 | SD/Flux 等 |
| #1 (24GB) | TTS | CosyVoice2 + IndexTTS-2 + Qwen3-TTS |
| #0+#1 | 视频生成 | CogVideoX 等（按需，跨卡） |

### TTS 模型显存策略

小模型常驻 (~10GB)，大模型按需切换：

| 引擎 | 显存 | 策略 |
|------|------|------|
| CosyVoice2 | ~2GB | 常驻 |
| IndexTTS-2 | ~4GB | 常驻 |
| Qwen3-TTS (1.7B) | ~4GB | 常驻 |
| MOSS-TTS (8B) | ~16GB | 按需加载，需卸载其他模型 |

### 资源优化技术

**P0（直接可用）：**
- **CUDA MPS** — 多进程真正共享 GPU SM 核心，利用率翻倍
- **torch.compile** — PyTorch 2.x 编译优化，推理提速 20-40%

**P1（中期）：**
- **MOSS-TTS 4-bit 量化** — 16GB→5GB，可与其他模型共存
- **LRU 模型调度器** — 自动管理显存，最久未用模型自动卸载

**P2（长期）：**
- **ONNX Runtime 导出** — 推理再提速 30%
- **动态 GPU 分配** — 根据任务队列动态调度双卡资源

## Two-Layer API Design

### Debug Mode（Console 使用）

同步接口，直接返回音频，用于调试和试听：

```
POST /api/v1/tts/synthesize
{
  "engine": "cosyvoice2",
  "text": "你好世界",
  "voice": "default",
  "speed": 1.0,
  "sample_rate": 24000,
  "reference_audio": "base64_or_path"
}
→ {
  "audio_base64": "...",
  "sample_rate": 24000,
  "duration_seconds": 4.83,
  "engine": "cosyvoice2",
  "rtf": 2.82
}
```

### Production Mode（mediahub 等使用）

异步 Celery 任务，通过 voice_preset 简化调用：

```
POST /api/v1/tts/generate
{
  "voice_preset": "主持人-晓晓",
  "text": "欢迎收听本期节目"
}
→ { "task_id": "xxx" }
```

### Batch Mode（多角色/播客）

一次提交多段文本 + 角色：

```
POST /api/v1/tts/batch
{
  "segments": [
    { "voice_preset": "主持人-晓晓", "text": "欢迎收听本期节目" },
    { "voice_preset": "嘉宾A-男声",  "text": "谢谢邀请" },
    { "voice_preset": "主持人-晓晓", "text": "今天我们聊聊AI" }
  ]
}
→ {
  "batch_id": "batch_xxxx",
  "tasks": [
    { "index": 0, "task_id": "task_001" },
    { "index": 1, "task_id": "task_002" },
    { "index": 2, "task_id": "task_003" }
  ]
}
```

## Backend API Endpoints

### Engine Management

```
GET    /api/v1/engines                    — 引擎列表 + 加载状态 + 显存占用
POST   /api/v1/engines/{name}/load        — 加载模型到 GPU
POST   /api/v1/engines/{name}/unload      — 卸载模型释放显存
```

### TTS

```
POST   /api/v1/tts/synthesize             — 同步合成 (debug)
POST   /api/v1/tts/generate               — 异步合成 (production)
POST   /api/v1/tts/batch                  — 批量合成 (多角色)
```

### Voice Presets

```
GET    /api/v1/voices                     — 预设列表
POST   /api/v1/voices                     — 创建预设
GET    /api/v1/voices/{id}                — 获取预设详情
PUT    /api/v1/voices/{id}                — 更新预设
DELETE /api/v1/voices/{id}                — 删除预设
GET    /api/v1/voices/groups              — 角色组列表
POST   /api/v1/voices/groups              — 创建角色组
```

## Voice Preset Data Model

```yaml
voice_preset:
  id: uuid
  name: "晓晓-温柔"
  engine: "cosyvoice2"
  params:
    voice: "default"
    speed: 1.0
    sample_rate: 24000
  reference_audio_path: "assets/voices/xiaoxiao.wav"  # optional
  reference_text: ""                                   # optional (Qwen3 needs)
  tags: ["中文", "女声", "温柔"]
  created_at: datetime
  updated_at: datetime

voice_preset_group:
  id: uuid
  name: "科技播客-三人组"
  presets:
    - role: "主持人"
      voice_preset_id: uuid
    - role: "嘉宾A"
      voice_preset_id: uuid
    - role: "嘉宾B"
      voice_preset_id: uuid
```

## Console Frontend

### Tech Stack

- Vite + React + TypeScript
- TailwindCSS
- Independent project: `mind-center-console`

### Priority

1. TTS Playground（调试）
2. Voice Preset 管理
3. Model Dashboard（引擎状态/显存）
4. Image/Video Debugging（后续）

### TTS Playground Layout

```
┌─────────────────────────────────────────────────────┐
│  mind-center-console            [Models] [TTS] ...  │
├──────────────┬──────────────────────────────────────┤
│  Engine List │   TTS Playground                     │
│  ┌────────┐  │   ┌──────────────────────────────┐   │
│  │cosyv…✅│  │   │ Text Input (textarea)        │   │
│  │index…✅│  │   └──────────────────────────────┘   │
│  │qwen3…⬚│  │                                      │
│  │moss… ⬚│  │   Engine: [dropdown]                 │
│  └────────┘  │   Voice:  [dropdown]                 │
│              │   Speed:  [slider] 1.0               │
│  Voice       │   Sample Rate: [dropdown]             │
│  Presets     │   Ref Audio: [upload/select]          │
│  ┌────────┐  │                                      │
│  │晓晓-温柔│  │   [Synthesize]  [Save as Preset]    │
│  │播客主持 │  │                                      │
│  │英文男声 │  │   Audio Player + Metadata            │
│  └────────┘  │   (duration, SR, engine, RTF)        │
│  [+ New]     │                                      │
│              │   History (session-local)              │
└──────────────┴──────────────────────────────────────┘
```

### Components

| Component | Purpose |
|-----------|---------|
| `EnginePanel` | 引擎列表，显示加载状态，load/unload 操作 |
| `VoicePresetPanel` | 预设列表，点击填充参数 |
| `SynthesizeForm` | 主表单：文本、引擎、参数、参考音频 |
| `AudioPlayer` | 播放合成结果，显示元数据 |
| `HistoryList` | 会话内合成历史，可回放对比 |

## Data Flow

```
Console (React)                    mind-center (FastAPI)
     │                                    │
     │── GET /api/v1/engines ────────────▶│ 引擎列表 + 状态
     │── POST /api/v1/engines/{n}/load ─▶│ 加载模型
     │── POST /api/v1/tts/synthesize ───▶│ 同步合成
     │◀─ { audio_base64, metadata } ─────│
     │── POST /api/v1/voices ───────────▶│ 保存预设
     │── GET /api/v1/voices ────────────▶│ 获取预设
     │                                    │
     │   Production (mediahub):           │
     │── POST /api/v1/tts/generate ─────▶│ 异步 Celery
     │── POST /api/v1/tts/batch ────────▶│ 批量异步
     │◀─ WS /ws/tasks ──────────────────│ 完成通知
```

## Error Handling

- 引擎未加载时调用 synthesize → 返回 HTTP 409 + 提示先 load
- GPU 显存不足时 load → 返回 HTTP 507 + 当前显存使用详情
- 合成超时 → 30s timeout，返回 HTTP 504
- 无效 voice_preset → 返回 HTTP 404

## Testing Strategy

- Backend: pytest + httpx AsyncClient，mock GPU 推理
- Frontend: Vitest + React Testing Library
- E2E: 手动测试（单人使用，暂不需要自动化 E2E）
