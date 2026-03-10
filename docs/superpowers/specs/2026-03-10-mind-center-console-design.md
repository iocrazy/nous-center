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
│  /api/v1/engines/*       — 引擎管理                           │
│  /api/v1/tts/synthesize  — 同步合成 (debug)     [NEW]        │
│  /api/v1/tts/generate    — 异步合成 (production) [重构路径]    │
│  /api/v1/tts/batch       — 批量合成 (多角色)     [NEW]        │
│  /api/v1/voices/*        — Voice Preset CRUD    [NEW]        │
│  /api/v1/audio/upload    — 参考音频上传          [NEW]        │
│  /ws/tasks/{task_id}     — 任务状态推送 (per-task)            │
├──────────────────────────────────────────────────────────────┤
│                    Celery Workers                             │
│  TTS Worker (GPU #1)  │  Image Worker (GPU #0)               │
│  Video Worker (GPU #0 + #1)                                  │
├──────────────────────────────────────────────────────────────┤
│  Redis (broker + result)  │  PostgreSQL (presets, history)    │
└──────────────────────────────────────────────────────────────┘
```

### Infrastructure Prerequisites

- **CORS**: FastAPI 需添加 `CORSMiddleware`，允许 console 跨域访问（console 运行在 `localhost:5173`，API 在 `localhost:8000`）
- **路由重构**: 当前后端路由为 `/api/v1/generate/tts`，需重构为 `/api/v1/tts/generate`（按领域分组更清晰）
- **认证**: 个人工作站暂不做认证，后续可加 API key 简单认证

## Engine Registry

6 个 TTS 引擎的标识符（用于 API 请求中的 `engine` 字段）：

| Engine ID | Model | VRAM (from config) |
|-----------|-------|-----|
| `cosyvoice2` | CosyVoice2-0.5B | 3 GB |
| `indextts2` | IndexTTS-2 | 4 GB |
| `qwen3_tts_base` | Qwen3-TTS-1.7B-Base | 4 GB |
| `qwen3_tts_customvoice` | Qwen3-TTS-1.7B-CustomVoice | 4 GB |
| `qwen3_tts_voicedesign` | Qwen3-TTS-1.7B-VoiceDesign | 4 GB |
| `moss_tts` | MOSS-TTS (8B) | 8 GB |

> Note: Qwen3-TTS 三个变体共享权重，同时只能加载一个。MOSS-TTS 的 8GB 是量化后预期值，原始 bfloat16 约 16GB。

## Dual GPU Design

双 RTX 3090 (各 24GB) 并行使用，不同于 ComfyUI 的单卡限制。

### GPU 分配

| GPU | 用途 | 常驻模型 |
|-----|------|---------|
| #0 (24GB) | 图像生成 | SDXL (~10GB) |
| #1 (24GB) | TTS | CosyVoice2 (3GB) — 唯一常驻 |
| #0+#1 | 视频生成 | Wan2.1 (~40GB, exclusive) |

### TTS 模型显存策略

`configs/models.yaml` 为 source of truth。当前只有 cosyvoice2 配置为 `resident: true`，其他引擎按需加载/卸载。

GPU #1 (24GB) 可同时加载的组合示例：
- CosyVoice2 (3GB) + IndexTTS-2 (4GB) + Qwen3-TTS (4GB) = 11GB ✅
- CosyVoice2 (3GB) + MOSS-TTS (16GB bfloat16) = 19GB ✅（紧凑）
- CosyVoice2 (3GB) + MOSS-TTS-4bit (5GB) + IndexTTS-2 (4GB) = 12GB ✅

LRU 调度器根据显存预算自动卸载最久未用的引擎。

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

同步接口，直接返回音频，用于调试和试听。**此接口需新建，当前后端不存在。**

```jsonc
// POST /api/v1/tts/synthesize
{
  "engine": "cosyvoice2",           // required, one of 6 engine IDs
  "text": "你好世界",                // required
  "voice": "default",               // optional, engine-specific
  "speed": 1.0,                     // optional
  "sample_rate": 24000,             // optional
  "reference_audio": "uuid-or-path", // optional, UUID from upload API or server path
  "reference_text": ""              // optional, needed by qwen3_tts_base for voice clone
}
// Response 200:
{
  "audio_base64": "UklGR...",
  "sample_rate": 24000,
  "duration_seconds": 4.83,
  "engine": "cosyvoice2",
  "rtf": 2.82,
  "format": "wav"
}
```

### Production Mode（mediahub 等使用）

异步 Celery 任务，通过 voice_preset 简化调用：

```jsonc
// POST /api/v1/tts/generate
{
  "voice_preset": "主持人-晓晓",     // preset name or ID
  "text": "欢迎收听本期节目"
}
// Response 202:
{ "task_id": "xxx" }
```

### Batch Mode（多角色/播客）

一次提交多段文本 + 角色：

```jsonc
// POST /api/v1/tts/batch
{
  "segments": [
    { "voice_preset": "主持人-晓晓", "text": "欢迎收听本期节目" },
    { "voice_preset": "嘉宾A-男声",  "text": "谢谢邀请" },
    { "voice_preset": "主持人-晓晓", "text": "今天我们聊聊AI" }
  ]
}
// Response 202:
{
  "batch_id": "batch_xxxx",
  "tasks": [
    { "index": 0, "task_id": "task_001" },
    { "index": 1, "task_id": "task_002" },
    { "index": 2, "task_id": "task_003" }
  ]
}

// Query batch status:
// GET /api/v1/tts/batch/{batch_id}
// Response:
{
  "batch_id": "batch_xxxx",
  "status": "partial",  // pending | partial | completed | failed
  "tasks": [
    { "index": 0, "task_id": "task_001", "status": "completed" },
    { "index": 1, "task_id": "task_002", "status": "running" },
    { "index": 2, "task_id": "task_003", "status": "pending" }
  ]
}
```

## Backend API Endpoints

### Engine Management

```
GET    /api/v1/engines
  → [{ "name": "cosyvoice2", "display_name": "CosyVoice2-0.5B",
       "type": "tts", "status": "loaded"|"unloaded",
       "gpu": 1, "vram_gb": 3, "resident": true }]

POST   /api/v1/engines/{name}/load
  → { "name": "cosyvoice2", "status": "loaded", "load_time_seconds": 12.3 }
  → 507 if insufficient VRAM (returns current usage)

POST   /api/v1/engines/{name}/unload
  → { "name": "cosyvoice2", "status": "unloaded" }
  → 409 if engine is resident and force=false
```

### TTS

```
POST   /api/v1/tts/synthesize             — 同步合成 (debug)     [NEW]
POST   /api/v1/tts/generate               — 异步合成 (production) [重构路径]
POST   /api/v1/tts/batch                  — 批量合成 (多角色)     [NEW]
GET    /api/v1/tts/batch/{batch_id}       — 批量状态查询          [NEW]
```

### Voice Presets [NEW]

```
GET    /api/v1/voices                     — 预设列表
POST   /api/v1/voices                     — 创建预设
GET    /api/v1/voices/{id}                — 获取预设详情
PUT    /api/v1/voices/{id}                — 更新预设
DELETE /api/v1/voices/{id}                — 删除预设
GET    /api/v1/voices/groups              — 角色组列表
POST   /api/v1/voices/groups              — 创建角色组
```

### Audio Upload [NEW]

```
POST   /api/v1/audio/upload               — 上传参考音频 (multipart/form-data)
  → { "id": "uuid", "path": "assets/voices/uploads/uuid.wav", "duration": 3.2 }

GET    /api/v1/audio/{id}                 — 获取音频文件信息
```

Console 上传参考音频流程：用户选择文件 → `POST /audio/upload` → 返回 UUID → synthesize 请求中使用 `"reference_audio": "uuid"`

## WebSocket Protocol

连接地址：`WS /ws/tasks/{task_id}`（per-task 订阅，与现有实现一致）

```jsonc
// Server → Client messages:
{ "type": "status", "task_id": "xxx", "status": "running" }
{ "type": "progress", "task_id": "xxx", "progress": 0.5 }
{ "type": "completed", "task_id": "xxx", "result": {
    "audio_url": "/outputs/tts/xxx.wav",
    "duration_seconds": 4.83,
    "rtf": 2.82
  }
}
{ "type": "failed", "task_id": "xxx", "error": "Engine not loaded" }
```

## Voice Preset Data Model

存储在 PostgreSQL：

```yaml
voice_preset:
  id: uuid
  name: "晓晓-温柔"
  engine: "cosyvoice2"              # one of 6 engine IDs
  params:
    voice: "default"
    speed: 1.0
    sample_rate: 24000
  reference_audio_id: uuid | null   # FK to uploaded audio
  reference_text: ""                # optional (qwen3_tts_base needs)
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
- **React Query (TanStack Query)** — server state 管理（引擎状态、预设列表等）
- **Zustand** — 轻量 client state（合成历史、表单状态）
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
│  ┌────────┐  │   Ref Text: [input] (Qwen3 only)    │
│  │晓晓-温柔│  │                                      │
│  │播客主持 │  │   [Synthesize]  [Save as Preset]    │
│  │英文男声 │  │                                      │
│  └────────┘  │   Audio Player + Metadata            │
│  [+ New]     │   (duration, SR, engine, RTF)        │
│              │                                      │
│              │   History (session-local, Zustand)    │
└──────────────┴──────────────────────────────────────┘
```

### Components

| Component | Purpose |
|-----------|---------|
| `EnginePanel` | 引擎列表，显示加载状态 + VRAM，load/unload 操作 |
| `VoicePresetPanel` | 预设列表，点击填充参数到表单 |
| `SynthesizeForm` | 主表单：文本、引擎、参数、参考音频上传、参考文本 |
| `AudioPlayer` | 播放合成结果，显示 duration/SR/RTF 元数据 |
| `HistoryList` | 会话内合成历史（Zustand store），可回放对比 |

## Data Flow

```
Console (React)                    mind-center (FastAPI)
     │                                    │
     │── GET /api/v1/engines ────────────▶│ 引擎列表 + 状态 + VRAM
     │── POST /api/v1/engines/{n}/load ─▶│ 加载模型到 GPU
     │── POST /api/v1/audio/upload ─────▶│ 上传参考音频 → UUID
     │── POST /api/v1/tts/synthesize ───▶│ 同步合成 (debug)
     │◀─ { audio_base64, metadata } ─────│
     │── POST /api/v1/voices ───────────▶│ 保存为预设
     │── GET /api/v1/voices ────────────▶│ 获取预设列表
     │                                    │
     │   Production (mediahub):           │
     │── POST /api/v1/tts/generate ─────▶│ 异步 Celery 任务
     │── POST /api/v1/tts/batch ────────▶│ 批量异步任务
     │── GET /api/v1/tts/batch/{id} ────▶│ 批量状态查询
     │◀─ WS /ws/tasks/{task_id} ────────│ 完成通知
```

## Error Handling

| Scenario | HTTP Status | Response |
|----------|-------------|----------|
| 引擎未加载时 synthesize | 409 Conflict | `{ "detail": "Engine cosyvoice2 not loaded", "hint": "POST /engines/cosyvoice2/load" }` |
| GPU 显存不足时 load | 507 Insufficient Storage | `{ "detail": "Not enough VRAM", "gpu": 1, "available_gb": 2.1, "required_gb": 4.0 }` |
| 合成超时 | 504 Gateway Timeout | `{ "detail": "Synthesis timed out after 30s" }` |
| 无效 voice_preset | 404 Not Found | `{ "detail": "Voice preset not found: xxx" }` |
| 无效 engine name | 422 Unprocessable | `{ "detail": "Unknown engine: xxx", "valid_engines": [...] }` |

## Testing Strategy

- Backend: pytest + httpx AsyncClient，mock GPU 推理
- Frontend: Vitest + React Testing Library
- E2E: 手动测试（单人使用，暂不需要自动化 E2E）
