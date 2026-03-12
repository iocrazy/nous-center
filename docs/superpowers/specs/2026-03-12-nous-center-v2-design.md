# Nous Center V2 — 全栈增强设计文档

> 日期：2026-03-12
> 状态：已确认
> 参考：`docs/volcengine-tts-reference.md`（火山引擎 API 设计借鉴）

---

## 1. 定位

Nous Center 是 AI 综合媒体工作台，支持 TTS 语音合成、图像生成、视频生成、多模态理解。V2 在已完成的后端 API 基础上，借鉴火山引擎 TTS 服务的设计模式，进行全栈增强。

## 2. Monorepo 架构

全局单一 Git 仓库，三个子项目各自管理依赖：

```
nous-center/
├── .gitignore                  # 全局 gitignore
├── backend/                    # Python — AI 推理 + 业务 API + 任务队列
│   ├── pyproject.toml          # uv 管理依赖
│   ├── uv.lock
│   ├── .python-version
│   ├── .env.example
│   ├── configs/
│   ├── scripts/
│   ├── src/
│   └── tests/
├── frontend/                   # React — 节点编辑器 + 开发者控制台
│   ├── package.json
│   ├── vite.config.ts
│   ├── tsconfig.json
│   ├── wasm/                   # Rust WASM 音频引擎
│   └── src/
├── nous-core/                  # Rust — 系统监控 + 音频 IO + 文件操作
│   ├── Cargo.toml
│   └── src/
├── assets/
│   └── voices/
├── docs/
└── scripts/                    # 全局启动/部署脚本
    ├── dev.sh
    └── deploy.sh
```

**Git 规范：**
- 每个子项目的 `.venv`、`node_modules`、`target/` 在全局 `.gitignore` 中
- `checkpoints/` 不进仓库
- Commit message 前缀：`backend:` / `frontend:` / `core:` / `docs:`

## 3. 服务间通信

```
                    用户浏览器
                        │
                        ▼
              ┌─────────────────┐
              │   frontend       │  :5173
              │   React + WASM   │
              └───┬─────────┬───┘
          HTTP/WS │         │ HTTP
                  ▼         ▼
        ┌──────────┐  ┌──────────┐
        │ backend  │  │nous-core │
        │ FastAPI  │  │  Axum    │
        │ :8000    │  │  :8001   │
        └────┬─────┘  └──────────┘
             │
     ┌───────┼────────┐
     ▼       ▼        ▼
   Redis   PostgreSQL  GPU Workers
   (队列+缓存) (持久化)  (Celery)
```

**调用关系：**
- 前端 → backend：所有业务 API（TTS、图像、任务管理、Voice Preset）
- 前端 → nous-core：系统监控（GPU 状态、磁盘、进程）— 轻量直连
- backend → nous-core：音频文件处理（重采样、格式转换、拼接）
- backend → Redis：Celery 任务队列 + 合成结果缓存
- backend → PostgreSQL：任务记录、Voice Preset、用量统计

## 4. TTS 增强设计（火山引擎借鉴）

### 4.1 SSE 流式合成

新增端点，引擎逐句生成，每句立即推送，前端边生成边播放：

```
GET /api/v1/tts/stream?text=...&engine=...&voice_preset=...

Response: text/event-stream
  event: audio
  data: {"seq": 1, "audio": "<base64 chunk>", "format": "wav"}

  event: audio
  data: {"seq": 2, "audio": "<base64 chunk>", "format": "wav"}

  event: done
  data: {"total_chunks": 2, "duration_ms": 3200, "usage": {"characters": 42, "rtf": 0.8}}
```

- 最后 `done` 事件附带用量统计（借鉴火山 UsageResponse）
- 不支持流式的引擎降级为单次返回

**来源：** 火山 HTTP 流式 TTS (Chunked/SSE)

### 4.2 WebSocket 会话级连接复用

当前 `/ws/tasks/{task_id}` 每任务一连接。改为会话级复用：

```
WS /ws/tts

客户端：
  {"type": "start_session", "session_id": "s1", "engine": "cosyvoice2", "voice_preset": "xiaoming"}
  {"type": "synthesize", "session_id": "s1", "text": "第一句台词"}
  {"type": "synthesize", "session_id": "s1", "text": "第二句台词"}
  {"type": "end_session", "session_id": "s1"}
  // 同一连接开新会话，切换音色
  {"type": "start_session", "session_id": "s2", "engine": "qwen3_tts", "voice_preset": "xiaohong"}

服务端：
  {"type": "audio", "session_id": "s1", "seq": 1, "audio": "..."}
  {"type": "session_ended", "session_id": "s1"}
```

- 一个 WS 连接复用多个串行会话
- 每个 session 独立配置引擎和音色
- 适合前端节点编辑器连续调试场景

**来源：** 火山连接/会话两级生命周期

### 4.3 批量合成 Round 模型

将批量合成从 N 个独立任务改为 Round 事件模型：

```
POST /api/v1/tts/batch
{
  "rounds": [
    {"round_id": 1, "voice_preset": "xiaoming", "text": "你好啊"},
    {"round_id": 2, "voice_preset": "xiaohong", "text": "你好"},
    {"round_id": 3, "voice_preset": "xiaoming", "text": "今天天气不错"}
  ]
}
→ 返回 batch_id

WS 推送：
  {"type": "round_start", "batch_id": "...", "round_id": 1}
  {"type": "round_audio", "round_id": 1, "audio": "..."}
  {"type": "round_end",   "round_id": 1}
  ...
  {"type": "batch_done",  "batch_id": "...", "total_rounds": 3}
```

**单轮重试：**
```
POST /api/v1/tts/batch/{batch_id}/retry
{"round_ids": [2]}
```

**来源：** 火山播客 Round 事件模型 + PodcastRetry

### 4.4 结果缓存

对 `(text + engine + voice_preset + params)` 做哈希，命中 Redis 缓存直接返回：

- 请求字段：`"cache": true/false`
- 响应头：`X-Cache: HIT/MISS`
- TTL 1 小时，可配置
- 调试时可传 `"cache": false` 强制重新合成

**来源：** 火山 cache_config

### 4.5 情感控制

合成请求新增可选字段：

```json
{
  "text": "我再也不想看到这种事情了",
  "emotion": "用痛心的语气说话",
  "engine": "qwen3_tts_base"
}
```

- `emotion` 为自然语言字符串，透传给支持情感控制的引擎
- 不支持的引擎忽略此字段

**来源：** 火山 TTS 2.0 context_texts

### 4.6 用量统计

每次合成记录到 PostgreSQL：
- 输入字符数
- 输出音频时长（秒）
- RTF（实时率 = 合成耗时 / 音频时长）
- 使用的引擎
- 时间戳

前端 Dashboard overlay 展示图表。

**来源：** 火山 UsageResponse

## 5. nous-core 职责

### 5.1 现有功能

```
GET  /sys/gpus              GPU 状态（NVML）
GET  /sys/stats             CPU/内存（sysinfo）
GET  /sys/processes         进程列表
GET  /sys/models            模型目录扫描
```

### 5.2 新增音频 IO

```
POST /audio/info            读取音频元信息（时长、采样率、通道数）
POST /audio/resample        重采样（改采样率/位深）
POST /audio/convert         格式转换（wav/mp3/ogg/flac 互转）
POST /audio/concat          多段音频拼接
POST /audio/split           按时间点切割
```

**技术依赖：** `symphonia`（解码）+ `hound`（WAV 编码），已在 Cargo.toml 中。

**选择 Rust 的理由：**
- 比 Python `pydub`/`soundfile` 快 5-10x
- 不占 Python GIL，不影响 FastAPI 并发
- 与前端 WASM 音频引擎共享 Rust 生态

**backend 封装：**
```python
# backend/src/services/audio_io.py
async def resample(file_path: str, target_sr: int) -> str:
    resp = await httpx.post("http://localhost:8001/audio/resample", json={...})
    return resp.json()["output_path"]
```

## 6. 前端补完

### 6.1 已完成

- 节点编辑器（8 种节点：文本输入、参考音频、TTS 引擎、重采样、混音、拼接、BGM 混合、输出）
- Layout（IconRail + Topbar + WorkflowTabs）
- 侧面板（ApiNodes、Collections、NodeLibrary、Presets、Workflows）
- Overlay（Dashboard、Models、Settings）
- 状态管理（Zustand：workspace、execution、history、panel、settings、theme、toast）
- API 层（client、tts、engines、voices、settings、system、websocket）
- WASM 音频引擎
- WavePlayer 组件

### 6.2 需要补完

| 模块 | 说明 |
|------|------|
| 节点执行引擎 | 拓扑排序 + 按序执行节点图，点「运行」真正工作 |
| 流式播放 | WavePlayer 对接 SSE 逐块接收边播放 |
| WS 连接管理 | websocket.ts 改为会话级复用 |
| Round 进度面板 | 批量合成每轮状态展示 + 单轮重试按钮 |
| 情感控制 UI | TTSEngine 节点增加 emotion 文本输入框 |
| 缓存指示器 | 合成结果旁 HIT/MISS 标记 |
| 用量 Dashboard | 统计图表（字符数、音频时长、RTF 趋势） |
| 系统监控面板 | 对接 nous-core `/sys/*` 实时 GPU/内存图表 |

### 6.3 节点执行引擎

```
用户点击「运行」
    ↓
拓扑排序所有节点
    ↓
按序执行：
    TextInput  → 输出 text
    RefAudio   → 输出 audio_path
    TTSEngine  → 调 /tts/stream，输出音频块
    Resample   → 调 nous-core /audio/resample
    Concat     → 调 nous-core /audio/concat
    Mixer      → WASM 本地处理
    BGMix      → WASM 本地混合
    Output     → WavePlayer 播放
```

**处理策略：**
- 轻量操作（混音、BGM）→ WASM 本地执行
- IO 密集操作（重采样、拼接、格式转换）→ nous-core
- AI 推理（TTS）→ backend

## 7. 相比 V1 的关键变化汇总

| 项 | V1 | V2 | 来源 |
|---|-----|-----|------|
| Git 结构 | 单体 | monorepo（backend + frontend + nous-core） | — |
| Rust 服务 | nous-center-sys（仅监控） | nous-core（监控 + 音频 IO） | 性能需求 |
| TTS 合成 | 仅同步/异步 | +SSE 流式 | 火山 HTTP 流式 |
| WebSocket | 每任务新建 | 会话级连接复用 | 火山两级生命周期 |
| 批量合成 | N 个独立 task | Round 模型 + 单轮重试 | 火山播客 Round |
| 缓存 | 无 | Redis 缓存，1h TTL | 火山 cache_config |
| 情感控制 | 无 | 自然语言 emotion 字段 | 火山 context_texts |
| 用量统计 | 无 | 字符数/时长/RTF 记录 + Dashboard | 火山 UsageResponse |
| 节点编辑器 | 仅连线 | 完整执行引擎 | — |

## 8. 实施阶段

### Phase 1：基础整固（Monorepo + 修复）
1. 完成 monorepo 重构，提交所有 rename
2. `nous-center-sys` → `nous-core` 重命名
3. 全局 `.gitignore` 统一
4. `scripts/dev.sh` 一键启动
5. 后端现有 API 测试补全

### Phase 2：TTS 增强（火山引擎借鉴）
6. 结果缓存 — Redis hash + `X-Cache`
7. 用量统计 — 合成记录 + 数据模型
8. SSE 流式合成 `/tts/stream`
9. 情感控制 `emotion` 字段
10. 批量合成 Round 模型 + 单轮重试
11. WebSocket 会话级连接复用 `/ws/tts`

### Phase 3：nous-core 扩展
12. `/audio/info` — 音频元信息
13. `/audio/resample` — 重采样
14. `/audio/convert` — 格式转换
15. `/audio/concat` — 拼接
16. `/audio/split` — 切割
17. backend 封装 `audio_io` 服务层

### Phase 4：前端补完
18. 节点执行引擎 — 拓扑排序 + 按序执行
19. 流式播放 — WavePlayer 对接 SSE
20. WS 连接复用 — websocket.ts 改造
21. Round 进度面板 + 重试按钮
22. 情感控制 UI
23. 缓存指示器
24. 用量 Dashboard
25. 系统监控面板
