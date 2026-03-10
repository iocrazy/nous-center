# Mind Center - AI 综合媒体工作台设计文档

## 定位

对外 API + 前端界面的综合媒体工作台，支持图像生成、视频生成、语音合成、多模态理解。

## 基础设施

| 设备 | 角色 | 说明 |
|------|------|------|
| Ubuntu 双3090 | 推理层 | 跑所有 AI 模型 |
| Mac | 协调层 | Redis、PostgreSQL、Nginx |
| NAS | 存储层 | 模型权重、生成文件 |

## 架构

```
┌─────────────────────────────────────────────────────┐
│  Mac (协调层)                                        │
│  ├── Redis (任务队列 + 结果缓存)                     │
│  ├── PostgreSQL (任务记录、用户数据)                  │
│  ├── Celery Beat (定时任务)                          │
│  └── Nginx (反向代理 → Ubuntu FastAPI)               │
└──────────────────┬──────────────────────────────────┘
                   │ 局域网
┌──────────────────▼──────────────────────────────────┐
│  Ubuntu 双3090 (推理层)                              │
│  ├── FastAPI (任务接收 + 状态查询 + WebSocket)       │
│  ├── vLLM Server (GPU1) → Qwen2.5-VL / Qwen TTS    │
│  ├── Celery Worker (GPU0) → diffusers 常驻           │
│  ├── Celery Worker (GPU1) → CosyVoice2              │
│  └── Celery Worker (动态) → Wan2.1 独占双卡          │
└──────────────────┬──────────────────────────────────┘
                   │ 挂载/SMB
┌──────────────────▼──────────────────────────────────┐
│  NAS                                                 │
│  ├── /models  (模型权重，Ubuntu 只读挂载)             │
│  └── /outputs (生成结果，统一存放)                    │
└─────────────────────────────────────────────────────┘
```

## 模型与 GPU 分配

### 推理引擎分工

| 引擎 | 模型 | 说明 |
|------|------|------|
| vLLM | Qwen2.5-VL, Qwen TTS | LLM/VLM 类，自带排队和批处理 |
| PyTorch 直接加载 | diffusers, Wan2.1, CosyVoice2 | Diffusion/音频模型 |

### GPU 分配

| 级别 | 模型 | GPU | 显存 | 策略 |
|------|------|-----|------|------|
| 常驻 | diffusers (SDXL) | GPU0 | ~10GB | 永不卸载 |
| 常驻 | CosyVoice2 + Qwen TTS | GPU1 | ~6GB | 轻量共存 |
| 按需 | Qwen2.5-VL 7B | GPU1 | ~16GB | 按需加载，空闲释放 |
| 独占 | Wan2.1 视频 | GPU0+GPU1 | ~40GB | 卸载所有模型，双卡推理 |

### 调度逻辑

```
收到任务 → 检查目标模型是否已加载
  ├── 已加载 → 直接执行
  ├── 未加载，显存够 → 加载模型 → 执行
  └── 未加载，显存不够 → 排队等待 / 卸载低优先级模型 → 加载 → 执行
```

Wan2.1 视频特殊处理：需独占双卡，触发前卸载所有模型，完成后恢复常驻模型。

## API 设计

### 端点

```
POST /api/v1/generate/image      — 文生图 (diffusers)
POST /api/v1/generate/video      — 文生视频 (Wan2.1)
POST /api/v1/generate/tts        — 文字转语音 (CosyVoice2 / Qwen TTS)
POST /api/v1/understand/image    — 图片理解 (Qwen2.5-VL)

GET  /api/v1/tasks/{task_id}     — 查询任务状态+结果
GET  /api/v1/tasks               — 任务列表
DELETE /api/v1/tasks/{task_id}   — 取消任务

GET  /api/v1/models              — 查看可用模型 & GPU状态
```

### 任务流程

```
用户请求 → Nginx(Mac) → FastAPI(Ubuntu) → Celery Task → Redis(Mac)
                                              ↓
                                         GPU Worker 执行
                                              ↓
                                    结果存 NAS + 状态写 PostgreSQL
                                              ↓
                              WebSocket 推送完成通知
```

- 文本类任务（understand）直接调 vLLM，不走 Celery
- 所有任务统一 WebSocket 推送，保留 GET 轮询作为 fallback

### 任务状态

`pending → running → completed / failed`

### 结果存储

| 类型 | 存储位置 |
|------|----------|
| 文件（图片/视频/音频） | NAS，数据库存文件路径 |
| 文本（理解/描述） | PostgreSQL |
| 任务元数据 | PostgreSQL |

## Celery 队列

| 队列 | 并发 | GPU | 说明 |
|------|------|-----|------|
| image | 1 | GPU0 | diffusers，对用户异步，GPU 串行防 OOM |
| tts | 1 | GPU1 | CosyVoice2/Qwen TTS |
| video | 1 | GPU0+1 | Wan2.1 独占，需先卸载其他模型 |
| understand | — | — | 不走 Celery，直接 HTTP 调 vLLM |

并发 1 = GPU 同一时间只跑一个同类任务（防显存溢出），但对用户是异步非阻塞的。不同类型任务在不同 GPU 上可并行。

## 技术栈

| 层 | 技术 |
|---|------|
| 包管理 / 虚拟环境 | uv |
| API 网关 | FastAPI + Uvicorn |
| 任务队列 | Celery + Redis |
| 数据库 | PostgreSQL |
| LLM 推理 | vLLM |
| 图像/视频/音频 | PyTorch 直接加载 |
| WebSocket | FastAPI WebSocket |
| 文件存储 | NAS (SMB/NFS 挂载) |
| 容器化 | Docker Compose (可选) |
| 前端 | 后续规划 (Next.js / React) |

## 项目结构

```
mind-center/
├── docker-compose.yml
├── pyproject.toml              # uv 管理依赖
├── uv.lock
├── .env                        # Redis地址、NAS路径、GPU分配
├── .python-version
│
├── src/
│   ├── api/                    # FastAPI 网关
│   │   ├── main.py
│   │   ├── routes/
│   │   │   ├── generate.py     # 图像/视频/TTS 生成端点
│   │   │   ├── understand.py   # 多模态理解端点 (→ vLLM)
│   │   │   └── tasks.py        # 任务查询/取消
│   │   └── websocket.py        # WebSocket 推送
│   │
│   ├── workers/                # Celery Workers
│   │   ├── image_worker.py     # diffusers
│   │   ├── video_worker.py     # Wan2.1
│   │   ├── tts_worker.py       # CosyVoice2 / Qwen TTS
│   │   └── celery_app.py       # Celery 配置
│   │
│   ├── gpu/                    # GPU 管理
│   │   ├── model_manager.py    # 模型加载/卸载/调度
│   │   └── vram_tracker.py     # 显存监控
│   │
│   ├── storage/                # 文件存储
│   │   └── nas.py              # NAS 读写
│   │
│   └── models/                 # 数据模型
│       ├── task.py             # 任务 ORM
│       └── schemas.py          # Pydantic 请求/响应
│
├── configs/
│   ├── models.yaml             # 模型配置（路径、GPU、显存）
│   └── vllm.yaml               # vLLM 启动配置
│
└── scripts/
    ├── start_vllm.sh           # 启动 vLLM 服务
    └── start_workers.sh        # 启动 Celery Workers
```

## 启动流程

### Mac 端

```bash
brew services start redis
brew services start postgresql
```

### Ubuntu 端

```bash
# 1. 挂载 NAS
mount -t cifs //nas/media /mnt/nas

# 2. 启动 vLLM (GPU1)
CUDA_VISIBLE_DEVICES=1 vllm serve Qwen2.5-VL-7B --port 8100

# 3. 启动 Celery Workers
CUDA_VISIBLE_DEVICES=0 celery -A src.workers.celery_app worker --queue=image,tts
celery -A src.workers.celery_app worker --queue=video --concurrency=1

# 4. 启动 FastAPI
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```
