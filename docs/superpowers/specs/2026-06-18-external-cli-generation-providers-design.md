# 外部 CLI 生成 provider 子系统(codex / 即梦 dreamina)— 账号登录态 CLI 转发,带并发节流护栏

- Date: 2026-06-18
- Status: 设计
- Trigger: 用户要把 Infinite-Canvas 的即梦 CLI、T8-penguin-canvas 的 codex CLI 抽成 nous-center 的「runner」,带并发/节流护栏,经 `/v1/images/generations` 给自己另一个平台(走 ZeroTier 内网)调用。单人自用,非对外多租户。

## 1. 背景 / 现状

两个外部仓库各有一套「账号登录态 CLI → subprocess 转发」的生图能力:

- **即梦(dreamina)**:`/media/heygo/.../github-repos/Infinite-Canvas/main.py`,`run_jimeng_cli`(L3871)+ `jimeng_submit_id`/`jimeng_queue_info`/`jimeng_collect_media_values`。Python `asyncio.create_subprocess_exec` 起 `dreamina` 二进制,提交拿 `submit_id` → 轮询 → 收图/视频。登录 `dreamina login`(扫码,账号额度 `user_credit`)。
- **codex**:`/media/heygo/.../github-repos/T8-penguin-canvas/backend/src/utils/codexCliRunner.js`,`runCodexExecStream`(L939)。Node `spawn` 起 `codex exec --json --enable image_generation`,流式收 JSON 事件 → 从 workspace 捞产物。登录 `codex login`(OAuth,ChatGPT 订阅额度)。

两者**形态完全一致**:本机账号登录态 CLI + subprocess 驱动 + 轮询/流式 + 收产物。**纯云转发,不吃本地 GPU**。

封号风险结论(已与用户对齐):CLI 进程跑在本机,出口是单账号单 IP;ZeroTier 只在「另一平台 ↔ 本机」这段私网,云厂商看不到。风险只来自**请求形态**(量/并发/自动化节奏),不来自 ZeroTier。护栏(并发上限 + 限速 + 节流)把账号画像压在「单人偶尔用」即安全。这是本设计内置护栏的根本动因。

## 2. 现状盘点(nous-center 落点)

| 关注点 | 现状坐标 |
|---|---|
| 引擎接口 | `backend/src/services/inference/base.py` `InferenceAdapter`(ABC,GPU 中心,`vram_mb` 概念) |
| GPU runner | `backend/src/runner/supervisor.py` 按 GPU 组 spawn 子进程;`GroupScheduler` 是**显存队列**,非账号限速 |
| 节点分流 | `backend/src/services/node_routing.py:26` `node_exec_class` → `dispatch`(GPU runner)vs `inline`(主进程);白名单 `DISPATCH_NODE_TYPES`(L22) |
| inline 节点 | `backend/src/services/nodes/registry.py` `@register`;`InvokableNode.invoke(data, inputs)`(`nodes/base.py:20`) |
| 出图终端 | `nodes/image.py:15` `image_output` sink → `{image_url, media_type, width, height}`;签名 URL 走 `image_output_storage` |
| service 分类检测 | `image_output` sink 命中 → category=image / meter=images(#467 口径) |
| 对外端点 | `api/routes/openai_compat.py` `POST /v1/images/generations`(L682)→ `workflow_service_runner.run_published_workflow`(L113) |
| 鉴权/计量 | M:N api_key + 配额(predictions / openai_compat 既有路径) |

**关键发现**:nous-center 无「纯外部云转发、不吃 GPU」的既有引擎(`VLLMAdapter` 实际是本机 GPU 子进程,不是先例)。但 inline 节点天然在主进程执行、不经 GPU runner —— 这正是 CLI relay 的正确落点。

## 3. 目标设计

### 3.1 总体:独立 provider 子系统 + inline 节点,绕开 GPU runner

```
backend/src/services/external_providers/
  base.py        ExternalCliProvider(ABC):probe_status / login_start / generate(req)->ExternalGenResult
  governor.py    ProviderGovernor:并发信号量 + 令牌桶限速 + 最小间隔节流 + 队列容量(满→503) + 可选结果缓存
  dreamina.py    DreaminaProvider:移植 run_jimeng_cli(Python→Python)
  codex.py       CodexProvider:移植 codexCliRunner(Node→Python),exec --json --enable image_generation
  config.py      读 configs/external_providers.yaml,构建 provider + governor 实例
nodes:
  src/services/nodes/external_gen.py  @register("external_image_gen") inline 节点 → 调子系统
configs/external_providers.yaml       每 provider 的 executable / modalities / 护栏参数
```

- `external_image_gen` **不进 `DISPATCH_NODE_TYPES`** → `node_exec_class=inline` → 主进程内 `await` → 不碰 GPU runner / 不占 GPU 组队列。
- 节点 `invoke()` 内:`async with governor.slot(provider_name): result = await provider.generate(req)` → 产物经 `image_output_storage` 落盘签名 URL → 出 `{image_url, ...}` 接 `image_output`。
- service(category=image)发布:`PrimitiveInput → external_image_gen(provider=dreamina|codex) → image_output`。`/v1/images/generations` → `run_published_workflow` 自动跑通,复用既有 M:N key + 配额 + run-history 产物。

### 3.2 ExternalCliProvider ABC

```python
class ProviderStatus(BaseModel):
    available: bool; logged_in: bool; version: str = ""
    quota: str | None = None; message: str = ""
    modalities: list[str] = []

class ExternalGenRequest(BaseModel):
    prompt: str; negative_prompt: str = ""
    width: int = 1024; height: int = 1024; num_images: int = 1
    input_images: list[str] = []          # 本地路径/签名 URL,provider 自行解析
    model: str | None = None; extra: dict = {}

class ExternalGenResult(BaseModel):
    artifacts: list[ArtifactRef]          # kind(image/video) + local_path
    text: str = ""; elapsed_ms: int = 0

class ExternalCliProvider(ABC):
    name: str; modalities: set[MediaModality]
    async def probe_status(self) -> ProviderStatus: ...
    async def login_start(self) -> dict: ...           # 代理 `codex login` / `dreamina login`
    async def generate(self, req: ExternalGenRequest) -> ExternalGenResult: ...
```

executable 解析、WindowsApps 兜底、artifact 提取(markdown 链接 + workspace 扫描 + 相对路径)直接移植两个上游实现(已在真实账号上跑过,降低风险)。

### 3.3 ProviderGovernor 护栏(子系统级,跨 provider 共用)

每 provider 一个 governor 实例,串在 `generate()` 外层:

- **并发**:`asyncio.Semaphore(concurrency)`(默认 1)。
- **限速**:令牌桶 `rate_per_min`(默认即梦 4 / codex 6)。
- **节流**:`min_interval_s` 两次调用最小间隔(默认即梦 5 / codex 3)。
- **队列容量**:等待槽位的请求数上限,超了直接 `503`(对齐 `GroupScheduler` 满→503 语义),不无限堆积。
- **缓存(可选)**:`cache_ttl_s>0` 时,(provider, 规整 prompt+参数) 哈希命中 → 直接返上次 artifacts,省一次云调用(默认关)。
- 全部 per-provider 可配;无 `Date.now`/随机以外副作用,便于单测。

护栏的意义:把账号在云侧看到的画像锁死在「单人、低频、串行」,这是规避封号的核心。

### 3.4 配置 configs/external_providers.yaml

```yaml
providers:
  dreamina:
    enabled: true
    executable: dreamina            # 或绝对路径;留空走 PATH 查找
    modalities: [image, video]
    concurrency: 1
    rate_per_min: 4
    min_interval_s: 5
    cache_ttl_s: 0
  codex:
    enabled: true
    executable: codex
    modalities: [image]
    concurrency: 1
    rate_per_min: 6
    min_interval_s: 3
    cache_ttl_s: 0
```

账号登录态**不落 nous-center**:留在各 CLI 自己的 home(`~/.codex` / dreamina config)。登录经 endpoint 代理触发 CLI 自身的 login 流程。

### 3.5 内网管理端点

`backend/src/api/routes/external_providers.py`(admin-gated,供前端面板 + 你另一平台诊断):
- `GET /api/v1/external-providers` — 列 provider + `probe_status`(装没装/登没登/额度)。
- `POST /api/v1/external-providers/{name}/login` — 触发登录流程,返回引导信息。
- 生成**不**在这里走 —— 生成统一走 service / `/v1/images/generations`(用户选定),保证对外接口单一。

## 4. PR 切分

> 分支独立、走 CI;CI 全程 mock subprocess(不碰真账号/真 CLI),真机 smoke 单人手动验。

### PR-1 子系统骨架 + 即梦 provider + 内网 status/login
- 新增 `external_providers/{base,governor,config}.py` + `dreamina.py`(移植 `run_jimeng_cli` + submit/poll/collect)。
- `configs/external_providers.yaml` + loader。
- `api/routes/external_providers.py`(status/login)。
- 测试:governor 并发/节流/限速/503 行为(mock 时钟,注入而非 `Date.now`);即梦 artifact 提取/submit_id 解析(mock subprocess 输出)。
- 暂不接工作流/对外端点。

### PR-2 codex provider 坐进同一子系统
- `codex.py`:移植 `codexCliRunner`(feature 探测 / `--enable image_generation` / `--enable` 不支持时剥离重试 / JSON 流 delta 提取 / reasoning 过滤 / artifact 提取)。
- 复用同一 `ProviderGovernor`。
- 测试:exec args 构建、feature 剥离重试、artifact 提取(移植上游既有单测用例)。

### PR-3 inline 节点 + service 接入 + 对外端点打通
- `nodes/external_gen.py` `@register("external_image_gen")`:解析上游图 → `governor.slot` → `provider.generate` → 产物落 `image_output_storage` → 出 `image_output` envelope。**确认不进 `DISPATCH_NODE_TYPES`**。
- quick-provision 支持 category=image 经 external provider(或手搭 `input→external_image_gen→image_output` 工作流发布)。
- `/v1/images/generations` → `run_published_workflow` 端到端(M:N key + 配额 + run-history 产物)。
- 测试:节点 invoke(mock provider)、service 分类检测=image、`num_images` 注入、配额计量。

### PR-4 护栏硬化 + 前端面板 + 部署
- 缓存 + 队列容量 + 限速精修;`/api/v1/external-providers` 状态/登录前端面板(lucide 图标,无 emoji)。
- ZeroTier 内网绑定说明写入 infra/CLAUDE 文档;不暴露公网生成。
- 真机 smoke(单人,真账号):即梦出图 + codex 出图 → 验产物落盘 + 签名 URL + governor 限并发。

## 5. 验证

- **CI**:ruff + 单测全 mock subprocess(governor 行为、artifact 提取、节点 invoke、exec args、service 分类);`NOUS_DISABLE_FRONTEND_MOUNT=1` + `ADMIN_PASSWORD=""` 既有约定。
- **真机**:`dreamina login` / `codex login` 后,经 `/v1/images/generations` 单人调用出图;验证并发被 governor 限到 1、节流间隔生效、产物进 `/history` 画廊。
- **封号护栏自检**:压测脚本确认超 `rate_per_min`/队列容量时返 503 而非继续打云。

## 6. 非目标 / follow-up

- **不**对公网多租户暴露(封号红线)。
- **不**塞进 GPU runner / GroupScheduler(纯云转发,inline 节点即可)。
- 视频模态(即梦 `seedance`)留 PR-3 后续:节点出 `video_url` 接 video sink,需先确认 nous-center 是否有 video_output sink + video service 分类。
- 账号额度耗尽/登录失效的优雅降级与前端提示,PR-4 起按真机表现补。
