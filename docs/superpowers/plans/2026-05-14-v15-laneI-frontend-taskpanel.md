# V1.5 Lane I: TaskPanel 重构为 Buildkite 风 runner 泳道 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 `frontend/src/components/panels/TaskPanel.tsx` 从「460px 抽屉 + 3 tab 扁平列表」**重构**为 Buildkite 风结构 —— 顶部 per-runner 泳道区（视觉 hero）+ 下方「最近完成」列表（image 任务带输出缩略图）。配套：点 Run 后的 toast + IconRail badge 反馈（面板不自动打开）、任务完成的 toast + 浏览器 Notification、runner 异常态内联表达、排队位置可展开列表、`<768px` 响应式全屏、a11y 必加清单、Dashboard GpuPanel 加 runner 标识。对应 spec §6 全部 DD3-DD9 + a11y 清单。

**Architecture:** 三块交付物，按依赖顺序：

1. **数据层** —— 新增 `frontend/src/api/runners.ts`（`useRunners()` hook，拉 runner 泳道数据：每个 runner 的 state / 当前任务 / 进度 / 排队列表）。`frontend/src/api/tasks.ts` 的 `ExecutionTask` 接口扩 V1.5 字段（`gpu_group` / `runner_id` / `queue_position` / `output_thumbnails`），与 Lane B 的 `execution_tasks` schema 对齐。完成通知用一个新 zustand store `frontend/src/stores/notifications.ts`（去重 + 浏览器 Notification 权限流）。
2. **TaskPanel 重构** —— `TaskPanel.tsx` 拆成 `RunnerLane`（泳道）+ `RecentList`（最近完成）+ `QueueExpand`（排队展开）+ `ImageThumb`（缩略图）四个 subview。原 `TaskCard` 逻辑保留给「最近完成」列表行复用。响应式 `<768px` 全屏 + 泳道堆叠。
3. **反馈链路** —— `Topbar.tsx` 的 `handleRun` 接 Lane S 的 202 异步契约：点 Run → 拿 `task_id` → toast「任务已入队」+ 不自动开面板；`useExecutionStore` 加 `taskIconBadge` 计数驱动 IconRail badge；任务转终态时 `notifications` store 发 toast + 浏览器通知。`DashboardOverlay.tsx` 的 `GpuCard` 加「Runner: A (image)」标签。

**Tech Stack:** Vite + React 19 + TypeScript 5.9 / `@tanstack/react-query` 5（已用于 `useTasks`）/ zustand 5（已有 `execution` / `toast` store）/ lucide-react 图标 / Vitest 4 + `@testing-library/react` 16（`environment: jsdom`，`setupFiles: ./src/test/setup.ts`，`globals: true`）。**无新增依赖** —— toast 用既有 `useToastStore` + `ToastContainer`，浏览器通知用原生 `Notification` API。

> **注意 — 与 spec 的偏差（已核实，须知会）：**
>
> 1. **后端没有 runner 状态端点。** spec §6.1/§6.2 的 per-runner 泳道需要「每个 GPU runner 的 state / 当前任务 / 进度 / 排队数」这份数据，但 grep 全后端（`backend/src/api/routes/`）**没有 `/api/v1/runners` 或等价端点**；`monitor.py` 只给 GPU 硬件指标（util/mem/温度），不给 runner 调度态。Lane G（GroupScheduler）/ Lane H（RunnerSupervisor + `/health` 扩展）的 plan 也未明确定义一个供前端消费的 runner 端点。**判断**：Lane I 假定一个 `GET /api/v1/runners` 契约（返回 `RunnerInfo[]`，形状见 Task 1），由 Lane G/H 提供。若 Lane G/H 落地时端点形状不同，Task 1 的 `RunnerInfo` 接口 + `useRunners` 的 `queryFn` 需对齐——`useRunners` 故意写成「端点 404 时降级为空泳道」，让 Lane I 能独立 merge、CI 绿，不硬阻塞在 Lane G/H 的端点上。已在 Self-Review 标注。
> 2. **`ExecutionTask` 缺 V1.5 字段 + 缩略图字段。** Lane B 给后端 `execution_tasks` 加了 8 列（`gpu_group` / `runner_id` / ...），但前端 `tasks.ts` 的 `ExecutionTask` 接口还是 V1 形状。spec §6.5 的 DD9 缩略图历史需要 `output_thumbnails`（image 任务输出图 URL 列表），spec §2.1 说 image 结果写 `outputs/{task_id}/`，Lane D 落盘 + 后端 `/tasks` 序列化时应带出缩略图 URL。**判断**：Lane I 扩 `ExecutionTask` 接口加 `gpu_group` / `runner_id` / `queue_position` / `output_thumbnails`（全部 optional，旧后端不返回时为 `undefined`），`ImageThumb` 在 `output_thumbnails` 为空时降级为既有的 `ImageIcon`。后端实际序列化这些字段是 Lane B/D/G 的 scope，不在 Lane I。已在 Self-Review 标注。
> 3. **`/ws/tasks` 事件 payload 未知。** `tasks.ts` 现有的 `useTasks()` 用 `/ws/tasks` 做「收到任意消息就 invalidate」的粗粒度刷新，不解析 payload。spec §3.7 的 WSEvent 词表（`execution_start` / `executed` / `execution_error` ...）是 `/ws/workflow/{id}` 的词表，不是 `/ws/tasks` 的。**判断**：Lane I **不改** `/ws/tasks` 的消费方式（继续 invalidate-on-message），完成通知靠「`useTasks` 数据里某 task 从非终态翻到终态」这个 diff 触发，不依赖解析具体 WS 事件类型。这样 Lane I 不和 Lane G 的 WS 推送实现耦合。已在 Self-Review 标注。

---

## File Structure

| 文件 | Lane I 动作 | 责任 |
|---|---|---|
| `frontend/src/api/runners.ts` | **新建** | `RunnerInfo` 接口 + `useRunners()` hook（拉 `/api/v1/runners`，404 降级空数组） |
| `frontend/src/api/tasks.ts` | **修改** | `ExecutionTask` 接口扩 `gpu_group` / `runner_id` / `queue_position` / `output_thumbnails`（全 optional） |
| `frontend/src/stores/notifications.ts` | **新建** | 完成通知 store：去重已通知的 task_id + 浏览器 Notification 权限流（首次询问 / 拒绝降级 toast-only / 仅失焦发系统通知） |
| `frontend/src/stores/execution.ts` | **修改** | 加 `taskIconBadge: number` + `bumpTaskBadge` / `clearTaskBadge`，驱动 IconRail badge（DD4） |
| `frontend/src/components/panels/TaskPanel.tsx` | **重写** | Buildkite 风结构：`RunnerLane` 泳道区 + `RecentList` 最近完成 + `QueueExpand` 排队展开 + `ImageThumb` 缩略图；响应式 `<768px` 全屏；a11y |
| `frontend/src/components/panels/TaskPanel.test.tsx` | **重写** | 对齐新结构的组件测试（泳道渲染 / 异常态 / 排队展开 / 缩略图 / 响应式 / a11y） |
| `frontend/src/hooks/useTaskCompletionNotifier.ts` | **新建** | 监听 `useTasks()` 数据，检测「非终态 → 终态」的 diff，触发 `notifications` store（DD6） |
| `frontend/src/components/layout/Topbar.tsx` | **修改** | `handleRun` 接 Lane S 202 契约：toast「已入队」+ `bumpTaskBadge`，**不**自动开面板（DD4） |
| `frontend/src/components/layout/IconRail.tsx` | **修改** | `TaskRailButton` 的 badge 改读 `taskIconBadge`（不再只数 running），点开面板时 `clearTaskBadge`（DD4） |
| `frontend/src/components/overlays/DashboardOverlay.tsx` | **修改** | `GpuCard` 加「Runner: A (image)」标签（DD3 末项） |
| `frontend/src/App.tsx` | **修改** | 挂载 `useTaskCompletionNotifier()`（全局监听，与 `ToastContainer` 同级） |
| `frontend/src/hooks/useTaskCompletionNotifier.test.tsx` | **新建** | diff 检测 + 去重 + 失焦判定的单元测试 |
| `frontend/src/stores/notifications.test.ts` | **新建** | 权限流 + 去重的单元测试 |

---

## Task 1: `runners.ts` —— `useRunners()` hook + `RunnerInfo` 契约

spec §6.1/§6.2 的泳道需要 runner 调度态。后端无此端点（见顶部偏差 1），Lane I 定义契约并写一个「404 降级空数组」的 hook，让本 Lane 能独立 merge。

**Files:**
- Create: `frontend/src/api/runners.ts`
- Test: `frontend/src/api/runners.test.ts`（新建）

- [ ] **Step 1: 跑现有前端 suite 建基线**

Run: `cd frontend && npm run test`
Expected: PASS（记下通过的测试文件数 + 用例数，作为 Lane I 改动后的回归对照）。

- [ ] **Step 2: 写失败测试 —— useRunners 解析 + 404 降级**

新建 `frontend/src/api/runners.test.ts`：
```tsx
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { renderHook, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { useRunners } from './runners'

vi.mock('./client', () => ({
  apiFetch: vi.fn(),
}))
import { apiFetch } from './client'

function wrapper() {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={qc}>{children}</QueryClientProvider>
  )
}

describe('useRunners', () => {
  beforeEach(() => vi.clearAllMocks())

  it('returns runner list from /api/v1/runners', async () => {
    vi.mocked(apiFetch).mockResolvedValue([
      {
        id: 'runner-i',
        label: 'Runner-I',
        role: 'image',
        state: 'busy',
        current_task: { task_id: '7k2m', workflow_name: 'flux2-人物立绘', progress: 0.6, detail: 'step 18/30' },
        queue: [{ task_id: '9p1q', workflow_name: 'sd-背景', position: 1 }],
        restart_attempt: null,
        load_error: null,
      },
    ])
    const { result } = renderHook(() => useRunners(), { wrapper: wrapper() })
    await waitFor(() => expect(result.current.data).toBeDefined())
    expect(result.current.data?.[0].id).toBe('runner-i')
    expect(result.current.data?.[0].current_task?.detail).toBe('step 18/30')
  })

  it('degrades to empty array when endpoint 404s', async () => {
    vi.mocked(apiFetch).mockRejectedValue(
      Object.assign(new Error('Not Found'), { status: 404 }),
    )
    const { result } = renderHook(() => useRunners(), { wrapper: wrapper() })
    await waitFor(() => expect(result.current.data).toEqual([]))
  })
})
```

- [ ] **Step 3: 跑测试确认失败**

Run: `cd frontend && npm run test -- src/api/runners.test.ts`
Expected: FAIL —— `Failed to resolve import "./runners"`（模块还没建）。

- [ ] **Step 4: 创建 `runners.ts`**

新建 `frontend/src/api/runners.ts`：
```ts
import { useQuery } from '@tanstack/react-query'
import { apiFetch } from './client'

/** runner 泳道当前正在跑的任务（spec §6.1 进度条 + step 文案的数据源）。 */
export interface RunnerCurrentTask {
  task_id: string
  workflow_name: string
  /** 0.0 ~ 1.0 —— 泳道进度条宽度。 */
  progress: number
  /** "step 18/30" 之类的细节文案，可空。 */
  detail: string | null
}

/** runner 排队列表里的一条（spec §6.4 排队展开的有序列表项）。 */
export interface RunnerQueueItem {
  task_id: string
  workflow_name: string
  /** 1-based 排队序号，spec §6.4 的 #1 #2 #3。 */
  position: number
}

/** 一个 GPU runner 泳道的完整状态（spec §6.1 / §6.2）。
 *
 * state 与 spec §6.2 异常态表一一对应：
 *   idle    — 灰点 + 「idle」
 *   busy    — 绿点 + current_task 名 + 进度条
 *   restarting — 黄色脉冲点 + 「重启中 N/M」（restart_attempt 提供 N/M）
 *   load_failed — 红点 + 「加载失败: ...」（load_error 提供文案）+ Retry 按钮
 */
export interface RunnerInfo {
  id: string
  /** 展示名，如 "Runner-I"。 */
  label: string
  role: 'image' | 'tts' | 'llm'
  state: 'idle' | 'busy' | 'restarting' | 'load_failed'
  current_task: RunnerCurrentTask | null
  queue: RunnerQueueItem[]
  /** restarting 态：[当前第几次, 总 backoff 次数]，如 [2, 4] → 「重启中 2/4」。 */
  restart_attempt: [number, number] | null
  /** load_failed 态：失败文案，如 "qwen3-35b OOM"。 */
  load_error: string | null
}

/** 拉 runner 泳道数据。
 *
 * 后端 /api/v1/runners 由 V1.5 Lane G/H 提供（RunnerSupervisor 调度态）。
 * 端点尚未落地时 hook 降级为空数组 —— TaskPanel 泳道区显示「暂无 runner 数据」，
 * 不阻塞 Lane I 独立 merge。
 */
export function useRunners() {
  return useQuery<RunnerInfo[]>({
    queryKey: ['runners'],
    queryFn: async () => {
      try {
        return await apiFetch<RunnerInfo[]>('/api/v1/runners')
      } catch (e) {
        // 端点未落地（404）/ 暂时不可达 → 降级空泳道，不让整个面板崩。
        if ((e as { status?: number }).status === 404) return []
        throw e
      }
    },
    // runner 状态变化频繁（进度条、排队数），3s 轮询；WS 推送由后续 Lane 接。
    refetchInterval: 3_000,
  })
}
```

- [ ] **Step 5: 跑测试确认通过**

Run: `cd frontend && npm run test -- src/api/runners.test.ts`
Expected: 两个用例 PASS。

- [ ] **Step 6: Commit**

```bash
cd /media/heygo/Program/projects-code/_playground/nous-center
git add frontend/src/api/runners.ts frontend/src/api/runners.test.ts
git commit -m "feat(frontend): useRunners hook for Buildkite-style runner lanes

RunnerInfo contract for the per-runner lane data (state / current task
/ queue / restart-attempt / load-error). Endpoint /api/v1/runners is
provided by V1.5 Lane G/H; hook degrades to [] on 404 so Lane I merges
independently. V1.5 Lane I, spec 6.1/6.2."
```

---

## Task 2: 扩 `ExecutionTask` 接口 —— V1.5 字段 + 缩略图

spec §6.5 的 DD9 缩略图历史 + §6.1 泳道需要 task 带 `gpu_group` / `runner_id` / `queue_position` / `output_thumbnails`。后端序列化是 Lane B/D scope，前端先把接口扩好（全 optional，旧后端不返回时 `undefined`）。

**Files:**
- Modify: `frontend/src/api/tasks.ts`
- Test: `frontend/src/api/tasks.test.ts`（新建 —— 仅做类型层面的编译断言，无运行时逻辑）

- [ ] **Step 1: 写失败测试 —— 接口含新字段（编译期断言）**

新建 `frontend/src/api/tasks.test.ts`：
```ts
import { describe, it, expect } from 'vitest'
import type { ExecutionTask } from './tasks'

describe('ExecutionTask V1.5 fields', () => {
  it('accepts V1.5 scheduler + thumbnail fields', () => {
    // 编译期断言：下面这个对象能赋给 ExecutionTask 就说明接口已扩。
    const t: ExecutionTask = {
      id: 'wf_1',
      workflow_id: null,
      workflow_name: 'flux2-人物立绘',
      status: 'completed',
      nodes_total: 2,
      nodes_done: 2,
      current_node: null,
      result: null,
      error: null,
      duration_ms: 34000,
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      task_type: 'image',
      image_width: 1024,
      image_height: 1024,
      gpu_group: 'image',
      runner_id: 'runner-i',
      queue_position: null,
      output_thumbnails: ['/files/outputs/wf_1/0.webp'],
    }
    expect(t.output_thumbnails?.[0]).toContain('outputs')
    expect(t.gpu_group).toBe('image')
  })

  it('V1.5 fields are optional (old backend payload still valid)', () => {
    const legacy: ExecutionTask = {
      id: 'wf_2',
      workflow_id: null,
      workflow_name: 'legacy',
      status: 'queued',
      nodes_total: 0,
      nodes_done: 0,
      current_node: null,
      result: null,
      error: null,
      duration_ms: null,
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      task_type: null,
      image_width: null,
      image_height: null,
    }
    expect(legacy.gpu_group).toBeUndefined()
  })
})
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd frontend && npm run test -- src/api/tasks.test.ts`
Expected: FAIL —— TypeScript 报 `gpu_group` / `output_thumbnails` 不在 `ExecutionTask` 上（`Object literal may only specify known properties`）。

- [ ] **Step 3: 扩 `ExecutionTask` 接口**

`frontend/src/api/tasks.ts` 的 `ExecutionTask` 接口，在 `image_height` 之后追加 V1.5 字段：
```ts
export interface ExecutionTask {
  id: string
  workflow_id: string | null
  workflow_name: string
  status: 'queued' | 'running' | 'completed' | 'failed' | 'cancelled'
  nodes_total: number
  nodes_done: number
  current_node: string | null
  result: any
  error: string | null
  duration_ms: number | null
  created_at: string
  updated_at: string
  /** Server-derived from result envelope. null until the run completes
   * with an image_output, then 'image'. Used for the task card badge. */
  task_type: 'image' | null
  image_width: number | null
  image_height: number | null

  // —— V1.5 新增（Lane I，对齐 Lane B execution_tasks schema）——
  // 全部 optional：旧后端 payload 不带这些字段时为 undefined，
  // 组件按 undefined 降级（缩略图 → ImageIcon，runner 标识 → 不显示）。
  /** 落到哪个 hardware.yaml group（"image" / "llm-tp" / "tts"）。 */
  gpu_group?: string | null
  /** 实际执行的 runner 实例 id。 */
  runner_id?: string | null
  /** queued 态时的排队序号（1-based）；非 queued 态为 null/undefined。 */
  queue_position?: number | null
  /** image 任务的输出缩略图 URL 列表（数据源 outputs/{task_id}/，
   * 后端 Lane D 落盘 + /tasks 序列化时带出）。空 → 降级 ImageIcon。 */
  output_thumbnails?: string[] | null
}
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd frontend && npm run test -- src/api/tasks.test.ts`
Expected: 两个用例 PASS。

- [ ] **Step 5: Commit**

```bash
cd /media/heygo/Program/projects-code/_playground/nous-center
git add frontend/src/api/tasks.ts frontend/src/api/tasks.test.ts
git commit -m "feat(frontend): extend ExecutionTask with V1.5 scheduler + thumbnail fields

gpu_group / runner_id / queue_position / output_thumbnails — all
optional so legacy backend payloads stay valid. Mirrors Lane B
execution_tasks columns; output_thumbnails feeds DD9 thumbnail
history. V1.5 Lane I, spec 6.5."
```

---

## Task 3: `notifications.ts` store —— 完成通知 + 浏览器权限流

spec §6.3 DD6：任务完成 → app 内 toast + 浏览器 Notification。§6.6 a11y：权限首次询问 → 拒绝降级 toast-only；仅页面失焦时发系统通知。本 Task 只做 store（去重 + 权限流），diff 检测在 Task 4。

**Files:**
- Create: `frontend/src/stores/notifications.ts`
- Test: `frontend/src/stores/notifications.test.ts`（新建）

- [ ] **Step 1: 写失败测试 —— 去重 + 权限流 + 失焦判定**

新建 `frontend/src/stores/notifications.test.ts`：
```ts
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { useNotificationStore } from './notifications'

describe('notifications store', () => {
  beforeEach(() => {
    useNotificationStore.setState({ notified: new Set(), permission: 'default' })
    vi.restoreAllMocks()
  })
  afterEach(() => vi.restoreAllMocks())

  it('notifyOnce dedupes by task_id', () => {
    const toast = vi.fn()
    const { notifyOnce } = useNotificationStore.getState()
    notifyOnce('task-1', 'flux2 完成 · 34s', 'success', toast)
    notifyOnce('task-1', 'flux2 完成 · 34s', 'success', toast)
    expect(toast).toHaveBeenCalledTimes(1)
  })

  it('always fires the in-app toast even without notification permission', () => {
    const toast = vi.fn()
    useNotificationStore.setState({ permission: 'denied' })
    useNotificationStore.getState().notifyOnce('task-2', '失败', 'error', toast)
    expect(toast).toHaveBeenCalledWith('失败', 'error')
  })

  it('only sends a system Notification when the page is unfocused + permission granted', () => {
    const NotificationCtor = vi.fn()
    vi.stubGlobal('Notification', Object.assign(NotificationCtor, { permission: 'granted' }))
    vi.spyOn(document, 'hasFocus').mockReturnValue(false) // 页面失焦
    useNotificationStore.setState({ permission: 'granted' })

    useNotificationStore.getState().notifyOnce('task-3', '完成', 'success', vi.fn())
    expect(NotificationCtor).toHaveBeenCalledTimes(1)

    // 页面有焦点时不发系统通知（只 toast）
    vi.spyOn(document, 'hasFocus').mockReturnValue(true)
    useNotificationStore.getState().notifyOnce('task-4', '完成', 'success', vi.fn())
    expect(NotificationCtor).toHaveBeenCalledTimes(1) // 没再增加
  })

  it('requestPermission updates store + degrades gracefully when API absent', async () => {
    // 无 Notification API 的环境（jsdom 默认）→ permission 落 'denied'
    vi.stubGlobal('Notification', undefined)
    await useNotificationStore.getState().requestPermission()
    expect(useNotificationStore.getState().permission).toBe('denied')
  })
})
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd frontend && npm run test -- src/stores/notifications.test.ts`
Expected: FAIL —— `Failed to resolve import "./notifications"`。

- [ ] **Step 3: 创建 `notifications.ts`**

新建 `frontend/src/stores/notifications.ts`：
```ts
import { create } from 'zustand'

type NotifyPermission = 'default' | 'granted' | 'denied'
type ToastFn = (message: string, type: 'success' | 'error' | 'info') => void

interface NotificationState {
  /** 已发过通知的 task_id —— 去重，避免同一 task 重复弹（轮询 + WS 双触发）。 */
  notified: Set<string>
  /** 浏览器 Notification 权限快照（'default' = 还没问过）。 */
  permission: NotifyPermission

  /** 首次询问浏览器通知权限。拒绝 / 无 API → permission 落 'denied'，降级 toast-only。 */
  requestPermission: () => Promise<void>
  /** 对某 task 发一次完成/失败通知：永远发 app 内 toast；
   * 仅当「权限 granted + 页面失焦」时额外发系统 Notification。已通知过的 task_id 跳过。 */
  notifyOnce: (
    taskId: string,
    message: string,
    type: 'success' | 'error',
    toast: ToastFn,
  ) => void
}

export const useNotificationStore = create<NotificationState>((set, get) => ({
  notified: new Set(),
  permission: 'default',

  requestPermission: async () => {
    // jsdom / 老浏览器无 Notification API → 直接降级。
    if (typeof Notification === 'undefined') {
      set({ permission: 'denied' })
      return
    }
    if (Notification.permission !== 'default') {
      set({ permission: Notification.permission as NotifyPermission })
      return
    }
    try {
      const result = await Notification.requestPermission()
      set({ permission: result as NotifyPermission })
    } catch {
      set({ permission: 'denied' })
    }
  },

  notifyOnce: (taskId, message, type, toast) => {
    const { notified } = get()
    if (notified.has(taskId)) return
    set({ notified: new Set(notified).add(taskId) })

    // app 内 toast —— 永远发，是降级基线（spec §6.6：拒绝权限就只 toast）。
    toast(message, type === 'error' ? 'error' : 'success')

    // 系统通知 —— 仅「权限 granted + 页面失焦」才发（spec §6.3）。
    const canSystemNotify =
      typeof Notification !== 'undefined' &&
      Notification.permission === 'granted' &&
      !document.hasFocus()
    if (canSystemNotify) {
      try {
        new Notification('nous-center', { body: message })
      } catch {
        // 某些环境构造会抛 —— 静默吞，toast 已经发了。
      }
    }
  },
}))
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd frontend && npm run test -- src/stores/notifications.test.ts`
Expected: 四个用例 PASS。

- [ ] **Step 5: Commit**

```bash
cd /media/heygo/Program/projects-code/_playground/nous-center
git add frontend/src/stores/notifications.ts frontend/src/stores/notifications.test.ts
git commit -m "feat(frontend): notifications store with browser Notification permission flow

notifyOnce dedupes by task_id, always fires the in-app toast (the
degraded baseline), and only sends a system Notification when
permission is granted AND the page is unfocused. requestPermission
degrades to 'denied' when the API is absent. V1.5 Lane I, spec 6.3/6.6."
```

---

## Task 4: `useTaskCompletionNotifier` hook —— 终态 diff 检测

spec §6.3 DD6：任务完成（含失败）触发通知。检测方式 = 监听 `useTasks()` 数据，某 task 从「非终态」翻到「终态」就触发 `notifications` store。不解析 WS payload（见顶部偏差 3）。

**Files:**
- Create: `frontend/src/hooks/useTaskCompletionNotifier.ts`
- Test: `frontend/src/hooks/useTaskCompletionNotifier.test.tsx`（新建）

- [ ] **Step 1: 写失败测试 —— 非终态→终态触发一次，去重，初次加载不误报**

新建 `frontend/src/hooks/useTaskCompletionNotifier.test.tsx`：
```tsx
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { renderHook } from '@testing-library/react'
import { useTaskCompletionNotifier } from './useTaskCompletionNotifier'
import { useNotificationStore } from '../stores/notifications'

const notifyOnce = vi.fn()
vi.mock('../stores/notifications', () => ({
  useNotificationStore: () => ({ notifyOnce }),
}))
const toastAdd = vi.fn()
vi.mock('../stores/toast', () => ({
  useToastStore: (sel: (s: { add: typeof toastAdd }) => unknown) => sel({ add: toastAdd }),
}))

let mockTasks: any[] = []
vi.mock('../api/tasks', () => ({
  useTasks: () => ({ data: mockTasks }),
}))

describe('useTaskCompletionNotifier', () => {
  beforeEach(() => {
    notifyOnce.mockClear()
    mockTasks = []
  })

  it('does NOT notify for tasks already terminal on first load', () => {
    mockTasks = [{ id: 't1', status: 'completed', workflow_name: 'wf', duration_ms: 1000 }]
    renderHook(() => useTaskCompletionNotifier())
    expect(notifyOnce).not.toHaveBeenCalled()
  })

  it('notifies when a running task transitions to completed', () => {
    mockTasks = [{ id: 't1', status: 'running', workflow_name: 'flux2-人物立绘', duration_ms: null }]
    const { rerender } = renderHook(() => useTaskCompletionNotifier())
    expect(notifyOnce).not.toHaveBeenCalled()

    mockTasks = [{ id: 't1', status: 'completed', workflow_name: 'flux2-人物立绘', duration_ms: 34000 }]
    rerender()
    expect(notifyOnce).toHaveBeenCalledTimes(1)
    expect(notifyOnce).toHaveBeenCalledWith(
      't1',
      expect.stringContaining('flux2-人物立绘'),
      'success',
      toastAdd,
    )
  })

  it('notifies with error type when a task transitions to failed', () => {
    mockTasks = [{ id: 't2', status: 'running', workflow_name: 'sd-bg', duration_ms: null }]
    const { rerender } = renderHook(() => useTaskCompletionNotifier())
    mockTasks = [{ id: 't2', status: 'failed', workflow_name: 'sd-bg', duration_ms: 2000 }]
    rerender()
    expect(notifyOnce).toHaveBeenCalledWith('t2', expect.stringContaining('失败'), 'error', toastAdd)
  })
})
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd frontend && npm run test -- src/hooks/useTaskCompletionNotifier.test.tsx`
Expected: FAIL —— `Failed to resolve import "./useTaskCompletionNotifier"`。

- [ ] **Step 3: 创建 `useTaskCompletionNotifier.ts`**

新建 `frontend/src/hooks/useTaskCompletionNotifier.ts`：
```ts
import { useEffect, useRef } from 'react'
import { useTasks, type ExecutionTask } from '../api/tasks'
import { useNotificationStore } from '../stores/notifications'
import { useToastStore } from '../stores/toast'

const TERMINAL: ReadonlySet<ExecutionTask['status']> = new Set([
  'completed',
  'failed',
  'cancelled',
])

function isTerminal(status: ExecutionTask['status']): boolean {
  return TERMINAL.has(status)
}

function fmtDuration(ms: number | null): string {
  if (ms == null || ms <= 0) return ''
  const s = ms / 1000
  if (s < 60) return ` · ${s.toFixed(0)}s`
  const m = Math.floor(s / 60)
  return ` · ${m}m${Math.round(s - m * 60)}s`
}

/** 全局挂载一次（App.tsx）。监听 useTasks() 数据，某 task 从非终态翻到
 * 终态时发一次完成/失败通知。spec §6.3 DD6。
 *
 * 检测靠「上一帧 status」与「这一帧 status」的 diff —— 不解析 WS 事件类型，
 * 所以不和 Lane G 的 /ws/tasks 推送实现耦合。初次加载时已是终态的 task
 * 不触发（prevStatus 为空 = 不算「转换」）。 */
export function useTaskCompletionNotifier(): void {
  const { data: tasks } = useTasks()
  const { notifyOnce } = useNotificationStore()
  const toast = useToastStore((s) => s.add)
  // task_id → 上一帧观测到的 status。
  const prevStatus = useRef<Map<string, ExecutionTask['status']>>(new Map())

  useEffect(() => {
    if (!tasks) return
    const seen = prevStatus.current
    for (const t of tasks) {
      const before = seen.get(t.id)
      const justFinished =
        before !== undefined && !isTerminal(before) && isTerminal(t.status)
      if (justFinished) {
        if (t.status === 'completed') {
          notifyOnce(t.id, `${t.workflow_name} 完成${fmtDuration(t.duration_ms)}`, 'success', toast)
        } else if (t.status === 'failed') {
          notifyOnce(t.id, `${t.workflow_name} 失败`, 'error', toast)
        }
        // cancelled 不发通知（用户自己点的取消，无需打扰）。
      }
      seen.set(t.id, t.status)
    }
  }, [tasks, notifyOnce, toast])
}
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd frontend && npm run test -- src/hooks/useTaskCompletionNotifier.test.tsx`
Expected: 三个用例 PASS。

- [ ] **Step 5: 在 App.tsx 挂载 hook**

先定位 `ToastContainer` 的挂载点：
```bash
cd frontend && grep -n "ToastContainer\|function App\|return (" src/App.tsx
```
在 `App.tsx` 的 `App` 组件体内（`ToastContainer` 渲染处附近）调用 hook。加 import：
```tsx
import { useTaskCompletionNotifier } from './hooks/useTaskCompletionNotifier'
```
在 `App` 组件函数体顶部（其它 hook 调用旁）加一行：
```tsx
  useTaskCompletionNotifier()
```
（hook 无返回值，纯副作用 —— 监听 `useTasks` 并触发通知。`useTasks` 内部已建 `/ws/tasks` 频道，此处不重复建。）

- [ ] **Step 6: 跑全 suite + tsc 确认无回归**

Run: `cd frontend && npm run test && npx tsc -b --noEmit`
Expected: 全 PASS，tsc 无报错。

- [ ] **Step 7: Commit**

```bash
cd /media/heygo/Program/projects-code/_playground/nous-center
git add frontend/src/hooks/useTaskCompletionNotifier.ts frontend/src/hooks/useTaskCompletionNotifier.test.tsx frontend/src/App.tsx
git commit -m "feat(frontend): useTaskCompletionNotifier — fire completion notification on terminal transition

Watches useTasks() data, fires notifyOnce when a task goes
non-terminal -> terminal. Diff-based (prev status vs current), so it
does not couple to Lane G's /ws/tasks event payload. Tasks already
terminal on first load do not fire. Mounted globally in App.tsx.
V1.5 Lane I, spec 6.3."
```

---

## Task 5: `execution` store 加 `taskIconBadge` —— Run 反馈 badge 计数

spec §6.3 DD4：点 Run → IconRail 任务图标 badge 计数。当前 `IconRail` 的 `TaskRailButton` 是数 `running` 的 task 数；DD4 要的是「点 Run 累加、打开面板清零」的语义。

**Files:**
- Modify: `frontend/src/stores/execution.ts`
- Test: `frontend/src/stores/execution.test.ts`（新建，仅覆盖新增的 badge action）

- [ ] **Step 1: 写失败测试 —— badge 累加 / 清零**

新建 `frontend/src/stores/execution.test.ts`：
```ts
import { describe, it, expect, beforeEach } from 'vitest'
import { useExecutionStore } from './execution'

describe('execution store — task icon badge', () => {
  beforeEach(() => {
    useExecutionStore.setState({ taskIconBadge: 0, taskPanelOpen: false })
  })

  it('bumpTaskBadge increments the badge count', () => {
    useExecutionStore.getState().bumpTaskBadge()
    useExecutionStore.getState().bumpTaskBadge()
    expect(useExecutionStore.getState().taskIconBadge).toBe(2)
  })

  it('clearTaskBadge resets to zero', () => {
    useExecutionStore.setState({ taskIconBadge: 5 })
    useExecutionStore.getState().clearTaskBadge()
    expect(useExecutionStore.getState().taskIconBadge).toBe(0)
  })
})
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd frontend && npm run test -- src/stores/execution.test.ts`
Expected: FAIL —— `taskIconBadge` / `bumpTaskBadge` 不在 store 上（TypeScript 报错 + 运行时 `undefined`）。

- [ ] **Step 3: 给 `execution.ts` 加 badge state + actions**

`frontend/src/stores/execution.ts`：

在 `ExecutionState` 接口的 `// Task panel` 段，把：
```ts
  // Task panel
  taskPanelOpen: boolean
  toggleTaskPanel: () => void
```
改为：
```ts
  // Task panel
  taskPanelOpen: boolean
  toggleTaskPanel: () => void
  /** IconRail 任务图标 badge 计数（DD4）：点 Run 累加，打开面板清零。
   * 与「running task 数」解耦 —— 它表达的是「有未查看的新提交」。 */
  taskIconBadge: number
  bumpTaskBadge: () => void
  clearTaskBadge: () => void
```

在 `create<ExecutionState>(...)` 的初始 state，`taskPanelOpen: false,` 之后加：
```ts
  taskIconBadge: 0,
```

在 `toggleTaskPanel` 实现之后加：
```ts
  bumpTaskBadge: () => set((s) => ({ taskIconBadge: s.taskIconBadge + 1 })),

  clearTaskBadge: () => set({ taskIconBadge: 0 }),
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd frontend && npm run test -- src/stores/execution.test.ts`
Expected: 两个用例 PASS。

- [ ] **Step 5: Commit**

```bash
cd /media/heygo/Program/projects-code/_playground/nous-center
git add frontend/src/stores/execution.ts frontend/src/stores/execution.test.ts
git commit -m "feat(frontend): execution store taskIconBadge for Run feedback

bumpTaskBadge on Run / clearTaskBadge on panel open. Decoupled from
the running-task count — it expresses 'there are unviewed
submissions'. V1.5 Lane I, spec 6.3 DD4."
```

---

## Task 6: `Topbar.handleRun` 接 Lane S 202 异步契约 —— Run 反馈

spec §6.3 DD4：D17 异步后，点 Run → toast「任务已入队 · image runner」+ badge 计数；**面板不自动打开**。当前 `handleRun` 调 `executeWorkflow`（同步阻塞到完成）。Lane S 把 `/api/v1/workflows/execute` 改成返回 `202 {task_id}`。

> **注意**：Lane S 重写 `executeWorkflow` 的后端契约时，前端 `workflowExecutor.ts` 的 `executeOnBackend` 也会被 Lane S 一并改成「POST → 拿 task_id → 不阻塞」。Lane I 这里只改 `Topbar.handleRun` 的**反馈** UX（toast 文案 + badge + 不自动开面板），不重写 `executeWorkflow` 本身。若 Lane S 尚未 merge，`executeWorkflow` 仍是旧的同步签名 —— 本 Task 的改动对「旧同步 / 新异步」两种签名都成立（见 Step 3 的兼容写法）。

**Files:**
- Modify: `frontend/src/components/layout/Topbar.tsx`
- Test: `frontend/src/components/layout/Topbar.test.tsx`（若已存在则追加；不存在则新建，仅覆盖 handleRun 反馈）

- [ ] **Step 1: 确认 Topbar 是否已有测试**

Run: `cd frontend && ls src/components/layout/Topbar.test.tsx 2>/dev/null && echo EXISTS || echo NEW`
Expected: `NEW`（当前无 Topbar 测试）或 `EXISTS`。下面 Step 2 按结果决定新建还是追加。

- [ ] **Step 2: 写失败测试 —— 点 Run 发「已入队」toast + bump badge + 不开面板**

新建（或追加到）`frontend/src/components/layout/Topbar.test.tsx`：
```tsx
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import Topbar from './Topbar'
import { useExecutionStore } from '../../stores/execution'

const toastAdd = vi.fn()
vi.mock('../../stores/toast', () => ({
  useToastStore: (sel: (s: { add: typeof toastAdd }) => unknown) => sel({ add: toastAdd }),
}))

// executeWorkflow 现在返回 202 契约的 { task_id }（Lane S）。
const executeWorkflow = vi.fn()
vi.mock('../../utils/workflowExecutor', () => ({
  executeWorkflow: (...a: unknown[]) => executeWorkflow(...a),
}))

vi.mock('../../api/workflows', () => ({ useUnpublishWorkflow: () => ({ mutate: vi.fn() }) }))

describe('Topbar handleRun — async 202 feedback', () => {
  beforeEach(() => {
    toastAdd.mockClear()
    executeWorkflow.mockReset()
    useExecutionStore.setState({ taskIconBadge: 0, taskPanelOpen: false, isRunning: false })
  })

  it('on Run: shows enqueued toast, bumps badge, does NOT open the panel', async () => {
    executeWorkflow.mockResolvedValue({ task_id: 'wf_exec_9z' })
    render(
      <MemoryRouter>
        <Topbar />
      </MemoryRouter>,
    )
    fireEvent.click(screen.getByText(/Run/))
    await waitFor(() => expect(executeWorkflow).toHaveBeenCalled())
    expect(toastAdd).toHaveBeenCalledWith(expect.stringContaining('入队'), 'info')
    expect(useExecutionStore.getState().taskIconBadge).toBe(1)
    // 面板不自动打开 —— DD4
    expect(useExecutionStore.getState().taskPanelOpen).toBe(false)
  })
})
```
> 若 `Topbar` 渲染依赖更多 store（`useWorkspaceStore` 等），按测试运行时报错补齐对应 `vi.mock` —— 对齐 `TaskPanel.test.tsx` 既有的 mock 风格。Step 3 改完后若仍有渲染依赖未 mock，补 mock 直到测试聚焦在 handleRun 行为上。

- [ ] **Step 3: 跑测试确认失败**

Run: `cd frontend && npm run test -- src/components/layout/Topbar.test.tsx`
Expected: FAIL —— 当前 `handleRun` 发的是「生成完成」toast、不 bump badge。

- [ ] **Step 4: 改 `Topbar.handleRun`**

`frontend/src/components/layout/Topbar.tsx`：

`useExecutionStore()` 解构加 `bumpTaskBadge`：
```tsx
  const { isRunning, progress, currentNodeType, start, succeed, fail, resetNodeStates, bumpTaskBadge } = useExecutionStore()
```

把 `handleRun` 整个函数体替换为：
```tsx
  const handleRun = async () => {
    if (isRunning) return
    const workflow = getActiveWorkflow()
    start()

    try {
      // Lane S（D17）：executeWorkflow 入队后立即返回 { task_id }，不再阻塞到完成。
      // 反馈 UX（spec §6.3 DD4）：toast「已入队」+ IconRail badge 计数 +
      // 面板【不】自动打开（toast 带「查看」语义由 ToastContainer 承接，
      // 用户点 IconRail 任务图标进面板）。
      const result = await executeWorkflow(workflow)
      const taskId = (result as { task_id?: string })?.task_id
      bumpTaskBadge()
      toast(taskId ? `任务已入队 · ${taskId}` : '任务已入队', 'info')
      // 入队即结束「本次 Run」的 UI busy 态 —— 后续进度由 TaskPanel 泳道接管。
      succeed(null)
      resetNodeStates()
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Unknown error'
      fail(msg)
      toast(msg, 'error')
      setTimeout(() => resetNodeStates(), 5000)
    }
  }
```
> 说明：`succeed` 的签名是 `(result: ExecutionState['result']) => void`，`ExecutionState['result']` 是 `{...} | null` —— 传 `null` 合法（异步契约下 Topbar 不再持有最终 result，结果由 TaskPanel 展示）。`updateNode(outputNode, ...)` 那段删掉 —— 异步契约下 Run 时刻拿不到 audio 结果，输出节点的回填由 WS 进度推送 / TaskPanel 承接（Lane S / 既有 `workflowExecutor` WS 通道负责）。

- [ ] **Step 5: 跑测试确认通过 + tsc**

Run: `cd frontend && npm run test -- src/components/layout/Topbar.test.tsx && npx tsc -b --noEmit`
Expected: 测试 PASS，tsc 无报错（若 `result` 未使用导致 lint warning，确认 `updateNode` 那段已删干净）。

- [ ] **Step 6: Commit**

```bash
cd /media/heygo/Program/projects-code/_playground/nous-center
git add frontend/src/components/layout/Topbar.tsx frontend/src/components/layout/Topbar.test.tsx
git commit -m "feat(frontend): Topbar handleRun async 202 feedback

Run -> enqueued toast + IconRail badge bump, panel does NOT auto-open
(spec DD4). Aligns with Lane S /run 202 contract: executeWorkflow
returns { task_id } instead of blocking to completion. V1.5 Lane I,
spec 6.3."
```

---

## Task 7: `IconRail.TaskRailButton` badge 改读 `taskIconBadge`

spec §6.3 DD4：IconRail 任务图标 badge = 未查看的新提交计数。当前 `TaskRailButton` 数的是 `running` task 数；改成读 `taskIconBadge`，点开面板时 `clearTaskBadge`。

**Files:**
- Modify: `frontend/src/components/layout/IconRail.tsx`
- Test: `frontend/src/components/layout/IconRail.test.tsx`（新建，仅覆盖 `TaskRailButton`）

- [ ] **Step 1: 写失败测试 —— badge 读 taskIconBadge + 点击清零**

新建 `frontend/src/components/layout/IconRail.test.tsx`：
```tsx
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import IconRail from './IconRail'
import { useExecutionStore } from '../../stores/execution'

vi.mock('../../api/tasks', () => ({ useTasks: () => ({ data: [] }) }))
vi.mock('../../api/admin', () => ({
  useAdminLogout: () => ({ mutate: vi.fn(), isPending: false }),
  useAdminMe: () => ({ data: { login_required: false } }),
}))

describe('IconRail TaskRailButton — badge from taskIconBadge', () => {
  beforeEach(() => {
    useExecutionStore.setState({ taskIconBadge: 0, taskPanelOpen: false })
  })

  it('shows the taskIconBadge count (not running-task count)', () => {
    useExecutionStore.setState({ taskIconBadge: 3 })
    render(<MemoryRouter><IconRail /></MemoryRouter>)
    expect(screen.getByText('3')).toBeTruthy()
  })

  it('clicking the Tasks button clears the badge', () => {
    useExecutionStore.setState({ taskIconBadge: 2 })
    render(<MemoryRouter><IconRail /></MemoryRouter>)
    fireEvent.click(screen.getByLabelText('Tasks'))
    expect(useExecutionStore.getState().taskIconBadge).toBe(0)
  })

  it('hides the badge when count is 0', () => {
    render(<MemoryRouter><IconRail /></MemoryRouter>)
    expect(screen.queryByText('0')).toBeNull()
  })
})
```
> `getByLabelText('Tasks')` 依赖 `TaskRailButton` 的按钮有 `aria-label="Tasks"` —— 当前 `RailButton` 只在 hover tooltip 显示 label，没有 `aria-label`。Step 3 会补 `aria-label`（同时满足 a11y 清单「键盘可达 + 可标注」）。

- [ ] **Step 2: 跑测试确认失败**

Run: `cd frontend && npm run test -- src/components/layout/IconRail.test.tsx`
Expected: FAIL —— badge 还在数 `runningCount`；`getByLabelText('Tasks')` 找不到（无 aria-label）。

- [ ] **Step 3: 改 `IconRail.tsx` 的 `TaskRailButton` + `RailButton`**

`frontend/src/components/layout/IconRail.tsx`：

`TaskRailButton` 整个函数替换为：
```tsx
function TaskRailButton() {
  const { taskPanelOpen, toggleTaskPanel, taskIconBadge, clearTaskBadge } = useExecutionStore()

  const handleClick = () => {
    // 打开面板即视为「已查看」—— 清掉未查看计数（spec §6.3 DD4）。
    if (!taskPanelOpen) clearTaskBadge()
    toggleTaskPanel()
  }

  return (
    <div className="relative">
      <RailButton active={taskPanelOpen} onClick={handleClick} label="Tasks">
        <ListTodo size={18} />
      </RailButton>
      {taskIconBadge > 0 && (
        <span
          aria-label={`${taskIconBadge} 个新任务`}
          className="absolute pointer-events-none"
          style={{
            top: 2,
            right: 2,
            minWidth: 14,
            height: 14,
            borderRadius: 7,
            background: 'var(--accent)',
            color: '#fff',
            fontSize: 9,
            fontWeight: 600,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            padding: '0 3px',
          }}
        >
          {taskIconBadge}
        </span>
      )}
    </div>
  )
}
```
（`useTasks` import 若 `IconRail.tsx` 已无其它使用处，一并删掉那行 import —— 跑 tsc 会提示 unused。）

`RailButton` 组件给 `<button>` 加 `aria-label`（a11y 清单「键盘可达 + 可标注」），把 `<button onClick={onClick} className=...>` 改为：
```tsx
    <button
      onClick={onClick}
      aria-label={label}
      className="group relative flex items-center justify-center mb-0.5"
```
（其余 `RailButton` 内容不变；`label` 已是入参。）

- [ ] **Step 4: 跑测试确认通过 + tsc**

Run: `cd frontend && npm run test -- src/components/layout/IconRail.test.tsx && npx tsc -b --noEmit`
Expected: 三个用例 PASS，tsc 无报错（确认 `useTasks` unused import 已删）。

- [ ] **Step 5: Commit**

```bash
cd /media/heygo/Program/projects-code/_playground/nous-center
git add frontend/src/components/layout/IconRail.tsx frontend/src/components/layout/IconRail.test.tsx
git commit -m "feat(frontend): IconRail task badge reads taskIconBadge, clears on open

Badge now counts unviewed submissions (bumped on Run) instead of
running tasks; opening the panel clears it. RailButton gains an
aria-label for keyboard reachability (a11y checklist). V1.5 Lane I,
spec 6.3/6.6."
```

---

## Task 8: TaskPanel 重写 — Buildkite 风结构骨架（DD3 + 响应式 + a11y）

spec §6.1 DD3：从「3-tab 扁平列表」重构为「顶部 per-runner 泳道区 + 下方最近完成列表」。本 Task 建结构骨架（泳道区 + 最近完成 + 响应式 + a11y），异常态 / 排队展开 / 缩略图在 Task 9-11 逐个加。

**Files:**
- Rewrite: `frontend/src/components/panels/TaskPanel.tsx`
- Rewrite: `frontend/src/components/panels/TaskPanel.test.tsx`

- [ ] **Step 1: 写失败测试 —— 新结构（泳道区 + 最近完成 + 响应式 + a11y）**

`frontend/src/components/panels/TaskPanel.test.tsx` 整个替换为：
```tsx
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import TaskPanel from './TaskPanel'

// Lane I: Buildkite 风结构 — per-runner 泳道区（hero）+ 最近完成列表。

const mockTasks = [
  {
    id: 'wf_done_1', workflow_name: 'flux2-人物立绘', status: 'completed',
    nodes_done: 2, nodes_total: 2, duration_ms: 34000, created_at: new Date().toISOString(),
    error: null, result: null, current_node: null, task_type: 'image',
    image_width: 1024, image_height: 1024, output_thumbnails: ['/files/outputs/wf_done_1/0.webp'],
  },
  {
    id: 'wf_done_2', workflow_name: 'cosy-旁白', status: 'completed',
    nodes_done: 1, nodes_total: 1, duration_ms: 8000, created_at: new Date().toISOString(),
    error: null, result: null, current_node: null, task_type: null,
    image_width: null, image_height: null,
  },
]

const mockRunners = [
  {
    id: 'runner-i', label: 'Runner-I', role: 'image', state: 'busy',
    current_task: { task_id: 'wf_run_x', workflow_name: 'flux2-人物立绘', progress: 0.6, detail: 'step 18/30' },
    queue: [{ task_id: 'q1', workflow_name: 'sd-背景', position: 1 }],
    restart_attempt: null, load_error: null,
  },
  {
    id: 'runner-l', label: 'Runner-L', role: 'llm', state: 'idle',
    current_task: null, queue: [], restart_attempt: null, load_error: null,
  },
]

vi.mock('../../api/tasks', () => ({
  useTasks: () => ({ data: mockTasks }),
  useCancelTask: () => ({ mutate: vi.fn(), isPending: false }),
  useRetryTask: () => ({ mutate: vi.fn(), isPending: false }),
  useDeleteTask: () => ({ mutate: vi.fn(), isPending: false }),
}))

let runnersData: unknown = mockRunners
vi.mock('../../api/runners', () => ({
  useRunners: () => ({ data: runnersData }),
}))

function withQuery(ui: React.ReactNode) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return <QueryClientProvider client={qc}>{ui}</QueryClientProvider>
}

describe('TaskPanel — Buildkite-style structure (DD3)', () => {
  beforeEach(() => {
    runnersData = mockRunners
    window.innerWidth = 1280
  })

  it('hidden when open=false', () => {
    render(withQuery(<TaskPanel open={false} onClose={() => {}} />))
    expect(screen.queryByText('任务面板')).toBeNull()
  })

  it('renders one lane per runner with label + role', () => {
    render(withQuery(<TaskPanel open={true} onClose={() => {}} />))
    expect(screen.getByText('Runner-I')).toBeTruthy()
    expect(screen.getByText('Runner-L')).toBeTruthy()
    // role 标注
    expect(screen.getByText(/image/)).toBeTruthy()
    expect(screen.getByText(/llm/)).toBeTruthy()
  })

  it('busy lane shows current task name + progress detail', () => {
    render(withQuery(<TaskPanel open={true} onClose={() => {}} />))
    expect(screen.getByText('flux2-人物立绘')).toBeTruthy()
    expect(screen.getByText('step 18/30')).toBeTruthy()
  })

  it('idle lane shows the "idle" text label (a11y: text not just color)', () => {
    render(withQuery(<TaskPanel open={true} onClose={() => {}} />))
    expect(screen.getByText('idle')).toBeTruthy()
  })

  it('renders a "最近完成" section with completed tasks', () => {
    render(withQuery(<TaskPanel open={true} onClose={() => {}} />))
    expect(screen.getByText('最近完成')).toBeTruthy()
    expect(screen.getByText('cosy-旁白')).toBeTruthy()
  })

  it('shows an empty-runner hint when /api/v1/runners returned []', () => {
    runnersData = []
    render(withQuery(<TaskPanel open={true} onClose={() => {}} />))
    expect(screen.getByText(/暂无 runner/)).toBeTruthy()
  })

  it('close button is a real button with aria-label', () => {
    render(withQuery(<TaskPanel open={true} onClose={() => {}} />))
    const closeBtn = screen.getByLabelText('关闭')
    expect(closeBtn.tagName).toBe('BUTTON')
  })

  it('drawer is fullscreen-width under 768px (DD7 responsive)', () => {
    window.innerWidth = 600
    render(withQuery(<TaskPanel open={true} onClose={() => {}} />))
    const drawer = screen.getByRole('complementary') // <aside>
    // <768px: 宽度 100vw（全屏），>=768px: 460px
    expect(drawer.style.width).toBe('100vw')
  })
})
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd frontend && npm run test -- src/components/panels/TaskPanel.test.tsx`
Expected: FAIL —— 旧 `TaskPanel` 还是 3-tab 结构，没有 `Runner-I` / `最近完成` / 响应式宽度。

- [ ] **Step 3: 重写 `TaskPanel.tsx` —— 骨架**

`frontend/src/components/panels/TaskPanel.tsx` 整个替换为：
```tsx
import { useEffect, useMemo, useState } from 'react'
import {
  AlertCircle,
  Ban,
  CheckCircle2,
  Image as ImageIcon,
  Loader2,
  RotateCcw,
  Trash2,
  X,
} from 'lucide-react'
import {
  useCancelTask,
  useDeleteTask,
  useRetryTask,
  useTasks,
  type ExecutionTask,
} from '../../api/tasks'
import { useRunners, type RunnerInfo } from '../../api/runners'

// Lane I（spec §6）：TaskPanel 从「3-tab 扁平列表」重构为 Buildkite 风：
//   - 顶部 per-runner 泳道区（视觉 hero）—— 每条泳道 = 一个 GPU runner 的
//     当前任务 + 进度条 + 排队数（可展开）+ 异常态内联。
//   - 下方「最近完成」列表 —— image 任务带输出缩略图。
// 响应式：<768px 抽屉变全屏、泳道堆叠。a11y：折叠 toggle 用真 button +
// aria-expanded、状态文字始终伴随色点、动作键盘可达。

const TERMINAL: ReadonlySet<ExecutionTask['status']> = new Set([
  'completed',
  'failed',
  'cancelled',
])

/** <768px 判定 —— 抽屉全屏 + 泳道堆叠（DD7）。 */
function useIsNarrow(): boolean {
  const [narrow, setNarrow] = useState(
    typeof window !== 'undefined' ? window.innerWidth < 768 : false,
  )
  useEffect(() => {
    const onResize = () => setNarrow(window.innerWidth < 768)
    window.addEventListener('resize', onResize)
    return () => window.removeEventListener('resize', onResize)
  }, [])
  return narrow
}

export default function TaskPanel({ open, onClose }: { open: boolean; onClose: () => void }) {
  const { data: tasks } = useTasks()
  const { data: runners } = useRunners()
  const isNarrow = useIsNarrow()

  const recent = useMemo(
    () => (tasks ?? []).filter((t) => TERMINAL.has(t.status)),
    [tasks],
  )

  if (!open) return null

  return (
    <>
      {/* 半透明遮罩 —— 点击关闭。<768px 全屏抽屉时遮罩仍铺满（点不到也无妨）。 */}
      <div
        onClick={onClose}
        style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.35)', zIndex: 49 }}
      />

      <aside
        style={{
          position: 'fixed',
          right: 0,
          top: 0,
          bottom: 0,
          // DD7 响应式：<768px 全屏，否则 460px 抽屉。
          width: isNarrow ? '100vw' : 460,
          background: 'var(--bg-accent)',
          borderLeft: isNarrow ? 'none' : '1px solid var(--border)',
          boxShadow: isNarrow ? 'none' : '-8px 0 24px rgba(0,0,0,0.3)',
          zIndex: 50,
          display: 'flex',
          flexDirection: 'column',
        }}
      >
        {/* header */}
        <div
          style={{
            padding: '14px 18px',
            borderBottom: '1px solid var(--border)',
            display: 'flex',
            alignItems: 'center',
            gap: 10,
          }}
        >
          <h2 style={{ flex: 1, fontSize: 14, fontWeight: 600, color: 'var(--text)' }}>
            任务面板
          </h2>
          <button
            type="button"
            onClick={onClose}
            aria-label="关闭"
            style={{
              width: 24,
              height: 24,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              background: 'transparent',
              border: 'none',
              color: 'var(--muted)',
              borderRadius: 4,
              cursor: 'pointer',
            }}
          >
            <X size={14} />
          </button>
        </div>

        {/* body —— 泳道区（hero）+ 最近完成 */}
        <div style={{ flex: 1, overflowY: 'auto', padding: '14px 18px' }}>
          {/* per-runner 泳道区 */}
          <section aria-label="GPU runner 泳道" style={{ marginBottom: 18 }}>
            {(runners ?? []).length === 0 ? (
              <div
                style={{
                  padding: 24,
                  textAlign: 'center',
                  fontSize: 12,
                  color: 'var(--muted)',
                  border: '1px dashed var(--border)',
                  borderRadius: 6,
                }}
              >
                暂无 runner 数据（调度器未就绪或端点未上线）
              </div>
            ) : (
              <div
                style={{
                  border: '1px solid var(--border)',
                  borderRadius: 6,
                  overflow: 'hidden',
                }}
              >
                {(runners ?? []).map((r) => (
                  <RunnerLane key={r.id} runner={r} />
                ))}
              </div>
            )}
          </section>

          {/* 最近完成 */}
          <section aria-label="最近完成">
            <div
              style={{
                fontSize: 11,
                color: 'var(--muted)',
                textTransform: 'uppercase',
                letterSpacing: 0.5,
                marginBottom: 8,
              }}
            >
              最近完成
            </div>
            {recent.length === 0 ? (
              <div
                style={{
                  padding: 24,
                  textAlign: 'center',
                  fontSize: 12,
                  color: 'var(--muted)',
                }}
              >
                还没有完成的任务。
              </div>
            ) : (
              recent.map((t) => <RecentRow key={t.id} task={t} />)
            )}
          </section>
        </div>
      </aside>
    </>
  )
}

// ---------- runner 泳道（Task 9 加异常态 + Task 10 加排队展开）----------

function RunnerLane({ runner }: { runner: RunnerInfo }) {
  return (
    <div
      style={{
        padding: '12px 14px',
        borderBottom: '1px solid var(--border)',
        background: 'var(--bg)',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <RunnerStateDot state={runner.state} />
        <span style={{ fontSize: 13, fontWeight: 600, color: 'var(--text)' }}>
          {runner.label}
        </span>
        <span style={{ fontSize: 11, color: 'var(--muted)' }}>({runner.role})</span>
        <span style={{ flex: 1 }} />
        {/* 状态文字始终伴随色点（a11y：色盲不能只靠颜色）。 */}
        <RunnerStateText runner={runner} />
      </div>

      {runner.state === 'busy' && runner.current_task && (
        <div style={{ marginTop: 8 }}>
          <div
            style={{
              fontSize: 12,
              color: 'var(--text)',
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              whiteSpace: 'nowrap',
            }}
          >
            {runner.current_task.workflow_name}
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 6 }}>
            <div
              style={{
                flex: 1,
                height: 4,
                background: 'var(--bg-accent)',
                borderRadius: 2,
                overflow: 'hidden',
              }}
            >
              <div
                style={{
                  width: `${Math.round((runner.current_task.progress ?? 0) * 100)}%`,
                  height: '100%',
                  background: 'var(--accent-2, #22c55e)',
                  transition: 'width 0.3s ease',
                }}
              />
            </div>
            {runner.current_task.detail && (
              <span
                style={{
                  fontSize: 11,
                  color: 'var(--muted)',
                  fontVariantNumeric: 'tabular-nums',
                }}
              >
                {runner.current_task.detail}
              </span>
            )}
          </div>
        </div>
      )}

      {/* 排队数 —— Task 10 替换为可展开的 QueueExpand。 */}
      {runner.queue.length > 0 && (
        <div style={{ marginTop: 8, fontSize: 11, color: 'var(--muted)' }}>
          排队 {runner.queue.length}
        </div>
      )}
    </div>
  )
}

function RunnerStateDot({ state }: { state: RunnerInfo['state'] }) {
  const color =
    state === 'busy'
      ? 'var(--accent-2, #22c55e)'
      : state === 'restarting'
        ? 'var(--warn, #f59e0b)'
        : state === 'load_failed'
          ? 'var(--accent, #ef4444)'
          : 'var(--muted)'
  return (
    <span
      aria-hidden="true"
      style={{
        width: 8,
        height: 8,
        borderRadius: '50%',
        background: color,
        flexShrink: 0,
        // restarting 态脉冲（Task 9 接入 keyframes）。
        animation: state === 'restarting' ? 'pulse 1.2s ease-in-out infinite' : undefined,
      }}
    />
  )
}

function RunnerStateText({ runner }: { runner: RunnerInfo }) {
  // Task 9 把 restarting / load_failed 的文案 + Retry 按钮做全；
  // 本 Task 先给 idle / busy 的文字。
  if (runner.state === 'idle') {
    return <span style={{ fontSize: 11, color: 'var(--muted)' }}>idle</span>
  }
  if (runner.state === 'busy') {
    return <span style={{ fontSize: 11, color: 'var(--accent-2, #22c55e)' }}>busy</span>
  }
  if (runner.state === 'restarting') {
    return <span style={{ fontSize: 11, color: 'var(--warn, #f59e0b)' }}>重启中</span>
  }
  return <span style={{ fontSize: 11, color: 'var(--accent, #ef4444)' }}>加载失败</span>
}

// ---------- 最近完成列表行（Task 11 加缩略图）----------

function RecentRow({ task }: { task: ExecutionTask }) {
  const cancelTask = useCancelTask()
  const retryTask = useRetryTask()
  const deleteTask = useDeleteTask()

  const isFailed = task.status === 'failed'
  const isCancelled = task.status === 'cancelled'
  const canRetry = isFailed || isCancelled

  return (
    <div
      style={{
        background: 'var(--bg)',
        border: '1px solid var(--border)',
        borderRadius: 6,
        padding: '10px 12px',
        marginBottom: 8,
        display: 'flex',
        alignItems: 'center',
        gap: 10,
      }}
    >
      {/* Task 11：image 任务这里放缩略图，否则状态图标。 */}
      <RecentLeading task={task} />
      <div style={{ flex: 1, minWidth: 0 }}>
        <div
          style={{
            fontSize: 13,
            fontWeight: 500,
            color: 'var(--text)',
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
          }}
        >
          {task.workflow_name || '未命名任务'}
        </div>
        <div style={{ fontSize: 11, color: 'var(--muted)', marginTop: 2 }}>
          {statusLabel(task.status)}
          {task.duration_ms != null && task.duration_ms > 0
            ? ` · ${formatDuration(task.duration_ms)}`
            : ''}
        </div>
        {task.error && (
          <div
            style={{
              marginTop: 6,
              fontSize: 11,
              color: 'var(--accent, #ef4444)',
              wordBreak: 'break-all',
            }}
          >
            {task.error}
          </div>
        )}
      </div>
      <div style={{ display: 'flex', gap: 6 }}>
        {canRetry && (
          <ActionBtn
            onClick={() => retryTask.mutate(task.id)}
            disabled={retryTask.isPending}
            icon={<RotateCcw size={11} />}
            label="重试"
          />
        )}
        <ActionBtn
          onClick={() => deleteTask.mutate(task.id)}
          disabled={deleteTask.isPending}
          icon={<Trash2 size={11} />}
          label="删除"
        />
        {task.status === 'queued' || task.status === 'running' ? (
          <ActionBtn
            onClick={() => cancelTask.mutate(task.id)}
            disabled={cancelTask.isPending}
            icon={<Ban size={11} />}
            label="取消"
            danger
          />
        ) : null}
      </div>
    </div>
  )
}

function RecentLeading({ task }: { task: ExecutionTask }) {
  // Task 11 用 ImageThumb 替换；本 Task 先放状态图标。
  if (task.status === 'completed') {
    return <CheckCircle2 size={16} style={{ color: 'var(--accent-2, #22c55e)', flexShrink: 0 }} />
  }
  if (task.status === 'failed') {
    return <AlertCircle size={16} style={{ color: 'var(--accent, #ef4444)', flexShrink: 0 }} />
  }
  return <Loader2 size={16} style={{ color: 'var(--muted)', flexShrink: 0 }} />
}

function ActionBtn({
  onClick,
  disabled,
  icon,
  label,
  danger,
}: {
  onClick: () => void
  disabled?: boolean
  icon: React.ReactNode
  label: string
  danger?: boolean
}) {
  return (
    <button
      type="button"
      onClick={(e) => {
        e.stopPropagation()
        onClick()
      }}
      disabled={disabled}
      aria-label={label}
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: 4,
        padding: '4px 10px',
        fontSize: 11,
        borderRadius: 4,
        border: '1px solid var(--border)',
        background: 'transparent',
        color: danger ? 'var(--accent, #ef4444)' : 'var(--text)',
        cursor: disabled ? 'wait' : 'pointer',
        opacity: disabled ? 0.5 : 1,
      }}
    >
      {icon}
      {label}
    </button>
  )
}

// ---------- helpers ----------

function statusLabel(status: string): string {
  const map: Record<string, string> = {
    completed: '已完成',
    failed: '失败',
    cancelled: '已取消',
    running: '运行中',
    queued: '排队',
  }
  return map[status] ?? status
}

function formatDuration(ms: number): string {
  if (ms < 1000) return `${ms}ms`
  const s = ms / 1000
  if (s < 60) return `${s.toFixed(1)}s`
  const m = Math.floor(s / 60)
  return `${m}m${Math.round(s - m * 60)}s`
}

// 占位：ImageIcon 在 Task 11 被 ImageThumb 用到，先 re-export 防 unused-import。
export const _ImageIcon = ImageIcon
```

> 说明：`_ImageIcon` export 是临时的 —— Task 11 把 `ImageThumb` 加进来后会真正用 `ImageIcon`，届时删掉这行。这样 Task 8 的 tsc 不会因 unused import 报错。`pulse` keyframes 在 Task 9 补（当前 `restarting` 态尚无 runner mock 命中，不影响 Task 8 测试）。

- [ ] **Step 4: 跑测试确认通过**

Run: `cd frontend && npm run test -- src/components/panels/TaskPanel.test.tsx`
Expected: 8 个用例全 PASS。

- [ ] **Step 5: tsc 确认无回归**

Run: `cd frontend && npx tsc -b --noEmit`
Expected: 无报错。

- [ ] **Step 6: Commit**

```bash
cd /media/heygo/Program/projects-code/_playground/nous-center
git add frontend/src/components/panels/TaskPanel.tsx frontend/src/components/panels/TaskPanel.test.tsx
git commit -m "refactor(frontend): rewrite TaskPanel to Buildkite-style runner lanes

Replaces the 3-tab flat list with a per-runner lane section (hero) +
a recent-completed list. Responsive: fullscreen drawer + stacked
lanes under 768px. a11y: real button + aria-label on close, state
text always paired with the color dot, section aria-labels. Abnormal
states / queue expand / thumbnails land in follow-up tasks. V1.5
Lane I, spec 6.1/6.5/6.6."
```

---

## Task 9: runner 泳道异常态 — 重启中脉冲 + 加载失败 + Retry（DD5）

spec §6.2 DD5：`restarting` = 黄色脉冲点 + 「重启中 2/4」；`load_failed` = 红点 + 「加载失败: qwen3-35b OOM」+ Retry 按钮。

**Files:**
- Modify: `frontend/src/components/panels/TaskPanel.tsx`（`RunnerStateText` + 加 Retry + `pulse` keyframes）
- Modify: `frontend/src/api/runners.ts`（加 `useRetryRunner` mutation）
- Modify: `frontend/src/components/panels/TaskPanel.test.tsx`（追加异常态用例）

- [ ] **Step 1: 写失败测试 —— 重启中文案 + 加载失败文案 + Retry 按钮**

在 `frontend/src/components/panels/TaskPanel.test.tsx` 末尾追加一个 describe：
```tsx
describe('TaskPanel — runner abnormal states (DD5)', () => {
  beforeEach(() => {
    window.innerWidth = 1280
  })

  it('restarting lane shows "重启中 N/M" with attempt numbers', () => {
    runnersData = [
      {
        id: 'runner-t', label: 'Runner-T', role: 'tts', state: 'restarting',
        current_task: null, queue: [], restart_attempt: [2, 4], load_error: null,
      },
    ]
    render(withQuery(<TaskPanel open={true} onClose={() => {}} />))
    expect(screen.getByText('重启中 2/4')).toBeTruthy()
  })

  it('load_failed lane shows the error text + a keyboard-reachable Retry button', () => {
    runnersData = [
      {
        id: 'runner-l', label: 'Runner-L', role: 'llm', state: 'load_failed',
        current_task: null, queue: [], restart_attempt: null,
        load_error: 'qwen3-35b OOM',
      },
    ]
    render(withQuery(<TaskPanel open={true} onClose={() => {}} />))
    expect(screen.getByText(/qwen3-35b OOM/)).toBeTruthy()
    const retry = screen.getByRole('button', { name: '重试加载' })
    expect(retry.tagName).toBe('BUTTON')
  })
})
```
> 该 describe 复用上一个 describe 的 `runnersData` / `withQuery` / mock —— 它们在同一文件作用域。`runnersData` 是 `let`，`beforeEach` 在两个 describe 各自设置；本 describe 的 `beforeEach` 只重置 `innerWidth`，每个 `it` 自己设 `runnersData`。

- [ ] **Step 2: 跑测试确认失败**

Run: `cd frontend && npm run test -- src/components/panels/TaskPanel.test.tsx`
Expected: 新增 2 个用例 FAIL —— `RunnerStateText` 现在只渲染「重启中」/「加载失败」纯文字，没有 `2/4`、没有 `load_error` 文案、没有 Retry 按钮。

- [ ] **Step 3: 给 `runners.ts` 加 `useRetryRunner`**

`frontend/src/api/runners.ts` 顶部 import 改为：
```ts
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { apiFetch } from './client'
```
文件末尾追加：
```ts
/** 触发某 runner 重新加载失败的模型（spec §6.2 DD5 的 Retry 按钮）。
 * 后端 POST /api/v1/runners/{id}/retry 由 Lane H 提供。 */
export function useRetryRunner() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (runnerId: string) =>
      apiFetch(`/api/v1/runners/${runnerId}/retry`, { method: 'POST' }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['runners'] }),
  })
}
```

- [ ] **Step 4: 改 `TaskPanel.tsx` —— 异常态文案 + Retry + pulse keyframes**

`frontend/src/components/panels/TaskPanel.tsx`：

import 行加 `useRetryRunner`：
```tsx
import { useRunners, useRetryRunner, type RunnerInfo } from '../../api/runners'
```

`RunnerStateText` 整个函数替换为：
```tsx
function RunnerStateText({ runner }: { runner: RunnerInfo }) {
  if (runner.state === 'idle') {
    return <span style={{ fontSize: 11, color: 'var(--muted)' }}>idle</span>
  }
  if (runner.state === 'busy') {
    return <span style={{ fontSize: 11, color: 'var(--accent-2, #22c55e)' }}>busy</span>
  }
  if (runner.state === 'restarting') {
    // restart_attempt = [当前第几次, 总次数] → 「重启中 2/4」（spec §6.2）。
    const attempt = runner.restart_attempt
    return (
      <span style={{ fontSize: 11, color: 'var(--warn, #f59e0b)' }}>
        {attempt ? `重启中 ${attempt[0]}/${attempt[1]}` : '重启中'}
      </span>
    )
  }
  // load_failed —— 文案 + Retry 在 RunnerLane 里渲染（这里只给状态词）。
  return <span style={{ fontSize: 11, color: 'var(--accent, #ef4444)' }}>加载失败</span>
}
```

`RunnerLane` 函数体内，在「排队数」那段 `{runner.queue.length > 0 && (...)}` **之前**插入 load_failed 区块：
```tsx
      {runner.state === 'load_failed' && (
        <div
          style={{
            marginTop: 8,
            display: 'flex',
            alignItems: 'center',
            gap: 8,
            padding: '6px 10px',
            borderRadius: 4,
            background: 'rgba(239,68,68,0.08)',
            border: '1px solid rgba(239,68,68,0.25)',
          }}
        >
          <span style={{ flex: 1, fontSize: 11, color: 'var(--accent, #ef4444)' }}>
            加载失败: {runner.load_error ?? '未知错误'}
          </span>
          <RunnerRetryButton runnerId={runner.id} />
        </div>
      )}
```

在 `RunnerStateDot` 函数**之后**加 `RunnerRetryButton`：
```tsx
function RunnerRetryButton({ runnerId }: { runnerId: string }) {
  const retry = useRetryRunner()
  return (
    <button
      type="button"
      onClick={() => retry.mutate(runnerId)}
      disabled={retry.isPending}
      aria-label="重试加载"
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: 4,
        padding: '3px 9px',
        fontSize: 11,
        borderRadius: 4,
        border: '1px solid var(--accent, #ef4444)',
        background: 'transparent',
        color: 'var(--accent, #ef4444)',
        cursor: retry.isPending ? 'wait' : 'pointer',
        opacity: retry.isPending ? 0.5 : 1,
      }}
    >
      <RotateCcw size={11} />
      重试加载
    </button>
  )
}
```

`pulse` keyframes —— `TaskPanel.tsx` 用内联 style，keyframes 需注入一个 `<style>`。在 `TaskPanel` 组件 `return` 的 `<>` 内、`遮罩 <div>` 之前加：
```tsx
      {/* restarting 态脉冲动画的 keyframes（内联 style 无法定义 @keyframes）。 */}
      <style>{`@keyframes pulse { 0%,100% { opacity: 1 } 50% { opacity: 0.3 } }`}</style>
```

- [ ] **Step 5: 跑测试确认通过 + tsc**

Run: `cd frontend && npm run test -- src/components/panels/TaskPanel.test.tsx && npx tsc -b --noEmit`
Expected: 全部用例（Task 8 的 8 个 + 本 Task 的 2 个）PASS，tsc 无报错。

- [ ] **Step 6: Commit**

```bash
cd /media/heygo/Program/projects-code/_playground/nous-center
git add frontend/src/components/panels/TaskPanel.tsx frontend/src/api/runners.ts frontend/src/components/panels/TaskPanel.test.tsx
git commit -m "feat(frontend): runner lane abnormal states — restarting pulse + load-failed Retry

restarting shows 重启中 N/M (yellow pulse dot), load_failed shows the
error text + a keyboard-reachable Retry button wired to
useRetryRunner. State text always pairs with the color dot (a11y).
V1.5 Lane I, spec 6.2 DD5."
```

---

## Task 10: 排队位置可展开列表（DD8）

spec §6.4 DD8：runner 泳道的「排队 N」可点击展开成有序列表，每条带序号 #1 #2 #3。a11y：折叠 toggle 用真 `<button>` + `aria-expanded`。

**Files:**
- Modify: `frontend/src/components/panels/TaskPanel.tsx`（`RunnerLane` 的排队段换成 `QueueExpand`）
- Modify: `frontend/src/components/panels/TaskPanel.test.tsx`（追加排队展开用例）

- [ ] **Step 1: 写失败测试 —— 排队 toggle + aria-expanded + 有序列表**

在 `frontend/src/components/panels/TaskPanel.test.tsx` 末尾追加：
```tsx
describe('TaskPanel — queue position expand (DD8)', () => {
  beforeEach(() => {
    window.innerWidth = 1280
    runnersData = [
      {
        id: 'runner-i', label: 'Runner-I', role: 'image', state: 'busy',
        current_task: { task_id: 'cur', workflow_name: 'cur-wf', progress: 0.3, detail: null },
        queue: [
          { task_id: 'q1', workflow_name: 'sd-背景', position: 1 },
          { task_id: 'q2', workflow_name: 'flux-头像', position: 2 },
        ],
        restart_attempt: null, load_error: null,
      },
    ]
  })

  it('queue toggle is a real button with aria-expanded, collapsed by default', () => {
    render(withQuery(<TaskPanel open={true} onClose={() => {}} />))
    const toggle = screen.getByRole('button', { name: /排队 2/ })
    expect(toggle.getAttribute('aria-expanded')).toBe('false')
    // 折叠时列表项不可见
    expect(screen.queryByText('sd-背景')).toBeNull()
  })

  it('clicking the toggle expands an ordered list with #position numbers', async () => {
    const { default: userEventModule } = await import('@testing-library/user-event')
    const user = userEventModule.setup()
    render(withQuery(<TaskPanel open={true} onClose={() => {}} />))
    const toggle = screen.getByRole('button', { name: /排队 2/ })
    await user.click(toggle)
    expect(toggle.getAttribute('aria-expanded')).toBe('true')
    expect(screen.getByText('sd-背景')).toBeTruthy()
    expect(screen.getByText('flux-头像')).toBeTruthy()
    expect(screen.getByText('#1')).toBeTruthy()
    expect(screen.getByText('#2')).toBeTruthy()
  })
})
```
> `@testing-library/user-event` 在 `node_modules` 里随 `@testing-library/react` 16 一起进来（peer）。若 `import` 失败（未装），Step 2 会报错 —— 那就改用 `fireEvent.click`（`TaskPanel.test.tsx` 顶部已 import `fireEvent`）。先按 user-event 写，跑不通再降级。

- [ ] **Step 2: 跑测试确认失败**

Run: `cd frontend && npm run test -- src/components/panels/TaskPanel.test.tsx`
Expected: 新增 2 个用例 FAIL —— 当前排队段是纯文字 `排队 N`，不是 button、无 `aria-expanded`、不可展开。
> 若报 `Cannot find module '@testing-library/user-event'`：把这 2 个用例里的 `user.click(toggle)` 换成 `fireEvent.click(toggle)`，删掉 `userEventModule` 那两行，重跑确认 FAIL。

- [ ] **Step 3: 改 `TaskPanel.tsx` —— `QueueExpand` 组件**

`frontend/src/components/panels/TaskPanel.tsx`：

`RunnerLane` 里把排队段：
```tsx
      {/* 排队数 —— Task 10 替换为可展开的 QueueExpand。 */}
      {runner.queue.length > 0 && (
        <div style={{ marginTop: 8, fontSize: 11, color: 'var(--muted)' }}>
          排队 {runner.queue.length}
        </div>
      )}
```
替换为：
```tsx
      {runner.queue.length > 0 && <QueueExpand queue={runner.queue} />}
```

在 `RunnerRetryButton` 之后加 `QueueExpand`：
```tsx
function QueueExpand({ queue }: { queue: RunnerInfo['queue'] }) {
  const [open, setOpen] = useState(false)
  return (
    <div style={{ marginTop: 8 }}>
      {/* a11y：折叠 toggle 用真 button + aria-expanded（spec §6.6）。 */}
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        style={{
          display: 'inline-flex',
          alignItems: 'center',
          gap: 4,
          padding: '2px 6px',
          fontSize: 11,
          color: 'var(--muted)',
          background: 'transparent',
          border: 'none',
          cursor: 'pointer',
        }}
      >
        排队 {queue.length}
        <span aria-hidden="true" style={{ fontSize: 9 }}>
          {open ? '▾' : '▸'}
        </span>
      </button>
      {open && (
        <ol
          style={{
            listStyle: 'none',
            margin: '6px 0 0',
            padding: 0,
            display: 'flex',
            flexDirection: 'column',
            gap: 4,
          }}
        >
          {queue.map((item) => (
            <li
              key={item.task_id}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 8,
                fontSize: 11,
                color: 'var(--text)',
                padding: '3px 6px',
                background: 'var(--bg-accent)',
                borderRadius: 3,
              }}
            >
              <span
                style={{
                  color: 'var(--muted)',
                  fontVariantNumeric: 'tabular-nums',
                  minWidth: 22,
                }}
              >
                #{item.position}
              </span>
              <span
                style={{
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  whiteSpace: 'nowrap',
                }}
              >
                {item.workflow_name}
              </span>
            </li>
          ))}
        </ol>
      )}
    </div>
  )
}
```
> 说明：spec §6.4 还要「刚提交的 / 当前用户的任务高亮」—— 但单 admin infra（CLAUDE.md：single-admin）下「当前用户」永远是同一个 admin，「当前用户高亮」无意义；「刚提交高亮」需要 Lane S 在 202 response 里回 task_id 后由 TaskPanel 比对。本 Task 不做高亮 —— 已在 Self-Review 标注为有意识的范围裁剪（单 admin 场景下高亮无信息量）。

- [ ] **Step 4: 跑测试确认通过 + tsc**

Run: `cd frontend && npm run test -- src/components/panels/TaskPanel.test.tsx && npx tsc -b --noEmit`
Expected: 全部用例 PASS，tsc 无报错。

- [ ] **Step 5: Commit**

```bash
cd /media/heygo/Program/projects-code/_playground/nous-center
git add frontend/src/components/panels/TaskPanel.tsx frontend/src/components/panels/TaskPanel.test.tsx
git commit -m "feat(frontend): click-to-expand queue position list

排队 N is now a real button with aria-expanded; expands to an ordered
list with #position numbers. Current-user highlight skipped — single
-admin infra makes it information-free. V1.5 Lane I, spec 6.4 DD8."
```

---

## Task 11: image 缩略图历史（DD9）

spec §6.5 DD9：image 类任务完成后，「最近完成」直接显示输出缩略图（数据源 `output_thumbnails`），不只是文字 + ImageIcon。空 → 降级 ImageIcon。

**Files:**
- Modify: `frontend/src/components/panels/TaskPanel.tsx`（`RecentLeading` 换成 `ImageThumb` 分支）
- Modify: `frontend/src/components/panels/TaskPanel.test.tsx`（追加缩略图用例）

- [ ] **Step 1: 写失败测试 —— image 任务显示缩略图 img，无缩略图降级 ImageIcon**

在 `frontend/src/components/panels/TaskPanel.test.tsx` 末尾追加：
```tsx
describe('TaskPanel — image thumbnail history (DD9)', () => {
  beforeEach(() => {
    window.innerWidth = 1280
    runnersData = []
  })

  it('image task with output_thumbnails renders an <img> thumbnail', () => {
    render(withQuery(<TaskPanel open={true} onClose={() => {}} />))
    // mockTasks[0] = flux2-人物立绘，task_type image，带 output_thumbnails
    const thumb = screen.getByAltText(/flux2-人物立绘/) as HTMLImageElement
    expect(thumb.tagName).toBe('IMG')
    expect(thumb.src).toContain('/files/outputs/wf_done_1/0.webp')
  })

  it('non-image completed task falls back to the status icon (no img)', () => {
    render(withQuery(<TaskPanel open={true} onClose={() => {}} />))
    // mockTasks[1] = cosy-旁白，task_type null —— 不应有 img alt 含它的名字
    expect(screen.queryByAltText(/cosy-旁白/)).toBeNull()
  })
})
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd frontend && npm run test -- src/components/panels/TaskPanel.test.tsx`
Expected: 新增 2 个用例里第一个 FAIL（`RecentLeading` 现在永远渲染图标，没有 `<img>`）；第二个可能已 PASS（巧合）。

- [ ] **Step 3: 改 `TaskPanel.tsx` —— `ImageThumb`**

`frontend/src/components/panels/TaskPanel.tsx`：

`RecentLeading` 整个函数替换为：
```tsx
function RecentLeading({ task }: { task: ExecutionTask }) {
  // DD9：image 任务且有缩略图 → 显示输出缩略图；否则降级状态图标。
  const thumb = task.output_thumbnails?.[0]
  if (task.task_type === 'image' && thumb) {
    return (
      <img
        src={thumb}
        alt={task.workflow_name || '图像任务输出'}
        loading="lazy"
        style={{
          width: 36,
          height: 36,
          objectFit: 'cover',
          borderRadius: 4,
          flexShrink: 0,
          background: 'var(--bg-accent)',
        }}
      />
    )
  }
  if (task.task_type === 'image') {
    // image 任务但还没拿到缩略图 URL（后端 outputs/ 落盘未就绪）→ ImageIcon。
    return <ImageIcon size={16} style={{ color: 'var(--info, #3b82f6)', flexShrink: 0 }} />
  }
  if (task.status === 'completed') {
    return <CheckCircle2 size={16} style={{ color: 'var(--accent-2, #22c55e)', flexShrink: 0 }} />
  }
  if (task.status === 'failed') {
    return <AlertCircle size={16} style={{ color: 'var(--accent, #ef4444)', flexShrink: 0 }} />
  }
  return <Loader2 size={16} style={{ color: 'var(--muted)', flexShrink: 0 }} />
}
```

删掉文件末尾的临时 `export const _ImageIcon = ImageIcon` 那行（`ImageIcon` 现在被 `RecentLeading` 真正用上了）。

- [ ] **Step 4: 跑测试确认通过 + tsc**

Run: `cd frontend && npm run test -- src/components/panels/TaskPanel.test.tsx && npx tsc -b --noEmit`
Expected: 全部用例 PASS，tsc 无报错（确认 `_ImageIcon` 已删、`ImageIcon` 不再 unused）。

- [ ] **Step 5: Commit**

```bash
cd /media/heygo/Program/projects-code/_playground/nous-center
git add frontend/src/components/panels/TaskPanel.tsx frontend/src/components/panels/TaskPanel.test.tsx
git commit -m "feat(frontend): image thumbnail history in recent-completed list

Completed image tasks with output_thumbnails render an <img>
thumbnail; image tasks without a thumbnail URL yet fall back to
ImageIcon, non-image tasks to the status icon. V1.5 Lane I, spec
6.5 DD9."
```

---

## Task 12: Dashboard GpuPanel 加 runner 标识（DD3 末项）

spec §6 / DD3 末项：Dashboard GpuPanel 加「Runner: A (image)」标签。`DashboardOverlay.tsx` 的 `GpuCard` 渲染每张 GPU；用 `useRunners()` 的数据，把「这张 GPU 属于哪个 runner」标出来。

> **判断**：spec 写「Runner: A (image)」，但 `RunnerInfo` 没有「runner 占用哪些 GPU index」的字段 —— runner ↔ GPU 的映射在后端 `hardware.yaml` 的 `groups[].gpus`（Lane A）。最干净的做法是 `RunnerInfo` 加一个 `gpus: number[]` 字段，`GpuCard` 按 `gpu.index ∈ runner.gpus` 匹配。本 Task 给 `RunnerInfo` 补 `gpus`，并在 `GpuCard` 渲染标签。已在 Self-Review 标注 `gpus` 字段是 Lane I 对 Lane G/H 端点契约的补充要求。

**Files:**
- Modify: `frontend/src/api/runners.ts`（`RunnerInfo` 加 `gpus: number[]`）
- Modify: `frontend/src/components/overlays/DashboardOverlay.tsx`（`GpuCard` 加 runner 标签）
- Modify: `frontend/src/api/runners.test.ts`（更新 mock 带 `gpus`）
- Test: `frontend/src/components/overlays/DashboardOverlay.test.tsx`（新建，仅覆盖 GpuCard runner 标签）

- [ ] **Step 1: 写失败测试 —— GpuCard 显示所属 runner 标签**

新建 `frontend/src/components/overlays/DashboardOverlay.test.tsx`：
```tsx
import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import DashboardOverlay from './DashboardOverlay'

// DashboardOverlay 拉一堆 api —— 全部 mock 成空/最小数据，只验 GpuCard runner 标签。
vi.mock('../../api/dashboard', () => ({ useDashboardSummary: () => ({ data: undefined }) }))
vi.mock('../../api/observability', () => ({ useRuntimeMetrics: () => ({ data: undefined, isLoading: false, error: null }) }))
vi.mock('../../api/vllm', () => ({
  useVLLMMetrics: () => ({ data: { instances: [] }, isLoading: false, error: null }),
  useUpdateLaunchParams: () => ({ mutate: vi.fn(), isPending: false }),
}))
vi.mock('../../api/engines', () => ({ useEngines: () => ({ data: [] }) }))
vi.mock('../../api/system', () => ({
  useSysGpus: () => ({
    data: {
      count: 1,
      gpus: [{
        index: 2, name: 'RTX Pro 6000', utilization_gpu: 40, utilization_memory: 30,
        temperature: 55, fan_speed: 30, power_draw_w: 200, power_limit_w: 600,
        memory_used_mb: 20000, memory_total_mb: 98000, memory_free_mb: 78000, processes: [],
      }],
    },
  }),
  useSysStats: () => ({ data: undefined }),
  useSysProcesses: () => ({ data: undefined }),
  useKillProcess: () => ({ mutate: vi.fn() }),
}))
vi.mock('../../api/runners', () => ({
  useRunners: () => ({
    data: [
      { id: 'runner-i', label: 'Runner-I', role: 'image', state: 'busy',
        current_task: null, queue: [], restart_attempt: null, load_error: null, gpus: [2] },
    ],
  }),
}))

describe('DashboardOverlay GpuCard — runner label (DD3)', () => {
  it('shows the owning runner label on the GPU card', () => {
    render(<DashboardOverlay />)
    // GPU index 2 属于 runner-i → 卡片上应有 "Runner-I (image)"
    expect(screen.getByText(/Runner-I \(image\)/)).toBeTruthy()
  })
})
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd frontend && npm run test -- src/components/overlays/DashboardOverlay.test.tsx`
Expected: FAIL —— `GpuCard` 还没渲染 runner 标签；且 `RunnerInfo` 无 `gpus` 字段（tsc 在 mock 里也会报，但 vitest 跑时是运行时，主要 FAIL 原因是找不到文字）。

- [ ] **Step 3: 给 `RunnerInfo` 加 `gpus` + 更新 `runners.test.ts` mock**

`frontend/src/api/runners.ts` 的 `RunnerInfo` 接口加字段（在 `load_error` 之后）：
```ts
  /** 该 runner 占用的 GPU index 列表（对齐 hardware.yaml groups[].gpus）。
   * Dashboard GpuPanel 用它给每张 GPU 标「属于哪个 runner」（spec §6 DD3）。 */
  gpus: number[]
```

`frontend/src/api/runners.test.ts` 里 `mockResolvedValue` 的 runner 对象加 `gpus: [2]`（与新接口对齐，否则 tsc 报缺字段）：
```ts
        restart_attempt: null,
        load_error: null,
        gpus: [2],
      },
```

- [ ] **Step 4: 改 `DashboardOverlay.tsx` —— `GpuCard` 加 runner 标签**

`frontend/src/components/overlays/DashboardOverlay.tsx`：

顶部 import 区加：
```tsx
import { useRunners } from '../../api/runners'
```

`CollapsibleSystem` 里拿 runners 数据 —— 在 `const { data: gpuData } = useSysGpus()` 旁加：
```tsx
  const { data: runners } = useRunners()
```

`GpuCard` 的调用处（`{(gpuData?.gpus ?? []).map((gpu) => (`）把 runner 传进去 —— 找到匹配该 GPU 的 runner：
```tsx
            {(gpuData?.gpus ?? []).map((gpu) => (
              <GpuCard
                key={gpu.index}
                gpu={gpu}
                runner={(runners ?? []).find((r) => r.gpus.includes(gpu.index)) ?? null}
                onKill={(pid, mem) => {
                  if (
                    window.confirm(
                      `Kill process PID ${pid}? This will free ~${(mem / 1024).toFixed(1)}G GPU memory.`,
                    )
                  ) {
                    killProcess.mutate(pid)
                  }
                }}
              />
            ))}
```

`GpuCard` 组件签名 + 渲染 —— 把：
```tsx
function GpuCard({
  gpu,
  onKill,
}: {
  gpu: SysGpuInfo
  onKill: (pid: number, mem: number) => void
}) {
```
改为：
```tsx
function GpuCard({
  gpu,
  runner,
  onKill,
}: {
  gpu: SysGpuInfo
  runner: import('../../api/runners').RunnerInfo | null
  onKill: (pid: number, mem: number) => void
}) {
```
然后在 `GpuCard` 的 `<div style={{ fontSize: 13, color: 'var(--text)', marginTop: 4 }}>{gpu.name}</div>` **之后**加 runner 标签：
```tsx
      {runner && (
        <div style={{ fontSize: 11, color: 'var(--info, #3b82f6)', marginTop: 2 }}>
          Runner: {runner.label} ({runner.role})
        </div>
      )}
```

- [ ] **Step 5: 跑测试确认通过 + tsc**

Run: `cd frontend && npm run test -- src/components/overlays/DashboardOverlay.test.tsx src/api/runners.test.ts && npx tsc -b --noEmit`
Expected: `DashboardOverlay.test.tsx` 1 个用例 + `runners.test.ts` 2 个用例 PASS，tsc 无报错。

- [ ] **Step 6: Commit**

```bash
cd /media/heygo/Program/projects-code/_playground/nous-center
git add frontend/src/api/runners.ts frontend/src/api/runners.test.ts frontend/src/components/overlays/DashboardOverlay.tsx frontend/src/components/overlays/DashboardOverlay.test.tsx
git commit -m "feat(frontend): Dashboard GpuCard shows owning runner label

RunnerInfo gains a gpus[] field (mirrors hardware.yaml groups[].gpus);
GpuCard matches gpu.index against it and renders 'Runner: X (role)'.
V1.5 Lane I, spec 6 DD3."
```

---

## Task 13: 浏览器通知权限流接入 + Lane I 整合验证 + preflight

spec §6.6 a11y：浏览器通知权限「首次询问」。挂一个时机触发 `requestPermission()`（不在页面加载时弹 —— 那很烦；在用户首次点 Run 时问，是合理时机）。然后跑 Lane I 全量验证 + CLAUDE.md 要求的 preflight。

**Files:**
- Modify: `frontend/src/components/layout/Topbar.tsx`（`handleRun` 里首次 Run 触发 `requestPermission`）
- Test: 复用 `Topbar.test.tsx`（追加 1 个用例）

- [ ] **Step 1: 写失败测试 —— 首次 Run 触发权限询问**

在 `frontend/src/components/layout/Topbar.test.tsx` 的 describe 里追加用例。先在文件顶部 mock 区加：
```tsx
const requestPermission = vi.fn().mockResolvedValue(undefined)
vi.mock('../../stores/notifications', () => ({
  useNotificationStore: () => ({ requestPermission }),
}))
```
追加用例：
```tsx
  it('first Run triggers the browser notification permission request', async () => {
    executeWorkflow.mockResolvedValue({ task_id: 'wf_x' })
    render(
      <MemoryRouter>
        <Topbar />
      </MemoryRouter>,
    )
    fireEvent.click(screen.getByText(/Run/))
    await waitFor(() => expect(requestPermission).toHaveBeenCalled())
  })
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd frontend && npm run test -- src/components/layout/Topbar.test.tsx`
Expected: 新用例 FAIL —— `handleRun` 还没调 `requestPermission`。

- [ ] **Step 3: 改 `Topbar.tsx` —— 首次 Run 问权限**

`frontend/src/components/layout/Topbar.tsx`：

import 加：
```tsx
import { useNotificationStore } from '../../stores/notifications'
```

组件体内加：
```tsx
  const requestNotifyPermission = useNotificationStore((s) => s.requestPermission)
```

`handleRun` 函数体最前面（`if (isRunning) return` 之后）加：
```tsx
    // spec §6.6：浏览器通知权限「首次询问」—— 在用户首次点 Run 时问
    // （比页面加载时弹更得体）。requestPermission 内部幂等：已问过就直接读快照。
    void requestNotifyPermission()
```
> 若 `useNotificationStore` 的 selector 写法（`(s) => s.requestPermission`）与 Step 1 的 mock（`useNotificationStore: () => ({ requestPermission })`，无视 selector 参数）不兼容 —— mock 返回的对象没走 selector。改 mock 为支持 selector：`useNotificationStore: (sel?: (s: { requestPermission: typeof requestPermission }) => unknown) => sel ? sel({ requestPermission }) : { requestPermission }`。两种写法择一，确保测试与实现的 selector 约定一致。

- [ ] **Step 4: 跑测试确认通过**

Run: `cd frontend && npm run test -- src/components/layout/Topbar.test.tsx`
Expected: Topbar 全部用例 PASS。

- [ ] **Step 5: Lane I 全量测试 green**

Run: `cd frontend && npm run test`
Expected: 全 PASS。新增测试文件：`runners.test.ts` / `tasks.test.ts` / `notifications.test.ts` / `useTaskCompletionNotifier.test.tsx` / `execution.test.ts` / `Topbar.test.tsx` / `IconRail.test.tsx` / `DashboardOverlay.test.tsx`；重写：`TaskPanel.test.tsx`。无 collection error、无 unhandled rejection。对照 Task 1 Step 1 基线 —— 用例数应净增（重写的 `TaskPanel.test.tsx` 用例数变化属预期）。

- [ ] **Step 6: preflight —— tsc + vite build（CLAUDE.md 要求）**

Run: `cd frontend && npm run build`
Expected: `tsc -b` 无类型错误 + `vite build` 成功产出 `dist/`。CLAUDE.md：生产 backend serve `frontend/dist/`，build 必须过。
> 若 `npm run build` 触发 `prebuild` 跑 `wasm:build`（`wasm-pack`）失败且本机无 `wasm-pack` —— Lane I 没碰 wasm，可单独验证类型与打包：`npx tsc -b --noEmit && npx vite build`。但 PR CI 跑的是完整 `npm run build`，本机 wasm 环境缺失只是本地 preflight 的局限，需在 PR 描述注明本地用 `tsc + vite build` 验证。

- [ ] **Step 7: lint 预检（CLAUDE.md 要求）**

Run: `cd frontend && npm run lint`
Expected: 无新增 eslint 错误（Lane I 新增/改动的文件无 unused import、无 `any` 滥用 —— `ExecutionTask.result` 既有 `any` 是历史遗留，不在本 Lane 引入）。

- [ ] **Step 8: Commit**

```bash
cd /media/heygo/Program/projects-code/_playground/nous-center
git add frontend/src/components/layout/Topbar.tsx frontend/src/components/layout/Topbar.test.tsx
git commit -m "feat(frontend): request notification permission on first Run

spec 6.6 — permission asked once, at the first Run click (more
graceful than prompting on page load). requestPermission is
idempotent. V1.5 Lane I, spec 6.6."
```

- [ ] **Step 9: 开 PR**

```bash
cd /media/heygo/Program/projects-code/_playground/nous-center
git push -u origin <lane-I-branch>
gh pr create --title "feat: V1.5 Lane I — TaskPanel Buildkite-style runner lanes + notifications" --body "$(cat <<'EOF'
## Summary
- TaskPanel 从「3-tab 扁平列表」重构为 Buildkite 风：顶部 per-runner 泳道区（hero）+ 下方「最近完成」列表（DD3）
- runner 泳道异常态内联：重启中脉冲 + 加载失败 + Retry 按钮（DD5）；排队位置可点击展开有序列表（DD8）；image 任务缩略图历史（DD9）
- 点 Run 反馈：toast「已入队」+ IconRail badge，面板不自动打开（DD4）；任务完成通知：toast + 浏览器 Notification，权限首次询问、拒绝降级 toast-only、仅失焦发系统通知（DD6）
- 响应式 `<768px` 全屏抽屉 + 泳道堆叠（DD7）；a11y：aria-expanded / 状态文字伴随色点 / 键盘可达动作 / aria-label（§6.6）
- Dashboard GpuCard 加「Runner: X (role)」标签

## 与 spec 的偏差（详见 plan 顶部）
- 后端无 `/api/v1/runners` 端点 —— Lane I 定义契约，`useRunners` 404 降级空泳道，由 Lane G/H 提供端点
- `RunnerInfo` 需 `gpus[]` 字段（GpuCard runner 标签用）—— 是 Lane I 对 Lane G/H 端点契约的补充
- `ExecutionTask` 扩 `gpu_group/runner_id/queue_position/output_thumbnails`（全 optional）—— 后端序列化由 Lane B/D 落地
- 排队列表「当前用户高亮」跳过 —— 单 admin infra 下无信息量

## Test plan
- [ ] `npm run test` 全 green（9 个新/重写测试文件）
- [ ] `npm run build`（tsc + vite build）过 —— 生产 backend serve dist/
- [ ] `npm run lint` 无新增错误
- [ ] 手测：点 Run → toast + badge、面板不自动开；打开面板看泳道 + 最近完成；窄屏全屏抽屉
EOF
)"
```
（分支名按项目惯例，如 `feat/v15-laneI-frontend-taskpanel`。）

---

## Self-Review

**Spec 覆盖检查：** Lane I 在 spec「实施分 Lane」表里的职责是「TaskPanel 重构为 Buildkite 风 runner 泳道（DD3-DD8）+ image 缩略图历史（DD9）+ 响应式 + a11y + Dashboard runner 标识。依赖：B, G, S」。逐项对照 spec §6：

- **DD3**（§6.1 Buildkite 风结构：顶部 per-runner 泳道 hero + 下方最近完成）→ Task 8（结构骨架重写）+ Task 12（Dashboard runner 标识）
- **DD4**（§6.3 点 Run 反馈：toast + IconRail badge，面板不自动开）→ Task 5（`taskIconBadge` store）+ Task 6（`Topbar.handleRun` 接 202 契约）+ Task 7（IconRail badge 改读 `taskIconBadge`）
- **DD5**（§6.2 runner 泳道异常态：重启中脉冲 / 加载失败 + Retry）→ Task 9
- **DD6**（§6.3 完成通知：toast + 浏览器 Notification，权限流）→ Task 3（`notifications` store）+ Task 4（`useTaskCompletionNotifier` diff 检测）+ Task 13（首次 Run 问权限）
- **DD7**（§6.5 响应式 `<768px` 全屏 + 泳道堆叠）→ Task 8（`useIsNarrow` + `width: 100vw`）
- **DD8**（§6.4 排队位置可展开有序列表）→ Task 10（`QueueExpand`）
- **DD9**（§6.5 image 缩略图历史）→ Task 2（`ExecutionTask.output_thumbnails`）+ Task 11（`ImageThumb` / `RecentLeading` 缩略图分支）
- **a11y 清单**（§6.6）→ 贯穿：aria-expanded（Task 10 `QueueExpand`）、状态文字伴随色点（Task 8/9 `RunnerStateText` + `RunnerStateDot`）、aria-live（见下「已知风险」—— ToastContainer 待补）、键盘可达（Task 7/9 真 button + aria-label）、通知权限流（Task 3/13）
- **设计系统复用**（§6 开头）→ 全程用既有 CSS vars（`--bg/--bg-accent/--border/--text/--muted/--accent/--accent-2/--info/--warn/--mono`）+ lucide 图标 + 既有 badge/进度条模式，无新依赖
- **依赖 B/G/S** → Lane B（`execution_tasks` 8 列）：Task 2 扩 `ExecutionTask` 接口对齐；Lane G（GroupScheduler + runner 状态）：Task 1 `useRunners` 假定 `/api/v1/runners` 契约；Lane S（`/run` 202 异步契约）：Task 6 `handleRun` 接 `{task_id}` 返回。三者都用「optional 字段 / 404 降级 / 兼容旧签名」让 Lane I 能独立 merge、CI 绿，不硬阻塞在 B/G/S 落地顺序上。

**与 spec 的偏差（已在 plan 顶部「注意」+ 此处显式标注）：**
1. **后端无 runner 状态端点** —— spec §6.1/§6.2 假设有 per-runner 数据源，但全后端无 `/api/v1/runners`。Lane I 定义 `RunnerInfo` 契约 + `useRunners` 404 降级。Lane G/H 落地端点时若形状不同，需对齐 Task 1 接口。
2. **`RunnerInfo.gpus[]` 是 Lane I 对端点契约的补充** —— spec 写 Dashboard「Runner: A (image)」标签，但没说 runner↔GPU 映射怎么传。Task 12 给 `RunnerInfo` 加 `gpus: number[]`（对齐 `hardware.yaml groups[].gpus`），这是 Lane I 给 Lane G/H 端点提的额外字段要求。
3. **`ExecutionTask` 缩略图 / V1.5 字段后端序列化不在 Lane I** —— Task 2 扩前端接口（全 optional），后端 `/tasks` 实际带出 `output_thumbnails` 等是 Lane B/D scope。`ImageThumb` 在字段缺失时降级 ImageIcon。
4. **`/ws/tasks` 事件 payload 不解析** —— spec §3.7 的 WSEvent 词表是 `/ws/workflow/{id}` 的；`/ws/tasks` 现有消费是「收消息就 invalidate」。Lane I 不改这个，完成通知靠 `useTasks` 数据的「非终态→终态」diff 触发（Task 4），不耦合 Lane G 的 WS 实现。
5. **排队列表「当前用户高亮」跳过** —— spec §6.4 写「刚提交的 / 当前用户的任务高亮」，但 CLAUDE.md 明确单 admin infra，「当前用户」永远同一人，高亮无信息量；「刚提交高亮」需 Lane S 回 task_id 后比对，价值低。Task 10 有意识裁掉。

**spec 模糊处的判断：**
- spec §6.2 异常态表「重启中 2/4」的 `2/4` 语义 = backoff「第几次 / 总次数」。判断：`RunnerInfo.restart_attempt: [number, number] | null`，`RunnerStateText` 渲染 `重启中 ${n}/${m}`。来源是 spec §4.2 `RESTART_BACKOFF = [5, 15, 60, 300]`（4 次）。
- spec §6.3「toast 带『查看』跳转」—— 当前 `useToastStore` 的 toast 是纯文本 + 点击 dismiss，没有「带 action button」的能力。判断：本 Lane **不**扩 toast 组件加 action button（那是 toast 系统的改造，超 Lane I scope）；toast 文案带 task_id，用户点 IconRail 任务图标进面板查看 —— 「查看」入口由 IconRail badge 承接。已在 Task 6 注释说明。这是有意识的范围控制，不是遗漏。
- spec §6.6「toast 用 aria-live」—— `ToastContainer.tsx` 现在没有 `aria-live`。判断：见「已知风险」—— 本 plan 未把 `ToastContainer` 加 `aria-live` 列为独立 Task，应在 Task 13 之前补一个 1-step 改动，或作为 follow-up。**这是 plan 的一个缺口，已在已知风险标注。**
- spec §6.1 泳道图里「排队 3 ▸」在泳道内 —— 判断：`QueueExpand` 渲染在 `RunnerLane` 内部、current_task 之下，与 spec ASCII 图一致。

**Placeholder 扫描：** 无 TBD / 「类似 Task N」。所有 `.tsx` / `.ts` 代码、测试代码、命令均完整给出。Task 8 的 `_ImageIcon` 临时 export 在 Task 11 Step 3 明确删除（不是遗留 placeholder，是跨 Task 的有序交接，已标注）。每个 Task 是「写失败测试 → 跑确认失败 → 最小实现 → 跑确认通过 → commit」闭环。

**类型一致性：**
- `RunnerInfo`（Task 1 定义，Task 9 加 `useRetryRunner` 不改接口，Task 12 加 `gpus: number[]`）—— `TaskPanel.tsx` / `DashboardOverlay.tsx` 消费处类型对齐；`runners.test.ts` 的 mock 在 Task 12 同步加 `gpus` 字段（否则 tsc 报缺字段）。
- `ExecutionTask` 扩的 4 个字段全 `optional`（`?:`）—— `tasks.test.ts` 的「legacy payload」用例验证旧形状仍可赋值。
- `useExecutionStore` 加的 `taskIconBadge: number` / `bumpTaskBadge` / `clearTaskBadge` —— `ExecutionState` 接口与 `create<ExecutionState>` 实现同步改（Task 5），`Topbar` / `IconRail` 消费处对齐。
- `succeed(null)`（Task 6）—— `succeed` 签名 `(result: ExecutionState['result']) => void`，`ExecutionState['result']` 含 `| null`，传 `null` 类型合法。
- `notifyOnce(taskId, message, type, toast)` 的 `toast` 参数类型 `ToastFn` 与 `useToastStore` 的 `add` 签名 `(message, type?) => void` 兼容（`notifyOnce` 调用时显式传 `'success'`/`'error'`）。

**已知风险：**
- **`/api/v1/runners` 端点未落地** —— 最大风险。`useRunners` 404 降级让 Lane I 能 merge + CI 绿，但「泳道区」在 Lane G/H 端点上线前会一直显示「暂无 runner 数据」。这是有意的解耦，但意味着 Lane I merge 后到 Lane G/H merge 前，泳道是空的 —— PR 描述需注明这是预期中间态。
- **`ToastContainer` 缺 `aria-live`** —— spec §6.6 a11y 清单明确要求 toast 用 `aria-live`，但本 plan 没有独立 Task 覆盖 `ToastContainer.tsx` 的改动。**缓解**：实施时应在 Task 13 之前插一个 1-step 微改 —— 给 `ToastContainer` 的外层 `<div>` 加 `role="status" aria-live="polite"`，并加一个断言用例。这是 plan 的已知缺口，列在此处而非藏着。
- **`npm run build` 的 `prebuild` 跑 `wasm:build`** —— `package.json` 的 `prebuild` 钩子跑 `wasm-pack`。本机若无 `wasm-pack`，`npm run build` 会在 wasm 阶段失败。Lane I 没碰 wasm，preflight 可用 `npx tsc -b --noEmit && npx vite build` 验证类型 + 打包；但 CI 跑完整 `npm run build`，需确保 CI 环境有 `wasm-pack`（这是既有 CI 配置，非 Lane I 引入）。已在 Task 13 Step 6 标注。
- **`TaskPanel.test.tsx` 跨 describe 共享 `let runnersData`** —— Task 8/9/10/11 的测试都在同一文件、共享 `runnersData` 这个 `let` 变量。每个 describe 的 `beforeEach` 必须重置它（Task 8 重置为 `mockRunners`，Task 11 重置为 `[]`，Task 9/10 各 `it` 自设）。实施时若用例顺序导致串台，检查 `beforeEach` 重置是否覆盖。已在 Task 9 Step 1 注释说明。
- **`@testing-library/user-event` 可能未安装** —— Task 10 的测试优先用 `user-event`，但 `package.json` 的 `devDependencies` 没显式列它（它是 `@testing-library/react` 的 peer，可能在也可能不在 `node_modules`）。Task 10 Step 2 已给降级路径（改用 `fireEvent.click`）。
- **Lane S 未 merge 时 `executeWorkflow` 仍是旧同步签名** —— Task 6 的 `handleRun` 用 `(result as { task_id?: string })?.task_id` 软取值：Lane S 已 merge 时拿到 `task_id`，未 merge 时 `result` 是旧的 `ExecutionResult`（无 `task_id`），`taskId` 为 `undefined`，toast 文案降级为「任务已入队」（无 id）。功能不崩，只是 id 缺失。Lane I 与 Lane S 的 merge 顺序无强约束。
