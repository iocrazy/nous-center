# 全局任务管理系统设计

## 概述

为 nous-center 添加全局任务管理，记录所有 workflow 执行历史（前端 Run + API 调用），支持取消、重试、结果查看。类似 ComfyUI 的任务队列面板。

---

## 数据模型

```sql
-- execution_tasks 表
id            BIGINT PRIMARY KEY (snowflake)
workflow_id   BIGINT NULL       -- 关联的 workflow（API 直接调用可能无）
workflow_name VARCHAR(100)
status        VARCHAR(20)       -- queued / running / completed / failed / cancelled
nodes_total   INT DEFAULT 0
nodes_done    INT DEFAULT 0
current_node  VARCHAR(100) NULL
result        JSON NULL          -- 执行结果（outputs）
error         TEXT NULL
duration_ms   INT NULL
output_dir    VARCHAR(500) NULL  -- outputs/{task_id}/ 结果文件目录
created_at    TIMESTAMP
updated_at    TIMESTAMP
```

结果文件存储在 `outputs/{task_id}/` 目录下（音频文件等）。

---

## API 端点

```
GET    /api/v1/tasks              — 任务列表（分页，?limit=20&offset=0&status=running）
GET    /api/v1/tasks/{id}         — 任务详情 + 结果
POST   /api/v1/tasks/{id}/cancel  — 取消正在执行的任务
POST   /api/v1/tasks/{id}/retry   — 重试失败的任务
DELETE /api/v1/tasks/{id}         — 删除任务记录 + 结果文件
```

---

## 执行流程

### 前端 Run

```
用户点 Run
  → 创建 task 记录 (status=queued)
  → 开始执行 (status=running)
  → 逐节点执行，更新 nodes_done / current_node
  → 完成：status=completed, 保存 result, 计算 duration_ms
  → 失败：status=failed, 保存 error
```

### API /run 调用

```
POST /v1/instances/{id}/run
  → 创建 task 记录 (status=queued)
  → 执行 workflow
  → 完成/失败更新 task
```

### 取消

```
POST /api/v1/tasks/{id}/cancel
  → 设置取消标志
  → 执行器在每个节点执行前检查标志
  → 如已取消，停止执行，status=cancelled
```

### 重试

```
POST /api/v1/tasks/{id}/retry
  → 读取原任务的 workflow 定义
  → 创建新 task 记录
  → 重新执行
```

---

## 前端

### 任务面板

- **入口**：IconRail 底部新增任务图标（ListTodo）
- **面板**：右侧可折叠面板，300px 宽
- **面板类型**：OverlayId 不适合（overlay 是全屏的），用独立的 `taskPanelOpen` 状态

### 任务列表项

```
┌─────────────────────────────────────┐
│ ● Qwen3语音测试          12s  ✓    │
│   completed · 3/3 nodes · 刚刚     │
├─────────────────────────────────────┤
│ ● 基础合成               running   │
│   ████████░░ 66% · tts_engine      │
├─────────────────────────────────────┤
│ ● API调用-发布测试2       failed   │
│   TTS 节点缺少文本输入 · 5分钟前   │
└─────────────────────────────────────┘
```

### 任务详情（点击展开）

- 每个节点的执行状态（pending/running/completed/error）
- 结果预览（音频播放器、文本显示）
- 执行耗时分解（每个节点用时）

### 右键菜单

- 取消（running 状态）
- 重试（failed 状态）
- 删除（任何状态）

### 实时更新

- 运行中的任务通过 WebSocket 实时更新进度
- 复用现有 `/ws/workflow/{instance_id}` WebSocket
- 前端执行时直接更新 store

---

## 和现有系统的关系

| 现有模块 | 变更 |
|---------|------|
| `workflow_executor.py` | 执行时创建/更新 task 记录 |
| `workflowExecutor.ts` | 前端执行时创建/更新 task 记录 |
| `instance_service.py` /run | API 执行时创建 task 记录 |
| `execution.ts` store | 新增 taskPanelOpen 状态 |
| WebSocket 进度 | 复用，task_id 关联 |
| 节点高亮 | 保持不变 |
| Topbar 进度条 | 保持不变 |

---

## 文件结构

### 后端新建

| 文件 | 职责 |
|------|------|
| `backend/src/models/execution_task.py` | ExecutionTask ORM 模型 |
| `backend/src/api/routes/tasks_v2.py` | 任务管理 API（避免和现有 tasks.py 冲突） |

### 后端修改

| 文件 | 变更 |
|------|------|
| `backend/src/api/main.py` | 注册模型 + 路由 |
| `backend/src/services/workflow_executor.py` | 执行时写 task 记录 |
| `backend/src/api/routes/instance_service.py` | /run 时写 task 记录 |

### 前端新建

| 文件 | 职责 |
|------|------|
| `frontend/src/api/tasks.ts` | Task API hooks |
| `frontend/src/components/panels/TaskPanel.tsx` | 右侧任务面板 |

### 前端修改

| 文件 | 变更 |
|------|------|
| `frontend/src/stores/execution.ts` | taskPanelOpen 状态 |
| `frontend/src/components/layout/IconRail.tsx` | 任务图标 |
| `frontend/src/components/nodes/NodeEditor.tsx` | 渲染 TaskPanel |
| `frontend/src/utils/workflowExecutor.ts` | 创建/更新 task |

---

## 非目标

- 任务优先级调度
- 并行任务执行（单用户，串行即可）
- 任务依赖链
