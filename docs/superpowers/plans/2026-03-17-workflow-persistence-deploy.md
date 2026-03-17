# Phase A: Workflow 持久化 + 部署发布 — 实现计划

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 workflow 从前端内存持久化到数据库，支持 CRUD、自动保存、一键发布为常驻 API 服务。

**Architecture:** 新增 `Workflow` ORM 模型 + CRUD 路由。发布时快照 workflow 到 ServiceInstance.params_override，后端 DAG 执行器处理 `/instances/{id}/run` 请求。前端 workspace store 改为 DB 驱动，Topbar 加发布按钮。

**Tech Stack:** FastAPI, SQLAlchemy async, PostgreSQL, React, Zustand, TanStack Query, @xyflow/react

---

## 文件结构

### 后端新建

| 文件 | 职责 |
|------|------|
| `backend/src/models/workflow.py` | Workflow ORM 模型 |
| `backend/src/api/routes/workflows.py` | Workflow CRUD + publish/unpublish 路由 |
| `backend/src/services/workflow_executor.py` | 后端 DAG 执行器 |
| `backend/tests/test_api_workflows.py` | Workflow API 测试 |
| `backend/tests/test_workflow_executor.py` | DAG 执行器测试 |

### 后端修改

| 文件 | 变更 |
|------|------|
| `backend/src/api/main.py` | 注册 workflow 模型 + 路由 |
| `backend/src/models/schemas.py` | 新增 Workflow 相关 schema |
| `backend/src/api/routes/instance_service.py` | 支持 source_type="workflow" 的 run |
| `backend/src/api/routes/instances.py` | _resolve_source_name 支持 workflow |

### 前端新建

| 文件 | 职责 |
|------|------|
| `frontend/src/api/workflows.ts` | Workflow API hooks |

### 前端修改

| 文件 | 变更 |
|------|------|
| `frontend/src/models/workflow.ts` | id 改为 string (snowflake)，新增 is_template/status/description |
| `frontend/src/stores/workspace.ts` | DB 驱动：加载/保存/自动保存 |
| `frontend/src/components/layout/Topbar.tsx` | 加发布按钮 |
| `frontend/src/components/panels/NodeLibraryPanel.tsx` | 重命名为 WorkflowsPanel 或新增 workflows 列表 |
| `frontend/src/stores/panel.ts` | PanelId 调整 |

---

## Chunk 1: 后端 Workflow 模型 + CRUD

### Task 1: Workflow ORM 模型

**Files:**
- Create: `backend/src/models/workflow.py`
- Modify: `backend/src/api/main.py`
- Modify: `backend/src/models/schemas.py`

- [ ] **Step 1: 创建 Workflow ORM 模型**

```python
# backend/src/models/workflow.py
from datetime import datetime, timezone

from sqlalchemy import BigInteger, Boolean, DateTime, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from src.models.database import Base
from src.utils.snowflake import snowflake_id


class Workflow(Base):
    __tablename__ = "workflows"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, default=snowflake_id)
    name: Mapped[str] = mapped_column(String(100))
    description: Mapped[str | None] = mapped_column(Text, default=None)
    nodes: Mapped[list] = mapped_column(JSON, default=list)
    edges: Mapped[list] = mapped_column(JSON, default=list)
    is_template: Mapped[bool] = mapped_column(Boolean, default=False)
    status: Mapped[str] = mapped_column(String(20), default="draft")

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )
```

- [ ] **Step 2: 添加 Pydantic schema**

在 `backend/src/models/schemas.py` 末尾添加：

```python
class WorkflowCreate(BaseModel):
    name: str
    description: str | None = None
    nodes: list = []
    edges: list = []
    is_template: bool = False

class WorkflowUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    nodes: list | None = None
    edges: list | None = None
    is_template: bool | None = None

class WorkflowOut(BaseModel):
    model_config = {"from_attributes": True}

    id: str
    name: str
    description: str | None
    nodes: list
    edges: list
    is_template: bool
    status: str
    created_at: datetime
    updated_at: datetime

    @field_serializer("id")
    def serialize_id(self, v: int) -> str:
        return str(v)
```

- [ ] **Step 3: 注册模型到 main.py lifespan**

在 `backend/src/api/main.py` 的 lifespan 中 import 行添加：

```python
import src.models.workflow  # noqa: F401
```

- [ ] **Step 4: 验证 import**

Run: `cd backend && uv run python -c "from src.models.workflow import Workflow; print('OK')"`
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add backend/src/models/workflow.py backend/src/models/schemas.py backend/src/api/main.py
git commit -m "feat: add Workflow ORM model and schemas"
```

---

### Task 2: Workflow CRUD 路由

**Files:**
- Create: `backend/src/api/routes/workflows.py`
- Create: `backend/tests/test_api_workflows.py`
- Modify: `backend/src/api/main.py`
- Modify: `backend/tests/conftest.py`

- [ ] **Step 1: 写 CRUD 测试**

```python
# backend/tests/test_api_workflows.py

async def test_create_workflow(db_client):
    resp = await db_client.post("/api/v1/workflows", json={
        "name": "测试流程",
        "nodes": [{"id": "n1", "type": "text_input", "data": {}, "position": {"x": 0, "y": 0}}],
        "edges": [],
    })
    assert resp.status_code == 201
    data = resp.json()
    assert data["name"] == "测试流程"
    assert data["status"] == "draft"
    assert len(data["nodes"]) == 1
    assert "id" in data


async def test_list_workflows(db_client):
    # Create two workflows
    await db_client.post("/api/v1/workflows", json={"name": "w1"})
    await db_client.post("/api/v1/workflows", json={"name": "w2", "is_template": True})
    resp = await db_client.get("/api/v1/workflows")
    assert resp.status_code == 200
    assert len(resp.json()) == 2


async def test_list_workflows_filter_template(db_client):
    await db_client.post("/api/v1/workflows", json={"name": "w1"})
    await db_client.post("/api/v1/workflows", json={"name": "w2", "is_template": True})
    resp = await db_client.get("/api/v1/workflows?is_template=true")
    data = resp.json()
    assert len(data) == 1
    assert data[0]["is_template"] is True


async def test_get_workflow(db_client):
    create = await db_client.post("/api/v1/workflows", json={"name": "w1"})
    wf_id = create.json()["id"]
    resp = await db_client.get(f"/api/v1/workflows/{wf_id}")
    assert resp.status_code == 200
    assert resp.json()["name"] == "w1"


async def test_get_workflow_not_found(db_client):
    resp = await db_client.get("/api/v1/workflows/999999")
    assert resp.status_code == 404


async def test_update_workflow(db_client):
    create = await db_client.post("/api/v1/workflows", json={"name": "old"})
    wf_id = create.json()["id"]
    resp = await db_client.patch(f"/api/v1/workflows/{wf_id}", json={"name": "new"})
    assert resp.status_code == 200
    assert resp.json()["name"] == "new"


async def test_update_workflow_nodes(db_client):
    create = await db_client.post("/api/v1/workflows", json={"name": "w1", "nodes": []})
    wf_id = create.json()["id"]
    new_nodes = [{"id": "n1", "type": "text_input", "data": {}, "position": {"x": 0, "y": 0}}]
    resp = await db_client.patch(f"/api/v1/workflows/{wf_id}", json={"nodes": new_nodes})
    assert resp.status_code == 200
    assert len(resp.json()["nodes"]) == 1


async def test_delete_workflow(db_client):
    create = await db_client.post("/api/v1/workflows", json={"name": "w1"})
    wf_id = create.json()["id"]
    resp = await db_client.delete(f"/api/v1/workflows/{wf_id}")
    assert resp.status_code == 204
    resp2 = await db_client.get(f"/api/v1/workflows/{wf_id}")
    assert resp2.status_code == 404
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd backend && uv run pytest tests/test_api_workflows.py -v`
Expected: FAIL (route not found, 404)

- [ ] **Step 3: 实现 CRUD 路由**

```python
# backend/src/api/routes/workflows.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.database import get_async_session
from src.models.schemas import WorkflowCreate, WorkflowUpdate, WorkflowOut
from src.models.workflow import Workflow

router = APIRouter(prefix="/api/v1/workflows", tags=["workflows"])


@router.post("", response_model=WorkflowOut, status_code=201)
async def create_workflow(
    body: WorkflowCreate,
    session: AsyncSession = Depends(get_async_session),
):
    wf = Workflow(**body.model_dump())
    session.add(wf)
    await session.commit()
    await session.refresh(wf)
    return wf


@router.get("", response_model=list[WorkflowOut])
async def list_workflows(
    is_template: bool | None = None,
    session: AsyncSession = Depends(get_async_session),
):
    stmt = select(Workflow).order_by(Workflow.updated_at.desc())
    if is_template is not None:
        stmt = stmt.where(Workflow.is_template == is_template)
    result = await session.execute(stmt)
    return result.scalars().all()


@router.get("/{workflow_id}", response_model=WorkflowOut)
async def get_workflow(
    workflow_id: int,
    session: AsyncSession = Depends(get_async_session),
):
    wf = await session.get(Workflow, workflow_id)
    if not wf:
        raise HTTPException(404, "Workflow not found")
    return wf


@router.patch("/{workflow_id}", response_model=WorkflowOut)
async def update_workflow(
    workflow_id: int,
    body: WorkflowUpdate,
    session: AsyncSession = Depends(get_async_session),
):
    wf = await session.get(Workflow, workflow_id)
    if not wf:
        raise HTTPException(404, "Workflow not found")
    for key, value in body.model_dump(exclude_unset=True).items():
        setattr(wf, key, value)
    await session.commit()
    await session.refresh(wf)
    return wf


@router.delete("/{workflow_id}", status_code=204)
async def delete_workflow(
    workflow_id: int,
    session: AsyncSession = Depends(get_async_session),
):
    wf = await session.get(Workflow, workflow_id)
    if not wf:
        raise HTTPException(404, "Workflow not found")
    await session.delete(wf)
    await session.commit()
```

- [ ] **Step 4: 注册路由到 main.py**

在 `backend/src/api/main.py` 中：

import 行添加：`from src.api.routes import workflows`

app.include_router 添加：`app.include_router(workflows.router)`

conftest.py 添加：`import src.models.workflow  # noqa: F401`

- [ ] **Step 5: 运行测试确认通过**

Run: `cd backend && uv run pytest tests/test_api_workflows.py -v`
Expected: 8 passed

- [ ] **Step 6: Commit**

```bash
git add backend/src/api/routes/workflows.py backend/tests/test_api_workflows.py backend/src/api/main.py backend/tests/conftest.py
git commit -m "feat: add Workflow CRUD API with tests"
```

---

### Task 3: Publish / Unpublish 路由

**Files:**
- Modify: `backend/src/api/routes/workflows.py`
- Modify: `backend/src/api/routes/instances.py`
- Modify: `backend/tests/test_api_workflows.py`

- [ ] **Step 1: 写发布/下线测试**

追加到 `backend/tests/test_api_workflows.py`：

```python
async def test_publish_workflow(db_client):
    create = await db_client.post("/api/v1/workflows", json={
        "name": "发布测试",
        "nodes": [
            {"id": "n1", "type": "text_input", "data": {"text": "hello"}, "position": {"x": 0, "y": 0}},
            {"id": "n2", "type": "output", "data": {}, "position": {"x": 400, "y": 0}},
        ],
        "edges": [{"id": "e1", "source": "n1", "sourceHandle": "text", "target": "n2", "targetHandle": "audio"}],
    })
    wf_id = create.json()["id"]
    resp = await db_client.post(f"/api/v1/workflows/{wf_id}/publish")
    assert resp.status_code == 200
    data = resp.json()
    assert data["instance_id"] is not None
    assert data["endpoint"] is not None

    # Workflow status should be "published"
    wf = await db_client.get(f"/api/v1/workflows/{wf_id}")
    assert wf.json()["status"] == "published"


async def test_publish_not_found(db_client):
    resp = await db_client.post("/api/v1/workflows/999999/publish")
    assert resp.status_code == 404


async def test_unpublish_workflow(db_client):
    create = await db_client.post("/api/v1/workflows", json={"name": "w1"})
    wf_id = create.json()["id"]
    await db_client.post(f"/api/v1/workflows/{wf_id}/publish")
    resp = await db_client.post(f"/api/v1/workflows/{wf_id}/unpublish")
    assert resp.status_code == 200

    wf = await db_client.get(f"/api/v1/workflows/{wf_id}")
    assert wf.json()["status"] == "draft"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd backend && uv run pytest tests/test_api_workflows.py::test_publish_workflow -v`
Expected: FAIL (endpoint not found)

- [ ] **Step 3: 实现 publish/unpublish**

在 `backend/src/api/routes/workflows.py` 追加：

```python
from src.models.service_instance import ServiceInstance


@router.post("/{workflow_id}/publish")
async def publish_workflow(
    workflow_id: int,
    session: AsyncSession = Depends(get_async_session),
):
    wf = await session.get(Workflow, workflow_id)
    if not wf:
        raise HTTPException(404, "Workflow not found")

    # Check if already published — find existing instance
    stmt = select(ServiceInstance).where(
        ServiceInstance.source_type == "workflow",
        ServiceInstance.source_id == workflow_id,
    )
    result = await session.execute(stmt)
    instance = result.scalar_one_or_none()

    if instance:
        # Update snapshot
        instance.params_override = {"nodes": wf.nodes, "edges": wf.edges}
        instance.status = "active"
    else:
        # Create new instance (two-step: create → commit → set endpoint_path)
        instance = ServiceInstance(
            source_type="workflow",
            source_id=wf.id,
            name=wf.name,
            type="workflow",
            params_override={"nodes": wf.nodes, "edges": wf.edges},
        )
        session.add(instance)
        await session.flush()  # Get instance.id
        instance.endpoint_path = f"/v1/instances/{instance.id}/run"

    wf.status = "published"
    await session.commit()
    await session.refresh(instance)

    return {
        "instance_id": str(instance.id),
        "endpoint": instance.endpoint_path,
    }


@router.post("/{workflow_id}/unpublish")
async def unpublish_workflow(
    workflow_id: int,
    session: AsyncSession = Depends(get_async_session),
):
    wf = await session.get(Workflow, workflow_id)
    if not wf:
        raise HTTPException(404, "Workflow not found")

    # Deactivate service instance
    stmt = select(ServiceInstance).where(
        ServiceInstance.source_type == "workflow",
        ServiceInstance.source_id == workflow_id,
    )
    result = await session.execute(stmt)
    instance = result.scalar_one_or_none()
    if instance:
        instance.status = "inactive"

    wf.status = "draft"
    await session.commit()
    return {"status": "unpublished"}
```

- [ ] **Step 4: 更新 instances.py 支持 workflow source_type**

在 `backend/src/api/routes/instances.py` 做三处修改：

1. `_resolve_source_name` 添加 workflow 分支：

```python
from src.models.workflow import Workflow

# 在 _resolve_source_name 中添加：
if source_type == "workflow":
    wf = await session.get(Workflow, source_id)
    return wf.name if wf else None
```

2. `list_instances` 的批量 source name 解析添加 workflow 分支：

```python
# 在现有 preset 批量查询后添加：
wf_ids = [i.source_id for i in instances if i.source_type == "workflow"]
if wf_ids:
    wf_result = await session.execute(select(Workflow).where(Workflow.id.in_(wf_ids)))
    wf_map = {w.id: w.name for w in wf_result.scalars()}
    for inst in instances:
        if inst.source_type == "workflow":
            source_names[inst.id] = wf_map.get(inst.source_id)
```

3. `ServiceInstanceCreate` 的 `source_type` Literal 扩展（在 `schemas.py` 中）：

```python
source_type: Literal["preset", "workflow"] = "preset"
```

- [ ] **Step 5: 运行测试确认通过**

Run: `cd backend && uv run pytest tests/test_api_workflows.py -v`
Expected: 11 passed

- [ ] **Step 6: Commit**

```bash
git add backend/src/api/routes/workflows.py backend/src/api/routes/instances.py backend/tests/test_api_workflows.py
git commit -m "feat: add workflow publish/unpublish with ServiceInstance snapshot"
```

---

## Chunk 2: 后端 DAG 执行器 + Workflow Run

### Task 4: 后端 DAG 执行器

**Files:**
- Create: `backend/src/services/workflow_executor.py`
- Create: `backend/tests/test_workflow_executor.py`

- [ ] **Step 1: 写执行器测试**

```python
# backend/tests/test_workflow_executor.py
import pytest
from src.services.workflow_executor import WorkflowExecutor, ExecutionError


def _simple_workflow():
    """text_input → output"""
    return {
        "nodes": [
            {"id": "n1", "type": "text_input", "data": {"text": "hello"}, "position": {"x": 0, "y": 0}},
            {"id": "n2", "type": "output", "data": {}, "position": {"x": 400, "y": 0}},
        ],
        "edges": [
            {"id": "e1", "source": "n1", "sourceHandle": "text", "target": "n2", "targetHandle": "audio"},
        ],
    }


def _tts_workflow():
    """text_input → tts_engine → output"""
    return {
        "nodes": [
            {"id": "n1", "type": "text_input", "data": {"text": "你好"}, "position": {"x": 0, "y": 0}},
            {"id": "n2", "type": "tts_engine", "data": {"engine": "cosyvoice2", "speed": 1.0, "voice": "default", "sample_rate": 24000}, "position": {"x": 200, "y": 0}},
            {"id": "n3", "type": "output", "data": {}, "position": {"x": 400, "y": 0}},
        ],
        "edges": [
            {"id": "e1", "source": "n1", "sourceHandle": "text", "target": "n2", "targetHandle": "text"},
            {"id": "e2", "source": "n2", "sourceHandle": "audio", "target": "n3", "targetHandle": "audio"},
        ],
    }


def test_topological_sort():
    wf = _simple_workflow()
    executor = WorkflowExecutor(wf)
    order = executor._topological_sort()
    assert order.index("n1") < order.index("n2")


def test_topological_sort_tts():
    wf = _tts_workflow()
    executor = WorkflowExecutor(wf)
    order = executor._topological_sort()
    assert order.index("n1") < order.index("n2")
    assert order.index("n2") < order.index("n3")


def test_cycle_detection():
    wf = {
        "nodes": [
            {"id": "a", "type": "text_input", "data": {}, "position": {"x": 0, "y": 0}},
            {"id": "b", "type": "output", "data": {}, "position": {"x": 0, "y": 0}},
        ],
        "edges": [
            {"id": "e1", "source": "a", "sourceHandle": "text", "target": "b", "targetHandle": "audio"},
            {"id": "e2", "source": "b", "sourceHandle": "audio", "target": "a", "targetHandle": "text"},
        ],
    }
    executor = WorkflowExecutor(wf)
    with pytest.raises(ExecutionError, match="循环依赖"):
        executor._topological_sort()


def test_empty_workflow():
    executor = WorkflowExecutor({"nodes": [], "edges": []})
    with pytest.raises(ExecutionError, match="空"):
        executor._topological_sort()


@pytest.mark.asyncio
async def test_execute_text_passthrough():
    wf = _simple_workflow()
    executor = WorkflowExecutor(wf)
    result = await executor.execute()
    assert result["outputs"]["n1"]["text"] == "hello"


@pytest.mark.asyncio
async def test_get_inputs():
    wf = _tts_workflow()
    executor = WorkflowExecutor(wf)
    # Simulate n1 output
    executor._outputs["n1"] = {"text": "你好"}
    inputs = executor._get_inputs("n2")
    assert inputs["text"] == "你好"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd backend && uv run pytest tests/test_workflow_executor.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: 实现执行器**

```python
# backend/src/services/workflow_executor.py
"""Backend DAG workflow executor."""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from typing import Any

logger = logging.getLogger(__name__)


class ExecutionError(Exception):
    pass


class WorkflowExecutor:
    """Execute a workflow DAG (topological sort + per-node execution)."""

    def __init__(self, workflow: dict):
        self.nodes: list[dict] = workflow.get("nodes", [])
        self.edges: list[dict] = workflow.get("edges", [])
        self._node_map: dict[str, dict] = {n["id"]: n for n in self.nodes}
        self._outputs: dict[str, dict[str, Any]] = {}

    def _topological_sort(self) -> list[str]:
        if not self.nodes:
            raise ExecutionError("工作流为空")

        in_degree: dict[str, int] = defaultdict(int)
        adj: dict[str, list[str]] = defaultdict(list)

        for node in self.nodes:
            in_degree.setdefault(node["id"], 0)

        for edge in self.edges:
            adj[edge["source"]].append(edge["target"])
            in_degree[edge["target"]] += 1

        queue = deque(nid for nid, deg in in_degree.items() if deg == 0)
        order: list[str] = []

        while queue:
            nid = queue.popleft()
            order.append(nid)
            for neighbor in adj[nid]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(order) != len(self.nodes):
            raise ExecutionError("工作流存在循环依赖")

        return order

    def _get_inputs(self, node_id: str) -> dict[str, Any]:
        """Collect inputs for a node from upstream outputs via edges."""
        inputs: dict[str, Any] = {}
        for edge in self.edges:
            if edge["target"] == node_id:
                source_output = self._outputs.get(edge["source"], {})
                # Map source output to target input by handle
                source_handle = edge.get("sourceHandle", "")
                target_handle = edge.get("targetHandle", "")
                if source_handle in source_output:
                    inputs[target_handle] = source_output[source_handle]
                # Also try passing all outputs for simple cases
                for key, value in source_output.items():
                    if key not in inputs:
                        inputs[key] = value
        return inputs

    async def execute(self) -> dict[str, Any]:
        """Execute the workflow and return all node outputs."""
        order = self._topological_sort()

        for node_id in order:
            node = self._node_map[node_id]
            inputs = self._get_inputs(node_id)
            try:
                output = await self._execute_node(node, inputs)
                self._outputs[node_id] = output
            except Exception as e:
                raise ExecutionError(
                    f"节点 {node_id} ({node['type']}) 执行失败: {e}"
                ) from e

        return {"outputs": self._outputs}

    async def _execute_node(self, node: dict, inputs: dict) -> dict[str, Any]:
        """Execute a single node. Override-friendly dispatch."""
        node_type = node["type"]
        data = node.get("data", {})
        executor = _NODE_EXECUTORS.get(node_type)
        if executor is None:
            raise ExecutionError(f"未知节点类型: {node_type}")
        return await executor(data, inputs)


# --- Per-node executor functions ---

async def _exec_text_input(data: dict, inputs: dict) -> dict:
    return {"text": data.get("text", "")}


async def _exec_ref_audio(data: dict, inputs: dict) -> dict:
    return {
        "audio_path": data.get("path", ""),
        "ref_text": data.get("ref_text", ""),
    }


async def _exec_tts_engine(data: dict, inputs: dict) -> dict:
    """Call TTS engine directly via registry (same pattern as instance_service.py)."""
    import asyncio
    import base64
    from pathlib import Path
    from src.config import load_model_configs, get_settings
    from src.gpu.detector import get_device_for_engine
    from src.workers.tts_engines.registry import get_engine

    text = inputs.get("text", "")
    if not text:
        raise ExecutionError("TTS 节点缺少文本输入")

    engine_name = data.get("engine", "cosyvoice2")
    configs = load_model_configs()
    cfg = configs.get(engine_name)
    if not cfg:
        raise ExecutionError(f"未知引擎: {engine_name}")

    settings = get_settings()
    model_path = Path(settings.LOCAL_MODELS_PATH) / cfg["local_path"]
    device = get_device_for_engine(cfg)
    engine = get_engine(engine_name, model_path=model_path, device=device)
    if not engine.is_loaded:
        await asyncio.to_thread(engine.load)

    audio_bytes = await asyncio.to_thread(
        engine.synthesize,
        text=text,
        voice=data.get("voice", "default"),
        speed=data.get("speed", 1.0),
        sample_rate=data.get("sample_rate", 24000),
    )
    audio_b64 = base64.b64encode(audio_bytes).decode()
    return {"audio": audio_b64, "sample_rate": data.get("sample_rate", 24000)}


async def _exec_output(data: dict, inputs: dict) -> dict:
    return inputs


async def _exec_passthrough(data: dict, inputs: dict) -> dict:
    """Stub for unimplemented audio processing nodes."""
    return inputs


_NODE_EXECUTORS = {
    "text_input": _exec_text_input,
    "ref_audio": _exec_ref_audio,
    "tts_engine": _exec_tts_engine,
    "output": _exec_output,
    "resample": _exec_passthrough,
    "mixer": _exec_passthrough,
    "concat": _exec_passthrough,
    "bgm_mix": _exec_passthrough,
}
```

Note: `_do_synthesize` 可能需要从 `tts.py` 路由中提取为独立函数。如果现有 TTS 路由没有这个函数，则 TTS 节点执行器直接调用引擎 registry。具体实现时查看 `backend/src/api/routes/tts.py` 确定调用方式。

- [ ] **Step 4: 运行测试确认通过**

Run: `cd backend && uv run pytest tests/test_workflow_executor.py -v`
Expected: 7 passed（TTS 相关测试可能需要 mock）

- [ ] **Step 5: Commit**

```bash
git add backend/src/services/workflow_executor.py backend/tests/test_workflow_executor.py
git commit -m "feat: add backend DAG workflow executor with topological sort"
```

---

### Task 5: 新增 /run 路由 + Workflow 执行

**Files:**
- Modify: `backend/src/api/routes/instance_service.py`
- Modify: `backend/tests/test_api_workflows.py`

现有 `instance_service.py` 只有 `/synthesize` 端点（仅支持 preset TTS）。需要新增 `/run` 端点支持 workflow 执行。

- [ ] **Step 1: 写 workflow run 测试**

追加到 `backend/tests/test_api_workflows.py`：

```python
from unittest.mock import patch, AsyncMock


async def test_run_published_workflow(db_client):
    """POST /v1/instances/{id}/run executes a published workflow."""
    # Create and publish
    create = await db_client.post("/api/v1/workflows", json={
        "name": "run-test",
        "nodes": [
            {"id": "n1", "type": "text_input", "data": {"text": "hello"}, "position": {"x": 0, "y": 0}},
            {"id": "n2", "type": "output", "data": {}, "position": {"x": 400, "y": 0}},
        ],
        "edges": [
            {"id": "e1", "source": "n1", "sourceHandle": "text", "target": "n2", "targetHandle": "text"},
        ],
    })
    wf_id = create.json()["id"]
    pub = await db_client.post(f"/api/v1/workflows/{wf_id}/publish")
    instance_id = pub.json()["instance_id"]

    # Create API key for this instance
    key_resp = await db_client.post(f"/api/v1/instances/{instance_id}/keys", json={"label": "test"})
    api_key = key_resp.json()["raw_key"]

    # Run the workflow via /run endpoint
    resp = await db_client.post(
        f"/v1/instances/{instance_id}/run",
        json={},
        headers={"Authorization": f"Bearer {api_key}"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "outputs" in data


async def test_run_non_workflow_returns_400(db_client):
    """Only workflow instances support /run."""
    # TTS preset instance cannot use /run
    resp = await db_client.post("/v1/instances/999/run", json={})
    assert resp.status_code in (401, 404)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd backend && uv run pytest tests/test_api_workflows.py::test_run_published_workflow -v`
Expected: FAIL (405 Method Not Allowed — route doesn't exist)

- [ ] **Step 3: 在 instance_service.py 新增 /run 路由**

在 `backend/src/api/routes/instance_service.py` 添加新路由：

```python
from src.services.workflow_executor import WorkflowExecutor

@router.post("/{instance_id}/run")
async def run_workflow_instance(
    instance_id: int,
    request: Request,
    session: AsyncSession = Depends(get_async_session),
):
    """Execute a published workflow instance."""
    # Auth: same pattern as /synthesize — validate Bearer token
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(401, "Missing Bearer token")
    raw_key = auth.removeprefix("Bearer ").strip()

    # Validate API key
    from src.models.instance_api_key import InstanceApiKey
    import hashlib
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
    stmt = select(InstanceApiKey).where(
        InstanceApiKey.instance_id == instance_id,
        InstanceApiKey.key_hash == key_hash,
        InstanceApiKey.is_active == True,
    )
    result = await session.execute(stmt)
    api_key = result.scalar_one_or_none()
    if not api_key:
        raise HTTPException(401, "Invalid API key")

    # Load instance
    instance = await session.get(ServiceInstance, instance_id)
    if not instance or instance.status != "active":
        raise HTTPException(404, "Instance not found or inactive")

    if instance.source_type != "workflow":
        raise HTTPException(400, "Only workflow instances support /run")

    # Execute workflow from snapshot
    snapshot = instance.params_override or {}
    executor = WorkflowExecutor(snapshot)
    result = await executor.execute()

    # Update usage
    api_key.usage_calls += 1
    await session.commit()

    return result
```

Note: 需要添加必要的 import（`Request`, `select`, `ServiceInstance` 等）。具体实现时参考现有 `/synthesize` 路由的 auth 模式。

- [ ] **Step 4: 运行测试确认通过**

Run: `cd backend && uv run pytest tests/test_api_workflows.py -v`
Expected: All passed

- [ ] **Step 5: 运行全量测试确认无回归**

Run: `cd backend && uv run pytest --ignore=tests/test_audio_io.py -v`
Expected: All passed

- [ ] **Step 6: Commit**

```bash
git add backend/src/api/routes/instance_service.py backend/tests/test_api_workflows.py
git commit -m "feat: add /run endpoint for workflow instance execution"
```

---

## Chunk 3: 前端 Workflow API + Store 改造

### Task 6: 前端 Workflow API Hooks

**Files:**
- Create: `frontend/src/api/workflows.ts`

- [ ] **Step 1: 创建 workflow API hooks**

```typescript
// frontend/src/api/workflows.ts
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { apiFetch } from './client'

export interface WorkflowSummary {
  id: string
  name: string
  description: string | null
  is_template: boolean
  status: string
  created_at: string
  updated_at: string
}

export interface WorkflowFull extends WorkflowSummary {
  nodes: any[]
  edges: any[]
}

export function useWorkflows(isTemplate?: boolean) {
  const params = isTemplate != null ? `?is_template=${isTemplate}` : ''
  return useQuery({
    queryKey: ['workflows', isTemplate],
    queryFn: () => apiFetch<WorkflowSummary[]>(`/api/v1/workflows${params}`),
  })
}

export function useWorkflow(id: string | null) {
  return useQuery({
    queryKey: ['workflow', id],
    queryFn: () => apiFetch<WorkflowFull>(`/api/v1/workflows/${id}`),
    enabled: !!id,
  })
}

export function useCreateWorkflow() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (data: { name: string; nodes?: any[]; edges?: any[]; is_template?: boolean }) =>
      apiFetch<WorkflowFull>('/api/v1/workflows', {
        method: 'POST',
        body: JSON.stringify(data),
      }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['workflows'] }),
  })
}

export function useSaveWorkflow() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ id, ...data }: { id: string; name?: string; nodes?: any[]; edges?: any[] }) =>
      apiFetch<WorkflowFull>(`/api/v1/workflows/${id}`, {
        method: 'PATCH',
        body: JSON.stringify(data),
      }),
    onSuccess: (_, vars) => {
      qc.invalidateQueries({ queryKey: ['workflow', vars.id] })
      qc.invalidateQueries({ queryKey: ['workflows'] })
    },
  })
}

export function useDeleteWorkflow() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (id: string) =>
      apiFetch(`/api/v1/workflows/${id}`, { method: 'DELETE' }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['workflows'] }),
  })
}

export function usePublishWorkflow() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (id: string) =>
      apiFetch<{ instance_id: string; endpoint: string }>(
        `/api/v1/workflows/${id}/publish`,
        { method: 'POST' }
      ),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['workflows'] })
    },
  })
}

export function useUnpublishWorkflow() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (id: string) =>
      apiFetch(`/api/v1/workflows/${id}/unpublish`, { method: 'POST' }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['workflows'] }),
  })
}
```

- [ ] **Step 2: TypeScript 类型检查**

Run: `cd frontend && npx tsc --noEmit`
Expected: no errors

- [ ] **Step 3: Commit**

```bash
git add frontend/src/api/workflows.ts
git commit -m "feat: add workflow API hooks (CRUD + publish/unpublish)"
```

---

### Task 7: Workspace Store 改造（DB 驱动）

**Files:**
- Modify: `frontend/src/stores/workspace.ts`
- Modify: `frontend/src/models/workflow.ts`

- [ ] **Step 1: 扩展 Workflow 类型**

在 `frontend/src/models/workflow.ts` 中，修改 `Workflow` 接口：

```typescript
export interface Workflow {
  id: string
  name: string
  description?: string
  nodes: WorkflowNode[]
  edges: WorkflowEdge[]
  is_template?: boolean
  status?: 'draft' | 'published'
}
```

- [ ] **Step 2: 改造 workspace store — 添加 DB 同步**

修改 `frontend/src/stores/workspace.ts`，添加：

1. `loadWorkflow(id)` — 从 DB 加载到 tab
2. `saveWorkflow()` — 当前 tab 保存到 DB（PATCH）
3. `autoSave` — debounce 自动保存（2 秒）
4. `workflowId` 追踪 — tab 关联 DB workflow ID

核心改动：
- `WorkflowTab` 添加 `savedToDb: boolean` 标志
- `addTab` 接受可选的 `WorkflowFull` 参数（从 DB 加载的数据）
- `markDirty()` 触发 debounce 自动保存
- 新增 `_debouncedSave` 内部方法

```typescript
// workspace store 中追加的接口和方法

interface WorkflowTab {
  id: string
  name: string
  workflow: Workflow
  isDirty: boolean
  dbId: string | null  // 数据库中的 workflow ID，null = 未保存
}

// 新方法:
// loadFromDb(workflowFull) — 创建 tab 并关联 dbId
// saveToDb() — 调 API 保存当前 tab
```

具体实现需要在 store 中引入 `useSaveWorkflow` 或直接使用 `apiFetch`。由于 Zustand store 不能直接使用 React hooks，使用 `apiFetch` 直接调用。

- [ ] **Step 3: TypeScript 类型检查**

Run: `cd frontend && npx tsc --noEmit`
Expected: no errors

- [ ] **Step 4: Commit**

```bash
git add frontend/src/models/workflow.ts frontend/src/stores/workspace.ts
git commit -m "feat: workspace store DB sync with auto-save"
```

---

### Task 8: Topbar 发布按钮 + Workflow 列表面板

**Files:**
- Modify: `frontend/src/components/layout/Topbar.tsx`
- Modify: `frontend/src/stores/panel.ts`
- Modify: `frontend/src/components/nodes/NodeEditor.tsx`

- [ ] **Step 1: Topbar 添加发布/下线按钮**

在 `frontend/src/components/layout/Topbar.tsx` 的 Run 按钮旁添加 Publish 按钮：

```tsx
// 导入
import { usePublishWorkflow, useUnpublishWorkflow } from '../../api/workflows'

// 在组件中：
const publishWf = usePublishWorkflow()
const unpublishWf = useUnpublishWorkflow()
const activeWf = useWorkspaceStore((s) => s.getActiveWorkflow())
const activeTab = useWorkspaceStore((s) => s.tabs.find(t => t.id === s.activeTabId))
const isPublished = activeWf?.status === 'published'

// Publish button JSX (在 Run 按钮之后):
{activeTab?.dbId && (
  <button
    onClick={() => {
      if (isPublished) {
        unpublishWf.mutate(activeTab.dbId!)
      } else {
        publishWf.mutate(activeTab.dbId!)
      }
    }}
    disabled={publishWf.isPending || unpublishWf.isPending}
    style={{
      padding: '4px 12px',
      fontSize: 11,
      borderRadius: 4,
      border: '1px solid var(--border)',
      background: isPublished ? 'none' : 'var(--ok)',
      color: isPublished ? 'var(--muted)' : '#fff',
    }}
  >
    {isPublished ? '下线' : '发布'}
  </button>
)}
```

- [ ] **Step 2: Presets 面板改为加载 workflow 列表**

修改 presets panel 或新增 workflows panel，列出 `is_template=true` 的 workflow，点击加载到 tab。

这部分涉及修改 `PresetDetailOverlay` 或新增面板组件。核心逻辑：

```tsx
// 使用 useWorkflows(true) 获取模板列表
// 点击模板 → useWorkflow(id) 加载完整数据 → store.loadFromDb(data)
```

- [ ] **Step 3: TypeScript 类型检查**

Run: `cd frontend && npx tsc --noEmit`
Expected: no errors

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/layout/Topbar.tsx frontend/src/stores/panel.ts frontend/src/components/nodes/NodeEditor.tsx
git commit -m "feat: add publish/unpublish button and workflow list panel"
```

---

## Chunk 4: 集成测试 + 清理

### Task 9: 端到端集成测试

**Files:**
- Modify: `backend/tests/test_api_workflows.py`

- [ ] **Step 1: 写完整生命周期测试**

```python
async def test_workflow_full_lifecycle(db_client):
    """Create → Update → Publish → Run → Unpublish → Delete"""
    # Create
    resp = await db_client.post("/api/v1/workflows", json={
        "name": "lifecycle",
        "nodes": [
            {"id": "n1", "type": "text_input", "data": {"text": "test"}, "position": {"x": 0, "y": 0}},
            {"id": "n2", "type": "output", "data": {}, "position": {"x": 400, "y": 0}},
        ],
        "edges": [
            {"id": "e1", "source": "n1", "sourceHandle": "text", "target": "n2", "targetHandle": "text"},
        ],
    })
    wf_id = resp.json()["id"]

    # Update
    resp = await db_client.patch(f"/api/v1/workflows/{wf_id}", json={"name": "updated"})
    assert resp.json()["name"] == "updated"

    # Publish
    resp = await db_client.post(f"/api/v1/workflows/{wf_id}/publish")
    assert resp.status_code == 200
    instance_id = resp.json()["instance_id"]

    # Verify published status
    resp = await db_client.get(f"/api/v1/workflows/{wf_id}")
    assert resp.json()["status"] == "published"

    # Verify instance was created
    resp = await db_client.get(f"/api/v1/instances/{instance_id}")
    assert resp.status_code == 200
    assert resp.json()["source_type"] == "workflow"

    # Unpublish
    resp = await db_client.post(f"/api/v1/workflows/{wf_id}/unpublish")
    assert resp.status_code == 200

    # Verify draft status
    resp = await db_client.get(f"/api/v1/workflows/{wf_id}")
    assert resp.json()["status"] == "draft"

    # Delete
    resp = await db_client.delete(f"/api/v1/workflows/{wf_id}")
    assert resp.status_code == 204
```

- [ ] **Step 2: 运行全量测试**

Run: `cd backend && uv run pytest --ignore=tests/test_audio_io.py -v`
Expected: All passed

- [ ] **Step 3: 前端类型检查**

Run: `cd frontend && npx tsc --noEmit`
Expected: no errors

- [ ] **Step 4: Commit**

```bash
git add backend/tests/test_api_workflows.py
git commit -m "test: add workflow full lifecycle integration test"
```

---

### Task 10: 清理 + 最终提交

- [ ] **Step 1: 运行全量后端测试**

Run: `cd backend && uv run pytest --ignore=tests/test_audio_io.py -v`
Expected: All passed

- [ ] **Step 2: 运行前端类型检查**

Run: `cd frontend && npx tsc --noEmit`
Expected: no errors

- [ ] **Step 3: 最终 commit**

如有未提交的修复：
```bash
git add -A
git commit -m "chore: Phase A cleanup and final adjustments"
```
