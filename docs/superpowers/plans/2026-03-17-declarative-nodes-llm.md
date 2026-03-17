# Phase C-1: 声明式组件模型 + LLM 节点 — 实现计划

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将节点系统从手写 React 组件改为声明式数据驱动，并新增 LLM、Prompt Template、Agent、If-Else 节点。

**Architecture:** 新增 `NodeRegistry`（JSON 定义 → UI 自动生成），`DeclarativeNode` 组件读取定义渲染 widgets。现有 8 个节点保留手写组件（复杂逻辑），新节点用声明式定义。后端 `_NODE_EXECUTORS` 同步扩展。

**Tech Stack:** FastAPI, httpx (OpenAI 兼容 API), React, Zustand, @xyflow/react

---

## 文件结构

### 后端新建/修改

| 文件 | 职责 |
|------|------|
| `backend/src/services/llm_service.py` | LLM 调用服务（vLLM / OpenAI 兼容） |
| `backend/src/services/workflow_executor.py` | 新增 llm / prompt_template / agent / if_else 执行器 |
| `backend/tests/test_llm_service.py` | LLM 服务测试 |
| `backend/tests/test_workflow_executor.py` | 新节点执行器测试 |

### 前端新建

| 文件 | 职责 |
|------|------|
| `frontend/src/models/nodeRegistry.ts` | 声明式节点定义注册表 |
| `frontend/src/components/nodes/DeclarativeNode.tsx` | 通用声明式节点渲染组件 |

### 前端修改

| 文件 | 职责 |
|------|------|
| `frontend/src/models/workflow.ts` | 扩展 NodeType 和 PortType |
| `frontend/src/components/nodes/nodeTypes.ts` | 注册新节点 |
| `frontend/src/components/panels/NodeLibraryPanel.tsx` | 新增节点分类 |

---

## Chunk 1: 声明式节点框架

### Task 1: 节点注册表 + DeclarativeNode

**Files:**
- Create: `frontend/src/models/nodeRegistry.ts`
- Create: `frontend/src/components/nodes/DeclarativeNode.tsx`
- Modify: `frontend/src/models/workflow.ts`

- [ ] **Step 1: 扩展 PortType 和 NodeType**

在 `frontend/src/models/workflow.ts` 中：

```typescript
// 扩展 PortType
export type PortType = 'text' | 'audio' | 'message' | 'data' | 'any'

// 扩展 NodeType — 新增节点类型
export type NodeType =
  | 'text_input' | 'ref_audio' | 'tts_engine' | 'resample'
  | 'mixer' | 'concat' | 'bgm_mix' | 'output'
  // Phase C 新增
  | 'llm' | 'prompt_template' | 'agent' | 'if_else'
```

同时在 `NODE_DEFS` 中添加新节点定义（inputs/outputs 端口定义）。

- [ ] **Step 2: 创建节点注册表**

```typescript
// frontend/src/models/nodeRegistry.ts
/**
 * 声明式节点定义注册表。
 * 新增节点只需在这里添加定义，UI 自动生成。
 */

export type WidgetType = 'input' | 'textarea' | 'select' | 'slider' | 'model_select' | 'agent_select' | 'checkbox'

export interface WidgetDef {
  name: string
  label: string
  widget: WidgetType
  type?: 'string' | 'number' | 'boolean'
  default?: unknown
  required?: boolean
  // select options
  options?: { label: string; value: string }[]
  // slider range
  min?: number
  max?: number
  step?: number
  // dynamic options
  dynamic?: boolean
  dependsOn?: string
  // textarea rows
  rows?: number
}

export interface DeclarativeNodeDef {
  type: string
  label: string
  category: string
  badge?: string
  badgeColor?: string
  icon?: string
  widgets: WidgetDef[]
}

/**
 * Registry of declarative node definitions.
 * Nodes defined here will be rendered by DeclarativeNode component.
 * Existing hand-written nodes (text_input, tts_engine, etc.) are NOT included here.
 */
export const DECLARATIVE_NODES: Record<string, DeclarativeNodeDef> = {
  llm: {
    type: 'llm',
    label: 'LLM',
    category: 'ai',
    badge: 'AI',
    badgeColor: 'var(--accent)',
    widgets: [
      { name: 'system', label: '系统提示', widget: 'textarea', rows: 3 },
      { name: 'model', label: '模型', widget: 'input', default: '' },
      { name: 'base_url', label: 'API 地址', widget: 'input', default: 'http://localhost:8100' },
      { name: 'api_key', label: 'API Key', widget: 'input' },
      { name: 'temperature', label: '温度', widget: 'slider', min: 0, max: 2, step: 0.1, default: 0.7 },
      { name: 'max_tokens', label: '最大 Token', widget: 'slider', min: 1, max: 8192, step: 1, default: 2048 },
    ],
  },
  prompt_template: {
    type: 'prompt_template',
    label: '提示模板',
    category: 'ai',
    badge: 'AI',
    badgeColor: 'var(--accent)',
    widgets: [
      { name: 'template', label: '模板', widget: 'textarea', rows: 5, default: '请将以下文本翻译为{language}：\n\n{text}' },
    ],
  },
  agent: {
    type: 'agent',
    label: 'Agent',
    category: 'ai',
    badge: 'AI',
    badgeColor: 'var(--accent)',
    widgets: [
      { name: 'agent_name', label: 'Agent', widget: 'agent_select' },
    ],
  },
  if_else: {
    type: 'if_else',
    label: '条件分支',
    category: 'control',
    badge: '控制',
    badgeColor: 'var(--warn)',
    widgets: [
      { name: 'condition', label: '条件', widget: 'input', default: '' },
      { name: 'match_type', label: '匹配方式', widget: 'select', options: [
        { label: '包含', value: 'contains' },
        { label: '等于', value: 'equals' },
        { label: '正则', value: 'regex' },
        { label: '非空', value: 'not_empty' },
      ], default: 'not_empty' },
    ],
  },
}

/** All categories for the node library panel */
export const NODE_CATEGORIES: Record<string, { label: string; order: number }> = {
  tts: { label: '语音合成 TTS', order: 0 },
  audio_processing: { label: '音频处理', order: 1 },
  ai: { label: 'AI / LLM', order: 2 },
  control: { label: '流程控制', order: 3 },
}
```

- [ ] **Step 3: 创建 DeclarativeNode 组件**

```tsx
// frontend/src/components/nodes/DeclarativeNode.tsx
import { type NodeProps } from '@xyflow/react'
import { DECLARATIVE_NODES } from '../../models/nodeRegistry'
import { NODE_DEFS } from '../../models/workflow'
import { useWorkspaceStore } from '../../stores/workspace'
import { useAgents } from '../../api/agents'
import BaseNode, { NodeWidgetRow, NodeInput, NodeSelect, NodeNumberDrag, NodeTextarea } from './BaseNode'

export default function DeclarativeNode({ id, data, selected, type }: NodeProps) {
  const updateNode = useWorkspaceStore((s) => s.updateNode)
  const def = DECLARATIVE_NODES[type as string]
  const nodeDef = NODE_DEFS[type as keyof typeof NODE_DEFS]

  if (!def || !nodeDef) return null

  return (
    <BaseNode
      title={def.label}
      badge={def.badge}
      selected={!!selected}
      inputs={nodeDef.inputs}
      outputs={nodeDef.outputs}
    >
      {def.widgets.map((w) => {
        const value = (data as Record<string, unknown>)[w.name] ?? w.default ?? ''
        const onChange = (v: unknown) => updateNode(id, { [w.name]: v })

        switch (w.widget) {
          case 'input':
            return (
              <NodeWidgetRow key={w.name} label={w.label}>
                <NodeInput
                  value={String(value)}
                  onChange={(e) => onChange(e.target.value)}
                  placeholder={w.label}
                />
              </NodeWidgetRow>
            )
          case 'textarea':
            return (
              <NodeWidgetRow key={w.name} label={w.label}>
                <NodeTextarea
                  value={String(value)}
                  onChange={(e) => onChange(e.target.value)}
                  rows={w.rows ?? 3}
                  placeholder={w.label}
                />
              </NodeWidgetRow>
            )
          case 'select':
            return (
              <NodeWidgetRow key={w.name} label={w.label}>
                <NodeSelect
                  value={String(value)}
                  onChange={(e) => onChange(e.target.value)}
                >
                  {(w.options ?? []).map((o) => (
                    <option key={o.value} value={o.value}>{o.label}</option>
                  ))}
                </NodeSelect>
              </NodeWidgetRow>
            )
          case 'slider':
            return (
              <NodeWidgetRow key={w.name} label={w.label}>
                <NodeNumberDrag
                  value={Number(value)}
                  onChange={(v) => onChange(v)}
                  min={w.min ?? 0}
                  max={w.max ?? 100}
                  step={w.step ?? 1}
                />
              </NodeWidgetRow>
            )
          case 'agent_select':
            return <AgentSelectWidget key={w.name} label={w.label} value={String(value)} onChange={onChange} />
          default:
            return null
        }
      })}
    </BaseNode>
  )
}

function AgentSelectWidget({ label, value, onChange }: { label: string; value: string; onChange: (v: unknown) => void }) {
  const { data: agents } = useAgents()
  return (
    <NodeWidgetRow label={label}>
      <NodeSelect value={value} onChange={(e) => onChange(e.target.value)}>
        <option value="">选择 Agent...</option>
        {(agents ?? []).map((a) => (
          <option key={a.name} value={a.name}>{a.display_name}</option>
        ))}
      </NodeSelect>
    </NodeWidgetRow>
  )
}
```

- [ ] **Step 4: 注册新节点到 nodeTypes**

修改 `frontend/src/components/nodes/nodeTypes.ts`：

```typescript
import DeclarativeNode from './DeclarativeNode'
import { DECLARATIVE_NODES } from '../../models/nodeRegistry'

// 现有手写节点
const handwrittenTypes: NodeTypes = {
  text_input: TextInputNode,
  ref_audio: RefAudioNode,
  tts_engine: TTSEngineNode,
  output: OutputNode,
  resample: ResampleNode,
  concat: ConcatNode,
  mixer: MixerNode,
  bgm_mix: BgmMixNode,
}

// 声明式节点（自动注册）
const declarativeTypes: NodeTypes = Object.fromEntries(
  Object.keys(DECLARATIVE_NODES).map((type) => [type, DeclarativeNode])
)

export const nodeTypes: NodeTypes = { ...handwrittenTypes, ...declarativeTypes }
```

- [ ] **Step 5: 更新 NODE_DEFS — 新增端口定义**

在 `frontend/src/models/workflow.ts` 的 `NODE_DEFS` 中添加：

```typescript
llm: {
  type: 'llm',
  label: 'LLM',
  inputs: [{ id: 'prompt', type: 'text', label: '提示' }],
  outputs: [{ id: 'text', type: 'text', label: '输出' }],
},
prompt_template: {
  type: 'prompt_template',
  label: '提示模板',
  inputs: [{ id: 'text', type: 'text', label: '输入' }],
  outputs: [{ id: 'text', type: 'text', label: '输出' }],
},
agent: {
  type: 'agent',
  label: 'Agent',
  inputs: [{ id: 'text', type: 'text', label: '输入' }],
  outputs: [{ id: 'text', type: 'text', label: '文本' }],
},
if_else: {
  type: 'if_else',
  label: '条件分支',
  inputs: [{ id: 'input', type: 'text', label: '输入' }],
  outputs: [
    { id: 'true', type: 'text', label: '真' },
    { id: 'false', type: 'text', label: '假' },
  ],
},
```

- [ ] **Step 6: 更新 NodeLibraryPanel — 新增分类**

修改 `frontend/src/components/panels/NodeLibraryPanel.tsx`：

将现有硬编码的 categories 替换为从 `NODE_CATEGORIES` + `DECLARATIVE_NODES` 导入的动态分类。新增 `ai` 和 `control` 分类，包含 llm、prompt_template、agent、if_else 节点。

- [ ] **Step 7: TypeScript 检查**

Run: `cd frontend && npx tsc --noEmit`
Expected: no errors

- [ ] **Step 8: Commit**

```bash
git add frontend/src/models/nodeRegistry.ts frontend/src/models/workflow.ts frontend/src/components/nodes/DeclarativeNode.tsx frontend/src/components/nodes/nodeTypes.ts frontend/src/components/panels/NodeLibraryPanel.tsx
git commit -m "feat: declarative node framework with LLM, Prompt, Agent, If-Else definitions"
```

---

## Chunk 2: 后端 LLM 服务 + 新节点执行器

### Task 2: LLM 调用服务

**Files:**
- Create: `backend/src/services/llm_service.py`
- Create: `backend/tests/test_llm_service.py`

- [ ] **Step 1: 写 LLM 服务测试**

```python
# backend/tests/test_llm_service.py
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from src.services.llm_service import call_llm


@pytest.mark.asyncio
async def test_call_llm_returns_text():
    """LLM service should return generated text."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "Hello world"}}]
    }

    with patch("src.services.llm_service.httpx.AsyncClient") as MockClient:
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        MockClient.return_value = mock_client

        result = await call_llm(
            prompt="Say hello",
            base_url="http://localhost:8100",
            model="test-model",
        )
    assert result == "Hello world"


@pytest.mark.asyncio
async def test_call_llm_with_system():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "你好"}}]
    }

    with patch("src.services.llm_service.httpx.AsyncClient") as MockClient:
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        MockClient.return_value = mock_client

        result = await call_llm(
            prompt="hello",
            system="你是翻译员",
            base_url="http://localhost:8100",
            model="test",
        )
    assert result == "你好"
    # Verify system message was sent
    call_args = mock_client.post.call_args
    messages = call_args[1]["json"]["messages"]
    assert messages[0]["role"] == "system"
```

- [ ] **Step 2: 实现 LLM 服务**

```python
# backend/src/services/llm_service.py
"""LLM service — calls vLLM or any OpenAI-compatible API."""

import logging

import httpx

logger = logging.getLogger(__name__)


async def call_llm(
    prompt: str,
    base_url: str = "http://localhost:8100",
    model: str = "",
    system: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
) -> str:
    """Call an OpenAI-compatible chat completions API."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    url = f"{base_url.rstrip('/')}/v1/chat/completions"

    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            url,
            json={
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            headers=headers,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
```

- [ ] **Step 3: 运行测试**

Run: `cd backend && uv run pytest tests/test_llm_service.py -v`
Expected: 2 passed

- [ ] **Step 4: Commit**

```bash
git add backend/src/services/llm_service.py backend/tests/test_llm_service.py
git commit -m "feat: add LLM service for OpenAI-compatible API calls"
```

---

### Task 3: 新节点执行器

**Files:**
- Modify: `backend/src/services/workflow_executor.py`
- Modify: `backend/tests/test_workflow_executor.py`

- [ ] **Step 1: 写新节点执行器测试**

追加到 `backend/tests/test_workflow_executor.py`：

```python
from unittest.mock import patch, AsyncMock


@pytest.mark.asyncio
async def test_exec_prompt_template():
    wf = {
        "nodes": [
            {"id": "n1", "type": "text_input", "data": {"text": "Hello"}, "position": {"x": 0, "y": 0}},
            {"id": "n2", "type": "prompt_template", "data": {"template": "Translate: {text}"}, "position": {"x": 200, "y": 0}},
            {"id": "n3", "type": "output", "data": {}, "position": {"x": 400, "y": 0}},
        ],
        "edges": [
            {"id": "e1", "source": "n1", "sourceHandle": "text", "target": "n2", "targetHandle": "text"},
            {"id": "e2", "source": "n2", "sourceHandle": "text", "target": "n3", "targetHandle": "text"},
        ],
    }
    executor = WorkflowExecutor(wf)
    result = await executor.execute()
    assert result["outputs"]["n2"]["text"] == "Translate: Hello"


@pytest.mark.asyncio
async def test_exec_llm():
    wf = {
        "nodes": [
            {"id": "n1", "type": "text_input", "data": {"text": "Hello"}, "position": {"x": 0, "y": 0}},
            {"id": "n2", "type": "llm", "data": {"model": "test", "base_url": "http://localhost:8100"}, "position": {"x": 200, "y": 0}},
            {"id": "n3", "type": "output", "data": {}, "position": {"x": 400, "y": 0}},
        ],
        "edges": [
            {"id": "e1", "source": "n1", "sourceHandle": "text", "target": "n2", "targetHandle": "prompt"},
            {"id": "e2", "source": "n2", "sourceHandle": "text", "target": "n3", "targetHandle": "text"},
        ],
    }
    with patch("src.services.workflow_executor.call_llm", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = "Hi there"
        executor = WorkflowExecutor(wf)
        result = await executor.execute()
    assert result["outputs"]["n2"]["text"] == "Hi there"


@pytest.mark.asyncio
async def test_exec_if_else_true():
    wf = {
        "nodes": [
            {"id": "n1", "type": "text_input", "data": {"text": "hello world"}, "position": {"x": 0, "y": 0}},
            {"id": "n2", "type": "if_else", "data": {"condition": "hello", "match_type": "contains"}, "position": {"x": 200, "y": 0}},
            {"id": "n3", "type": "output", "data": {}, "position": {"x": 400, "y": 0}},
        ],
        "edges": [
            {"id": "e1", "source": "n1", "sourceHandle": "text", "target": "n2", "targetHandle": "input"},
            {"id": "e2", "source": "n2", "sourceHandle": "true", "target": "n3", "targetHandle": "text"},
        ],
    }
    executor = WorkflowExecutor(wf)
    result = await executor.execute()
    assert result["outputs"]["n2"]["true"] == "hello world"
    assert result["outputs"]["n2"]["false"] == ""


@pytest.mark.asyncio
async def test_exec_if_else_false():
    wf = {
        "nodes": [
            {"id": "n1", "type": "text_input", "data": {"text": "goodbye"}, "position": {"x": 0, "y": 0}},
            {"id": "n2", "type": "if_else", "data": {"condition": "hello", "match_type": "contains"}, "position": {"x": 200, "y": 0}},
            {"id": "n3", "type": "output", "data": {}, "position": {"x": 400, "y": 0}},
        ],
        "edges": [
            {"id": "e1", "source": "n1", "sourceHandle": "text", "target": "n2", "targetHandle": "input"},
            {"id": "e2", "source": "n2", "sourceHandle": "false", "target": "n3", "targetHandle": "text"},
        ],
    }
    executor = WorkflowExecutor(wf)
    result = await executor.execute()
    assert result["outputs"]["n2"]["true"] == ""
    assert result["outputs"]["n2"]["false"] == "goodbye"
```

- [ ] **Step 2: 实现新节点执行器**

在 `backend/src/services/workflow_executor.py` 中添加：

```python
import re
from src.services.llm_service import call_llm


async def _exec_llm(data: dict, inputs: dict) -> dict:
    """Call LLM via OpenAI-compatible API."""
    prompt = inputs.get("prompt", inputs.get("text", ""))
    if not prompt:
        raise ExecutionError("LLM 节点缺少输入")

    result = await call_llm(
        prompt=prompt,
        base_url=data.get("base_url", "http://localhost:8100"),
        model=data.get("model", ""),
        system=data.get("system"),
        api_key=data.get("api_key"),
        temperature=data.get("temperature", 0.7),
        max_tokens=data.get("max_tokens", 2048),
    )
    return {"text": result}


async def _exec_prompt_template(data: dict, inputs: dict) -> dict:
    """Replace {variables} in template with inputs."""
    template = data.get("template", "")
    # Merge all inputs as variables
    variables = {**inputs}
    # Simple string format — replace {key} with value
    result = template
    for key, value in variables.items():
        result = result.replace(f"{{{key}}}", str(value))
    return {"text": result}


async def _exec_agent(data: dict, inputs: dict) -> dict:
    """Execute a pre-configured agent (placeholder — full implementation in later iteration)."""
    agent_name = data.get("agent_name", "")
    input_text = inputs.get("text", "")
    if not agent_name:
        raise ExecutionError("Agent 节点未选择 Agent")
    # TODO: Load agent config, assemble system prompt from MD files, call LLM
    # For now, pass through with agent context
    return {"text": f"[Agent:{agent_name}] {input_text}"}


async def _exec_if_else(data: dict, inputs: dict) -> dict:
    """Conditional branching: routes input to 'true' or 'false' output."""
    input_text = inputs.get("input", inputs.get("text", ""))
    condition = data.get("condition", "")
    match_type = data.get("match_type", "not_empty")

    matched = False
    if match_type == "contains":
        matched = condition in str(input_text)
    elif match_type == "equals":
        matched = str(input_text) == condition
    elif match_type == "regex":
        matched = bool(re.search(condition, str(input_text)))
    elif match_type == "not_empty":
        matched = bool(input_text)

    return {
        "true": str(input_text) if matched else "",
        "false": str(input_text) if not matched else "",
    }
```

然后在 `_NODE_EXECUTORS` dict 中添加：

```python
_NODE_EXECUTORS = {
    # ... existing ...
    "llm": _exec_llm,
    "prompt_template": _exec_prompt_template,
    "agent": _exec_agent,
    "if_else": _exec_if_else,
}
```

- [ ] **Step 3: 运行测试**

Run: `cd backend && uv run pytest tests/test_workflow_executor.py -v`
Expected: all passed

- [ ] **Step 4: 运行全量测试**

Run: `cd backend && uv run pytest --ignore=tests/test_audio_io.py -v`
Expected: all passed

- [ ] **Step 5: Commit**

```bash
git add backend/src/services/workflow_executor.py backend/tests/test_workflow_executor.py
git commit -m "feat: add LLM, Prompt Template, Agent, If-Else node executors"
```

---

## Chunk 3: 集成验证

### Task 4: 端到端验证 + 清理

- [ ] **Step 1: 运行全量后端测试**

Run: `cd backend && uv run pytest --ignore=tests/test_audio_io.py -v`
Expected: all passed

- [ ] **Step 2: 运行前端类型检查**

Run: `cd frontend && npx tsc --noEmit`
Expected: no errors

- [ ] **Step 3: Commit（如有修复）**

```bash
git add -A && git commit -m "chore: Phase C-1 cleanup"
```
