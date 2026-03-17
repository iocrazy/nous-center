# Phase B: Agent 系统 + Skills/Prompts 管理 — 实现计划

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现文件驱动的 Agent 系统，支持 Skills（SKILL.md）和 Prompts（AGENT.md/SOUL.md/IDENTITY.md）管理，提供 CRUD API 和前端管理界面。

**Architecture:** 文件系统为唯一数据源（`~/.nous-center/agents/` + `~/.nous-center/skills/`）。后端扫描目录、解析 MD frontmatter、提供 REST API。前端新增 Agent 管理 Overlay 含内嵌 MD 编辑器。

**Tech Stack:** FastAPI, Python pathlib + yaml, React, Zustand, TanStack Query, CodeMirror（MD 编辑器）

---

## 文件结构

### 后端新建

| 文件 | 职责 |
|------|------|
| `backend/src/services/agent_manager.py` | Agent 加载、CRUD（文件系统操作） |
| `backend/src/services/skill_manager.py` | Skill 加载、解析 SKILL.md frontmatter |
| `backend/src/api/routes/agents.py` | Agent REST API |
| `backend/src/api/routes/skills.py` | Skill REST API |
| `backend/tests/test_api_agents.py` | Agent API 测试 |
| `backend/tests/test_api_skills.py` | Skill API 测试 |

### 后端修改

| 文件 | 变更 |
|------|------|
| `backend/src/api/main.py` | 注册 agents + skills 路由 |
| `backend/src/config.py` | 新增 `NOUS_CENTER_HOME` 设置 |

### 前端新建

| 文件 | 职责 |
|------|------|
| `frontend/src/api/agents.ts` | Agent API hooks |
| `frontend/src/api/skills.ts` | Skill API hooks |
| `frontend/src/components/overlays/AgentManagementOverlay.tsx` | Agent 管理页面 |

### 前端修改

| 文件 | 变更 |
|------|------|
| `frontend/src/stores/panel.ts` | 新增 'agents' OverlayId |
| `frontend/src/components/layout/IconRail.tsx` | 新增 Agent 图标 |
| `frontend/src/components/nodes/NodeEditor.tsx` | 渲染 AgentManagementOverlay |

---

## Chunk 1: 后端 Agent 管理

### Task 1: Config + Agent 管理服务

**Files:**
- Modify: `backend/src/config.py`
- Create: `backend/src/services/agent_manager.py`

- [ ] **Step 1: 添加 NOUS_CENTER_HOME 到 Settings**

在 `backend/src/config.py` 的 `Settings` 类中添加：

```python
NOUS_CENTER_HOME: str = "~/.nous-center"
```

- [ ] **Step 2: 创建 agent_manager.py**

```python
# backend/src/services/agent_manager.py
"""File-based agent management. ~/.nous-center/agents/ is the source of truth."""

import json
import logging
from pathlib import Path
from typing import Any

import yaml

from src.config import get_settings

logger = logging.getLogger(__name__)

PROMPT_FILES = ["AGENT.md", "SOUL.md", "IDENTITY.md"]


def _agents_dir() -> Path:
    settings = get_settings()
    return Path(settings.NOUS_CENTER_HOME).expanduser() / "agents"


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def list_agents() -> list[dict[str, Any]]:
    """List all agents by scanning the agents directory."""
    agents_dir = _agents_dir()
    if not agents_dir.exists():
        return []
    result = []
    for agent_dir in sorted(agents_dir.iterdir()):
        if not agent_dir.is_dir():
            continue
        config = _load_agent_config(agent_dir)
        config["name"] = agent_dir.name
        result.append(config)
    return result


def get_agent(name: str) -> dict[str, Any] | None:
    """Load a single agent's config + prompt file contents."""
    agent_dir = _agents_dir() / name
    if not agent_dir.is_dir():
        return None
    config = _load_agent_config(agent_dir)
    config["name"] = name
    # Load prompt files
    prompts = {}
    for fname in PROMPT_FILES:
        fpath = agent_dir / fname
        if fpath.exists():
            prompts[fname] = fpath.read_text(encoding="utf-8")
        else:
            prompts[fname] = ""
    config["prompts"] = prompts
    return config


def create_agent(name: str, display_name: str | None = None) -> dict[str, Any]:
    """Create a new agent directory with default files."""
    agent_dir = _agents_dir() / name
    if agent_dir.exists():
        raise ValueError(f"Agent '{name}' already exists")
    _ensure_dir(agent_dir)

    config = {
        "display_name": display_name or name,
        "model": {"engine_key": None, "fallback_api": None},
        "skills": [],
        "tools_policy": {},
        "status": "active",
    }
    _save_agent_config(agent_dir, config)

    # Create default prompt files
    for fname in PROMPT_FILES:
        (agent_dir / fname).write_text("", encoding="utf-8")

    config["name"] = name
    return config


def update_agent(name: str, updates: dict[str, Any]) -> dict[str, Any] | None:
    """Update agent config.json fields."""
    agent_dir = _agents_dir() / name
    if not agent_dir.is_dir():
        return None
    config = _load_agent_config(agent_dir)
    for key in ("display_name", "model", "skills", "tools_policy", "status"):
        if key in updates:
            config[key] = updates[key]
    _save_agent_config(agent_dir, config)
    config["name"] = name
    return config


def delete_agent(name: str) -> bool:
    """Delete agent directory."""
    import shutil
    agent_dir = _agents_dir() / name
    if not agent_dir.is_dir():
        return False
    shutil.rmtree(agent_dir)
    return True


def get_prompt(name: str, filename: str) -> str | None:
    """Get a prompt file's content."""
    if filename not in PROMPT_FILES:
        return None
    fpath = _agents_dir() / name / filename
    if not fpath.exists():
        return None
    return fpath.read_text(encoding="utf-8")


def save_prompt(name: str, filename: str, content: str) -> bool:
    """Save a prompt file."""
    if filename not in PROMPT_FILES:
        return False
    agent_dir = _agents_dir() / name
    if not agent_dir.is_dir():
        return False
    (agent_dir / filename).write_text(content, encoding="utf-8")
    return True


def _load_agent_config(agent_dir: Path) -> dict[str, Any]:
    config_path = agent_dir / "config.json"
    if config_path.exists():
        return json.loads(config_path.read_text(encoding="utf-8"))
    return {
        "display_name": agent_dir.name,
        "model": {"engine_key": None, "fallback_api": None},
        "skills": [],
        "tools_policy": {},
        "status": "active",
    }


def _save_agent_config(agent_dir: Path, config: dict[str, Any]) -> None:
    config_path = agent_dir / "config.json"
    # Don't save 'name' or 'prompts' in config.json (derived from directory)
    save_data = {k: v for k, v in config.items() if k not in ("name", "prompts")}
    config_path.write_text(
        json.dumps(save_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
```

- [ ] **Step 3: 验证 import**

Run: `cd backend && uv run python -c "from src.services.agent_manager import list_agents; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add backend/src/config.py backend/src/services/agent_manager.py
git commit -m "feat: add file-based agent manager service"
```

---

### Task 2: Skill 管理服务

**Files:**
- Create: `backend/src/services/skill_manager.py`

- [ ] **Step 1: 创建 skill_manager.py**

```python
# backend/src/services/skill_manager.py
"""File-based skill management. ~/.nous-center/skills/ is the source of truth."""

import logging
from pathlib import Path
from typing import Any

import yaml

from src.config import get_settings

logger = logging.getLogger(__name__)


def _skills_dir() -> Path:
    settings = get_settings()
    return Path(settings.NOUS_CENTER_HOME).expanduser() / "skills"


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _parse_skill_md(content: str) -> dict[str, Any]:
    """Parse SKILL.md with YAML frontmatter."""
    if not content.startswith("---"):
        return {"body": content}
    parts = content.split("---", 2)
    if len(parts) < 3:
        return {"body": content}
    try:
        frontmatter = yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError:
        frontmatter = {}
    return {**frontmatter, "body": parts[2].strip()}


def list_skills() -> list[dict[str, Any]]:
    """List all skills by scanning the skills directory."""
    skills_dir = _skills_dir()
    if not skills_dir.exists():
        return []
    result = []
    for skill_dir in sorted(skills_dir.iterdir()):
        if not skill_dir.is_dir():
            continue
        skill_md = skill_dir / "SKILL.md"
        if not skill_md.exists():
            continue
        parsed = _parse_skill_md(skill_md.read_text(encoding="utf-8"))
        result.append({
            "name": parsed.get("name", skill_dir.name),
            "description": parsed.get("description", ""),
            "requires": parsed.get("requires", {}),
            "dir_name": skill_dir.name,
        })
    return result


def get_skill(name: str) -> dict[str, Any] | None:
    """Get full skill content (frontmatter + body)."""
    skill_dir = _skills_dir() / name
    skill_md = skill_dir / "SKILL.md"
    if not skill_md.exists():
        return None
    content = skill_md.read_text(encoding="utf-8")
    parsed = _parse_skill_md(content)
    return {
        "name": parsed.get("name", name),
        "description": parsed.get("description", ""),
        "requires": parsed.get("requires", {}),
        "body": parsed.get("body", ""),
        "raw": content,
        "dir_name": name,
    }


def create_skill(name: str, description: str = "", body: str = "") -> dict[str, Any]:
    """Create a new skill directory with SKILL.md."""
    skill_dir = _skills_dir() / name
    if skill_dir.exists():
        raise ValueError(f"Skill '{name}' already exists")
    _ensure_dir(skill_dir)

    content = f"""---
name: {name}
description: {description}
requires:
  models: []
---

{body}
"""
    (skill_dir / "SKILL.md").write_text(content.strip() + "\n", encoding="utf-8")
    return get_skill(name)


def update_skill(name: str, raw_content: str) -> dict[str, Any] | None:
    """Update SKILL.md raw content."""
    skill_dir = _skills_dir() / name
    if not skill_dir.is_dir():
        return None
    (skill_dir / "SKILL.md").write_text(raw_content, encoding="utf-8")
    return get_skill(name)


def delete_skill(name: str) -> bool:
    """Delete skill directory."""
    import shutil
    skill_dir = _skills_dir() / name
    if not skill_dir.is_dir():
        return False
    shutil.rmtree(skill_dir)
    return True
```

- [ ] **Step 2: 验证 import**

Run: `cd backend && uv run python -c "from src.services.skill_manager import list_skills; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add backend/src/services/skill_manager.py
git commit -m "feat: add file-based skill manager with SKILL.md parser"
```

---

### Task 3: Agent API 路由 + 测试

**Files:**
- Create: `backend/src/api/routes/agents.py`
- Create: `backend/tests/test_api_agents.py`
- Modify: `backend/src/api/main.py`

- [ ] **Step 1: 写 Agent API 测试**

```python
# backend/tests/test_api_agents.py
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture
def agents_home(tmp_path):
    """Override NOUS_CENTER_HOME to temp dir for testing."""
    home = tmp_path / ".nous-center"
    home.mkdir()
    (home / "agents").mkdir()
    (home / "skills").mkdir()
    with patch("src.services.agent_manager.get_settings") as mock:
        mock.return_value.NOUS_CENTER_HOME = str(home)
        yield home


async def test_create_agent(client, agents_home):
    with patch("src.services.agent_manager.get_settings") as mock:
        mock.return_value.NOUS_CENTER_HOME = str(agents_home.parent / ".nous-center")
        resp = await client.post("/api/v1/agents", json={
            "name": "test-agent",
            "display_name": "测试 Agent",
        })
    assert resp.status_code == 201
    data = resp.json()
    assert data["name"] == "test-agent"
    assert data["display_name"] == "测试 Agent"
    assert data["status"] == "active"


async def test_create_agent_duplicate(client, agents_home):
    with patch("src.services.agent_manager.get_settings") as mock:
        mock.return_value.NOUS_CENTER_HOME = str(agents_home.parent / ".nous-center")
        await client.post("/api/v1/agents", json={"name": "dup"})
        resp = await client.post("/api/v1/agents", json={"name": "dup"})
    assert resp.status_code == 409


async def test_list_agents(client, agents_home):
    with patch("src.services.agent_manager.get_settings") as mock:
        mock.return_value.NOUS_CENTER_HOME = str(agents_home.parent / ".nous-center")
        await client.post("/api/v1/agents", json={"name": "a1"})
        await client.post("/api/v1/agents", json={"name": "a2"})
        resp = await client.get("/api/v1/agents")
    assert resp.status_code == 200
    assert len(resp.json()) == 2


async def test_get_agent(client, agents_home):
    with patch("src.services.agent_manager.get_settings") as mock:
        mock.return_value.NOUS_CENTER_HOME = str(agents_home.parent / ".nous-center")
        await client.post("/api/v1/agents", json={"name": "myagent"})
        resp = await client.get("/api/v1/agents/myagent")
    assert resp.status_code == 200
    data = resp.json()
    assert "prompts" in data
    assert "AGENT.md" in data["prompts"]


async def test_get_agent_not_found(client, agents_home):
    with patch("src.services.agent_manager.get_settings") as mock:
        mock.return_value.NOUS_CENTER_HOME = str(agents_home.parent / ".nous-center")
        resp = await client.get("/api/v1/agents/nonexist")
    assert resp.status_code == 404


async def test_update_agent(client, agents_home):
    with patch("src.services.agent_manager.get_settings") as mock:
        mock.return_value.NOUS_CENTER_HOME = str(agents_home.parent / ".nous-center")
        await client.post("/api/v1/agents", json={"name": "upd"})
        resp = await client.patch("/api/v1/agents/upd", json={
            "display_name": "Updated",
            "skills": ["tts-synthesis"],
        })
    assert resp.status_code == 200
    assert resp.json()["display_name"] == "Updated"
    assert resp.json()["skills"] == ["tts-synthesis"]


async def test_delete_agent(client, agents_home):
    with patch("src.services.agent_manager.get_settings") as mock:
        mock.return_value.NOUS_CENTER_HOME = str(agents_home.parent / ".nous-center")
        await client.post("/api/v1/agents", json={"name": "del"})
        resp = await client.delete("/api/v1/agents/del")
    assert resp.status_code == 204


async def test_update_prompt(client, agents_home):
    with patch("src.services.agent_manager.get_settings") as mock:
        mock.return_value.NOUS_CENTER_HOME = str(agents_home.parent / ".nous-center")
        await client.post("/api/v1/agents", json={"name": "prompt-test"})
        resp = await client.put(
            "/api/v1/agents/prompt-test/prompts/SOUL.md",
            content="你是一个播客主播",
            headers={"Content-Type": "text/plain"},
        )
    assert resp.status_code == 200

    with patch("src.services.agent_manager.get_settings") as mock:
        mock.return_value.NOUS_CENTER_HOME = str(agents_home.parent / ".nous-center")
        resp = await client.get("/api/v1/agents/prompt-test")
    assert resp.json()["prompts"]["SOUL.md"] == "你是一个播客主播"
```

- [ ] **Step 2: 实现 Agent API 路由**

```python
# backend/src/api/routes/agents.py
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from src.services import agent_manager

router = APIRouter(prefix="/api/v1/agents", tags=["agents"])


class AgentCreate(BaseModel):
    name: str
    display_name: str | None = None


class AgentUpdate(BaseModel):
    display_name: str | None = None
    model: dict | None = None
    skills: list[str] | None = None
    tools_policy: dict | None = None
    status: str | None = None


@router.post("", status_code=201)
async def create_agent(body: AgentCreate):
    try:
        return agent_manager.create_agent(body.name, body.display_name)
    except ValueError as e:
        raise HTTPException(409, str(e))


@router.get("")
async def list_agents():
    return agent_manager.list_agents()


@router.get("/{name}")
async def get_agent(name: str):
    agent = agent_manager.get_agent(name)
    if not agent:
        raise HTTPException(404, "Agent not found")
    return agent


@router.patch("/{name}")
async def update_agent(name: str, body: AgentUpdate):
    result = agent_manager.update_agent(name, body.model_dump(exclude_unset=True))
    if not result:
        raise HTTPException(404, "Agent not found")
    return result


@router.delete("/{name}", status_code=204)
async def delete_agent(name: str):
    if not agent_manager.delete_agent(name):
        raise HTTPException(404, "Agent not found")


@router.put("/{name}/prompts/{filename}")
async def update_prompt(name: str, filename: str, request: Request):
    content = (await request.body()).decode("utf-8")
    if not agent_manager.save_prompt(name, filename, content):
        raise HTTPException(404, "Agent or file not found")
    return {"status": "saved"}
```

- [ ] **Step 3: 注册路由**

在 `backend/src/api/main.py`:
- Import: `from src.api.routes import agents`
- Router: `app.include_router(agents.router)`

- [ ] **Step 4: 运行测试**

Run: `cd backend && uv run pytest tests/test_api_agents.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add backend/src/api/routes/agents.py backend/tests/test_api_agents.py backend/src/api/main.py
git commit -m "feat: add Agent CRUD API with file-based storage"
```

---

### Task 4: Skill API 路由 + 测试

**Files:**
- Create: `backend/src/api/routes/skills.py`
- Create: `backend/tests/test_api_skills.py`
- Modify: `backend/src/api/main.py`

- [ ] **Step 1: 写 Skill API 测试**

```python
# backend/tests/test_api_skills.py
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture
def skills_home(tmp_path):
    home = tmp_path / ".nous-center"
    home.mkdir()
    (home / "skills").mkdir()
    with patch("src.services.skill_manager.get_settings") as mock:
        mock.return_value.NOUS_CENTER_HOME = str(home)
        yield home


async def test_create_skill(client, skills_home):
    with patch("src.services.skill_manager.get_settings") as mock:
        mock.return_value.NOUS_CENTER_HOME = str(skills_home.parent / ".nous-center")
        resp = await client.post("/api/v1/skills", json={
            "name": "tts-synthesis",
            "description": "将文本合成为语音",
        })
    assert resp.status_code == 201
    data = resp.json()
    assert data["name"] == "tts-synthesis"
    assert data["description"] == "将文本合成为语音"


async def test_list_skills(client, skills_home):
    with patch("src.services.skill_manager.get_settings") as mock:
        mock.return_value.NOUS_CENTER_HOME = str(skills_home.parent / ".nous-center")
        await client.post("/api/v1/skills", json={"name": "s1", "description": "d1"})
        await client.post("/api/v1/skills", json={"name": "s2", "description": "d2"})
        resp = await client.get("/api/v1/skills")
    assert resp.status_code == 200
    assert len(resp.json()) == 2


async def test_get_skill(client, skills_home):
    with patch("src.services.skill_manager.get_settings") as mock:
        mock.return_value.NOUS_CENTER_HOME = str(skills_home.parent / ".nous-center")
        await client.post("/api/v1/skills", json={"name": "my-skill", "description": "test"})
        resp = await client.get("/api/v1/skills/my-skill")
    assert resp.status_code == 200
    data = resp.json()
    assert "raw" in data
    assert "body" in data


async def test_update_skill(client, skills_home):
    with patch("src.services.skill_manager.get_settings") as mock:
        mock.return_value.NOUS_CENTER_HOME = str(skills_home.parent / ".nous-center")
        await client.post("/api/v1/skills", json={"name": "upd-skill"})
        new_content = "---\nname: upd-skill\ndescription: updated\n---\n\nNew body"
        resp = await client.put("/api/v1/skills/upd-skill", content=new_content,
                                headers={"Content-Type": "text/plain"})
    assert resp.status_code == 200
    assert resp.json()["description"] == "updated"


async def test_delete_skill(client, skills_home):
    with patch("src.services.skill_manager.get_settings") as mock:
        mock.return_value.NOUS_CENTER_HOME = str(skills_home.parent / ".nous-center")
        await client.post("/api/v1/skills", json={"name": "del-skill"})
        resp = await client.delete("/api/v1/skills/del-skill")
    assert resp.status_code == 204


async def test_get_skill_not_found(client, skills_home):
    with patch("src.services.skill_manager.get_settings") as mock:
        mock.return_value.NOUS_CENTER_HOME = str(skills_home.parent / ".nous-center")
        resp = await client.get("/api/v1/skills/nope")
    assert resp.status_code == 404
```

- [ ] **Step 2: 实现 Skill API 路由**

```python
# backend/src/api/routes/skills.py
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from src.services import skill_manager

router = APIRouter(prefix="/api/v1/skills", tags=["skills"])


class SkillCreate(BaseModel):
    name: str
    description: str = ""
    body: str = ""


@router.post("", status_code=201)
async def create_skill(body: SkillCreate):
    try:
        return skill_manager.create_skill(body.name, body.description, body.body)
    except ValueError as e:
        raise HTTPException(409, str(e))


@router.get("")
async def list_skills():
    return skill_manager.list_skills()


@router.get("/{name}")
async def get_skill(name: str):
    skill = skill_manager.get_skill(name)
    if not skill:
        raise HTTPException(404, "Skill not found")
    return skill


@router.put("/{name}")
async def update_skill(name: str, request: Request):
    content = (await request.body()).decode("utf-8")
    result = skill_manager.update_skill(name, content)
    if not result:
        raise HTTPException(404, "Skill not found")
    return result


@router.delete("/{name}", status_code=204)
async def delete_skill(name: str):
    if not skill_manager.delete_skill(name):
        raise HTTPException(404, "Skill not found")
```

- [ ] **Step 3: 注册路由**

在 `backend/src/api/main.py`:
- Import: `from src.api.routes import skills`
- Router: `app.include_router(skills.router)`

- [ ] **Step 4: 运行测试**

Run: `cd backend && uv run pytest tests/test_api_skills.py -v`
Expected: 6 passed

- [ ] **Step 5: 运行全量后端测试**

Run: `cd backend && uv run pytest --ignore=tests/test_audio_io.py -v`
Expected: All passed

- [ ] **Step 6: Commit**

```bash
git add backend/src/api/routes/skills.py backend/tests/test_api_skills.py backend/src/api/main.py
git commit -m "feat: add Skill CRUD API with SKILL.md frontmatter parsing"
```

---

## Chunk 2: 前端 Agent 管理

### Task 5: 前端 API Hooks

**Files:**
- Create: `frontend/src/api/agents.ts`
- Create: `frontend/src/api/skills.ts`

- [ ] **Step 1: 创建 agents.ts**

```typescript
// frontend/src/api/agents.ts
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { apiFetch } from './client'

export interface AgentSummary {
  name: string
  display_name: string
  status: string
  skills: string[]
  model: { engine_key: string | null; fallback_api: string | null }
}

export interface AgentFull extends AgentSummary {
  tools_policy: Record<string, unknown>
  prompts: Record<string, string>
}

export function useAgents() {
  return useQuery({
    queryKey: ['agents'],
    queryFn: () => apiFetch<AgentSummary[]>('/api/v1/agents'),
  })
}

export function useAgent(name: string | null) {
  return useQuery({
    queryKey: ['agent', name],
    queryFn: () => apiFetch<AgentFull>(`/api/v1/agents/${name}`),
    enabled: !!name,
  })
}

export function useCreateAgent() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (data: { name: string; display_name?: string }) =>
      apiFetch('/api/v1/agents', { method: 'POST', body: JSON.stringify(data) }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['agents'] }),
  })
}

export function useUpdateAgent() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ name, ...data }: { name: string; display_name?: string; skills?: string[]; model?: any; status?: string }) =>
      apiFetch(`/api/v1/agents/${name}`, { method: 'PATCH', body: JSON.stringify(data) }),
    onSuccess: (_, vars) => {
      qc.invalidateQueries({ queryKey: ['agent', vars.name] })
      qc.invalidateQueries({ queryKey: ['agents'] })
    },
  })
}

export function useDeleteAgent() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (name: string) =>
      apiFetch(`/api/v1/agents/${name}`, { method: 'DELETE' }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['agents'] }),
  })
}

export function useSavePrompt() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ name, filename, content }: { name: string; filename: string; content: string }) =>
      apiFetch(`/api/v1/agents/${name}/prompts/${filename}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'text/plain' },
        body: content,
      }),
    onSuccess: (_, vars) => qc.invalidateQueries({ queryKey: ['agent', vars.name] }),
  })
}
```

- [ ] **Step 2: 创建 skills.ts**

```typescript
// frontend/src/api/skills.ts
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { apiFetch } from './client'

export interface SkillSummary {
  name: string
  description: string
  requires: Record<string, unknown>
  dir_name: string
}

export interface SkillFull extends SkillSummary {
  body: string
  raw: string
}

export function useSkills() {
  return useQuery({
    queryKey: ['skills'],
    queryFn: () => apiFetch<SkillSummary[]>('/api/v1/skills'),
  })
}

export function useSkill(name: string | null) {
  return useQuery({
    queryKey: ['skill', name],
    queryFn: () => apiFetch<SkillFull>(`/api/v1/skills/${name}`),
    enabled: !!name,
  })
}

export function useCreateSkill() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (data: { name: string; description?: string; body?: string }) =>
      apiFetch('/api/v1/skills', { method: 'POST', body: JSON.stringify(data) }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['skills'] }),
  })
}

export function useUpdateSkill() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ name, raw }: { name: string; raw: string }) =>
      apiFetch(`/api/v1/skills/${name}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'text/plain' },
        body: raw,
      }),
    onSuccess: (_, vars) => {
      qc.invalidateQueries({ queryKey: ['skill', vars.name] })
      qc.invalidateQueries({ queryKey: ['skills'] })
    },
  })
}

export function useDeleteSkill() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (name: string) =>
      apiFetch(`/api/v1/skills/${name}`, { method: 'DELETE' }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['skills'] }),
  })
}
```

- [ ] **Step 3: TypeScript 检查**

Run: `cd frontend && npx tsc --noEmit`
Expected: no errors

- [ ] **Step 4: Commit**

```bash
git add frontend/src/api/agents.ts frontend/src/api/skills.ts
git commit -m "feat: add Agent and Skill API hooks"
```

---

### Task 6: Agent 管理 Overlay

**Files:**
- Create: `frontend/src/components/overlays/AgentManagementOverlay.tsx`
- Modify: `frontend/src/stores/panel.ts`
- Modify: `frontend/src/components/layout/IconRail.tsx`
- Modify: `frontend/src/components/nodes/NodeEditor.tsx`

- [ ] **Step 1: 注册 Overlay + Icon**

在 `frontend/src/stores/panel.ts`：
- `OverlayId` 添加 `'agents'`

在 `frontend/src/components/layout/IconRail.tsx`：
- Import `Bot` from lucide-react
- 在 OVERLAY_ITEMS 中添加 `{ id: 'agents', icon: Bot, label: 'Agents' }`

在 `frontend/src/components/nodes/NodeEditor.tsx`：
- Import `AgentManagementOverlay`
- 添加 `{activeOverlay === 'agents' && <AgentManagementOverlay />}`

- [ ] **Step 2: 创建 AgentManagementOverlay**

这是一个 master-detail 布局的 overlay，类似已有的 `ApiManagementOverlay`：
- 左侧：Agent 列表 + 新建按钮
- 右侧：Agent 详情（config 编辑 + MD 文件编辑器 + skills 勾选）

核心功能：
- 列出所有 agents（`useAgents`）
- 选中后显示详情（`useAgent(name)`）
- 编辑 display_name、skills 勾选（`useUpdateAgent`）
- 三个 MD 文件的文本编辑器（AGENT.md, SOUL.md, IDENTITY.md）
- 保存 MD 文件（`useSavePrompt`）
- 新建/删除 agent

使用 textarea 作为 MD 编辑器（简单实现，不需要 CodeMirror）。

```tsx
// frontend/src/components/overlays/AgentManagementOverlay.tsx
// 参考 ApiManagementOverlay 的布局模式实现
// 左侧列表 + 右侧详情 + 底部 skills 勾选
```

具体实现参考现有的 `ApiManagementOverlay.tsx`（约 748 行），但更简单：
- 不需要 instance cards
- 用 textarea 替代 key management
- Skills 用 checkbox list（从 `useSkills()` 获取可用列表）

- [ ] **Step 3: TypeScript 检查**

Run: `cd frontend && npx tsc --noEmit`
Expected: no errors

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/overlays/AgentManagementOverlay.tsx frontend/src/stores/panel.ts frontend/src/components/layout/IconRail.tsx frontend/src/components/nodes/NodeEditor.tsx
git commit -m "feat: add Agent management overlay with MD editor"
```

---

### Task 7: 集成测试 + 清理

- [ ] **Step 1: 运行全量后端测试**

Run: `cd backend && uv run pytest --ignore=tests/test_audio_io.py -v`
Expected: All passed

- [ ] **Step 2: 运行前端类型检查**

Run: `cd frontend && npx tsc --noEmit`
Expected: no errors

- [ ] **Step 3: Commit（如有修复）**

```bash
git add -A && git commit -m "chore: Phase B cleanup"
```
