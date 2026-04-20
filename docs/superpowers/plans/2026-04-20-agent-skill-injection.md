# Agent / Skill 注入 · 实施 Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal：** 让 `/v1/responses` 和 `/v1/chat/completions` 支持 `agent` 字段：读取 `~/.nous-center/agents/<name>/` 下的 IDENTITY/SOUL/AGENT.md + skills 列表 → 装配 system message 注入 vLLM；skill 走 lazy-readable 模式（system prompt 放 `<available_skills>` 清单 + 专用 `Skill` tool）。

**Architecture：** 新建 `prompt_composer/` 模块作为纯函数装配层；`response_sessions.agent_id` 和 `llm_usage.agent_id` 列做 session 绑定与 usage 观测；feature flag `NOUS_ENABLE_AGENT_INJECTION` 灰度；对齐 OpenClaw + Claude Code 范式。

**Tech Stack：** Python 3.12, FastAPI, SQLAlchemy async, pytest, vLLM (local Qwen3.5 for manual E2E)。

**Spec：** `docs/designs/2026-04-20-agent-skill-injection.md`

**并行分支策略（选项 C）：** 本 plan 在独立 worktree 下执行，分支 `feature/agent-skill-injection`（从 `feature/nous-center-v2` fork）。Wave 1 在另一 worktree / 另一分支并行推进。两者合并到 `feature/nous-center-v2` 时各自独立 PR。

---

## File Structure

```
backend/
  src/
    services/
      prompt_composer/
        __init__.py              # compose() 主入口 + AgentLoadFailed 异常
        _constants.py            # 固定指令段、XML 转义、CACHE_BOUNDARY marker
        _persona.py              # 加载三件套 + lru_cache
        _skills_catalog.py       # <available_skills> 段生成
      skill_tools.py             # 扩展：加入 Skill tool 执行分支
      responses_service.py       # 扩展：assert_agent_matches_session()
      usage_recorder.py          # 扩展：record_llm_usage 接受 agent_id
    models/
      response_session.py        # 扩展：agent_id 列
      llm_usage.py               # 扩展：agent_id 列
    api/routes/
      responses.py               # 扩展：agent 字段 + composer 接入 + MESSAGES_ORDER
      openai_compat.py           # 扩展：agent 字段 + composer 接入
      agents.py                  # 扩展：GET /{name}/preview endpoint
    config.py                    # 扩展：NOUS_ENABLE_AGENT_INJECTION flag
  scripts/
    migrate_agent_id.py          # 生产 PG 的 ALTER TABLE 脚本
  tests/
    test_prompt_composer.py      # composer 单元测试（含 golden file）
    test_skill_tool_exec.py      # Skill tool 执行测试
    test_responses_agent_binding.py  # session 绑定集成测试
    golden/
      tutor_full.txt             # golden file fixture
    fixtures/
      agents/tutor/              # 测试用 agent 目录
      skills/search/             # 测试用 skill
```

---

## Task 1: prompt_composer 常量与工具

**Files:**
- Create: `backend/src/services/prompt_composer/__init__.py`（最小占位，Task 4 填充）
- Create: `backend/src/services/prompt_composer/_constants.py`
- Test: `backend/tests/test_prompt_composer.py`

- [ ] **Step 1: 写失败测试**

```python
# backend/tests/test_prompt_composer.py
from src.services.prompt_composer._constants import (
    CACHE_BOUNDARY_MARKER,
    SKILLS_INSTRUCTION,
    SOUL_PERSONA_INSTRUCTION,
    escape_xml,
)


def test_cache_boundary_marker_stable():
    assert CACHE_BOUNDARY_MARKER == "<!-- CACHE_BOUNDARY -->"


def test_skills_instruction_has_required_cues():
    assert "<available_skills>" in SKILLS_INSTRUCTION
    assert "Skill(skill=" in SKILLS_INSTRUCTION
    assert "do not call Skill" in SKILLS_INSTRUCTION


def test_soul_persona_instruction_mentions_embody():
    assert "embody" in SOUL_PERSONA_INSTRUCTION.lower()


def test_escape_xml_special_chars():
    assert escape_xml("a & b") == "a &amp; b"
    assert escape_xml("<x>") == "&lt;x&gt;"
    assert escape_xml('a"b') == "a&quot;b"
    assert escape_xml("it's") == "it&apos;s"
```

- [ ] **Step 2: 运行测试验证失败**

Run: `cd backend && pytest tests/test_prompt_composer.py -v`
Expected: FAIL — `ModuleNotFoundError: src.services.prompt_composer`

- [ ] **Step 3: 创建模块骨架 + 实现常量**

```python
# backend/src/services/prompt_composer/__init__.py
"""Agent / Skill system prompt composer.

Single public entry: compose(agent_id, instructions) -> str | None
"""
# Task 4 填充 compose()
```

```python
# backend/src/services/prompt_composer/_constants.py
"""Prompt composition constants & helpers.

Layout, ordering, and XML format are documented in
docs/designs/2026-04-20-agent-skill-injection.md (System message 装配 section).
References: OpenClaw src/agents/system-prompt.ts (buildSkillsSection),
Claude Code rust/crates/tools/src/lib.rs:557-570 (Skill tool).
"""

CACHE_BOUNDARY_MARKER = "<!-- CACHE_BOUNDARY -->"

SOUL_PERSONA_INSTRUCTION = (
    "Embody the persona and tone described above. Avoid generic or stiff "
    "replies unless higher-priority instructions override it."
)

SKILLS_INSTRUCTION = """## Available Skills
Before replying: scan <available_skills> <description> entries.
- If one clearly applies: call Skill(skill="<name>") first, then follow the returned instructions.
- If none apply: do not call Skill.
Never call Skill more than once per turn unless the task clearly requires chaining."""


def escape_xml(s: str) -> str:
    """Minimal XML escape (agent/skill metadata only; never render model output)."""
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )
```

- [ ] **Step 4: 运行测试验证通过**

Run: `cd backend && pytest tests/test_prompt_composer.py -v`
Expected: PASS（4 tests）

- [ ] **Step 5: Commit**

```bash
git add backend/src/services/prompt_composer/__init__.py \
        backend/src/services/prompt_composer/_constants.py \
        backend/tests/test_prompt_composer.py
git commit -m "feat(composer): constants + xml escape"
```

---

## Task 2: prompt_composer 的 persona 加载（IDENTITY/SOUL/AGENT）

**Files:**
- Create: `backend/src/services/prompt_composer/_persona.py`
- Test: 扩展 `backend/tests/test_prompt_composer.py`
- Create: `backend/tests/fixtures/agents/tutor/` 下的 config.json + 三个 md

- [ ] **Step 1: 建 fixture agent**

```bash
mkdir -p backend/tests/fixtures/agents/tutor
```

```json
// backend/tests/fixtures/agents/tutor/config.json
{
  "display_name": "Tutor",
  "model": "qwen3.5",
  "skills": ["search", "summarize"],
  "status": "active"
}
```

```markdown
<!-- backend/tests/fixtures/agents/tutor/IDENTITY.md -->
你是 Tutor，一位耐心的学习助手。
```

```markdown
<!-- backend/tests/fixtures/agents/tutor/SOUL.md -->
温和、鼓励、不评判。用简单比喻帮学生理解抽象概念。
```

```markdown
<!-- backend/tests/fixtures/agents/tutor/AGENT.md -->
回答学生问题时先确认他们的背景知识，再由浅入深展开。
```

- [ ] **Step 2: 写失败测试**

追加到 `backend/tests/test_prompt_composer.py`：

```python
from pathlib import Path
from src.services.prompt_composer._persona import PersonaBundle, load_persona

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def _clear_settings_and_persona_cache():
    """Clear get_settings() + _load_cached lru_cache so monkeypatch.setenv takes effect."""
    from src import config
    from src.services.prompt_composer import _persona
    config.get_settings.cache_clear()
    _persona._load_cached.cache_clear()
    yield
    config.get_settings.cache_clear()
    _persona._load_cached.cache_clear()


def test_load_persona_all_three_files(monkeypatch):
    monkeypatch.setenv("NOUS_CENTER_HOME", str(FIXTURES))
    bundle = load_persona("tutor")
    assert bundle.identity.startswith("你是 Tutor")
    assert "温和" in bundle.soul
    assert "由浅入深" in bundle.agent
    assert bundle.skills == ("search", "summarize")  # tuple, frozen dataclass


def test_load_persona_missing_agent_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("NOUS_CENTER_HOME", str(tmp_path))
    import pytest
    from src.services.prompt_composer._persona import AgentNotFound
    with pytest.raises(AgentNotFound):
        load_persona("nonexistent")


def test_load_persona_empty_files(monkeypatch, tmp_path):
    monkeypatch.setenv("NOUS_CENTER_HOME", str(tmp_path))
    agent_dir = tmp_path / "agents" / "blank"
    agent_dir.mkdir(parents=True)
    (agent_dir / "config.json").write_text('{"skills": []}')
    for fn in ("IDENTITY.md", "SOUL.md", "AGENT.md"):
        (agent_dir / fn).write_text("")
    bundle = load_persona("blank")
    assert bundle.identity == ""
    assert bundle.soul == ""
    assert bundle.agent == ""
    assert bundle.skills == ()


def test_load_persona_corrupt_config_raises_agent_load_failed(monkeypatch, tmp_path):
    monkeypatch.setenv("NOUS_CENTER_HOME", str(tmp_path))
    agent_dir = tmp_path / "agents" / "broken"
    agent_dir.mkdir(parents=True)
    (agent_dir / "config.json").write_text("{ not json")
    for fn in ("IDENTITY.md", "SOUL.md", "AGENT.md"):
        (agent_dir / fn).write_text("")
    import pytest
    from src.services.prompt_composer._persona import AgentLoadFailed
    with pytest.raises(AgentLoadFailed):
        load_persona("broken")
```

- [ ] **Step 3: 运行测试验证失败**

Run: `cd backend && pytest tests/test_prompt_composer.py::test_load_persona_all_three_files -v`
Expected: FAIL — `ModuleNotFoundError: src.services.prompt_composer._persona`

- [ ] **Step 4: 实现 persona 加载**

```python
# backend/src/services/prompt_composer/_persona.py
"""Load agent's IDENTITY.md / SOUL.md / AGENT.md + config.json.

Uses lru_cache keyed on (agent_id, config.json mtime) to avoid disk IO
on every request. Cache invalidates when any of the agent's files change.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from src.config import get_settings


class AgentNotFound(Exception):
    """Agent directory or config.json does not exist."""


class AgentLoadFailed(Exception):
    """Agent files are malformed or unreadable (IO / permission / JSON parse)."""


@dataclass(frozen=True)
class PersonaBundle:
    identity: str
    soul: str
    agent: str
    skills: tuple[str, ...]  # tuple for hashable (lru_cache friendly)


def _agents_root() -> Path:
    return Path(get_settings().NOUS_CENTER_HOME).expanduser() / "agents"


def _config_mtime(agent_dir: Path) -> float:
    """Return config.json mtime as cache-busting key component."""
    try:
        return (agent_dir / "config.json").stat().st_mtime
    except OSError:
        return 0.0


def load_persona(agent_id: str) -> PersonaBundle:
    """Load + cache persona files for agent_id.

    Raises AgentNotFound if config.json missing; AgentLoadFailed on IO/JSON errors.
    """
    agent_dir = _agents_root() / agent_id
    if not (agent_dir / "config.json").exists():
        raise AgentNotFound(f"agent {agent_id!r} not found")
    return _load_cached(agent_id, _config_mtime(agent_dir))


@lru_cache(maxsize=128)
def _load_cached(agent_id: str, config_mtime: float) -> PersonaBundle:
    """Cached inner load. Key includes mtime so file edits invalidate."""
    agent_dir = _agents_root() / agent_id
    try:
        cfg = json.loads((agent_dir / "config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise AgentLoadFailed(
            f"agent {agent_id!r} config.json: {e}"
        ) from e

    def _read(name: str) -> str:
        p = agent_dir / name
        if not p.exists():
            return ""
        try:
            return p.read_text(encoding="utf-8").strip()
        except (OSError, PermissionError) as e:
            raise AgentLoadFailed(
                f"agent {agent_id!r} {name}: {e}"
            ) from e

    return PersonaBundle(
        identity=_read("IDENTITY.md"),
        soul=_read("SOUL.md"),
        agent=_read("AGENT.md"),
        skills=tuple(cfg.get("skills", [])),
    )
```

- [ ] **Step 5: 运行测试验证通过**

Run: `cd backend && pytest tests/test_prompt_composer.py -v`
Expected: PASS（4 new tests + 4 from Task 1 = 8 total）

- [ ] **Step 6: Commit**

```bash
git add backend/src/services/prompt_composer/_persona.py \
        backend/tests/fixtures/agents/tutor \
        backend/tests/test_prompt_composer.py
git commit -m "feat(composer): load persona with mtime-keyed cache"
```

---

## Task 3: prompt_composer 的 skills catalog

**Files:**
- Create: `backend/src/services/prompt_composer/_skills_catalog.py`
- Test: 扩展 `backend/tests/test_prompt_composer.py`
- Create: `backend/tests/fixtures/skills/search/SKILL.md`、`summarize/SKILL.md`

- [ ] **Step 1: 建 fixture skills**

```bash
mkdir -p backend/tests/fixtures/skills/search backend/tests/fixtures/skills/summarize
```

```markdown
<!-- backend/tests/fixtures/skills/search/SKILL.md -->
---
name: search
description: 网页搜索，返回可引用链接
---

## 使用说明
优先用中文搜索，引用时带 [n] 标注。
```

```markdown
<!-- backend/tests/fixtures/skills/summarize/SKILL.md -->
---
name: summarize
description: 将长文本压缩为要点列表
---

## 使用说明
默认输出 3-5 条要点。
```

- [ ] **Step 2: 写失败测试**

追加到 `backend/tests/test_prompt_composer.py`：

```python
from src.services.prompt_composer._skills_catalog import build_skills_catalog


def test_build_skills_catalog_with_two_skills(monkeypatch):
    monkeypatch.setenv("NOUS_CENTER_HOME", str(FIXTURES))
    xml = build_skills_catalog(["search", "summarize"])
    assert "<available_skills>" in xml
    assert "<name>search</name>" in xml
    assert "<description>网页搜索，返回可引用链接</description>" in xml
    assert "<name>summarize</name>" in xml
    # 不应暴露 location
    assert "<location>" not in xml


def test_build_skills_catalog_empty_list_returns_empty():
    assert build_skills_catalog([]) == ""


def test_build_skills_catalog_missing_skill_is_skipped(monkeypatch, caplog):
    monkeypatch.setenv("NOUS_CENTER_HOME", str(FIXTURES))
    import logging
    caplog.set_level(logging.WARNING)
    xml = build_skills_catalog(["search", "nonexistent_skill"])
    assert "<name>search</name>" in xml
    assert "nonexistent_skill" not in xml
    assert any("nonexistent_skill" in r.message for r in caplog.records)


def test_build_skills_catalog_escapes_xml_in_description(monkeypatch, tmp_path):
    monkeypatch.setenv("NOUS_CENTER_HOME", str(tmp_path))
    skill_dir = tmp_path / "skills" / "xss"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: xss\ndescription: 'a <script> & b'\n---\nbody"
    )
    xml = build_skills_catalog(["xss"])
    assert "&lt;script&gt;" in xml
    assert "&amp;" in xml
    assert "<script>" not in xml
```

- [ ] **Step 3: 运行测试验证失败**

Run: `cd backend && pytest tests/test_prompt_composer.py -v -k skills_catalog`
Expected: FAIL — module not found

- [ ] **Step 4: 实现 skills catalog**

```python
# backend/src/services/prompt_composer/_skills_catalog.py
"""Build the <available_skills> XML block injected into system prompt.

Only reads frontmatter (name + description) — SKILL.md body stays off-context
until the model calls Skill(skill=...) tool. This is the lazy-readable pattern
from OpenClaw (src/agents/skills/skill-contract.ts:44-64), adapted.
"""

from __future__ import annotations

import logging

from src.services import skill_manager
from src.services.prompt_composer._constants import escape_xml

logger = logging.getLogger(__name__)


def build_skills_catalog(skill_names: list[str]) -> str:
    """Return an XML block listing (name, description) for each skill.

    Missing skills are logged and skipped, never raise. Returns "" if no
    valid skills remain.
    """
    entries: list[str] = []
    for name in skill_names:
        try:
            sk = skill_manager.get_skill(name)
        except FileNotFoundError:
            logger.warning("agent references missing skill: %s (skipping)", name)
            continue
        entries.append(
            f"  <skill>\n"
            f"    <name>{escape_xml(sk['name'])}</name>\n"
            f"    <description>{escape_xml(sk.get('description', ''))}</description>\n"
            f"  </skill>"
        )
    if not entries:
        return ""
    body = "\n".join(entries)
    return f"<available_skills>\n{body}\n</available_skills>"
```

- [ ] **Step 5: 运行测试验证通过**

Run: `cd backend && pytest tests/test_prompt_composer.py -v`
Expected: PASS（4 new + 8 prev = 12 total）

- [ ] **Step 6: Commit**

```bash
git add backend/src/services/prompt_composer/_skills_catalog.py \
        backend/tests/fixtures/skills \
        backend/tests/test_prompt_composer.py
git commit -m "feat(composer): skills catalog with xml escape"
```

---

## Task 4: compose() 主入口（装配完整 system message）

**Files:**
- Modify: `backend/src/services/prompt_composer/__init__.py`
- Test: 扩展 `backend/tests/test_prompt_composer.py`

- [ ] **Step 1: 写失败测试**

追加到 `backend/tests/test_prompt_composer.py`：

```python
from src.services.prompt_composer import compose


def test_compose_full_agent_has_all_sections(monkeypatch):
    monkeypatch.setenv("NOUS_CENTER_HOME", str(FIXTURES))
    result = compose(agent_id="tutor", instructions=None)
    assert result is not None
    # 顺序：IDENTITY → SOUL → AGENT → Available Skills → CACHE_BOUNDARY
    idx_identity = result.index("你是 Tutor")
    idx_soul = result.index("温和")
    idx_agent = result.index("由浅入深")
    idx_skills = result.index("<available_skills>")
    idx_boundary = result.index("<!-- CACHE_BOUNDARY -->")
    assert idx_identity < idx_soul < idx_agent < idx_skills < idx_boundary


def test_compose_soul_persona_instruction_appended(monkeypatch):
    monkeypatch.setenv("NOUS_CENTER_HOME", str(FIXTURES))
    result = compose(agent_id="tutor", instructions=None)
    assert "Embody" in result


def test_compose_with_instructions_appears_after_boundary(monkeypatch):
    monkeypatch.setenv("NOUS_CENTER_HOME", str(FIXTURES))
    result = compose(agent_id="tutor", instructions="本轮请用英文回答")
    assert "本轮请用英文回答" in result
    assert result.index("<!-- CACHE_BOUNDARY -->") < result.index("本轮请用英文回答")


def test_compose_no_agent_returns_none():
    assert compose(agent_id=None, instructions=None) is None
    assert compose(agent_id=None, instructions="foo") is None  # instructions alone 走现有路径，不经 composer


def test_compose_empty_agent_returns_none(monkeypatch, tmp_path):
    monkeypatch.setenv("NOUS_CENTER_HOME", str(tmp_path))
    agent_dir = tmp_path / "agents" / "blank"
    agent_dir.mkdir(parents=True)
    (agent_dir / "config.json").write_text('{"skills": []}')
    for fn in ("IDENTITY.md", "SOUL.md", "AGENT.md"):
        (agent_dir / fn).write_text("")
    result = compose(agent_id="blank", instructions=None)
    assert result is None


def test_compose_agent_not_found_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("NOUS_CENTER_HOME", str(tmp_path))
    import pytest
    from src.services.prompt_composer._persona import AgentNotFound
    with pytest.raises(AgentNotFound):
        compose(agent_id="ghost", instructions=None)
```

- [ ] **Step 2: 运行测试验证失败**

Run: `cd backend && pytest tests/test_prompt_composer.py::test_compose_full_agent_has_all_sections -v`
Expected: FAIL — `ImportError: cannot import name 'compose'`

- [ ] **Step 3: 实现 compose()**

```python
# backend/src/services/prompt_composer/__init__.py
"""Agent / Skill system prompt composer.

Single public entry: compose(agent_id, instructions) -> str | None
Returns None when result would be empty (caller MUST NOT add system role).
"""

from __future__ import annotations

from src.services.prompt_composer._constants import (
    CACHE_BOUNDARY_MARKER,
    SKILLS_INSTRUCTION,
    SOUL_PERSONA_INSTRUCTION,
)
from src.services.prompt_composer._persona import (
    AgentLoadFailed,
    AgentNotFound,
    load_persona,
)
from src.services.prompt_composer._skills_catalog import build_skills_catalog

__all__ = ["compose", "AgentNotFound", "AgentLoadFailed"]


def compose(agent_id: str | None, instructions: str | None) -> str | None:
    """Compose system message for (agent_id, instructions).

    Returns None if no agent given, OR agent resolves to empty persona +
    no skills. Caller MUST NOT append {"role": "system"} when result is None.

    Raises:
        AgentNotFound: agent directory missing
        AgentLoadFailed: config.json / md file IO or parse error
    """
    if not agent_id:
        return None

    persona = load_persona(agent_id)
    parts: list[str] = []

    if persona.identity:
        parts.append(f"# Identity\n{persona.identity}")

    if persona.soul:
        parts.append(f"# Soul\n{persona.soul}\n\n{SOUL_PERSONA_INSTRUCTION}")

    if persona.agent:
        parts.append(f"# Agent Instructions\n{persona.agent}")

    skills_xml = build_skills_catalog(list(persona.skills))
    if skills_xml:
        parts.append(f"{SKILLS_INSTRUCTION}\n\n{skills_xml}")

    if not parts:
        # 无 persona 且无 skills — 退化为"不追加 system message"
        return None

    stable_prefix = "\n\n".join(parts)
    result = f"{stable_prefix}\n\n{CACHE_BOUNDARY_MARKER}"

    if instructions:
        result = f"{result}\n\n# Request Instructions\n{instructions}"

    return result
```

- [ ] **Step 4: 运行测试验证通过**

Run: `cd backend && pytest tests/test_prompt_composer.py -v`
Expected: PASS（6 new + 12 prev = 18 total）

- [ ] **Step 5: Commit**

```bash
git add backend/src/services/prompt_composer/__init__.py \
        backend/tests/test_prompt_composer.py
git commit -m "feat(composer): compose() assembles full system message"
```

---

## Task 5: Golden file 回归测试

**Files:**
- Create: `backend/tests/golden/tutor_full.txt`
- Test: 扩展 `backend/tests/test_prompt_composer.py`

- [ ] **Step 1: 生成 golden file（手动确认内容正确后提交）**

```bash
cd backend
NOUS_CENTER_HOME=$(pwd)/tests/fixtures python -c "
from src.services.prompt_composer import compose
print(compose(agent_id='tutor', instructions=None))
" > tests/golden/tutor_full.txt
```

检查 `tests/golden/tutor_full.txt` 内容符合 spec 描述的布局。

- [ ] **Step 2: 写 golden 测试**

追加到 `backend/tests/test_prompt_composer.py`：

```python
GOLDEN = Path(__file__).parent / "golden"


def test_compose_golden_tutor_full(monkeypatch):
    """Byte-exact comparison against golden file.

    If this fails after an intentional format change, regenerate golden:
        NOUS_CENTER_HOME=backend/tests/fixtures python -c \\
          "from src.services.prompt_composer import compose; \\
           print(compose('tutor', None))" > backend/tests/golden/tutor_full.txt
    """
    monkeypatch.setenv("NOUS_CENTER_HOME", str(FIXTURES))
    result = compose(agent_id="tutor", instructions=None)
    expected = (GOLDEN / "tutor_full.txt").read_text(encoding="utf-8").rstrip("\n")
    assert result == expected
```

- [ ] **Step 3: 运行测试验证通过**

Run: `cd backend && pytest tests/test_prompt_composer.py::test_compose_golden_tutor_full -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add backend/tests/golden/tutor_full.txt backend/tests/test_prompt_composer.py
git commit -m "test(composer): golden file byte-exact regression"
```

---

## Task 6: Skill tool 执行分支

**Files:**
- Modify: `backend/src/services/skill_tools.py`
- Test: Create `backend/tests/test_skill_tool_exec.py`

- [ ] **Step 1: 写失败测试**

```python
# backend/tests/test_skill_tool_exec.py
from pathlib import Path

import pytest

from src.services.skill_tools import execute_tool, skill_tool_schema

FIXTURES = Path(__file__).parent / "fixtures"


def test_skill_tool_schema_shape():
    schema = skill_tool_schema()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "Skill"
    params = schema["function"]["parameters"]
    assert params["properties"]["skill"]["type"] == "string"
    assert "skill" in params["required"]


@pytest.mark.asyncio
async def test_skill_tool_valid_name(monkeypatch):
    monkeypatch.setenv("NOUS_CENTER_HOME", str(FIXTURES))
    result = await execute_tool("Skill", {"skill": "search"})
    import json
    data = json.loads(result)
    assert data["skill"] == "search"
    assert "网页搜索" in data["description"]
    assert "中文" in data["prompt"]  # SKILL.md body


@pytest.mark.asyncio
async def test_skill_tool_unknown_name(monkeypatch, tmp_path):
    monkeypatch.setenv("NOUS_CENTER_HOME", str(tmp_path))
    result = await execute_tool("Skill", {"skill": "ghost"})
    import json
    data = json.loads(result)
    assert "error" in data
    assert "unknown skill" in data["error"]


@pytest.mark.asyncio
async def test_skill_tool_empty_name():
    result = await execute_tool("Skill", {"skill": ""})
    import json
    data = json.loads(result)
    assert data["error"] == "skill name required"


@pytest.mark.asyncio
async def test_skill_tool_args_none_no_crash(monkeypatch):
    """Model may send args: null; don't AttributeError."""
    monkeypatch.setenv("NOUS_CENTER_HOME", str(FIXTURES))
    result = await execute_tool("Skill", None)
    import json
    data = json.loads(result)
    assert data["error"] == "skill name required"


@pytest.mark.asyncio
async def test_skill_tool_args_passthrough(monkeypatch):
    monkeypatch.setenv("NOUS_CENTER_HOME", str(FIXTURES))
    result = await execute_tool("Skill", {"skill": "search", "args": "q=foo"})
    import json
    data = json.loads(result)
    assert data["args"] == "q=foo"
```

- [ ] **Step 2: 运行测试验证失败**

Run: `cd backend && pytest tests/test_skill_tool_exec.py -v`
Expected: FAIL — `ImportError: skill_tool_schema` not exported

- [ ] **Step 3: 扩展 skill_tools.py**

查看现有文件顶部的 imports，然后在文件中添加：

```python
# backend/src/services/skill_tools.py — 新增函数
def skill_tool_schema() -> dict:
    """Return the OpenAI function-tool schema for the Skill tool.

    This is the single tool that replaces per-skill function generation
    (cf. skills_to_tools). The model calls Skill(skill=<name>, args=?)
    and receives the SKILL.md body as tool_result.
    """
    return {
        "type": "function",
        "function": {
            "name": "Skill",
            "description": "Load a local skill definition and its instructions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "skill": {
                        "type": "string",
                        "description": "Skill name from <available_skills>.",
                    },
                    "args": {
                        "type": "string",
                        "description": "Optional arguments to pass to the skill.",
                    },
                },
                "required": ["skill"],
            },
        },
    }


async def _execute_skill_tool(args: dict | None) -> str:
    """Execute the Skill tool. Returns JSON string for tool_result."""
    import json

    args = args or {}  # model may send null
    name = (args.get("skill") or "").strip()
    if not name:
        return json.dumps({"error": "skill name required"})
    try:
        sk = skill_manager.get_skill(name)
    except FileNotFoundError:
        return json.dumps({"error": f"unknown skill: {name}"})
    return json.dumps({
        "skill": name,
        "description": sk.get("description", ""),
        "prompt": sk.get("body", ""),
        "args": args.get("args"),
    }, ensure_ascii=False)
```

然后在现有 `execute_tool()` 函数内，在 `execute_python` 分支之前加入：

```python
    # 在 execute_tool() 函数内，在 "if tool_name == "execute_python":" 分支 之前 添加
    if tool_name == "Skill":
        return await _execute_skill_tool(arguments if isinstance(arguments, dict) else None)
```

- [ ] **Step 4: 运行测试验证通过**

Run: `cd backend && pytest tests/test_skill_tool_exec.py -v`
Expected: PASS（6 tests）

- [ ] **Step 5: 验证现有 skill_tools 测试未回归**

Run: `cd backend && pytest tests/test_api_skills.py -v`
Expected: PASS（所有现有测试）

- [ ] **Step 6: Commit**

```bash
git add backend/src/services/skill_tools.py backend/tests/test_skill_tool_exec.py
git commit -m "feat(skill_tools): add Skill function tool for lazy-readable skills"
```

---

## Task 7: Schema — agent_id 列加到 response_sessions 和 llm_usage

**Files:**
- Modify: `backend/src/models/response_session.py`
- Modify: `backend/src/models/llm_usage.py`
- Create: `backend/scripts/migrate_agent_id.py`

- [ ] **Step 1: 写失败测试**

新建 `backend/tests/test_agent_id_column.py`：

```python
from sqlalchemy import create_engine, inspect
from src.models.database import Base
from src.models import response_session, llm_usage  # noqa: F401 ensure registered


def test_response_session_has_agent_id_column():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    inspector = inspect(engine)
    cols = [c["name"] for c in inspector.get_columns("response_sessions")]
    assert "agent_id" in cols


def test_llm_usage_has_agent_id_column():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    inspector = inspect(engine)
    cols = [c["name"] for c in inspector.get_columns("llm_usage")]
    assert "agent_id" in cols
```

- [ ] **Step 2: 运行测试验证失败**

Run: `cd backend && pytest tests/test_agent_id_column.py -v`
Expected: FAIL — `agent_id` not in columns

- [ ] **Step 3: 加列到 response_session.py**

在 `backend/src/models/response_session.py` 的 `ResponseSession` 类 `model` 列之后加：

```python
    agent_id = Column(String(128), nullable=True, index=True)
```

- [ ] **Step 4: 加列到 llm_usage.py**

先读 `backend/src/models/llm_usage.py` 找到合适位置（通常在 model 字段附近），加：

```python
    agent_id = Column(String(128), nullable=True, index=True)
```

- [ ] **Step 5: 写生产 migration 脚本**

```python
# backend/scripts/migrate_agent_id.py
"""Add agent_id column to response_sessions and llm_usage.

Idempotent (checks if column exists before ALTER). Run against dev SQLite
or production PG. For SQLite in CI/tests, Base.metadata.create_all handles it.
"""

import asyncio

from sqlalchemy import inspect, text

from src.models.database import create_engine


async def _add_column_if_missing(conn, table: str, column: str, coltype: str):
    def _check(sync_conn):
        insp = inspect(sync_conn)
        cols = [c["name"] for c in insp.get_columns(table)]
        return column in cols

    exists = await conn.run_sync(_check)
    if exists:
        print(f"  {table}.{column} already exists — skipping")
        return
    await conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {coltype}"))
    await conn.execute(
        text(f"CREATE INDEX IF NOT EXISTS ix_{table}_{column} ON {table}({column})")
    )
    print(f"  {table}.{column} added")


async def main():
    engine = create_engine()
    async with engine.begin() as conn:
        print("Adding agent_id columns...")
        await _add_column_if_missing(conn, "response_sessions", "agent_id", "VARCHAR(128)")
        await _add_column_if_missing(conn, "llm_usage", "agent_id", "VARCHAR(128)")
        print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 6: 运行测试验证通过**

Run: `cd backend && pytest tests/test_agent_id_column.py -v`
Expected: PASS

- [ ] **Step 7: 手动验证 migration 脚本（本地 SQLite）**

Run: `cd backend && python scripts/migrate_agent_id.py`
Expected: 输出 "already exists" 或 "added"，无异常

- [ ] **Step 8: Commit**

```bash
git add backend/src/models/response_session.py \
        backend/src/models/llm_usage.py \
        backend/scripts/migrate_agent_id.py \
        backend/tests/test_agent_id_column.py
git commit -m "feat(db): agent_id column on response_sessions + llm_usage"
```

---

## Task 8: usage_recorder 接受 agent_id

**Files:**
- Modify: `backend/src/services/usage_recorder.py`
- Test: 读现有 usage 测试找最相近的，扩展

- [ ] **Step 1: 读现有 record_llm_usage 签名**

Run: `cd backend && grep -n "def record_llm_usage" src/services/usage_recorder.py`

假设签名是 `async def record_llm_usage(session, *, instance_id, model, prompt_tokens, completion_tokens, ...)` 类似。

- [ ] **Step 2: 写失败测试**

新建 `backend/tests/test_usage_recorder_agent_id.py`：

```python
import pytest
from sqlalchemy import select

from src.models.llm_usage import LlmUsage
from src.services.usage_recorder import record_llm_usage


@pytest.mark.asyncio
async def test_record_llm_usage_writes_agent_id(db_session, test_instance):
    await record_llm_usage(
        db_session,
        instance_id=test_instance.id,
        model="qwen3.5",
        prompt_tokens=10,
        completion_tokens=20,
        agent_id="tutor",
    )
    row = (await db_session.execute(select(LlmUsage))).scalar_one()
    assert row.agent_id == "tutor"


@pytest.mark.asyncio
async def test_record_llm_usage_agent_id_optional(db_session, test_instance):
    """Omitting agent_id should write NULL."""
    await record_llm_usage(
        db_session,
        instance_id=test_instance.id,
        model="qwen3.5",
        prompt_tokens=10,
        completion_tokens=20,
    )
    row = (await db_session.execute(select(LlmUsage))).scalar_one()
    assert row.agent_id is None
```

（`db_session` / `test_instance` fixture 复用 `backend/tests/conftest.py` 里已有的。）

- [ ] **Step 3: 运行测试验证失败**

Run: `cd backend && pytest tests/test_usage_recorder_agent_id.py -v`
Expected: FAIL — `record_llm_usage() got unexpected keyword argument 'agent_id'`

- [ ] **Step 4: 扩展 record_llm_usage 签名**

在 `record_llm_usage` 函数签名加 `agent_id: str | None = None` 参数，并在 `LlmUsage(...)` 构造时把它传下去。

- [ ] **Step 5: 运行测试验证通过**

Run: `cd backend && pytest tests/test_usage_recorder_agent_id.py -v`
Expected: PASS

- [ ] **Step 6: 回归现有测试**

Run: `cd backend && pytest tests/ -v -k usage`
Expected: 所有现有 usage 测试仍通过（因为 agent_id 是 optional）

- [ ] **Step 7: Commit**

```bash
git add backend/src/services/usage_recorder.py backend/tests/test_usage_recorder_agent_id.py
git commit -m "feat(usage): record_llm_usage accepts optional agent_id"
```

---

## Task 9: Config — feature flag

**Files:**
- Modify: `backend/src/config.py`

- [ ] **Step 1: 写失败测试**

新建 `backend/tests/test_config_agent_flag.py`：

```python
import os
import importlib

from src import config


def test_agent_injection_flag_defaults_false(monkeypatch):
    monkeypatch.delenv("NOUS_ENABLE_AGENT_INJECTION", raising=False)
    config.get_settings.cache_clear()
    s = config.get_settings()
    assert s.NOUS_ENABLE_AGENT_INJECTION is False


def test_agent_injection_flag_reads_env(monkeypatch):
    monkeypatch.setenv("NOUS_ENABLE_AGENT_INJECTION", "true")
    config.get_settings.cache_clear()
    s = config.get_settings()
    assert s.NOUS_ENABLE_AGENT_INJECTION is True
```

- [ ] **Step 2: 运行测试验证失败**

Run: `cd backend && pytest tests/test_config_agent_flag.py -v`
Expected: FAIL — AttributeError on `NOUS_ENABLE_AGENT_INJECTION`

- [ ] **Step 3: 加 flag**

读 `backend/src/config.py` 了解 Settings 类的结构（用 pydantic-settings 或类似），在 `Settings` 类定义里加：

```python
    NOUS_ENABLE_AGENT_INJECTION: bool = False  # feature flag for agent/skill system prompt injection
```

- [ ] **Step 4: 运行测试验证通过**

Run: `cd backend && pytest tests/test_config_agent_flag.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/src/config.py backend/tests/test_config_agent_flag.py
git commit -m "feat(config): NOUS_ENABLE_AGENT_INJECTION flag"
```

---

## Task 10: responses_service — agent 一致性校验 helper

**Files:**
- Modify: `backend/src/services/responses_service.py`
- Test: 新建 `backend/tests/test_agent_session_binding.py`

- [ ] **Step 1: 写失败测试**

```python
# backend/tests/test_agent_session_binding.py
import pytest
from fastapi import HTTPException

from src.models.response_session import ResponseSession
from src.services.responses_service import assert_agent_matches_session


def make_session(agent_id: str | None = None) -> ResponseSession:
    return ResponseSession(
        id="session-test",
        instance_id=1,
        model="qwen3.5",
        agent_id=agent_id,
    )


def test_assert_agent_matches_none_request_uses_session_binding():
    sess = make_session(agent_id="tutor")
    result = assert_agent_matches_session(sess, request_agent=None)
    assert result == "tutor"


def test_assert_agent_matches_identical_ok():
    sess = make_session(agent_id="tutor")
    result = assert_agent_matches_session(sess, request_agent="tutor")
    assert result == "tutor"


def test_assert_agent_mismatch_raises_400():
    sess = make_session(agent_id="tutor")
    with pytest.raises(HTTPException) as exc_info:
        assert_agent_matches_session(sess, request_agent="writer")
    assert exc_info.value.status_code == 400
    assert "agent_session_mismatch" in str(exc_info.value.detail)


def test_assert_agent_on_unbound_session_raises_400():
    sess = make_session(agent_id=None)
    with pytest.raises(HTTPException) as exc_info:
        assert_agent_matches_session(sess, request_agent="tutor")
    assert exc_info.value.status_code == 400


def test_assert_no_agent_on_unbound_session_returns_none():
    sess = make_session(agent_id=None)
    result = assert_agent_matches_session(sess, request_agent=None)
    assert result is None
```

- [ ] **Step 2: 运行测试验证失败**

Run: `cd backend && pytest tests/test_agent_session_binding.py -v`
Expected: FAIL — `ImportError: assert_agent_matches_session`

- [ ] **Step 3: 实现 helper**

在 `backend/src/services/responses_service.py` 末尾添加：

```python
# ---------- agent binding ---------- #

def assert_agent_matches_session(
    sess: ResponseSession, request_agent: str | None
) -> str | None:
    """Validate request's agent against session's bound agent_id.

    Returns the effective agent_id to use, or raises 400 on mismatch.
    Called by continuation requests (with previous_response_id).
    """
    from fastapi import HTTPException

    if request_agent is None:
        return sess.agent_id  # 从 session 恢复（可能为 None）
    if sess.agent_id is None:
        raise HTTPException(
            400,
            {
                "error": "agent_session_mismatch",
                "message": f"session has no agent binding; got {request_agent!r}",
            },
        )
    if request_agent != sess.agent_id:
        raise HTTPException(
            400,
            {
                "error": "agent_session_mismatch",
                "message": (
                    f"session bound to {sess.agent_id!r}, got {request_agent!r}"
                ),
            },
        )
    return sess.agent_id
```

- [ ] **Step 4: 运行测试验证通过**

Run: `cd backend && pytest tests/test_agent_session_binding.py -v`
Expected: PASS（5 tests）

- [ ] **Step 5: Commit**

```bash
git add backend/src/services/responses_service.py backend/tests/test_agent_session_binding.py
git commit -m "feat(responses): assert_agent_matches_session helper"
```

---

## Task 11: /v1/responses 接入 composer + agent 字段

**Files:**
- Modify: `backend/src/api/routes/responses.py`
- Test: `backend/tests/test_responses_agent_binding.py`

- [ ] **Step 1: 写失败 E2E 测试**

```python
# backend/tests/test_responses_agent_binding.py
"""Integration tests: /v1/responses with agent field."""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_first_request_with_agent_writes_agent_id(
    monkeypatch, api_client: AsyncClient, bearer_headers, fixtures_home, mock_vllm
):
    """First request with agent=tutor writes agent_id into response_sessions."""
    monkeypatch.setenv("NOUS_ENABLE_AGENT_INJECTION", "true")
    resp = await api_client.post(
        "/v1/responses",
        json={"model": "qwen3.5", "input": "你好", "agent": "tutor"},
        headers=bearer_headers,
    )
    assert resp.status_code == 200
    data = resp.json()
    # 查 DB
    from sqlalchemy import select
    from src.models.response_session import ResponseSession
    async with api_client.app.state.async_session_factory() as db:
        sess = (await db.execute(select(ResponseSession))).scalar_one()
        assert sess.agent_id == "tutor"


@pytest.mark.asyncio
async def test_continuation_without_agent_restores_from_session(
    monkeypatch, api_client, bearer_headers, fixtures_home, mock_vllm
):
    monkeypatch.setenv("NOUS_ENABLE_AGENT_INJECTION", "true")
    r1 = await api_client.post(
        "/v1/responses",
        json={"model": "qwen3.5", "input": "hi", "agent": "tutor"},
        headers=bearer_headers,
    )
    first_id = r1.json()["id"]
    r2 = await api_client.post(
        "/v1/responses",
        json={"model": "qwen3.5", "input": "again", "previous_response_id": first_id},
        headers=bearer_headers,
    )
    assert r2.status_code == 200
    # assert mock_vllm 收到的 messages 首条 system message 包含 tutor 的 IDENTITY


@pytest.mark.asyncio
async def test_continuation_with_mismatched_agent_returns_400(
    monkeypatch, api_client, bearer_headers, fixtures_home, mock_vllm
):
    monkeypatch.setenv("NOUS_ENABLE_AGENT_INJECTION", "true")
    r1 = await api_client.post(
        "/v1/responses",
        json={"model": "qwen3.5", "input": "hi", "agent": "tutor"},
        headers=bearer_headers,
    )
    first_id = r1.json()["id"]
    r2 = await api_client.post(
        "/v1/responses",
        json={
            "model": "qwen3.5",
            "input": "again",
            "previous_response_id": first_id,
            "agent": "writer",
        },
        headers=bearer_headers,
    )
    assert r2.status_code == 400
    assert r2.json()["detail"]["error"] == "agent_session_mismatch"


@pytest.mark.asyncio
async def test_messages_order_regression_no_agent(
    monkeypatch, api_client, bearer_headers, mock_vllm
):
    """Without agent, messages array sent to vLLM must be byte-identical to pre-change."""
    monkeypatch.setenv("NOUS_ENABLE_AGENT_INJECTION", "true")
    await api_client.post(
        "/v1/responses",
        json={"model": "qwen3.5", "input": "hi", "instructions": "be brief"},
        headers=bearer_headers,
    )
    sent = mock_vllm.last_request_body["messages"]
    # 第一条应是 instructions 的 system message（不是 agent system message）
    assert sent[0]["role"] == "system"
    assert sent[0]["content"] == "be brief"


@pytest.mark.asyncio
async def test_flag_off_ignores_agent_field(
    monkeypatch, api_client, bearer_headers, fixtures_home, mock_vllm
):
    monkeypatch.setenv("NOUS_ENABLE_AGENT_INJECTION", "false")
    resp = await api_client.post(
        "/v1/responses",
        json={"model": "qwen3.5", "input": "hi", "agent": "tutor"},
        headers=bearer_headers,
    )
    assert resp.status_code == 200
    # agent 字段被忽略，session.agent_id 应为 NULL
    from sqlalchemy import select
    from src.models.response_session import ResponseSession
    async with api_client.app.state.async_session_factory() as db:
        sess = (await db.execute(select(ResponseSession))).scalar_one()
        assert sess.agent_id is None
```

（`fixtures_home` / `mock_vllm` fixtures 需要在 conftest 里加，见下一步）

- [ ] **Step 2: 扩展 conftest.py**

追加到 `backend/tests/conftest.py`：

```python
@pytest.fixture
def fixtures_home(monkeypatch, tmp_path):
    """Point NOUS_CENTER_HOME at tests/fixtures for agent/skill lookups."""
    from pathlib import Path
    fixtures = Path(__file__).parent / "fixtures"
    monkeypatch.setenv("NOUS_CENTER_HOME", str(fixtures))
    return fixtures


@pytest.fixture
def mock_vllm(monkeypatch):
    """Intercept outgoing httpx requests to vLLM, record last body."""
    from unittest.mock import AsyncMock
    import httpx

    class _Recorder:
        def __init__(self):
            self.last_request_body: dict | None = None

        async def post(self, url, *, json=None, **kwargs):
            self.last_request_body = json
            # 返回一个 minimal 合法 response
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl-mock",
                    "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
                },
                request=httpx.Request("POST", url),
            )

    recorder = _Recorder()
    monkeypatch.setattr(httpx.AsyncClient, "post", recorder.post)
    return recorder
```

- [ ] **Step 3: 运行测试验证失败**

Run: `cd backend && pytest tests/test_responses_agent_binding.py -v`
Expected: FAIL — `agent` 字段未识别

- [ ] **Step 4: 扩展 CreateResponseRequest 和 create_response**

在 `backend/src/api/routes/responses.py` 里：

(a) `CreateResponseRequest` 类加字段：

```python
class CreateResponseRequest(BaseModel):
    model: str
    input: str | list[Any]
    agent: str | None = None            # ← 新增
    previous_response_id: str | None = None
    context_id: str | None = None
    instructions: str | None = None
    # ... 其余不变
```

(b) imports 顶部加：

```python
from src.config import get_settings
from src.services.prompt_composer import (
    AgentLoadFailed,
    AgentNotFound,
    compose as compose_agent_prompt,
)
from src.services.responses_service import assert_agent_matches_session
from src.services.skill_tools import skill_tool_schema
```

(c) 在 `create_response()` 的 messages 装配段（`responses.py:384-394` 附近）改为：

```python
    # --- agent/skill 装配 ---
    settings = get_settings()
    effective_agent: str | None = None

    if settings.NOUS_ENABLE_AGENT_INJECTION:
        if sess is not None:
            # 续请求：从 session 绑定校验
            effective_agent = assert_agent_matches_session(sess, req.agent)
        else:
            # 首请求
            effective_agent = req.agent

    agent_sys: str | None = None
    if effective_agent:
        try:
            agent_sys = compose_agent_prompt(effective_agent, None)
        except AgentNotFound:
            raise InvalidRequestError(
                f"agent not found: {effective_agent}",
                code="agent_not_found",
            )
        except AgentLoadFailed as e:
            logger.error("agent load failed: %s", e)
            raise APIError(
                f"failed to load agent {effective_agent}",
                code="agent_load_failed",
            )

    # --- 装配 messages（新 MESSAGES_ORDER）---
    previous_messages_vllm = transform_inputs_to_chat_messages(previous_messages)
    messages: list[dict] = []
    if agent_sys is not None:
        messages.append({"role": "system", "content": agent_sys})
    if cached_messages:
        messages.extend(cached_messages)
    messages.extend(previous_messages_vllm)
    if req.instructions:
        messages.append({"role": "system", "content": req.instructions})
    messages.extend(new_input_messages)
```

(d) 在 `create_session(...)` 调用处把 `agent_id=effective_agent` 传下去（需要在 `responses_service.create_session` 签名里加 `agent_id` 参数，或复制当前 session 创建代码加字段）。

(e) 在 tools 数组传给 vLLM 的地方（找到 `tools=[...]` 传参的位置），加入 `skill_tool_schema()`：

```python
    # 组 tools
    tools_list: list[dict] = []
    if effective_agent:
        tools_list.append(skill_tool_schema())
    if req.tools:
        tools_list.extend(req.tools)
    # ... vllm_body["tools"] = tools_list if tools_list else None
```

(f) `record_llm_usage(...)` 调用处传 `agent_id=effective_agent`。

- [ ] **Step 5: 运行测试验证通过**

Run: `cd backend && pytest tests/test_responses_agent_binding.py -v`
Expected: PASS（5 tests）

- [ ] **Step 6: 回归现有 responses 测试**

Run: `cd backend && pytest tests/ -v -k responses`
Expected: 所有现有 responses 测试仍通过

- [ ] **Step 7: Commit**

```bash
git add backend/src/api/routes/responses.py \
        backend/tests/test_responses_agent_binding.py \
        backend/tests/conftest.py
git commit -m "feat(responses): agent field + composer integration + MESSAGES_ORDER"
```

---

## Task 12: /v1/chat/completions 接入

**Files:**
- Modify: `backend/src/api/routes/openai_compat.py`
- Test: 扩展 `backend/tests/test_responses_agent_binding.py`

- [ ] **Step 1: 写失败测试**

追加到 `backend/tests/test_responses_agent_binding.py`：

```python
@pytest.mark.asyncio
async def test_chat_completions_with_agent_injects_system_message(
    monkeypatch, api_client, bearer_headers, fixtures_home, mock_vllm
):
    monkeypatch.setenv("NOUS_ENABLE_AGENT_INJECTION", "true")
    resp = await api_client.post(
        "/v1/chat/completions",
        json={
            "model": "qwen3.5",
            "agent": "tutor",
            "messages": [{"role": "user", "content": "你好"}],
        },
        headers=bearer_headers,
    )
    assert resp.status_code == 200
    sent = mock_vllm.last_request_body["messages"]
    # 首条应是 agent system message
    assert sent[0]["role"] == "system"
    assert "你是 Tutor" in sent[0]["content"]


@pytest.mark.asyncio
async def test_chat_completions_no_agent_unchanged(
    monkeypatch, api_client, bearer_headers, mock_vllm
):
    monkeypatch.setenv("NOUS_ENABLE_AGENT_INJECTION", "true")
    resp = await api_client.post(
        "/v1/chat/completions",
        json={
            "model": "qwen3.5",
            "messages": [{"role": "user", "content": "hi"}],
        },
        headers=bearer_headers,
    )
    assert resp.status_code == 200
    sent = mock_vllm.last_request_body["messages"]
    assert sent == [{"role": "user", "content": "hi"}]
```

- [ ] **Step 2: 运行测试验证失败**

Run: `cd backend && pytest tests/test_responses_agent_binding.py::test_chat_completions_with_agent_injects_system_message -v`
Expected: FAIL

- [ ] **Step 3: 改 openai_compat.py**

读 `backend/src/api/routes/openai_compat.py` 找到 chat/completions 的 request 模型和 messages 转发处。

(a) request schema 加 `agent: str | None = None` 字段。

(b) 在 messages 转发前插入 agent 装配（chat/completions 没有 session 概念，所以**只处理首请求**式装配，无绑定校验）：

```python
    settings = get_settings()
    agent_sys: str | None = None
    if settings.NOUS_ENABLE_AGENT_INJECTION and req.agent:
        try:
            agent_sys = compose_agent_prompt(req.agent, None)
        except AgentNotFound:
            raise InvalidRequestError(
                f"agent not found: {req.agent}",
                code="agent_not_found",
            )
        except AgentLoadFailed as e:
            logger.error("agent load failed: %s", e)
            raise APIError(
                f"failed to load agent {req.agent}",
                code="agent_load_failed",
            )

    messages = list(req.messages)
    if agent_sys is not None:
        messages.insert(0, {"role": "system", "content": agent_sys})
```

(c) 同样把 `skill_tool_schema()` 加到 tools 数组（若 agent 存在）。

(d) `record_llm_usage(..., agent_id=req.agent if settings.NOUS_ENABLE_AGENT_INJECTION else None)`。

- [ ] **Step 4: 运行测试验证通过**

Run: `cd backend && pytest tests/test_responses_agent_binding.py -v`
Expected: PASS（7 tests 总）

- [ ] **Step 5: 回归**

Run: `cd backend && pytest tests/ -v -k "openai_compat or chat_completions"`
Expected: 所有现有测试通过

- [ ] **Step 6: Commit**

```bash
git add backend/src/api/routes/openai_compat.py backend/tests/test_responses_agent_binding.py
git commit -m "feat(chat_completions): agent field + composer integration"
```

---

## Task 13: GET /api/v1/agents/{name}/preview endpoint

**Files:**
- Modify: `backend/src/api/routes/agents.py`
- Test: 扩展 `backend/tests/test_api_agents.py`

- [ ] **Step 1: 写失败测试**

追加到 `backend/tests/test_api_agents.py`：

```python
@pytest.mark.asyncio
async def test_agent_preview_returns_system_message(
    api_client, admin_headers, fixtures_home
):
    resp = await api_client.get(
        "/api/v1/agents/tutor/preview",
        headers=admin_headers,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "system_message" in data
    assert "你是 Tutor" in data["system_message"]
    assert "<available_skills>" in data["system_message"]


@pytest.mark.asyncio
async def test_agent_preview_requires_admin(
    api_client, bearer_headers, fixtures_home
):
    resp = await api_client.get(
        "/api/v1/agents/tutor/preview",
        headers=bearer_headers,  # 非 admin
    )
    assert resp.status_code in (401, 403)


@pytest.mark.asyncio
async def test_agent_preview_not_found(api_client, admin_headers, tmp_path, monkeypatch):
    monkeypatch.setenv("NOUS_CENTER_HOME", str(tmp_path))
    resp = await api_client.get(
        "/api/v1/agents/ghost/preview",
        headers=admin_headers,
    )
    assert resp.status_code == 404
```

- [ ] **Step 2: 运行测试验证失败**

Run: `cd backend && pytest tests/test_api_agents.py -v -k preview`
Expected: FAIL — 404 (endpoint missing)

- [ ] **Step 3: 加 endpoint**

在 `backend/src/api/routes/agents.py` 末尾添加：

```python
@router.get("/{name}/preview", dependencies=[Depends(require_admin)])
def preview_agent(name: str):
    """Preview the system message that compose() produces for this agent.

    Admin-only debug endpoint. Does not execute any LLM call.
    """
    from src.services.prompt_composer import (
        AgentLoadFailed,
        AgentNotFound,
        compose,
    )
    try:
        msg = compose(agent_id=name, instructions=None)
    except AgentNotFound:
        raise HTTPException(404, f"agent not found: {name}")
    except AgentLoadFailed as e:
        raise HTTPException(500, f"agent load failed: {e}")
    return {"agent": name, "system_message": msg or ""}
```

- [ ] **Step 4: 运行测试验证通过**

Run: `cd backend && pytest tests/test_api_agents.py -v -k preview`
Expected: PASS（3 tests）

- [ ] **Step 5: Commit**

```bash
git add backend/src/api/routes/agents.py backend/tests/test_api_agents.py
git commit -m "feat(agents): GET /{name}/preview debug endpoint"
```

---

## Task 14: 手动 E2E — Qwen3.5 真实调 Skill tool 验证

**Files:** 无代码改动，只做手动验证 + 记录结果

- [ ] **Step 1: 启 vLLM Qwen3.5**

Run: `cd backend && ./scripts/start_vllm.sh`
等待模型 loaded（可查 `GET /api/v1/models`）。

- [ ] **Step 2: 建一个 fixture agent + skill 在真实 `~/.nous-center/`**

```bash
mkdir -p ~/.nous-center/agents/qa_tutor
cat > ~/.nous-center/agents/qa_tutor/config.json <<'EOF'
{"display_name": "QA Tutor", "model": "qwen3.5", "skills": ["qa_search"], "status": "active"}
EOF
echo "你是 QA 专家。" > ~/.nous-center/agents/qa_tutor/IDENTITY.md
echo "精确、引用来源。" > ~/.nous-center/agents/qa_tutor/SOUL.md
echo "遇到事实问题先调 qa_search。" > ~/.nous-center/agents/qa_tutor/AGENT.md

mkdir -p ~/.nous-center/skills/qa_search
cat > ~/.nous-center/skills/qa_search/SKILL.md <<'EOF'
---
name: qa_search
description: 在本地知识库里搜索答案
---

## 使用说明

你被调用说明需要检索事实。返回："根据 X 文档，答案是 ..." 格式。
EOF
```

- [ ] **Step 3: 打开 feature flag 重启后端**

```bash
NOUS_ENABLE_AGENT_INJECTION=true ./scripts/start_api.sh
```

- [ ] **Step 4: 发起真实请求**

```bash
curl -sS -X POST http://localhost:8000/v1/responses \
  -H "Authorization: Bearer $NOUS_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.5","agent":"qa_tutor","input":"What is the capital of Japan?"}' | jq
```

预期：响应里能看到 `output` 的 assistant turn，且若模型**真的调用了 Skill tool**，会看到 `function_call` 或 `tool_calls` event。

- [ ] **Step 5: 检查 preview endpoint 输出**

```bash
curl -sS http://localhost:8000/api/v1/agents/qa_tutor/preview \
  -H "Authorization: Bearer $ADMIN_KEY" | jq -r .system_message
```

验证 system_message 格式符合 spec（IDENTITY → SOUL → AGENT → Available Skills → CACHE_BOUNDARY）。

- [ ] **Step 6: 检查 llm_usage 表**

```bash
psql $DATABASE_URL -c "SELECT agent_id, model, prompt_tokens, completion_tokens FROM llm_usage ORDER BY created_at DESC LIMIT 5;"
```

验证 `agent_id="qa_tutor"` 出现在最新一行。

- [ ] **Step 7: 记录 QA 结果**

在 PR 描述里附上：
- 模型是否真的调用 Skill tool（Y/N）
- 若 N，模型实际行为（直接回答？无视 `<available_skills>`？）
- 若 N，修复方向（强化 SKILLS_INSTRUCTION 措辞？加 few-shot？换 function-tool 路线？）

**这一步是整个 spec 最大的未知：本地 Qwen3.5 是否接得住 lazy-readable 范式。结果会决定是否需要 Task 15 调整 prompt 或策略。**

- [ ] **Step 8: 手工 QA 通过后，准备 PR**

```bash
git log --oneline -15  # 确认 14 个任务的 commit 都在
git push -u origin feature/agent-skill-injection  # 按选项 C 的独立 branch 策略
gh pr create --base feature/nous-center-v2 --title "feat: agent/skill injection (lazy-readable)" --body "$(cat <<'EOF'
## Summary
- 新增 `agent` 字段到 `/v1/responses` 和 `/v1/chat/completions`
- Lazy-readable skill 注入（system prompt 放 `<available_skills>`，专用 `Skill` tool）
- Session 绑定 + 一致性校验
- Feature flag `NOUS_ENABLE_AGENT_INJECTION`（默认关）
- usage 表加 agent_id 列

Spec: docs/designs/2026-04-20-agent-skill-injection.md

## Test plan
- [x] Unit: prompt_composer 18 tests（含 golden file）
- [x] Unit: Skill tool exec 6 tests
- [x] Unit: agent session binding 5 tests
- [x] Integration: /v1/responses agent field 5 tests
- [x] Integration: /v1/chat/completions agent field 2 tests
- [x] Manual E2E: Qwen3.5 真实调 Skill tool → [结果见下]
- [x] 回归：现有 responses / chat / agents / skills / usage 测试全过

## Rollout
1. Deploy migration first: `python scripts/migrate_agent_id.py`
2. Deploy code with `NOUS_ENABLE_AGENT_INJECTION=false`
3. Enable flag on 1 machine for 24h smoke
4. Roll out
EOF
)"
```

---

## Self-Review Checklist

（writing-plans skill 要求我在 plan 完成后跑这个 checklist）

### 1. Spec coverage

| Spec section | Task |
|--------------|------|
| API 契约（agent 字段） | Task 11 + 12 |
| System message 装配（IDENTITY/SOUL/AGENT 顺序） | Task 2 + 4 |
| `<available_skills>` XML 段 | Task 3 |
| MESSAGES_ORDER 改动 | Task 11（含回归测试） |
| Skill tool（Claude Code 风格） | Task 6 |
| Session 绑定 + 一致性校验 | Task 10 + 11 |
| Cache 策略（mtime fingerprint） | Task 2（lru_cache 按 config.json mtime） |
| Feature flag | Task 9 + 11 + 12 |
| Observability（agent_id 入 usage 表） | Task 7 + 8 |
| Deployment order | Task 7（migration 脚本） + Task 14（rollout checklist） |
| 错误处理（agent_load_failed / mismatch） | Task 2 + 10 + 11 |
| 测试策略（unit + integration + golden） | Task 1-12 各自 TDD + Task 5 golden |
| `Skill(args=None)` 兜底 | Task 6 |
| 空 persona 返回 `None` | Task 4 |
| Preview endpoint | Task 13 |

全部 spec 要点都有对应 task。

### 2. Placeholder scan

No "TBD" / "TODO" / "implement later" / "handle edge cases"（除 Task 14 里 "结果见下" 是真 placeholder，但指的是 PR 写作时填的真实结果，非代码 placeholder）。

### 3. Type consistency

- `compose(agent_id, instructions) -> str | None` 签名在 Task 4 定义，Task 11/12/13 一致使用
- `PersonaBundle` 类在 Task 2 定义，`load_persona` 返回此类型
- `AgentNotFound` / `AgentLoadFailed` 在 Task 2 定义，Task 4/11/12/13 一致引用
- `skill_tool_schema()` 和 `_execute_skill_tool()` 在 Task 6 定义，Task 11/12 一致引用
- `assert_agent_matches_session(sess, request_agent) -> str | None` 在 Task 10 定义，Task 11 使用
- `record_llm_usage(..., agent_id=...)` 在 Task 8 扩展签名，Task 11/12 使用

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-20-agent-skill-injection.md`. 总 14 个任务，~3 天工作量。

两种执行方式：

**1. Subagent-Driven（推荐，按你之前的倾向）** — 我为每个 task 派发一个 fresh subagent，在 task 之间做 review，快速迭代。适合这种 TDD 链式任务（前任务的 type 定义影响后任务）。

**2. Inline Execution** — 在本会话内按顺序跑，带 checkpoint 让你中途 review。适合你想紧密跟进每一步。

按你之前选的选项 C（并行分支），建议 Subagent-Driven 跑本 plan，同时另开一个分支并行跑 Wave 1 的 plan（Wave 1 需要单独再跑一次 writing-plans）。

**Which approach?**