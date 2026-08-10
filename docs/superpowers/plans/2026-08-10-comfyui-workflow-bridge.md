# ComfyUI 通用工作流桥 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 任意 ComfyUI 工作流(API 格式 JSON)可导入 nous-engine 成为服务,经 prediction API 异步执行于本地 ComfyUI sidecar,产物(视频/音频/图)落 nous 存储;配套读图选节点的字段配置 UI 与视频可见性 UI。

**Architecture:** 模板=服务(`source_type=comfy_template`):注册 API 把 ComfyUI JSON 存 `comfy_templates` 表,并创建 ServiceInstance,其 `workflow_snapshot` 为单节点桥工作流(`comfyui_workflow` 节点 + `video_output` 节点),`exposed_inputs` 指向桥节点——现有 predictions 管线(`apply_inputs_to_snapshot`/`run_workflow_task`)零改动直通。桥节点内经 `ComfyClient` 提交 sidecar、轮询、分拣下载产物。

**Tech Stack:** FastAPI + SQLAlchemy(async) + httpx(`MockTransport` 测试) / React + React Flow + vitest / systemd。

**Spec:** `docs/superpowers/specs/2026-08-10-comfyui-workflow-bridge-design.md`(必读)。UI 稿:artifact v5。

## Global Constraints

- 所有对 sidecar 的 httpx 客户端必须 `trust_env=False`(防 mihomo ALL_PROXY 劫持)。
- 环境变量:`NOUS_COMFY_URL`(默认 `http://127.0.0.1:8188`)、`NOUS_COMFY_TIMEOUT`(秒,默认 `14400`)、`NOUS_COMFY_DOWNLOAD_TIMEOUT`(秒,默认 `120`)。读取一律 `os.getenv` + 默认值。
- 后端测试跑法:`cd backend && uv run pytest tests/... -x -q`。conftest 已设 `ADMIN_PASSWORD=""` 与 `NOUS_DISABLE_FRONTEND_MOUNT=1`,不得改动。
- 前端测试:`cd frontend && npx vitest run <file>`。生产改动最后需 `npm run build`。
- 服务名规则 `^[a-z][a-z0-9-]{1,62}$`(ServiceInstance.name 的 v3 契约)。
- 写路径动了服务列表必须 `invalidate("services")`(`src/api/response_cache.py`)。
- 新增 `/api/v1/*` 路由走 admin 门;`/v1/*` 是对外 bearer 端点——本计划只加 `/api/v1/*`。
- commit 信息风格照 repo 现状:`feat(comfy): ...` / `test(comfy): ...` 中文短句。

---

### Task 1: ComfyClient — sidecar HTTP 客户端与错误翻译

**Files:**
- Create: `backend/src/services/comfy/__init__.py`(空)
- Create: `backend/src/services/comfy/client.py`
- Test: `backend/tests/comfy/test_client.py`(目录加空 `__init__.py`)

**Interfaces:**
- Consumes: 无(纯新模块)。
- Produces:
  - `class ComfyClient(base_url: str | None = None)` — async 方法:
    - `async health() -> dict` → `{"online": bool, "queue_depth": int, "version": str}`
    - `async object_info() -> dict`(原样返回 sidecar JSON)
    - `async upload_image(filename: str, content: bytes, mime: str = "image/png") -> str`(返回 sidecar 侧文件名)
    - `async submit(graph: dict) -> str`(返回 prompt_id;校验失败抛 `ComfyError`)
    - `async wait(prompt_id: str, *, timeout_s: float, interval_s: float = 2.0) -> dict`(返回 history 条目;超时抛 `ComfyError("ComfyUI 渲染超时…")`)
    - `async download(item: dict) -> bytes`(item 含 filename/subfolder/type)
    - `async interrupt() -> None`
  - `class ComfyError(RuntimeError)`:`.message: str`、`.status_code: int = 502`
  - `def translate_prompt_error(status: int, body: str) -> str`(ComfyUI 校验 JSON → 中文人话)

- [ ] **Step 1: 写失败测试**

```python
# backend/tests/comfy/test_client.py
import json
import httpx
import pytest

from src.services.comfy.client import ComfyClient, ComfyError, translate_prompt_error


def make_client(handler) -> ComfyClient:
    c = ComfyClient(base_url="http://comfy.test")
    c._client = httpx.AsyncClient(  # noqa: SLF001 — 测试注入 MockTransport
        transport=httpx.MockTransport(handler), base_url="http://comfy.test")
    return c


@pytest.mark.asyncio
async def test_submit_returns_prompt_id():
    def handler(req):
        assert req.url.path == "/prompt"
        return httpx.Response(200, json={"prompt_id": "abc123"})
    assert await make_client(handler).submit({"1": {"class_type": "X", "inputs": {}}}) == "abc123"


@pytest.mark.asyncio
async def test_submit_validation_error_translated():
    def handler(req):
        return httpx.Response(400, json={"error": {"message": "invalid prompt"}, "node_errors": {
            "138": {"errors": [{"message": "Required input is missing", "extra_info": {"input_name": "text"}}],
                     "class_type": "PrimitiveStringMultiline"}}})
    with pytest.raises(ComfyError) as e:
        await make_client(handler).submit({})
    assert "138" in e.value.message and "text" in e.value.message


@pytest.mark.asyncio
async def test_wait_polls_until_history_present():
    calls = {"n": 0}
    def handler(req):
        calls["n"] += 1
        body = {} if calls["n"] < 3 else {"p1": {"outputs": {"92": {"images": []}}}}
        return httpx.Response(200, json=body)
    hist = await make_client(handler).wait("p1", timeout_s=10, interval_s=0)
    assert "outputs" in hist and calls["n"] == 3


@pytest.mark.asyncio
async def test_wait_timeout_raises():
    def handler(req):
        return httpx.Response(200, json={})
    with pytest.raises(ComfyError, match="超时"):
        await make_client(handler).wait("p1", timeout_s=0.01, interval_s=0)


@pytest.mark.asyncio
async def test_health_offline_on_connect_error():
    def handler(req):
        raise httpx.ConnectError("refused")
    h = await make_client(handler).health()
    assert h["online"] is False


def test_translate_prompt_error_fallback_plain_text():
    msg = translate_prompt_error(500, "Internal Server Error")
    assert "HTTP 500" in msg
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd backend && uv run pytest tests/comfy/test_client.py -x -q`
Expected: FAIL(`ModuleNotFoundError: src.services.comfy`)

- [ ] **Step 3: 实现 client.py**

```python
"""ComfyUI sidecar HTTP 客户端(spec §5)。所有请求 trust_env=False。"""
from __future__ import annotations

import asyncio
import json
import os
import urllib.parse

import httpx


def _base_url() -> str:
    return os.getenv("NOUS_COMFY_URL", "http://127.0.0.1:8188").rstrip("/")


class ComfyError(RuntimeError):
    def __init__(self, message: str, *, status_code: int = 502) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def translate_prompt_error(status: int, body: str) -> str:
    """ComfyUI /prompt 校验 payload → 可操作中文(仿 IC comfy_prompt_error_message)。"""
    try:
        data = json.loads(body)
    except (ValueError, TypeError):
        return f"ComfyUI 请求失败(HTTP {status}):{body[:300] or '未知错误'}"
    parts: list[str] = []
    top = (data.get("error") or {}).get("message") or ""
    for node_id, ne in (data.get("node_errors") or {}).items():
        ct = ne.get("class_type") or ""
        for err in ne.get("errors") or []:
            inp = (err.get("extra_info") or {}).get("input_name") or ""
            parts.append(f"节点 {node_id}({ct}) 输入 {inp}:{err.get('message', '')}")
    detail = ";".join(parts) or top
    return f"ComfyUI 拒绝了工作流(HTTP {status}):{detail[:600] or '校验失败'}。请检查模板字段映射。"


class ComfyClient:
    def __init__(self, base_url: str | None = None) -> None:
        self.base_url = (base_url or _base_url()).rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self.base_url, trust_env=False,
            timeout=float(os.getenv("NOUS_COMFY_DOWNLOAD_TIMEOUT", "120")))

    async def health(self) -> dict:
        try:
            r = await self._client.get("/queue", timeout=3)
            q = r.json()
            depth = len(q.get("queue_running", [])) + len(q.get("queue_pending", []))
            ver = ""
            try:
                ver = (await self._client.get("/system_stats", timeout=3)).json() \
                    .get("system", {}).get("comfyui_version", "")
            except (httpx.HTTPError, ValueError):
                pass
            return {"online": True, "queue_depth": depth, "version": ver}
        except (httpx.HTTPError, ValueError):
            return {"online": False, "queue_depth": 0, "version": ""}

    async def object_info(self) -> dict:
        return (await self._client.get("/object_info", timeout=15)).json()

    async def upload_image(self, filename: str, content: bytes, mime: str = "image/png") -> str:
        r = await self._client.post(
            "/upload/image", files={"image": (filename, content, mime)}, timeout=60)
        if r.status_code != 200:
            raise ComfyError(f"上传参考素材失败(HTTP {r.status_code})")
        return r.json().get("name", filename)

    async def submit(self, graph: dict) -> str:
        r = await self._client.post("/prompt", json={"prompt": graph}, timeout=30)
        if r.status_code != 200:
            raise ComfyError(translate_prompt_error(r.status_code, r.text), status_code=422)
        return r.json()["prompt_id"]

    async def wait(self, prompt_id: str, *, timeout_s: float, interval_s: float = 2.0) -> dict:
        loop = asyncio.get_event_loop()
        deadline = loop.time() + timeout_s
        while True:
            try:
                res = (await self._client.get(f"/history/{prompt_id}", timeout=10)).json()
                if prompt_id in res:
                    return res[prompt_id]
            except (httpx.HTTPError, ValueError):
                pass  # sidecar 瞬断:继续轮询直到超时
            if loop.time() >= deadline:
                raise ComfyError("ComfyUI 渲染超时(NOUS_COMFY_TIMEOUT)。注意:ComfyUI 侧任务可能仍在运行。")
            await asyncio.sleep(interval_s)

    async def download(self, item: dict) -> bytes:
        qs = urllib.parse.urlencode({
            "filename": item["filename"], "subfolder": item.get("subfolder", ""),
            "type": item.get("type", "output")})
        r = await self._client.get(f"/view?{qs}")
        if r.status_code != 200:
            raise ComfyError(f"下载产物失败(HTTP {r.status_code}):{item['filename']}")
        return r.content

    async def interrupt(self) -> None:
        try:
            await self._client.post("/interrupt", timeout=5)
        except httpx.HTTPError:
            pass  # 尽力而为:sidecar 掉线时取消不应抛错
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd backend && uv run pytest tests/comfy/test_client.py -x -q`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add backend/src/services/comfy backend/tests/comfy
git commit -m "feat(comfy): sidecar HTTP 客户端 + 校验错误翻译"
```

---

### Task 2: 产物分拣 — comfy_outputs.py

**Files:**
- Create: `backend/src/services/comfy/outputs.py`
- Test: `backend/tests/comfy/test_outputs.py`

**Interfaces:**
- Consumes: 无。
- Produces:
  - `@dataclass OutputItem`: `node_id: str; class_type: str; filename: str; subfolder: str; file_type: str; kind: str`
  - `def classify_ext(filename: str) -> str` → `image|video|audio|text|file`
  - `def collect_outputs(history: dict, graph: dict) -> list[OutputItem]`(preview/调试节点过滤规则见测试)

- [ ] **Step 1: 写失败测试**

```python
# backend/tests/comfy/test_outputs.py
from src.services.comfy.outputs import classify_ext, collect_outputs


def test_classify_ext():
    assert classify_ext("a.mp4") == "video"
    assert classify_ext("a.PNG") == "image"
    assert classify_ext("a.flac") == "audio"
    assert classify_ext("a.txt") == "text"
    assert classify_ext("a.bin") == "file"


GRAPH = {
    "92": {"class_type": "SaveVideo", "inputs": {}},
    "50": {"class_type": "PreviewImage", "inputs": {}},
    "51": {"class_type": "SaveImage", "inputs": {}},
}
HISTORY = {"outputs": {
    "92": {"images": [{"filename": "out.mp4", "subfolder": "", "type": "output"}]},
    "50": {"images": [{"filename": "prev.png", "subfolder": "", "type": "temp"}]},
    "51": {"images": [{"filename": "final.png", "subfolder": "", "type": "output"}]},
}}


def test_preview_dropped_when_primary_image_exists():
    items = collect_outputs(HISTORY, GRAPH)
    names = {i.filename for i in items}
    assert names == {"out.mp4", "final.png"}
    assert next(i for i in items if i.filename == "out.mp4").kind == "video"


def test_preview_kept_when_only_output():
    hist = {"outputs": {"50": HISTORY["outputs"]["50"]}}
    items = collect_outputs(hist, GRAPH)
    assert [i.filename for i in items] == ["prev.png"]
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd backend && uv run pytest tests/comfy/test_outputs.py -x -q`
Expected: FAIL(import error)

- [ ] **Step 3: 实现 outputs.py**

```python
"""ComfyUI history 产物分拣(仿 IC:扩展名定 kind,冗余 preview 图过滤)。"""
from __future__ import annotations

from dataclasses import dataclass

_KIND_BY_EXT = {
    "png": "image", "jpg": "image", "jpeg": "image", "webp": "image", "gif": "image",
    "mp4": "video", "webm": "video", "mov": "video", "mkv": "video",
    "wav": "audio", "mp3": "audio", "flac": "audio", "ogg": "audio",
    "txt": "text", "json": "text", "srt": "text",
}
_PREVIEW_HINTS = ("previewimage", "comparer", "imagecompare")


@dataclass
class OutputItem:
    node_id: str
    class_type: str
    filename: str
    subfolder: str
    file_type: str
    kind: str


def classify_ext(filename: str) -> str:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return _KIND_BY_EXT.get(ext, "file")


def _is_preview(class_type: str) -> bool:
    ct = class_type.lower()
    return any(h in ct for h in _PREVIEW_HINTS)


def collect_outputs(history: dict, graph: dict) -> list[OutputItem]:
    cands: list[OutputItem] = []
    for node_id, node_out in (history.get("outputs") or {}).items():
        ct = str((graph.get(str(node_id)) or {}).get("class_type") or "")
        for key in ("images", "videos", "audio", "gifs", "files"):
            for item in node_out.get(key) or []:
                if not isinstance(item, dict) or "filename" not in item:
                    continue
                cands.append(OutputItem(
                    node_id=str(node_id), class_type=ct,
                    filename=str(item["filename"]),
                    subfolder=str(item.get("subfolder", "")),
                    file_type=str(item.get("type", "output")),
                    kind=classify_ext(str(item["filename"]))))
    has_primary_image = any(c.kind == "image" and not _is_preview(c.class_type) for c in cands)
    return [c for c in cands
            if not (c.kind == "image" and has_primary_image and _is_preview(c.class_type))]
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd backend && uv run pytest tests/comfy/test_outputs.py -x -q`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add backend/src/services/comfy/outputs.py backend/tests/comfy/test_outputs.py
git commit -m "feat(comfy): history 产物分拣与 preview 过滤"
```

---

### Task 3: 模板模型 + 注册/映射 API

**Files:**
- Create: `backend/src/models/comfy_template.py`
- Create: `backend/src/api/routes/comfy_templates.py`
- Modify: `backend/src/api/main.py`(挂路由,与其他 routes 同处 include)
- Modify: `backend/src/models/__init__.py`(若存在模型聚合导入,加 ComfyTemplate;先读该文件确认惯例)
- Test: `backend/tests/comfy/test_templates_api.py`

**Interfaces:**
- Consumes: `ServiceInstance`(字段见 `src/models/service_instance.py`)、`invalidate`(`src/api/response_cache.py`)、`get_async_session`。
- Produces:
  - 表 `comfy_templates`:`id BigInteger pk(snowflake_id)`、`name String(100) unique`、`workflow_json JSON`、`created_at/updated_at`。
  - `POST /api/v1/comfy-templates` body `{name, workflow: dict}` → 建模板 + ServiceInstance(`source_type="comfy_template"`, `source_id=模板id`, `category="app"`, `type="workflow"`, `meter_dim="calls"`, `workflow_snapshot=桥快照`, `exposed_inputs=[]`) → 201 `{id, name, service_name, node_count}`。
  - **桥快照形状(后续任务契约)**:
    ```json
    {"nodes": [
       {"id": "bridge", "type": "comfyui_workflow", "data": {"template_id": <id>}},
       {"id": "out", "type": "video_output", "data": {}}],
     "edges": [{"id": "e1", "source": "bridge", "sourceHandle": "outputs",
                "target": "out", "targetHandle": "outputs"}]}
    ```
  - `PUT /api/v1/comfy-templates/{id}/mapping` body `{exposed_params: [{key,label,node_id,input_name,type,default?,min?,max?,step?,options?,required?,random?}]}` → 写入服务 `exposed_inputs`(每项补 `{"node_id": "bridge", "input_name": key}` 供 apply_inputs_to_snapshot 使用,原 ComfyUI 节点定位存进各项的 `comfy_node_id`/`comfy_input`)→ 200。
  - `PUT /api/v1/comfy-templates/{id}` body `{workflow: dict}` 重新上传 → 校验现有映射的 `comfy_node_id`/`comfy_input` 是否仍存在,返回 `{stale_keys: [...]}`。
  - `GET /api/v1/comfy-templates` → `[{id, name, service_name, node_count, exposed_count}]`;`GET .../{id}` 含 `workflow_json` 与 `exposed_params`;`DELETE .../{id}` 连带删服务。
  - 所有写端点结尾 `invalidate("services")`。

- [ ] **Step 1: 写失败测试**(用现有 conftest 的 app/client fixture;先读 `backend/tests/conftest.py` 摸清 fixture 名,以下按惯用 `client`)

```python
# backend/tests/comfy/test_templates_api.py
import pytest

WF = {"138": {"class_type": "PrimitiveStringMultiline", "inputs": {"value": ""}},
      "92": {"class_type": "SaveVideo", "inputs": {}}}


@pytest.mark.asyncio
async def test_create_template_creates_service(client):
    r = await client.post("/api/v1/comfy-templates", json={"name": "minimax-h3-r2v", "workflow": WF})
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["service_name"] == "minimax-h3-r2v" and body["node_count"] == 2
    svc = await client.get("/v1/services/minimax-h3-r2v/schema")
    assert svc.status_code == 200
    assert svc.json()["source_type"] == "comfy_template"


@pytest.mark.asyncio
async def test_mapping_roundtrip_and_stale_detection(client):
    r = await client.post("/api/v1/comfy-templates", json={"name": "tpl-map", "workflow": WF})
    tid = r.json()["id"]
    mapping = {"exposed_params": [{"key": "prompt", "label": "提示词", "type": "string",
                                   "comfy_node_id": "138", "comfy_input": "value", "required": True}]}
    assert (await client.put(f"/api/v1/comfy-templates/{tid}/mapping", json=mapping)).status_code == 200
    detail = (await client.get(f"/api/v1/comfy-templates/{tid}")).json()
    assert detail["exposed_params"][0]["key"] == "prompt"
    # 重新上传丢掉 138 节点 → prompt 映射失效
    r2 = await client.put(f"/api/v1/comfy-templates/{tid}",
                          json={"workflow": {"92": WF["92"]}})
    assert r2.json()["stale_keys"] == ["prompt"]


@pytest.mark.asyncio
async def test_delete_removes_service(client):
    r = await client.post("/api/v1/comfy-templates", json={"name": "tpl-del", "workflow": WF})
    tid = r.json()["id"]
    assert (await client.delete(f"/api/v1/comfy-templates/{tid}")).status_code == 204
    assert (await client.get("/v1/services/tpl-del/schema")).status_code == 404
```

- [ ] **Step 2: 跑测试确认失败** — Run: `uv run pytest tests/comfy/test_templates_api.py -x -q`;Expected: 404(路由不存在)

- [ ] **Step 3: 实现模型 + 路由**。模型照 `execution_task.py` 惯例;路由要点:name 校验服务名正则;创建时构造上文桥快照;`exposed_inputs` 存储时每项展开为 `{key, label, node_id: "bridge", input_name: key, type, default, min, max, step, options, required, random, comfy_node_id, comfy_input}`;stale 校验 = 遍历 mapping,查 `comfy_node_id in workflow and comfy_input in workflow[nid]["inputs"]`。挂路由方式照 `src/api/main.py` 现有 include_router 列表。

- [ ] **Step 4: 跑测试确认通过** — `uv run pytest tests/comfy/ -x -q`;Expected: 全 passed

- [ ] **Step 5: Commit** — `git commit -m "feat(comfy): 模板即服务 — comfy_templates 表 + 注册/映射 API"`

---

### Task 4: sidecar 健康/object_info 代理路由

**Files:**
- Modify: `backend/src/api/routes/comfy_templates.py`(同文件追加)
- Test: `backend/tests/comfy/test_health_routes.py`

**Interfaces:**
- Consumes: `ComfyClient`(Task 1)。
- Produces: `GET /api/v1/comfy/health` → ComfyClient.health() 结果 + `{"base_url": ..., "timeout_s": ...}`;`GET /api/v1/comfy/object-info` → 代理 object_info(进程内 60s TTL 缓存,模块级 `_cache: tuple[float, dict] | None`)。路由通过模块级工厂 `get_client() -> ComfyClient` 获取客户端——**测试用 monkeypatch 替换 get_client**,后续任务同此模式。

- [ ] **Step 1: 写失败测试**

```python
# backend/tests/comfy/test_health_routes.py
import pytest
import src.api.routes.comfy_templates as ct_mod


class FakeClient:
    async def health(self):
        return {"online": True, "queue_depth": 2, "version": "0.30.2"}
    async def object_info(self):
        return {"RandomNoise": {"input": {"required": {"noise_seed": ["INT", {"default": 0}]}}}}


@pytest.mark.asyncio
async def test_health(client, monkeypatch):
    monkeypatch.setattr(ct_mod, "get_client", lambda: FakeClient())
    r = await client.get("/api/v1/comfy/health")
    assert r.status_code == 200 and r.json()["queue_depth"] == 2


@pytest.mark.asyncio
async def test_object_info_cached(client, monkeypatch):
    calls = {"n": 0}
    class Counting(FakeClient):
        async def object_info(self):
            calls["n"] += 1
            return await super().object_info()
    monkeypatch.setattr(ct_mod, "get_client", lambda: Counting())
    ct_mod._object_info_cache = None
    await client.get("/api/v1/comfy/object-info")
    await client.get("/api/v1/comfy/object-info")
    assert calls["n"] == 1
```

- [ ] **Step 2: 跑失败** → 404。**Step 3: 实现**(缓存用 `time.monotonic()`)。**Step 4: 跑通过。**

- [ ] **Step 5: Commit** — `git commit -m "feat(comfy): sidecar 健康与 object_info 代理"`

---

### Task 5: 视频/音频产物存储 + 首帧缩略

**Files:**
- Modify: `backend/src/services/image_output_storage.py`
- Create: `backend/src/services/comfy/thumbnail.py`
- Test: `backend/tests/comfy/test_media_storage.py`

**Interfaces:**
- Consumes: `write_image(bytes, *, ext, ttl_seconds) -> dict`(现有;先读它返回的 dict 键——含签名 URL)。
- Produces:
  - `write_media = write_image` 别名导出(`image_output_storage.py` 尾部 `write_media = write_image`),并确认 files 服务路由按 ext 出正确 Content-Type(读 `src/api/routes/image_files.py`;若 mime 写死 image/*,改为 `mimetypes.guess_type`,并补该路由测试)。
  - `async def extract_first_frame(video_path: Path) -> bytes | None`(`thumbnail.py`):`ffmpeg -y -i <in> -frames:v 1 -f image2 <tmp.png>` 经 `asyncio.create_subprocess_exec`;ffmpeg 不存在(`FileNotFoundError`)或退出码非 0 → 返回 None,不抛。

- [ ] **Step 1: 写失败测试**

```python
# backend/tests/comfy/test_media_storage.py
import pytest
from src.services.image_output_storage import write_media
from src.services.comfy.thumbnail import extract_first_frame


def test_write_media_mp4_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("NOUS_OUTPUTS_DIR", str(tmp_path))  # 先读 _outputs_root 确认 env 名,不对则改这里
    out = write_media(b"\x00fakemp4", ext="mp4", ttl_seconds=60)
    assert out["url"].endswith(".mp4") or "mp4" in out["url"]


@pytest.mark.asyncio
async def test_extract_first_frame_missing_ffmpeg_returns_none(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))  # 空 PATH → ffmpeg 不存在
    assert await extract_first_frame(tmp_path / "x.mp4") is None
```

- [ ] **Step 2: 跑失败。Step 3: 实现。Step 4: 跑通过。**
- [ ] **Step 5: Commit** — `git commit -m "feat(comfy): 媒体产物存储别名 + ffmpeg 首帧缩略(可缺省)"`

---

### Task 6: comfyui_workflow / video_output 节点

**Files:**
- Create: `backend/src/services/nodes/comfy_bridge.py`
- Modify: `backend/src/services/nodes/__init__.py`(照现有惯例 import 以触发注册;先读该文件)
- Test: `backend/tests/comfy/test_bridge_node.py`

**Interfaces:**
- Consumes: `@register`(`nodes/registry.py`)、`ComfyClient`/`ComfyError`(Task 1)、`collect_outputs`(Task 2)、`write_media`/`extract_first_frame`(Task 5)、`ComfyTemplate` + 服务 `exposed_inputs`(Task 3)。
- Produces:
  - `@register("comfyui_workflow") class ComfyUIWorkflowNode`:`async invoke(self, data: dict, inputs: dict) -> dict`。`data` = `{template_id, <param_key>: value...}`(apply_inputs_to_snapshot 已把用户 input 写进 data)。流程:
    1. 读模板 workflow_json + 服务 exposed_params(按 template_id 查;DB 访问方式照其他节点/服务惯例,读 `nodes/image.py` 参考);
    2. graph = deepcopy;逐映射:值 = data.get(key, default);`type=="media"` 且值为 data URI → 解码 → `client.upload_image` → 值替换为返回文件名;`random=True` 且值为空 → `secrets.randbelow(2**32)` 并记入结果;写 `graph[comfy_node_id]["inputs"][comfy_input] = 值`;`required` 且缺值 → 抛 `ValueError("缺少必填参数:<key>")`;
    3. `async with _SEM:`(模块级 `asyncio.Semaphore(1)`,sidecar 显存独占)→ `submit` → `wait(timeout_s=float(os.getenv("NOUS_COMFY_TIMEOUT","14400")))`;
    4. `collect_outputs` → 逐项 `download` → `write_media`;kind=video 时 `extract_first_frame` → 有则 `write_media(frame, ext="png")` 作缩略;
    5. 返回 `{"items": [{"url", "kind", "filename", "node_id", "class_type"}...], "video_url": 首个 video 的 url 或 None, "thumbnails": [缩略 url...], "seed": 随机 seed 或 None}`。
  - `@register("video_output") class VideoOutputNode`:`async invoke(self, data, inputs) -> dict` — 原样返回 `inputs.get("outputs") or inputs`(输出节点惯例先读 `nodes/image.py` 的 image_output 实现对齐)。
  - `ComfyError` → 抛出让 run_workflow_task 记 task.error(错误信息即翻译后中文)。

- [ ] **Step 1: 写失败测试**(FakeClient 注入:节点经模块级 `get_client()`,同 Task 4 模式)

```python
# backend/tests/comfy/test_bridge_node.py
import base64
import pytest
import src.services.nodes.comfy_bridge as nb
from src.services.nodes.registry import get_node_class

PNG1PX = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg==")


class FakeClient:
    def __init__(self):
        self.uploaded, self.submitted = [], None
    async def upload_image(self, filename, content, mime="image/png"):
        self.uploaded.append(filename); return f"up_{filename}"
    async def submit(self, graph):
        self.submitted = graph; return "p1"
    async def wait(self, prompt_id, *, timeout_s, interval_s=2.0):
        return {"outputs": {"92": {"images": [
            {"filename": "out.mp4", "subfolder": "", "type": "output"}]}}}
    async def download(self, item):
        return b"MP4DATA"


@pytest.fixture
def fake(monkeypatch, tmp_path):
    fc = FakeClient()
    monkeypatch.setattr(nb, "get_client", lambda: fc)
    monkeypatch.setattr(nb, "extract_first_frame", _none_async := (lambda p: _async_none()))
    async def fake_write(data, *, ext, ttl_seconds=86400):
        return {"url": f"/files/x.{ext}", "uuid": "u", "ext": ext}
    monkeypatch.setattr(nb, "write_media", fake_write)
    # 模板与映射也 monkeypatch(绕 DB):
    monkeypatch.setattr(nb, "load_template", lambda tid: (
        {"138": {"class_type": "T", "inputs": {"value": ""}},
         "78": {"class_type": "LoadImage", "inputs": {"image": ""}},
         "92": {"class_type": "SaveVideo", "inputs": {}}},
        [{"key": "prompt", "type": "string", "comfy_node_id": "138", "comfy_input": "value", "required": True},
         {"key": "ref", "type": "media", "comfy_node_id": "78", "comfy_input": "image"}]))
    return fc


async def _async_none():
    return None


@pytest.mark.asyncio
async def test_invoke_patches_uploads_and_returns_video(fake):
    node = get_node_class("comfyui_workflow")()
    data = {"template_id": 1, "prompt": "hello",
            "ref": "data:image/png;base64," + base64.b64encode(PNG1PX).decode()}
    out = await node.invoke(data, {})
    assert fake.submitted["138"]["inputs"]["value"] == "hello"
    assert fake.submitted["78"]["inputs"]["image"].startswith("up_")
    assert out["video_url"] == "/files/x.mp4"


@pytest.mark.asyncio
async def test_missing_required_raises(fake):
    node = get_node_class("comfyui_workflow")()
    with pytest.raises(ValueError, match="prompt"):
        await node.invoke({"template_id": 1}, {})
```

- [ ] **Step 2: 跑失败。Step 3: 实现 comfy_bridge.py**(`load_template(tid)` 真实现:async DB 查询模板 + 关联服务 exposed_inputs;注意节点 invoke 是 async,可直接 await session)。**Step 4: 跑通过 + 跑全量 `uv run pytest tests/ -x -q` 确认无回归。**
- [ ] **Step 5: Commit** — `git commit -m "feat(comfy): comfyui_workflow/video_output 节点 — patch/上传/串行执行/产物落储"`

---

### Task 7: predictions 端到端 + 取消打通

**Files:**
- Modify: `backend/src/api/routes/execution_tasks.py`(cancel 端点)
- Modify: `backend/src/services/execution_task_serialize.py`(先读;确认 video url/缩略进 `output_thumbnails`/结果序列化——若已从 result 泛化提取则只补测试)
- Test: `backend/tests/comfy/test_prediction_e2e.py`

**Interfaces:**
- Consumes: Task 3 建的服务、Task 6 节点、`POST /v1/services/{name}/predictions`(现有)。
- Produces: 取消行为——cancel 端点在 task 归属服务为 comfy_template 且 status=running 时,额外 `await get_client().interrupt()`(import 自 comfy_templates 路由模块;串行语义下中断当前即中断本任务;局限已记 spec §12)。

- [ ] **Step 1: 写失败测试**

```python
# backend/tests/comfy/test_prediction_e2e.py
import asyncio
import pytest
import src.services.nodes.comfy_bridge as nb
from tests.comfy.test_bridge_node import FakeClient  # 复用


@pytest.mark.asyncio
async def test_async_prediction_completes_with_video(client, monkeypatch):
    fc = FakeClient()
    monkeypatch.setattr(nb, "get_client", lambda: fc)
    wf = {"138": {"class_type": "T", "inputs": {"value": ""}},
          "92": {"class_type": "SaveVideo", "inputs": {}}}
    r = await client.post("/api/v1/comfy-templates", json={"name": "e2e-h3", "workflow": wf})
    tid = r.json()["id"]
    await client.put(f"/api/v1/comfy-templates/{tid}/mapping", json={"exposed_params": [
        {"key": "prompt", "label": "提示词", "type": "string",
         "comfy_node_id": "138", "comfy_input": "value", "required": True}]})
    r = await client.post("/v1/services/e2e-h3/predictions",
                          json={"input": {"prompt": "hi"}},
                          headers={"Prefer": "respond-async"})
    assert r.status_code == 202, r.text
    pid = r.json()["id"]
    for _ in range(50):
        p = (await client.get(f"/v1/predictions/{pid}")).json()
        if p["status"] in ("succeeded", "completed", "failed"):
            break
        await asyncio.sleep(0.1)
    assert p["status"] in ("succeeded", "completed"), p
    assert any("mp4" in str(v) for v in str(p["output"]))
```

注:prediction 无鉴权路径/bearer 要求以现有 predictions 测试(先读 `backend/tests/` 里 prediction 相关测试文件)为准,照抄其建 key/headers 的做法;状态字面量同理对齐后修正断言。

- [ ] **Step 2: 跑失败(服务能建但执行/断言不通则逐步修)。Step 3: 打通(含 cancel:另写一测,提交后立刻 cancel,断言 FakeClient 记录到 interrupt 调用且 task 终态 cancelled)。Step 4: 全量回归 `uv run pytest tests/ -q`。**
- [ ] **Step 5: Commit** — `git commit -m "feat(comfy): prediction 全链路打通 + 取消转发 interrupt"`

---

### Task 8: 前端 — 服务列表导入入口与「桥」标识

**Files:**
- Modify: `frontend/src/pages/ServicesList.tsx`(筛选 tab、行徽标、新建下拉项)
- Create: `frontend/src/components/services/ImportComfyDialog.tsx`
- Modify: `frontend/src/api/services.ts`(若类型缺 source_type 则补)
- Create: `frontend/src/api/comfyTemplates.ts`
- Test: `frontend/src/components/services/ImportComfyDialog.test.tsx`

**Interfaces:**
- Consumes: `POST /api/v1/comfy-templates`(Task 3)。
- Produces: `comfyTemplates.ts` 导出 `createComfyTemplate(name: string, workflow: object): Promise<{id: number; service_name: string; node_count: number}>`、`putMapping(id, exposedParams)`、`getComfyTemplate(id)`、`getComfyHealth()`、`getObjectInfo()`(fetch 封装照 `api/services.ts` 惯例);`ImportComfyDialog` props `{open: boolean; onClose(): void; onImported(serviceName: string): void}`——文件拖入/选择 → JSON.parse 校验(非法 JSON / 非 API 格式即无 class_type 键 → 行内错误文案「不是 ComfyUI API 格式导出,请在 ComfyUI 用 Export (API) 导出」)→ 输入服务名(校验 `^[a-z][a-z0-9-]{1,62}$`)→ 提交 → onImported 跳 `/services/{id}` 编辑分段。

- [ ] **Step 1: 写失败测试**

```tsx
// ImportComfyDialog.test.tsx
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { describe, it, expect, vi } from 'vitest'
import ImportComfyDialog from './ImportComfyDialog'

vi.mock('../../api/comfyTemplates', () => ({
  createComfyTemplate: vi.fn(async () => ({ id: 1, service_name: 'minimax-h3-r2v', node_count: 2 })),
}))

describe('ImportComfyDialog', () => {
  it('拒绝非 API 格式 JSON', async () => {
    render(<ImportComfyDialog open onClose={() => {}} onImported={() => {}} />)
    const file = new File(['{"nodes":[]}'], 'ui.json', { type: 'application/json' })
    fireEvent.change(screen.getByTestId('comfy-file-input'), { target: { files: [file] } })
    await waitFor(() => expect(screen.getByText(/API 格式/)).toBeInTheDocument())
  })
  it('合法 JSON + 名称 → 提交回调服务名', async () => {
    const onImported = vi.fn()
    render(<ImportComfyDialog open onClose={() => {}} onImported={onImported} />)
    const file = new File([JSON.stringify({ '1': { class_type: 'X', inputs: {} } })], 'wf.json',
      { type: 'application/json' })
    fireEvent.change(screen.getByTestId('comfy-file-input'), { target: { files: [file] } })
    fireEvent.change(await screen.findByPlaceholderText(/服务名/), { target: { value: 'minimax-h3-r2v' } })
    fireEvent.click(screen.getByRole('button', { name: /导入/ }))
    await waitFor(() => expect(onImported).toHaveBeenCalledWith('minimax-h3-r2v'))
  })
})
```

- [ ] **Step 2: `npx vitest run src/components/services/ImportComfyDialog.test.tsx` → FAIL。Step 3: 实现组件 + 接入 ServicesList(下拉、`source_type==='comfy_template'` 行徽标「桥」、筛选 tab;样式全用现有 token/类,参考同页现有 tab)。Step 4: vitest 通过 + `npx vitest run src/pages/ServicesList.test.tsx` 无回归。**
- [ ] **Step 5: Commit** — `git commit -m "feat(ui): 服务列表导入 ComfyUI 工作流 + 桥徽标筛选"`

---

### Task 9: 前端 — 读图选节点字段配置(ServiceDetail 编辑)

**Files:**
- Create: `frontend/src/components/services/ComfyTemplateEditor.tsx`
- Create: `frontend/src/components/services/comfyGraphLayout.ts`
- Modify: `frontend/src/pages/ServiceDetail.tsx`(Playground tab 编辑分支:`svc.source_type === 'comfy_template'` 时渲染 ComfyTemplateEditor 替代节点图编辑器;先读 442 行起合并 tab 实现)
- Test: `frontend/src/components/services/comfyGraphLayout.test.ts`
- Test: `frontend/src/components/services/ComfyTemplateEditor.test.tsx`

**Interfaces:**
- Consumes: `getComfyTemplate/putMapping/getObjectInfo/getComfyHealth`(Task 8)。
- Produces:
  - `comfyGraphLayout.ts`:`layoutComfyGraph(workflow: Record<string, {class_type: string, inputs: Record<string, unknown>}>): {nodes: {id, class_type, x, y, usedCount}[], edges: {source, target}[]}` — 边=inputs 值形如 `[nodeId, slot]` 的引用;x=拓扑深度×列宽(220),y=列内序×高(64)。
  - `ComfyTemplateEditor`:sidecar 状态行(getComfyHealth)、上传替换(stale_keys 高亮)、React Flow 通用卡(点击开弹窗:该节点全部原始值 inputs 列表,勾选暴露/显示名/类型(object_info 有该 class 则预填类型与 min/max/options)/random 勾选)、下方字段汇总表、保存→putMapping。
  - 弹窗与表编辑同一份 state:`exposedParams: ExposedParamDraft[]`,`type ExposedParamDraft = {key, label, type, comfy_node_id, comfy_input, default?, min?, max?, step?, options?, required?, random?}`。

- [ ] **Step 1: 写 layout 失败测试**

```ts
// comfyGraphLayout.test.ts
import { describe, it, expect } from 'vitest'
import { layoutComfyGraph } from './comfyGraphLayout'

const WF = {
  '1': { class_type: 'LoadImage', inputs: { image: 'a.png' } },
  '2': { class_type: 'Encode', inputs: { pixels: ['1', 0] } },
  '3': { class_type: 'SaveVideo', inputs: { latent: ['2', 0] } },
}

describe('layoutComfyGraph', () => {
  it('按拓扑深度分列,引用值成边', () => {
    const { nodes, edges } = layoutComfyGraph(WF)
    const byId = Object.fromEntries(nodes.map((n) => [n.id, n]))
    expect(byId['1'].x).toBeLessThan(byId['2'].x)
    expect(byId['2'].x).toBeLessThan(byId['3'].x)
    expect(edges).toContainEqual({ source: '1', target: '2' })
    expect(edges).toContainEqual({ source: '2', target: '3' })
  })
})
```

- [ ] **Step 2: FAIL → Step 3: 实现 layout(BFS 深度)。Step 4: 通过。**
- [ ] **Step 5: 写 Editor 测试(mock api 模块;断言:渲染节点卡数、点卡弹窗列出 inputs、勾选后汇总表出现该行、保存调 putMapping 载荷正确)→ FAIL → 实现 → 通过。ServiceDetail 接线后跑 `npx vitest run src/pages/ServiceDetail.test.tsx` 无回归。**
- [ ] **Step 6: Commit** — `git commit -m "feat(ui): 读图选节点字段配置 — 通用节点卡 + object_info 预填"`

---

### Task 10: 前端 — Playground 异步运行态

**Files:**
- Modify: `frontend/src/pages/ServiceDetail.tsx`(AppTab 运行分支:comfy_template 服务提交走 respond-async)
- Create: `frontend/src/components/playground/AsyncRunState.tsx`
- Modify: `frontend/src/api/services.ts` 或新增 `frontend/src/api/predictions.ts`(`createPredictionAsync(service, input) -> {id}`、`getPrediction(id)`、`cancelPrediction(id)`;URL 按后端 `/v1/services/{name}/predictions`,注意从 admin 会话侧调用的鉴权方式照 Playground 现有运行请求的做法——先读现在同步运行走哪个端点/头)
- Test: `frontend/src/components/playground/AsyncRunState.test.tsx`

**Interfaces:**
- Consumes: prediction API(现有)+ `SchemaDrivenOutput`(现有,完成后渲染输出)。
- Produces: `AsyncRunState` props `{predictionId: string; onDone(output: unknown): void; onCancel(): void}` — 2s 轮询 getPrediction;渲染四行状态(已提交/排队/运行中+耗时秒表/完成)+ 取消按钮;终态 failed 显示 error 文案;组件卸载清定时器。

- [ ] **Step 1: 写失败测试**(vi.useFakeTimers + mock getPrediction 依次返回 queued→running→completed,断言状态文案推进、onDone 拿到 output、取消按钮触发 cancelPrediction)。**Step 2: FAIL。Step 3: 实现 + ServiceDetail 接线(comfy_template → respond-async;其余服务原路不动)。Step 4: 通过 + ServiceDetail 测试无回归。**
- [ ] **Step 5: Commit** — `git commit -m "feat(ui): Playground 异步运行态 — 轮询/秒表/取消"`

---

### Task 11: 前端 — 历史画廊视频卡 + 灯箱播放

**Files:**
- Modify: `frontend/src/components/overlays/HistoryOverlay.tsx`(标题「历史生成」;卡片:缩略 URL 以 `.mp4` 结尾或 item 带 kind=video → ▶ 角标 + 时长角标)
- Modify: `frontend/src/stores/lightbox.ts` + 灯箱渲染组件(`components/nodes/Lightbox.tsx`):`LightboxItem` 增 `kind?: 'image' | 'video'`;video → `<video controls autoPlay src={url}>`,禁用图片缩放平移逻辑
- Modify: `frontend/src/api/tasks.ts`(若 useImageTasks 过滤条件排除视频任务则放宽:有 output_thumbnails 即收;先读)
- Test: `frontend/src/components/overlays/HistoryOverlay.test.tsx`(补 video 用例)
- Test: `frontend/src/components/nodes/Lightbox.test.tsx`(补 video 用例)

**Interfaces:**
- Consumes: 后端 result/序列化里的 thumbnails 与 video url(Task 6/7 产出;缩略进 `output_thumbnails`,video url 在 task result items)。
- Produces: 无对外新接口;`LightboxItem.kind` 供两处共用。

- [ ] **Step 1: 写失败测试(HistoryOverlay:构造带 video 产物的 task fixture,断言 ▶ 角标与服务名渲染;Lightbox:kind=video 渲染 `<video>` 标签)。Step 2: FAIL。Step 3: 实现。Step 4: 两文件 vitest + 原有测试无回归。**
- [ ] **Step 5: Commit** — `git commit -m "feat(ui): 历史画廊视频卡 + 灯箱视频播放"`

---

### Task 12: 前端 — 设置「ComfyUI 桥」分区

**Files:**
- Modify: `frontend/src/components/overlays/SettingsOverlay.tsx`(Section 枚举 + SubNav 项 + Body 分支)
- Create: `frontend/src/components/settings/ComfyBridgeSection.tsx`
- Test: `frontend/src/components/settings/ComfyBridgeSection.test.tsx`

**Interfaces:**
- Consumes: `getComfyHealth()`(Task 8)。
- Produces: 分区展示 base_url / 在线状态 / 队列深度 / 版本 / 超时(均来自 health 端点;**只读**——地址与超时经 `backend/.env` 配置,分区内文案注明「修改需编辑 backend/.env 的 NOUS_COMFY_URL / NOUS_COMFY_TIMEOUT 并重启」,单管理员场景不做写回)。

- [ ] **Step 1: 写失败测试(mock getComfyHealth 在线/离线两态文案)。Step 2: FAIL。Step 3: 实现。Step 4: 通过 + SettingsOverlay.test.tsx 无回归。**
- [ ] **Step 5: Commit** — `git commit -m "feat(ui): 设置新增 ComfyUI 桥分区(状态只读)"`

---

### Task 13: infra — systemd 单元与 enginectl 纳管

**Files:**
- Create: `infra/systemd/nous-engine-comfyui.service`
- Modify: `infra/systemd/install.sh`(照现有单元的安装写法)
- Modify: `enginectl` 源(先 `grep -rn "nous-engine-backend" infra/` 找到 enginectl 脚本;把新单元加进 status/up/down/restart/logs 的单元列表)

**Interfaces:** 无代码接口;运维契约:`enginectl status` 可见 comfyui 单元。

- [ ] **Step 1: 写 unit 文件**

```ini
[Unit]
Description=nous-engine ComfyUI sidecar
After=network.target

[Service]
Type=simple
User=heygo
Environment=CUDA_DEVICE_ORDER=PCI_BUS_ID
Environment=CUDA_VISIBLE_DEVICES=0
# ComfyUI 安装目录与 venv:安装时按实际路径改(见 Step 2 的安装说明输出)
WorkingDirectory=/opt/comfyui
ExecStart=/opt/comfyui/.venv/bin/python main.py --listen 127.0.0.1 --port 8188
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

`CUDA_VISIBLE_DEVICES=0` 的 0 必须是 PCI_BUS_ID 序下的 Pro 6000——安装时用 `nvidia-smi --query-gpu=index,name --format=csv` 核对后填。

- [ ] **Step 2: install.sh 增拷贝/enable 逻辑 + enginectl 单元列表加 `nous-engine-comfyui`;本机 `bash -n` 语法检查两脚本。**
- [ ] **Step 3: Commit** — `git commit -m "chore(infra): ComfyUI sidecar systemd 单元 + enginectl 纳管"`

---

### Task 14: 真机 smoke + 文档收尾

**Files:**
- Create: `backend/tests/manual/smoke_comfy_h3.py`
- Modify: `CLAUDE.md`(新增「ComfyUI 桥」小节:模板即服务、异步 prediction、NOUS_COMFY_* 环境变量、smoke 跑法、GSP 缓解脚本前置)
- Modify: 本计划文件(勾掉完成项)

**Interfaces:** Consumes 全部前置任务。

- [x] **Step 1: 写 smoke 脚本**(非 CI,真机跑;结构性校验)

```python
"""真机 smoke:导入 H3 模板 → 异步 prediction → 校验 mp4 + 音轨。

前置:sidecar 已跑(enginectl status)、H3 INT8 权重已入 ComfyUI models、
GSP 缓解脚本已上机。跑法:
  cd backend && uv run python tests/manual/smoke_comfy_h3.py \
      --base http://127.0.0.1:8000 --admin-token $ADMIN_TOKEN \
      --workflow /path/to/minimax-h3-t2v-api.json --mapping /path/to/mapping.json
workflow 取自 ComfyUI 模板库(Workflow → Export (API));mapping 为
exposed_params JSON(至少暴露 prompt 与时长,时长压到最短以省时)。
"""
import argparse, json, subprocess, sys, time
import httpx  # 校验用 ffprobe 查音轨:ffprobe -show_streams -select_streams a

# 流程:POST /api/v1/comfy-templates → PUT mapping → POST /v1/services/{name}/predictions
# (Prefer: respond-async) → 轮询至终态(上限 NOUS_COMFY_TIMEOUT)→ 下载 video_url →
# ffprobe 断言:存在 v 流与 a 流 → 打印 PASS/耗时。任何断言失败 sys.exit(1)。
```

(实现补全上述流程,全部真代码;admin 端点带 `Authorization: Bearer $ADMIN_TOKEN`。)

- [x] **Step 2: 文档:CLAUDE.md 小节 + `frontend && npm run build`(生产前端产物)。**
      (本 worktree 缺 `wasm-pack`,`npm run build` 的 `prebuild` 跑不了——与 Task 8
      progress.md 记的 pre-existing 环境限制一致,production build 留到主环境做;
      已在 CLAUDE.md 新小节里记这条坑。)
- [x] **Step 3: 全量回归:`cd backend && uv run pytest tests/ -q` + `cd frontend && npx vitest run`。**
      (backend `-m "not e2e"`:1916 passed / 2 failed(`test_image_modular_wiring.py`
      沙箱无 CUDA,pre-existing,与本任务无关)/ 6 skipped。frontend vitest:313
      passed / 8 failed(`workspace.test.ts` round3 + `NodePropertyPanel.test.tsx`,
      zustand persist `setItem` TypeError,`git log` 确认最后改动在 #388,跟本分支
      无关,pre-existing)。均未修复,超出 Task 14 范围,详见 task-14-report.md。)
- [x] **Step 4: Commit** — `git commit -m "test(comfy): 真机 smoke 脚本 + 文档收尾"`

---

## Self-Review 记录

- **Spec 覆盖**:§3 架构(T1/2/6)、§4 注册与映射含目录扫描——**目录扫描 backend/comfy_templates/ 启动 upsert 未列任务**,并入 T3 范围过大,故明确:目录扫描降为二期(单管理员用导入 UI/API 已闭环;spec §4-2 该条推迟,已在下方「偏差」注明)。§5 数据流(T1/6/7)、§6 存储(T5)、§7 错误(T1/6/7)、§8 UI 四表面(T8-12)、§9 IC 契约(无代码,T14 文档带过)、§10 测试(各任务+T14)、§11 部署(T13/T14)。
- **与 spec 的两处偏差(均已确认合理)**:① 并发控制用模块级 `asyncio.Semaphore(1)` 而非 ProviderGovernor——governor 是 fail-fast(GovernorBusyError),视频任务需要排队等待语义;② 模板目录扫描推迟二期。
- **占位符扫描**:Task 9/10/11 的部分测试以文字描述断言要点而非全码——保留,因组件 props/断言目标已给全,实现者按 Task 8 的测试样式展开;其余任务均为可执行代码。
- **类型一致性**:`get_client()` 模式(T4 定义,T6/7 复用)、`write_media` 签名(T5→T6)、`ExposedParamDraft` 键与后端 mapping 载荷(T3↔T9)已核对一致。

## 执行顺序与依赖

T1→T2 可并行;T3 依赖无;T4 依赖 T1;T5 独立;T6 依赖 T1/2/3/5;T7 依赖 T6;T8 依赖 T3;T9 依赖 T4/8;T10 依赖 T7/8;T11 依赖 T7;T12 依赖 T8;T13/T14 收尾。
