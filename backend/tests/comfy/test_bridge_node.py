"""comfyui_workflow / video_output 桥节点(Task 6)。

FakeClient 经模块级 `get_client()` 注入(同 Task 4 模式)。`load_template` /
`write_media` / `extract_first_frame` 也经 monkeypatch 绕开真 DB / 真 ffmpeg —
其中 `load_template` 真实现是 async DB 查询(见 test_load_template_real_db_lookup),
其余测试都 monkeypatch 成异步替身,贴合真实签名(都是 `await`-ed)。
"""
from __future__ import annotations

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
        self.uploaded.append(filename)
        return f"up_{filename}"

    async def submit(self, graph):
        self.submitted = graph
        return "p1"

    async def wait(self, prompt_id, *, timeout_s, interval_s=2.0):
        return {"outputs": {"92": {"images": [
            {"filename": "out.mp4", "subfolder": "", "type": "output"}]}}}

    async def download(self, item):
        return b"MP4DATA"


TEMPLATE_GRAPH = {
    "138": {"class_type": "T", "inputs": {"value": ""}},
    "78": {"class_type": "LoadImage", "inputs": {"image": ""}},
    "92": {"class_type": "SaveVideo", "inputs": {}},
}
TEMPLATE_MAPPING = [
    {"key": "prompt", "type": "string", "comfy_node_id": "138", "comfy_input": "value", "required": True},
    {"key": "ref", "type": "media", "comfy_node_id": "78", "comfy_input": "image"},
]


@pytest.fixture
def fake(monkeypatch):
    fc = FakeClient()
    monkeypatch.setattr(nb, "get_client", lambda: fc)

    async def _no_thumbnail(_path):
        return None
    monkeypatch.setattr(nb, "extract_first_frame", _no_thumbnail)

    async def fake_write(data, *, ext, ttl_seconds=86400):
        return {"url": f"/files/x.{ext}", "uuid": "u", "ext": ext}
    monkeypatch.setattr(nb, "write_media", fake_write)

    # 模板与映射也 monkeypatch(绕 DB)—— 真实现是 async DB 查询,替身跟着 async。
    async def fake_load_template(tid):
        return TEMPLATE_GRAPH, TEMPLATE_MAPPING
    monkeypatch.setattr(nb, "load_template", fake_load_template)
    return fc


@pytest.mark.asyncio
async def test_invoke_patches_uploads_and_returns_video(fake):
    node = get_node_class("comfyui_workflow")()
    data = {"template_id": 1, "prompt": "hello",
            "ref": "data:image/png;base64," + base64.b64encode(PNG1PX).decode()}
    out = await node.invoke(data, {})
    assert fake.submitted["138"]["inputs"]["value"] == "hello"
    assert fake.submitted["78"]["inputs"]["image"].startswith("up_")
    assert out["video_url"] == "/files/x.mp4"
    assert out["items"][0]["kind"] == "video"
    assert out["items"][0]["node_id"] == "92"
    assert out["thumbnails"] == []
    assert out["seed"] is None


@pytest.mark.asyncio
async def test_missing_required_raises(fake):
    node = get_node_class("comfyui_workflow")()
    with pytest.raises(ValueError, match="prompt"):
        await node.invoke({"template_id": 1}, {})


@pytest.mark.asyncio
async def test_random_param_generates_seed(monkeypatch, fake):
    async def fake_load_template_random(tid):
        return (
            {"1": {"class_type": "Seed", "inputs": {"seed": 0}}},
            [{"key": "seed", "type": "int", "comfy_node_id": "1", "comfy_input": "seed",
              "required": False, "random": True}],
        )
    monkeypatch.setattr(nb, "load_template", fake_load_template_random)
    node = get_node_class("comfyui_workflow")()
    out = await node.invoke({"template_id": 1}, {})
    assert isinstance(out["seed"], int)
    assert 0 <= out["seed"] < 2 ** 32
    assert fake.submitted["1"]["inputs"]["seed"] == out["seed"]


@pytest.mark.asyncio
async def test_video_output_passthrough():
    node = get_node_class("video_output")()
    out = await node.invoke({}, {"outputs": {"video_url": "/files/x.mp4"}})
    assert out == {"video_url": "/files/x.mp4"}
    out2 = await node.invoke({}, {"video_url": "/files/x.mp4"})
    assert out2 == {"video_url": "/files/x.mp4"}


@pytest.mark.asyncio
async def test_load_template_real_db_lookup(client):
    """真实现:不 monkeypatch load_template,直接建模板+映射后查询。"""
    wf = {"138": {"class_type": "T", "inputs": {"value": ""}},
          "92": {"class_type": "SaveVideo", "inputs": {}}}
    r = await client.post("/api/v1/comfy-templates", json={"name": "bridge-db-test", "workflow": wf})
    assert r.status_code == 201, r.text
    tid = int(r.json()["id"])
    mapping = {"exposed_params": [{"key": "prompt", "label": "提示词", "type": "string",
                                    "comfy_node_id": "138", "comfy_input": "value", "required": True}]}
    r2 = await client.put(f"/api/v1/comfy-templates/{tid}/mapping", json=mapping)
    assert r2.status_code == 200, r2.text

    graph, exposed = await nb.load_template(tid)
    assert graph == wf
    assert len(exposed) == 1 and exposed[0]["key"] == "prompt"
    assert exposed[0]["comfy_node_id"] == "138"


@pytest.mark.asyncio
async def test_load_template_missing_raises(client):
    with pytest.raises(ValueError):
        await nb.load_template(999999999)
