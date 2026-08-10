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
