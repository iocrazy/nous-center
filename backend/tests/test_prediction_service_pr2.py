"""服务层 API spec PR-2:统一 prediction 契约纯逻辑 + 端点 wiring。

apply_inputs_to_snapshot(旧 /run 丢弃 inputs 的修复)/ task_to_prediction(ExecutionTask→Cog 形)/
Prefer 头解析。端到端(发布→POST predictions→轮询)真机验。CI 安全(无 torch/DB)。
"""
from __future__ import annotations

import pathlib
import types

from src.services.prediction_service import (
    apply_inputs_to_snapshot,
    prediction_status,
    snapshot_to_executor_form,
    task_to_prediction,
)

_SRC = pathlib.Path(__file__).parent.parent / "src"


# ---- apply_inputs_to_snapshot ----

def test_inject_into_dict_snapshot():
    snap = {"nodes": {
        "in": {"class_type": "text_input", "inputs": {"value": "旧"}},
        "ks": {"class_type": "ksampler", "inputs": {"steps": 20}},
    }}
    exposed = [
        {"node_id": "in", "key": "prompt", "input_name": "value"},
        {"node_id": "ks", "key": "steps", "input_name": "steps"},
    ]
    out = apply_inputs_to_snapshot(snap, exposed, {"prompt": "新提示", "steps": 30})
    assert out["nodes"]["in"]["inputs"]["value"] == "新提示"
    assert out["nodes"]["ks"]["inputs"]["steps"] == 30
    # 原快照不被改(深拷贝)
    assert snap["nodes"]["in"]["inputs"]["value"] == "旧"


def test_missing_key_keeps_default():
    snap = {"nodes": {"in": {"inputs": {"value": "冻结默认"}}}}
    exposed = [{"node_id": "in", "key": "prompt", "input_name": "value"}]
    out = apply_inputs_to_snapshot(snap, exposed, {})  # 不传 prompt
    assert out["nodes"]["in"]["inputs"]["value"] == "冻结默认"


def test_inject_into_list_snapshot_data_form():
    snap = {"nodes": [{"id": "in", "type": "text_input", "data": {"value": "旧"}}]}
    exposed = [{"node_id": "in", "key": "prompt", "input_name": "value"}]
    out = apply_inputs_to_snapshot(snap, exposed, {"prompt": "X"})
    assert out["nodes"][0]["data"]["value"] == "X"


def test_legacy_alias_keys():
    snap = {"nodes": {"n": {"inputs": {}}}}
    exposed = [{"node_id": "n", "api_name": "p", "param_key": "field"}]  # legacy 别名
    out = apply_inputs_to_snapshot(snap, exposed, {"p": "v"})
    assert out["nodes"]["n"]["inputs"]["field"] == "v"


# ---- snapshot_to_executor_form(发布 api-shape dict → executor 编辑形 list)----

def test_snapshot_dict_to_list():
    snap = {"schema": "comfy/api-1", "nodes": {
        "t1": {"class_type": "text_input", "inputs": {"text": "hi"}, "_meta": {"x": 1}},
    }, "edges": [{"a": "b"}]}
    out = snapshot_to_executor_form(snap)
    assert isinstance(out["nodes"], list)
    n = out["nodes"][0]
    assert n["id"] == "t1" and n["type"] == "text_input"
    assert n["data"] == {"text": "hi"} and n["meta"] == {"x": 1}
    assert out["edges"] == [{"a": "b"}]  # edges 原样


def test_snapshot_already_list_passthrough():
    snap = {"nodes": [{"id": "t1", "type": "text_input", "data": {}}]}
    assert snapshot_to_executor_form(snap) is snap


def test_inject_then_convert_carries_value():
    """注入(api-shape inputs)→ 转编辑形(inputs→data),注入值带过去。"""
    snap = {"nodes": {"in": {"class_type": "text_input", "inputs": {"text": "旧"}}}}
    patched = apply_inputs_to_snapshot(snap, [{"node_id": "in", "key": "text", "input_name": "text"}], {"text": "新"})
    exe = snapshot_to_executor_form(patched)
    assert exe["nodes"][0]["data"]["text"] == "新"


# ---- task_to_prediction ----

def _task(**kw):
    base = dict(id=123, workflow_name="my-svc", status="completed", result={"image": "u"},
               error=None, duration_ms=4520, input_json={"prompt": "hi"},
               created_at=None, started_at=None, finished_at=None)
    base.update(kw)
    return types.SimpleNamespace(**base)


def test_prediction_shape_succeeded():
    p = task_to_prediction(_task(), service="my-svc")
    assert p["id"] == "123" and p["service"] == "my-svc"
    assert p["status"] == "succeeded"
    assert p["output"] == {"image": "u"}          # output 只在 succeeded
    assert p["input"] == {"prompt": "hi"}          # input 回显(传入优先,否则 input_json)
    assert p["metrics"]["predict_time"] == 4.52


def test_prediction_status_mapping():
    assert prediction_status("queued") == "starting"
    assert prediction_status("running") == "processing"
    assert prediction_status("failed") == "failed"
    assert prediction_status("cancelled") == "canceled"


def test_prediction_no_output_until_succeeded():
    p = task_to_prediction(_task(status="running", result={"partial": 1}))
    assert p["status"] == "processing"
    assert p["output"] is None                     # 未终态不给 output


# ---- Prefer 头解析 + wiring ----

def test_parse_prefer():
    from src.api.routes.predictions import _parse_prefer  # noqa: PLC0415
    assert _parse_prefer("respond-async") == (True, None)
    assert _parse_prefer("wait=5") == (False, 5.0)
    assert _parse_prefer(None) == (False, None)
    assert _parse_prefer("wait=30, handling=lenient") == (False, 30.0)


def test_route_wired_and_run_deleted():
    pred = (_SRC / "api/routes/predictions.py").read_text()
    assert '"/services/{name}/predictions"' in pred
    assert '"/predictions/{prediction_id}"' in pred
    assert "apply_inputs_to_snapshot(" in pred
    assert "validate_service_input(" in pred  # 接进 PR-1 校验
    # 旧 /run 已删(clean cut)
    inst = (_SRC / "api/routes/instance_service.py").read_text()
    assert '"/{instance_id}/run"' not in inst
    assert "def instance_run" not in inst
    # router 注册
    main = (_SRC / "api/main.py").read_text()
    assert "predictions_routes.router" in main
