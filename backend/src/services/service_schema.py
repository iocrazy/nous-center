"""Per-service I/O JSON-Schema 生成 + 调用期校验(服务层 API spec 2026-06-03,PR-1)。

发布的 ServiceInstance 的 `exposed_inputs/outputs`(元数据)+ 各节点 node.yaml `widgets` 定义
→ per-service JSON-Schema(input/output)。机器可发现(`GET /services/{name}/schema`)+ 调用期校验。
对齐 Cog「声明即 schema」+ ComfyUI object_info(节点 widget 即类型源)。**无 jsonschema 依赖,手写校验。**

PR-1 只生成 schema + 校验函数 + 端点;把校验**接进调用路径**是 PR-2(统一 prediction 端点)的事。
"""
from __future__ import annotations

from typing import Any

# node.yaml widget → JSON-Schema base type(select/slider 另行处理)。
_WIDGET_BASE_TYPE = {"slider": "number", "checkbox": "boolean", "seed": "integer"}
# ExposedParam.type 粗类型 → JSON-Schema type。
_EXPOSED_TYPE = {
    "int": "integer", "integer": "integer", "float": "number", "number": "number",
    "bool": "boolean", "boolean": "boolean", "string": "string",
    "object": "object", "array": "array",
}
# 输出里这些被当作「文件/产物」→ string + format=uri(交付契约 PR-4 落实)。
_FILE_OUT_TYPES = {"image", "file", "audio", "video", "latent"}


def _node_class_map(snapshot: Any) -> dict[str, str]:
    """node_id → class_type。兼容 snapshot 两种形:{"nodes": {id: {...}}} 或 {"nodes": [{id,...}]}。"""
    out: dict[str, str] = {}
    nodes = snapshot.get("nodes") if isinstance(snapshot, dict) else None
    if isinstance(nodes, dict):
        for nid, node in nodes.items():
            if isinstance(node, dict):
                out[str(nid)] = node.get("class_type") or node.get("type")
    elif isinstance(nodes, list):
        for n in nodes:
            if isinstance(n, dict) and n.get("id") is not None:
                out[str(n.get("id"))] = n.get("class_type") or n.get("type")
    return out


def _widget_index(class_type: str | None) -> dict[str, dict]:
    """node 类型的 widgets → {name: widget_def}。拿不到定义/无 widgets → 空(回退 ExposedParam.type)。"""
    if not class_type:
        return {}
    try:
        from nodes import get_all_definitions  # noqa: PLC0415
        defn = get_all_definitions().get(class_type) or {}
    except Exception:  # noqa: BLE001 — 定义加载失败不该拖垮 schema 生成
        return {}
    out: dict[str, dict] = {}
    for w in (defn.get("widgets") or []):
        if isinstance(w, dict) and w.get("name"):
            out[w["name"]] = w
    return out


def _option_values(options: Any) -> list:
    """node.yaml options(['a','b'] 或 [{value,label,description}])→ 值列表。"""
    out = []
    for o in options or []:
        if isinstance(o, dict):
            if "value" in o:
                out.append(o["value"])
        else:
            out.append(o)
    return out


def _all_numeric(values: list) -> bool:
    return bool(values) and all(
        isinstance(v, (int, float)) and not isinstance(v, bool) for v in values)


def _input_property(exposed: dict, widget: dict | None) -> dict:
    """一个 exposed_input(+ 可选 widget 定义)→ JSON-Schema property。"""
    prop: dict[str, Any] = {}
    wtype = (widget or {}).get("widget")
    if wtype == "select":
        enum = _option_values((widget or {}).get("options"))
        if enum:
            prop["enum"] = enum
        prop["type"] = "number" if _all_numeric(enum) else "string"
    elif wtype in _WIDGET_BASE_TYPE:
        prop["type"] = _WIDGET_BASE_TYPE[wtype]
        if wtype == "slider":
            if (widget or {}).get("min") is not None:
                prop["minimum"] = widget["min"]
            if (widget or {}).get("max") is not None:
                prop["maximum"] = widget["max"]
    else:
        # 无 widget / 文本类:回退 ExposedParam.type。
        t = str(exposed.get("type") or "string").lower()
        prop["type"] = _EXPOSED_TYPE.get(t, "string")
    # default:exposed 优先,其次 widget。
    default = exposed.get("default")
    if default is None and widget is not None:
        default = widget.get("default")
    if default is not None:
        prop["default"] = default
    if exposed.get("label"):
        prop["description"] = exposed["label"]
    return prop


def _input_key(exposed: dict) -> str | None:
    return exposed.get("key") or exposed.get("api_name") or exposed.get("input_name") or exposed.get("param_key")


def build_service_io_schema(exposed_inputs, exposed_outputs, snapshot) -> dict:
    """→ {"input_schema": <JSON-Schema obj>, "output_schema": <JSON-Schema obj>}。"""
    cmap = _node_class_map(snapshot or {})

    in_props: dict[str, dict] = {}
    required: list[str] = []
    for raw in (exposed_inputs or []):
        e = dict(raw)
        key = _input_key(e)
        if not key:
            continue
        ct = cmap.get(str(e.get("node_id")))
        widget = _widget_index(ct).get(e.get("input_name") or e.get("param_key"))
        in_props[key] = _input_property(e, widget)
        if e.get("required", True):
            required.append(key)
    input_schema: dict[str, Any] = {"type": "object", "properties": in_props}
    if required:
        input_schema["required"] = required

    out_props: dict[str, dict] = {}
    for raw in (exposed_outputs or []):
        e = dict(raw)
        key = e.get("key") or e.get("api_name") or e.get("input_name")
        if not key:
            continue
        t = str(e.get("type") or "string").lower()
        if t in _FILE_OUT_TYPES:
            prop: dict[str, Any] = {"type": "string", "format": "uri"}
        else:
            prop = {"type": _EXPOSED_TYPE.get(t, "string")}
        if e.get("label"):
            prop["description"] = e["label"]
        out_props[key] = prop
    output_schema = {"type": "object", "properties": out_props}

    return {"input_schema": input_schema, "output_schema": output_schema}


def validate_service_input(input_schema: dict, payload: Any) -> list[str]:
    """手写校验(无 jsonschema 依赖):required / type / enum / min-max。返回错误列表(空=通过)。

    多余的未声明字段放过(passthrough,不报错)—— 工作流可能有 schema 没覆盖的内部入参。
    """
    errors: list[str] = []
    if not isinstance(payload, dict):
        return ["input must be a JSON object"]
    props = (input_schema or {}).get("properties") or {}
    for k in (input_schema or {}).get("required") or []:
        if payload.get(k) is None:
            errors.append(f"missing required input: {k}")
    for k, v in payload.items():
        spec = props.get(k)
        if spec is None or v is None:
            continue
        t = spec.get("type")
        if t == "string" and not isinstance(v, str):
            errors.append(f"{k}: expected string")
        elif t == "integer" and (not isinstance(v, int) or isinstance(v, bool)):
            errors.append(f"{k}: expected integer")
        elif t == "number" and (not isinstance(v, (int, float)) or isinstance(v, bool)):
            errors.append(f"{k}: expected number")
        elif t == "boolean" and not isinstance(v, bool):
            errors.append(f"{k}: expected boolean")
        if "enum" in spec and v not in spec["enum"]:
            errors.append(f"{k}: must be one of {spec['enum']}")
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            if "minimum" in spec and v < spec["minimum"]:
                errors.append(f"{k}: must be >= {spec['minimum']}")
            if "maximum" in spec and v > spec["maximum"]:
                errors.append(f"{k}: must be <= {spec['maximum']}")
    return errors
